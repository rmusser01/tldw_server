"""Public archive recovery preserves Explainer identity and collection Media links."""

import json
import zipfile
from contextlib import ExitStack
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Chatbooks.chatbook_models import (
    ChatbookManifest,
    ChatbookVersion,
    ConflictResolution,
    ContentItem,
    ContentType,
)
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.api import get_media_by_id
from tldw_Server_API.app.core.Explainer.chatbook_adapter import build_explainer_chatbook_payload

pytestmark = pytest.mark.integration


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "users"))
    with ExitStack() as stack:
        services = []
        for owner in (101, 202):
            db = CharactersRAGDB(tmp_path / f"chat-{owner}.db", client_id=str(owner))
            stack.callback(db.close_all_connections)
            service = ChatbookService(user_id=str(owner), user_id_int=owner, db=db)
            services.append(service)
        yield services
        for service in services:
            if service._media_db is not None:
                stack.callback(service._media_db.close_connection)
            if service._chroma_manager is not None:
                stack.callback(service._chroma_manager.close)
            if service._explainer_db is not None:
                stack.callback(service._explainer_db.close_all_connections)


def session(repo, owner, title="Rowan explanation"):
    return repo.create_session(
        owner_user_id=str(owner),
        title=title,
        mode="goal",
        output_intent="explain",
        grounding="open",
        depth_preset="quick",
        selected_sources=[],
        root_prompt="Why Selene?",
    )


async def explainer_archive(source, destination, lane):
    repo = source._get_explainer_repo()
    original = session(repo, 101)
    if lane == "first-class":
        okay, message, archive = await source.create_chatbook(
            name="Rowan Explainer",
            description="Real session",
            async_mode=False,
            content_selections={ContentType.EXPLAINER_SESSION: [original.id]},
        )
        assert okay, message  # nosec B101 -- Require a genuine public archive.
        target = destination.import_dir / "explainer.chatbook"
        target.write_bytes(Path(archive).read_bytes())
    else:
        payload = build_explainer_chatbook_payload(repo=repo, session_id=original.id, owner_user_id="101")
        path = f"content/generated_documents/document_{original.id}.json"
        manifest = ChatbookManifest(
            version=ChatbookVersion.V1,
            name="Legacy Explainer",
            description="Supported fallback",
            content_items=[
                ContentItem(
                    id=original.id,
                    type=ContentType.GENERATED_DOCUMENT,
                    title=original.title,
                    file_path=path,
                    metadata={"subtype": "explainer_session"},
                )
            ],
        )
        target = destination.import_dir / "explainer.chatbook"
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest.to_dict()))
            archive.writestr(path, json.dumps({"type": "generated_document", "content": payload}))
    return target


@pytest.mark.parametrize("lane", ["first-class", "fallback"])
@pytest.mark.parametrize("prefix", [False, True])
async def test_skip_retry_does_not_duplicate_explainer_sessions(stores, lane, prefix):
    source, destination = stores
    archive = await explainer_archive(source, destination, lane)
    repo = destination._get_explainer_repo()
    foreign = session(repo, 303, "[Imported] Rowan explanation" if prefix else "Rowan explanation")
    for attempt in range(2):
        okay, message, result = await destination.import_chatbook(
            str(archive),
            conflict_resolution=ConflictResolution.SKIP,
            prefix_imported=prefix,
            async_mode=False,
        )
        assert okay, message  # nosec B101 -- Each public attempt must complete honestly.
        owned = repo.list_sessions(owner_user_id="202")
        assert len(owned) == 1  # nosec B101 -- Retry must preserve the first successful session.
        assert result["successful_items"] == (1 if attempt == 0 else 0)  # nosec B101 -- Truthful counters.
        assert result["skipped_items"] == attempt  # nosec B101 -- Existing owner item is skipped.
        assert owned[0].title == ("[Imported] Rowan explanation" if prefix else "Rowan explanation")  # nosec B101 -- Prefix precedes conflict lookup.
        assert owned[0].nodes[owned[0].root_node_ids[0]].title == "Why Selene?"  # nosec B101 -- Actual tree survives.
    assert repo.get_session(foreign.id, owner_user_id="303") == foreign  # nosec B101 -- Same-title foreign content stays intact.


@pytest.mark.parametrize("lane", ["first-class", "fallback"])
async def test_rename_keeps_original_explainer_and_creates_distinct_title(stores, lane):
    source, destination = stores
    archive = await explainer_archive(source, destination, lane)
    repo = destination._get_explainer_repo()
    original = session(repo, 202)
    okay, message, result = await destination.import_chatbook(
        str(archive),
        conflict_resolution=ConflictResolution.RENAME,
        prefix_imported=False,
        async_mode=False,
    )
    assert okay, message  # nosec B101 -- Rename is a supported public import contract.
    rows = repo.list_sessions(owner_user_id="202")
    assert len(rows) == 2 and len({row.title for row in rows}) == 2  # nosec B101 -- No duplicate display identity.
    assert repo.get_session(original.id, owner_user_id="202") == original  # nosec B101 -- Preserve destination body.
    assert result["successful_items"] == 1  # nosec B101 -- Count the one new session.


@pytest.mark.parametrize("mode", ["fresh", "unselected", "skip", "same-title"])
async def test_collection_media_links_bind_only_to_selected_restored_media(stores, mode):
    source, destination = stores
    restore_media = mode != "unselected"
    src_media = source._get_media_db()
    dst_media = destination._get_media_db()
    source_id, source_uuid, _ = src_media.add_media_with_keywords(
        url="https://example.test/rowan",
        title="Rowan source",
        media_type="document",
        content="The telescope is Selene.",
        keywords=[],
        owner_user_id=101,
    )
    other_id, other_uuid, _ = dst_media.add_media_with_keywords(
        url="https://example.test/unrelated",
        title="Rowan source" if mode == "same-title" else "Destination original",
        media_type="document",
        content="Unrelated owner content.",
        keywords=[],
        owner_user_id=202,
    )
    assert source_id == other_id  # nosec B101 -- Old numeric ID deliberately collides with unrelated destination content.
    chroma = source._get_chroma_manager()
    chroma.store_in_chroma(
        "rowan_links",
        ["Selene", "Unselected", "No Media link", "Conflicting link"],
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.1, 0.9]],
        ["selected", "unselected", "plain", "conflicting"],
        [
            {"media_id": source_id, "media_uuid": source_uuid, "label": "keep"},
            {"media_id": 99999, "media_uuid": "unselected-source", "label": "unselected"},
            {"label": "plain"},
            {"media_id": source_id, "media_uuid": "unmapped-uuid", "label": "conflicting"},
        ],
    )
    okay, message, path = await source.create_chatbook(
        name="Rowan vectors",
        description="Real collection and Media",
        format_version=ChatbookVersion.V1_1,
        content_selections={ContentType.MEDIA: [str(source_id)], ContentType.EMBEDDING: ["rowan_links"]},
        include_embeddings=True,
        async_mode=False,
    )
    assert okay, message  # nosec B101 -- Use the actual exporter and archive.
    target = destination.import_dir / "vectors.chatbook"
    target.write_bytes(Path(path).read_bytes())
    if mode == "skip":
        okay, message, _ = await destination.import_chatbook(
            str(target),
            content_selections={ContentType.MEDIA: [str(source_id)]},
            conflict_resolution=ConflictResolution.SKIP,
            prefix_imported=False,
            async_mode=False,
        )
        assert okay, message  # nosec B101 -- Seed only the genuine first successful Media import.
    okay, message, result = await destination.import_chatbook(
        str(target),
        content_selections=None if restore_media else {ContentType.EMBEDDING: ["collection:rowan_links"]},
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=False,
        async_mode=False,
    )
    assert okay, message  # nosec B101 -- References must not break independent vector restoration.
    collection = destination._get_chroma_manager().get_collection("rowan_links")
    data = collection.get(include=["documents", "metadatas", "embeddings"])
    metadata = dict(zip(data["ids"], data["metadatas"]))
    expected = {"label": "keep"}
    if restore_media and mode != "same-title":
        restored = dst_media.get_media_by_title("Rowan source")
        assert restored["id"] != source_id and restored["uuid"] != source_uuid  # nosec B101 -- Both destination identities genuinely differ.
        expected.update(media_id=restored["id"], media_uuid=restored["uuid"])
    assert metadata["selected"] == expected  # nosec B101 -- Never bind old IDs to unrelated destination Media.
    assert metadata["unselected"] == {"label": "unselected"}  # nosec B101 -- Unselected source links clear explicitly.
    assert metadata["conflicting"] == {"label": "conflicting"}  # nosec B101 -- Every supplied source identity must agree.
    assert metadata["plain"] == {"label": "plain"}  # nosec B101 -- Unrelated metadata is preserved.
    assert any("not imported" in warning for warning in result["warnings"])  # nosec B101 -- Disclose unlinked references.
    assert get_media_by_id(dst_media, other_id)["uuid"] == other_uuid  # nosec B101 -- Existing destination remains untouched.


@pytest.mark.parametrize("lane", ["first-class", "fallback"])
async def test_failed_explainer_node_restore_cannot_be_skipped_as_complete(stores, monkeypatch, lane):
    source, destination = stores
    archive = await explainer_archive(source, destination, lane)
    repo = destination._get_explainer_repo()
    update = repo.update_node

    def failed_update(*_args, **_kwargs):
        raise RuntimeError("Synthetic node write failure")

    with monkeypatch.context() as failing:
        failing.setattr(repo, "update_node", failed_update)
        okay, _message, result = await destination.import_chatbook(
            str(archive),
            conflict_resolution=ConflictResolution.SKIP,
            prefix_imported=False,
            async_mode=False,
        )
    assert not okay and result["failed_items"] == 1  # nosec B101 -- Failed content must remain failed.
    assert repo.list_sessions(owner_user_id="202") == []  # nosec B101 -- Partial trees cannot become Skip candidates.
    assert repo.update_node == update  # nosec B101 -- Restore real storage before legitimate Retry.
    okay, message, result = await destination.import_chatbook(
        str(archive),
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=False,
        async_mode=False,
    )
    assert okay, message  # nosec B101 -- One legitimate recovery restores the full tree.
    assert result["successful_items"] == 1 and len(repo.list_sessions(owner_user_id="202")) == 1  # nosec B101 -- Retry creates exactly one complete session.


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_nested_explainer_transactions_do_not_commit_outer_failed_restore(stores, failure):
    _source, destination = stores
    repo = destination._get_explainer_repo()
    with pytest.raises(failure), repo.db.transaction():
        created = session(repo, 202)
        assert repo.get_session(created.id, owner_user_id="202") is not None  # nosec B101 -- A genuine inner write happened.
        raise failure("Synthetic outer interruption")
    assert repo.list_sessions(owner_user_id="202") == []  # nosec B101 -- Inner repository commits cannot escape outer failure.
    session(repo, 202)
    assert len(repo.list_sessions(owner_user_id="202")) == 1  # nosec B101 -- Connection remains usable after rollback.


async def test_archived_explainer_is_not_an_active_skip_candidate(stores):
    source, destination = stores
    archive = await explainer_archive(source, destination, "first-class")
    repo = destination._get_explainer_repo()
    old = session(repo, 202)
    repo.archive_session(old.id, owner_user_id="202")
    okay, message, result = await destination.import_chatbook(
        str(archive),
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=False,
        async_mode=False,
    )
    assert okay, message  # nosec B101 -- Archived data must not block restoring active content.
    assert result["successful_items"] == 1  # nosec B101 -- Count the new active session.
    assert len(repo.list_sessions(owner_user_id="202", include_archived=True)) == 2  # nosec B101 -- Preserve archived history.


@pytest.mark.parametrize("case", ["duplicate-uuid", "malformed-provenance", "foreign-owner"])
async def test_unqualified_archive_media_aliases_do_not_bind(stores, case):
    _source, destination = stores
    media = destination._get_media_db()
    existing = None
    if case != "duplicate-uuid":
        existing, _uuid, _ = media.add_media_with_keywords(
            url="https://example.test/existing",
            title="Alias source a",
            media_type="document",
            content="Keep existing body.",
            keywords=[],
            owner_user_id=303 if case == "foreign-owner" else 202,
            safe_metadata=json.dumps(
                {"chatbook_import": {"source_uuid": "shared-source-uuid"} if case == "foreign-owner" else None}
            ),
        )
    members = {}
    items = []
    source_ids = ["a", "b"] if case == "duplicate-uuid" else ["a"]
    for source_id in source_ids:
        path = f"content/media/media_{source_id}.json"
        members[path] = {
            "id": source_id,
            "uuid": "shared-source-uuid",
            "title": f"Alias source {source_id}",
            "url": f"https://example.test/source-{source_id}",
            "type": "document",
            "content": f"Imported body {source_id}.",
        }
        items.append(
            ContentItem(id=source_id, type=ContentType.MEDIA, title=f"Alias source {source_id}", file_path=path)
        )
    path = "content/embeddings/collection_aliases.json"
    members[path] = {
        "embedding_set_id": "alias_links",
        "chunks": [
            {
                "id": "uuid-only",
                "document": "Must not bind arbitrarily",
                "embedding": [1.0, 0.0],
                "metadata": {"media_uuid": "shared-source-uuid", "label": "keep"},
            },
            {
                "id": "id-only",
                "document": "Selected independent numeric identity",
                "embedding": [0.0, 1.0],
                "metadata": {"media_id": "a", "label": "id"},
            },
        ],
    }
    items.append(ContentItem(id="collection:alias_links", type=ContentType.EMBEDDING, title="Aliases", file_path=path))
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1, name="Alias archive", description="Public trust-boundary case", content_items=items
    )
    target = destination.import_dir / "aliases.chatbook"
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest.to_dict()))
        for name, payload in members.items():
            archive.writestr(name, json.dumps(payload))
    okay, message, result = await destination.import_chatbook(
        str(target),
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=False,
        async_mode=False,
    )
    assert okay, message  # nosec B101 -- Bad provenance must not turn ordinary Skip into a failed import.
    data = destination._get_chroma_manager().get_collection("alias_links").get(include=["metadatas"])
    metadata = dict(zip(data["ids"], data["metadatas"]))
    if case == "duplicate-uuid":
        first = media.get_media_by_title("Alias source a")
        second = media.get_media_by_title("Alias source b")
        assert first["id"] != second["id"] and first["uuid"] != second["uuid"]  # nosec B101 -- Require two actual records rather than normal content deduplication.
    assert metadata["uuid-only"] == {"label": "keep"}  # nosec B101 -- Never bind ambiguous/unqualified source UUIDs.
    if case == "duplicate-uuid":
        restored = media.get_media_by_title("Alias source a")
        assert (
            metadata["id-only"]["media_id"] == restored["id"] and metadata["id-only"]["media_uuid"] == restored["uuid"]
        )  # nosec B101 -- Independent unique source ID still maps correctly.
    else:
        assert metadata["id-only"] == {"label": "id"}  # nosec B101 -- No unqualified title/foreign-owner binding.
        assert get_media_by_id(media, existing)["content"] == "Keep existing body."  # nosec B101 -- Existing destination body is preserved.
    assert any("not imported" in warning for warning in result["warnings"])  # nosec B101 -- Truthful unresolved-link disclosure.
