"""Portable workspaces require verified restore identities and exact PDF locations."""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.api.v1.endpoints.chatbooks import _import_job_response
from tldw_Server_API.app.core.Chatbooks.chatbook_models import (
    ChatbookManifest,
    ChatbookVersion,
    ConflictResolution,
    ContentItem,
    ContentType,
    ImportJob,
    ImportStatus,
)
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService, record_failed_import_result
from tldw_Server_API.app.core.Chatbooks.services import jobs_worker
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import get_document_version
from tldw_Server_API.tests.Chatbooks.test_chatbooks_retry_identity import stores  # noqa: F401

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("valid_map", [True, False])
async def test_real_media_note_roundtrip_returns_destination_ids_and_exact_page_map(stores, valid_map):  # noqa: F811
    source, destination = stores
    text = "\nFirst PDF page.\n\nSecond PDF page.\n"
    page_map = [
        {"page_number": 1, "start_char": 0, "end_char": 18, "printed_page_label": "i"},
        {"page_number": 2, "start_char": 18, "end_char": len(text), "printed_page_label": "1"},
    ]
    metadata = {
        "page_map": page_map,
        "page_count": 2,
        "page_map_content_hash": hashlib.sha256((text if valid_map else "stale").encode()).hexdigest(),
        "api_key": "must-not-export",
        "source_path": "/private/server/original.pdf",
        "nested": {"credential": "must-not-export"},
    }
    media_id, media_uuid, _ = source._get_media_db().add_media_with_keywords(
        url="https://example.test/portable.pdf",
        title="Portable PDF",
        media_type="document",
        content=text,
        keywords=[],
        owner_user_id=101,
        safe_metadata=json.dumps(metadata),
    )
    note_id = source.db.add_note("Portable note", "Exact quotation and review.")
    original = DatabasePaths.resolve_user_base_directory(101) / "original.pdf"
    original.write_bytes(b"original portable PDF bytes")
    source._get_media_db().insert_media_file(
        media_id=media_id,
        file_type="original",
        storage_path="original.pdf",
        original_filename="original.pdf",
    )
    collision, _, _ = destination._get_media_db().add_media_with_keywords(
        url="https://example.test/unrelated",
        title="Unrelated",
        media_type="document",
        content="Destination private text",
        keywords=[],
        owner_user_id=202,
    )
    assert collision == media_id
    selections = {ContentType.MEDIA: [str(media_id)], ContentType.NOTE: [note_id]}
    okay, message, archive_path = await source.create_chatbook(
        name="Portable",
        description="Selected content only",
        format_version=ChatbookVersion.V1_1,
        content_selections=selections,
        include_media=True,
        include_embeddings=False,
        async_mode=False,
    )
    assert okay, message
    with zipfile.ZipFile(archive_path) as archive:
        exported = json.loads(archive.read(f"content/media/media_{media_id}.json"))
    assert exported["safe_metadata"] == (
        {
            "page_map": page_map,
            "page_count": 2,
            "page_map_content_hash": metadata["page_map_content_hash"],
        }
        if valid_map
        else {}
    )
    target = destination.import_dir / "portable.chatbook"
    target.write_bytes(Path(archive_path).read_bytes())
    okay, message, result = await destination.import_chatbook(
        str(target),
        content_selections=selections,
        conflict_resolution=ConflictResolution.RENAME,
        prefix_imported=False,
        async_mode=False,
    )
    assert okay, message
    restored_id = result["id_maps"]["media"][str(media_id)]
    assert restored_id != collision
    assert result["id_maps"]["media"][media_uuid] == restored_id
    restored_note = destination.db.get_note_by_id(result["id_maps"]["notes"][note_id])
    assert restored_note["content"] == "Exact quotation and review."
    version = get_document_version(destination._get_media_db(), restored_id)
    assert version["content"] == text
    restored_metadata = json.loads(version["safe_metadata"])
    assert restored_metadata.get("page_map") == (page_map if valid_map else None)
    assert "must-not-export" not in json.dumps(restored_metadata)
    assert "/private/server" not in json.dumps(restored_metadata)
    # Verified Skip may reuse an identical import, but not a modified original.
    okay, message, retry = await destination.import_chatbook(
        str(target),
        content_selections={ContentType.MEDIA: [str(media_id)]},
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=False,
        async_mode=False,
    )
    assert okay, message
    assert retry["id_maps"]["media"][str(media_id)] == restored_id
    stored_file = destination._get_media_db().get_media_files(restored_id)[0]
    restored_original = DatabasePaths.resolve_user_base_directory(202) / stored_file["storage_path"]
    assert restored_original.read_bytes() == original.read_bytes()
    restored_original.write_bytes(b"changed destination original")
    okay, message, retry = await destination.import_chatbook(
        str(target),
        content_selections={ContentType.MEDIA: [str(media_id)]},
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=False,
        async_mode=False,
    )
    assert okay, message
    assert retry["id_maps"]["media"] == {}


@pytest.mark.parametrize("terminal", [ImportStatus.FAILED, ImportStatus.CANCELLED])
def test_partial_receipt_survives_terminal_job_projection(terminal):
    job = ImportJob(job_id="partial", user_id="202", status=terminal, chatbook_path="pack.chatbook")
    maps = {"media": {"1": 8, "ambiguous": None}, "notes": {"old": "new"}}
    record_failed_import_result(
        job,
        {
            "id_maps": maps,
            "successful_items": 2,
            "failed_items": 1,
            "_account_restore_payload": {"secret": "never-persist"},
        },
    )
    public = _import_job_response(job).model_dump()
    assert public["id_maps"] == maps
    assert public["status"] == terminal
    assert "never-persist" not in json.dumps(public, default=str)


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["legacy", "jobs"])
@pytest.mark.parametrize("cancelled", [False, True])
async def test_async_import_persists_verified_maps_without_overwriting_cancellation(
    stores,  # noqa: F811
    monkeypatch,
    lane,
    cancelled,
):
    _, service = stores
    job = ImportJob(job_id="legacy-async", user_id="202", status=ImportStatus.PENDING, chatbook_path="pack.chatbook")
    service._save_import_job(job)
    maps = {"media": {"1": 8}, "notes": {"old": "new"}}

    def restore(*_):
        if cancelled:
            current = service._get_import_job(job.job_id)
            current.status = ImportStatus.CANCELLED
            service._save_import_job(current)
        return True, "ok", {"imported_items": {"media": 1, "note": 1}, "id_maps": maps}

    monkeypatch.setattr(service, "_import_chatbook_sync", restore)
    if lane == "legacy":
        await service._import_chatbook_async(
            job.job_id, "pack.chatbook", None, ConflictResolution.SKIP, False, True, False
        )
    else:
        archive = service.import_dir / "pack.chatbook"
        archive.write_bytes(b"unused; actual importer replaced")
        await jobs_worker._handle_import(service, {"file_token": str(archive)}, job.job_id)
    public = _import_job_response(service._get_import_job(job.job_id)).model_dump()
    assert public["id_maps"] == maps
    assert public["status"] == (ImportStatus.CANCELLED if cancelled else ImportStatus.COMPLETED)


@given(st.lists(st.integers(min_value=1, max_value=1000), min_size=1, max_size=25))
def test_alias_is_usable_only_when_all_restores_agree(destination_ids):
    mapping = {}
    for media_id in destination_ids:
        ChatbookService._record_imported_media_ids(mapping, "1", "source-uuid", media_id)
    expected = destination_ids[0] if len(set(destination_ids)) == 1 else None
    assert mapping == {"1": expected, "source-uuid": expected}


@pytest.mark.parametrize("title_match", [True, False])
def test_skip_does_not_map_previously_imported_but_edited_canonical_text(stores, tmp_path, title_match):  # noqa: F811
    _, service = stores
    service._get_media_db().add_media_with_keywords(
        url="https://example.test/edited",
        title="Imported then edited",
        media_type="document",
        content="Edited destination text",
        keywords=[],
        owner_user_id=202,
        safe_metadata=json.dumps({"chatbook_import": {"source_uuid": "source-uuid"}}),
    )
    payload = tmp_path / "content/media/media_1.json"
    payload.parent.mkdir(parents=True)
    payload.write_text(
        json.dumps(
            {
                "id": 1,
                "uuid": "source-uuid",
                "title": "Imported then edited" if title_match else "Different archive title",
                "url": "https://example.test/edited",
                "content": "Original citation text",
                "type": "document",
            }
        )
    )
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1,
        name="Edited",
        description="No stale mapping",
        content_items=[
            ContentItem(
                id="1", type=ContentType.MEDIA, title="Imported then edited", file_path="content/media/media_1.json"
            ),
        ],
    )
    status = ImportJob(job_id="edited", user_id="202", status=ImportStatus.IN_PROGRESS, chatbook_path="pack.chatbook")
    mapping = {}
    service._import_media_items(tmp_path, manifest, ["1"], ConflictResolution.SKIP, False, status, media_id_map=mapping)
    assert mapping == {}
    assert (status.skipped_items, status.failed_items) == ((1, 0) if title_match else (0, 1))


def test_failed_original_restore_never_yields_usable_mapping_even_after_skip_retry(stores, tmp_path, monkeypatch):  # noqa: F811
    _, service = stores
    payload = tmp_path / "content/media/media_1.json"
    payload.parent.mkdir(parents=True)
    original = payload.parent / "original.pdf"
    original.write_bytes(b"original PDF fixture")
    payload.write_text(
        json.dumps(
            {
                "id": 1,
                "uuid": "source-uuid",
                "title": "Original failure",
                "content": "Exact text",
                "type": "document",
                "stored_artifacts": [
                    {
                        "bundled": True,
                        "file_type": "original",
                        "original_filename": "original.pdf",
                        "archive_path": "content/media/original.pdf",
                    }
                ],
            }
        )
    )
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1,
        name="Original failure",
        description="Partial restore",
        content_items=[
            ContentItem(
                id="1", type=ContentType.MEDIA, title="Original failure", file_path="content/media/media_1.json"
            ),
        ],
    )

    def fail_insert(**_kwargs):
        raise OSError("Synthetic MediaFiles write failure")

    monkeypatch.setattr(service._get_media_db(), "insert_media_file", fail_insert)
    for attempt in range(2):
        status = ImportJob(
            job_id=f"partial-{attempt}", user_id="202", status=ImportStatus.IN_PROGRESS, chatbook_path="pack.chatbook"
        )
        mapping = {}
        service._import_media_items(
            tmp_path, manifest, ["1"], ConflictResolution.SKIP, False, status, media_id_map=mapping
        )
        assert mapping == {}
        assert (status.failed_items, status.skipped_items) == ((1, 0) if attempt == 0 else (0, 1))
