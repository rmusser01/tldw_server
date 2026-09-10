"""Tests for Notes Studio sidecar storage and shared schema models."""

from __future__ import annotations

from datetime import datetime

import pytest

from tldw_Server_API.app.api.v1.schemas.notes_schemas import NoteResponse
from tldw_Server_API.app.api.v1.schemas.notes_studio import (
    NoteStudioDocumentCreateRequest,
    NoteStudioDocumentResponse,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError, InputError
from tldw_Server_API.app.core.Notes.studio_markdown import stable_content_hash
from tldw_Server_API.app.core.Sync.v2.notes_moodboard_studio_contract import (
    diagram_render_hash,
    notes_studio_document_object_hash,
    parse_notes_studio_document_v1,
)
from tldw_Server_API.app.core.Sync.v2.server_origin import canonical_payload_hash


@pytest.fixture
def db(tmp_path: pytest.TempPathFactory) -> CharactersRAGDB:
    database = CharactersRAGDB(db_path=str(tmp_path / "chacha_studio.db"), client_id="studio-user")
    yield database
    database.close_connection()


def test_note_studio_schema_models_validate_core_fields() -> None:
    request = NoteStudioDocumentCreateRequest(
        note_id="note-1",
        payload_json={"meta": {"source_note_id": "note-1"}, "sections": []},
        template_type="lined",
        handwriting_mode="accented",
        source_note_id="note-1",
        excerpt_snapshot="beta",
        excerpt_hash="sha256:demo",
        companion_content_hash="sha256:markdown",
        render_version=1,
    )

    assert request.note_id == "note-1"  # nosec B101
    assert request.template_type == "lined"  # nosec B101
    assert request.handwriting_mode == "accented"  # nosec B101
    assert request.render_version == 1  # nosec B101

    response = NoteStudioDocumentResponse.model_validate(
        {
            **request.model_dump(),
            "created_at": "2026-03-28T00:00:00Z",
            "last_modified": "2026-03-28T00:00:00Z",
        }
    )
    assert response.note_id == "note-1"  # nosec B101
    assert response.payload_json["meta"]["source_note_id"] == "note-1"  # nosec B101
    assert isinstance(response.created_at, datetime)  # nosec B101
    assert isinstance(response.last_modified, datetime)  # nosec B101
    assert response.created_at.isoformat().startswith("2026-03-28T00:00:00")  # nosec B101


def test_notes_db_creates_note_studio_documents_table(db: CharactersRAGDB) -> None:
    conn = db.get_connection()
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
        ("note_studio_documents",),
    ).fetchone()

    assert row is not None  # nosec B101
    assert row["name"] == "note_studio_documents"  # nosec B101


def test_create_and_fetch_note_studio_document_by_note_id(db: CharactersRAGDB) -> None:
    note_id = db.add_note(title="Source", content="Alpha beta gamma")

    created = db.create_note_studio_document(
        note_id=note_id,
        payload_json=_canonical_sections(),
        template_type="lined",
        handwriting_mode="accented",
        source_note_id=None,
        excerpt_snapshot=None,
        excerpt_hash=None,
        companion_content_hash="sha256:markdown",
        render_version=1,
    )

    assert created["note_id"] == note_id  # nosec B101
    assert created["owner_user_id"] == db.client_id  # nosec B101
    assert created["dataset_id"] == "local-unbound"  # nosec B101
    assert created["template_type"] == "lined"  # nosec B101
    assert created["handwriting_mode"] == "accented"  # nosec B101
    assert created["version"] == 1  # nosec B101
    assert created["canonical_revision"] == 1  # nosec B101
    assert created["canonical_hash"].startswith("sha256:")  # nosec B101

    studio = db.get_note_studio_document(note_id)
    assert studio is not None  # nosec B101
    assert studio["note_id"] == note_id  # nosec B101
    assert studio["template_type"] == "lined"  # nosec B101
    assert studio["handwriting_mode"] == "accented"  # nosec B101
    assert studio["payload_json"] == _canonical_sections()  # nosec B101


def test_create_note_studio_document_translates_duplicate_to_conflict(db: CharactersRAGDB) -> None:
    note_id = db.add_note(title="Source", content="Alpha beta gamma")
    payload = {
        "note_id": note_id,
        "payload_json": _canonical_sections(),
        "template_type": "lined",
        "handwriting_mode": "accented",
        "source_note_id": None,
        "excerpt_snapshot": None,
        "excerpt_hash": None,
        "companion_content_hash": "sha256:markdown",
        "render_version": 1,
    }

    db.create_note_studio_document(**payload)

    with pytest.raises(ConflictError, match="already exists"):
        db.create_note_studio_document(**payload)


def test_create_note_studio_document_translates_missing_note_to_conflict(db: CharactersRAGDB) -> None:
    with pytest.raises(ConflictError, match="Note not found"):
        db.create_note_studio_document(
            note_id="missing-note",
            payload_json={"meta": {}, "sections": []},
            template_type="lined",
            handwriting_mode="accented",
            source_note_id=None,
            excerpt_snapshot="beta",
            excerpt_hash="sha256:demo",
            companion_content_hash="sha256:markdown",
            render_version=1,
        )


def test_note_studio_write_requires_a_live_owned_source_note(db: CharactersRAGDB) -> None:
    note_id = db.add_note(title="Companion", content="Rendered")
    source_note_id = db.add_note(title="Source", content="Original")

    assert db.soft_delete_note(source_note_id, expected_version=1) is True  # nosec B101
    with pytest.raises(ConflictError, match="Source note not found or not live"):
        db.create_note_studio_document(
            note_id=note_id,
            payload_json={"meta": {"source_note_id": source_note_id}, "sections": []},
            template_type="lined",
            handwriting_mode="accented",
            source_note_id=source_note_id,
            excerpt_snapshot="Original",
            excerpt_hash="sha256:demo",
            companion_content_hash="sha256:markdown",
            render_version=1,
        )


def test_diagram_update_advances_scoped_studio_lineage(db: CharactersRAGDB) -> None:
    note_id = db.add_note(title="Source", content="Alpha beta gamma")
    before = db.create_note_studio_document(
        note_id=note_id,
        payload_json=_canonical_sections(),
        template_type="lined",
        handwriting_mode="accented",
        source_note_id=None,
        excerpt_snapshot=None,
        excerpt_hash=None,
        companion_content_hash="sha256:markdown",
        render_version=1,
    )

    after = db.update_note_studio_diagram_manifest(
        note_id=note_id,
        diagram_manifest_json=_canonical_diagram("Accepted content", "diagram-1"),
        expected_companion_content_hash=before["companion_content_hash"],
        expected_render_version=1,
        expected_last_modified=before["last_modified"],
    )

    assert after["owner_user_id"] == db.client_id  # nosec B101
    assert after["dataset_id"] == "local-unbound"  # nosec B101
    assert after["version"] == before["version"] + 1  # nosec B101
    assert after["canonical_revision"] == before["canonical_revision"] + 1  # nosec B101
    assert after["canonical_hash"] != before["canonical_hash"]  # nosec B101


def test_note_studio_document_rejects_non_dict_json_shapes(db: CharactersRAGDB) -> None:
    note_id = db.add_note(title="Source", content="Alpha beta gamma")

    with pytest.raises(InputError, match="payload_json must be a JSON object"):
        db.create_note_studio_document(
            note_id=note_id,
            payload_json=["unexpected", "list"],
            template_type="lined",
            handwriting_mode="accented",
            source_note_id=note_id,
            excerpt_snapshot="beta",
            excerpt_hash="sha256:demo",
            companion_content_hash="sha256:markdown",
            render_version=1,
        )

    with pytest.raises(InputError, match="diagram_manifest_json must be a JSON object"):
        db.create_note_studio_document(
            note_id=note_id,
            payload_json={"meta": {"source_note_id": note_id}, "sections": []},
            template_type="lined",
            handwriting_mode="accented",
            source_note_id=note_id,
            excerpt_snapshot="beta",
            excerpt_hash="sha256:demo",
            diagram_manifest_json='{"unexpected":"string"}',
            companion_content_hash="sha256:markdown",
            render_version=1,
        )


def test_upsert_note_studio_document_uses_explicit_transaction_connection(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    note_id = db.add_note(title="Source", content="Alpha beta gamma")

    monkeypatch.setattr(
        db,
        "execute_query",
        lambda *_args, **_kwargs: pytest.fail("write helper should read back through the explicit transaction connection"),
    )

    with db.transaction() as conn:
        created = db.upsert_note_studio_document(
            note_id=note_id,
            payload_json=_canonical_sections("Intro"),
            template_type="lined",
            handwriting_mode="accented",
            source_note_id=None,
            excerpt_snapshot=None,
            excerpt_hash=None,
            diagram_manifest_json=_canonical_diagram("Intro", "d-1"),
            companion_content_hash="sha256:markdown",
            render_version=1,
            conn=conn,
        )

        updated = db.upsert_note_studio_document(
            note_id=note_id,
            payload_json=_canonical_sections("Revised"),
            template_type="cornell",
            handwriting_mode="off",
            source_note_id=None,
            excerpt_snapshot=None,
            excerpt_hash=None,
            diagram_manifest_json=_canonical_diagram("Revised", "d-2"),
            companion_content_hash="sha256:markdown-2",
            render_version=1,
            conn=conn,
        )

    monkeypatch.undo()

    assert created["payload_json"]["sections"][0]["content"] == "Intro"  # nosec B101
    assert created["diagram_manifest_json"]["diagram"].endswith("d-1")  # nosec B101
    assert updated["template_type"] == "cornell"  # nosec B101
    assert updated["render_version"] == 1  # nosec B101

    persisted = db.get_note_studio_document(note_id)
    assert persisted is not None  # nosec B101
    assert persisted["payload_json"]["sections"][0]["content"] == "Revised"  # nosec B101
    assert persisted["diagram_manifest_json"]["diagram"].endswith("d-2")  # nosec B101


def test_studio_scope_resolution_occurs_inside_the_write_transaction(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_id = db.add_note(title="Source", content="Alpha beta gamma")
    original = db.resolve_studio_compatibility_dataset_id
    seen_connections: list[object | None] = []

    def recording_resolver(*, owner_user_id: str, conn=None):
        seen_connections.append(conn)
        return original(owner_user_id=owner_user_id, conn=conn)

    monkeypatch.setattr(db, "resolve_studio_compatibility_dataset_id", recording_resolver)
    db.create_note_studio_document(
        note_id=note_id,
        payload_json=_canonical_sections(),
        template_type="lined",
        handwriting_mode="accented",
        source_note_id=None,
        excerpt_snapshot=None,
        excerpt_hash=None,
        companion_content_hash="sha256:markdown",
        render_version=1,
    )

    assert seen_connections  # nosec B101
    assert all(conn is not None for conn in seen_connections)  # nosec B101


def test_add_and_update_note_accept_explicit_transaction_connection(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with db.transaction() as conn:
        monkeypatch.setattr(
            db,
            "transaction",
            lambda: pytest.fail("note helpers should use the explicit transaction connection"),
        )

        note_id = db.add_note(
            title="Source",
            content="Alpha beta gamma",
            conn=conn,
        )
        assert note_id is not None  # nosec B101

        updated = db.update_note(
            note_id=note_id,
            update_data={"title": "Updated Source", "content": "Updated alpha beta gamma"},
            expected_version=1,
            conn=conn,
        )
        assert updated is True  # nosec B101

    persisted = db.get_note_by_id(note_id)
    assert persisted is not None  # nosec B101
    assert persisted["title"] == "Updated Source"  # nosec B101
    assert persisted["content"] == "Updated alpha beta gamma"  # nosec B101


def test_soft_delete_preserves_sidecar_and_restore_reuses_same_row(db: CharactersRAGDB) -> None:
    note_id = db.add_note(title="Source", content="Alpha beta gamma")
    db.create_note_studio_document(
        note_id=note_id,
        payload_json={"sections": []},
        template_type="cornell",
        handwriting_mode="accented",
        source_note_id=None,
        excerpt_snapshot=None,
        excerpt_hash=None,
        companion_content_hash="sha256:markdown",
        render_version=1,
    )

    before_delete = db.get_note_studio_document(note_id)
    assert before_delete is not None  # nosec B101

    deleted = db.soft_delete_note(note_id, expected_version=1)
    assert deleted is True  # nosec B101

    after_delete = db.get_note_studio_document(note_id)
    assert after_delete is not None  # nosec B101
    assert after_delete["deleted"] == 1  # nosec B101
    assert after_delete["version"] == before_delete["version"] + 1  # nosec B101
    assert after_delete["canonical_revision"] == before_delete["canonical_revision"] + 1  # nosec B101
    assert after_delete["canonical_hash"] != before_delete["canonical_hash"]  # nosec B101
    for retained in (
        "payload_json",
        "note_revision",
        "note_hash",
        "accepted_provenance_json",
    ):
        assert after_delete[retained] == before_delete[retained]  # nosec B101

    restored = db.restore_note(note_id, expected_version=2)
    assert restored is True  # nosec B101

    after_restore = db.get_note_studio_document(note_id)
    assert after_restore is not None  # nosec B101
    assert after_restore["deleted"] == 0  # nosec B101
    assert after_restore["version"] == after_delete["version"] + 1  # nosec B101
    assert after_restore["canonical_revision"] == after_delete["canonical_revision"] + 1  # nosec B101
    assert after_restore["canonical_hash"] != after_delete["canonical_hash"]  # nosec B101
    for retained in (
        "payload_json",
        "note_revision",
        "note_hash",
        "accepted_provenance_json",
    ):
        assert after_restore[retained] == before_delete[retained]  # nosec B101


def test_sync_note_tombstone_and_restore_advance_studio_lifecycle(
    db: CharactersRAGDB,
) -> None:
    note_id = db.add_note(title="Source", content="Alpha beta gamma")
    db.create_note_studio_document(
        note_id=note_id,
        payload_json={"sections": []},
        template_type="cornell",
        handwriting_mode="accented",
        render_version=1,
    )

    assert db.tombstone_note_from_sync(  # nosec B101
        note_id=note_id,
        sync_client_id=str(db.client_id),
        object_revision=2,
        object_hash="sha256:" + "0" * 64,
    )
    tombstone = db.get_note_studio_document(note_id)
    assert tombstone is not None and tombstone["deleted"] == 1  # nosec B101

    assert db.upsert_note_from_sync(  # nosec B101
        note_id=note_id,
        title="Source",
        content="Alpha beta gamma",
        conversation_id=None,
        message_id=None,
        sync_client_id=str(db.client_id),
        object_revision=3,
        object_hash="sha256:" + "1" * 64,
    )
    restored = db.get_note_studio_document(note_id)
    assert restored is not None and restored["deleted"] == 0  # nosec B101
    assert restored["canonical_revision"] == tombstone["canonical_revision"] + 1  # nosec B101


def test_hard_delete_removes_sidecar(db: CharactersRAGDB) -> None:
    note_id = db.add_note(title="Source", content="Alpha beta gamma")
    db.create_note_studio_document(
        note_id=note_id,
        payload_json=_canonical_sections(),
        template_type="grid",
        handwriting_mode="off",
        source_note_id=None,
        excerpt_snapshot=None,
        excerpt_hash=None,
        companion_content_hash="sha256:markdown",
        render_version=1,
    )

    deleted = db.delete_note(note_id, hard_delete=True)
    assert deleted is True  # nosec B101
    assert db.get_note_studio_document(note_id) is None  # nosec B101

    conn = db.get_connection()
    row = conn.execute(
        "SELECT note_id FROM note_studio_documents WHERE note_id = ?",
        (note_id,),
    ).fetchone()
    assert row is None  # nosec B101


def test_stale_state_hashes_are_persisted_and_compared_explicitly(db: CharactersRAGDB) -> None:
    note_id = db.add_note(title="Source", content="Alpha beta gamma")
    db.create_note_studio_document(
        note_id=note_id,
        payload_json=_canonical_sections(),
        template_type="lined",
        handwriting_mode="accented",
        source_note_id=None,
        excerpt_snapshot=None,
        excerpt_hash=None,
        companion_content_hash="sha256:markdown",
        render_version=1,
    )

    studio = db.get_note_studio_document(note_id)
    assert studio is not None  # nosec B101
    assert studio["excerpt_hash"] is None  # nosec B101
    assert studio["companion_content_hash"] == stable_content_hash("Alpha beta gamma")  # nosec B101
    assert studio["companion_content_hash"] != "sha256:changed-markdown"  # nosec B101


def test_note_fetch_can_include_lightweight_studio_summary(db: CharactersRAGDB) -> None:
    note_id = db.add_note(title="Source", content="Alpha beta gamma")
    db.create_note_studio_document(
        note_id=note_id,
        payload_json=_canonical_sections(),
        template_type="grid",
        handwriting_mode="off",
        source_note_id=None,
        excerpt_snapshot=None,
        excerpt_hash=None,
        companion_content_hash="sha256:markdown",
        render_version=1,
    )

    note = db.get_note_by_id(note_id, include_studio_summary=True)
    assert note is not None  # nosec B101
    assert note["studio"]["note_id"] == note_id  # nosec B101
    assert note["studio"]["template_type"] == "grid"  # nosec B101
    assert "payload_json" not in note["studio"]  # nosec B101

    validated = NoteResponse.model_validate(note)
    assert validated.studio is not None  # nosec B101
    assert validated.studio.note_id == note_id  # nosec B101
    assert validated.studio.template_type == "grid"  # nosec B101


def _canonical_sections(content: str = "Accepted content") -> dict[str, object]:
    return {
        "sections": [
            {
                "id": "notes-1",
                "kind": "notes",
                "title": "Notes",
                "content": content,
            }
        ]
    }


def _canonical_diagram(content: str, diagram_id: str) -> dict[str, object]:
    diagram = f"graph TD; A-->{diagram_id}"
    context = f"Notes\n{content}"
    return {
        "diagram_type": "flowchart",
        "source_section_ids": ["notes-1"],
        "source_graph": [
            {
                "id": "notes-1",
                "title": "Notes",
                "kind": "notes",
                "content": content,
            }
        ],
        "diagram": diagram,
        "format": "mermaid",
        "status": "ready",
        "render_hash": diagram_render_hash(
            diagram_type="flowchart", context=context, diagram=diagram
        ),
    }


def test_v61_studio_write_reduces_equal_legacy_aliases_and_rebuilds_all_hashes(
    db: CharactersRAGDB,
) -> None:
    source_id = db.add_note(title="Source", content="Alpha\r\nbeta gamma")
    note_id = db.add_note(title="Companion", content="# Companion\r\n\r\nAccepted content")
    assert source_id is not None and note_id is not None

    stored = db.create_note_studio_document(
        note_id=note_id,
        payload_json={
            "meta": {"title": "Companion", "source_note_id": source_id},
            "layout": {"template_type": "lined", "render_version": 1},
            **_canonical_sections(),
        },
        template_type="lined",
        handwriting_mode="accented",
        source_note_id=source_id,
        excerpt_snapshot="Alpha\r\nbeta",
        excerpt_hash="sha256:" + "0" * 64,
        companion_content_hash="sha256:" + "1" * 64,
        render_version=1,
    )

    assert stored["payload_json"] == _canonical_sections()  # nosec B101
    assert stored["excerpt_snapshot"] == "Alpha\nbeta"  # nosec B101
    assert stored["excerpt_hash"] == stable_content_hash("Alpha\nbeta")  # nosec B101
    assert stored["companion_content_hash"] == stable_content_hash(
        "# Companion\n\nAccepted content"
    )  # nosec B101
    note_hash, _ = canonical_payload_hash(
        {
            "title": "Companion",
            "content": "# Companion\r\n\r\nAccepted content",
            "conversation_id": None,
            "message_id": None,
        }
    )
    assert stored["note_hash"] == note_hash  # nosec B101
    parsed = parse_notes_studio_document_v1(
        {
            "note_id": stored["note_id"],
            "source_note_id": stored["source_note_id"],
            "payload_json": stored["payload_json"],
            "template_type": stored["template_type"],
            "handwriting_mode": stored["handwriting_mode"],
            "excerpt_snapshot": stored["excerpt_snapshot"],
            "excerpt_hash": stored["excerpt_hash"],
            "diagram_manifest_json": stored["diagram_manifest_json"],
            "companion_content_hash": stored["companion_content_hash"],
            "render_version": stored["render_version"],
            "note_revision": stored["note_revision"],
            "note_hash": stored["note_hash"],
            "accepted_provenance": stored["accepted_provenance_json"],
        },
        bound_attestation="server",
        bound_accepted_at=stored["accepted_provenance_json"]["accepted_at"],
    )
    assert stored["canonical_hash"] == notes_studio_document_object_hash(
        parsed,
        revision=stored["canonical_revision"],
        deleted=False,
    )  # nosec B101


@pytest.mark.parametrize(
    "overrides",
    [
        {"render_version": 2},
        {"payload_json": {"meta": {"title": "Wrong"}, **_canonical_sections()}},
        {"payload_json": {"layout": {"handwriting_mode": "off"}, **_canonical_sections()}},
        {"excerpt_snapshot": "not in source"},
        {
            "diagram_manifest_json": {
                "canonical_source": [],
                "source_graph": [{"id": "notes-1"}],
            }
        },
    ],
)
def test_v61_studio_write_rejects_noncanonical_or_untruthful_compatibility_input(
    db: CharactersRAGDB,
    overrides: dict[str, object],
) -> None:
    source_id = db.add_note(title="Source", content="source excerpt")
    note_id = db.add_note(title="Companion", content="Accepted content")
    assert source_id is not None and note_id is not None
    values: dict[str, object] = {
        "note_id": note_id,
        "payload_json": _canonical_sections(),
        "template_type": "lined",
        "handwriting_mode": "accented",
        "source_note_id": source_id,
        "excerpt_snapshot": "source excerpt",
        "excerpt_hash": stable_content_hash("source excerpt"),
        "diagram_manifest_json": None,
        "companion_content_hash": stable_content_hash("Accepted content"),
        "render_version": 1,
    }
    values.update(overrides)
    with pytest.raises(InputError):
        db.create_note_studio_document(**values)


def test_v61_studio_write_rejects_effective_sync_envelope_over_limit(
    db: CharactersRAGDB,
) -> None:
    note_id = db.add_note(title="Companion", content="Accepted content")
    assert note_id is not None
    with pytest.raises(InputError, match="262144"):
        db.create_note_studio_document(
            note_id=note_id,
            payload_json=_canonical_sections("x" * 65_536),
            template_type="lined",
            handwriting_mode="accented",
            source_note_id=None,
            excerpt_snapshot=None,
            excerpt_hash=None,
            diagram_manifest_json={
                "diagram_type": "flowchart",
                "source_section_ids": ["notes-1"],
                "source_graph": [
                    {
                        "id": "notes-1",
                        "title": "Notes",
                        "kind": "notes",
                        "content": "x" * 65_536,
                    }
                ],
                "diagram": "y" * 131_072,
                "format": "mermaid",
                "status": "ready",
                "render_hash": "sha256:" + "2" * 64,
            },
            companion_content_hash=stable_content_hash("Accepted content"),
            render_version=1,
        )
