"""Tests for Notes Studio sidecar storage and shared schema models."""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest

from tldw_Server_API.app.api.v1.schemas.notes_schemas import NoteResponse
from tldw_Server_API.app.api.v1.schemas.notes_studio import (
    NoteStudioDocumentCreateRequest,
    NoteStudioDocumentResponse,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError, InputError


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
        payload_json={"meta": {"source_note_id": note_id}, "sections": []},
        template_type="lined",
        handwriting_mode="accented",
        source_note_id=note_id,
        excerpt_snapshot="beta",
        excerpt_hash="sha256:demo",
        companion_content_hash="sha256:markdown",
        render_version=1,
    )

    assert created["note_id"] == note_id  # nosec B101
    assert created["template_type"] == "lined"  # nosec B101
    assert created["handwriting_mode"] == "accented"  # nosec B101

    studio = db.get_note_studio_document(note_id)
    assert studio is not None  # nosec B101
    assert studio["note_id"] == note_id  # nosec B101
    assert studio["template_type"] == "lined"  # nosec B101
    assert studio["handwriting_mode"] == "accented"  # nosec B101
    assert studio["payload_json"]["meta"]["source_note_id"] == note_id  # nosec B101


def test_create_note_studio_document_translates_duplicate_to_conflict(db: CharactersRAGDB) -> None:
    note_id = db.add_note(title="Source", content="Alpha beta gamma")
    payload = {
        "note_id": note_id,
        "payload_json": {"meta": {"source_note_id": note_id}, "sections": []},
        "template_type": "lined",
        "handwriting_mode": "accented",
        "source_note_id": note_id,
        "excerpt_snapshot": "beta",
        "excerpt_hash": "sha256:demo",
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
            payload_json={"meta": {"source_note_id": note_id}, "sections": [{"title": "Intro"}]},
            template_type="lined",
            handwriting_mode="accented",
            source_note_id=note_id,
            excerpt_snapshot="beta",
            excerpt_hash="sha256:demo",
            diagram_manifest_json={"diagrams": [{"id": "d-1"}]},
            companion_content_hash="sha256:markdown",
            render_version=1,
            conn=conn,
        )

        updated = db.upsert_note_studio_document(
            note_id=note_id,
            payload_json={"meta": {"source_note_id": note_id}, "sections": [{"title": "Revised"}]},
            template_type="cornell",
            handwriting_mode="off",
            source_note_id=note_id,
            excerpt_snapshot="gamma",
            excerpt_hash="sha256:demo-2",
            diagram_manifest_json={"diagrams": [{"id": "d-2"}]},
            companion_content_hash="sha256:markdown-2",
            render_version=2,
            conn=conn,
        )

    monkeypatch.undo()

    assert created["payload_json"]["sections"][0]["title"] == "Intro"  # nosec B101
    assert created["diagram_manifest_json"]["diagrams"][0]["id"] == "d-1"  # nosec B101
    assert updated["template_type"] == "cornell"  # nosec B101
    assert updated["render_version"] == 2  # nosec B101

    persisted = db.get_note_studio_document(note_id)
    assert persisted is not None  # nosec B101
    assert persisted["payload_json"]["sections"][0]["title"] == "Revised"  # nosec B101
    assert persisted["diagram_manifest_json"]["diagrams"][0]["id"] == "d-2"  # nosec B101


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
        payload_json={"meta": {"source_note_id": note_id}, "sections": []},
        template_type="cornell",
        handwriting_mode="accented",
        source_note_id=note_id,
        excerpt_snapshot="beta",
        excerpt_hash="sha256:demo",
        companion_content_hash="sha256:markdown",
        render_version=1,
    )

    before_delete = db.get_note_studio_document(note_id)
    assert before_delete is not None  # nosec B101

    deleted = db.soft_delete_note(note_id, expected_version=1)
    assert deleted is True  # nosec B101

    after_delete = db.get_note_studio_document(note_id)
    assert after_delete is not None  # nosec B101
    assert after_delete == before_delete  # nosec B101

    restored = db.restore_note(note_id, expected_version=2)
    assert restored is True  # nosec B101

    after_restore = db.get_note_studio_document(note_id)
    assert after_restore is not None  # nosec B101
    assert after_restore == before_delete  # nosec B101


def test_hard_delete_removes_sidecar(db: CharactersRAGDB) -> None:
    note_id = db.add_note(title="Source", content="Alpha beta gamma")
    db.create_note_studio_document(
        note_id=note_id,
        payload_json={"meta": {"source_note_id": note_id}, "sections": []},
        template_type="grid",
        handwriting_mode="off",
        source_note_id=note_id,
        excerpt_snapshot="beta",
        excerpt_hash="sha256:demo",
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
        payload_json={"meta": {"source_note_id": note_id}, "sections": []},
        template_type="lined",
        handwriting_mode="accented",
        source_note_id=note_id,
        excerpt_snapshot="beta",
        excerpt_hash="sha256:excerpt",
        companion_content_hash="sha256:markdown",
        render_version=1,
    )

    studio = db.get_note_studio_document(note_id)
    assert studio is not None  # nosec B101
    assert studio["excerpt_hash"] == "sha256:excerpt"  # nosec B101
    assert studio["companion_content_hash"] == "sha256:markdown"  # nosec B101
    assert studio["companion_content_hash"] != "sha256:changed-markdown"  # nosec B101


def test_note_fetch_can_include_lightweight_studio_summary(db: CharactersRAGDB) -> None:
    note_id = db.add_note(title="Source", content="Alpha beta gamma")
    db.create_note_studio_document(
        note_id=note_id,
        payload_json={"meta": {"source_note_id": note_id}, "sections": []},
        template_type="grid",
        handwriting_mode="off",
        source_note_id=note_id,
        excerpt_snapshot="beta",
        excerpt_hash="sha256:excerpt",
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
