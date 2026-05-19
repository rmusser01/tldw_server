"""Tests for web clipper sidecar storage and schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.web_clipper_schemas import (
    WebClipperEnrichmentPayload,
    WebClipperOutcomeState,
    WebClipperSaveRequest,
    WebClipperSaveResult,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="clipper-client")
    database.upsert_workspace("ws-1", "Research Workspace")
    yield database
    database.close_connection()


def test_web_clipper_document_round_trip_and_table_creation(db):
    tables = {
        row["name"]
        for row in db.execute_query("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }
    assert "note_clipper_documents" in tables
    assert "note_clipper_workspace_placements" in tables

    note_id = db.add_note(title="Clip", content="Visible body", note_id="clip-123")
    document = db.upsert_note_clipper_document(
        clip_id="clip-123",
        note_id=note_id,
        clip_type="article",
        source_url="https://example.com/story",
        source_title="Example Story",
        capture_metadata={"fallback_path": ["article"]},
        enrichments={"ocr": {"status": "pending"}},
        content_budget={"inline_body_chars": 1200},
        source_note_version=1,
    )

    clip_doc_by_clip_id = db.get_note_clipper_document_by_clip_id("clip-123")
    clip_doc_by_note_id = db.get_note_clipper_document_by_note_id(note_id)

    assert document["clip_id"] == "clip-123"
    assert clip_doc_by_clip_id is not None
    assert clip_doc_by_clip_id["note_id"] == note_id
    assert clip_doc_by_clip_id["clip_type"] == "article"
    assert clip_doc_by_clip_id["capture_metadata_json"] == {"fallback_path": ["article"]}
    assert clip_doc_by_clip_id["analysis_json"] == {"ocr": {"status": "pending"}}
    assert clip_doc_by_note_id is not None
    assert clip_doc_by_note_id["clip_id"] == "clip-123"


def test_web_clipper_workspace_placement_upsert_is_idempotent(db):
    note_id = db.add_note(title="Clip", content="Visible body", note_id="clip-456")
    db.upsert_note_clipper_document(
        clip_id="clip-456",
        note_id=note_id,
        clip_type="bookmark",
        source_url="https://example.com/bookmark",
        source_title="Bookmark",
        capture_metadata={"fallback_path": ["bookmark"]},
        enrichments={},
        content_budget={"inline_body_chars": 120},
        source_note_version=1,
    )

    first = db.upsert_note_clipper_workspace_placement(
        clip_id="clip-456",
        workspace_id="ws-1",
        workspace_note_id=41,
        source_note_id=note_id,
        source_note_version=1,
    )
    second = db.upsert_note_clipper_workspace_placement(
        clip_id="clip-456",
        workspace_id="ws-1",
        workspace_note_id=42,
        source_note_id=note_id,
        source_note_version=1,
    )
    placements = db.list_note_clipper_workspace_placements("clip-456")

    assert first["workspace_note_id"] == 41
    assert second["workspace_note_id"] == 42
    assert len(placements) == 1
    assert placements[0]["clip_id"] == "clip-456"
    assert placements[0]["workspace_id"] == "ws-1"
    assert placements[0]["workspace_note_id"] == 42
    assert placements[0]["source_note_id"] == note_id


def test_web_clipper_cleanup_follows_canonical_note_deletion(db):
    note_id = db.add_note(title="Clip", content="Visible body", note_id="clip-789")
    db.upsert_note_clipper_document(
        clip_id="clip-789",
        note_id=note_id,
        clip_type="screenshot",
        source_url="https://example.com/screenshot",
        source_title="Screenshot",
        capture_metadata={"fallback_path": ["screenshot"]},
        enrichments={},
        content_budget={"inline_body_chars": 0},
        source_note_version=1,
    )
    db.upsert_note_clipper_workspace_placement(
        clip_id="clip-789",
        workspace_id="ws-1",
        workspace_note_id=99,
        source_note_id=note_id,
        source_note_version=1,
    )

    assert db.get_note_clipper_document_by_clip_id("clip-789") is not None
    assert db.list_note_clipper_workspace_placements("clip-789")

    assert db.delete_note(note_id, hard_delete=True) is True

    assert db.get_note_clipper_document_by_clip_id("clip-789") is None
    assert db.get_note_clipper_document_by_note_id(note_id) is None
    assert db.list_note_clipper_workspace_placements("clip-789") == []


def test_web_clipper_soft_delete_invalidates_and_restore_reactivates_sidecars(db):
    note_id = db.add_note(title="Clip", content="Visible body", note_id="clip-soft")
    db.upsert_note_clipper_document(
        clip_id="clip-soft",
        note_id=note_id,
        clip_type="article",
        source_url="https://example.com/article",
        source_title="Soft Delete Story",
        capture_metadata={"fallback_path": ["article"]},
        enrichments={"ocr": {"status": "pending"}},
        content_budget={"inline_body_chars": 500},
        source_note_version=1,
    )
    db.upsert_note_clipper_workspace_placement(
        clip_id="clip-soft",
        workspace_id="ws-1",
        workspace_note_id=7,
        source_note_id=note_id,
        source_note_version=1,
    )

    assert db.soft_delete_note(note_id, expected_version=1) is True
    assert db.get_note_clipper_document_by_clip_id("clip-soft") is None
    assert db.list_note_clipper_workspace_placements("clip-soft") == []

    conn = db.get_connection()
    deleted_document = conn.execute(
        "SELECT deleted FROM note_clipper_documents WHERE clip_id = ?",
        ("clip-soft",),
    ).fetchone()
    deleted_placement = conn.execute(
        "SELECT deleted FROM note_clipper_workspace_placements WHERE clip_id = ? AND workspace_id = ?",
        ("clip-soft", "ws-1"),
    ).fetchone()
    assert deleted_document is not None
    assert bool(deleted_document["deleted"]) is True
    assert deleted_placement is not None
    assert bool(deleted_placement["deleted"]) is True

    assert db.restore_note(note_id, expected_version=2) is True

    restored_document = db.get_note_clipper_document_by_clip_id("clip-soft")
    restored_placements = db.list_note_clipper_workspace_placements("clip-soft")
    assert restored_document is not None
    assert restored_document["note_id"] == note_id
    assert restored_placements == [
        {
            "clip_id": "clip-soft",
            "workspace_id": "ws-1",
            "workspace_note_id": 7,
            "source_note_id": note_id,
            "source_note_version": 1,
        }
    ]


def test_web_clipper_helpers_use_explicit_transaction_connection(db, monkeypatch: pytest.MonkeyPatch):
    note_id = db.add_note(title="Clip", content="Visible body", note_id="clip-txn")

    monkeypatch.setattr(
        db,
        "execute_query",
        lambda *_args, **_kwargs: pytest.fail(
            "web clipper helpers should use the explicit transaction connection for reads and writes"
        ),
    )

    with db.transaction() as conn:
        document = db.upsert_note_clipper_document(
            clip_id="clip-txn",
            note_id=note_id,
            clip_type="article",
            source_url="https://example.com/txn",
            source_title="Transactional Story",
            capture_metadata={"fallback_path": ["article"]},
            enrichments={},
            content_budget={"inline_body_chars": 100},
            source_note_version=1,
            conn=conn,
        )
        placement = db.upsert_note_clipper_workspace_placement(
            clip_id="clip-txn",
            workspace_id="ws-1",
            workspace_note_id=21,
            source_note_id=note_id,
            source_note_version=1,
            conn=conn,
        )

    assert document["clip_id"] == "clip-txn"
    assert placement["workspace_note_id"] == 21


def test_web_clipper_schema_models_accept_minimal_payloads():
    request = WebClipperSaveRequest(
        clip_id="clip-abc",
        clip_type="article",
        source_url="https://example.com/article",
        source_title="Example Article",
        destination_mode="both",
        note=WebClipperSaveRequest.NotePayload(
            title="Example Article",
            folder_id=3,
            keywords=["example"],
        ),
        workspace=WebClipperSaveRequest.WorkspacePayload(workspace_id="ws-1"),
        content=WebClipperSaveRequest.ContentPayload(
            visible_body="Visible body",
            full_extract="Visible body plus more context",
        ),
        attachments=[
            WebClipperSaveRequest.AttachmentPayload(
                slot="page-screenshot",
                media_type="image/png",
                source_url="https://example.com/article.png",
            )
        ],
        enhancements=WebClipperSaveRequest.EnhancementOptions(run_ocr=True, run_vlm=False),
    )
    enrichment = WebClipperEnrichmentPayload(
        clip_id="clip-abc",
        enrichment_type="ocr",
        status="pending",
        inline_summary="Detected headline and body text.",
        structured_payload={"text": "Detected headline and body text."},
        source_note_version=1,
    )
    result = WebClipperSaveResult(
        clip_id="clip-abc",
        note_id="clip-abc",
        status="saved",
    )

    assert request.destination_mode == "both"
    assert request.workspace is not None
    assert request.workspace.workspace_id == "ws-1"
    assert request.note.folder_id == 3
    assert request.enhancements.run_ocr is True
    assert enrichment.status == "pending"
    assert enrichment.source_note_version == 1
    assert result.status == "saved"
    assert WebClipperOutcomeState.__args__ == ("saved", "saved_with_warnings", "partially_saved", "failed")


def test_web_clipper_schema_models_require_workspace_target_for_workspace_destinations():
    with pytest.raises(ValidationError, match="workspace is required"):
        WebClipperSaveRequest(
            clip_id="clip-invalid",
            clip_type="article",
            source_url="https://example.com/article",
            source_title="Example Article",
            destination_mode="workspace",
        )


def test_web_clipper_enrichment_requires_source_note_version():
    with pytest.raises(ValidationError, match="source_note_version"):
        WebClipperEnrichmentPayload(
            clip_id="clip-invalid",
            enrichment_type="vlm",
            status="pending",
        )
