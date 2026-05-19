"""Unit tests for web clipper service orchestration."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from tldw_Server_API.app.api.v1.schemas.web_clipper_schemas import (
    WebClipperEnrichmentPayload,
    WebClipperSaveRequest,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
)
from tldw_Server_API.app.core.WebClipper import service as web_clipper_service_module
from tldw_Server_API.app.core.WebClipper.service import WebClipperService


pytestmark = pytest.mark.unit


@pytest.fixture()
def clipper_db(tmp_path: Path):
    db = CharactersRAGDB(str(tmp_path / "web_clipper_unit.db"), client_id="web_clipper_unit")
    db.upsert_workspace("ws-1", "Research Workspace")
    yield db
    db.close_connection()


def _save_request(
    *,
    clip_id: str = "clip-123",
    destination_mode: str = "both",
    include_attachment: bool = False,
    full_extract: str | None = "Alpha paragraph.\n\nBeta paragraph.",
) -> WebClipperSaveRequest:
    attachments: list[WebClipperSaveRequest.AttachmentPayload] = []
    if include_attachment:
        attachments.append(
            WebClipperSaveRequest.AttachmentPayload(
                slot="page-screenshot",
                file_name="page-screenshot.txt",
                media_type="text/plain",
                text_content="captured attachment payload",
            )
        )
    return WebClipperSaveRequest(
        clip_id=clip_id,
        clip_type="article",
        source_url="https://example.com/story",
        source_title="Example Story",
        destination_mode=destination_mode,  # type: ignore[arg-type]
        note=WebClipperSaveRequest.NotePayload(
            title="Example Story",
            comment="Saved from the browser clipper.",
            keywords=["example"],
        ),
        workspace=(
            WebClipperSaveRequest.WorkspacePayload(workspace_id="ws-1")
            if destination_mode in {"workspace", "both"}
            else None
        ),
        content=WebClipperSaveRequest.ContentPayload(
            visible_body="Alpha paragraph.",
            full_extract=full_extract,
            selected_text="Alpha paragraph.",
        ),
        attachments=attachments,
        enhancements=WebClipperSaveRequest.EnhancementOptions(run_ocr=False, run_vlm=False),
        capture_metadata={"fallback_path": ["article"]},
    )


def test_save_clip_creates_canonical_note_and_sidecar(clipper_db):
    service = WebClipperService(db=clipper_db, user_id=1)

    result = service.save_clip(_save_request())

    assert result.status == "saved"
    assert result.note.id == "clip-123"
    assert result.note.title == "Example Story"
    assert result.workspace_placement is not None
    assert result.workspace_placement.workspace_id == "ws-1"

    note = clipper_db.get_note_by_id("clip-123")
    assert note is not None
    assert note["title"] == "Example Story"
    assert "Capture date:" in note["content"]
    assert "Clip type: article" in note["content"]

    clip_doc = clipper_db.get_note_clipper_document_by_clip_id("clip-123")
    assert clip_doc is not None
    assert clip_doc["note_id"] == "clip-123"


def test_save_clip_retry_reuses_note_workspace_and_attachment(clipper_db):
    service = WebClipperService(db=clipper_db, user_id=1)
    request = _save_request(include_attachment=True)

    first = service.save_clip(request)
    second = service.save_clip(request)
    status = service.get_clip_status("clip-123")

    assert first.note.id == second.note.id == "clip-123"
    assert first.workspace_placement is not None
    assert second.workspace_placement is not None
    assert first.workspace_placement.workspace_note_id == second.workspace_placement.workspace_note_id
    assert len(status.workspace_placements) == 1
    assert len(status.attachments) == 1
    assert status.attachments[0].slot == "page-screenshot"


def test_save_clip_returns_partially_saved_when_workspace_creation_fails_after_note(clipper_db, monkeypatch: pytest.MonkeyPatch):
    service = WebClipperService(db=clipper_db, user_id=1)

    def _boom(*_args, **_kwargs):
        raise CharactersRAGDBError("workspace note persistence failed")

    monkeypatch.setattr(clipper_db, "add_workspace_note", _boom)

    result = service.save_clip(_save_request())

    assert result.status == "partially_saved"
    assert result.note.id == "clip-123"
    assert any("workspace" in warning.lower() for warning in result.warnings)
    assert clipper_db.get_note_by_id("clip-123") is not None


def test_persist_enrichment_skips_inline_writeback_on_version_mismatch_but_stores_structured_data(clipper_db):
    service = WebClipperService(db=clipper_db, user_id=1)
    save_result = service.save_clip(_save_request(destination_mode="note"))

    current_note = clipper_db.get_note_by_id("clip-123")
    assert current_note is not None
    clipper_db.update_note(
        note_id="clip-123",
        update_data={"content": f"{current_note['content']}\n\nUser edit."},
        expected_version=int(current_note["version"]),
    )

    enrichment = service.persist_enrichment(
        "clip-123",
        WebClipperEnrichmentPayload(
            clip_id="clip-123",
            enrichment_type="ocr",
            status="complete",
            inline_summary="Captured text summary.",
            structured_payload={"raw_text": "Captured text summary."},
            source_note_version=save_result.note.version,
        ),
    )

    assert enrichment.inline_applied is False
    assert enrichment.conflict_reason == "source_note_version_mismatch"

    clip_doc = clipper_db.get_note_clipper_document_by_clip_id("clip-123")
    assert clip_doc is not None
    assert clip_doc["analysis_json"]["ocr"]["structured_payload"]["raw_text"] == "Captured text summary."

    note_after = clipper_db.get_note_by_id("clip-123")
    assert note_after is not None
    assert "Captured text summary." not in note_after["content"]
    assert note_after["content"].endswith("User edit.")


def test_persist_enrichment_stores_ocr_and_vlm_structured_payloads_when_inline_writeback_is_skipped(clipper_db):
    service = WebClipperService(db=clipper_db, user_id=1)
    save_result = service.save_clip(_save_request(destination_mode="note", clip_id="clip-enrichment-mismatch"))

    current_note = clipper_db.get_note_by_id("clip-enrichment-mismatch")
    assert current_note is not None
    clipper_db.update_note(
        note_id="clip-enrichment-mismatch",
        update_data={"content": f"{current_note['content']}\n\nUser edit remains."},
        expected_version=int(current_note["version"]),
    )

    ocr_enrichment = service.persist_enrichment(
        "clip-enrichment-mismatch",
        WebClipperEnrichmentPayload(
            clip_id="clip-enrichment-mismatch",
            enrichment_type="ocr",
            status="complete",
            inline_summary="OCR machine summary.",
            structured_payload={"raw_text": "OCR raw text"},
            source_note_version=save_result.note.version,
        ),
    )
    vlm_enrichment = service.persist_enrichment(
        "clip-enrichment-mismatch",
        WebClipperEnrichmentPayload(
            clip_id="clip-enrichment-mismatch",
            enrichment_type="vlm",
            status="complete",
            inline_summary="VLM machine summary.",
            structured_payload={"caption": "VLM caption"},
            source_note_version=save_result.note.version,
        ),
    )

    assert ocr_enrichment.inline_applied is False
    assert ocr_enrichment.conflict_reason == "source_note_version_mismatch"
    assert vlm_enrichment.inline_applied is False
    assert vlm_enrichment.conflict_reason == "source_note_version_mismatch"

    clip_doc = clipper_db.get_note_clipper_document_by_clip_id("clip-enrichment-mismatch")
    assert clip_doc is not None
    assert clip_doc["analysis_json"]["ocr"]["structured_payload"]["raw_text"] == "OCR raw text"
    assert clip_doc["analysis_json"]["vlm"]["structured_payload"]["caption"] == "VLM caption"

    note_after = clipper_db.get_note_by_id("clip-enrichment-mismatch")
    assert note_after is not None
    assert "OCR machine summary." not in note_after["content"]
    assert "VLM machine summary." not in note_after["content"]
    assert note_after["content"].endswith("User edit remains.")


def test_persist_enrichment_preserves_visible_body_without_comment(clipper_db):
    service = WebClipperService(db=clipper_db, user_id=1)
    request = _save_request(destination_mode="note", clip_id="clip-visible-preserve", full_extract=None)
    request.note = WebClipperSaveRequest.NotePayload(
        title="Example Story",
        comment=None,
        keywords=[],
    )
    request.content = WebClipperSaveRequest.ContentPayload(
        visible_body="para1\n\npara2",
        full_extract=None,
        selected_text=None,
    )

    save_result = service.save_clip(request)
    enrichment = service.persist_enrichment(
        "clip-visible-preserve",
        WebClipperEnrichmentPayload(
            clip_id="clip-visible-preserve",
            enrichment_type="ocr",
            status="complete",
            inline_summary="Captured text summary.",
            structured_payload={"raw_text": "Captured text summary."},
            source_note_version=save_result.note.version,
        ),
    )

    note = clipper_db.get_note_by_id("clip-visible-preserve")
    assert enrichment.inline_applied is True
    assert note is not None
    content_before_analysis = note["content"].split("## Web Clipper Analysis", maxsplit=1)[0]
    assert "para1" in content_before_analysis
    assert "para2" in content_before_analysis


def test_persist_enrichment_clamps_inline_summaries_to_per_type_and_combined_budgets(clipper_db):
    service = WebClipperService(db=clipper_db, user_id=1)
    service.save_clip(_save_request(destination_mode="note", clip_id="clip-enrichment-budgets"))

    ocr_summary = "O" * 1800
    vlm_summary = "V" * 1400

    ocr_result = service.persist_enrichment(
        "clip-enrichment-budgets",
        WebClipperEnrichmentPayload(
            clip_id="clip-enrichment-budgets",
            enrichment_type="ocr",
            status="complete",
            inline_summary=ocr_summary,
            structured_payload={"raw_text": ocr_summary},
            source_note_version=1,
        ),
    )
    vlm_result = service.persist_enrichment(
        "clip-enrichment-budgets",
        WebClipperEnrichmentPayload(
            clip_id="clip-enrichment-budgets",
            enrichment_type="vlm",
            status="complete",
            inline_summary=vlm_summary,
            structured_payload={"caption": vlm_summary},
            source_note_version=ocr_result.source_note_version,
        ),
    )

    assert ocr_result.inline_applied is True
    assert len(ocr_result.inline_summary or "") == 1500
    assert vlm_result.inline_applied is True
    assert len(vlm_result.inline_summary or "") == 1000

    clip_doc = clipper_db.get_note_clipper_document_by_clip_id("clip-enrichment-budgets")
    assert clip_doc is not None
    assert len(clip_doc["analysis_json"]["ocr"]["inline_summary"]) == 1500
    assert len(clip_doc["analysis_json"]["vlm"]["inline_summary"]) == 1000

    note = clipper_db.get_note_by_id("clip-enrichment-budgets")
    assert note is not None
    machine_section = note["content"].split("## Web Clipper Analysis", maxsplit=1)[1]
    ocr_section, vlm_section = machine_section.split("### VLM", maxsplit=1)
    ocr_body = ocr_section.split("### OCR", maxsplit=1)[1]
    assert ocr_body.count("O") == 1500
    assert vlm_section.count("V") == 1000
    assert ocr_body.count("O") + vlm_section.count("V") == 2500


def test_save_clip_prefers_visible_body_over_full_extract(clipper_db):
    service = WebClipperService(db=clipper_db, user_id=1)
    request = _save_request(destination_mode="note", clip_id="clip-visible-vs-full")
    request.content = WebClipperSaveRequest.ContentPayload(
        visible_body="VISIBLE BODY",
        full_extract="FULL EXTRACT",
        selected_text="SELECTED TEXT",
    )

    service.save_clip(request)

    note = clipper_db.get_note_by_id("clip-visible-vs-full")
    clip_doc = clipper_db.get_note_clipper_document_by_clip_id("clip-visible-vs-full")
    assert note is not None
    assert clip_doc is not None
    assert note["content"].endswith("VISIBLE BODY")
    assert "FULL EXTRACT" not in note["content"]
    assert clip_doc["content_budget_json"]["visible_body"] == "VISIBLE BODY"


def test_save_clip_truncates_visible_body_to_content_budget(clipper_db):
    service = WebClipperService(db=clipper_db, user_id=1)
    oversized_extract = "\n\n".join(["Paragraph " + ("A" * 3000) for _ in range(8)])
    request = _save_request(destination_mode="note", full_extract=oversized_extract)
    request.content = WebClipperSaveRequest.ContentPayload(
        visible_body=oversized_extract,
        full_extract=oversized_extract,
        selected_text=None,
    )

    result = service.save_clip(request)

    note = clipper_db.get_note_by_id("clip-123")
    assert note is not None
    assert result.status in {"saved", "saved_with_warnings"}
    assert "Truncated. Full extract attached." not in note["content"]
    assert "Truncated. Full extract preserved in clip metadata." in note["content"]

    clip_doc = clipper_db.get_note_clipper_document_by_clip_id("clip-123")
    assert clip_doc is not None
    assert clip_doc["content_budget_json"]["visible_body_truncated"] is True
    assert len(clip_doc["content_budget_json"]["visible_body"]) <= 12029


def test_save_clip_records_pending_enhancement_requests_and_status(clipper_db):
    service = WebClipperService(db=clipper_db, user_id=1)
    request = _save_request(destination_mode="note")
    request.enhancements = WebClipperSaveRequest.EnhancementOptions(run_ocr=True, run_vlm=True)

    result = service.save_clip(request)
    status = service.get_clip_status("clip-123")

    assert result.status == "saved"
    assert status.status == "saved"
    assert status.analysis["ocr"]["status"] == "pending"
    assert status.analysis["vlm"]["status"] == "pending"


def test_save_clip_attachment_io_failure_becomes_saved_with_warnings(clipper_db, monkeypatch: pytest.MonkeyPatch):
    service = WebClipperService(db=clipper_db, user_id=1)

    def _boom(*_args, **_kwargs):
        raise CharactersRAGDBError("disk full")

    monkeypatch.setattr(WebClipperService, "_persist_attachment", _boom)

    result = service.save_clip(_save_request(destination_mode="note", include_attachment=True))
    status = service.get_clip_status("clip-123")

    assert result.status == "saved_with_warnings"
    assert status.status == "saved_with_warnings"
    assert any("attachment" in warning.lower() for warning in result.warnings)


def test_save_clip_resyncs_existing_workspace_note_and_keywords(clipper_db):
    service = WebClipperService(db=clipper_db, user_id=1)

    first_request = _save_request(destination_mode="both", clip_id="clip-workspace-sync", full_extract=None)
    first_request.note = WebClipperSaveRequest.NotePayload(
        title="Title One",
        comment="First comment.",
        keywords=["one"],
    )
    first_request.content = WebClipperSaveRequest.ContentPayload(
        visible_body="body one",
        full_extract=None,
        selected_text=None,
    )
    service.save_clip(first_request)

    second_request = _save_request(destination_mode="both", clip_id="clip-workspace-sync", full_extract=None)
    second_request.note = WebClipperSaveRequest.NotePayload(
        title="Title Two",
        comment="Second comment.",
        keywords=["two"],
    )
    second_request.content = WebClipperSaveRequest.ContentPayload(
        visible_body="body two",
        full_extract=None,
        selected_text=None,
    )
    result = service.save_clip(second_request)

    placement = clipper_db.list_note_clipper_workspace_placements("clip-workspace-sync")[0]
    workspace_note = clipper_db.execute_query(
        "SELECT title, content, keywords_json FROM workspace_notes WHERE workspace_id = ? AND id = ? AND deleted = 0",
        ("ws-1", int(placement["workspace_note_id"])),
    ).fetchone()
    note = clipper_db.get_note_by_id("clip-workspace-sync")
    keywords = [row["keyword"] for row in clipper_db.get_keywords_for_note("clip-workspace-sync")]

    assert result.workspace_placement is not None
    assert note is not None
    assert workspace_note is not None
    assert note["title"] == "Title Two"
    assert "body two" in note["content"]
    assert keywords == ["two"]
    assert workspace_note["title"] == "Title Two"
    assert "body two" in workspace_note["content"]
    assert json.loads(workspace_note["keywords_json"]) == ["two"]


def test_save_clip_returns_failed_when_canonical_note_cannot_be_created(clipper_db, monkeypatch: pytest.MonkeyPatch):
    service = WebClipperService(db=clipper_db, user_id=1)

    def _boom(*_args, **_kwargs):
        raise CharactersRAGDBError("write failed")

    monkeypatch.setattr(clipper_db, "add_note", _boom)

    result = service.save_clip(_save_request(destination_mode="note"))

    assert result.status == "failed"
    assert result.note is None
    assert clipper_db.get_note_clipper_document_by_clip_id("clip-123") is None


def test_save_clip_rejects_oversized_attachment_payloads_as_warnings(clipper_db, monkeypatch: pytest.MonkeyPatch):
    service = WebClipperService(db=clipper_db, user_id=1)
    monkeypatch.setattr(web_clipper_service_module, "_MAX_ATTACHMENT_BYTES", 16)

    request = _save_request(destination_mode="note", clip_id="clip-oversized")
    request.attachments = [
        WebClipperSaveRequest.AttachmentPayload(
            slot="page-screenshot",
            file_name="page-screenshot.txt",
            media_type="text/plain",
            content_base64=base64.b64encode(b"x" * 17).decode("ascii"),
        )
    ]

    result = service.save_clip(request)
    status = service.get_clip_status("clip-oversized")

    assert result.status == "saved_with_warnings"
    assert status.status == "saved_with_warnings"
    assert status.attachments == []
