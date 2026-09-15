"""Requested analysis must not store legacy error strings as useful analysis."""
import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Plaintext import Plaintext_Files as docs


@pytest.mark.unit
@pytest.mark.parametrize("analysis", ["Error: Model is required for provider 'llama.cpp'", ""])
def test_document_analysis_failure_keeps_source_and_reports_warning(tmp_path, monkeypatch, analysis):
    source = tmp_path / "cedar.md"
    source.write_text("Project Cedar launches on 22 November 2026.", encoding="utf-8")
    monkeypatch.setattr(docs, "analyze", lambda **kwargs: analysis)
    result = docs.process_document_content(
        source, False, None, True, False, "llama", None, None, None,
    )
    assert result["status"] == "Warning"
    assert result["analysis"] is None
    assert result["warnings"]
    assert "Project Cedar" in result["content"]
    assert result["chunks"][0]["metadata"].get("analysis") is None


@pytest.mark.unit
def test_document_analysis_success_is_retained(tmp_path, monkeypatch):
    source = tmp_path / "cedar.md"
    source.write_text("Project Cedar launches in November.", encoding="utf-8")
    monkeypatch.setattr(docs, "analyze", lambda **kwargs: "Cedar launches in November.")
    result = docs.process_document_content(
        source, False, None, True, False, "llama", None, None, None,
    )
    assert result["status"] == "Success"
    assert result["analysis"] == "Cedar launches in November."
