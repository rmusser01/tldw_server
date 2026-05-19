from pathlib import Path

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Plaintext import Plaintext_Files as plaintext


@pytest.mark.unit
def test_process_document_content_rejects_path_outside_base_dir(tmp_path: Path) -> None:
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    outside_path = tmp_path / "outside" / "note.txt"

    result = plaintext.process_document_content(
        doc_path=outside_path,
        perform_chunking=False,
        chunk_options=None,
        perform_analysis=False,
        summarize_recursively=False,
        api_name=None,
        api_key=None,
        custom_prompt=None,
        system_prompt=None,
        base_dir=allowed_dir,
    )

    assert result["status"] == "Error"
    assert "rejected outside allowed base directory" in (result.get("error") or "")


@pytest.mark.unit
def test_process_document_content_sanitizes_unexpected_processing_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    doc_path = tmp_path / "note.txt"
    doc_path.write_text("hello", encoding="utf-8")

    def _fail_convert(*_args, **_kwargs):
        raise RuntimeError("plaintext converter exploded at /private/docs/note.txt")

    monkeypatch.setattr(plaintext, "convert_document_to_text", _fail_convert)

    result = plaintext.process_document_content(
        doc_path=doc_path,
        perform_chunking=False,
        chunk_options=None,
        perform_analysis=False,
        summarize_recursively=False,
        api_name=None,
        api_key=None,
        custom_prompt=None,
        system_prompt=None,
        base_dir=tmp_path,
    )

    assert result["status"] == "Error"
    assert result["error"] == "Document processing failed"
    assert "plaintext converter exploded" not in str(result)
    assert "/private/docs/note.txt" not in str(result)
