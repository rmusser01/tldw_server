from pathlib import Path

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Books import Book_Processing_Lib as books


@pytest.mark.unit
def test_process_markup_or_plain_text_sanitizes_unexpected_processing_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    html_path = tmp_path / "note.html"
    html_path.write_text("<html><body>Plain text body</body></html>", encoding="utf-8")

    class _FailingRe:
        @staticmethod
        def sub(*_args, **_kwargs):
            raise RuntimeError("markup normalizer exploded at /private/books/note.html")

    monkeypatch.setattr(books, "re", _FailingRe)

    result = books._process_markup_or_plain_text(
        file_path=str(html_path),
        file_type="html",
        perform_chunking=False,
        perform_analysis=False,
        base_dir=tmp_path,
    )

    assert result["status"] == "Error"
    assert result["error"] == "Document processing failed"
    assert "markup normalizer exploded" not in str(result)
    assert "/private/books/note.html" not in str(result)
