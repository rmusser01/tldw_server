from pathlib import Path

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Books import Book_Processing_Lib as books


@pytest.mark.unit
def test_process_epub_rejects_path_outside_base_dir(tmp_path: Path) -> None:
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    outside_path = tmp_path / "outside" / "book.epub"

    result = books.process_epub(
        file_path=str(outside_path),
        perform_chunking=False,
        perform_analysis=False,
        base_dir=allowed_dir,
    )

    assert result["status"] == "Error"
    assert "rejected outside allowed base directory" in (result.get("error") or "")


@pytest.mark.unit
def test_process_epub_sanitizes_unexpected_processing_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    epub_path = tmp_path / "book.epub"
    epub_path.write_bytes(b"fake epub")

    class _FakeBook:
        metadata = {}

    monkeypatch.setattr(
        books,
        "read_epub_filtered",
        lambda _file_path: ("chapter text", _FakeBook()),
    )

    def _fail_metadata(_ebook_obj):
        raise TypeError("metadata parser exploded at /private/books/book.epub")

    monkeypatch.setattr(books, "extract_epub_metadata_from_epub_obj", _fail_metadata)

    result = books.process_epub(
        file_path=str(epub_path),
        perform_chunking=False,
        perform_analysis=False,
        base_dir=tmp_path,
    )

    assert result["status"] == "Error"
    assert result["error"] == "Ebook processing failed"
    assert "metadata parser exploded" not in str(result)
    assert "/private/books/book.epub" not in str(result)
