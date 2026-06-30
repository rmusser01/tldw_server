from __future__ import annotations

from contextlib import contextmanager

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Books import Book_Processing_Lib as books


@pytest.mark.unit
def test_ingest_text_file_uses_managed_media_database(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text_file = tmp_path / "sample.txt"
    text_file.write_text("Book body", encoding="utf-8")

    class _FakeDb:
        def __init__(self) -> None:
            self.closed = False

        def close_connection(self) -> None:
            self.closed = True

    fake_db = _FakeDb()
    managed_calls: list[dict[str, object]] = []
    add_calls: list[dict[str, object]] = []

    @contextmanager
    def _fake_managed_media_database(client_id, *, initialize=True, **kwargs):
        managed_calls.append(
            {
                "client_id": client_id,
                "initialize": initialize,
                "kwargs": kwargs,
            }
        )
        try:
            yield fake_db
        finally:
            fake_db.close_connection()

    def _fake_add_media_with_keywords(**kwargs):
        add_calls.append(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(
        books,
        "managed_media_database",
        _fake_managed_media_database,
        raising=False,
    )
    monkeypatch.setattr(
        books,
        "get_media_repository",
        lambda db_instance: type(
            "_FakeWriter",
            (),
            {"add_media_with_keywords": staticmethod(_fake_add_media_with_keywords)},
        )(),
        raising=False,
    )

    result = books.ingest_text_file(
        str(text_file),
        title="Sample Book",
        author="Author Name",
        keywords="fiction,novel",
        base_dir=tmp_path,
    )

    assert "ingested successfully" in result
    assert fake_db.closed is True
    assert managed_calls == [
        {
            "client_id": "book_ingest",
            "initialize": False,
            "kwargs": {},
        }
    ]
    assert len(add_calls) == 1
    assert add_calls[0]["db_instance"] is fake_db
    assert add_calls[0]["title"] == "Sample Book"
    assert add_calls[0]["author"] == "Author Name"
    assert add_calls[0]["keywords"] == ["text_file", "epub_converted", "fiction", "novel"]


@pytest.mark.unit
def test_ingest_text_file_sanitizes_backend_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text_file = tmp_path / "sample.txt"
    text_file.write_text("Book body", encoding="utf-8")

    @contextmanager
    def _fake_managed_media_database(*_args, **_kwargs):
        yield object()

    def _fail_add_media_with_keywords(**_kwargs):
        raise RuntimeError("media database exploded at /private/db/books.sqlite")

    monkeypatch.setattr(
        books,
        "managed_media_database",
        _fake_managed_media_database,
        raising=False,
    )
    monkeypatch.setattr(
        books,
        "get_media_repository",
        lambda _db_instance: type(
            "_FakeWriter",
            (),
            {"add_media_with_keywords": staticmethod(_fail_add_media_with_keywords)},
        )(),
        raising=False,
    )

    result = books.ingest_text_file(
        str(text_file),
        title="Sample Book",
        author="Author Name",
        keywords="fiction,novel",
        base_dir=tmp_path,
    )

    assert result == "Error ingesting text file"
    assert "media database exploded" not in result
    assert "/private/db/books.sqlite" not in result


@pytest.mark.unit
def test_ingest_folder_sanitizes_unexpected_listing_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _FailingOs:
        @staticmethod
        def listdir(_folder_path):
            raise RuntimeError("folder lister exploded at /private/books")

    monkeypatch.setattr(books, "os", _FailingOs)

    result = books.ingest_folder(str(tmp_path / "books"))

    assert result == "Error ingesting folder"
    assert "folder lister exploded" not in result
    assert "/private/books" not in result
