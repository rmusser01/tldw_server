from unittest.mock import MagicMock
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import reading


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_create_saved_search_sanitizes_backend_log(monkeypatch):
    logger_stub = MagicMock()

    class _FailingDB:
        def create_saved_search(self, **_kwargs):
            raise RuntimeError("saved search create exploded at /private/collections.db")

    monkeypatch.setattr(reading, "logger", logger_stub)
    monkeypatch.setattr(reading, "_ensure_reading_saved_searches_enabled", lambda: None)

    with pytest.raises(HTTPException) as excinfo:
        await reading.create_reading_saved_search(
            payload=reading.ReadingSavedSearchCreateRequest(name="Morning", query={"q": "ai"}),
            collections_db=_FailingDB(),
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "reading_saved_search_create_failed"
    logger_stub.error.assert_called_once_with("reading_saved_search_create_failed")


@pytest.mark.asyncio
async def test_list_saved_searches_sanitizes_backend_log(monkeypatch):
    logger_stub = MagicMock()

    class _FailingDB:
        def list_saved_searches(self, **_kwargs):
            raise RuntimeError("saved search list exploded at /private/collections.db")

    monkeypatch.setattr(reading, "logger", logger_stub)
    monkeypatch.setattr(reading, "_ensure_reading_saved_searches_enabled", lambda: None)

    with pytest.raises(HTTPException) as excinfo:
        await reading.list_reading_saved_searches(limit=10, offset=0, collections_db=_FailingDB())

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "reading_saved_search_list_failed"
    logger_stub.error.assert_called_once_with("reading_saved_search_list_failed")


@pytest.mark.asyncio
async def test_update_saved_search_sanitizes_backend_log(monkeypatch):
    logger_stub = MagicMock()

    class _FailingDB:
        def update_saved_search(self, *_args, **_kwargs):
            raise RuntimeError("saved search update exploded at /private/collections.db")

    monkeypatch.setattr(reading, "logger", logger_stub)
    monkeypatch.setattr(reading, "_ensure_reading_saved_searches_enabled", lambda: None)

    with pytest.raises(HTTPException) as excinfo:
        await reading.update_reading_saved_search(
            search_id=123,
            payload=reading.ReadingSavedSearchUpdateRequest(query={"q": "ml"}),
            collections_db=_FailingDB(),
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "reading_saved_search_update_failed"
    logger_stub.error.assert_called_once_with("reading_saved_search_update_failed")


@pytest.mark.asyncio
async def test_delete_saved_search_sanitizes_backend_log(monkeypatch):
    logger_stub = MagicMock()

    class _FailingDB:
        def delete_saved_search(self, *_args, **_kwargs):
            raise RuntimeError("saved search delete exploded at /private/collections.db")

    monkeypatch.setattr(reading, "logger", logger_stub)
    monkeypatch.setattr(reading, "_ensure_reading_saved_searches_enabled", lambda: None)

    with pytest.raises(HTTPException) as excinfo:
        await reading.delete_reading_saved_search(search_id=123, collections_db=_FailingDB())

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "reading_saved_search_delete_failed"
    logger_stub.error.assert_called_once_with("reading_saved_search_delete_failed")


@pytest.mark.asyncio
async def test_create_note_link_sanitizes_backend_log(monkeypatch):
    logger_stub = MagicMock()

    class _FailingDB:
        def get_content_item(self, *_args, **_kwargs):
            return SimpleNamespace(title="Research paper")

        def link_note_to_content_item(self, **_kwargs):
            raise RuntimeError("note link create exploded at /private/collections.db")

    monkeypatch.setattr(reading, "logger", logger_stub)
    monkeypatch.setattr(reading, "_ensure_reading_note_links_enabled", lambda: None)
    monkeypatch.setattr(reading, "_ensure_note_exists_or_404", lambda *_args, **_kwargs: None)

    with pytest.raises(HTTPException) as excinfo:
        await reading.link_note_to_reading_item(
            item_id=123,
            payload=reading.ReadingNoteLinkCreateRequest(note_id="note-1"),
            current_user=SimpleNamespace(id=42),
            notes_db=object(),
            collections_db=_FailingDB(),
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "reading_note_link_create_failed"
    logger_stub.error.assert_called_once_with("reading_note_link_create_failed")


@pytest.mark.asyncio
async def test_list_note_links_sanitizes_backend_log(monkeypatch):
    logger_stub = MagicMock()

    class _FailingDB:
        def list_note_links_for_content_item(self, *_args, **_kwargs):
            raise RuntimeError("note link list exploded at /private/collections.db")

    monkeypatch.setattr(reading, "logger", logger_stub)
    monkeypatch.setattr(reading, "_ensure_reading_note_links_enabled", lambda: None)

    with pytest.raises(HTTPException) as excinfo:
        await reading.list_reading_item_note_links(item_id=123, collections_db=_FailingDB())

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "reading_note_link_list_failed"
    logger_stub.error.assert_called_once_with("reading_note_link_list_failed")


@pytest.mark.asyncio
async def test_delete_note_link_sanitizes_backend_log(monkeypatch):
    logger_stub = MagicMock()

    class _FailingDB:
        def get_content_item(self, *_args, **_kwargs):
            return SimpleNamespace(title="Research paper")

        def list_note_links_for_content_item(self, *_args, **_kwargs):
            return []

        def unlink_note_from_content_item(self, **_kwargs):
            raise RuntimeError("note link delete exploded at /private/collections.db")

    monkeypatch.setattr(reading, "logger", logger_stub)
    monkeypatch.setattr(reading, "_ensure_reading_note_links_enabled", lambda: None)

    with pytest.raises(HTTPException) as excinfo:
        await reading.unlink_note_from_reading_item(
            item_id=123,
            note_id="note-1",
            current_user=SimpleNamespace(id=42),
            collections_db=_FailingDB(),
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "reading_note_link_delete_failed"
    logger_stub.error.assert_called_once_with("reading_note_link_delete_failed")


@pytest.mark.asyncio
async def test_import_reading_items_sanitizes_file_read_log(monkeypatch):
    logger_stub = MagicMock()

    class _UnreadableUpload:
        filename = "reading.csv"

        async def read(self):
            raise RuntimeError("import read exploded at /private/uploads/reading.csv")

    monkeypatch.setattr(reading, "logger", logger_stub)

    with pytest.raises(HTTPException) as excinfo:
        await reading.import_reading_items(
            file=_UnreadableUpload(),
            source="auto",
            merge_tags=True,
            current_user=SimpleNamespace(id=42),
            jm=object(),
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "reading_import_failed"
    logger_stub.error.assert_called_once_with("reading_import_read_failed")


@pytest.mark.asyncio
async def test_import_reading_items_sanitizes_job_create_log(monkeypatch):
    logger_stub = MagicMock()

    class _Upload:
        filename = "reading.csv"

        async def read(self):
            return b"url,title\nhttps://example.test,Example"

    class _StagedPath:
        name = "reading-import-token.csv"

        def unlink(self, **_kwargs):
            return None

    class _FailingJobManager:
        def create_job(self, **_kwargs):
            raise RuntimeError("job create exploded at /private/jobs.db")

    monkeypatch.setattr(reading, "logger", logger_stub)
    monkeypatch.setattr(reading, "stage_reading_import_file", lambda **_kwargs: _StagedPath())

    with pytest.raises(HTTPException) as excinfo:
        await reading.import_reading_items(
            file=_Upload(),
            source="auto",
            merge_tags=True,
            current_user=SimpleNamespace(id=42),
            jm=_FailingJobManager(),
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "reading_import_failed"
    logger_stub.error.assert_called_once_with("reading_import_job_create_failed")


@pytest.mark.asyncio
async def test_import_reading_items_sanitizes_staged_cleanup_log(monkeypatch):
    logger_stub = MagicMock()

    class _Upload:
        filename = "reading.csv"

        async def read(self):
            return b"url,title\nhttps://example.test,Example"

    class _StagedPath:
        name = "reading-import-token.csv"

        def unlink(self, **_kwargs):
            raise OSError("cleanup exploded at /private/staged/reading.csv")

    class _FailingJobManager:
        def create_job(self, **_kwargs):
            raise RuntimeError("job create exploded at /private/jobs.db")

    monkeypatch.setattr(reading, "logger", logger_stub)
    monkeypatch.setattr(reading, "stage_reading_import_file", lambda **_kwargs: _StagedPath())

    with pytest.raises(HTTPException) as excinfo:
        await reading.import_reading_items(
            file=_Upload(),
            source="auto",
            merge_tags=True,
            current_user=SimpleNamespace(id=42),
            jm=_FailingJobManager(),
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "reading_import_failed"
    logger_stub.debug.assert_called_once_with("reading_import_staged_file_cleanup_failed")
