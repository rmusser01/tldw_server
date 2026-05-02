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
async def test_list_reading_import_jobs_includes_canonical_pagination():
    class _JobManager:
        def list_jobs(self, **_kwargs):
            return [
                {
                    "id": 1,
                    "uuid": "job-1",
                    "status": "queued",
                    "payload_json": "{}",
                    "result_json": None,
                    "error_message": None,
                    "progress": 0,
                    "created_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-01-01T00:00:00",
                    "started_at": None,
                    "completed_at": None,
                },
                {
                    "id": 2,
                    "uuid": "job-2",
                    "status": "processing",
                    "payload_json": "{}",
                    "result_json": None,
                    "error_message": None,
                    "progress": 25,
                    "created_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-01-01T00:00:00",
                    "started_at": None,
                    "completed_at": None,
                },
            ]

        def count_jobs(self, **_kwargs):
            return 5

    response = await reading.list_reading_import_jobs(
        status=None,
        limit=2,
        offset=1,
        current_user=SimpleNamespace(id=42),
        jm=_JobManager(),
    )

    assert response.total == 5
    assert response.limit == 2
    assert response.offset == 1
    assert [job.job_id for job in response.jobs] == [2]
    assert response.pagination.mode == "offset"
    assert response.pagination.total == 5
    assert response.pagination.limit == 2
    assert response.pagination.offset == 1
    assert response.pagination.has_more is True
    assert response.pagination.next_offset == 3
    assert response.has_more is True
    assert response.next_offset == 3


@pytest.mark.asyncio
async def test_list_reading_digest_outputs_includes_canonical_pagination():
    row = SimpleNamespace(
        id=11,
        title="Digest A",
        format="md",
        created_at="2026-01-01T00:00:00",
        metadata_json='{"schedule_id":"sched-1","schedule_name":"Morning","item_count":3}',
    )

    class _CollectionsDB:
        def list_output_artifacts(self, **kwargs):
            if kwargs["offset"] == 0:
                return [row], 1
            return [], 1

    response = await reading.list_reading_digest_outputs(
        schedule_id="sched-1",
        limit=1,
        offset=0,
        _current_user=SimpleNamespace(id=42),
        collections_db=_CollectionsDB(),
    )

    assert response.total == 1
    assert response.limit == 1
    assert response.offset == 0
    assert [item.output_id for item in response.items] == [11]
    assert response.pagination.mode == "offset"
    assert response.pagination.total == 1
    assert response.pagination.limit == 1
    assert response.pagination.offset == 0
    assert response.pagination.has_more is False
    assert response.pagination.next_offset is None
    assert response.has_more is False
    assert response.next_offset is None


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


@pytest.mark.asyncio
async def test_save_reading_item_sanitizes_backend_log(monkeypatch):
    logger_stub = MagicMock()

    class _FailingService:
        async def save_url(self, **_kwargs):
            raise RuntimeError("reading save exploded at /private/collections.db")

    monkeypatch.setattr(reading, "logger", logger_stub)
    monkeypatch.setattr(reading, "_service_for_user", lambda _user: _FailingService())

    with pytest.raises(HTTPException) as excinfo:
        await reading.save_reading_item(
            payload=reading.ReadingSaveRequest(url="https://example.test/article"),
            current_user=SimpleNamespace(id=42),
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "reading_save_failed"
    logger_stub.error.assert_called_once_with("reading_save_failed")


@pytest.mark.asyncio
async def test_get_reading_item_sanitizes_backend_log(monkeypatch):
    logger_stub = MagicMock()

    class _FailingService:
        def get_item(self, *_args, **_kwargs):
            raise RuntimeError("reading get exploded at /private/collections.db")

    monkeypatch.setattr(reading, "logger", logger_stub)
    monkeypatch.setattr(reading, "_service_for_user", lambda _user: _FailingService())

    with pytest.raises(HTTPException) as excinfo:
        await reading.get_reading_item(item_id=123, current_user=SimpleNamespace(id=42))

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "reading_get_failed"
    logger_stub.error.assert_called_once_with("reading_get_failed")


@pytest.mark.asyncio
async def test_update_reading_item_sanitizes_backend_log(monkeypatch):
    logger_stub = MagicMock()

    class _FailingService:
        def update_item(self, **_kwargs):
            raise RuntimeError("reading update exploded at /private/collections.db")

    monkeypatch.setattr(reading, "logger", logger_stub)
    monkeypatch.setattr(reading, "_service_for_user", lambda _user: _FailingService())

    with pytest.raises(HTTPException) as excinfo:
        await reading.update_reading_item(
            item_id=123,
            payload=reading.ReadingUpdateRequest(status="read"),
            current_user=SimpleNamespace(id=42),
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "reading_update_failed"
    logger_stub.error.assert_called_once_with("reading_update_failed")


@pytest.mark.asyncio
async def test_delete_reading_item_sanitizes_backend_log(monkeypatch):
    logger_stub = MagicMock()

    class _FailingService:
        def update_item(self, *_args, **_kwargs):
            raise RuntimeError("reading delete exploded at /private/collections.db")

    monkeypatch.setattr(reading, "logger", logger_stub)
    monkeypatch.setattr(reading, "_service_for_user", lambda _user: _FailingService())

    with pytest.raises(HTTPException) as excinfo:
        await reading.delete_reading_item(
            item_id=123,
            hard=False,
            current_user=SimpleNamespace(id=42),
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "reading_delete_failed"
    logger_stub.error.assert_called_once_with("reading_delete_failed")


def _reading_action_row() -> SimpleNamespace:
    return SimpleNamespace(
        id=123,
        media_id=None,
        title="Research item",
        url="https://example.test/research",
        canonical_url=None,
        domain="example.test",
        summary="Existing summary",
        notes="Existing notes",
        published_at=None,
        status="saved",
        favorite=False,
        tags=[],
        created_at=None,
        updated_at=None,
        read_at=None,
        content_hash="hash",
        origin_type="web",
        metadata_json='{"text": "Action body text"}',
    )


@pytest.mark.asyncio
async def test_summarize_reading_item_sanitizes_fetch_log(monkeypatch):
    logger_stub = MagicMock()

    class _FailingService:
        def get_item(self, *_args, **_kwargs):
            raise RuntimeError("summary fetch exploded at /private/collections.db")

    monkeypatch.setattr(reading, "logger", logger_stub)
    monkeypatch.setattr(reading, "_service_for_user", lambda _user: _FailingService())

    with pytest.raises(HTTPException) as excinfo:
        await reading.summarize_reading_item(
            item_id=123,
            payload=reading.ReadingSummarizeRequest(provider="openai"),
            current_user=SimpleNamespace(id=42),
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "reading_item_fetch_failed"
    logger_stub.error.assert_called_once_with("reading_summary_get_failed")


@pytest.mark.asyncio
async def test_summarize_reading_item_sanitizes_backend_log(monkeypatch):
    logger_stub = MagicMock()

    class _Service:
        def get_item(self, *_args, **_kwargs):
            return _reading_action_row()

    def _failing_summarize(*_args, **_kwargs):
        raise RuntimeError("summarizer exploded at /private/llm.key")

    monkeypatch.setattr(reading, "logger", logger_stub)
    monkeypatch.setattr(reading, "_service_for_user", lambda _user: _Service())
    monkeypatch.setattr(reading, "summarize_analyze", _failing_summarize)

    with pytest.raises(HTTPException) as excinfo:
        await reading.summarize_reading_item(
            item_id=123,
            payload=reading.ReadingSummarizeRequest(provider="openai"),
            current_user=SimpleNamespace(id=42),
        )

    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == "reading_summarize_failed"
    logger_stub.error.assert_called_once_with("reading_summarize_failed")


@pytest.mark.asyncio
async def test_summarize_reading_item_sanitizes_error_string_log(monkeypatch):
    logger_stub = MagicMock()

    class _Service:
        def get_item(self, *_args, **_kwargs):
            return _reading_action_row()

    monkeypatch.setattr(reading, "logger", logger_stub)
    monkeypatch.setattr(reading, "_service_for_user", lambda _user: _Service())
    monkeypatch.setattr(reading, "summarize_analyze", lambda *_args, **_kwargs: "Error: /private/llm.key")

    with pytest.raises(HTTPException) as excinfo:
        await reading.summarize_reading_item(
            item_id=123,
            payload=reading.ReadingSummarizeRequest(provider="openai"),
            current_user=SimpleNamespace(id=42),
        )

    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == "reading_summarize_failed"
    logger_stub.error.assert_called_once_with("reading_summarize_error")


@pytest.mark.asyncio
async def test_tts_reading_item_sanitizes_fetch_log(monkeypatch):
    logger_stub = MagicMock()

    class _FailingService:
        def get_item(self, *_args, **_kwargs):
            raise RuntimeError("tts fetch exploded at /private/collections.db")

    monkeypatch.setattr(reading, "logger", logger_stub)
    monkeypatch.setattr(reading, "_service_for_user", lambda _user: _FailingService())

    with pytest.raises(HTTPException) as excinfo:
        await reading.tts_reading_item(
            item_id=123,
            payload=reading.ReadingTTSRequest(model="kokoro"),
            current_user=SimpleNamespace(id=42),
        )

    assert excinfo.value.status_code == 400
    assert excinfo.value.detail == "reading_item_fetch_failed"
    logger_stub.error.assert_called_once_with("reading_tts_get_failed")


class _ArchiveService:
    def __init__(self, collections):
        self.collections = collections

    def get_item(self, *_args, **_kwargs):
        return _reading_action_row()


@pytest.mark.asyncio
async def test_create_reading_archive_sanitizes_outputs_dir_log(monkeypatch):
    logger_stub = MagicMock()

    class _FailingDir:
        def mkdir(self, **_kwargs):
            raise RuntimeError("mkdir exploded at /private/reading-outputs")

    monkeypatch.setattr(reading, "logger", logger_stub)
    monkeypatch.setattr(reading, "_service_for_user", lambda _user: _ArchiveService(collections=object()))
    monkeypatch.setattr(reading, "_outputs_dir_for_user", lambda _user_id: _FailingDir())

    with pytest.raises(HTTPException) as excinfo:
        await reading.create_reading_archive(
            item_id=123,
            payload=reading.ReadingArchiveCreateRequest(format="md", source="text"),
            current_user=SimpleNamespace(id=42),
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "storage_unavailable"
    logger_stub.error.assert_called_once_with("reading_archive_outputs_dir_failed")


@pytest.mark.asyncio
async def test_create_reading_archive_sanitizes_write_log(monkeypatch):
    logger_stub = MagicMock()

    class _OutputDir:
        def mkdir(self, **_kwargs):
            return None

    class _FailingPath:
        def write_text(self, *_args, **_kwargs):
            raise OSError("write exploded at /private/reading-outputs/archive.md")

    monkeypatch.setattr(reading, "logger", logger_stub)
    monkeypatch.setattr(reading, "_service_for_user", lambda _user: _ArchiveService(collections=object()))
    monkeypatch.setattr(reading, "_outputs_dir_for_user", lambda _user_id: _OutputDir())
    monkeypatch.setattr(reading, "_resolve_output_path_for_user", lambda *_args, **_kwargs: _FailingPath())

    with pytest.raises(HTTPException) as excinfo:
        await reading.create_reading_archive(
            item_id=123,
            payload=reading.ReadingArchiveCreateRequest(format="md", source="text"),
            current_user=SimpleNamespace(id=42),
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "reading_archive_write_failed"
    logger_stub.error.assert_called_once_with("reading_archive_write_failed")


@pytest.mark.asyncio
async def test_create_reading_archive_sanitizes_db_insert_log(monkeypatch):
    logger_stub = MagicMock()

    class _OutputDir:
        def mkdir(self, **_kwargs):
            return None

    class _Path:
        def write_text(self, *_args, **_kwargs):
            return None

        def unlink(self, **_kwargs):
            return None

    class _FailingCollections:
        def create_output_artifact(self, **_kwargs):
            raise RuntimeError("output insert exploded at /private/collections.db")

    monkeypatch.setattr(reading, "logger", logger_stub)
    monkeypatch.setattr(reading, "_service_for_user", lambda _user: _ArchiveService(_FailingCollections()))
    monkeypatch.setattr(reading, "_outputs_dir_for_user", lambda _user_id: _OutputDir())
    monkeypatch.setattr(reading, "_resolve_output_path_for_user", lambda *_args, **_kwargs: _Path())

    with pytest.raises(HTTPException) as excinfo:
        await reading.create_reading_archive(
            item_id=123,
            payload=reading.ReadingArchiveCreateRequest(format="md", source="text"),
            current_user=SimpleNamespace(id=42),
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "reading_archive_db_failed"
    logger_stub.error.assert_called_once_with("reading_archive_db_failed")


@pytest.mark.asyncio
async def test_create_reading_archive_sanitizes_cleanup_log(monkeypatch):
    logger_stub = MagicMock()

    class _OutputDir:
        def mkdir(self, **_kwargs):
            return None

    class _Path:
        def write_text(self, *_args, **_kwargs):
            return None

        def unlink(self, **_kwargs):
            raise OSError("cleanup exploded at /private/reading-outputs/archive.md")

    class _FailingCollections:
        def create_output_artifact(self, **_kwargs):
            raise RuntimeError("output insert exploded at /private/collections.db")

    monkeypatch.setattr(reading, "logger", logger_stub)
    monkeypatch.setattr(reading, "_service_for_user", lambda _user: _ArchiveService(_FailingCollections()))
    monkeypatch.setattr(reading, "_outputs_dir_for_user", lambda _user_id: _OutputDir())
    monkeypatch.setattr(reading, "_resolve_output_path_for_user", lambda *_args, **_kwargs: _Path())

    with pytest.raises(HTTPException) as excinfo:
        await reading.create_reading_archive(
            item_id=123,
            payload=reading.ReadingArchiveCreateRequest(format="md", source="text"),
            current_user=SimpleNamespace(id=42),
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "reading_archive_db_failed"
    logger_stub.warning.assert_called_once_with("reading_archive_cleanup_failed")


@pytest.mark.asyncio
async def test_export_reading_items_sanitizes_highlights_log(monkeypatch):
    logger_stub = MagicMock()

    class _FailingCollections:
        def list_highlights_by_item(self, **_kwargs):
            raise RuntimeError("highlight lookup exploded at /private/highlights.db")

    class _Service:
        collections = _FailingCollections()

        def list_items(self, **_kwargs):
            return [_reading_action_row()], 1

    monkeypatch.setattr(reading, "logger", logger_stub)
    monkeypatch.setattr(reading, "_service_for_user", lambda _user: _Service())

    response = await reading.export_reading_items(
        include_highlights=True,
        format="jsonl",
        current_user=SimpleNamespace(id=42),
    )

    assert response.status_code == 200
    logger_stub.debug.assert_called_once_with("reading_export_highlights_fetch_failed")
