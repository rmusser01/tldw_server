import json

import pytest

from tldw_Server_API.app.api.v1.schemas.data_tables_schemas import DATA_TABLES_MAX_ROWS_LIMIT
from tldw_Server_API.app.core.Data_Tables import jobs_worker
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.exceptions import DataTablesJobError
from tldw_Server_API.app.core.Jobs.manager import JobManager

pytestmark = pytest.mark.unit


class _StubAdapter:
    def __init__(self, payload):
        self._payload = payload

    def chat(self, _request):
        return {"choices": [{"message": {"content": json.dumps(self._payload)}}]}


class _LoggerStub:
    def __init__(self):
        self.warnings: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.debugs: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def warning(self, message, *args, **kwargs):
        self.warnings.append((str(message), args, kwargs))

    def debug(self, message, *args, **kwargs):
        self.debugs.append((str(message), args, kwargs))


class _FailingCloseDb:
    def close_connection(self):
        raise RuntimeError("data tables backend exploded /private/data-tables.db")


def _assert_sanitized_warning(logger_stub: _LoggerStub, expected_message: str) -> None:
    rendered = repr(logger_stub.warnings)
    assert logger_stub.warnings == [(expected_message, (), {})]
    assert "data tables backend exploded" not in rendered
    assert "/private/data-tables.db" not in rendered
    assert "user-4242" not in rendered
    assert "424242" not in rendered


def _assert_sanitized_debug(logger_stub: _LoggerStub, expected_message: str) -> None:
    rendered = repr(logger_stub.debugs)
    assert logger_stub.debugs == [(expected_message, (), {})]
    assert "data tables backend exploded" not in rendered
    assert "/private/data-tables.db" not in rendered
    assert "424242" not in rendered


def test_close_media_db_sanitizes_close_failure_log(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(jobs_worker, "logger", logger_stub)

    jobs_worker._close_media_db("user-4242", _FailingCloseDb())

    _assert_sanitized_warning(
        logger_stub,
        "data_tables: failed to close media db",
    )


def test_close_chacha_db_sanitizes_close_failure_log(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(jobs_worker, "logger", logger_stub)

    jobs_worker._close_chacha_db("user-4242", _FailingCloseDb())

    _assert_sanitized_warning(
        logger_stub,
        "data_tables: failed to close chacha db",
    )


@pytest.mark.asyncio
async def test_data_tables_worker_generates_rows(monkeypatch, tmp_path):
    media_path = tmp_path / "media.db"
    chacha_path = tmp_path / "chacha.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.setattr(jobs_worker, "get_user_media_db_path", lambda _user_id: str(media_path))
    monkeypatch.setattr(jobs_worker, "get_user_chacha_db_path", lambda _user_id: str(chacha_path))

    db = MediaDatabase(db_path=str(media_path), client_id="1")
    table = db.create_data_table(
        name="Worker Table",
        prompt="Summarize chunks",
        description="unit",
        status="queued",
        row_count=0,
    )
    table_id = int(table.get("id"))

    snapshot = {
        "query": "summarize",
        "retrieval": {"sources": ["media_db"]},
        "chunks": [
            {
                "chunk_id": "chunk_1",
                "chunk_text": "alpha beta gamma",
                "score": 0.9,
                "rank": 1,
            }
        ],
    }

    llm_payload = {
        "columns": [
            {"name": "Term", "type": "text"},
            {"name": "Count", "type": "number"},
        ],
        "rows": [
            ["alpha", 1],
            ["beta", 2],
        ],
    }

    monkeypatch.setattr(jobs_worker, "_get_adapter", lambda _provider: _StubAdapter(llm_payload))
    monkeypatch.setattr(jobs_worker, "_resolve_model", lambda *_args, **_kwargs: "test-model")
    monkeypatch.setattr(jobs_worker, "provider_requires_api_key", lambda _p: False)
    monkeypatch.setattr(jobs_worker, "resolve_provider_api_key", lambda *_a, **_k: ("", {}))
    monkeypatch.setattr(jobs_worker, "DEFAULT_LLM_PROVIDER", "openai")
    monkeypatch.setattr(jobs_worker, "load_and_log_configs", lambda: {})

    job = {
        "id": 1,
        "job_type": "data_table_generate",
        "owner_user_id": "1",
        "payload": {
            "table_id": table_id,
            "table_uuid": table.get("uuid"),
            "prompt": "Summarize chunks",
            "sources": [
                {
                    "source_type": "rag_query",
                    "source_id": "summarize",
                    "snapshot": snapshot,
                }
            ],
            "max_rows": 5,
        },
    }

    jm = JobManager()
    result = await jobs_worker._handle_job(job, jm)
    assert result["row_count"] == 2

    refreshed = db.get_data_table(table_id, include_deleted=True)
    assert refreshed["status"] == "ready"
    assert refreshed["row_count"] == 2

    columns = db.list_data_table_columns(table_id)
    rows = db.list_data_table_rows(table_id, limit=10, offset=0)
    assert len(columns) == 2
    assert len(rows) == 2

    row_json = json.loads(rows[0]["row_json"])
    assert set(row_json.keys()) == {columns[0]["column_id"], columns[1]["column_id"]}


def test_dedupe_column_names_is_case_insensitive():
    deduped = jobs_worker._dedupe_column_names(["Name", "name", "NAME"])
    assert deduped == ["Name", "name (2)", "NAME (3)"]


def test_worker_max_rows_limit_matches_api_schema_limit():
    assert jobs_worker._MAX_ROWS_LIMIT == DATA_TABLES_MAX_ROWS_LIMIT


def test_extract_json_payload_accepts_fenced_json_with_think_tags():
    raw = (
        "<think>reasoning</think>\n"
        "```json\n"
        "{\"columns\":[{\"name\":\"A\"}],\"rows\":[[1]]}\n"
        "```"
    )
    payload = jobs_worker._extract_json_payload(raw)
    assert isinstance(payload, dict)
    assert payload["columns"][0]["name"] == "A"


def test_extract_json_payload_rejects_malformed_json():
    with pytest.raises(DataTablesJobError) as exc:
        jobs_worker._extract_json_payload("not valid json")
    assert "llm_response_invalid_json" in str(exc.value)


def test_extract_media_text_uses_document_version_before_transcript(monkeypatch):
    class StubDb:
        def get_media_by_id(
            self,
            media_id: int,
            include_deleted: bool = False,
            include_trash: bool = False,
        ):
            return {"id": media_id, "content": ""}

    monkeypatch.setattr(
        jobs_worker,
        "get_document_version",
        lambda db, media_id, version_number=None, include_content=True: {
            "content": "document version fallback"
        },
    )

    def _should_not_call_transcription(*args, **kwargs):
        raise AssertionError("get_latest_transcription should not be called when document version exists")

    monkeypatch.setattr(
        jobs_worker,
        "get_latest_transcription",
        _should_not_call_transcription,
    )

    result = jobs_worker._extract_media_text(StubDb(), 7)
    assert result == "document version fallback"


def test_extract_media_text_uses_transcript_when_document_version_missing(monkeypatch):
    class StubDb:
        def get_media_by_id(
            self,
            media_id: int,
            include_deleted: bool = False,
            include_trash: bool = False,
        ):
            return {"id": media_id, "content": ""}

    monkeypatch.setattr(
        jobs_worker,
        "get_document_version",
        lambda db, media_id, version_number=None, include_content=True: None,
    )
    monkeypatch.setattr(
        jobs_worker,
        "get_latest_transcription",
        lambda db, media_id: "transcript fallback",
    )

    result = jobs_worker._extract_media_text(StubDb(), 8)
    assert result == "transcript fallback"


def test_extract_media_text_sanitizes_document_version_failure_log(monkeypatch):
    class StubDb:
        def get_media_by_id(
            self,
            media_id: int,
            include_deleted: bool = False,
            include_trash: bool = False,
        ):
            return {"id": media_id, "content": ""}

    logger_stub = _LoggerStub()
    monkeypatch.setattr(jobs_worker, "logger", logger_stub)

    def _raise_backend_failure(*args, **kwargs):
        raise RuntimeError("data tables backend exploded /private/data-tables.db")

    monkeypatch.setattr(jobs_worker, "get_document_version", _raise_backend_failure)
    monkeypatch.setattr(
        jobs_worker,
        "get_latest_transcription",
        lambda db, media_id: "transcript fallback",
    )

    result = jobs_worker._extract_media_text(StubDb(), 424242)

    assert result == "transcript fallback"
    _assert_sanitized_debug(
        logger_stub,
        "data_tables: get_document_version failed while extracting media text",
    )


def test_extract_media_text_sanitizes_latest_transcription_failure_log(monkeypatch):
    class StubDb:
        def get_media_by_id(
            self,
            media_id: int,
            include_deleted: bool = False,
            include_trash: bool = False,
        ):
            return {"id": media_id, "content": ""}

    logger_stub = _LoggerStub()
    monkeypatch.setattr(jobs_worker, "logger", logger_stub)
    monkeypatch.setattr(
        jobs_worker,
        "get_document_version",
        lambda db, media_id, version_number=None, include_content=True: None,
    )

    def _raise_backend_failure(*args, **kwargs):
        raise RuntimeError("data tables backend exploded /private/data-tables.db")

    monkeypatch.setattr(jobs_worker, "get_latest_transcription", _raise_backend_failure)

    with pytest.raises(DataTablesJobError) as exc:
        jobs_worker._extract_media_text(StubDb(), 424242)

    assert "media_missing_content:424242" in str(exc.value)
    _assert_sanitized_debug(
        logger_stub,
        "data_tables: get_latest_transcription failed while extracting media text",
    )
