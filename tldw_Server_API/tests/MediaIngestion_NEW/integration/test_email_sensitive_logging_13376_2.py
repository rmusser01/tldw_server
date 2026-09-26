"""Capture actual INFO+ Loguru output using exclusively synthetic email data."""

import json
import zipfile
from email.message import EmailMessage
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile
from loguru import logger

from tldw_Server_API.app.api.v1.endpoints import email as email_endpoint
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.Ingestion_Media_Processing import input_sourcing
from tldw_Server_API.app.core.Ingestion_Media_Processing.Email import Email_Processing_Lib as email_lib
from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import (
    FileValidationError,
    FileValidator,
    ValidationResult,
)
from tldw_Server_API.tests.MediaIngestion_NEW.integration.test_email_offline_ingestion import (
    OFFLINE_OPTIONS,
)
from tldw_Server_API.tests.MediaIngestion_NEW.integration.test_email_offline_ingestion import (
    offline_client as offline_client,
)

pytestmark = pytest.mark.integration

SENTINELS = (
    "email_body_sentinel_13376",
    "email_header_sentinel_13376",
    "email_credential_sentinel_13376",
    "email_metadata_sentinel_13376",
)
ECHO = " ".join(SENTINELS)
FILENAME = f"{SENTINELS[3]}.eml"


@pytest.fixture()
def captured_logs():
    """Include structured extras and tracebacks so hidden exception data fails."""
    output = []
    sink = logger.add(output.append, level="INFO", format="{level} {message} {extra}", backtrace=True, diagnose=True,
        filter=lambda record: not (record["name"] or "").startswith(("httpx", "httpcore")))
    try:
        yield output
    finally:
        logger.remove(sink)


def assert_private(output):
    """Sentinels must be absent from rendered messages, extras and tracebacks."""
    rendered = "\n".join(str(message) for message in output)
    assert not any(sentinel in rendered for sentinel in SENTINELS), rendered


def sensitive_message():
    """Use body, headers and credentials that have no real account meaning."""
    message = EmailMessage()
    message["From"] = f"{SENTINELS[1]} <sender@example.test>"
    message["To"] = "reader@example.test"
    message["Subject"] = SENTINELS[1]
    message["Message-ID"] = f"<{SENTINELS[3]}@example.test>"
    message["Authorization"] = f"Bearer {SENTINELS[2]}"
    message.set_content(SENTINELS[0])
    return message.as_bytes()


@pytest.mark.asyncio
async def test_saved_email_upload_logs_safe_outcome(tmp_path, captured_logs):
    uploaded = UploadFile(filename=FILENAME, file=BytesIO(sensitive_message()))
    saved, errors = await input_sourcing.save_uploaded_files(
        [uploaded], tmp_path, FileValidator(), allowed_extensions=[".eml"]
    )
    assert len(saved) == 1 and not errors
    assert any(message.record["level"].name == "INFO" for message in captured_logs)
    assert_private(captured_logs)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "issues", "rejection"])
async def test_upload_validation_failure_logs_safe_diagnostics(tmp_path, monkeypatch, captured_logs, failure):
    def fail_validation(*_args, **_kwargs):
        if failure == "exception":
            raise RuntimeError(ECHO)
        if failure == "issues":
            raise FileValidationError(ECHO)
        return ValidationResult(False, issues=[ECHO])

    monkeypatch.setattr(input_sourcing, "process_and_validate_file", fail_validation)
    uploaded = UploadFile(filename=FILENAME, file=BytesIO(sensitive_message()))
    saved, errors = await input_sourcing.save_uploaded_files(
        [uploaded], tmp_path, FileValidator(), allowed_extensions=[".eml"]
    )
    assert not saved and errors
    assert any(message.record["level"].no >= 30 for message in captured_logs)
    assert_private(captured_logs)


@pytest.mark.parametrize("member", [FILENAME, f"../{FILENAME}"])
def test_email_archive_validation_logs_safe_success_or_rejection(tmp_path, captured_logs, member):
    archive = tmp_path / f"{SENTINELS[3]}.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member, sensitive_message())
    result = FileValidator().validate_archive_contents(archive)
    assert bool(result) == (not member.startswith("../"))
    assert captured_logs
    assert_private(captured_logs)


@pytest.mark.parametrize("stage", ["parse", "chunk"])
def test_parser_failure_logs_exception_type_without_email_data(monkeypatch, captured_logs, stage):
    def echo_error(*_args, **_kwargs):
        raise ValueError(ECHO)

    monkeypatch.setattr(email_lib, "parse_eml_bytes" if stage == "parse" else "improved_chunking_process", echo_error)
    result = email_lib.process_email_task(
        file_bytes=sensitive_message(), filename=FILENAME, perform_chunking=(stage == "chunk"), perform_analysis=False
    )
    assert result["status"] == ("Error" if stage == "parse" else "Success")
    assert any("ValueError" in message.record["message"] for message in captured_logs)
    assert_private(captured_logs)


@pytest.mark.asyncio
async def test_email_search_failure_logs_type_without_query_or_exception_data(monkeypatch, captured_logs):
    def fail_search(**_kwargs):
        raise DatabaseError(ECHO)

    monkeypatch.setitem(email_endpoint.settings, "EMAIL_OPERATOR_SEARCH_ENABLED", True)
    with pytest.raises(HTTPException) as error:
        await email_endpoint.search_email_messages(q=ECHO, limit=50, offset=0, cursor=None, db=SimpleNamespace(search_email_messages=fail_search))
    assert error.value.status_code == 500
    assert any("DatabaseError" in message.record["message"] for message in captured_logs)
    assert_private(captured_logs)


@pytest.mark.parametrize("container", ["eml", "zip", "mbox"])
def test_email_upload_persistence_search_logs_remain_private(offline_client, captured_logs, container):
    filename, content, options = sensitive_upload(container)
    response = offline_client.post(
        "/api/v1/media/add", files={"files": (filename, content, "application/octet-stream")}, data=options
    )
    assert response.status_code == 200, response.text
    assert response.json()["results"][0]["status"] == "Success", response.text
    found = offline_client.get("/api/v1/email/search", params={"q": SENTINELS[0]})
    assert found.status_code == 200 and found.json()["pagination"]["total"] == 1
    assert_private(captured_logs)


@pytest.mark.parametrize("method", ["single", "batch"])
def test_backend_native_email_query_failure_logs_type_without_sql_data(captured_logs, method):
    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
    from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import execution_ops

    def fail_execute(*_args, **_kwargs):
        raise BackendDatabaseError(ECHO)

    db = SimpleNamespace(
        backend_type=BackendType.POSTGRESQL,
        _prepare_backend_statement=lambda query, params: (query, params),
        _prepare_backend_many_statement=lambda query, params: (query, params),
        backend=SimpleNamespace(execute=fail_execute, execute_many=fail_execute),
    )
    with pytest.raises(DatabaseError):
        if method == "single":
            execution_ops._execute_with_connection(db, object(), f"SELECT '{ECHO}'", (ECHO,))
        else:
            execution_ops._executemany_with_connection(db, object(), f"SELECT '{ECHO}'", [(ECHO,)])
    assert any(message.record["level"].name == "ERROR" for message in captured_logs)
    assert_private(captured_logs)


@pytest.mark.parametrize("method", ["single", "batch", "sync"])
def test_sqlite_email_query_failure_logs_type_without_driver_data(captured_logs, method):
    import sqlite3

    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import execution_ops

    def fail_execute(*_args, **_kwargs):
        if method == "sync":
            raise sqlite3.IntegrityError(f"sync error {ECHO}")
        raise sqlite3.OperationalError(ECHO)

    conn = SimpleNamespace(cursor=lambda: SimpleNamespace(execute=fail_execute, executemany=fail_execute))
    db = SimpleNamespace(
        backend_type=BackendType.SQLITE,
        _prepare_backend_statement=lambda query, params: (query, params),
        _prepare_backend_many_statement=lambda query, params: (query, params),
        _get_txn_conn=lambda: conn,
    )
    with pytest.raises((DatabaseError, sqlite3.IntegrityError)):
        if method == "batch":
            execution_ops.execute_many(db, "SELECT ?", [(ECHO,)])
        else:
            execution_ops.execute_query(db, "SELECT ?", (ECHO,))
    assert any(message.record["level"].name == "ERROR" for message in captured_logs)
    assert_private(captured_logs)


@pytest.mark.parametrize("operation", ["sync", "version", "invalid_version"])
def test_email_sync_failure_logs_safe_type_and_preserves_failure(tmp_path, captured_logs, caplog, operation):
    import sqlite3

    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import sync_utility_ops

    def fail_execute(*_args, **_kwargs):
        if operation == "invalid_version":
            return SimpleNamespace(fetchone=lambda: {"version": ECHO})
        raise sqlite3.OperationalError(ECHO)

    db = SimpleNamespace(
        backend_type=BackendType.SQLITE,
        client_id="synthetic",
        _get_current_utc_timestamp_str=lambda: "2026-09-26T00:00:00Z",
        _resolve_scope_ids=lambda: (None, None),
    )
    conn = SimpleNamespace(execute=fail_execute)
    if operation == "invalid_version":
        assert sync_utility_ops._get_next_version(db, conn, "Media", "id", ECHO) is None
    else:
        with pytest.raises(DatabaseError):
            if operation == "sync":
                sync_utility_ops._log_sync_event(db, conn, "Media", ECHO, "create", 1, {"content": ECHO})
            else:
                sync_utility_ops._get_next_version(db, conn, "Media", "id", ECHO)
    assert_private([*captured_logs, caplog.text])
    assert any(message.record["level"].name == "ERROR" for message in captured_logs)


def test_email_repository_rollback_does_not_log_exception_or_locals(tmp_path, monkeypatch, captured_logs):
    from tldw_Server_API.app.core.DB_Management.media_db.errors import InputError
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

    db = MediaDatabase(db_path=str(tmp_path / "rollback.db"), client_id="synthetic")

    def fail_sync(*_args, **_kwargs):
        raise InputError(ECHO)

    monkeypatch.setattr(db, "_log_sync_event", fail_sync)
    try:
        with pytest.raises(InputError):
            db.add_media_with_keywords(
                url=FILENAME, title=SENTINELS[1], media_type="email", content=SENTINELS[0],
                safe_metadata=json.dumps({"email": {"message_id": f"<{SENTINELS[3]}@example.test>", "authorization": SENTINELS[2]}}),
            )
        assert any(message.record["level"].name == "ERROR" for message in captured_logs)
        assert_private(captured_logs)
    finally:
        db.close_connection()


def test_html_email_sanitizer_fallback_does_not_log_content(monkeypatch, captured_logs):
    import bleach

    def fail_cleaner(*_args, **_kwargs):
        raise ValueError(ECHO)

    monkeypatch.setattr(bleach, "Cleaner", fail_cleaner)
    sanitized = FileValidator().sanitize_html_content(f"<p>{SENTINELS[0]}</p>")
    assert SENTINELS[0] in sanitized
    assert any("ValueError" in message.record["message"] for message in captured_logs)
    assert_private(captured_logs)


def test_yara_email_match_logs_count_without_sensitive_metadata(tmp_path, captured_logs):
    validator = FileValidator()
    validator.yara_available = True
    match = SimpleNamespace(rule=SENTINELS[1], namespace=SENTINELS[3], tags=[SENTINELS[2]], meta={"body": SENTINELS[0]})
    validator.compiled_yara_rules = SimpleNamespace(match=lambda **_kwargs: [match])
    passed, details = validator._scan_file_with_yara(tmp_path / FILENAME)
    assert not passed and details
    assert any("match_count=1" in message.record["message"] for message in captured_logs)
    assert_private(captured_logs)


def test_email_chunk_options_log_counts_without_option_values(captured_logs):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.chunking_options import prepare_chunking_options_dict

    options = prepare_chunking_options_dict({"media_type": "email", "perform_chunking": True, "chunk_method": SENTINELS[3]})
    assert options["method"] == SENTINELS[3]
    assert captured_logs
    assert_private(captured_logs)



def sensitive_upload(container):
    """Create real upload containers with sensitive-looking synthetic metadata."""
    if container == "zip":
        data = BytesIO()
        with zipfile.ZipFile(data, "w") as bundle:
            bundle.writestr(FILENAME, sensitive_message())
        return f"{SENTINELS[3]}.zip", data.getvalue(), {**OFFLINE_OPTIONS, "accept_archives": "true"}
    if container == "mbox":
        data = b"From sender@example.test Sat Sep 26 00:00:00 2026\n" + sensitive_message() + b"\n"
        return f"{SENTINELS[3]}.mbox", data, {**OFFLINE_OPTIONS, "accept_mbox": "true"}
    return FILENAME, sensitive_message(), OFFLINE_OPTIONS


@pytest.mark.parametrize("container", ["eml", "zip"])
def test_native_email_upsert_failure_is_visible_without_echoing_data(offline_client, monkeypatch, captured_logs, container):
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

    def fail_native(*_args, **_kwargs):
        raise DatabaseError(ECHO)

    monkeypatch.setattr(MediaDatabase, "upsert_email_message_graph", fail_native)
    filename, content, options = sensitive_upload(container)
    response = offline_client.post(
        "/api/v1/media/add", files={"files": (filename, content, "application/octet-stream")}, data=options
    )
    assert response.status_code == 200, response.text
    assert response.json()["results"][0]["status"] == "Success", response.text
    assert any("DatabaseError" in message.record["message"] and message.record["level"].no >= 30 for message in captured_logs)
    assert_private(captured_logs)



def test_email_keyword_failure_logs_safe_type_without_keyword_or_traceback(tmp_path, monkeypatch, captured_logs):
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
    from tldw_Server_API.app.core.DB_Management.media_db.repositories.keywords_repository import KeywordsRepository

    db = MediaDatabase(db_path=str(tmp_path / "keyword.db"), client_id="synthetic")

    def fail_query(*_args, **_kwargs):
        raise DatabaseError(ECHO)

    monkeypatch.setattr(db, "_fetchone_with_connection", fail_query)
    try:
        with pytest.raises(DatabaseError):
            KeywordsRepository(db).add(SENTINELS[3])
        assert any(message.record["level"].name == "ERROR" for message in captured_logs)
        assert_private(captured_logs)
    finally:
        db.close_connection()


def test_email_sqlite_rollback_failure_logs_only_type(tmp_path, captured_logs, caplog):
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

    db = MediaDatabase(db_path=str(tmp_path / "transaction.db"), client_id="synthetic")
    try:
        with pytest.raises(ValueError):
            with db.transaction():
                raise ValueError(ECHO)
        assert_private([*captured_logs, caplog.text])
        assert any("ValueError" in message.record["message"] for message in captured_logs)
    finally:
        db.close_connection()


@pytest.mark.parametrize("index", ["media", "keyword"])
def test_email_fts_failure_logs_type_without_indexed_data(captured_logs, index):
    import sqlite3

    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import fts_ops

    def fail_index(*_args, **_kwargs):
        raise sqlite3.OperationalError(ECHO)

    db = SimpleNamespace(backend_type=BackendType.SQLITE)
    conn = SimpleNamespace(execute=fail_index)
    with pytest.raises(DatabaseError):
        if index == "media":
            fts_ops._update_fts_media(db, conn, 1, SENTINELS[1], SENTINELS[0])
        else:
            fts_ops._update_fts_keyword(db, conn, 1, SENTINELS[3])
    assert any("OperationalError" in message.record["message"] for message in captured_logs)
    assert_private(captured_logs)


def test_email_document_version_failure_logs_type_without_content(tmp_path, monkeypatch, captured_logs):
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

    db = MediaDatabase(db_path=str(tmp_path / "version.db"), client_id="synthetic")
    try:
        media_id, _, _ = db.add_media_with_keywords(
            title=SENTINELS[1], media_type="email", content=SENTINELS[0],
            safe_metadata=json.dumps({"email": {"message_id": f"<{SENTINELS[3]}@example.test>"}}),
        )

        def fail_sync(*_args, **_kwargs):
            raise DatabaseError(ECHO)

        monkeypatch.setattr(db, "_log_sync_event", fail_sync)
        with pytest.raises(DatabaseError):
            db.create_document_version(media_id, ECHO, safe_metadata=json.dumps({"authorization": SENTINELS[2]}))
        assert any("DatabaseError" in message.record["message"] for message in captured_logs)
        assert_private(captured_logs)
    finally:
        db.close_connection()


@pytest.mark.parametrize("path", ["/api/v1/email/search", "/api/v1/media/search", "/api/v1/email/messages/1"])
@pytest.mark.parametrize("encoded", [False, True])
def test_uvicorn_email_request_query_is_redacted_in_actual_loguru_output(uvicorn_capture, path, encoded):
    import logging
    from urllib.parse import quote

    interceptor, captured_logs = uvicorn_capture

    values = ["".join(f"%{ord(character):02X}" for character in value) if encoded else value for value in SENTINELS]
    query = "&".join(f"q{number}={value}" for number, value in enumerate(values))
    query += f"&quoted='{SENTINELS[1]}'"
    if encoded:
        query += f"&header={quote('Authorization: Bearer ' + SENTINELS[2], safe='')}"
    record = logging.LogRecord(
        name="uvicorn.access", level=logging.INFO, pathname=__file__, lineno=1,
        msg=f'127.0.0.1:54000 - "GET {path}?{query} HTTP/1.1" 200', args=(), exc_info=None,
    )
    interceptor.emit(record)
    assert any(path in message.record["message"] and 'HTTP/1.1" 200' in message.record["message"] for message in captured_logs)
    assert not any("q0=" in message.record["message"] for message in captured_logs)
    assert_private(captured_logs)


def test_unrelated_access_query_is_preserved(uvicorn_capture):
    import logging

    interceptor, captured_logs = uvicorn_capture

    record = logging.LogRecord(
        name="uvicorn.access", level=logging.INFO, pathname=__file__, lineno=1,
        msg='127.0.0.1:54000 - "GET /api/v1/jobs?status=running HTTP/1.1" 200', args=(), exc_info=None,
    )
    interceptor.emit(record)
    assert any("/api/v1/jobs?status=running" in message.record["message"] for message in captured_logs)



@pytest.fixture()
def uvicorn_capture():
    """The app configures sinks during import, so capture after that boundary."""
    from tldw_Server_API.app import main as app_main

    output = []
    sink = logger.add(output.append, level="INFO", format="{level} {message} {extra}", backtrace=True, diagnose=True)
    try:
        yield app_main.InterceptHandler(), output
    finally:
        logger.remove(sink)


def test_email_multipart_worker_failure_logs_type_without_echoed_data(offline_client, monkeypatch, captured_logs):
    from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence

    async def fail_worker(**_kwargs):
        raise ValueError(ECHO)

    monkeypatch.setattr(persistence, "process_document_like_item", fail_worker)
    response = offline_client.post(
        "/api/v1/media/add", files={"files": (FILENAME, sensitive_message(), "message/rfc822")}, data=OFFLINE_OPTIONS
    )
    assert response.json()["results"][0]["status"] == "Error", response.text
    assert any("ValueError" in str(message) and message.record["level"].no >= 30 for message in captured_logs)
    assert_private(captured_logs)


def test_email_multipart_setup_failure_logs_only_error_type(offline_client, monkeypatch, captured_logs):
    def fail_temp_dir(_self):
        raise OSError(ECHO)

    monkeypatch.setattr(input_sourcing.TempDirManager, "__enter__", fail_temp_dir)
    response = offline_client.post(
        "/api/v1/media/add", files={"files": (FILENAME, sensitive_message(), "message/rfc822")}, data=OFFLINE_OPTIONS
    )
    assert response.status_code == 500, response.text
    assert any("OSError" in str(message) and message.record["level"].no >= 30 for message in captured_logs)
    assert_private(captured_logs)


@pytest.mark.parametrize("failure_type", [ValueError, HTTPException])
def test_email_multipart_setup_exception_keeps_type_only(offline_client, monkeypatch, captured_logs, failure_type):
    def fail_temp_dir(_self):
        if failure_type is HTTPException:
            raise HTTPException(status_code=422, detail=ECHO)
        raise failure_type(ECHO)

    monkeypatch.setattr(input_sourcing.TempDirManager, "__enter__", fail_temp_dir)
    response = offline_client.post(
        "/api/v1/media/add", files={"files": (FILENAME, sensitive_message(), "message/rfc822")}, data=OFFLINE_OPTIONS
    )
    assert response.status_code == (422 if failure_type is HTTPException else 500), response.text
    assert any(failure_type.__name__ in str(message) for message in captured_logs)
    assert_private(captured_logs)


@pytest.mark.parametrize("stage", ["initialize", "write"])
def test_email_collections_failure_does_not_echo_metadata(monkeypatch, captured_logs, stage):
    from tldw_Server_API.app.core.DB_Management import Collections_DB
    from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence

    def fail(*_args, **_kwargs):
        raise ValueError(ECHO)

    fake = SimpleNamespace(upsert_content_item=fail, close=lambda: None)
    monkeypatch.setattr(Collections_DB.CollectionsDatabase, "for_user", fail if stage == "initialize" else lambda **_kwargs: fake)
    results = [{"status": "Success", "db_id": 1, "media_type": "email", "input_ref": FILENAME,
                "content": SENTINELS[0], "metadata": {"title": SENTINELS[1]}}]
    persistence.sync_media_add_results_to_collections(
        results=results, form_data=SimpleNamespace(media_type="email"), current_user=SimpleNamespace(id=1), db=SimpleNamespace()
    )
    assert any("ValueError" in str(message) for message in captured_logs)
    assert_private(captured_logs)


def test_email_metadata_contract_diagnostic_excludes_input_ref(captured_logs):
    from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence

    result = {"status": "Success", "input_ref": FILENAME, "metadata": {}}
    persistence._enforce_metadata_contract_on_result(
        result=result, media_type="email", form_data=SimpleNamespace(metadata_contract_policy="warn"),
        path_kind="upload", processor="email",
    )
    assert result["warnings"]
    assert captured_logs
    assert_private(captured_logs)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["jobs", "background_result", "background_exception"])
async def test_email_embedding_dispatch_diagnostics_exclude_content(monkeypatch, captured_logs, stage):
    from fastapi import BackgroundTasks

    from tldw_Server_API.app.api.v1.endpoints import media_embeddings
    from tldw_Server_API.app.core.Embeddings import jobs_adapter
    from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence

    def fail(*_args, **_kwargs):
        raise ValueError(ECHO)

    async def get_content(*_args, **_kwargs):
        return {"media_item": {"content": SENTINELS[0], "metadata": {"header": SENTINELS[1]}}}

    async def generate(*_args, **_kwargs):
        if stage == "background_exception":
            raise ValueError(ECHO)
        return {"status": "error", "error": ECHO, "metadata": {"credential": SENTINELS[2]}}

    monkeypatch.setattr(jobs_adapter, "EmbeddingsJobsAdapter", lambda: SimpleNamespace(create_job=fail))
    monkeypatch.setattr(media_embeddings, "get_media_content", get_content)
    monkeypatch.setattr(media_embeddings, "generate_embeddings_for_media", generate)
    tasks = BackgroundTasks()
    results = [{"status": "Success", "db_id": 1, "media_type": "email", "input_ref": FILENAME}]
    await persistence.schedule_media_add_embeddings(
        results=results, form_data=SimpleNamespace(media_type="email", generate_embeddings=True,
            embedding_dispatch_mode="jobs" if stage == "jobs" else "background"),
        background_tasks=tasks, db=SimpleNamespace(mark_embeddings_error=fail), current_user=SimpleNamespace(id=1),
    )
    await tasks()
    assert captured_logs
    assert any("ValueError" in str(message) for message in captured_logs)
    assert_private(captured_logs)


def test_embedding_completion_failure_logs_only_type(monkeypatch, captured_logs):
    from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence

    def fail(*_args, **_kwargs):
        raise ValueError(ECHO)

    monkeypatch.setattr(persistence, "mark_media_as_processed", fail)
    assert persistence._mark_media_embeddings_complete(SimpleNamespace(), 1) is False
    assert any("ValueError" in str(message) for message in captured_logs)
    assert_private(captured_logs)


@pytest.mark.parametrize("stage", ["context", "clear", "close_all"])
def test_sqlite_pool_failures_keep_diagnostics_without_exception_locals(captured_logs, stage):
    import threading

    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
    from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteConnectionPool

    pool = SQLiteConnectionPool(":memory:", DatabaseConfig(backend_type=BackendType.SQLITE))
    failure = RuntimeError(ECHO)
    try:
        if stage == "context":
            with pytest.raises(RuntimeError) as raised:
                with pool.connection():
                    private_local = ECHO
                    assert private_local
                    raise failure
            assert raised.value is failure
        else:
            def fail_close():
                raise failure

            pool._connections[threading.get_ident()] = SimpleNamespace(close=fail_close)
            if stage == "clear":
                pool.clear_thread_local_connection()
                assert not pool._connections
            else:
                pool.close_all()
                assert pool._closed and not pool._connections
        assert any("RuntimeError" in str(message) for message in captured_logs)
        assert_private(captured_logs)
    finally:
        pool.close_all()


@pytest.mark.parametrize("stage", ["columns", "schema"])
def test_email_collections_database_failure_diagnostics_exclude_caller_data(captured_logs, stage):
    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
    from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase

    failure = ValueError(ECHO)

    def fail(*_args, **_kwargs):
        raise failure

    # Invoke the real bootstrap boundary without another user's runtime database.
    database = SimpleNamespace(backend=SimpleNamespace(backend_type=BackendType.SQLITE, execute=fail, create_tables=fail))
    if stage == "columns":
        assert CollectionsDatabase._sqlite_columns(database, "content_items") == set()
    else:
        with pytest.raises(ValueError) as raised:
            CollectionsDatabase.ensure_schema(database)
        assert raised.value is failure
    assert any("ValueError" in str(message) for message in captured_logs)
    assert_private(captured_logs)


@pytest.mark.parametrize("stage", ["backend", "collections"])
def test_email_sqlite_schema_failure_logs_safe_diagnostics_and_preserves_cause(monkeypatch, captured_logs, stage):
    import sqlite3

    from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
    from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
    from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteBackend
    from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase

    failure = sqlite3.OperationalError(ECHO)

    def fail_schema(_schema):
        raise failure

    # Keep the real backend schema/error boundary; only the failing connection is injected.
    connection = SimpleNamespace(executescript=fail_schema)
    backend = SQLiteBackend(DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=":memory:"))
    monkeypatch.setattr(backend, "get_pool", lambda: SimpleNamespace(get_connection=lambda: connection))
    with pytest.raises(BackendDatabaseError) as raised:
        if stage == "backend":
            backend.create_tables("CREATE TABLE synthetic_email (id INTEGER)", connection=connection)
        else:
            CollectionsDatabase.ensure_schema(SimpleNamespace(backend=backend))
    assert raised.value.__cause__ is failure
    assert any("OperationalError" in str(message) for message in captured_logs)
    assert_private(captured_logs)
