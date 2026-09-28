"""Queued content authority must survive the real document persistence executor."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest
from psycopg import Cursor, sql

from tldw_Server_API.app.core.Ingestion_Media_Processing.Plaintext import Plaintext_Files as plaintext_files
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database, search_media
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.scope_context import get_scope, scoped_context
from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.services import media_ingest_jobs_worker as worker

pytestmark = pytest.mark.integration
CONTENT = "The amber observatory opens in October."


@pytest.fixture(params=["postgresql", "sqlite"])
def ingest_store(request, monkeypatch, tmp_path):
    """Official fixture owns the database; a disposable role cannot bypass its RLS."""
    admin = backend = None
    role = None
    sessions = []
    receipts = []
    sequences = []
    failure_types = []
    driver_failures = []
    kind = request.param
    if kind == "postgresql":
        config = request.getfixturevalue("pg_database_config")
        admin = DatabaseBackendFactory.create_backend(config)
        role = "ingest_scope_" + uuid4().hex
        password = uuid4().hex
        with admin.transaction() as conn:
            conn.execute(
                sql.SQL(
                    "CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOBYPASSRLS NOINHERIT NOCREATEDB NOCREATEROLE"
                ).format(sql.Identifier(role), sql.Literal(password))
            )
            conn.execute(sql.SQL("GRANT USAGE, CREATE ON SCHEMA public TO {}").format(sql.Identifier(role)))
            conn.execute(
                sql.SQL("GRANT CREATE ON DATABASE {} TO {}").format(
                    sql.Identifier(config.pg_database), sql.Identifier(role)
                )
            )
        backend = DatabaseBackendFactory.create_backend(
            replace(config, pg_user=role, pg_password=password, connection_string=None, pool_size=1, max_overflow=1)
        )

        class ObservedCursor(Cursor):
            def execute(self, query, params=None, **kwargs):
                try:
                    return super().execute(query, params, **kwargs)
                except Exception as exc:
                    if str(query).strip().lower().startswith("insert into media ("):
                        driver_failures.append(
                            {
                                "sqlstate": getattr(exc, "sqlstate", None),
                                "media_primary_key": getattr(getattr(exc, "diag", None), "constraint_name", None)
                                == "media_pkey",
                            }
                        )
                    raise

        pool = backend.get_pool()
        get_connection = pool.get_connection

        def observed_connection():
            conn = get_connection()
            conn.cursor_factory = ObservedCursor
            return conn

        monkeypatch.setattr(pool, "get_connection", observed_connection)
        execute = backend.execute

        def record_insert(query, params=None, connection=None, **kwargs):
            media_insert = str(query).strip().lower().startswith("insert into media (")
            if media_insert:
                receipt = execute(
                    "SELECT current_setting('app.current_user_id') AS owner, "
                    "current_setting('app.is_admin') AS admin, rolsuper, rolbypassrls "
                    "FROM pg_roles WHERE rolname=current_user",
                    connection=connection,
                ).rows[0]
                receipts.append(dict(receipt))
                sequences.append(
                    dict(execute("SELECT last_value, is_called FROM media_id_seq", connection=connection).rows[0])
                )
            try:
                return execute(query, params, connection=connection, **kwargs)
            except Exception as exc:
                if media_insert:
                    failure_types.append(type(exc).__name__)
                raise

        monkeypatch.setattr(backend, "execute", record_insert)

    def factory(client_id, *, db_path=None, **kwargs):
        db = create_media_database(str(client_id), db_path=db_path, backend=backend, **kwargs)
        sessions.append(db)
        return db

    monkeypatch.setattr(worker, "create_media_database", factory)
    monkeypatch.setattr(persistence, "create_media_database", factory)
    monkeypatch.setattr(worker.DatabasePaths, "get_media_db_path", lambda owner: tmp_path / f"media-{owner}.db")
    monkeypatch.setattr(worker, "_mark_collection_item_status", lambda **_: None)
    monkeypatch.setattr(worker, "_sync_collection_item_terminal_result", lambda **_: None)
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    try:
        yield SimpleNamespace(
            kind=kind,
            backend=backend,
            factory=factory,
            receipts=receipts,
            sequences=sequences,
            failure_types=failure_types,
            driver_failures=driver_failures,
            root=tmp_path,
        )
    finally:
        for db in sessions:
            db.close_connection()
        if backend is not None:
            backend.get_pool().close_all()
        if role is not None:
            with admin.transaction() as conn:
                conn.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
                conn.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))
        if admin is not None:
            admin.get_pool().close_all()


def queued_job(store, owner="1", **payload_overrides):
    source = store.root / (uuid4().hex + ".txt")
    source.write_text(CONTENT)
    payload = {
        "source": str(source),
        "source_kind": "file",
        "input_ref": source.name,
        "temp_dir": str(store.root),
        "cleanup_temp_dir": False,
        "options": {
            "media_type": "document",
            "perform_analysis": False,
            "perform_chunking": False,
            "generate_embeddings": False,
        },
        **payload_overrides,
    }
    jm = JobManager()
    row = jm.create_job(
        domain="media_ingest", queue="default", job_type="media_ingest_item", payload=payload, owner_user_id=owner
    )
    return jm, row


def processor(monkeypatch, warnings=None, status="Success", error=None):
    result = {
        "status": status,
        "content": CONTENT,
        "metadata": {"title": "Amber source"},
        "warnings": warnings,
        "error": error,
    }

    def extract(**_):
        return result

    monkeypatch.setattr(plaintext_files, "process_document_content", extract)
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ["1", "2"])
async def test_actual_worker_inserts_with_queued_owner_not_payload_or_ambient_scope(ingest_store, monkeypatch, owner):
    store = ingest_store
    processor(monkeypatch)
    jm, job = queued_job(store, owner, user_id="777", is_admin=True, org_ids=[99])
    with scoped_context(user_id=998, org_ids=[88], is_admin=True) as outer:
        result = await worker._handle_job(job, jm, worker._ProgressState())
        assert get_scope() is outer
    if store.kind == "postgresql":
        assert store.receipts == [{"owner": owner, "admin": "0", "rolsuper": False, "rolbypassrls": False}]
    assert result["media_id"] is not None, result
    db = store.factory(owner, db_path=store.root / f"media-{owner}.db")
    with scoped_context(user_id=int(owner)):
        row = db.get_media_by_id(result["media_id"])
        assert row["content"] == CONTENT
        assert row["owner_user_id"] == int(owner)
    with scoped_context(user_id=777):
        if store.kind == "postgresql":
            assert db.get_media_by_id(result["media_id"]) is None
        assert search_media(db, "Amber", search_fields=["title", "content"])[0] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("warnings", [[], ["Analysis truncated"], ["Analysis truncated", "Parser fallback"]])
async def test_document_terminal_warnings_are_not_duplicated_or_mutated(ingest_store, monkeypatch, warnings, failed):
    supplied = list(warnings)
    status, error = ("Error", "controlled analysis failure") if failed else ("Warning", None)
    result_from_processor = processor(monkeypatch, supplied, status=status, error=error)
    jm, job = queued_job(ingest_store)
    with scoped_context(user_id=1):
        result = await worker._handle_job(job, jm, worker._ProgressState())
    assert result["warnings"] == (warnings or None)
    assert result_from_processor["warnings"] == warnings
    assert result["status"] == status
    assert result["error"] == error
    assert (result["media_id"] is None) is failed


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_owner", [None, "", "device-label", "0", "-1", True])
async def test_worker_rejects_missing_or_invalid_queued_authority(ingest_store, monkeypatch, bad_owner):
    processor(monkeypatch)
    jm, job = queued_job(ingest_store, user_id="1")
    job["owner_user_id"] = bad_owner
    with pytest.raises(worker.MediaIngestJobError, match="owner_user_id"):
        await worker._handle_job(job, jm, worker._ProgressState())


@pytest.mark.asyncio
async def test_unscoped_and_foreign_direct_writes_stay_denied(ingest_store):
    if ingest_store.kind == "sqlite":
        # SQLite's per-file device/provenance writes remain supported.
        db = ingest_store.factory("old-device", db_path=ingest_store.root / "legacy.db")
        assert db.add_media_with_keywords(title="Legacy", media_type="document", content=CONTENT, keywords=[])[0]
        return
    db = ingest_store.factory("1", db_path=ingest_store.root / "denied.db")
    for authority in (None, 2):
        with scoped_context(user_id=authority), pytest.raises(DatabaseError):
            db.add_media_with_keywords(title="Denied", media_type="document", content=CONTENT, keywords=[])


@pytest.mark.asyncio
async def test_sequential_jobs_restore_executor_scope(ingest_store, monkeypatch):
    processor(monkeypatch)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    original = loop.run_in_executor
    monkeypatch.setattr(loop, "run_in_executor", lambda _executor, fn, *args: original(executor, fn, *args))
    try:
        for owner in ("1", "2"):
            jm, job = queued_job(ingest_store, owner)
            result = await worker._handle_job(job, jm, worker._ProgressState())
            if result["media_id"] is None:
                print(
                    "INSERT_SCOPE_PROBE",
                    {
                        "scopes": ingest_store.receipts,
                        "before_insert_sequences": ingest_store.sequences,
                        "driver_failures": ingest_store.driver_failures,
                    },
                )
            assert result["media_id"] is not None, (
                result,
                ingest_store.receipts,
                ingest_store.sequences,
                ingest_store.failure_types,
            )
            assert await original(executor, get_scope) is None
        assert get_scope() is None
    finally:
        executor.shutdown(wait=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_kind", ["error", "cancel"])
async def test_worker_scope_retires_after_failed_or_cancelled_executor(ingest_store, monkeypatch, exit_kind):
    processor(monkeypatch)
    jm, job = queued_job(ingest_store)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    original_submit = loop.run_in_executor
    monkeypatch.setattr(loop, "run_in_executor", lambda _, fn, *args: original_submit(executor, fn, *args))
    original_session = persistence._with_media_db_session
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    observed, restored = [], []

    def controlled_session(**kwargs):
        observed.append(get_scope())
        entered.set()
        try:
            if not release.wait(5):
                raise RuntimeError("fixture release timed out")
            if exit_kind == "error":
                raise DatabaseError("controlled persistence failure")
            return original_session(**kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(persistence, "_with_media_db_session", controlled_session)

    async def run():
        with scoped_context(user_id=77, is_admin=True) as outer:
            try:
                return await worker._handle_job(job, jm, worker._ProgressState())
            finally:
                restored.append(get_scope() is outer)

    task = asyncio.create_task(run())
    try:
        for _ in range(500):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
        if exit_kind == "cancel":
            jm.cancel_job(job["id"], reason="controlled cancellation")
            task.cancel()
            # Persistence currently converts interruption to a Warning; the worker's
            # existing authoritative Jobs cancellation check must still return {}.
            assert await task == {}
        release.set()
        if exit_kind == "error":
            result = await task
            assert result["media_id"] is None
            assert "controlled persistence failure" in result["error"]
        await original_submit(executor, finished.wait, 5)
        assert observed[0] is not None and observed[0].user_id == 1
        assert observed[0].is_admin is False
        assert restored == [True]
        assert await original_submit(executor, get_scope) is None
    finally:
        release.set()
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        executor.shutdown(wait=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["document", "attachment", "archive", "audio", "pdf"])
async def test_direct_authorized_persistence_preserves_scope_for_primary_and_children(ingest_store, monkeypatch, path):
    store = ingest_store
    form = SimpleNamespace(ingest_attachments=True, accept_archives=True)
    result = {
        "status": "Success",
        "content": CONTENT,
        "metadata": {"title": "Direct source"},
        "children": [{"status": "Success", "content": "Child content", "metadata": {"filename": "child.txt"}}],
    }
    if path == "archive":
        result["content"] = None
    if path == "audio":
        result["metadata"]["model"] = "fixture-model"
        result["normalized_stt"] = {"text": CONTENT, "metadata": {"model": "fixture-model", "provider": "fixture"}}
    observed = []
    original_session = persistence._with_media_db_session

    def observe(**kwargs):
        observed.append(get_scope())
        return original_session(**kwargs)

    monkeypatch.setattr(persistence, "_with_media_db_session", observe)
    if path == "pdf":
        from tldw_Server_API.app.core.Ingestion_Media_Processing import visual_ingestion

        original_visual = visual_ingestion.persist_visual_documents_from_analysis

        def observe_visual(**kwargs):
            observed.append(get_scope())
            return original_visual(**kwargs)

        monkeypatch.setattr(visual_ingestion, "persist_visual_documents_from_analysis", observe_visual)
    with scoped_context(user_id=2, org_ids=[7], team_ids=[8]) as scope:
        common = {
            "form_data": form,
            "chunk_options": None,
            "path_kind": "upload",
            "db_path": str(store.root / "direct.db"),
            "client_id": "2",
            "loop": asyncio.get_running_loop(),
            "claims_context": None,
        }
        if path in {"audio", "pdf"}:
            await persistence.persist_primary_av_item(
                process_result=result, media_type=path, original_input_ref="direct.wav", **common
            )
        else:
            await persistence.persist_doc_item_and_children(
                final_result=result,
                media_type="document" if path == "document" else "email",
                item_input_ref="direct.eml",
                processing_filename="direct.eml",
                **common,
            )
        assert observed and all(item is scope for item in observed)
        if path in {"attachment", "archive"}:
            assert result["child_db_results"][0]["db_id"] is not None
        if path != "archive":
            assert result["db_id"] is not None, result
            await persistence._fetch_unvectorized_chunk_count(
                db_path=common["db_path"], client_id="2", media_id=result["db_id"], loop=common["loop"]
            )
        assert all(item is scope for item in observed)
        if path in {"audio", "pdf", "attachment"}:
            assert len(observed) == 3  # primary, transcript/visual/child, chunk count


@pytest.mark.asyncio
async def test_failed_keyword_write_rolls_back_media_and_scope(ingest_store, monkeypatch):
    store = ingest_store
    processor(monkeypatch)
    jm, job = queued_job(store)
    if store.backend is None:
        # Keep this transaction fault at the same real repository boundary on SQLite.
        from tldw_Server_API.app.core.DB_Management.media_db.repositories.keywords_repository import KeywordsRepository

        monkeypatch.setattr(
            KeywordsRepository, "add", lambda *_, **__: (_ for _ in ()).throw(DatabaseError("fixture keyword failure"))
        )
    else:
        execute = store.backend.execute

        def fail_keyword(query, params=None, connection=None, **kwargs):
            if str(query).strip().lower().startswith("insert into keywords"):
                raise DatabaseError("fixture keyword failure")
            return execute(query, params, connection=connection, **kwargs)

        monkeypatch.setattr(store.backend, "execute", fail_keyword)
    # Avoid a fake writer: the real document's keywords trigger the failure after Media INSERT.
    import json

    payload = job["payload"] if isinstance(job["payload"], dict) else json.loads(job["payload"])
    payload["options"]["keywords"] = "rollback-fixture"
    job["payload"] = payload
    result = await worker._handle_job(job, jm, worker._ProgressState())
    assert result["media_id"] is None
    db = store.factory("1", db_path=store.root / "media-1.db")
    with scoped_context(user_id=1):
        assert search_media(db, "Amber", search_fields=["title"])[0] == []
    assert get_scope() is None
