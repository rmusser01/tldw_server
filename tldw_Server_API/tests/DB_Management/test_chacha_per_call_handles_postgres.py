"""Background code that opens a ChaCha handle per call returns its checkouts (TASK-13861 AC2).

Each case runs the real code that opens the handle, on the shared PostgreSQL content
backend, with garbage collection off: a dropped handle is cyclic garbage, so a
return that waited for the collector would not count. Services that only decide
what to read are stubbed with one read through the handle.
"""

import asyncio
import contextvars
import gc
import threading
from collections import Counter
from contextlib import asynccontextmanager
from types import SimpleNamespace

import aiosqlite
import pytest

from tldw_Server_API.app.core.DB_Management import ChaChaNotes_DB as chacha_module
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.Jobs.manager import JobManager

pytestmark = pytest.mark.integration


class _ReadingChatbookService:
    """Stands in for ChatbookService: every call reads once through the handle it was given."""

    def __init__(self, user_id, db, user_id_int=None):  # noqa: ARG002 - ChatbookService's signature
        self.db = db
        db.get_character_card_by_name("nobody")

    def _read(self, *_args, **_kwargs):
        self.db.get_character_card_by_name("nobody")
        return 0

    cleanup_expired_exports = cleanup_import_orphans = _read

    def list_export_jobs(self, **_kwargs):
        self._read()
        return []


@pytest.fixture
def shared_backend(pg_database_config, tmp_path, monkeypatch):
    """The shared content backend every per-call handle resolves, on a recording pool."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    pool = backend.get_pool()
    holders, taken, returned = {}, Counter(), Counter()
    original_get, original_return = pool.get_connection, pool.return_connection

    def get_connection():
        raw = original_get()
        taken[id(raw)] += 1
        holders[id(raw)] = threading.current_thread().name
        return raw

    def return_connection(raw):
        returned[id(raw)] += 1
        return original_return(raw)

    monkeypatch.setattr(pool, "get_connection", get_connection)
    monkeypatch.setattr(pool, "return_connection", return_connection)
    monkeypatch.setattr(chacha_module, "get_content_backend", lambda _config: backend)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    (tmp_path / "user_dbs" / "1").mkdir(parents=True)

    def outstanding():
        return sorted(holders[key] for key, count in (taken - returned).items() for _ in range(count))

    gc.disable()
    try:
        yield SimpleNamespace(outstanding=outstanding, tmp_path=tmp_path)
    finally:
        gc.enable()
        gc.collect()
        pool.close_all()


async def _chatbooks_cleanup_pass(env, monkeypatch):
    from tldw_Server_API.app.services import chatbooks_cleanup_service as cleanup

    stop = asyncio.Event()

    class Service(_ReadingChatbookService):
        def cleanup_import_orphans(self, *_args, **_kwargs):
            stop.set()  # one pass is enough
            return self._read()

    monkeypatch.setattr(cleanup, "ChatbookService", Service)
    monkeypatch.setattr(cleanup, "_enumerate_user_ids", lambda: [1])
    monkeypatch.setenv("CHATBOOKS_CLEANUP_INTERVAL_SEC", "1")
    # The pass builds the ChaCha schema in a fresh database: 37 s once under a loaded sweep.
    await asyncio.wait_for(cleanup.run_chatbooks_cleanup_loop(stop), timeout=60)


async def _chatbooks_core_job(env, monkeypatch):
    from tldw_Server_API.app.services import core_jobs_worker as core_worker

    jobs = JobManager(env.tmp_path / "core-jobs.db")
    jobs.create_job(domain="chatbooks", queue="default", job_type="chatbooks", payload={"action": "noop"}, owner_user_id="1")
    stop = asyncio.Event()
    original_fail = jobs.fail_job

    def fail_and_stop(*args, **kwargs):  # the unknown action ends the only job
        stop.set()
        return original_fail(*args, **kwargs)

    monkeypatch.setattr(jobs, "fail_job", fail_and_stop)
    monkeypatch.setattr(core_worker, "JobManager", lambda: jobs)
    monkeypatch.setattr(core_worker, "ChatbookService", _ReadingChatbookService)
    monkeypatch.setenv("JOBS_POLL_INTERVAL_SECONDS", "0.05")
    await asyncio.wait_for(core_worker.run_chatbooks_core_jobs_worker(stop), timeout=60)


async def _ingestion_notes_sync(env, monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Sources.service import create_source, ensure_ingestion_sources_schema
    from tldw_Server_API.app.services import ingestion_sources_worker as ingestion

    root = env.tmp_path / "allowed" / "docs"
    root.mkdir(parents=True)
    (root / "alpha.md").write_text("# Alpha\n\nbody\n", encoding="utf-8")
    monkeypatch.setenv("INGESTION_SOURCE_ALLOWED_ROOTS", str(env.tmp_path / "allowed"))
    async with aiosqlite.connect(str(env.tmp_path / "sources.sqlite3")) as meta:
        meta.row_factory = aiosqlite.Row
        await ensure_ingestion_sources_schema(meta)
        source = await create_source(
            meta, user_id=1,
            payload={"source_type": "local_directory", "sink_type": "notes", "policy": "canonical", "config": {"path": str(root)}},
        )

        class Pool:
            @asynccontextmanager
            async def transaction(self):
                yield meta

        async def get_pool():
            return Pool()

        jobs = SimpleNamespace(renew_job_lease=lambda *a, **k: None, complete_job=lambda *a, **k: None, fail_job=lambda *a, **k: None)
        monkeypatch.setattr(ingestion, "get_db_pool", get_pool, raising=False)
        await ingestion._process_sync_job(jobs, jid=1, lease_id="lease-1", worker_id="w", source_id=int(source["id"]), user_id=1)


async def _workflow_chatbooks_step(env, monkeypatch, *, context=None):
    from tldw_Server_API.app.core.Chatbooks import chatbook_service
    from tldw_Server_API.app.core.Workflows.adapters.integration import messaging

    monkeypatch.setattr(chatbook_service, "ChatbookService", _ReadingChatbookService)
    monkeypatch.setattr(messaging, "is_test_mode", lambda: False)  # test mode simulates the step
    adapter = messaging.registry.get_adapter("chatbooks")  # what the engine runs (engine.py:1906)
    step = adapter({"action": "list_jobs"}, {"user_id": "1"})
    result = await asyncio.create_task(step, context=context)
    assert "error" not in result, result


async def _workflow_chatbooks_step_after_its_request(env, monkeypatch):
    """A request starts engine.continue_run with asyncio.create_task (workflows.py:3497), which outlives it."""
    with chacha_operation(independent=True):
        inherited = contextvars.copy_context()  # what that task copies; the operation closes with the response
    await _workflow_chatbooks_step(env, monkeypatch, context=inherited)


async def _notification_into_chatbook(env, monkeypatch):
    from tldw_Server_API.app.core.Notifications import service as notifications

    class Documents:
        def __init__(self, db, user_id=None):  # noqa: ARG002
            self.db = db

        def create_manual_document(self, **_kwargs):
            self.db.get_character_card_by_name("nobody")
            return 7

    monkeypatch.setattr(notifications, "DocumentGeneratorService", Documents)
    result = notifications.NotificationsService(user_id=1).deliver_chatbook(title="t", content="c")
    assert result.status == "stored", result


@pytest.mark.parametrize(
    "site",
    [
        _chatbooks_cleanup_pass, _chatbooks_core_job, _ingestion_notes_sync, _workflow_chatbooks_step,
        _workflow_chatbooks_step_after_its_request, _notification_into_chatbook,
    ],
    ids=[
        "chatbooks-cleanup", "chatbooks-core-jobs", "ingestion-notes-sink", "workflows-chatbooks",
        "workflows-chatbooks-after-its-request", "notifications-chatbook",
    ],
)
async def test_a_per_call_handle_returns_its_checkouts_without_waiting_for_gc(shared_backend, monkeypatch, site):
    await site(shared_backend, monkeypatch)

    assert shared_backend.outstanding() == []
