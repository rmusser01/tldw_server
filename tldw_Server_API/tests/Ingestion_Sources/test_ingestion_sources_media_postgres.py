"""Sources background jobs must establish the owner for restricted PostgreSQL writes."""

import asyncio
from contextlib import nullcontext

import aiosqlite
import pytest

from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database, get_media_by_id
from tldw_Server_API.app.core.DB_Management.scope_context import get_scope, scoped_context
from tldw_Server_API.app.core.Ingestion_Sources.service import (
    create_source,
    ensure_ingestion_sources_schema,
    list_source_items,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.services import ingestion_sources_worker as worker
from tldw_Server_API.tests.Ingestion_Sources.test_ingestion_sources_worker import _SQLitePool

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
def source_worker_runtime(tmp_path, monkeypatch):
    """Bind the real source worker and queue to this test's private resource paths."""
    source_root = tmp_path / "allowed" / "rowan"
    source_root.mkdir(parents=True)
    (source_root / "rowan.txt").write_text("Rowan Observatory telescope is Selene.")
    monkeypatch.setenv("INGESTION_SOURCE_ALLOWED_ROOTS", str(tmp_path / "allowed"))
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    monkeypatch.setenv("INGESTION_SOURCES_WORKER_ENABLED", "1")
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    return source_root, JobManager()


@pytest.mark.parametrize("inherited_admin", [False, True])
async def test_source_worker_uses_owner_scope_for_postgres_media(
    tmp_path,
    monkeypatch,
    pg_restricted_backend,
    inherited_admin,
    source_worker_runtime,
):
    source_root, jm = source_worker_runtime
    # Match the schema already initialized before a real API worker starts.
    with scoped_context(user_id=1, is_admin=False):
        media = create_media_database(client_id="1", db_path=tmp_path / "media.db", backend=pg_restricted_backend)
    seen = []

    def sink(*, sink_type, user_id):
        scope = get_scope()
        seen.append(None if scope is None else (scope.user_id, scope.is_admin))
        return media

    monkeypatch.setattr(worker, "_create_sink_db", sink)
    try:
        async with aiosqlite.connect(str(tmp_path / "sources.db")) as db:
            db.row_factory = aiosqlite.Row
            await ensure_ingestion_sources_schema(db)
            source = await create_source(
                db,
                user_id=1,
                payload={
                    "source_type": "local_directory",
                    "sink_type": "media",
                    "policy": "canonical",
                    "config": {"path": str(source_root)},
                },
            )

            async def pool():
                return _SQLitePool(db)

            monkeypatch.setattr(worker, "get_db_pool", pool)
            stop = asyncio.Event()

            queued = jm.create_job(
                domain=worker.DOMAIN, queue=worker.ingestion_sources_queue(), job_type="sync",
                owner_user_id="1", payload={"source_id": source["id"], "user_id": 999},
            )

            async def stop_when_terminal():
                try:
                    while jm.get_job(queued["id"])["status"] not in {"completed", "failed", "cancelled"}:
                        await asyncio.sleep(0.01)
                finally:
                    stop.set()

            context = scoped_context(user_id=999, is_admin=True) if inherited_admin else nullcontext()
            with context:
                before = get_scope()
                watcher = asyncio.create_task(stop_when_terminal())
                try:
                    async with asyncio.timeout(10):
                        await worker.run_ingestion_sources_worker(stop)
                finally:
                    watcher.cancel()
                    await asyncio.gather(watcher, return_exceptions=True)
                assert get_scope() == before  # nosec B101 -- The worker must restore its caller's authorization scope.
            completed = jm.get_job(queued["id"])
            assert completed["status"] == "completed" and completed["result"]["sink_failed_items"] == 0  # nosec B101 -- Real restricted PostgreSQL content write succeeds.
            assert seen == [(1, False)]  # nosec B101 -- Persisted owner replaces absent/admin scope and spoofed payload authority.
            items = await list_source_items(db, source_id=source["id"])
            assert len(items) == 1 and items[0]["sync_status"] == "active"  # nosec B101 -- The source binds the real persisted document.
            media_id = items[0]["binding"]["media_id"]
            with scoped_context(user_id=1, is_admin=False):
                assert get_media_by_id(media, media_id)["content"] == "Rowan Observatory telescope is Selene."  # nosec B101 -- Exact owner content survives.
            with scoped_context(user_id=2, is_admin=False):
                assert get_media_by_id(media, media_id) is None  # nosec B101 -- Another tenant cannot read the imported content.
    finally:
        media.close_connection()
