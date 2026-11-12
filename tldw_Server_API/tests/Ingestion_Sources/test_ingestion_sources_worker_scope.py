"""Invalid Sources job ownership cannot become content authorization."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.DB_Management.scope_context import get_scope, scoped_context
from tldw_Server_API.app.services import ingestion_sources_worker as worker
from tldw_Server_API.tests.Ingestion_Sources.test_ingestion_sources_worker import _FakeJobManager


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner", [None, 0, -1, True, 1.5, "", "owner", "9" * 1000, 2**63, pytest.param(10**5000, id="oversized-integer")]
)
async def test_invalid_persisted_source_owner_fails_without_dispatch(monkeypatch, owner):
    stop = asyncio.Event()

    class Jobs(_FakeJobManager):
        acquired = False

        def acquire_next_job(self, **kwargs):
            if self.acquired:
                raise AssertionError("Another job was acquired")
            self.acquired = True
            return {"id": 41, "lease_id": "lease-1", "owner_user_id": owner, "payload": {"source_id": 1, "user_id": 1}}

        def fail_job(self, *args, **kwargs):
            super().fail_job(*args, **kwargs)
            stop.set()

    jobs = Jobs()
    process = AsyncMock(side_effect=lambda *args, **kwargs: stop.set())
    monkeypatch.setenv("INGESTION_SOURCES_WORKER_ENABLED", "1")
    monkeypatch.setattr(worker, "JobManager", lambda: jobs)
    monkeypatch.setattr(worker, "_process_sync_job", process)
    with scoped_context(user_id=999, is_admin=True):
        before = get_scope()
        await worker.run_ingestion_sources_worker(stop)
        assert get_scope() == before  # nosec B101 -- Invalid jobs cannot replace the caller's context.
    process.assert_not_awaited()
    assert jobs.completed is None and jobs.failed["retryable"] is False  # nosec B101 -- Fail closed without granting payload authority.
    assert jobs.failed["jid"] == 41 and jobs.failed["lease_id"] == "lease-1"  # nosec B101 -- Fail only the acquired owned lease.


@pytest.mark.asyncio
async def test_string_source_owner_is_scoped_and_restored_after_handler_error(monkeypatch):
    stop = asyncio.Event()
    seen = []

    class Jobs:
        def acquire_next_job(self, **kwargs):
            stop.set()
            return {"id": 41, "lease_id": "lease-1", "owner_user_id": "1", "payload": {"source_id": 1, "user_id": 999}}

    async def process(*args, **kwargs):
        scope = get_scope()
        seen.append((scope.user_id, scope.is_admin))
        raise RuntimeError("Controlled handler failure")

    monkeypatch.setenv("INGESTION_SOURCES_WORKER_ENABLED", "1")
    monkeypatch.setenv("INGESTION_SOURCES_POLL_INTERVAL_SECONDS", "0")
    monkeypatch.setattr(worker, "JobManager", Jobs)
    monkeypatch.setattr(worker, "_process_sync_job", process)
    with scoped_context(user_id=999, is_admin=True):
        before = get_scope()
        await worker.run_ingestion_sources_worker(stop)
        assert get_scope() == before  # nosec B101 -- Restore authorization after a handler exception.
    assert seen == [(1, False)]  # nosec B101 -- Numeric stored owners are accepted without inherited administrator rights.
