from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables


class _PoolAcquireContext:
    def __init__(self, pool: _FakeConnectionPool) -> None:
        self._pool = pool

    def __enter__(self) -> sqlite3.Connection:
        self._pool.acquired += 1
        return self._pool.conn

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._pool.released += 1
        return False


class _FakeConnectionPool:
    def __init__(self, db_path: Path) -> None:
        self.conn = sqlite3.connect(db_path)
        self.acquired = 0
        self.released = 0
        self.closed = 0

    def acquire(self) -> _PoolAcquireContext:
        return _PoolAcquireContext(self)

    def close(self) -> None:
        self.closed += 1
        self.conn.close()


@pytest.fixture
def sqlite_jobs_repo(tmp_path):
    from tldw_Server_API.app.core.DB_Management.Jobs_Repository import JobsRepository

    db_path = tmp_path / "jobs_repo.db"
    ensure_jobs_tables(db_path)
    return JobsRepository.for_sqlite(db_path)


class TestJobsRepositorySqlite:
    def test_count_active_jobs_for_user_uses_existing_session(self, sqlite_jobs_repo):
        with sqlite_jobs_repo.session() as session:
            sqlite_jobs_repo.insert_job(
                domain="chatbooks",
                queue="default",
                job_type="export",
                payload_json='{"first": true}',
                owner_user_id="42",
                project_id=None,
                batch_group=None,
                idempotency_key=None,
                priority=5,
                max_retries=3,
                available_at=None,
                request_id="req-1",
                trace_id="trace-1",
                created_at=datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
                session=session,
            )
            sqlite_jobs_repo.insert_job(
                domain="chatbooks",
                queue="default",
                job_type="export",
                payload_json='{"second": true}',
                owner_user_id="42",
                project_id=None,
                batch_group=None,
                idempotency_key=None,
                priority=4,
                max_retries=3,
                available_at=None,
                request_id="req-2",
                trace_id="trace-2",
                created_at=datetime(2026, 3, 17, 12, 1, tzinfo=timezone.utc),
                session=session,
            )

            assert sqlite_jobs_repo.count_active_jobs_for_user("42", session=session) == 2
            assert sqlite_jobs_repo.count_active_jobs_for_user("7", session=session) == 0

    def test_insert_job_returns_created_row_from_same_session(self, sqlite_jobs_repo):
        with sqlite_jobs_repo.session() as session:
            row = sqlite_jobs_repo.insert_job(
                domain="chatbooks",
                queue="default",
                job_type="export",
                payload_json='{"created": true}',
                owner_user_id="55",
                project_id=9,
                batch_group="batch-a",
                idempotency_key="idem-1",
                priority=3,
                max_retries=7,
                available_at=datetime(2026, 3, 18, 9, 30, tzinfo=timezone.utc),
                request_id="req-55",
                trace_id="trace-55",
                created_at=datetime(2026, 3, 17, 12, 30, tzinfo=timezone.utc),
                session=session,
            )

            counted = sqlite_jobs_repo.count_active_jobs_for_user("55", session=session)

        assert row["owner_user_id"] == "55"
        assert row["status"] == "queued"
        assert row["queue"] == "default"
        assert row["job_type"] == "export"
        assert row["batch_group"] == "batch-a"
        assert row["idempotency_key"] == "idem-1"
        assert int(row["priority"]) == 3
        assert int(row["max_retries"]) == 7
        assert counted == 1

    def test_session_uses_injected_connection_pool(self, tmp_path):
        from tldw_Server_API.app.core.DB_Management.Jobs_Repository import JobsRepository

        db_path = tmp_path / "jobs_repo_pool.db"
        ensure_jobs_tables(db_path)
        pool = _FakeConnectionPool(db_path)
        repo = JobsRepository.for_sqlite(db_path, connection_pool=pool)

        try:
            with repo.session() as session:
                row = repo.insert_job(
                    domain="chatbooks",
                    queue="default",
                    job_type="export",
                    payload_json='{"pooled": true}',
                    owner_user_id="77",
                    project_id=None,
                    batch_group=None,
                    idempotency_key=None,
                    priority=4,
                    max_retries=2,
                    available_at=None,
                    request_id="req-pool",
                    trace_id="trace-pool",
                    created_at=datetime(2026, 3, 17, 13, 0, tzinfo=timezone.utc),
                    session=session,
                )
                counted = repo.count_active_jobs_for_user("77", session=session)
        finally:
            repo.close_pool()

        assert row["owner_user_id"] == "77"
        assert counted == 1
        assert pool.acquired == 1
        assert pool.released == 1
        assert pool.closed == 1
