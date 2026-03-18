"""
Tests for fair-share scheduler integration with JobManager.

Verifies that:
- create_job enforces per-user concurrency limits via FairShareScheduler
- create_job adjusts priority using fair-share calculation
- Jobs are allowed when under the limit
- Jobs are blocked when at or over the limit
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tldw_Server_API.app.core.DB_Management.Jobs_Repository import JobsRepository, JobsSession
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.app.core.exceptions import BadRequestError


@pytest.fixture
def job_manager(tmp_path, monkeypatch):
    """Create a JobManager backed by a temporary SQLite database."""
    monkeypatch.delenv("JOBS_DISABLE_LEASE_ENFORCEMENT", raising=False)
    db_path = tmp_path / "jobs_fs.db"
    ensure_jobs_tables(db_path)
    return JobManager(db_path)


class TestFairShareAdmissionControl:
    """Verify that create_job blocks users who exceed the concurrency limit."""

    def test_allows_job_under_limit(self, job_manager, monkeypatch):
        monkeypatch.setenv("JOBS_MAX_PER_USER", "3")
        # Reset the cached scheduler so it picks up new env
        import tldw_Server_API.app.core.Jobs.manager as mgr_mod
        mgr_mod._fair_share = None

        job = job_manager.create_job(
            domain="chatbooks",
            queue="default",
            job_type="export",
            payload={"test": True},
            owner_user_id="42",
        )
        assert job is not None
        assert job.get("status") in ("queued", None) or "uuid" in job

    def test_blocks_when_at_limit(self, job_manager, monkeypatch):
        monkeypatch.setenv("JOBS_MAX_PER_USER", "2")
        import tldw_Server_API.app.core.Jobs.manager as mgr_mod
        mgr_mod._fair_share = None

        # Create 2 jobs (at the limit)
        job_manager.create_job(
            domain="chatbooks", queue="default", job_type="export",
            payload={}, owner_user_id="42",
        )
        job_manager.create_job(
            domain="chatbooks", queue="default", job_type="export",
            payload={}, owner_user_id="42",
        )

        # Third job should be blocked
        with pytest.raises(ValueError, match="maximum concurrent job limit"):
            job_manager.create_job(
                domain="chatbooks", queue="default", job_type="export",
                payload={}, owner_user_id="42",
            )

    def test_different_users_independent(self, job_manager, monkeypatch):
        monkeypatch.setenv("JOBS_MAX_PER_USER", "1")
        import tldw_Server_API.app.core.Jobs.manager as mgr_mod
        mgr_mod._fair_share = None

        # User 1 creates a job
        job_manager.create_job(
            domain="chatbooks", queue="default", job_type="export",
            payload={}, owner_user_id="1",
        )

        # User 2 should still be allowed
        job2 = job_manager.create_job(
            domain="chatbooks", queue="default", job_type="export",
            payload={}, owner_user_id="2",
        )
        assert job2 is not None

    def test_no_owner_skips_fair_share(self, job_manager, monkeypatch):
        """Jobs without an owner_user_id bypass fair-share checks."""
        monkeypatch.setenv("JOBS_MAX_PER_USER", "1")
        import tldw_Server_API.app.core.Jobs.manager as mgr_mod
        mgr_mod._fair_share = None

        # Should not raise even though limit is 1
        job_manager.create_job(
            domain="chatbooks", queue="default", job_type="export",
            payload={}, owner_user_id=None,
        )
        job_manager.create_job(
            domain="chatbooks", queue="default", job_type="export",
            payload={}, owner_user_id=None,
        )


class TestFairSharePriorityAdjustment:
    """Verify that priority is adjusted upward based on fair-share calculation."""

    def test_priority_boosted_for_low_active_count(self, job_manager, monkeypatch):
        monkeypatch.setenv("JOBS_MAX_PER_USER", "10")
        import tldw_Server_API.app.core.Jobs.manager as mgr_mod
        mgr_mod._fair_share = None

        job = job_manager.create_job(
            domain="chatbooks", queue="default", job_type="export",
            payload={}, owner_user_id="42", priority=5,
        )
        stored = job_manager.get_job(int(job["id"]))
        assert stored is not None
        assert int(stored["priority"]) < 5

    def test_fails_closed_when_fair_share_check_errors(self, job_manager, monkeypatch):
        monkeypatch.setenv("JOBS_MAX_PER_USER", "10")
        import tldw_Server_API.app.core.Jobs.manager as mgr_mod
        mgr_mod._fair_share = None

        with patch.object(JobManager, "_count_active_jobs_for_user", side_effect=RuntimeError("boom")), \
             patch.object(mgr_mod.logger, "warning") as mock_warning, \
             pytest.raises(BadRequestError, match="Unable to evaluate fair-share policy; please retry"):
            job_manager.create_job(
                domain="chatbooks",
                queue="default",
                job_type="export",
                payload={},
                owner_user_id="42",
                priority=5,
            )

        mock_warning.assert_called_once()


class TestCountActiveJobs:
    """Verify the _count_active_jobs_for_user helper."""

    def test_counts_queued_jobs(self, job_manager):
        job_manager.create_job(
            domain="chatbooks", queue="default", job_type="export",
            payload={}, owner_user_id="99",
        )
        count = job_manager._count_active_jobs_for_user("99")
        assert count == 1

    def test_does_not_count_completed_jobs(self, job_manager):
        job = job_manager.create_job(
            domain="chatbooks", queue="default", job_type="export",
            payload={}, owner_user_id="99",
        )
        # Acquire and complete the job
        acq = job_manager.acquire_next_job(
            domain="chatbooks", queue="default", lease_seconds=60, worker_id="w1",
        )
        if acq:
            job_manager.complete_job(
                int(acq["id"]),
                result={"ok": True},
                worker_id="w1",
                lease_id=str(acq.get("lease_id")),
            )
        count = job_manager._count_active_jobs_for_user("99")
        assert count == 0

    def test_counts_zero_for_unknown_user(self, job_manager):
        count = job_manager._count_active_jobs_for_user("nonexistent")
        assert count == 0


class TrackingJobsRepository(JobsRepository):
    def __init__(
        self,
        db_path: Path | None = None,
        *,
        backend: str = "sqlite",
        db_url: str | None = None,
        events: list[str] | None = None,
    ) -> None:
        super().__init__(
            backend=backend,
            db_path=db_path,
            db_url=db_url,
        )
        self.count_sessions: list[JobsSession | None] = []
        self.insert_sessions: list[JobsSession | None] = []
        self.events = events if events is not None else []

    def count_active_jobs_for_user(
        self,
        user_id: str,
        *,
        session: JobsSession | None = None,
    ) -> int:
        self.count_sessions.append(session)
        self.events.append("count_active_jobs")
        if session is not None and (
            session.backend == "postgres" or getattr(session.conn, "is_fake_sqlite", False)
        ):
            return 0
        return super().count_active_jobs_for_user(user_id, session=session)

    def insert_job(
        self,
        *,
        domain: str,
        queue: str,
        job_type: str,
        payload_json: str,
        owner_user_id: str | None,
        project_id: int | None,
        batch_group: str | None,
        idempotency_key: str | None,
        priority: int,
        max_retries: int,
        available_at: Any,
        request_id: str | None,
        trace_id: str | None,
        created_at: Any,
        session: JobsSession | None = None,
    ) -> dict[str, Any]:
        self.insert_sessions.append(session)
        self.events.append("insert_job")
        if session is not None and (
            session.backend == "postgres" or getattr(session.conn, "is_fake_sqlite", False)
        ):
            return {
                "id": 1,
                "uuid": "job-tracked-1",
                "domain": domain,
                "queue": queue,
                "job_type": job_type,
                "owner_user_id": owner_user_id,
                "project_id": project_id,
                "batch_group": batch_group,
                "idempotency_key": idempotency_key,
                "payload": payload_json,
                "status": "queued",
                "priority": priority,
                "max_retries": max_retries,
                "retry_count": 0,
                "available_at": available_at,
                "created_at": created_at,
                "updated_at": created_at,
                "request_id": request_id,
                "trace_id": trace_id,
                "acquired_at": None,
            }
        return super().insert_job(
            domain=domain,
            queue=queue,
            job_type=job_type,
            payload_json=payload_json,
            owner_user_id=owner_user_id,
            project_id=project_id,
            batch_group=batch_group,
            idempotency_key=idempotency_key,
            priority=priority,
            max_retries=max_retries,
            available_at=available_at,
            request_id=request_id,
            trace_id=trace_id,
            created_at=created_at,
            session=session,
        )


class _FakePostgresConnection:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def __enter__(self) -> "_FakePostgresConnection":
        self._events.append("conn_enter")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._events.append("conn_exit")
        return False

    def commit(self) -> None:
        self._events.append("conn_commit")

    def rollback(self) -> None:
        self._events.append("conn_rollback")

    def close(self) -> None:
        self._events.append("conn_close")


class _FakePostgresCursor:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def __enter__(self) -> "_FakePostgresCursor":
        self._events.append("pg_cursor_enter")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._events.append("pg_cursor_exit")
        return False

    def execute(self, query: str, params: tuple[Any, ...] | None = None) -> None:
        self._events.append("pg_execute")

    def fetchone(self) -> dict[str, int]:
        return {"c": 0}


class _FakePostgresIdempotencyCursor:
    def __init__(self, existing_row: dict[str, Any]) -> None:
        self._existing_row = existing_row
        self._last_query = ""

    def __enter__(self) -> "_FakePostgresIdempotencyCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query: str, params: tuple[Any, ...] | None = None) -> None:
        self._last_query = query

    def fetchone(self) -> dict[str, Any] | None:
        if "SELECT * FROM jobs WHERE domain = %s" in self._last_query:
            return self._existing_row
        return None


class _FakeSqliteConnection:
    is_fake_sqlite = True

    def __init__(self, events: list[str]) -> None:
        self._events = events

    def __enter__(self) -> "_FakeSqliteConnection":
        self._events.append("conn_enter")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._events.append("conn_exit")
        return False

    def execute(self, query: str, params: tuple[Any, ...] | None = None) -> None:
        self._events.append("sqlite_execute")

    def commit(self) -> None:
        self._events.append("conn_commit")

    def rollback(self) -> None:
        self._events.append("conn_rollback")

    def close(self) -> None:
        self._events.append("conn_close")


class TestFairShareRepositoryIntegration:
    def test_idempotent_retry_returns_existing_job_even_when_at_limit(self, job_manager, monkeypatch):
        monkeypatch.setenv("JOBS_MAX_PER_USER", "1")
        import tldw_Server_API.app.core.Jobs.manager as mgr_mod
        mgr_mod._fair_share = None

        first = job_manager.create_job(
            domain="chatbooks",
            queue="default",
            job_type="export",
            payload={"attempt": 1},
            owner_user_id="42",
            idempotency_key="idem-limit-replay",
        )

        replay = job_manager.create_job(
            domain="chatbooks",
            queue="default",
            job_type="export",
            payload={"attempt": 2},
            owner_user_id="42",
            idempotency_key="idem-limit-replay",
        )

        assert replay["id"] == first["id"]

    def test_create_job_reuses_repository_session_for_fair_share(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JOBS_MAX_PER_USER", "10")
        import tldw_Server_API.app.core.Jobs.manager as mgr_mod
        mgr_mod._fair_share = None

        db_path = tmp_path / "jobs_repo_integration.db"
        ensure_jobs_tables(db_path)
        repo = TrackingJobsRepository(db_path)
        job_manager = JobManager(db_path, jobs_repository=repo)

        job = job_manager.create_job(
            domain="chatbooks",
            queue="default",
            job_type="export",
            payload={"test": True},
            owner_user_id="42",
            priority=5,
        )

        assert job is not None
        assert repo.count_sessions
        assert repo.insert_sessions
        assert repo.count_sessions[0] is repo.insert_sessions[0]
        stored = job_manager.get_job(int(job["id"]))
        assert stored is not None
        assert stored["payload"] == {"test": True}

    def test_create_job_uses_repository_session_context(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JOBS_MAX_PER_USER", "10")
        import tldw_Server_API.app.core.Jobs.manager as mgr_mod
        mgr_mod._fair_share = None

        db_path = tmp_path / "jobs_repo_session_context.db"
        ensure_jobs_tables(db_path)
        repo = TrackingJobsRepository(db_path)
        job_manager = JobManager(db_path, jobs_repository=repo)

        with patch.object(repo, "session", wraps=repo.session) as session_spy:
            job = job_manager.create_job(
                domain="chatbooks",
                queue="default",
                job_type="export",
                payload={"test": True},
                owner_user_id="42",
                priority=5,
            )

        assert job is not None
        session_spy.assert_called_once_with()

    def test_postgres_fair_share_count_runs_after_pg_cursor_setup(self, monkeypatch):
        monkeypatch.setenv("JOBS_MAX_PER_USER", "10")
        monkeypatch.setenv("JOBS_PG_SKIP_SCHEMA_INIT", "1")
        import tldw_Server_API.app.core.Jobs.manager as mgr_mod
        mgr_mod._fair_share = None

        events: list[str] = []
        repo = TrackingJobsRepository(
            backend="postgres",
            db_url="postgres://jobs.test/review_cleanup",
            events=events,
        )
        job_manager = JobManager(
            backend="postgres",
            db_url="postgres://jobs.test/review_cleanup",
            jobs_repository=repo,
        )
        fake_conn = _FakePostgresConnection(events)
        monkeypatch.setattr(repo, "_connect", lambda: fake_conn)
        monkeypatch.setattr(job_manager, "_pg_cursor", lambda conn: _FakePostgresCursor(events))

        with patch.object(mgr_mod, "increment_created"), \
             patch.object(mgr_mod, "emit_job_event"), \
             patch.object(mgr_mod, "submit_job_audit_event"):
            job = job_manager.create_job(
                domain="chatbooks",
                queue="default",
                job_type="export",
                payload={"test": True},
                owner_user_id="42",
                priority=5,
            )

        assert job is not None
        assert repo.count_sessions
        assert repo.insert_sessions
        assert events.index("pg_cursor_enter") < events.index("count_active_jobs")
        assert repo.count_sessions[0] is repo.insert_sessions[0]

    def test_postgres_idempotent_retry_returns_existing_job_when_fair_share_blocks(
        self,
        monkeypatch,
    ):
        monkeypatch.setenv("JOBS_MAX_PER_USER", "1")
        monkeypatch.setenv("JOBS_PG_SKIP_SCHEMA_INIT", "1")
        import tldw_Server_API.app.core.Jobs.manager as mgr_mod
        mgr_mod._fair_share = None

        repo = TrackingJobsRepository(
            backend="postgres",
            db_url="postgres://jobs.test/review_cleanup",
        )
        job_manager = JobManager(
            backend="postgres",
            db_url="postgres://jobs.test/review_cleanup",
            jobs_repository=repo,
        )
        fake_conn = _FakePostgresConnection([])
        existing_row = {
            "id": 321,
            "uuid": "job-existing-321",
            "domain": "chatbooks",
            "queue": "default",
            "job_type": "export",
            "owner_user_id": "42",
            "request_id": None,
            "trace_id": None,
            "retry_count": 0,
        }
        monkeypatch.setattr(repo, "_connect", lambda: fake_conn)
        monkeypatch.setattr(
            job_manager,
            "_pg_cursor",
            lambda conn: _FakePostgresIdempotencyCursor(existing_row),
        )

        with patch.object(
            job_manager,
            "_apply_fair_share_submission_policy",
            side_effect=BadRequestError("User 42 has reached the maximum concurrent job limit (1)"),
        ):
            replay = job_manager.create_job(
                domain="chatbooks",
                queue="default",
                job_type="export",
                payload={"attempt": 2},
                owner_user_id="42",
                idempotency_key="idem-pg-limit-replay",
            )

        assert replay["id"] == existing_row["id"]

    def test_sqlite_fair_share_count_runs_inside_write_transaction(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JOBS_MAX_PER_USER", "10")
        import tldw_Server_API.app.core.Jobs.manager as mgr_mod
        mgr_mod._fair_share = None

        db_path = tmp_path / "jobs_repo_sqlite_ordering.db"
        ensure_jobs_tables(db_path)
        events: list[str] = []
        repo = TrackingJobsRepository(db_path, events=events)
        job_manager = JobManager(db_path, jobs_repository=repo)
        fake_conn = _FakeSqliteConnection(events)
        monkeypatch.setattr(repo, "_connect", lambda: fake_conn)
        monkeypatch.setattr(job_manager, "_update_gauges", lambda **kwargs: None)

        with patch.object(mgr_mod, "increment_created"), \
             patch.object(mgr_mod, "emit_job_event"), \
             patch.object(mgr_mod, "submit_job_audit_event"):
            job = job_manager.create_job(
                domain="chatbooks",
                queue="default",
                job_type="export",
                payload={"test": True},
                owner_user_id="42",
                priority=5,
            )

        assert job is not None
        assert events.index("conn_enter") < events.index("count_active_jobs")
        assert events.index("count_active_jobs") < events.index("insert_job")
        assert repo.count_sessions[0] is repo.insert_sessions[0]

    def test_constructor_rejects_backend_mismatch(self, tmp_path):
        db_path = tmp_path / "jobs_repo_backend_mismatch.db"
        ensure_jobs_tables(db_path)
        repo = TrackingJobsRepository(
            backend="postgres",
            db_url="postgres://jobs.test/review_cleanup",
        )

        with pytest.raises(ValueError, match="does not match manager backend"):
            JobManager(db_path, jobs_repository=repo)

    def test_constructor_rejects_missing_repository_methods(self, tmp_path):
        db_path = tmp_path / "jobs_repo_invalid_repo.db"
        ensure_jobs_tables(db_path)

        class InvalidJobsRepository:
            backend = "sqlite"

        with pytest.raises(TypeError, match="jobs_repository must expose"):
            JobManager(db_path, jobs_repository=InvalidJobsRepository())
