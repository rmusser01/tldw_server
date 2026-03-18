"""
Tests for fair-share scheduler integration with JobManager.

Verifies that:
- create_job enforces per-user concurrency limits via FairShareScheduler
- create_job adjusts priority using fair-share calculation
- Jobs are allowed when under the limit
- Jobs are blocked when at or over the limit
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from tldw_Server_API.app.core.DB_Management.Jobs_Repository import JobsRepository
from tldw_Server_API.app.core.Jobs.manager import JobManager, _get_fair_share
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables


@pytest.fixture()
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

    def test_warns_when_fair_share_check_is_skipped(self, job_manager, monkeypatch):
        monkeypatch.setenv("JOBS_MAX_PER_USER", "10")
        import tldw_Server_API.app.core.Jobs.manager as mgr_mod
        mgr_mod._fair_share = None

        with patch.object(JobManager, "_count_active_jobs_for_user", side_effect=RuntimeError("boom")), \
             patch.object(mgr_mod.logger, "warning") as mock_warning:
            job = job_manager.create_job(
                domain="chatbooks",
                queue="default",
                job_type="export",
                payload={},
                owner_user_id="42",
                priority=5,
            )

        assert job is not None
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
    def __init__(self, db_path):
        super().__init__(backend="sqlite", db_path=db_path)
        self.count_sessions = []
        self.insert_sessions = []

    def count_active_jobs_for_user(self, user_id: str, *, session=None) -> int:
        self.count_sessions.append(session)
        return super().count_active_jobs_for_user(user_id, session=session)

    def insert_job(self, *, session=None, **kwargs):
        self.insert_sessions.append(session)
        return super().insert_job(session=session, **kwargs)


class TestFairShareRepositoryIntegration:
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
