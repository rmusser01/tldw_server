import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.operations.postgres import admission


pytestmark = pytest.mark.pg_jobs


def test_counter_failure_rolls_back_to_savepoint_and_commits_job_event(jobs_pg_dsn, monkeypatch):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")

    def fail_counter(cur, *, command, available_at):
        del command, available_at
        cur.execute("SELECT definitely_missing_column FROM job_counters")

    monkeypatch.setattr(admission, "_bump_counters", fail_counter)
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)

    created = manager.create_job(
        domain="admission-fault",
        queue="default",
        job_type="counter",
        payload={},
        owner_user_id="owner-1",
    )

    assert created["status"] == "queued"
    events = manager.list_job_events_after(after_id=0, domain="admission-fault", limit=10)
    assert [event["event_type"] for event in events] == ["job.created"]
