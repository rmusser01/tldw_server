"""Outbox setup failures must restore the caller's actual Jobs authority."""

import pytest

from tldw_Server_API.app.core.Jobs.event_stream import emit_job_event
from tldw_Server_API.app.core.Jobs.manager import JobManager

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("failure_stage", ["construct", "connect"])
def test_outbox_setup_failure_restores_caller_scope(monkeypatch, failure_stage):
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "1")
    monkeypatch.setenv("JOBS_DB_URL", "")

    def fail(_self):
        raise RuntimeError("controlled outbox setup failure")

    if failure_stage == "construct":
        monkeypatch.setattr(JobManager, "__init__", fail)
    else:
        # Isolate the connection failure after construction; real context variables remain.
        monkeypatch.setattr(JobManager, "__init__", lambda _self: None)
        monkeypatch.setattr(JobManager, "_connect", fail)
    with JobManager.scoped_rls_context(is_admin=False, domain_allowlist="media_ingest", owner_user_id="11"):
        before = (
            JobManager._RLS_IS_ADMIN.get(),
            JobManager._RLS_DOMAIN_ALLOWLIST.get(),
            JobManager._RLS_OWNER_USER_ID.get(),
        )
        emit_job_event("job.acquired", job={"id": 1, "domain": "media_ingest"})
        after = (
            JobManager._RLS_IS_ADMIN.get(),
            JobManager._RLS_DOMAIN_ALLOWLIST.get(),
            JobManager._RLS_OWNER_USER_ID.get(),
        )
        assert after == before  # nosec B101 # Temporary outbox authority must not escape a failed setup.


def test_sqlite_outbox_preserves_scope_and_writes_event(monkeypatch, tmp_path):
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "1")
    monkeypatch.setenv("JOBS_DB_URL", "")
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    manager = JobManager()
    with JobManager.scoped_rls_context(is_admin=False, domain_allowlist="media_ingest", owner_user_id="11"):
        job = manager.create_job(
            domain="media_ingest", queue="default", job_type="media_ingest_item", owner_user_id="11", payload={}
        )
        before = (
            JobManager._RLS_IS_ADMIN.get(),
            JobManager._RLS_DOMAIN_ALLOWLIST.get(),
            JobManager._RLS_OWNER_USER_ID.get(),
        )
        emit_job_event("job.acquired", job=job)
        after = (
            JobManager._RLS_IS_ADMIN.get(),
            JobManager._RLS_DOMAIN_ALLOWLIST.get(),
            JobManager._RLS_OWNER_USER_ID.get(),
        )
        assert after == before  # nosec B101 # A successful write also preserves caller authority.
        events = manager.list_job_events_after(after_id=0, job_id=job["id"], event_types=("job.acquired",))
        assert [(row["job_id"], row["owner_user_id"]) for row in events] == [
            (job["id"], "11")
        ]  # nosec B101 # Actual durable event retains the owner.
