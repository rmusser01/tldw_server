import os
import sqlite3

import pytest

from tldw_Server_API.app.core.Jobs import manager as manager_module
from tldw_Server_API.app.core.Jobs import metrics as metrics_module
from tldw_Server_API.app.core.Jobs.manager import JobManager


def _set_env(monkeypatch, tmp_path):


    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    # SQLite DB path under test dir
    db_path = os.path.join(os.getcwd(), "Databases", "jobs.db")
    monkeypatch.setenv("JOBS_DB_PATH", db_path)
    monkeypatch.setenv("JOBS_ENFORCE_LEASE_ACK", "true")
    monkeypatch.setenv("JOBS_REQUIRE_COMPLETION_TOKEN", "true")


def test_complete_idempotent_with_token(monkeypatch, tmp_path):

    _set_env(monkeypatch, tmp_path)
    jm = JobManager()
    j = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w1")
    assert acq and acq["id"] == j["id"]
    token = str(acq.get("lease_id"))
    ok1 = jm.complete_job(
        int(j["id"]), result={"ok": True}, worker_id="w1", lease_id=str(acq.get("lease_id")), completion_token=token
    )
    assert ok1 is True
    # Repeat with same token: should be idempotent success
    ok2 = jm.complete_job(
        int(j["id"]), result={"ok": True}, worker_id="w1", lease_id=str(acq.get("lease_id")), completion_token=token
    )
    assert ok2 is True
    # Different token: should not re-complete (status terminal); returns False
    ok3 = jm.complete_job(
        int(j["id"]),
        result={"ok": True},
        worker_id="w1",
        lease_id=str(acq.get("lease_id")),
        completion_token=token + "-x",
    )
    assert ok3 is False


def test_fail_idempotent_with_token(monkeypatch, tmp_path):

    _set_env(monkeypatch, tmp_path)
    jm = JobManager()
    j = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w2")
    assert acq and acq["id"] == j["id"]
    token = str(acq.get("lease_id"))
    ok1 = jm.fail_job(
        int(j["id"]),
        error="boom",
        retryable=False,
        worker_id="w2",
        lease_id=str(acq.get("lease_id")),
        completion_token=token,
    )
    assert ok1 is True
    # Repeat with same token: idempotent success
    ok2 = jm.fail_job(
        int(j["id"]),
        error="boom",
        retryable=False,
        worker_id="w2",
        lease_id=str(acq.get("lease_id")),
        completion_token=token,
    )
    assert ok2 is True
    # Different token: job is already terminal
    ok3 = jm.fail_job(
        int(j["id"]),
        error="boom",
        retryable=False,
        worker_id="w2",
        lease_id=str(acq.get("lease_id")),
        completion_token=token + "-x",
    )
    assert ok3 is False


def test_worker_terminalizer_is_exact_idempotent_cas(monkeypatch, tmp_path):
    _set_env(monkeypatch, tmp_path)
    jm = JobManager()
    job = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key="terminal-cas",
    )
    acquired = jm.acquire_next_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        lease_seconds=30,
        worker_id="slides-worker",
    )
    assert acquired is not None
    lease_id = str(acquired["lease_id"])
    arguments = {
        "job_id": int(job["id"]),
        "job_uuid": str(job["uuid"]),
        "owner_user_id": "owner-1",
        "domain": "slides",
        "queue": "default",
        "job_type": "presentation.generate",
        "worker_id": "slides-worker",
        "lease_id": lease_id,
        "completion_token": lease_id,
        "status": "failed",
        "error_code": "slides_render_failed",
        "error_message": "stable safe detail",
    }

    assert jm.terminalize_job_from_worker(**arguments) == "APPLIED"
    assert jm.terminalize_job_from_worker(**arguments) == "IDEMPOTENT"
    assert (
        jm.terminalize_job_from_worker(
            **{**arguments, "error_message": "different detail"},
        )
        == "CONFLICT"
    )
    assert (
        jm.terminalize_job_from_worker(
            **{**arguments, "completion_token": lease_id + "-different"},
        )
        == "CONFLICT"
    )


@pytest.mark.parametrize(
    ("status", "event_type"),
    [("failed", "job.failed"), ("cancelled", "job.cancelled")],
)
def test_worker_terminalizer_applied_bookkeeping_happens_exactly_once(
    monkeypatch,
    tmp_path,
    status,
    event_type,
):
    _set_env(monkeypatch, tmp_path)
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    jm = JobManager()
    job = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key=f"terminal-bookkeeping-{status}",
    )
    acquired = jm.acquire_next_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        lease_seconds=30,
        worker_id="slides-worker",
    )
    assert acquired is not None
    lease_id = str(acquired["lease_id"])
    with sqlite3.connect(os.environ["JOBS_DB_PATH"]) as conn:
        conn.execute(
            """
            UPDATE job_counters SET processing_count=1
            WHERE domain='slides' AND queue='default' AND job_type='presentation.generate'
            """
        )
        conn.commit()
    arguments = {
        "job_id": int(job["id"]),
        "job_uuid": str(job["uuid"]),
        "owner_user_id": "owner-1",
        "domain": "slides",
        "queue": "default",
        "job_type": "presentation.generate",
        "worker_id": "slides-worker",
        "lease_id": lease_id,
        "completion_token": lease_id,
        "status": status,
        "error_code": f"slides_render_{status}",
        "error_message": f"{status} safely",
    }

    assert jm.terminalize_job_from_worker(**arguments) == "APPLIED"
    assert jm.terminalize_job_from_worker(**arguments) == "IDEMPOTENT"

    with sqlite3.connect(os.environ["JOBS_DB_PATH"]) as conn:
        processing_count = conn.execute(
            """
            SELECT processing_count FROM job_counters
            WHERE domain='slides' AND queue='default' AND job_type='presentation.generate'
            """
        ).fetchone()[0]
        event_count = conn.execute(
            "SELECT COUNT(*) FROM job_events WHERE job_id=? AND event_type=?",
            (int(job["id"]), event_type),
        ).fetchone()[0]
    assert processing_count == 0
    assert event_count == 1


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_worker_terminalizer_applied_side_effects_are_not_repeated(
    monkeypatch,
    tmp_path,
    status,
):
    _set_env(monkeypatch, tmp_path)
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")
    jm = JobManager()
    job = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key=f"terminal-side-effects-{status}",
    )
    acquired = jm.acquire_next_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        lease_seconds=30,
        worker_id="slides-worker",
    )
    assert acquired is not None
    calls = {
        "gauges": 0,
        "failures": 0,
        "failure_codes": 0,
        "cancelled": 0,
        "events": 0,
        "audits": 0,
    }

    def record(name):
        calls[name] += 1

    monkeypatch.setattr(jm, "_update_gauges", lambda **_kwargs: record("gauges"))
    monkeypatch.setattr(
        manager_module,
        "increment_failures",
        lambda *_args, **_kwargs: record("failures"),
    )
    monkeypatch.setattr(
        metrics_module,
        "increment_failures_by_code",
        lambda *_args, **_kwargs: record("failure_codes"),
    )
    monkeypatch.setattr(
        manager_module,
        "increment_cancelled",
        lambda *_args, **_kwargs: record("cancelled"),
    )
    monkeypatch.setattr(
        manager_module,
        "emit_job_event",
        lambda *_args, **_kwargs: record("events"),
    )
    monkeypatch.setattr(
        manager_module,
        "submit_job_audit_event",
        lambda *_args, **_kwargs: record("audits"),
    )
    lease_id = str(acquired["lease_id"])
    arguments = {
        "job_id": int(job["id"]),
        "job_uuid": str(job["uuid"]),
        "owner_user_id": "owner-1",
        "domain": "slides",
        "queue": "default",
        "job_type": "presentation.generate",
        "worker_id": "slides-worker",
        "lease_id": lease_id,
        "completion_token": lease_id,
        "status": status,
        "error_code": f"slides_render_{status}",
        "error_message": f"{status} safely",
    }

    assert jm.terminalize_job_from_worker(**arguments) == "APPLIED"
    assert jm.terminalize_job_from_worker(**arguments) == "IDEMPOTENT"

    assert calls["gauges"] == 1
    assert calls["events"] == 1
    assert calls["audits"] == 1
    assert calls["failures"] == (1 if status == "failed" else 0)
    assert calls["failure_codes"] == (1 if status == "failed" else 0)
    assert calls["cancelled"] == (1 if status == "cancelled" else 0)


def test_worker_terminalizer_rejects_wrong_uuid_owner_scope_and_lease(monkeypatch, tmp_path):
    _set_env(monkeypatch, tmp_path)
    jm = JobManager()
    job = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key="terminal-correlation",
    )
    acquired = jm.acquire_next_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        lease_seconds=30,
        worker_id="slides-worker",
    )
    assert acquired is not None
    base = {
        "job_id": int(job["id"]),
        "job_uuid": str(job["uuid"]),
        "owner_user_id": "owner-1",
        "domain": "slides",
        "queue": "default",
        "job_type": "presentation.generate",
        "worker_id": "slides-worker",
        "lease_id": str(acquired["lease_id"]),
        "completion_token": str(acquired["lease_id"]),
        "status": "cancelled",
        "error_code": "slides_render_cancelled",
        "error_message": "cancelled safely",
    }

    for changed in (
        {"job_uuid": "00000000-0000-0000-0000-000000000000"},
        {"owner_user_id": "owner-2"},
        {"domain": "other"},
        {"queue": "other"},
        {"job_type": "other"},
        {"worker_id": "other-worker"},
        {"lease_id": "other-lease"},
    ):
        assert jm.terminalize_job_from_worker(**{**base, **changed}) == "CONFLICT"

    assert jm.get_job(int(job["id"]))["status"] == "processing"


def test_worker_terminalizer_rejects_exact_non_slides_job(monkeypatch, tmp_path):
    _set_env(monkeypatch, tmp_path)
    jm = JobManager()
    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="owner-1",
    )
    acquired = jm.acquire_next_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        lease_seconds=30,
        worker_id="generic-worker",
    )
    assert acquired is not None
    lease_id = str(acquired["lease_id"])

    assert (
        jm.terminalize_job_from_worker(
            job_id=int(job["id"]),
            job_uuid=str(job["uuid"]),
            owner_user_id="owner-1",
            domain="chatbooks",
            queue="default",
            job_type="export",
            worker_id="generic-worker",
            lease_id=lease_id,
            completion_token=lease_id,
            status="failed",
            error_code="generic_failure",
            error_message="safe detail",
        )
        == "CONFLICT"
    )
    assert jm.get_job(int(job["id"]))["status"] == "processing"
