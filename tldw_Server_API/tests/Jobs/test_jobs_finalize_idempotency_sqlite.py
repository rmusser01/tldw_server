import os
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
