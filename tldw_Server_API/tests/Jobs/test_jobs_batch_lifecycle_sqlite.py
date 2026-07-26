import os

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables


def _env(monkeypatch, tmp_path):


    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("JOBS_DB_PATH", os.path.join(os.getcwd(), "Databases", "jobs.db"))
    # Enforce lease ack to ensure worker_id/lease_id are honored
    monkeypatch.setenv("JOBS_ENFORCE_LEASE_ACK", "true")


def test_batch_renew_complete_fail_sqlite(monkeypatch, tmp_path):


    _env(monkeypatch, tmp_path)
    ensure_jobs_tables(tmp_path / "jobs.db")
    jm = JobManager()

    # Seed multiple jobs and acquire them
    ids = []
    for i in range(3):
        j = jm.create_job(domain="d", queue="default", job_type="t", payload={"i": i}, owner_user_id="u")
        ids.append(int(j["id"]))

    acquired = []
    for jid in ids:
        acq = jm.acquire_next_job(domain="d", queue="default", lease_seconds=1, worker_id="w1")
        assert acq and int(acq["id"]) == jid
        acquired.append(acq)

    # Batch renew
    items = [{"job_id": int(a["id"]), "worker_id": a.get("worker_id") or "w1", "lease_id": a.get("lease_id"), "seconds": 2} for a in acquired]
    n = jm.batch_renew_leases(items)
    assert n >= 1

    # Complete two, fail one
    complete_items = [
        {"job_id": int(acquired[0]["id"]), "worker_id": "w1", "lease_id": acquired[0].get("lease_id"), "completion_token": "tok-1", "result": {"ok": 1}},
        {"job_id": int(acquired[1]["id"]), "worker_id": "w1", "lease_id": acquired[1].get("lease_id"), "completion_token": "tok-2", "result": {"ok": 1}},
    ]
    done = jm.batch_complete_jobs(complete_items)
    assert done == 2

    fail_items = [
        {"job_id": int(acquired[2]["id"]), "worker_id": "w1", "lease_id": acquired[2].get("lease_id"), "completion_token": "tok-3", "error": "boom", "error_code": "E"},
    ]
    failed = jm.batch_fail_jobs(fail_items)
    assert failed == 1

    for job_id, expected_status in zip(ids, ("completed", "completed", "failed")):
        terminal = jm.get_job(job_id)
        assert terminal is not None
        assert terminal["status"] == expected_status
        assert terminal["leased_until"] is None
        assert terminal["worker_id"] is None
        assert terminal["lease_id"] is None

    # Idempotency: re-complete with same token should affect 0
    again = jm.batch_complete_jobs(complete_items)
    assert again == 0
