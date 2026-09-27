import os

import pytest

pytestmark = pytest.mark.pg_jobs


def _env(monkeypatch):


     # Requires JOBS_DB_URL and psycopg installed
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    # Enforce lease ack
    monkeypatch.setenv("JOBS_ENFORCE_LEASE_ACK", "true")


def test_batch_renew_complete_fail_postgres(monkeypatch):


    if not os.getenv("JOBS_DB_URL", "").startswith("postgres"):
        pytest.skip("JOBS_DB_URL not set to Postgres")
    _env(monkeypatch)
    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg
    ensure_jobs_tables_pg(os.getenv("JOBS_DB_URL"))
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    jm = JobManager(backend="postgres", db_url=os.getenv("JOBS_DB_URL"))

    ids = []
    for i in range(3):
        j = jm.create_job(domain="d", queue="default", job_type="t", payload={"i": i}, owner_user_id="u")
        ids.append(int(j["id"]))

    acquired = []
    for jid in ids:
        acq = jm.acquire_next_job(domain="d", queue="default", lease_seconds=1, worker_id="w1")
        assert acq and int(acq["id"]) == jid
        acquired.append(acq)

    items = [{"job_id": int(a["id"]), "worker_id": a.get("worker_id") or "w1", "lease_id": a.get("lease_id"), "seconds": 2} for a in acquired]
    n = jm.batch_renew_leases(items)
    assert n >= 1

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

    # Idempotency
    again = jm.batch_complete_jobs(complete_items)
    assert again == 0
