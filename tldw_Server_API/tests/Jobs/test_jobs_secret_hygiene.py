import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager


def test_secret_reject_prevents_insert_sqlite(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBS_SECRET_REJECT", "true")
    monkeypatch.delenv("JOBS_SECRET_REDACT", raising=False)
    manager = JobManager(tmp_path / "jobs.db")

    with pytest.raises(ValueError, match="Payload appears to contain secrets"):
        manager.create_job(
            domain="secret-hygiene",
            queue="default",
            job_type="reject",
            payload={"api_key": "do-not-store"},
            owner_user_id="owner-1",
        )

    assert manager.count_jobs(domain="secret-hygiene", owner_user_id="owner-1") == 0


def test_secret_redact_persists_only_redacted_value(tmp_path, monkeypatch):
    monkeypatch.delenv("JOBS_SECRET_REJECT", raising=False)
    monkeypatch.setenv("JOBS_SECRET_REDACT", "true")
    manager = JobManager(tmp_path / "jobs.db")

    created = manager.create_job(
        domain="secret-hygiene",
        queue="default",
        job_type="redact",
        payload={"api_key": "do-not-store"},
        owner_user_id="owner-1",
    )

    persisted = manager.get_job(int(created["id"]))

    assert persisted["payload"] == {"api_key": "***REDACTED***"}


def test_secret_scan_failure_preserves_original_payload(tmp_path, monkeypatch):
    manager = JobManager(tmp_path / "jobs.db")
    payload = {"value": "original"}

    def unavailable_scanner(_payload):
        raise RuntimeError("scanner unavailable")

    monkeypatch.setattr(manager, "_scan_and_redact_secrets", unavailable_scanner)

    created = manager.create_job(
        domain="secret-hygiene",
        queue="default",
        job_type="scanner-failure",
        payload=payload,
        owner_user_id="owner-1",
    )

    persisted = manager.get_job(int(created["id"]))

    assert persisted["payload"] == payload


@pytest.mark.pg_jobs
def test_secret_reject_prevents_insert_postgres(jobs_pg_dsn, monkeypatch):
    monkeypatch.setenv("JOBS_SECRET_REJECT", "true")
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)

    with pytest.raises(ValueError, match="Payload appears to contain secrets"):
        manager.create_job(
            domain="secret-hygiene-pg",
            queue="default",
            job_type="reject",
            payload={"api_key": "do-not-store"},
            owner_user_id="owner-1",
        )

    assert manager.count_jobs(domain="secret-hygiene-pg", owner_user_id="owner-1") == 0
