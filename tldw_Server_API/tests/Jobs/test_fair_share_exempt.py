import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager

pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_isolated_fixtures"]
USE_SHARED_JOBS_POSTGRES = True


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.pg_jobs)])
def jobs(request, tmp_path, monkeypatch) -> JobManager:
    if request.param == "postgres":
        return JobManager(backend="postgres", db_url=request.getfixturevalue("jobs_pg_dsn"))
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    return JobManager(db_path=tmp_path / "jobs.db")


@pytest.mark.unit
def test_fair_share_exempt_skips_the_limit_but_keeps_the_priority_adjustment(tmp_path, monkeypatch):
    monkeypatch.setenv("JOBS_MAX_PER_USER", "1")
    jm = JobManager(tmp_path / "jobs_fair_share_exempt.db")
    first = jm.create_job(domain="audio", queue="default", job_type="audio_convert", payload={}, owner_user_id="7")

    with pytest.raises(ValueError):
        jm.create_job(domain="audio", queue="default", job_type="audio_convert", payload={}, owner_user_id="7")

    exempt = jm.create_job(
        domain="audio", queue="default", job_type="audio_transcribe", payload={},
        owner_user_id="7", priority=5, fair_share_exempt=True,
    )
    # Fair-share lowered the first job's priority below the requested 5; an exempt create gets the same adjustment.
    assert int(first["priority"]) < 5
    assert int(exempt["priority"]) == int(first["priority"])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("quota", "refusal"),
    [("JOBS_QUOTA_SUBMITS_PER_MIN_AUDIO", "submits per minute"), ("JOBS_QUOTA_MAX_QUEUED_AUDIO", "max queued")],
)
def test_quota_exempt_skips_the_submit_quotas(jobs, monkeypatch, quota, refusal):
    monkeypatch.setenv(quota, "1")
    jobs.create_job(domain="audio", queue="default", job_type="audio_convert", payload={}, owner_user_id="7")

    with pytest.raises(ValueError, match=refusal):
        jobs.create_job(domain="audio", queue="default", job_type="audio_convert", payload={}, owner_user_id="7")

    exempt = jobs.create_job(
        domain="audio", queue="default", job_type="audio_transcribe", payload={}, owner_user_id="7", quota_exempt=True,
    )
    assert exempt["status"] == "queued"
