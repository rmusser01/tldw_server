import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager


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
