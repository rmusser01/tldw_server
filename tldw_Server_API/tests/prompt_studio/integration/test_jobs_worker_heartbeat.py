import asyncio
from datetime import datetime

import pytest

from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerSDK
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.services import jobs_worker


pytestmark = pytest.mark.integration


def _parse_dt(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


@pytest.mark.asyncio
async def test_prompt_studio_heartbeat_renew_and_reclaim(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs_ps_heartbeat.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)

    jm.create_job(domain="prompt_studio", queue="default", job_type="optimization", payload={}, owner_user_id="u")
    acq = jm.acquire_next_job(domain="prompt_studio", queue="default", lease_seconds=5, worker_id="w1", owner_user_id="u")
    assert acq is not None
    job_id = int(acq["id"])

    monkeypatch.setenv("PROMPT_STUDIO_JOBS_LEASE_SECONDS", "5")
    monkeypatch.setenv("PROMPT_STUDIO_JOBS_RENEW_JITTER_SECONDS", "0")
    monkeypatch.setenv("TLDW_PS_HEARTBEAT_SECONDS", "2")

    cfg = jobs_worker._build_worker_config(worker_id="w1", queue="default")
    sdk = WorkerSDK(jm, cfg)

    renew_calls = []
    orig_renew = jm.renew_job_lease

    def renew_wrapper(**kwargs):
        renew_calls.append(kwargs)
        return orig_renew(**kwargs)

    monkeypatch.setattr(jm, "renew_job_lease", lambda **kwargs: renew_wrapper(**kwargs))

    _orig_sleep = asyncio.sleep

    async def fast_sleep(_):
        await _orig_sleep(0)

    sdk._sleep = fast_sleep

    before = jm.get_job(job_id).get("leased_until")
    task = asyncio.create_task(sdk._auto_renew(acq))
    for _ in range(20):
        if renew_calls:
            break
        await _orig_sleep(0)
    sdk.stop()
    try:
        await asyncio.wait_for(task, timeout=1)
    except asyncio.TimeoutError:
        task.cancel()
        raise

    after = jm.get_job(job_id).get("leased_until")
    assert renew_calls
    before_dt = _parse_dt(before)
    after_dt = _parse_dt(after)
    assert before_dt is not None and after_dt is not None
    assert after_dt >= before_dt

    reacquire = jm.acquire_next_job(domain="prompt_studio", queue="default", lease_seconds=5, worker_id="w2", owner_user_id="u")
    assert reacquire is None

    conn = jm._connect()
    try:
        conn.execute(
            "UPDATE jobs SET leased_until = ? WHERE id = ?",
            ("2000-01-01 00:00:00", job_id),
        )
        conn.commit()
    finally:
        conn.close()

    reclaimed = jm.acquire_next_job(domain="prompt_studio", queue="default", lease_seconds=5, worker_id="w2", owner_user_id="u")
    assert reclaimed is not None


@pytest.mark.asyncio
async def test_prompt_studio_heartbeat_high_latency_renewal(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs_ps_latency.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)

    jm.create_job(domain="prompt_studio", queue="default", job_type="optimization", payload={}, owner_user_id="u")
    acq = jm.acquire_next_job(domain="prompt_studio", queue="default", lease_seconds=4, worker_id="w1", owner_user_id="u")
    assert acq is not None
    job_id = int(acq["id"])

    monkeypatch.setenv("PROMPT_STUDIO_JOBS_LEASE_SECONDS", "4")
    monkeypatch.setenv("PROMPT_STUDIO_JOBS_RENEW_JITTER_SECONDS", "0")
    monkeypatch.setenv("PROMPT_STUDIO_JOBS_RENEW_THRESHOLD_SECONDS", "1")
    monkeypatch.setenv("JOBS_WORKER_MAX_ITERATIONS", "1")

    cfg = jobs_worker._build_worker_config(worker_id="w1", queue="default")
    sdk = WorkerSDK(jm, cfg)

    expired_snapshot = {}
    _orig_sleep = asyncio.sleep

    async def delayed_sleep(_):
        conn = jm._connect()
        try:
            conn.execute(
                "UPDATE jobs SET leased_until = ? WHERE id = ?",
                ("2000-01-01 00:00:00", job_id),
            )
            conn.commit()
        finally:
            conn.close()
        expired_snapshot["leased_until"] = jm.get_job(job_id).get("leased_until")
        await _orig_sleep(0)

    sdk._sleep = delayed_sleep

    task = asyncio.create_task(sdk._auto_renew(acq))
    try:
        await asyncio.wait_for(task, timeout=1)
    except asyncio.TimeoutError:
        task.cancel()
        raise

    assert "leased_until" in expired_snapshot
    expired_dt = _parse_dt(expired_snapshot["leased_until"])
    renewed_dt = _parse_dt(jm.get_job(job_id).get("leased_until"))
    assert expired_dt is not None and renewed_dt is not None
    assert renewed_dt > expired_dt

    reacquire = jm.acquire_next_job(domain="prompt_studio", queue="default", lease_seconds=4, worker_id="w2", owner_user_id="u")
    assert reacquire is None
