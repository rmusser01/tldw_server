"""A job handler run by WorkerSDK sees the job's owner as the LLM usage user (spec §1)."""

import asyncio

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_user_id

pytestmark = pytest.mark.unit


def _sdk(jm: JobManager) -> WorkerSDK:
    """A worker on the test queue."""
    return WorkerSDK(
        jm,
        WorkerConfig(
            domain="writing", queue="default", worker_id="w-llm-user",
            lease_seconds=5, renew_threshold_seconds=1, renew_jitter_seconds=0,
        ),
    )


async def test_each_job_runs_for_its_own_owner_and_none_leaks(tmp_path) -> None:
    """Owners "7", "u" and "9" are seen as 7, no user and 9, in order, and no user remains afterwards."""
    db_path = tmp_path / "jobs_llm_user.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    for owner in ("7", "u", "9"):
        jm.create_job(domain="writing", queue="default", job_type="t", payload={"owner": owner}, owner_user_id=owner)
    sdk = _sdk(jm)
    seen: list[tuple[str, int | None]] = []

    async def handler(job_row):
        """Record the user this job sees."""
        seen.append((str(job_row["payload"]["owner"]), current_llm_user_id()))
        if len(seen) == 3:
            sdk.stop()
        return {"ok": True}

    await asyncio.wait_for(sdk.run(handler=handler, job_type="t"), timeout=5)

    assert seen == [("7", 7), ("u", None), ("9", 9)]
    assert current_llm_user_id() is None


async def test_a_failing_handler_still_releases_the_user(tmp_path) -> None:
    """A handler that raises does not leave its owner set for the next job."""
    db_path = tmp_path / "jobs_llm_user_fail.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    jm.create_job(domain="writing", queue="default", job_type="t", payload={"n": 1}, owner_user_id="7")
    jm.create_job(domain="writing", queue="default", job_type="t", payload={"n": 2}, owner_user_id="u")
    sdk = _sdk(jm)
    seen: list[int | None] = []

    async def handler(job_row):
        """Fail on the owned job, then report on the unowned one."""
        seen.append(current_llm_user_id())
        if len(seen) == 2:
            sdk.stop()
            return {"ok": True}
        raise RuntimeError("handler failed")

    await asyncio.wait_for(sdk.run(handler=handler, job_type="t"), timeout=5)

    assert seen == [7, None]
