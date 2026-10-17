"""Roots with their own loop give their work a user (LLM usage accounting spec §1)."""

import asyncio

import pytest

from tldw_Server_API.app.core.Scheduler.handlers import watchlists as watchlists_handlers
from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_user_id

pytestmark = pytest.mark.unit


async def test_watchlist_output_enrichment_runs_for_the_payload_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """The enrichment handler sees the payload's user, and none remains afterwards."""
    seen: list[int | None] = []

    async def _enrich(**kwargs: object) -> None:
        """Record the user the enrichment sees."""
        seen.append(current_llm_user_id())

    monkeypatch.setattr(watchlists_handlers.output_enrichment_handler, "enrich_output", _enrich)
    result = await watchlists_handlers.watchlists_enrich_output({"output_id": 5, "user_id": "7"})
    assert result == {"output_id": 5, "status": "completed"}
    assert seen == [7]
    assert current_llm_user_id() is None


async def test_the_agent_task_worker_runs_a_job_for_its_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    """One iteration of the agent-task worker loop: the handler sees the job's owner."""
    from tldw_Server_API.app.core.Scheduled_Tasks import automation_executors
    from tldw_Server_API.app.services import agent_task_jobs_worker as worker

    stop = asyncio.Event()
    seen: list[int | None] = []

    class Manager:
        """Hands out one job, then stops the loop when it completes."""

        def acquire_next_job(self, **kwargs: object) -> dict:
            return {"id": 77, "lease_id": "lease", "job_type": "agent_task_run", "owner_user_id": "7", "payload": {}}

        def complete_job(self, *args: object, **kwargs: object) -> None:
            stop.set()

        def fail_job(self, *args: object, **kwargs: object) -> None:
            stop.set()
            raise AssertionError("the job must complete, not fail")

    async def handler(job: dict, **kwargs: object) -> dict:
        """Record the user the job runs as."""
        seen.append(current_llm_user_id())
        return {"status": "completed"}

    monkeypatch.setattr(worker, "jobs_manager_from_env", Manager)
    monkeypatch.setattr(worker, "handle_agent_task_job", handler)
    monkeypatch.setattr(automation_executors, "register_automation_executors", lambda: None)
    await asyncio.wait_for(worker.run_agent_task_jobs_worker(stop), timeout=5)
    assert seen == [7]
    assert current_llm_user_id() is None


async def test_the_audio_jobs_worker_runs_analysis_for_the_job_owner(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """An audio_analyze job: the analysis call, on its thread, sees the job's owner."""
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs_llm_user.db"))
    monkeypatch.setenv("JOBS_POLL_INTERVAL_SECONDS", "0.05")

    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as summarization
    import tldw_Server_API.app.services.audio_jobs_worker as worker
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    async def _unlimited(user_id: int) -> dict:
        return {"daily_minutes": None, "concurrent_streams": None, "concurrent_jobs": None, "max_file_size_mb": None}

    async def _yes(user_id: int) -> tuple[bool, str]:
        return True, ""

    async def _noop(user_id: int) -> None:
        return None

    monkeypatch.setattr(worker, "get_limits_for_user", _unlimited)
    monkeypatch.setattr(worker, "can_start_job", _yes)
    monkeypatch.setattr(worker, "increment_jobs_started", _noop)
    monkeypatch.setattr(worker, "finish_job", _noop)

    seen: list[int | None] = []

    def _analyze(api_name: str, input_data: str, *args: object, **kwargs: object) -> str:
        """Record the user the analysis runs as."""
        seen.append(current_llm_user_id())
        return "analysis"

    monkeypatch.setattr(summarization, "analyze", _analyze)

    jm = JobManager()
    jm.create_job(
        domain="audio", queue="default", job_type="audio_analyze",
        payload={"text": "hello worker", "api_name": "openai", "perform_analysis": True}, owner_user_id="7",
    )
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run_audio_jobs_worker(stop))
    try:
        for _ in range(100):
            await asyncio.sleep(0.05)
            if task.done():
                pytest.fail(f"audio worker crashed: {task.exception()!r}")
            if seen:
                break
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)
    assert seen == [7]
