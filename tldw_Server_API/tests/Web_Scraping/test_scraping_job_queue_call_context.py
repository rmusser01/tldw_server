"""Scraping queue workers run each job as the request that submitted it (TASK-13864)."""

import asyncio
from typing import Any

import pytest

from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Usage.llm_call_context import (
    current_llm_call_context,
    llm_call_context,
    mark_llm_quota_prechecked,
)
from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import ScrapingJob, ScrapingJobQueue

pytestmark = pytest.mark.unit


def _job(job_id: str) -> ScrapingJob:
    return ScrapingJob(job_id=job_id, url=f"https://example.com/{job_id}", method="trafilatura")


@pytest.mark.asyncio
async def test_a_job_runs_with_its_submitters_user_not_the_admission_of_the_request_that_started_the_workers() -> None:
    """The workers start lazily inside a request; a later request's job must not run as that request."""
    queue = ScrapingJobQueue(max_concurrent=1)
    seen: list[dict[str, Any]] = []

    async def _execute(job: ScrapingJob) -> dict[str, Any]:
        context = current_llm_call_context()
        seen.append({"user_id": context.user_id, "prechecked": context.prechecked})
        return {"extraction_successful": True}

    queue._execute_job = _execute  # type: ignore[method-assign]

    async def first_request() -> None:
        """User 7, admitted at the boundary, starts the service's workers."""
        with llm_call_context(user_id=7):
            mark_llm_quota_prechecked()
            await queue.start()

    async def second_request() -> dict[str, Any]:
        """User 9, not admitted."""
        with llm_call_context(user_id=9):
            future = await queue.add_job(_job("j2"))
            return await asyncio.wait_for(future, timeout=5)

    try:
        await asyncio.create_task(first_request())
        result = await asyncio.create_task(second_request())
    finally:
        await queue.stop()

    assert result == {"extraction_successful": True}
    assert seen == [{"user_id": 9, "prechecked": False}]


@pytest.mark.asyncio
async def test_a_refused_job_fails_with_the_refusal_and_the_worker_runs_the_next_job() -> None:
    """A quota refusal in one job is that job's failure, not the worker's."""
    queue = ScrapingJobQueue(max_concurrent=1)

    async def _execute(job: ScrapingJob) -> dict[str, Any]:
        if job.job_id == "refused":
            raise LLMQuotaExceededError(
                category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
            )
        return {"extraction_successful": True, "url": job.url}

    queue._execute_job = _execute  # type: ignore[method-assign]
    await queue.start()
    try:
        refused = await queue.add_job(_job("refused"))
        with pytest.raises(LLMQuotaExceededError):
            await asyncio.wait_for(refused, timeout=5)
        following = await queue.add_job(_job("next"))
        result = await asyncio.wait_for(following, timeout=5)
    finally:
        await queue.stop()

    assert result == {"extraction_successful": True, "url": "https://example.com/next"}
