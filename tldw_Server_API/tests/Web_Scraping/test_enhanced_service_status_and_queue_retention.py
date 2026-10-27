from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import (
    JobStatus,
    ScrapingJob,
    ScrapingJobQueue,
)
from tldw_Server_API.app.services.enhanced_web_scraping_service import WebScrapingService


@pytest.mark.unit
def test_get_service_status_when_scraper_is_unavailable():
    svc = WebScrapingService()
    svc._initialized = True
    svc.scraper = None

    status = svc.get_service_status()

    assert status["status"] == "unavailable"
    assert status["initialized"] is True
    assert status["scraper_available"] is False
    assert status["active_jobs"] == 0
    assert status["queue"]["completed_jobs"] == 0


@pytest.mark.unit
def test_scraping_job_queue_completed_retention_evicts_oldest_entries():
    queue = ScrapingJobQueue(completed_retention=2)

    for job_id in ("job-1", "job-2", "job-3"):
        job = ScrapingJob(job_id=job_id, url=f"https://example.com/{job_id}", method="auto")
        job.status = JobStatus.COMPLETED
        queue._record_completed_job(job)

    assert len(queue._completed_jobs) == 2
    assert "job-1" not in queue._completed_jobs
    assert queue.get_job("job-1") is None
    assert queue.get_job("job-2") is not None
    assert queue.get_job("job-3") is not None


@pytest.mark.unit
def test_scraping_job_queue_zero_completed_retention_discards_completed_jobs():
    queue = ScrapingJobQueue(completed_retention=0)
    job = ScrapingJob(job_id="job-1", url="https://example.com/1", method="auto")
    job.status = JobStatus.COMPLETED

    queue._record_completed_job(job)

    assert queue._completed_jobs == {}
    assert queue.get_job("job-1") is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_summarize_content_runs_analyze_off_the_event_loop_thread(monkeypatch):
    """The summary runs on a worker thread, where it can be quota-checked (spec §4b)."""
    import threading

    from tldw_Server_API.app.services import enhanced_web_scraping_service as enhanced_service

    seen = []

    def fake_analyze(**kwargs):
        """Record the thread."""
        seen.append(threading.get_ident())
        return "summary"

    monkeypatch.setattr(enhanced_service, "analyze", fake_analyze)
    svc = WebScrapingService()

    assert await svc._summarize_content("body", None, "openai", None, None, 0.7) == "summary"
    assert seen and seen[0] != threading.get_ident()
