"""Crawl receipts retain terminal facts through the list-compatible ingest API."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException, Request

from tldw_Server_API.app.api.v1.endpoints import media
from tldw_Server_API.app.api.v1.endpoints.media import ingest_web_content as endpoint
from tldw_Server_API.app.api.v1.schemas.media_request_models import IngestWebContentRequest
from tldw_Server_API.app.core.Web_Scraping.contracts.execution import CrawlReport
from tldw_Server_API.app.services import web_scraping_service as service
from tldw_Server_API.app.services.ephemeral_store import ephemeral_storage

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]
PARTIAL = {"state": "partial", "stop_reason": "deadline_exhausted", "has_partial_results": True}


@pytest.mark.parametrize("method", ["url_level", "recursive_scraping"])
@pytest.mark.parametrize("stored", [False, True])
async def test_ingestion_receipt_preserves_outcome_and_releases_ephemeral(monkeypatch, method, stored):
    envelope = {
        "articles": [{"content": "Body", "extraction_successful": True, "summary": "Summary"}],
        "crawl_outcome": dict(PARTIAL),
    }
    result_id = ephemeral_storage.store_data({"result": envelope}) if stored else None

    async def scrape_task(**kwargs):
        return {"ephemeral_id": result_id, "crawl_outcome": PARTIAL} if stored else envelope

    monkeypatch.setattr(media, "process_web_scraping_task", scrape_task)
    result = await endpoint.ingest_web_content(
        IngestWebContentRequest(
            urls=["https://example.com"], scrape_method=method, perform_analysis=False, perform_chunking=False
        ),
        Request({"type": "http"}),
        BackgroundTasks(),
        db=SimpleNamespace(),
        usage_log=SimpleNamespace(),
        current_user=SimpleNamespace(id=1),
    )
    assert result["crawl_outcome"] == PARTIAL
    assert result["results"][0]["analysis"] == "Summary"
    if result_id:
        assert ephemeral_storage.get_data(result_id) is None


async def test_failed_empty_ingestion_never_becomes_warning(monkeypatch):
    async def scrape_task(**kwargs):
        return {
            "articles": [],
            "crawl_outcome": {"state": "failed", "stop_reason": "policy_denied", "has_partial_results": False},
        }

    monkeypatch.setattr(media, "process_web_scraping_task", scrape_task)
    with pytest.raises(HTTPException) as error:
        await service.ingest_web_content_orchestrate(
            IngestWebContentRequest(
                urls=["https://example.com"], scrape_method="recursive_scraping", perform_analysis=False
            ),
            SimpleNamespace(),
            SimpleNamespace(),
        )
    assert error.value.status_code == 403


async def test_completed_empty_ingestion_keeps_terminal_outcome(monkeypatch):
    async def scrape_task(**kwargs):
        return {
            "articles": [],
            "crawl_outcome": {"state": "completed", "stop_reason": "frontier_exhausted", "has_partial_results": False},
        }

    monkeypatch.setattr(media, "process_web_scraping_task", scrape_task)
    result = await endpoint.ingest_web_content(
        IngestWebContentRequest(
            urls=["https://example.com"],
            scrape_method="recursive_scraping",
            perform_analysis=False,
            perform_chunking=False,
        ),
        Request({"type": "http"}),
        BackgroundTasks(),
        db=SimpleNamespace(),
        usage_log=SimpleNamespace(),
        current_user=SimpleNamespace(id=1),
    )
    assert result["status"] == "warning"
    assert result["crawl_outcome"]["state"] == "completed"


async def test_sitemap_ingestion_retains_partial_and_does_not_summarize_failure(monkeypatch):
    calls = []

    async def acquire(*args, **kwargs):
        return CrawlReport.interrupted(
            [
                {"content": "Body", "extraction_successful": True},
                {"content": "Failure body", "extraction_successful": False, "error": "failed"},
            ],
            "deadline_exhausted",
        )

    def analyze(**kwargs):
        calls.append(kwargs)
        return "Summary"

    monkeypatch.setattr(service, "execute_crawl", acquire)
    monkeypatch.setattr(service, "analyze", analyze)
    result = await service.ingest_web_content_orchestrate(
        IngestWebContentRequest(
            urls=["https://example.com/map.xml"],
            scrape_method="sitemap",
            api_name="openai",
            perform_analysis=True,
            perform_chunking=False,
        ),
        SimpleNamespace(),
        SimpleNamespace(),
    )
    assert result.crawl_outcome == PARTIAL
    assert [item["input_data"] for item in calls] == ["Body"]


async def test_sitemap_pending_cancellation_cannot_launch_a_summary(monkeypatch):
    calls = []

    async def acquire(*args, **kwargs):
        asyncio.current_task().cancel()
        return CrawlReport.interrupted([{"content": "Body", "extraction_successful": True}], "deadline_exhausted")

    monkeypatch.setattr(service, "execute_crawl", acquire)
    monkeypatch.setattr(service, "analyze", lambda **kwargs: calls.append(kwargs))
    task = asyncio.create_task(
        service.ingest_web_content_orchestrate(
            IngestWebContentRequest(urls=["https://example.com/map.xml"], scrape_method="sitemap", api_name="openai"),
            SimpleNamespace(),
            SimpleNamespace(),
        )
    )
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == []


async def test_endpoint_reads_aggregate_before_normalizing_articles(monkeypatch):
    class Articles(list):
        viewed = False

        @property
        def crawl_outcome(self):
            self.viewed = True
            return PARTIAL

        def __iter__(self):
            assert self.viewed, "aggregate must be consumed before list normalization"
            return super().__iter__()

    async def ingest(**kwargs):
        return Articles([{"content": "Body", "extraction_successful": True}])

    monkeypatch.setattr(endpoint, "ingest_web_content_orchestrate", ingest)
    result = await endpoint.ingest_web_content(
        IngestWebContentRequest(urls=["https://example.com"], perform_analysis=False, perform_chunking=False),
        Request({"type": "http"}),
        BackgroundTasks(),
        db=SimpleNamespace(),
        usage_log=SimpleNamespace(),
        current_user=SimpleNamespace(id=1),
    )
    assert result["crawl_outcome"] == PARTIAL


async def test_expired_receipt_keeps_aggregate_outcome_in_sanitized_http_failure(monkeypatch):
    result_id = ephemeral_storage.store_data({"result": {"articles": []}}, ttl_seconds=0)

    async def scrape_task(**kwargs):
        return {"ephemeral_id": result_id, "crawl_outcome": PARTIAL}

    monkeypatch.setattr(media, "process_web_scraping_task", scrape_task)
    with pytest.raises(HTTPException) as error:
        await endpoint.ingest_web_content(
            IngestWebContentRequest(
                urls=["https://example.com"],
                scrape_method="recursive_scraping",
                perform_analysis=False,
                perform_chunking=False,
            ),
            Request({"type": "http"}),
            BackgroundTasks(),
            db=SimpleNamespace(),
            usage_log=SimpleNamespace(),
            current_user=SimpleNamespace(id=1),
        )
    assert error.value.status_code == 500
    assert error.value.detail["crawl_outcome"] == PARTIAL
