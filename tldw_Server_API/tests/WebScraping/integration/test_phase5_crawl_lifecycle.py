"""Real browser, canonical crawl, queue, executor, checkpoint and service together."""

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as enhanced
from tldw_Server_API.app.core.Web_Scraping.contracts.requests import CrawlRequest
from tldw_Server_API.app.core.Web_Scraping.crawl.checkpoints import CheckpointError, CheckpointStore
from tldw_Server_API.app.core.Web_Scraping.crawl.models import CheckpointLimits, CheckpointScope, CrawlLimits
from tldw_Server_API.app.core.Web_Scraping.orchestration import article, crawl
from tldw_Server_API.app.core.Web_Scraping.orchestration.executor import ExtractionExecutorManager
from tldw_Server_API.app.core.Web_Scraping.runtime.controls import ParentStop
from tldw_Server_API.app.services.enhanced_web_scraping_service import WebScrapingService
from tldw_Server_API.tests.Web_Scraping.test_phase4_article_orchestration import _allowed_target
from tldw_Server_API.tests.WebScraping.integration.test_phase5_enhanced_browser_native import _legacy_allowed, _sites

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def _until(predicate, *, timeout=8):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.fixture
def native_crawl(monkeypatch, tmp_path):
    """Only authorize controlled loopback and inject the real test-owned executor."""
    executor = ExtractionExecutorManager(worker_count_loader=lambda: 1, lifecycle_observer=None)
    contexts, checks, plans = [], [], []
    original_dependencies, original_context = article._build_default_dependencies, crawl.CrawlContext
    resolve = crawl.resolve_crawl_plan

    class Guard:
        async def decide(self, url, **kwargs):
            checks.append(url)
            return SimpleNamespace(allowed=urlsplit(url).path != "/deny")

    async def allowed(url, **kwargs):
        return _allowed_target(url)

    async def policy(url, **kwargs):
        return SimpleNamespace(allowed=True)

    def dependencies(cookies):
        value = original_dependencies(cookies)
        value.browser._egress_guard = Guard()
        value.browser._transport_decision = _legacy_allowed
        return replace(value, evaluate_target=allowed, executor=executor)

    def context(*args, **kwargs):
        value = original_context(*args, **kwargs)
        contexts.append(value)
        return value

    def plan(*args, **kwargs):
        value = resolve(*args, **kwargs)
        plans.append(value)
        return value

    monkeypatch.setattr(article, "_build_default_dependencies", dependencies)
    monkeypatch.setattr(crawl, "CrawlContext", context)
    monkeypatch.setattr(crawl, "resolve_crawl_plan", plan)
    monkeypatch.setattr(crawl, "decide_web_outbound_policy", policy)
    monkeypatch.setattr(enhanced, "get_database_dir", lambda: str(tmp_path))
    scraper = enhanced.EnhancedWebScraper({"max_concurrent": 1})
    scraper.cookie_manager = enhanced.CookieManager(tmp_path / "cookies.json")
    scraper.deduplicator = enhanced.ContentDeduplicator(tmp_path / "dedup.json")
    service = WebScrapingService()
    service.scraper, service._initialized = scraper, True
    return SimpleNamespace(
        executor=executor, contexts=contexts, checks=checks, plans=plans, scraper=scraper, service=service
    )


def _execute(harness, url, **kwargs):
    return crawl.execute_crawl(
        CrawlRequest(url),
        profile="enhanced_recursive",
        options={"max_pages": 1, "max_depth": 0, "method": "playwright", "allow_llm_extraction": False},
        supplied=frozenset({"max_pages", "max_depth", "method", "allow_llm_extraction"}),
        config={
            "web_scraper_default_backend": "playwright",
            "web_scraper_respect_robots": False,
            "web_scraper_retry_count": 1,
            "web_scraper_retry_timeout": 5,
        },
        credentials={},
        queue=harness.scraper.job_queue,
        **kwargs,
    )


@pytest.mark.parametrize("stop", ["cancel", "budget"])
async def test_shutdown_retains_live_thread_capacity_and_resume_exclusion(native_crawl, monkeypatch, tmp_path, stop):
    from playwright.async_api import Browser

    harness = native_crawl
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    browsers, extracted = [], []
    close = Browser.close

    async def observed_close(self, *args, **kwargs):
        browsers.append(self)
        return await close(self, *args, **kwargs)

    def extract(document, url, **kwargs):
        entered.set()
        try:
            assert release.wait(30), "test must release the physical extractor"
            extracted.append(url)
            return {"url": url, "content": "late facts", "extraction_successful": True, "custom_extra": {"rank": 7}}
        finally:
            exited.set()

    monkeypatch.setattr(Browser, "close", observed_close)
    build = article._build_default_dependencies
    monkeypatch.setattr(
        article, "_build_default_dependencies", lambda cookies: replace(build(cookies), extract=extract)
    )
    scope = CheckpointScope("owner", "native", root=tmp_path)
    path = tmp_path / "progress.json"
    store = CheckpointStore.open(path, scope=scope, limits=CheckpointLimits())
    task = shutdown = None
    async with _sites() as (bases, seen):
        await harness.scraper.job_queue.start()
        task = asyncio.create_task(_execute(harness, bases[0] + "/children", checkpoint=store))
        try:
            await _until(entered.is_set)
            assert browsers and all(not browser.is_connected() for browser in browsers)
            assert {"/frame", "/frame-image", "/image", "/worker-fetch", "/popup"} <= {p for _, p, _ in seen}
            context = harness.contexts[0]
            generation = harness.executor.current_generation
            assert generation is not None and len(generation.outstanding) == 1
            if stop == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(task), 5)
            else:
                with pytest.raises(ParentStop, match="budget_exhausted"):
                    context.charge_bytes(CrawlLimits().transfer_bytes)
                report = await asyncio.wait_for(asyncio.shield(task), 5)
                assert report.outcome.stop_reason == "budget_exhausted" and report.articles == ()
            shutdown = asyncio.create_task(harness.service.shutdown())
            await _until(lambda: harness.service.get_service_status()["status"] == "stopping")
            assert harness.scraper.job_queue.get_status()["state"] == "stopping"
            assert not shutdown.done() and not exited.is_set()
            assert len(generation.outstanding) == 1
            assert not generation.permits.acquire(blocking=False), "cancellation is not physical capacity"
            with pytest.raises(CheckpointError, match="checkpoint_in_use"):
                CheckpointStore.open(path, scope=scope, limits=CheckpointLimits())
            initializing = asyncio.create_task(harness.service.initialize())
            await asyncio.sleep(0)
            assert not initializing.done(), "startup is serialized behind the entered shutdown"
            initializing.cancel()
            await asyncio.gather(initializing, return_exceptions=True)
            before = len(seen)
            release.set()
            await asyncio.wait_for(shutdown, 8)
            await _until(lambda: not context.owned_handles())
            assert exited.is_set() and extracted == [bases[0] + "/children"]
            assert harness.scraper.job_queue.get_status()["state"] == "stopped"
            assert len(seen) == before
            assert harness.scraper.job_queue.get_status()["active_jobs"] == 0
            assert generation.outstanding == {}
            assert generation.permits.acquire(blocking=False)
            generation.permits.release()
            await _until(lambda: store._closed)
            contender = CheckpointStore.open(path, scope=scope, limits=CheckpointLimits())
            try:
                state = contender.load(plan=harness.plans[0], scope=scope)
                assert state.frontier.results == () and state.pages_scraped == 0
            finally:
                contender.close()
        finally:
            release.set()
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await harness.service.shutdown()
            if shutdown is not None:
                await asyncio.gather(shutdown, return_exceptions=True)
            await harness.executor.shutdown()
            store.close()


@pytest.mark.parametrize("path", ["/start", "/websocket", "/stream-child"])
async def test_combined_native_redirect_guard_and_transfer_stop(native_crawl, path):
    harness = native_crawl
    stream = SimpleNamespace(started=asyncio.Event(), peer_closed=asyncio.Event(), release=asyncio.Event())
    async with _sites(stream=stream) as (bases, seen):
        await harness.scraper.job_queue.start()
        try:
            limits = CrawlLimits(transfer_bytes=1024) if path == "/stream-child" else CrawlLimits()
            report = await asyncio.wait_for(_execute(harness, bases[0] + path, limits=limits), 15)
            assert not any(url == "/deny" for _, url, _ in seen)
            if path == "/stream-child":
                assert report.outcome.stop_reason == "budget_exhausted" and report.articles == ()
                await asyncio.wait_for(stream.peer_closed.wait(), 3)
                assert not stream.release.is_set(), "budget stop must not wait for held EOF"
            elif path == "/websocket":
                assert any(url.endswith("/deny") for url in harness.checks)
                assert report.articles == ()
            else:
                assert {"/start", "/step", "/done"} <= {url for _, url, _ in seen}
                assert report.outcome.state == "completed"
            context = harness.contexts[0]
            assert context.snapshot().network_requests == len(seen), "denied WebSockets never dispatch"
        finally:
            stream.release.set()
            await harness.service.shutdown()
            await harness.executor.shutdown()
        assert harness.scraper.job_queue.get_status()["state"] == "stopped"
        assert harness.contexts[0].owned_handles() == ()
