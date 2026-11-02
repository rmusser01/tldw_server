"""Task9 compatibility cutover: real canonical results, ownership and adapters."""

import asyncio
from dataclasses import replace

import pytest

from tldw_Server_API.app.core.Web_Scraping.contracts import conversion
from tldw_Server_API.app.core.Web_Scraping.contracts.execution import CrawlInterrupted, CrawlReport
from tldw_Server_API.app.core.Web_Scraping.crawl.models import CrawlLimits
from tldw_Server_API.app.core.Web_Scraping.orchestration import article
from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext
from tldw_Server_API.app.core.Web_Scraping.runtime import RuntimeRequestContext
from tldw_Server_API.tests.Web_Scraping.test_phase4_article_orchestration import URL, _harness

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_transform_after_conversion_before_discovery_without_extraction_fallback(monkeypatch):
    fixture = _harness()
    monkeypatch.setattr(article, "_build_default_dependencies", lambda *_args: fixture.dependencies)
    seen, discovered = [], []

    def duplicate(url, data):
        seen.append((url, data["content"]))
        return {"url": url, "error": "Duplicate content", "extraction_successful": False, "is_duplicate": True}

    handle = article.start_page(
        URL,
        None,
        controls=CrawlContext(CrawlLimits(), clock=lambda: 0),
        request_context=RuntimeRequestContext(),
        discover=lambda *args: discovered.append(args),
        discovery_on_success=True,
        transform=duplicate,
    )
    page = await handle.result
    await handle.drained
    assert seen == [(URL, "converted:article body")]
    assert discovered == []
    assert fixture.browser.calls == []
    assert fixture.extract.call_count == 1
    assert page.article["is_duplicate"] is True
    assert page.article["preflight_analysis"] == {"analysis": {"results": {}}}


@pytest.mark.asyncio
async def test_transformed_success_discovery_is_once_after_transform(monkeypatch):
    fixture = _harness()
    monkeypatch.setattr(article, "_build_default_dependencies", lambda *_args: fixture.dependencies)
    events = []

    def transform(url, data):
        events.append("transform")
        return {**data, "custom_extra": {"labels": ["kept"]}}

    def discover(document, request, final):
        from tldw_Server_API.app.core.Web_Scraping.contracts.execution import DiscoveryResult

        events.append("discover")
        return DiscoveryResult(request, final, ())

    handle = article.start_page(
        URL,
        None,
        controls=CrawlContext(CrawlLimits(), clock=lambda: 0),
        request_context=RuntimeRequestContext(),
        discover=discover,
        transform=transform,
    )
    page = await handle.result
    await handle.drained
    assert events == ["transform", "discover"]
    assert page.article["custom_extra"] == {"labels": ["kept"]}


def test_ambiguous_interruption_raises_original_report_with_partials():
    report = CrawlReport.interrupted([{"url": URL, "content": "kept article"}], "deadline_exhausted")
    with pytest.raises(CrawlInterrupted) as raised:
        conversion.to_legacy_crawl(report, profile="legacy_recursive")
    assert raised.value.report is report
    assert raised.value.report.articles[0]["content"] == "kept article"


def test_established_midcrawl_transport_failure_list_is_lossless():
    articles = [
        {"url": URL, "content": "kept article"},
        {
            "url": URL + "/next",
            "error": "browser_transport_unavailable",
            "extraction_successful": False,
        },
    ]
    report = CrawlReport.interrupted(articles, "transport_unavailable")
    assert conversion.to_legacy_crawl(report, profile="legacy_recursive") == articles


def test_legacy_transport_failure_does_not_invent_placeholder_article_fields():
    failure = {"url": URL, "error": "browser_transport_unavailable", "extraction_successful": False}
    canonical = {**failure, "title": "N/A", "author": "N/A", "date": "N/A", "content": ""}
    report = CrawlReport.interrupted([canonical], "transport_unavailable")
    assert conversion.to_legacy_crawl(report, profile="legacy_recursive") == [failure]


def test_queue_value_exports_are_canonical_identity():
    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as enhanced
    from tldw_Server_API.app.core.Web_Scraping.jobs import models

    assert enhanced.ScrapingJob is models.ScrapingJob
    assert enhanced.JobStatus is models.JobStatus
    assert enhanced.JobPriority is models.JobPriority


@pytest.mark.asyncio
async def test_queue_shim_owns_canonical_retry_and_detached_subscriber():
    from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import ScrapingJob, ScrapingJobQueue
    from tldw_Server_API.app.core.Web_Scraping.jobs.queue import ProcessLocalJobQueue

    entered, release = asyncio.Event(), asyncio.Event()
    attempts = []

    class Parent:
        async def scrape_article(self, url, method, **kwargs):
            attempts.append((url, method, kwargs))
            entered.set()
            await release.wait()
            return {"url": url, "content": "kept article", "extraction_successful": True}

    queue = ScrapingJobQueue(1, Parent(), completed_retention=0)
    assert isinstance(queue, ProcessLocalJobQueue)
    await queue.start()
    future = await queue.add_job(ScrapingJob("one", URL, "beautifulsoup", metadata={"user_agent": "agent"}))
    await entered.wait()
    future.cancel()
    assert queue.get_status()["live_jobs"] == 1
    release.set()
    for _ in range(100):
        if not queue.get_status()["live_jobs"]:
            break
        await asyncio.sleep(0)
    assert queue.get_status()["live_jobs"] == 0
    assert queue.get_job("one") is None
    assert attempts == [
        (
            URL,
            "beautifulsoup",
            {
                "custom_cookies": None,
                "user_agent": "agent",
                "custom_headers": None,
                "allow_llm_extraction": True,
            },
        )
    ]
    await queue.stop()


@pytest.mark.asyncio
async def test_queue_private_canonical_attempt_waits_instance_limiter_before_dispatch():
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Web_Scraping.contracts.execution import DiscoveryResult, PageData
    from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import ScrapingJob, ScrapingJobQueue
    from tldw_Server_API.app.core.Web_Scraping.runtime.controls import WorkHandle

    events = []
    release = asyncio.Event()

    async def acquire():
        events.append("limiter")
        await release.wait()

    queue = ScrapingJobQueue(1, SimpleNamespace(rate_limiter=SimpleNamespace(acquire=acquire)))
    context = CrawlContext(CrawlLimits(), clock=lambda: 0)

    def start(payload, controls):
        events.append("page")
        loop = asyncio.get_running_loop()
        result, drained = loop.create_future(), loop.create_future()
        result.set_result(PageData({"url": URL, "content": "kept"}, DiscoveryResult(URL, URL, ())))
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: None)

    await queue.start()
    handle = await queue.submit_page(ScrapingJob("one", URL, "auto"), controls=context, execute=start)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert events == ["limiter"]
    release.set()
    assert (await handle.result).article["content"] == "kept"
    await handle.drained
    assert events == ["limiter", "page"]
    await queue.stop()


@pytest.mark.asyncio
async def test_midcrawl_transport_failure_retains_list_checkpoint_without_completed_callback(tmp_path):
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as legacy
    from tldw_Server_API.tests.Web_Scraping.phase5_fakes import Phase5Harness

    child = "https://example.com/blog/child"
    fixture = Phase5Harness(
        profile="legacy_recursive",
        outcomes=[
            {"url": URL, "links": [child]},
            {"url": child, "transport_error": True},
        ],
    )
    # Legacy recursion admits only descendants of its seed.
    fixture.outcomes[URL]["links"] = [URL + "/blog/child"]
    fixture.outcomes[URL + "/blog/child"] = fixture.outcomes.pop(child)
    fixture.outcomes[URL + "/blog/child"]["url"] = URL + "/blog/child"
    checkpoint = tmp_path / "crawl.json"
    with fixture.boundaries():
        result = await legacy.recursive_scrape(
            URL,
            5,
            3,
            delay=0,
            resume_file=str(checkpoint),
            progress_callback=fixture.callbacks.append,
        )
    assert result[0]["content"] == "kept article"
    assert result[1]["error"] == "browser_transport_unavailable"
    assert not any("completed" in call for call in fixture.callbacks)
    assert checkpoint.exists()


def test_sync_interruption_keeps_writer_until_actual_work_exits_then_releases(monkeypatch, tmp_path):
    import threading
    import time

    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as legacy
    from tldw_Server_API.app.core.Web_Scraping.contracts.requests import CrawlRequest
    from tldw_Server_API.app.core.Web_Scraping.crawl.checkpoints import CheckpointError, CheckpointStore
    from tldw_Server_API.app.core.Web_Scraping.crawl.models import CheckpointLimits, CheckpointScope
    from tldw_Server_API.app.core.Web_Scraping.orchestration import crawl

    entered, release, exited = threading.Event(), threading.Event(), threading.Event()
    scope = CheckpointScope("local", "sync", trusted_local_path=True)
    path = tmp_path / "progress.json"
    fixture = _harness()
    original = fixture.dependencies.resolve_plan
    loops = []

    def resolve(*args):
        entered.set()
        try:
            assert release.wait(8), "test must release held physical work"
            return original(*args)
        finally:
            exited.set()

    monkeypatch.setattr(
        article,
        "_build_default_dependencies",
        lambda *_args: replace(
            fixture.dependencies,
            resolve_plan=resolve,
        ),
    )

    async def interrupted(*args, **kwargs):
        loops.append(asyncio.get_running_loop())
        store = CheckpointStore.open(path, scope=scope, limits=CheckpointLimits())
        report = await crawl.execute_crawl(
            CrawlRequest(URL),
            profile="enhanced_recursive",
            options={},
            supplied=frozenset(),
            config={},
            credentials={},
            checkpoint=store,
            limits=CrawlLimits(execution_seconds=0.15),
        )
        return conversion.to_legacy_crawl(report, profile="enhanced_recursive")

    monkeypatch.setattr(legacy, "recursive_scrape", interrupted)
    try:
        with pytest.raises(CrawlInterrupted) as raised:
            legacy.sync_recursive_scrape(URL, 2, 2, delay=0)
        assert raised.value.report.outcome.stop_reason == "deadline_exhausted"
        assert entered.is_set() and not exited.is_set()
        with pytest.raises(CheckpointError, match="checkpoint_in_use"):
            CheckpointStore.open(path, scope=scope, limits=CheckpointLimits())
        # Release only after the sync call returned and actual lock contention proved ownership.
        release.set()
        assert exited.wait(2)
        deadline = time.monotonic() + 2
        while True:
            try:
                replacement = CheckpointStore.open(path, scope=scope, limits=CheckpointLimits())
            except CheckpointError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
            else:
                replacement.close()
                break
    finally:
        release.set()
        # Clean up the pre-cutover abandoned loop on RED; never interrupt a live bridge.
        for loop in loops:
            if not loop.is_running() and not loop.is_closed():
                loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(loop)))
                loop.run_until_complete(loop.shutdown_default_executor())
                loop.close()


@pytest.mark.asyncio
async def test_stored_cookie_snapshot_is_selected_per_exact_netloc_and_charged(monkeypatch):
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Web_Scraping.contracts.execution import DiscoveryResult, PageData
    from tldw_Server_API.app.core.Web_Scraping.contracts.requests import CrawlRequest
    from tldw_Server_API.app.core.Web_Scraping.jobs.queue import ProcessLocalJobQueue
    from tldw_Server_API.app.core.Web_Scraping.orchestration import crawl
    from tldw_Server_API.app.core.Web_Scraping.runtime.controls import WorkHandle

    urls = ["https://example.com/a", "https://example.com:443/b", "https://other.example/c"]
    jar = {
        "example.com": [{"name": "sid", "value": "stored"}, {"name": "root", "value": "root-secret"}],
        "example.com:443": [{"name": "port", "value": "port-secret"}],
        "other.example": [{"name": "other", "value": "other-secret"}],
    }
    seen, payloads = [], []

    async def allow(*args, **kwargs):
        return SimpleNamespace(allowed=True)

    def start(url, cookies, *, controls, **kwargs):
        seen.append((url, {item["name"]: item["value"] for item in cookies}))
        jar["other.example"][0]["value"] = "mutated"
        controls.reserve("page_attempts").commit()
        loop = asyncio.get_running_loop()
        result, drained = loop.create_future(), loop.create_future()
        result.set_result(PageData({"url": url, "content": "kept"}, DiscoveryResult(url, url, ())))
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: None)

    class Queue(ProcessLocalJobQueue):
        async def submit_page(self, job, **kwargs):
            payloads.append(job.metadata)
            assert "stored_cookies_by_netloc" not in job.metadata
            return await super().submit_page(job, **kwargs)

    monkeypatch.setattr(crawl, "decide_web_outbound_policy", allow)
    monkeypatch.setattr(article, "start_page", start)
    queue = Queue(execute=lambda *_args: None, max_concurrent=1)
    await queue.start()
    try:
        report = await crawl.execute_crawl(
            CrawlRequest(urls[0]),
            profile="enhanced_batch",
            options={},
            supplied=frozenset(),
            config={},
            credentials={"stored_cookies_by_netloc": jar, "custom_cookies": [{"name": "sid", "value": "explicit"}]},
            seeds=urls,
            queue=queue,
        )
        assert seen == [
            (urls[0], {"root": "root-secret", "sid": "explicit"}),
            (urls[1], {"port": "port-secret", "sid": "explicit"}),
            (urls[2], {"other": "other-secret", "sid": "explicit"}),
        ]
        assert [{item["name"]: item["value"] for item in payload["custom_cookies"]} for payload in payloads] == [
            values for _, values in seen
        ]
        assert "secret" not in repr(report)
        assert "secret" not in repr(queue.get_job(next(iter(queue._history.entries)))).lower()
    finally:
        await queue.stop()


def test_cookie_manager_private_snapshot_is_detached(tmp_path):
    from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import CookieManager

    manager = CookieManager(tmp_path / "cookies.json")
    manager.add_cookies("Example.com:443", [{"name": "sid", "value": "before"}])
    snapshot = manager._snapshot()
    manager._cookies["Example.com:443"][0]["value"] = "after"
    assert snapshot == {"Example.com:443": [{"name": "sid", "value": "before"}]}


@pytest.mark.asyncio
async def test_batch_parent_stop_is_not_an_ambiguous_failure_list():
    from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import EnhancedWebScraper
    from tldw_Server_API.app.core.Web_Scraping.runtime.controls import ParentStop

    scraper = EnhancedWebScraper(config={})

    async def stopped(*args, **kwargs):
        raise ParentStop("deadline_exhausted")

    scraper.scrape_article = stopped
    await scraper.job_queue.start()
    try:
        with pytest.raises(CrawlInterrupted) as raised:
            await scraper.scrape_multiple([URL])
        assert raised.value.report.outcome.stop_reason == "deadline_exhausted"
    finally:
        await scraper.job_queue.stop()


@pytest.mark.asyncio
async def test_enhanced_public_conversion_does_not_mark_partial_progress_completed(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as enhanced
    from tldw_Server_API.tests.Web_Scraping.phase5_fakes import Phase5Harness

    report = CrawlReport.interrupted([{"url": URL, "content": "kept article"}], "deadline_exhausted")

    async def interrupted(*args, **kwargs):
        return report

    monkeypatch.setattr(enhanced, "execute_crawl", interrupted, raising=False)
    fixture = Phase5Harness(profile="enhanced_recursive", outcomes=[{"url": URL}])
    with fixture.boundaries():
        fixture.scraper._progress["partial"] = {"status": "completed", "completed_at": "old-run"}
        await fixture.scraper.job_queue.start()
        try:
            with pytest.raises(CrawlInterrupted) as raised:
                await fixture.scraper.recursive_scrape(URL, task_id="partial")
            assert raised.value.report is report
            assert fixture.scraper.get_progress("partial")["status"] == "partial"
            assert "completed_at" not in fixture.scraper.get_progress("partial")
        finally:
            await fixture.scraper.job_queue.stop()


@pytest.mark.asyncio
async def test_duplicate_attempts_transform_before_retry_without_discovery_debit(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import ContentDeduplicator, EnhancedWebScraper
    from tldw_Server_API.app.core.Web_Scraping.runtime import FetchResponse

    fixture = _harness(extract_results=[{"url": URL, "content": "body", "extraction_successful": True}] * 4)
    fixture.fetch.outcomes = [
        FetchResponse(
            url=URL,
            status=200,
            text='<a href="/unused">unused</a>' * 10,
            headers={},
            backend="httpx",
        )
    ] * 4
    monkeypatch.setattr(article, "_build_default_dependencies", lambda *_args: fixture.dependencies)
    scraper = EnhancedWebScraper(config={})
    scraper.deduplicator = ContentDeduplicator(tmp_path / "hashes.json")
    scraper.deduplicator.add_content(URL + "/original", "converted:body")
    attempts = []
    transform = scraper._apply_dedup

    def apply(url, data):
        attempts.append(data["content"])
        return transform(url, data)

    scraper._apply_dedup = apply
    # Any unused discovery would exceed this exact seed-only allowance.
    from tldw_Server_API.app.core.Web_Scraping.orchestration import crawl

    execute = crawl.execute_crawl

    async def limited(*args, **kwargs):
        return await execute(*args, limits=CrawlLimits(discovered_candidates=1), **kwargs)

    from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as enhanced

    monkeypatch.setattr(enhanced, "execute_crawl", limited)
    try:
        report = await scraper._crawl_report(URL, profile="enhanced_recursive", options={})
        assert report.articles == () and report.outcome.state == "completed"
        assert attempts == ["converted:body"] * 4
        assert fixture.browser.calls == []
        assert len(fixture.fetch.requests) == 4
    finally:
        await scraper.job_queue.stop()


@pytest.mark.asyncio
async def test_duplicate_only_interruption_checkpoint_has_no_successful_partial(monkeypatch, tmp_path):
    import json

    from tldw_Server_API.app.core.Web_Scraping.contracts.requests import CrawlRequest
    from tldw_Server_API.app.core.Web_Scraping.crawl.checkpoints import CheckpointStore
    from tldw_Server_API.app.core.Web_Scraping.crawl.models import CheckpointLimits, CheckpointScope, CrawlHooks
    from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import ContentDeduplicator, EnhancedWebScraper
    from tldw_Server_API.app.core.Web_Scraping.orchestration import crawl
    from tldw_Server_API.app.core.Web_Scraping.runtime.controls import ParentStop

    fixture = _harness()
    monkeypatch.setattr(article, "_build_default_dependencies", lambda *_args: fixture.dependencies)
    scraper = EnhancedWebScraper(config={})
    scraper.deduplicator = ContentDeduplicator(tmp_path / "hashes.json")
    scraper.deduplicator.add_content(URL + "/original", "converted:article body")

    def interrupt(snapshot):
        if snapshot["status"] == "in_progress":
            raise ParentStop("deadline_exhausted")

    path = tmp_path / "checkpoint.json"
    store = CheckpointStore.open(
        path, scope=CheckpointScope("local", "duplicate", trusted_local_path=True), limits=CheckpointLimits()
    )
    report = await crawl.execute_crawl(
        CrawlRequest(URL),
        profile="enhanced_recursive",
        options={},
        supplied=frozenset(),
        config={},
        credentials={"stored_cookies_by_netloc": {"unused.example": [{"name": "secret", "value": "hidden-cookie"}]}},
        checkpoint=store,
        hooks=CrawlHooks(on_progress=interrupt),
        transform=scraper._apply_dedup,
    )
    assert report.outcome.state == "failed" and not report.outcome.has_partial_results
    assert report.outcome.stop_reason == "deadline_exhausted"
    saved = json.loads(path.read_text())
    assert saved["frontier"]["results"][0]["is_duplicate"] is True
    assert not saved["frontier"]["results"][0]["extraction_successful"]
    assert "hidden-cookie" not in path.read_text()
    assert "stored_cookies_by_netloc" not in path.read_text()


@pytest.mark.asyncio
async def test_transform_capability_retained_only_until_actual_drain(monkeypatch):
    import gc
    import threading
    import weakref

    entered, release = threading.Event(), threading.Event()
    fixture = _harness()
    resolve = fixture.dependencies.resolve_plan

    def held(*args):
        entered.set()
        assert release.wait(5), "test must release physical plan resolution"
        return resolve(*args)

    monkeypatch.setattr(
        article, "_build_default_dependencies", lambda *_args: replace(fixture.dependencies, resolve_plan=held)
    )

    class Transform:
        def __call__(self, url, value):
            return value

    transform = Transform()
    retained = weakref.ref(transform)
    handle = article.start_page(
        URL,
        None,
        controls=CrawlContext(CrawlLimits(), clock=lambda: 0),
        request_context=RuntimeRequestContext(),
        discover=lambda *_args: None,
        transform=transform,
    )
    del transform
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        handle.cancel()
        with pytest.raises(asyncio.CancelledError):
            await handle.result
        gc.collect()
        assert not handle.drained.done() and retained() is not None
    finally:
        release.set()
        await asyncio.wait_for(asyncio.shield(handle.drained), 2)
    await asyncio.sleep(0)
    gc.collect()
    assert retained() is None


@pytest.mark.asyncio
async def test_discovery_cross_redirect_does_not_reattach_stored_or_explicit_secrets(monkeypatch):
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Web_Scraping.contracts.requests import CrawlRequest
    from tldw_Server_API.app.core.Web_Scraping.orchestration import crawl
    from tldw_Server_API.app.core.Web_Scraping.runtime import FetchResponse

    seen = []

    async def allow(*args, **kwargs):
        return SimpleNamespace(allowed=True)

    async def fetch(dependencies, request):
        seen.append((request.url, dict(request.cookies), dict(request.headers)))
        if len(seen) == 1:
            return FetchResponse(
                url=request.url,
                status=302,
                text="",
                headers={"Location": "https://other.example/sitemap"},
                backend="httpx",
            )
        return FetchResponse(url=request.url, status=200, text="<urlset/>", headers={}, backend="httpx")

    monkeypatch.setattr(crawl, "decide_web_outbound_policy", allow)
    monkeypatch.setattr(article, "_fetch_response", fetch)
    report = await crawl.execute_crawl(
        CrawlRequest(URL),
        profile="enhanced_sitemap",
        options={},
        supplied=frozenset(),
        config={},
        credentials={
            "stored_cookies_by_netloc": {
                "example.com": [{"name": "stored", "value": "first-secret"}],
                "other.example": [{"name": "other", "value": "other-secret"}],
            },
            "custom_cookies": [{"name": "explicit", "value": "explicit-secret"}],
            "custom_headers": {"Authorization": "private-header"},
        },
    )
    assert report.outcome.state == "completed"
    assert seen[0][1] == {"stored": "first-secret", "explicit": "explicit-secret"}
    assert seen[1][1] == {} and "Authorization" not in seen[1][2]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent, headers, expected_agent",
    [
        ("explicit-agent", {"User-Agent": "header-agent"}, "explicit-agent"),
        (None, {"User-Agent": "header-agent"}, "header-agent"),
        (None, {}, "route-agent"),
    ],
)
async def test_article_stored_explicit_route_cookie_precedence(monkeypatch, agent, headers, expected_agent):
    from tldw_Server_API.app.core.Web_Scraping.contracts.requests import CrawlRequest
    from tldw_Server_API.app.core.Web_Scraping.orchestration import crawl

    fixture = _harness()
    resolve = fixture.dependencies.resolve_plan
    fixture.dependencies = replace(
        fixture.dependencies,
        resolve_plan=lambda *args: replace(
            resolve(*args), cookies={"sid": "route"}, headers={"User-Agent": "route-agent"}
        ),
    )
    monkeypatch.setattr(article, "_build_default_dependencies", lambda *_args: fixture.dependencies)
    report = await crawl.execute_crawl(
        CrawlRequest(URL),
        profile="enhanced_batch",
        options={},
        supplied=frozenset(),
        config={},
        credentials={
            "stored_cookies_by_netloc": {"example.com": [{"name": "sid", "value": "stored"}]},
            "custom_cookies": [{"name": "sid", "value": "explicit"}],
            "user_agent": agent,
            "custom_headers": headers,
        },
    )
    assert report.outcome.state == "completed"
    assert fixture.fetch.requests[0].cookies == {"sid": "route"}
    assert fixture.fetch.requests[0].headers["User-Agent"] == expected_agent


@pytest.mark.asyncio
async def test_progress_save_load_remains_plain_snapshot(tmp_path):
    from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import EnhancedWebScraper

    scraper = EnhancedWebScraper(config={})
    snapshot = {"status": "partial", "owner_user_id": "7", "pages_scraped": 1, "custom_extra": {"labels": ["kept"]}}
    scraper._progress["task"] = snapshot
    path = tmp_path / "progress.json"
    await scraper.save_progress("task", path)
    scraper._progress.clear()
    await scraper.load_progress("task", path)
    assert scraper.get_progress("task", owner_user_id="7") == snapshot
    assert scraper.get_progress("task", owner_user_id="9") == {}


def test_sync_bridge_loop_startup_failure_is_delivered_not_stranded(monkeypatch):
    from concurrent.futures import Future

    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as legacy

    class BoundedFuture(Future):
        def result(self, timeout=None):
            return super().result(timeout=0.2)

    error = RuntimeError("cannot start test loop")

    def fail():
        raise error

    async def unused():
        return None

    monkeypatch.setattr(legacy, "Future", BoundedFuture)
    monkeypatch.setattr(legacy.asyncio, "new_event_loop", fail)
    with pytest.raises(RuntimeError) as raised:
        legacy._run_crawl_sync(unused())
    assert raised.value is error


@pytest.mark.asyncio
async def test_direct_deadline_after_article_raises_with_partials_and_checkpoint(tmp_path):
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as legacy
    from tldw_Server_API.app.core.Web_Scraping.runtime.controls import ParentStop
    from tldw_Server_API.tests.Web_Scraping.phase5_fakes import Phase5Harness

    child = URL + "/blog/child"
    fixture = Phase5Harness(profile="legacy_recursive", outcomes=[{"url": URL, "links": [child]}, {"url": child}])

    def progress(message):
        fixture.callbacks.append(message)
        if "page 2/" in message:
            raise ParentStop("deadline_exhausted")

    path = tmp_path / "crawl.json"
    with fixture.boundaries():
        with pytest.raises(CrawlInterrupted) as raised:
            await legacy.recursive_scrape(URL, 3, 2, delay=0, resume_file=str(path), progress_callback=progress)
    assert raised.value.report.articles[0]["content"] == "kept article"
    assert raised.value.report.outcome.stop_reason == "deadline_exhausted"
    assert path.exists() and not any("completed" in call for call in fixture.callbacks)


def test_repeated_interrupted_sync_calls_retire_loops_executors_tasks_and_private_refs(monkeypatch, tmp_path):
    import gc
    import threading
    import time
    import weakref

    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as legacy
    from tldw_Server_API.app.core.Web_Scraping.contracts.requests import CrawlRequest
    from tldw_Server_API.app.core.Web_Scraping.crawl.checkpoints import CheckpointError, CheckpointStore
    from tldw_Server_API.app.core.Web_Scraping.crawl.models import CheckpointLimits, CheckpointScope
    from tldw_Server_API.app.core.Web_Scraping.orchestration import crawl

    loops, workers, callbacks, checkpoints = [], [], [], []
    releases = [threading.Event() for _ in range(3)]
    entered = [threading.Event() for _ in range(3)]
    fixture = _harness()
    resolve = fixture.dependencies.resolve_plan
    baseline_bridges = {thread.ident for thread in threading.enumerate() if thread.name == "crawl-compat-drain"}

    class Transform:
        def __init__(self):
            self.secret = "private-live-capability"

        def __call__(self, url, data):
            return data

    async def interrupted(index, *args, **kwargs):
        loop = asyncio.get_running_loop()
        loops.append(weakref.ref(loop))

        def held(*args):
            entered[index].set()
            assert releases[index].wait(8), "test must release physical work"
            return resolve(*args)

        monkeypatch.setattr(
            article, "_build_default_dependencies", lambda *_args: replace(fixture.dependencies, resolve_plan=held)
        )
        transform = Transform()
        callbacks.append(weakref.ref(transform))
        path = tmp_path / f"progress-{index}.json"
        scope = CheckpointScope("local", str(index), trusted_local_path=True)
        checkpoints.append((path, scope))
        store = CheckpointStore.open(path, scope=scope, limits=CheckpointLimits())
        report = await crawl.execute_crawl(
            CrawlRequest(URL),
            profile="enhanced_recursive",
            options={},
            supplied=frozenset(),
            config={},
            credentials={},
            checkpoint=store,
            transform=transform,
            limits=CrawlLimits(execution_seconds=0.15),
        )
        workers.extend(weakref.ref(thread) for thread in loop._default_executor._threads)
        return conversion.to_legacy_crawl(report, profile="enhanced_recursive")

    try:
        for index in range(3):

            async def call(*args, index=index, **kwargs):
                return await interrupted(index, *args, **kwargs)

            monkeypatch.setattr(legacy, "recursive_scrape", call)
            with pytest.raises(CrawlInterrupted):
                legacy.sync_recursive_scrape(URL, 2, 2, delay=0)
            assert entered[index].is_set() and callbacks[index]() is not None
            path, scope = checkpoints[index]
            with pytest.raises(CheckpointError, match="checkpoint_in_use"):
                CheckpointStore.open(path, scope=scope, limits=CheckpointLimits())
        assert len(loops) == 3
    finally:
        for release in releases:
            release.set()

    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        gc.collect()
        bridges = [
            thread
            for thread in threading.enumerate()
            if thread.name == "crawl-compat-drain" and thread.ident not in baseline_bridges
        ]
        if not bridges and all(reference() is None or not reference().is_alive() for reference in workers):
            break
        time.sleep(0.01)
    assert not bridges
    assert all(reference() is None or not reference().is_alive() for reference in workers)
    assert all(reference() is None or reference().is_closed() for reference in loops)
    assert all(not asyncio.all_tasks(reference()) for reference in loops if reference() is not None)
    gc.collect()
    assert all(reference() is None for reference in callbacks)
    for path, scope in checkpoints:
        replacement = CheckpointStore.open(path, scope=scope, limits=CheckpointLimits())
        replacement.close()


@pytest.mark.asyncio
async def test_instance_stop_does_not_close_browser_ahead_of_owned_callback_drain():
    import threading
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import EnhancedWebScraper, ScrapingJob
    from tldw_Server_API.app.core.Web_Scraping.runtime.controls import owned_to_thread

    entered, release = threading.Event(), threading.Event()
    closed = []
    scraper = EnhancedWebScraper(config={})

    def held():
        entered.set()
        assert release.wait(5), "test must release physical callback work"

    async def scrape(*args, **kwargs):
        await owned_to_thread(held)
        return {"url": URL, "content": "kept", "extraction_successful": True}

    async def close():
        closed.append(True)

    scraper.scrape_article = scrape
    scraper._browser = SimpleNamespace(close=close)
    await scraper.job_queue.start()
    result = await scraper.job_queue.add_job(ScrapingJob("held", URL, "auto"))
    assert await asyncio.to_thread(entered.wait, 2)
    stop = asyncio.create_task(scraper.stop())
    try:
        for _ in range(20):
            await asyncio.sleep(0)
        assert result.done()
        assert closed == [] and not stop.done()
    finally:
        release.set()
        await asyncio.wait_for(stop, 2)
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["legacy_recursive", "legacy_sitemap", "legacy_url_level"])
async def test_sync_adapter_is_materialized_inside_an_existing_event_loop(profile):
    from tldw_Server_API.tests.Web_Scraping.phase5_fakes import Phase5Harness

    caller = asyncio.get_running_loop()
    fixture = Phase5Harness(profile=profile, outcomes=[{"url": URL}])
    result = fixture.run_sync()
    assert isinstance(result, list) and result[0]["content"] == "kept article"
    assert asyncio.get_running_loop() is caller
