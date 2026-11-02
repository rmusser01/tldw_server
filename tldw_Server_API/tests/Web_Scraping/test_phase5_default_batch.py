"""Task9 real default batch acquisition with only outbound transport substituted."""

import asyncio
import gc
import threading
import weakref
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core import http_client
from tldw_Server_API.app.core.Web_Scraping import enhanced_web_scraping as enhanced
from tldw_Server_API.app.core.Web_Scraping.crawl import models, profiles
from tldw_Server_API.app.core.Web_Scraping.orchestration import crawl_context
from tldw_Server_API.app.core.Web_Scraping.runtime import fetch

pytestmark = pytest.mark.unit
BASE = "https://93.184.216.34/article/"
PARAGRAPH = (
    "Researchers compared the experimental observations with the independently collected measurements. "
    "The resulting analysis explains the evidence, limitations, and practical consequences in detail. "
)
HTML = (
    "<html><head><title>Research findings</title></head><body><article><h1>Research findings</h1>"
    + ("<p>" + PARAGRAPH * 5 + "</p>") * 8
    + "</article></body></html>"
)


@pytest.fixture
def batch(monkeypatch, tmp_path):
    scraper = enhanced.EnhancedWebScraper(
        config={
            "web_scraper_default_backend": "httpx",
            "web_scraper_respect_robots": False,
            "max_concurrent": 1,
            "max_rps": 10000,
            "max_rpm": 10000,
            "max_rph": 10000,
            "web_scraper_retry_count": 0,
        }
    )
    scraper.cookie_manager = enhanced.CookieManager(tmp_path / "cookies.json")
    scraper.deduplicator = enhanced.ContentDeduplicator(tmp_path / "dedup.json")
    calls, contexts, plans = [], [], []
    state = SimpleNamespace(
        scraper=scraper,
        calls=calls,
        contexts=contexts,
        plans=plans,
        body=HTML,
        before_send=None,
        status=lambda: 200,
        unique=True,
    )

    def outbound(url=None, *args, **kwargs):
        calls.append((url, kwargs))
        if state.before_send:
            state.before_send(url, kwargs)
        body = state.body
        if state.unique:
            body = body.replace("Research findings", str(url)).replace("Researchers", f"Researchers of {url}")
        observer = kwargs.get("on_response_chunk")
        if observer:
            observer(len(body.encode()))
        return SimpleNamespace(url=url, text=body, status_code=state.status(), headers={"content-type": "text/html"})

    async def async_outbound(*, method, url, **kwargs):
        return outbound(url, **kwargs)

    real_context, real_resolve = crawl_context.CrawlContext, profiles.resolve_crawl_plan

    def captured_context(limits, **kwargs):
        context = real_context(limits, **kwargs)
        contexts.append(context)
        return context

    def captured_plan(*args, **kwargs):
        plan = real_resolve(*args, **kwargs)
        plans.append(plan)
        return plan

    def limits(**kwargs):
        selected = replace(models.CrawlLimits(), **kwargs)
        monkeypatch.setattr(models, "CrawlLimits", lambda: selected)

    state.limits = limits
    monkeypatch.setattr(crawl_context, "CrawlContext", captured_context)
    monkeypatch.setattr(profiles, "resolve_crawl_plan", captured_plan)
    monkeypatch.setattr(enhanced, "afetch", async_outbound)
    monkeypatch.setattr(http_client, "afetch", async_outbound)
    monkeypatch.setattr(fetch, "http_fetch", outbound)
    return state


@pytest.mark.asyncio
async def test_default_batch_stops_at_real_parent_network_limit(batch):
    batch.limits(network_requests=1)
    try:
        report = await batch.scraper._batch_report([BASE + "one", BASE + "two"])
        assert report.outcome.stop_reason == "budget_exhausted"
        assert batch.plans[0].limits.network_requests == batch.contexts[0]._limits.network_requests == 1
        assert len(batch.calls) == batch.contexts[0].snapshot().network_requests == 1
        assert batch.contexts[0].snapshot().page_attempts == 2
        assert [item["url"] for item in report.articles if item.get("content")] == [BASE + "one"]
    finally:
        await batch.scraper.stop()


@pytest.mark.asyncio
async def test_default_batch_real_analyzer_parent_stop_cannot_fail_open_or_fetch_article(batch):
    batch.limits(network_requests=1)
    batch.scraper.config["web_scraper_preflight_analyzers"] = True
    batch.scraper.config["web_scraper_preflight_enable_external_tools"] = False
    try:
        report = await batch.scraper._batch_report([BASE + "one"])
        assert report.outcome.stop_reason == "budget_exhausted"
        assert len(batch.calls) == batch.contexts[0].snapshot().network_requests == 1
        assert batch.calls[0][0].endswith("/robots.txt")
        assert batch.contexts[0].snapshot().page_attempts == 1
        assert batch.contexts[0].snapshot().browser_attempts == 0
        assert not any(item.get("content") for item in report.articles)
    finally:
        await batch.scraper.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "page_limit, expected_calls, reason", [(1000, 5, "frontier_exhausted"), (4, 4, "budget_exhausted")]
)
async def test_default_batch_dedup_retries_charge_each_canonical_attempt(batch, page_limit, expected_calls, reason):
    batch.limits(page_attempts=page_limit)
    batch.body = "<html><body><article>" + ("<p>" + PARAGRAPH * 5 + "</p>") * 8 + "</article></body></html>"
    batch.unique = False
    try:
        report = await batch.scraper._batch_report([BASE + "one", BASE + "two"])
        assert report.outcome.stop_reason == reason
        snapshot = batch.contexts[0].snapshot()
        assert snapshot.page_attempts == snapshot.network_requests == len(batch.calls) == expected_calls
        assert report.articles[0]["extraction_successful"]
        assert not any(item.get("content") for item in report.articles[1:])
        if reason == "frontier_exhausted":
            assert report.articles[1]["is_duplicate"]
    finally:
        await batch.scraper.stop()


@pytest.mark.asyncio
async def test_default_batch_cancel_keeps_real_transport_and_instance_owned_until_exit(batch):
    entered, release, exited = threading.Event(), threading.Event(), threading.Event()

    def hold(url, kwargs):
        entered.set()
        try:
            assert release.wait(10), "test must release actual transport after ownership assertions"
        finally:
            exited.set()

    batch.before_send = hold
    task = asyncio.create_task(batch.scraper._batch_report([BASE + "one", BASE + "two"]))
    stop = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(task, 4)
        status = batch.scraper.job_queue.get_status()
        assert status["live_jobs"] == 1 and status["payload_bytes"] > 0
        assert not exited.is_set()
        assert batch.contexts[0].snapshot().page_attempts == 1
        stop = asyncio.create_task(batch.scraper.stop())
        await asyncio.sleep(0)
        assert not stop.done()
    finally:
        release.set()
        assert await asyncio.to_thread(exited.wait, 2)
        if stop is None:
            stop = asyncio.create_task(batch.scraper.stop())
        await asyncio.wait_for(stop, 3)
    assert batch.scraper.job_queue.get_status()["live_jobs"] == 0
    assert len(batch.calls) == 1
    retained = weakref.ref(batch.scraper)
    batch.scraper = None
    del stop
    gc.collect()
    assert raised.value.report.outcome.state == "cancelled"
    assert retained() is None, "retained cancelled caller must not own the drained instance graph"


@pytest.mark.asyncio
async def test_default_batch_snapshots_instance_config_cookies_and_headers_before_rate_wait(
    batch, monkeypatch, tmp_path
):
    import json

    import yaml

    from tldw_Server_API.app.core.Web_Scraping.policy import DefaultWebOutboundPolicyChecker

    route_file = tmp_path / "routes.yaml"
    route_file.write_text(
        yaml.safe_dump(
            {
                "domains": {
                    "93.184.216.34": {
                        "backend": "auto",
                        "respect_robots": False,
                        "cookies": {"sid": "route-secret"},
                        "extra_headers": {"X-Route": "route-header"},
                        "proxies": {"https": "http://proxy.example:8080"},
                    }
                }
            }
        )
    )
    monkeypatch.setattr(http_client, "PROXY_ALLOWLIST", {"proxy.example"})
    batch.scraper.config.update(custom_scrapers_yaml_path=str(route_file), web_scraper_default_backend="curl")
    batch.scraper.cookie_manager.add_cookies("93.184.216.34", [{"sid": "stored-secret", "stored": "kept-stored"}])
    batch.scraper.cookie_manager.add_cookies("93.184.216.34:8443", [{"other": "wrong-port-secret"}])
    entered, release = asyncio.Event(), asyncio.Event()
    waits, contexts = [], []
    original_decide = DefaultWebOutboundPolicyChecker.decide

    async def observed_decide(self, *args, **kwargs):
        contexts.append(kwargs["context"])
        return await original_decide(self, *args, **kwargs)

    async def held_rate():
        waits.append(True)
        entered.set()
        await release.wait()

    monkeypatch.setattr(DefaultWebOutboundPolicyChecker, "decide", observed_decide)
    monkeypatch.setattr(batch.scraper.rate_limiter, "acquire", held_rate)
    options = {
        "custom_cookies": [{"sid": "explicit-secret", "explicit": "kept-explicit"}],
        "custom_headers": {"Authorization": "private-header"},
        "user_agent": "selected-agent",
    }
    task = asyncio.create_task(
        batch.scraper._batch_report([BASE + "one"], method="beautifulsoup", user_id="owner", options=options)
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        batch.scraper.config["web_scraper_default_backend"] = "httpx"
        batch.scraper.cookie_manager._cookies.clear()
        options["custom_headers"]["Authorization"] = "mutated-header"
        options["custom_cookies"][0]["explicit"] = "mutated-cookie"
        private = next(iter(batch.scraper.job_queue._entries.values())).payload
        assert "wrong-port-secret" not in json.dumps(private.options)
        public = batch.scraper.job_queue.get_job(private.job_id)
        assert not any(
            secret in repr(public) for secret in ("route-secret", "kept-stored", "private-header", "kept-explicit")
        )
        release.set()
        report = await task
        assert report.outcome.state == "completed"
        assert report.articles[0]["method"] == "beautifulsoup"
        assert waits == [True] and len(batch.calls) == 1
        kwargs = batch.calls[0][1]
        assert kwargs["backend"] == "curl"
        assert kwargs["proxies"] == {"https": "http://proxy.example:8080"}
        assert kwargs["cookies"] == {"sid": "route-secret", "stored": "kept-stored", "explicit": "kept-explicit"}
        assert kwargs["headers"]["Authorization"] == "private-header"
        assert kwargs["headers"]["X-Route"] == "route-header" and kwargs["headers"]["User-Agent"] == "selected-agent"
        assert contexts[0].source == "enhanced_batch" and contexts[0].user_id == "owner"
    finally:
        release.set()
        await batch.scraper.stop()


@pytest.mark.asyncio
async def test_default_batch_execution_credentials_are_byte_admitted_before_dispatch(batch):
    try:
        report = await batch.scraper._batch_report(
            [BASE + "one"], options={"custom_headers": {"X-Private": "s" * 262144}}
        )
        assert report.outcome.stop_reason == "job_payload_too_large"
        assert batch.calls == []
        assert batch.contexts[0].snapshot().page_attempts == 0
        assert batch.scraper.job_queue.get_status()["payload_bytes"] == 0
    finally:
        await batch.scraper.stop()


@pytest.mark.asyncio
async def test_default_batch_keeps_beautifulsoup_fallback_after_unsuccessful_route_pipeline(batch, tmp_path):
    import yaml

    route_file = tmp_path / "fallback.yaml"
    route_file.write_text(
        yaml.safe_dump(
            {
                "domains": {
                    "93.184.216.34": {
                        "backend": "httpx",
                        "respect_robots": False,
                        "strategy_order": ["regex"],
                    }
                }
            }
        )
    )
    batch.scraper.config["custom_scrapers_yaml_path"] = str(route_file)
    batch.unique = False
    batch.body = "<html><title>Brief</title><main>Plain short text.</main></html>"
    try:
        report = await batch.scraper._batch_report([BASE + "one"], method="beautifulsoup", allow_llm_extraction=False)
        assert report.articles[0]["extraction_successful"]
        assert report.articles[0]["extraction_strategy"] == "beautifulsoup"
        assert "Plain short text." in report.articles[0]["content"]
        assert len(batch.calls) == batch.contexts[0].snapshot().page_attempts == 1
    finally:
        await batch.scraper.stop()


@pytest.mark.asyncio
async def test_public_default_batch_interrupts_with_real_partial_report(batch):
    from tldw_Server_API.app.core.Web_Scraping.contracts.execution import CrawlInterrupted

    batch.limits(network_requests=1)
    try:
        with pytest.raises(CrawlInterrupted) as raised:
            await batch.scraper.scrape_multiple([BASE + "one", BASE + "two"])
        assert raised.value.report.outcome.state == "partial"
        assert raised.value.report.outcome.stop_reason == "budget_exhausted"
        assert len(batch.calls) == 1
        assert any(item.get("content") for item in raised.value.report.articles)
    finally:
        await batch.scraper.stop()


@pytest.mark.asyncio
async def test_default_batch_keeps_instance_routing_and_extraction_capabilities(batch, monkeypatch):
    seen = []
    resolver = batch.scraper._resolve_scrape_plan
    extractor = batch.scraper._extract_from_html_with_pipeline

    def observed_resolve(url):
        seen.append(("route", url))
        return resolver(url)

    def observed_extract(html, url, **kwargs):
        seen.append(("extract", url))
        data = extractor(html, url, **kwargs)
        data["custom_extra"] = {"labels": ["instance"]}
        return data

    monkeypatch.setattr(batch.scraper, "_resolve_scrape_plan", observed_resolve)
    monkeypatch.setattr(batch.scraper, "_extract_from_html_with_pipeline", observed_extract)
    try:
        report = await batch.scraper._batch_report([BASE + "one"], allow_llm_extraction=False)
        assert seen == [("route", BASE + "one"), ("extract", BASE + "one")]
        assert report.articles[0]["custom_extra"] == {"labels": ["instance"]}
        assert batch.contexts[0].snapshot().network_requests == 1
    finally:
        await batch.scraper.stop()


@pytest.mark.asyncio
async def test_public_default_batch_policy_failure_keeps_enhanced_failure_shape(batch):
    try:
        results = await batch.scraper.scrape_multiple(["file:///private/article"])
        assert batch.calls == []
        assert results[0]["extraction_successful"] is False
        assert results[0]["policy_reason"]
        assert not any(key in results[0] for key in ("title", "author", "date", "content"))
    finally:
        await batch.scraper.stop()


@pytest.mark.asyncio
async def test_default_queue_late_replacement_keeps_original_exception_and_submitter_context(batch):
    import contextvars

    from tldw_Server_API.app.core.Web_Scraping.runtime.controls import ParentStop

    marker = contextvars.ContextVar("batch_submitter", default="outside")
    failure = ParentStop("deadline_exhausted")
    cause = ValueError("original cause")
    failure.__cause__ = cause
    seen = []

    async def opaque(job):
        seen.append((job.method, marker.get(), job.metadata.get("custom_cookies")))
        raise failure

    controls = crawl_context.CrawlContext(models.CrawlLimits(), clock=lambda: 0)
    queue = batch.scraper.job_queue
    await queue.start()
    token = marker.set("submitted")
    try:
        handle = await queue.submit_page(enhanced.ScrapingJob("late", BASE + "one", "beautifulsoup"), controls=controls)
    finally:
        marker.reset(token)
    queue._execute_job = opaque
    try:
        with pytest.raises(ParentStop) as raised:
            await handle.result
        await handle.drained
        assert raised.value is failure and failure.__cause__ is cause
        assert seen == [("beautifulsoup", "submitted", None)]
        assert batch.calls == [] and controls.snapshot().page_attempts == 1
    finally:
        await batch.scraper.stop()


@pytest.mark.asyncio
async def test_default_batch_observes_real_parent_transfer_limit(batch):
    batch.limits(transfer_bytes=100)
    try:
        report = await batch.scraper._batch_report([BASE + "one", BASE + "two"])
        assert report.outcome.stop_reason == "budget_exhausted"
        assert len(batch.calls) == batch.contexts[0].snapshot().network_requests == 1
        assert batch.contexts[0].snapshot().page_attempts == 1
        assert not report.outcome.has_partial_results
        assert not any(item.get("content") for item in report.articles)
    finally:
        await batch.scraper.stop()


@pytest.mark.asyncio
async def test_default_batch_charges_one_page_per_actual_attempt_and_keeps_input_order(batch):
    try:
        report = await batch.scraper._batch_report(
            [BASE + "one", BASE + "two"], method="beautifulsoup", user_id="owner"
        )
        assert report.outcome.state == "completed"
        assert [item["url"] for item in report.articles] == [BASE + "one", BASE + "two"]
        assert all(item["extraction_successful"] and item["method"] == "beautifulsoup" for item in report.articles)
        snapshot = batch.contexts[0].snapshot()
        assert snapshot.page_attempts == snapshot.network_requests == len(batch.calls) == 2
        assert snapshot.transfer_bytes > 0
    finally:
        await batch.scraper.stop()
