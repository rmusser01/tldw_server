"""Actual batch and canonical acquisition must bound retained URL identities."""

import asyncio
import time

import pytest

from tldw_Server_API.app.core.Web_Scraping.contracts.execution import DiscoveryResult
from tldw_Server_API.app.core.Web_Scraping.crawl.models import CrawlLimits
from tldw_Server_API.app.core.Web_Scraping.jobs.models import ScrapingJob
from tldw_Server_API.app.core.Web_Scraping.orchestration import article, batch
from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext
from tldw_Server_API.app.core.Web_Scraping.runtime import FetchResponse, RuntimeRequestContext
from tldw_Server_API.app.core.Web_Scraping.runtime.controls import ParentStop, WorkHandle
from tldw_Server_API.tests.Web_Scraping.phase5_queue_fakes import page
from tldw_Server_API.tests.Web_Scraping.test_phase4_article_orchestration import URL, _harness
from tldw_Server_API.tests.Web_Scraping.test_phase5_batch_window import setup_batch

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("method", ["check_url", "resolve_url"])
async def test_url_stop_prevents_already_accepted_queue_sibling_from_starting(method):
    from tldw_Server_API.tests.Web_Scraping.phase5_queue_fakes import job, settle

    _, context, executor, queue = setup_batch(workers=1, limits=CrawlLimits(max_url_bytes=32))
    await queue.start()
    first = await queue.submit_page(job("first"), controls=context)
    attempt = await asyncio.wait_for(executor.started.get(), 1)
    sibling = await queue.submit_page(job("accepted-sibling"), controls=context)
    assert queue.get_status()["live_jobs"] == 2
    try:
        args = ("https://e.test/" + "x" * 33,) if method == "check_url" else ("https://e.test/a/", "x" * 30)
        with pytest.raises(ParentStop, match="budget_exhausted"):
            getattr(context, method)(*args)
        attempt.finish(page(content="first physical return"))
        await first.result
        # A broken owner latch would start and hold a second real queue attempt.
        # Bound that defect path without manufacturing the sibling's exception.
        done, _ = await asyncio.wait({sibling.result}, timeout=0.1)
        assert done, "accepted sibling started despite the parent URL stop"
        with pytest.raises(ParentStop, match="budget_exhausted"):
            await sibling.result
        assert len(executor.attempts) == 1
        assert context.snapshot().page_attempts == 1
    finally:
        for owned in executor.attempts:
            owned.finish()
        await settle(queue)
        await queue.stop()


@pytest.mark.parametrize(
    "url,maximum,allowed",
    [
        ("https://example.com/" + "x" * 70000, 65536, False),
        ("https://e.test/" + "x" * 33, 48, True),
        ("https://e.test/" + "x" * 34, 48, False),
        ("https://e.test/x" + "\u00e9" * 16, 48, True),
        ("https://e.test/x" + "\u00e9" * 17, 48, False),
    ],
)
async def test_actual_batch_bounds_before_job_materialization(url, maximum, allowed):
    plan, context, _, queue = setup_batch(limits=CrawlLimits(max_url_bytes=maximum))
    made, dispatched = [], []

    def execute(payload, controls):
        dispatched.append(payload.url)
        controls.reserve("page_attempts").commit()
        loop = asyncio.get_running_loop()
        result, drained = loop.create_future(), loop.create_future()
        result.set_result(page(url=payload.url))
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: None)

    def make(url):
        made.append(url)
        return ScrapingJob(str(len(made)), url, "auto", max_retries=0)

    await queue.start()
    try:
        report = await batch.execute_batch(
            [url], plan=plan, context=context, queue=queue, make_job=make, execute=execute
        )
    finally:
        await queue.stop()
    assert report.outcome.stop_reason == ("frontier_exhausted" if allowed else "budget_exhausted")
    assert made == dispatched == ([url] if allowed else [])
    assert context.snapshot().discovered_url_bytes == (len(url.encode("utf-8")) if allowed else 0)


async def test_batch_cumulative_url_budget_preserves_duplicates_order_and_partial():
    url = "https://e.test/"
    plan, context, executor, queue = setup_batch(workers=1, limits=CrawlLimits(discovered_url_bytes=len(url) * 2))
    made = []

    def make(value):
        made.append(value)
        return ScrapingJob(str(len(made)), value, "auto", max_retries=0)

    await queue.start()
    task = asyncio.create_task(batch.execute_batch([url] * 3, plan=plan, context=context, queue=queue, make_job=make))
    try:
        for index in range(2):
            attempt = await asyncio.wait_for(executor.started.get(), 1)
            attempt.finish(page(input_index=index))
        report = await asyncio.wait_for(asyncio.shield(task), 1)
        assert report.outcome.stop_reason == "budget_exhausted"
        assert [item["input_index"] for item in report.articles] == [0, 1]
        assert made == [url, url]
        assert context.snapshot().discovered_url_bytes == len(url) * 2
    finally:
        task.cancel()
        for attempt in executor.attempts:
            attempt.finish()
        await asyncio.gather(task, return_exceptions=True)
        await queue.stop()


@pytest.mark.parametrize("boundary", ["initial", "redirect", "relative", "final", "browser_final"])
async def test_controlled_canonical_url_bounds_without_discovery(boundary):
    maximum = 65536 if boundary == "relative" else 48
    initial = "https://example.com/" + "a" * 40000 + "/" if boundary == "relative" else URL
    overflow = "b" * 40000 if boundary == "relative" else "https://example.test/" + "x" * 50
    if boundary == "initial":
        initial = overflow
    response = FetchResponse(
        url=overflow if boundary == "final" else initial,
        status=302 if boundary in {"redirect", "relative"} else 200,
        text="<html>body</html>",
        headers={"Location": overflow} if boundary in {"redirect", "relative"} else {},
        backend="httpx",
    )
    harness = _harness(fetch_outcomes=[response], backend="playwright" if boundary == "browser_final" else "httpx")
    harness.browser.final_url = overflow
    controls = CrawlContext(CrawlLimits(max_url_bytes=maximum), clock=time.monotonic)
    discoveries = []
    handle = article.start_page(
        initial,
        None,
        controls=controls,
        request_context=RuntimeRequestContext(),
        discover=lambda *args: discoveries.append(args) or DiscoveryResult(initial, initial, ()),
        discovery_gate=lambda _result: False,
        dependencies=harness.dependencies,
    )
    try:
        with pytest.raises(ParentStop, match="budget_exhausted"):
            await handle.result
    finally:
        await handle.drained
    assert len(harness.fetch.requests) == (0 if boundary in {"initial", "browser_final"} else 1)
    assert harness.extract.call_count == 0 and discoveries == []


@pytest.mark.parametrize("extra", [0, 1])
async def test_controlled_relative_redirect_exact_trusted_boundary(extra):
    base = "https://example.test/a/"
    reference = "x" * 25 + "y" * extra
    target = base + reference
    harness = _harness(
        fetch_outcomes=[
            FetchResponse(url=base, status=302, text="", headers={"Location": reference}, backend="httpx"),
            FetchResponse(url=target, status=200, text="body", headers={}, backend="httpx"),
        ]
    )
    handle = article.start_page(
        base,
        None,
        controls=CrawlContext(CrawlLimits(max_url_bytes=48), clock=time.monotonic),
        request_context=RuntimeRequestContext(),
        discover=lambda *args: DiscoveryResult(base, base, ()),
        discovery_gate=lambda _result: False,
        dependencies=harness.dependencies,
    )
    try:
        if extra:
            with pytest.raises(ParentStop):
                await handle.result
        else:
            assert (await handle.result).article["extraction_successful"]
    finally:
        await handle.drained
    assert [request.url for request in harness.fetch.requests] == ([base] if extra else [base, target])
