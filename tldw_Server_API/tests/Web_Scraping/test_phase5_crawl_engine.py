"""The lower engine owns accepted handles, commits and terminal publication."""

import asyncio
import weakref

import pytest

from tldw_Server_API.app.core.Web_Scraping.contracts.execution import CrawlOutcome, DiscoveryResult, PageData
from tldw_Server_API.app.core.Web_Scraping.crawl.engine import run_crawl
from tldw_Server_API.app.core.Web_Scraping.crawl.models import (
    CheckpointScope,
    CheckpointState,
    CrawlHooks,
    CrawlLimits,
    FrontierEntry,
    FrontierSnapshot,
)
from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext
from tldw_Server_API.app.core.Web_Scraping.runtime.controls import ParentStop, WorkHandle
from tldw_Server_API.tests.Web_Scraping.test_phase5_frontier import BASE, plan

pytestmark = pytest.mark.asyncio


class MemoryCheckpoint:
    """Trusted bound scope; storage never supplies authorization."""

    scope = CheckpointScope(owner="fixture", namespace="crawl")

    def __init__(self):
        self.state = None
        self.closed = False
        self.loads = []

    def load(self, *, plan, scope):
        self.loads.append((plan, scope))
        return self.state

    def save(self, state):
        self.state = state

    def clear(self):
        self.state = None

    def close(self):
        self.closed = True


async def test_seed_charged_once_and_page_attempts_belong_to_acquirer():
    resolved = plan()
    context = CrawlContext(resolved.limits, clock=lambda: 0)
    submitted = []

    async def acquire(entry):
        submitted.append(entry.url)
        context.reserve("page_attempts").commit()
        loop = asyncio.get_running_loop()
        result, drained = loop.create_future(), loop.create_future()
        result.set_result(
            PageData(
                {"url": entry.url, "content": "kept", "extraction_successful": True},
                DiscoveryResult(entry.url, entry.url, ()),
            )
        )
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: None)

    report = await run_crawl(resolved, context=context, acquire=acquire)
    assert submitted == [BASE]
    assert report.outcome.stop_reason == "frontier_exhausted"
    assert context.snapshot().page_attempts == 1
    assert context.snapshot().discovered_candidates == 1
    assert context.snapshot().discovered_url_bytes == len(BASE)


async def test_discovery_interruption_leaves_current_entry_unfinished():
    resolved = plan()
    context = CrawlContext(resolved.limits, clock=lambda: 0)
    checkpoint = MemoryCheckpoint()

    async def acquire(_entry):
        loop = asyncio.get_running_loop()
        result, drained = loop.create_future(), loop.create_future()
        result.set_exception(ParentStop("budget_exhausted"))
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: None)

    report = await run_crawl(resolved, context=context, acquire=acquire, checkpoint=checkpoint)
    assert report.outcome.stop_reason == "budget_exhausted"
    assert report.articles == ()
    assert [entry.url for entry in checkpoint.state.frontier.inflight] == [BASE]
    assert checkpoint.state.frontier.completed == ()
    assert checkpoint.loads == [(resolved, checkpoint.scope)]
    assert checkpoint.closed


async def test_caller_cancel_is_not_a_failed_report_and_does_not_cancel_shared_futures():
    resolved = plan()
    context = CrawlContext(resolved.limits, clock=lambda: 0)
    accepted = asyncio.Event()
    loop = asyncio.get_running_loop()
    result, drained = loop.create_future(), loop.create_future()
    cancelled = []

    async def acquire(_entry):
        accepted.set()
        return WorkHandle(result, drained, lambda: cancelled.append(True))

    task = asyncio.create_task(run_crawl(resolved, context=context, acquire=acquire))
    await accepted.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled and not result.cancelled() and not drained.done()
    drained.set_result(None)
    result.cancel()


async def test_no_completed_progress_callback_on_interruption():
    resolved = plan("legacy_recursive", max_pages=3, max_depth=2, delay=0)
    context = CrawlContext(CrawlLimits(), clock=lambda: 0)
    messages = []

    async def acquire(_entry):
        raise ParentStop("deadline_exhausted")

    report = await run_crawl(
        resolved, context=context, acquire=acquire, hooks=CrawlHooks(progress_callback=messages.append)
    )
    assert report.outcome.stop_reason == "deadline_exhausted"
    assert messages == [f"Scraping page 1/3: {BASE}"]


async def test_restoration_preserves_unfinished_order_and_known_cumulative_usage():
    resolved = plan("legacy_recursive", max_pages=3, max_depth=2, delay=0)
    old_context = CrawlContext(resolved.limits, clock=lambda: 0)
    old_context.reserve("page_attempts").commit()
    old_context.reserve("discovered_candidates", 3).commit()
    old_context.reserve("discovered_url_bytes", 100).commit()
    first = FrontierEntry(BASE, None, 0, 0, 0)
    second = FrontierEntry(BASE + "/second", BASE, 1, 1, 0)
    checkpoint = MemoryCheckpoint()
    checkpoint.state = CheckpointState(
        checkpoint.scope,
        resolved,
        FrontierSnapshot((second,), (first,)),
        old_context.snapshot(),
        "execution_failed",
        False,
    )
    submitted = []
    context = CrawlContext(resolved.limits, clock=lambda: 10)

    async def acquire(entry):
        submitted.append(entry.url)
        context.reserve("page_attempts").commit()
        loop = asyncio.get_running_loop()
        result, drained = loop.create_future(), loop.create_future()
        result.set_result(
            PageData(
                {"url": entry.url, "content": "kept", "extraction_successful": True},
                DiscoveryResult(entry.url, entry.url, ()),
            )
        )
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: None)

    report = await run_crawl(resolved, context=context, acquire=acquire, checkpoint=checkpoint)
    assert submitted == [BASE, BASE + "/second"]
    assert len(report.articles) == 2
    assert context.snapshot().page_attempts == 3
    assert context.snapshot().discovered_candidates == 3
    assert context.snapshot().discovered_url_bytes == 100
    assert context.snapshot().accounting_complete is False


@pytest.mark.parametrize("successful", [1, "yes", True])
async def test_truthy_profile_count_is_separate_from_actual_retained_success(successful):
    resolved = plan(max_pages=1)
    context = CrawlContext(resolved.limits, clock=lambda: 0)

    async def acquire(entry):
        loop = asyncio.get_running_loop()
        result, drained = loop.create_future(), loop.create_future()
        result.set_result(
            PageData(
                {"url": entry.url, "content": "", "extraction_successful": successful},
                DiscoveryResult(entry.url, entry.url, ()),
            )
        )
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: None)

    report = await run_crawl(resolved, context=context, acquire=acquire)
    assert report.outcome.stop_reason == "page_cap_reached"
    assert report.articles == ({"url": BASE, "content": "", "extraction_successful": successful},)
    assert not report.outcome.has_partial_results


async def test_trusted_checkpoint_scope_mismatch_prevents_any_execution():
    resolved = plan()
    context = CrawlContext(resolved.limits, clock=lambda: 0)
    checkpoint = MemoryCheckpoint()
    checkpoint.state = CheckpointState(
        CheckpointScope("other-owner", "crawl"), resolved, FrontierSnapshot(), context.snapshot()
    )
    submitted = []

    async def acquire(entry):
        submitted.append(entry.url)
        raise AssertionError("untrusted state must not execute")

    report = await run_crawl(resolved, context=context, acquire=acquire, checkpoint=checkpoint)
    assert report.outcome.stop_reason == "checkpoint_failed"
    assert submitted == []


async def test_finished_crawl_releases_private_graph_promptly_at_physical_drain():
    class Secret:
        pass

    secret = Secret()
    retained = weakref.ref(secret)
    loop = asyncio.get_running_loop()
    result, drained = loop.create_future(), loop.create_future()
    result.set_result(
        PageData({"url": BASE, "content": "kept", "extraction_successful": True}, DiscoveryResult(BASE, BASE, ()))
    )
    context = CrawlContext(CrawlLimits(), clock=lambda: 0)

    async def acquire(_entry, bound=secret):
        return WorkHandle(result, drained, lambda: bound)

    report = await run_crawl(
        plan(), context=context, acquire=acquire, hooks=CrawlHooks(on_progress=lambda _event, bound=secret: bound)
    )
    del secret, acquire
    assert report.outcome.state == "completed" and retained() is not None
    drained.set_result(None)
    await asyncio.sleep(0)
    assert retained() is None


async def test_queued_page_wait_is_bounded_by_parent_deadline_without_cancelling_shared_result():
    import time

    resolved = plan()
    context = CrawlContext(CrawlLimits(execution_seconds=0.02), clock=time.monotonic)
    loop = asyncio.get_running_loop()
    result, drained = loop.create_future(), loop.create_future()
    cancelled = []

    async def acquire(_entry):
        return WorkHandle(result, drained, lambda: cancelled.append(True))

    report = await asyncio.wait_for(run_crawl(resolved, context=context, acquire=acquire), 0.2)
    assert report.outcome.stop_reason == "deadline_exhausted"
    assert cancelled and not result.cancelled()
    drained.set_result(None)
    result.cancel()


async def test_secondary_cleanup_cancellation_does_not_rewrite_budget_stop():
    context = CrawlContext(CrawlLimits(), clock=lambda: 0)

    def cancel():
        raise asyncio.CancelledError("secondary")

    async def acquire(_entry):
        loop = asyncio.get_running_loop()
        result, drained = loop.create_future(), loop.create_future()
        result.set_exception(ParentStop("budget_exhausted"))
        drained.set_result(None)
        return WorkHandle(result, drained, cancel)

    report = await run_crawl(plan(), context=context, acquire=acquire)
    assert report.outcome.stop_reason == "budget_exhausted"


async def test_cancel_during_candidate_policy_keeps_page_and_children_uncommitted():
    from tldw_Server_API.app.core.Web_Scraping.contracts.execution import DiscoveredLink
    from tldw_Server_API.app.core.Web_Scraping.crawl.models import TraversalDecision

    entered = asyncio.Event()
    checkpoint = MemoryCheckpoint()
    context = CrawlContext(CrawlLimits(), clock=lambda: 0)

    class Policy:
        async def evaluate(self, entry, *, phase):
            if phase == "discovery":
                entered.set()
                await asyncio.Event().wait()
            return TraversalDecision()

    async def acquire(entry):
        loop = asyncio.get_running_loop()
        result, drained = loop.create_future(), loop.create_future()
        result.set_result(
            PageData(
                {"url": BASE, "content": "kept", "extraction_successful": True},
                DiscoveryResult(BASE, BASE, (DiscoveredLink(BASE + "/child", BASE),)),
            )
        )
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: None)

    task = asyncio.create_task(
        run_crawl(plan(max_pages=3), context=context, acquire=acquire, checkpoint=checkpoint, policy=Policy())
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert checkpoint.state.frontier.completed == ()
    assert checkpoint.state.frontier.results == ()
    assert [entry.url for entry in checkpoint.state.frontier.inflight] == [BASE]
    assert not context.publish(CrawlOutcome("cancelled", "cancelled", False)).has_partial_results


async def test_live_seed_producer_stops_at_first_candidate_overflow():
    from dataclasses import replace

    pulled, submitted = [], []
    resolved = replace(plan("enhanced_batch"), limits=CrawlLimits(discovered_candidates=2))
    context = CrawlContext(resolved.limits, clock=lambda: 0)

    def seeds():
        for index in range(10):
            pulled.append(index)
            yield BASE + str(index)

    async def acquire(entry):
        submitted.append(entry.url)
        loop = asyncio.get_running_loop()
        result, drained = loop.create_future(), loop.create_future()
        result.set_result(
            PageData(
                {"url": entry.url, "content": "kept", "extraction_successful": True},
                DiscoveryResult(entry.url, entry.url, ()),
            )
        )
        drained.set_result(None)
        return WorkHandle(result, drained, lambda: None)

    report = await run_crawl(resolved, context=context, acquire=acquire, seeds=seeds())
    assert report.outcome.stop_reason == "budget_exhausted" and report.outcome.has_partial_results
    assert pulled == [0, 1, 2] and submitted == [BASE + "0", BASE + "1"]


async def test_multi_input_checkpoint_rejected_without_consuming_live_input():
    pulled = []

    def seeds():
        pulled.append(True)
        yield BASE

    async def acquire(_entry):
        raise AssertionError("unsupported continuation cannot execute")

    checkpoint = MemoryCheckpoint()
    report = await run_crawl(
        plan("enhanced_batch"),
        context=CrawlContext(CrawlLimits(), clock=lambda: 0),
        acquire=acquire,
        checkpoint=checkpoint,
        seeds=seeds(),
    )
    assert report.outcome.stop_reason == "checkpoint_failed" and not pulled
    assert checkpoint.state is None and checkpoint.closed


async def test_cancel_during_seed_policy_preserves_seed_for_continuation():
    from tldw_Server_API.app.core.Web_Scraping.crawl.models import TraversalDecision

    entered = asyncio.Event()
    checkpoint = MemoryCheckpoint()
    context = CrawlContext(CrawlLimits(), clock=lambda: 0)
    submitted = []

    class Policy:
        async def evaluate(self, entry, *, phase):
            if phase == "seed":
                entered.set()
                await asyncio.Event().wait()
            return TraversalDecision()

    async def acquire(entry):
        submitted.append(entry.url)
        raise AssertionError("seed policy has not completed")

    task = asyncio.create_task(
        run_crawl(plan(), context=context, acquire=acquire, checkpoint=checkpoint, policy=Policy())
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert submitted == []
    assert [entry.url for entry in checkpoint.state.frontier.queued] == [BASE]
    assert checkpoint.state.frontier.completed == ()


async def test_real_successful_acquisition_interrupted_in_discovery_keeps_only_prior_commits(monkeypatch):
    from dataclasses import replace

    from tldw_Server_API.app.core.Web_Scraping.contracts.requests import CrawlRequest
    from tldw_Server_API.app.core.Web_Scraping.crawl.discovery import discover
    from tldw_Server_API.app.core.Web_Scraping.orchestration import article
    from tldw_Server_API.app.core.Web_Scraping.runtime import FetchResponse, RuntimeRequestContext
    from tldw_Server_API.tests.Web_Scraping.test_phase4_article_orchestration import URL, _harness

    child = URL + "/next"
    resolved = replace(plan(max_pages=3), request=CrawlRequest(URL), limits=CrawlLimits(discovered_candidates=2))
    context = CrawlContext(resolved.limits, clock=lambda: 0)
    fixture = _harness(
        fetch_outcomes=[
            FetchResponse(url=URL, status=200, text=f'<a href="{child}">next</a>'),
            FetchResponse(url=child, status=200, text='<a href="/other">other</a>'),
        ],
        extract_results=[
            {"url": URL, "content": "first", "summary": "kept summary", "extraction_successful": True},
            {"url": child, "content": "unfinished", "extraction_successful": True},
        ],
    )
    monkeypatch.setattr(article, "_build_default_dependencies", lambda *_args: fixture.dependencies)
    checkpoint = MemoryCheckpoint()
    handles = []

    async def acquire(entry):
        handle = article.start_page(
            entry.url,
            None,
            controls=context,
            request_context=RuntimeRequestContext(),
            discovery_on_success=True,
            discover=lambda document, request, final: discover(
                document, request_url=request, final_url=final, plan=resolved, controls=context
            ),
        )
        handles.append(handle)
        return handle

    report = await run_crawl(resolved, context=context, acquire=acquire, checkpoint=checkpoint)
    assert report.outcome.stop_reason == "budget_exhausted" and report.outcome.has_partial_results
    assert [item["url"] for item in report.articles] == [URL]
    assert report.articles[0]["summary"] == "kept summary"
    assert [item.url for item in checkpoint.state.frontier.inflight] == [child]
    assert [item.url for item in checkpoint.state.frontier.completed] == [URL]
    assert [request.url for request in fixture.fetch.requests] == [URL, child]
    assert fixture.extract.call_count == 2
    await asyncio.gather(*(asyncio.shield(handle.drained) for handle in handles))
