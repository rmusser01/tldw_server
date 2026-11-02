"""Private execution capabilities are per-entry and live through actual drain."""

import asyncio
import gc
import weakref

import pytest

from tldw_Server_API.app.core.Web_Scraping.crawl.models import CrawlLimits, FrontierEntry
from tldw_Server_API.app.core.Web_Scraping.jobs.models import QueueLimits
from tldw_Server_API.app.core.Web_Scraping.jobs.queue import ProcessLocalJobQueue
from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext
from tldw_Server_API.tests.Web_Scraping.phase5_queue_fakes import Executor, job, page, settle, turn

pytestmark = pytest.mark.asyncio


async def test_execute_binding_is_captured_at_admission_and_isolated_from_default_and_other_crawls():
    default, first, second = Executor(), Executor(), Executor()
    queue = ProcessLocalJobQueue(execute=default)
    a = CrawlContext(CrawlLimits(), clock=lambda: 0)
    b = CrawlContext(CrawlLimits(), clock=lambda: 0)
    one = await queue.submit_page(job("a"), controls=a, execute=first)
    two = await queue.submit_page(job("b"), controls=b, execute=second)
    three = await queue.submit_page(job("c"), controls=a)
    queue._execute = second
    await queue.start()
    for executor in (first, second, default):
        (await executor.started.get()).finish()
    assert await asyncio.gather(one.result, two.result, three.result) == [page(), page(), page()]
    assert [attempt.controls for attempt in first.attempts + second.attempts + default.attempts] == [a, b, a]
    await settle(queue)
    await queue.stop()


async def test_override_retry_reuses_full_queue_reservation_and_same_controls():
    default, selected = Executor(), Executor()
    queue = ProcessLocalJobQueue(execute=default, max_concurrent=1, limits=QueueLimits(max_pending=1, max_live=1))
    context = CrawlContext(CrawlLimits(page_attempts=2), clock=lambda: 0)
    await queue.start()
    handle = await queue.submit_page(job(max_retries=1), controls=context, execute=selected)
    first = await selected.started.get()
    size = queue.get_status()["payload_bytes"]
    first.result.set_result(page(error="retry", extraction_successful=False))
    await turn()
    assert not handle.result.done() and selected.started.empty()
    assert queue.get_status()["pending_reservations"] == 1
    first.drained.set_result(None)
    second = await selected.started.get()
    assert second.payload is first.payload and second.controls is context
    assert queue.get_status()["payload_bytes"] == size
    second.finish()
    assert await handle.result == page()
    await handle.drained
    assert not default.attempts and context.snapshot().page_attempts == 2
    await settle(queue)
    await queue.stop()


@pytest.mark.parametrize("failure", [False, True])
async def test_retained_handle_or_exception_releases_override_graph_only_after_actual_drain(failure):
    executor = Executor()
    context = CrawlContext(CrawlLimits(), clock=lambda: 0)

    class Callback:
        def __call__(self, payload, controls):
            handle = executor(payload, controls)
            if failure:
                try:
                    raise RuntimeError("private failure")
                except RuntimeError as error:
                    handle.result.set_exception(error)
            return handle

    callback = Callback()
    reference = weakref.ref(callback)
    queue = ProcessLocalJobQueue(execute=Executor())
    await queue.start()
    handle = await queue.submit_page(job(), controls=context, execute=callback)
    del callback
    attempt = await executor.started.get()
    if not failure:
        attempt.result.set_result(page())
    delivered = await asyncio.gather(handle.result, return_exceptions=True)
    gc.collect()
    assert reference() is not None
    assert queue.get_status()["payload_bytes"] > 0
    attempt.drained.set_result(None)
    await handle.drained
    await settle(queue)
    gc.collect()
    assert reference() is None
    assert bool(delivered) and handle.result.done()
    await queue.stop()


async def test_queued_page_atomically_attaches_private_work_and_detached_waiter_does_not_cancel_it():
    from tldw_Server_API.app.core.Web_Scraping.orchestration.batch import queued_page

    executor = Executor()
    queue = ProcessLocalJobQueue(execute=Executor(), completed_retention=0)
    context = CrawlContext(CrawlLimits(), clock=lambda: 0)
    await queue.start()
    handle = await queued_page(
        FrontierEntry("https://example.com/a", None, 0, 0, 0),
        context=context,
        queue=queue,
        make_job=lambda url: job(url),
        execute=executor,
    )
    attempt = await executor.started.get()
    waiter = asyncio.ensure_future(asyncio.shield(handle.result))
    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)
    assert attempt.cancellations == 0 and len(context.owned_handles()) == 1
    handle.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handle.result
    attempt.finish()
    await handle.drained
    await settle(queue)
    assert context.owned_handles() == ()
    await queue.stop()


@pytest.mark.parametrize(
    "profile",
    [
        "legacy_recursive",
        "enhanced_recursive",
        "legacy_sitemap",
        "enhanced_sitemap",
        "legacy_url_level",
        "enhanced_url_level",
        "enhanced_batch",
    ],
)
async def test_production_queue_reuses_profile_conversion_order_and_single_terminal_publisher(monkeypatch, profile):
    from tldw_Server_API.app.core.Web_Scraping.orchestration import crawl
    from tldw_Server_API.tests.Web_Scraping.test_phase5_crawl_composition import run_report

    outcomes = [
        {"url": "https://example.com", "links": ["https://example.com/blog/child"]},
        {"url": "https://example.com/blog/child", "content": ""},
    ]
    published = []
    original = CrawlContext.publish

    def publish(self, outcome):
        published.append(outcome)
        return original(self, outcome)

    monkeypatch.setattr(crawl.CrawlContext, "publish", publish)
    reports, executed, budgets = [], [], []
    for queued in (False, True):
        with monkeypatch.context() as scoped:
            report, fixture, _, calls = await run_report(
                scoped,
                profile,
                outcomes,
                production_queue=queued,
                max_pages=3,
                max_depth=2,
                delay=0,
                seeds=[item["url"] for item in outcomes] if profile == "enhanced_batch" else None,
            )
            reports.append(report)
            executed.append(calls)
            budgets.append(fixture.budget.page_attempts)
    assert reports[0] == reports[1]
    assert executed[0] == executed[1] and budgets == [2, 2]
    assert len(published) == 2


async def test_private_normalized_crawl_inputs_cannot_bypass_queue_payload_ceiling(monkeypatch):
    from tldw_Server_API.tests.Web_Scraping.test_phase5_crawl_composition import BASE, run_report

    report, fixture, _, executed = await run_report(
        monkeypatch,
        "enhanced_batch",
        [{"url": BASE}],
        production_queue=True,
        custom_headers={"X-Private": "x" * 270000},
    )
    assert report.outcome.stop_reason == "job_payload_too_large"
    assert not executed and fixture.budget.page_attempts == 0


async def test_pending_owned_cancellation_releases_unexecuted_override_graph():
    callback = Executor()
    reference = weakref.ref(callback)
    queue = ProcessLocalJobQueue(execute=Executor())
    handle = await queue.submit_page(
        job(),
        controls=CrawlContext(CrawlLimits(), clock=lambda: 0),
        execute=callback,
    )
    del callback
    gc.collect()
    assert reference() is not None
    handle.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handle.result
    await handle.drained
    gc.collect()
    assert reference() is None and queue.get_status()["payload_bytes"] == 0
    await queue.stop()


async def test_queued_page_waits_only_for_prior_owned_draining_capacity():
    from tldw_Server_API.app.core.Web_Scraping.orchestration.batch import queued_page

    executor = Executor()
    queue = ProcessLocalJobQueue(execute=executor, max_concurrent=1, limits=QueueLimits(max_pending=1, max_live=1))
    context = CrawlContext(CrawlLimits(), clock=lambda: 0)
    await queue.start()
    first = await queued_page(
        FrontierEntry("first", None, 0, 0, 0),
        context=context,
        queue=queue,
        make_job=job,
    )
    attempt = await executor.started.get()
    attempt.result.set_result(page())
    await first.result
    task = None
    try:
        task = asyncio.create_task(
            queued_page(
                FrontierEntry("second", None, 0, 1, 0),
                context=context,
                queue=queue,
                make_job=job,
                owned=(first,),
            )
        )
        await turn()
        assert not task.done() and queue.get_status()["live_jobs"] == 1
        attempt.drained.set_result(None)
        second = await task
        following = await executor.started.get()
        following.finish()
        assert await second.result == page()
        await second.drained
    finally:
        for item in executor.attempts:
            item.finish()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await settle(queue)
        await queue.stop()


async def test_production_crawl_passes_its_prior_queue_drain_when_next_page_is_overloaded(monkeypatch):
    from tldw_Server_API.app.core.Web_Scraping.jobs import queue as queue_module
    from tldw_Server_API.tests.Web_Scraping.test_phase5_crawl_composition import run_report

    next_attempt = asyncio.Event()
    drains = asyncio.Queue()
    retained_drains = []
    original_type = queue_module.ProcessLocalJobQueue
    queues = []
    first, second = "https://example.com/first", "https://example.com/second"

    def create_queue(**kwargs):
        result = original_type(
            **kwargs,
            max_concurrent=1,
            limits=QueueLimits(max_pending=1, max_live=1),
        )
        submit = result.submit_page

        async def observed(item, **options):
            if item.url == second:
                next_attempt.set()
            return await submit(item, **options)

        monkeypatch.setattr(result, "submit_page", observed)
        queues.append(result)
        return result

    monkeypatch.setattr(queue_module, "ProcessLocalJobQueue", create_queue)
    task = asyncio.create_task(
        run_report(
            monkeypatch,
            "enhanced_batch",
            [{"url": first}, {"url": second}],
            production_queue=True,
            seeds=[first, second],
            page_drains=drains,
        )
    )
    try:
        url, drain = await asyncio.wait_for(drains.get(), 1)
        retained_drains.append(drain)
        assert url == first
        await asyncio.wait_for(next_attempt.wait(), 1)
        await turn()
        assert not task.done() and queues[0].get_status()["live_jobs"] == 1
        drain.set_result(None)
        url, drain = await asyncio.wait_for(drains.get(), 1)
        retained_drains.append(drain)
        assert url == second
        drain.set_result(None)
        report, fixture, _, executed = await task
        assert executed == [first, second]
        assert [item["url"] for item in report.articles] == [first, second]
        assert report.outcome.state == "completed" and fixture.budget.page_attempts == 2
    finally:
        for drain in retained_drains:
            if not drain.done():
                drain.set_result(None)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for queue in queues:
            await settle(queue)
