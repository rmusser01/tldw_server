"""Bounded lazy production against the real queue, without network or LLMs."""

import asyncio
import gc
import weakref

import pytest

from tldw_Server_API.app.core.Web_Scraping.contracts.requests import CrawlRequest
from tldw_Server_API.app.core.Web_Scraping.crawl.models import CrawlLimits
from tldw_Server_API.app.core.Web_Scraping.crawl.profiles import resolve_crawl_plan
from tldw_Server_API.app.core.Web_Scraping.jobs.models import QueueLimits, ScrapingJob
from tldw_Server_API.app.core.Web_Scraping.jobs.queue import ProcessLocalJobQueue
from tldw_Server_API.app.core.Web_Scraping.orchestration import batch
from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext
from tldw_Server_API.app.core.Web_Scraping.runtime.controls import ParentStop
from tldw_Server_API.tests.Web_Scraping.phase5_queue_fakes import Executor, job, page, settle, turn

pytestmark = pytest.mark.asyncio


async def test_context_cancel_wakes_collected_batch_drain_wait_into_bounded_cleanup(monkeypatch):
    plan, context, executor, queue = setup_batch(workers=1)
    drain_wait, cleanup_entered = asyncio.Event(), asyncio.Event()
    progress, cleanup = batch._progress, batch._cleanup

    async def observe_progress(children, context):
        if all(child.collected for child in children.values()):
            drain_wait.set()
        return await progress(children, context)

    async def observe_cleanup(*args, **kwargs):
        cleanup_entered.set()
        return await cleanup(*args, **kwargs)

    class Callback:
        def __call__(self, payload, controls):
            return executor(payload, controls)

    monkeypatch.setattr(batch, "_progress", observe_progress)
    monkeypatch.setattr(batch, "_cleanup", observe_cleanup)
    callback = Callback()
    retained = weakref.ref(callback)
    await queue.start()
    task = asyncio.create_task(
        batch.execute_batch(
            ["first", "later"],
            plan=plan,
            context=context,
            queue=queue,
            make_job=lambda url: job(url, max_retries=0),
            execute=callback,
        )
    )
    del callback
    attempt = await started(executor)
    try:
        attempt.result.set_result(page(content="collected before cancel"))
        await asyncio.wait_for(drain_wait.wait(), 1)
        context.cancel()
        await asyncio.wait_for(cleanup_entered.wait(), 1)
        assert not attempt.drained.done() and queue.get_status()["payload_bytes"] > 0
        assert queue.get_status()["live_jobs"] == 1 and retained() is not None
        done, _ = await asyncio.wait((task,), timeout=3)
        assert task in done, "logical cleanup exceeded its aggregate allowance"
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert caught.value.report.articles[0]["content"] == "collected before cancel"
        assert caught.value.report.outcome.stop_reason == "cancelled"
        assert len(executor.attempts) == 1 and attempt.cancellations == 1
        assert not attempt.drained.done() and retained() is not None
        assert not context.stop_signal().cancelled()
    finally:
        task.cancel()
        attempt.finish()
        await asyncio.gather(task, return_exceptions=True)
        await settle(queue)
        await queue.stop()
    gc.collect()
    assert retained() is None


def setup_batch(*, workers=5, limits=None, queue_limits=None):
    limits = limits or CrawlLimits()
    plan = resolve_crawl_plan(
        CrawlRequest("https://example.com"),
        profile="enhanced_batch",
        options={},
        supplied=frozenset(),
        config={},
        limits=limits,
    )
    context = CrawlContext(limits, clock=lambda: 0)
    executor = Executor()
    queue = ProcessLocalJobQueue(execute=executor, max_concurrent=workers, limits=queue_limits or QueueLimits())
    return plan, context, executor, queue


async def started(executor):
    return await asyncio.wait_for(executor.started.get(), 1)


async def test_batch_larger_than_window_fills_before_release_and_preserves_order():
    plan, context, executor, queue = setup_batch()
    pulled, made, completion_order = [], [], []

    def urls():
        for index in range(23):
            pulled.append(index)
            yield f"https://example.com/{index}"

    def make(url):
        made.append(url)
        return ScrapingJob(url, url, "auto", max_retries=0)

    await queue.start()
    task = asyncio.create_task(batch.execute_batch(urls(), plan=plan, context=context, queue=queue, make_job=make))
    try:
        for offset in range(0, 23, 5):
            window = [await started(executor) for _ in range(min(5, 23 - offset))]
            assert len(made) == len(pulled) == min(offset + 5, 23)
            assert queue.get_status()["live_jobs"] == len(window)
            for attempt in window:
                attempt.result.add_done_callback(
                    lambda result: completion_order.append(result.result().article["input_index"])
                )
            for attempt in reversed(window):
                index = int(attempt.payload.url.rsplit("/", 1)[1])
                attempt.finish(page(input_index=index))
        report = await task
        assert [item["input_index"] for item in report.articles] == list(range(23))
        assert completion_order == [4, 3, 2, 1, 0, 9, 8, 7, 6, 5, 14, 13, 12, 11, 10, 19, 18, 17, 16, 15, 22, 21, 20]
        assert report.outcome.state == "completed"
        assert context.snapshot().page_attempts == 23
    finally:
        task.cancel()
        for attempt in executor.attempts:
            attempt.finish()
        await asyncio.gather(task, return_exceptions=True)
        await queue.stop()
        await settle(queue)


async def test_no_owned_progress_overload_is_immediate_and_does_not_wait_for_other_job():
    plan, context, executor, queue = setup_batch(workers=1, queue_limits=QueueLimits(max_pending=1, max_live=1))
    await queue.start()
    other = await queue.add_job(job("unrelated"))
    attempt = await started(executor)
    report = await asyncio.wait_for(
        batch.execute_batch(
            ["https://example.com/owned"], plan=plan, context=context, queue=queue, make_job=lambda url: job(url)
        ),
        1,
    )
    assert report.outcome.stop_reason == "queue_overloaded"
    assert attempt.cancellations == 0
    attempt.finish()
    await other
    await settle(queue)
    await queue.stop()


async def test_own_progress_allows_retry_after_physical_capacity_release():
    plan, context, executor, queue = setup_batch(workers=2, queue_limits=QueueLimits(max_pending=2, max_live=3))
    await queue.start()
    other = await queue.add_job(job("unrelated"))
    unrelated = await started(executor)
    task = asyncio.create_task(
        batch.execute_batch(
            ["https://example.com/0", "https://example.com/1"],
            plan=plan,
            context=context,
            queue=queue,
            make_job=lambda url: ScrapingJob(url, url, "auto"),
        )
    )
    first = await started(executor)
    await turn()
    assert queue.get_status()["live_jobs"] == 2
    first.result.set_result(page(input_index=0))
    await turn()
    assert not task.done() and executor.started.empty()
    first.drained.set_result(None)
    second = await started(executor)
    second.finish(page(input_index=1))
    report = await task
    assert [item["input_index"] for item in report.articles] == [0, 1]
    assert unrelated.cancellations == 0
    unrelated.finish()
    await other
    await settle(queue)
    await queue.stop()


async def test_oversized_next_payload_retains_completed_partial_and_stops_production():
    plan, context, executor, queue = setup_batch(workers=1, queue_limits=QueueLimits(max_payload_bytes=300))
    await queue.start()
    made = []

    def make(url):
        made.append(url)
        return job(str(len(made)), metadata={} if len(made) == 1 else {"value": "x" * 400})

    task = asyncio.create_task(
        batch.execute_batch(
            iter(["first", "oversized", "never"]),
            plan=plan,
            context=context,
            queue=queue,
            make_job=make,
        )
    )
    (await started(executor)).finish(page(content="already completed"))
    report = await task
    assert report.articles[0]["content"] == "already completed"
    assert report.outcome.state == "partial" and report.outcome.stop_reason == "job_payload_too_large"
    assert made == ["first", "oversized"]
    await settle(queue)
    await queue.stop()


@pytest.mark.parametrize("stop", ["deadline", "cancel"])
async def test_partial_abort_cancels_only_owned_and_keeps_drain_charged(stop):
    plan, context, executor, queue = setup_batch(workers=3)
    await queue.start()
    unrelated_future = await queue.add_job(job("unrelated"))
    unrelated = await started(executor)
    task = asyncio.create_task(
        batch.execute_batch(
            ["first", "second"],
            plan=plan,
            context=context,
            queue=queue,
            make_job=lambda url: job(url, max_retries=0),
        )
    )
    first, second = await started(executor), await started(executor)
    first.finish(page(content="already completed"))
    await turn()
    await turn()
    try:
        if stop == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            report = caught.value.report
        else:
            context._clock = lambda: 2000
            second.result.set_result(page(content="late"))
            report = await task
        assert report.articles[0]["content"] == "already completed"
        assert report.outcome.stop_reason == ("cancelled" if stop == "cancel" else "deadline_exhausted")
        assert unrelated.cancellations == 0 and second.cancellations == 1
        assert not second.drained.done() and queue.get_status()["payload_bytes"] > 0
    finally:
        second.finish()
        unrelated.finish()
        await unrelated_future
        await settle(queue)
        await queue.stop()
    assert context.owned_handles() == ()


async def test_remaining_parent_allowance_bounds_materialization_without_double_debit():
    plan, context, executor, queue = setup_batch(limits=CrawlLimits(page_attempts=2))
    await queue.start()
    made = []

    def make(url):
        made.append(url)
        return job(url, max_retries=0)

    task = asyncio.create_task(
        batch.execute_batch(
            (str(i) for i in range(10)),
            plan=plan,
            context=context,
            queue=queue,
            make_job=make,
        )
    )
    attempts = [await started(executor) for _ in range(2)]
    assert made == ["0", "1"]
    for attempt in attempts:
        attempt.finish()
    report = await task
    assert report.outcome.stop_reason == "budget_exhausted" and len(report.articles) == 2
    assert context.snapshot().page_attempts == 2 and made == ["0", "1"]
    await settle(queue)
    await queue.stop()


async def test_failed_and_empty_truthy_success_payloads_retained_but_not_actual_partial_data():
    plan, context, executor, queue = setup_batch(workers=1)
    await queue.start()
    task = asyncio.create_task(
        batch.execute_batch(
            ["failed", "empty", "oversized"],
            plan=plan,
            context=context,
            queue=queue,
            make_job=lambda url: job(
                url, max_retries=0, metadata={"large": "x" * 270000} if url == "oversized" else {}
            ),
        )
    )
    (await started(executor)).finish(page(content="", error="source failure", extraction_successful=False))
    (await started(executor)).finish(page(content="", extraction_successful=1))
    report = await task
    assert len(report.articles) == 2 and report.articles[0]["error"] == "source failure"
    assert report.articles[1]["extraction_successful"] == 1
    assert report.outcome.state == "failed" and not report.outcome.has_partial_results
    await settle(queue)
    await queue.stop()


async def test_explicit_child_cancellation_is_per_item_not_parent_cancellation():
    plan, context, executor, queue = setup_batch(workers=3)
    await queue.start()
    other_future = await queue.add_job(job("other"))
    other = await started(executor)
    task = asyncio.create_task(
        batch.execute_batch(
            ["cancelled-child", "sibling"],
            plan=plan,
            context=context,
            queue=queue,
            make_job=lambda url: job(url, max_retries=0),
        )
    )
    child, sibling = await started(executor), await started(executor)
    try:
        await queue.cancel_job("cancelled-child")
        await turn()
        context.check()
        sibling.finish(page(content="sibling retained"))
        child.drained.set_result(None)
        report = await task
        assert report.outcome.state == "completed"
        assert report.articles[0] == {
            "url": "cancelled-child",
            "error": "Job cancelled",
            "extraction_successful": False,
        }
        assert report.articles[1]["content"] == "sibling retained"
        assert sibling.cancellations == other.cancellations == 0
    finally:
        child.finish()
        sibling.finish()
        other.finish()
        await other_future
        await settle(queue)
        await queue.stop()


async def test_owned_logical_completion_waits_on_its_drain_before_retrying_pressure(monkeypatch):
    plan, context, executor, queue = setup_batch(workers=2, queue_limits=QueueLimits(max_pending=2, max_live=2))
    submissions = []
    original = queue.submit_page

    async def counted(item, **kwargs):
        submissions.append(item.job_id)
        return await original(item, **kwargs)

    monkeypatch.setattr(queue, "submit_page", counted)
    await queue.start()
    other_future = await queue.add_job(job("other"))
    other = await started(executor)
    task = asyncio.create_task(
        batch.execute_batch(
            ["first", "second"],
            plan=plan,
            context=context,
            queue=queue,
            make_job=lambda url: job(url, max_retries=0),
        )
    )
    first = await started(executor)
    try:
        first.result.set_result(page(content="first result"))
        # FIFO turns give already-scheduled result handling a chance to run;
        # no elapsed-time sleep or poll of admission is needed.
        for _ in range(4):
            await turn()
        assert submissions == ["first", "second", "second"]
        for _ in range(4):
            await turn()
        assert submissions == ["first", "second", "second"]
        assert not task.done() and queue.get_status()["live_jobs"] == 2
        first.drained.set_result(None)
        second = await started(executor)
        second.finish(page(content="second result"))
        report = await task
        assert submissions == ["first", "second", "second", "second"]
        assert [item["content"] for item in report.articles] == ["first result", "second result"]
        assert other.cancellations == 0
    finally:
        first.finish()
        for attempt in executor.attempts:
            attempt.finish()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        other.finish()
        await other_future
        await settle(queue)
        await queue.stop()


async def test_empty_batch_and_exact_remaining_allowance_finish_without_false_truncation():
    for count in (0, 2):
        plan, context, executor, queue = setup_batch(limits=CrawlLimits(page_attempts=2))
        await queue.start()
        task = asyncio.create_task(
            batch.execute_batch(
                (str(i) for i in range(count)),
                plan=plan,
                context=context,
                queue=queue,
                make_job=lambda url: job(url, max_retries=0),
            )
        )
        for _ in range(count):
            (await started(executor)).finish()
        report = await task
        assert len(report.articles) == count and report.outcome.state == "completed"
        await settle(queue)
        await queue.stop()


async def test_last_logical_result_returns_with_bounded_cleanup_not_parent_deadline(monkeypatch):
    plan, context, executor, queue = setup_batch(workers=1)
    cap = context.cap_timeout
    calls = []

    def deadline_after_logical_result(requested):
        calls.append(requested)
        if len(calls) > 1:
            raise ParentStop("deadline_exhausted")
        return cap(requested)

    monkeypatch.setattr(context, "cap_timeout", deadline_after_logical_result)
    await queue.start()
    task = asyncio.create_task(
        batch.execute_batch(
            ["only"],
            plan=plan,
            context=context,
            queue=queue,
            make_job=lambda url: job(url, max_retries=0),
        )
    )
    attempt = await started(executor)
    try:
        attempt.result.set_result(page())
        report = await task
        assert report.outcome.state == "completed"
        assert not attempt.drained.done() and queue.get_status()["payload_bytes"] > 0
        assert attempt.cancellations == 0 and len(calls) == 1
    finally:
        attempt.finish()
        await settle(queue)
        await queue.stop()


async def test_retry_consuming_remaining_budget_does_not_drop_already_rejected_next_item():
    plan, context, executor, queue = setup_batch(
        workers=2,
        limits=CrawlLimits(page_attempts=2),
        queue_limits=QueueLimits(max_pending=2, max_live=3),
    )
    await queue.start()
    other_future = await queue.add_job(job("other"))
    other = await started(executor)
    made = []

    def make(url):
        made.append(url)
        return job(url, max_retries=1)

    task = asyncio.create_task(
        batch.execute_batch(
            ["first", "second"],
            plan=plan,
            context=context,
            queue=queue,
            make_job=make,
        )
    )
    first = await started(executor)
    try:
        first.finish(page(error="retry", extraction_successful=False))
        retry = await started(executor)
        retry.finish(page(content="retained retry"))
        report = await task
        assert made == ["first", "second"]
        assert report.outcome.stop_reason == "budget_exhausted" and report.outcome.state == "partial"
        assert [item["content"] for item in report.articles] == ["retained retry"]
        assert context.snapshot().page_attempts == 2 and other.cancellations == 0
    finally:
        for attempt in executor.attempts:
            attempt.finish()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        other.finish()
        await other_future
        await settle(queue)
        await queue.stop()


@pytest.mark.parametrize("cancel_owner", ["task", "context"])
async def test_cancellation_during_final_cleanup_still_propagates_and_keeps_physical_ownership(
    monkeypatch, cancel_owner
):
    plan, context, executor, queue = setup_batch(workers=1)
    entered = asyncio.Event()
    cleanup = batch._cleanup

    async def signal_cleanup(*args, **kwargs):
        entered.set()
        return await cleanup(*args, **kwargs)

    monkeypatch.setattr(batch, "_cleanup", signal_cleanup)
    await queue.start()
    task = asyncio.create_task(
        batch.execute_batch(
            ["only"],
            plan=plan,
            context=context,
            queue=queue,
            make_job=lambda url: job(url, max_retries=0),
        )
    )
    attempt = await started(executor)
    try:
        attempt.result.set_result(page(content="committed before cancel"))
        await asyncio.wait_for(entered.wait(), 1)
        if cancel_owner == "task":
            task.cancel()
        else:
            context.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert caught.value.report.articles[0]["content"] == "committed before cancel"
        assert caught.value.report.outcome.stop_reason == "cancelled"
        assert not attempt.drained.done() and queue.get_status()["live_jobs"] == 1
        assert attempt.cancellations == 1
    finally:
        attempt.finish()
        await settle(queue)
        await queue.stop()
