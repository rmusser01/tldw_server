"""Owned completion, retries, cancellation, stop and independent physical drain."""

import asyncio
import builtins
import contextvars
import gc
import weakref

import pytest

from tldw_Server_API.app.core.Web_Scraping.crawl.models import CrawlLimits
from tldw_Server_API.app.core.Web_Scraping.jobs.admission import QueueRejected
from tldw_Server_API.app.core.Web_Scraping.jobs.models import JobPriority, JobStatus, QueueLimits
from tldw_Server_API.app.core.Web_Scraping.jobs.queue import ProcessLocalJobQueue
from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext
from tldw_Server_API.app.core.Web_Scraping.runtime.controls import ParentStop, WorkHandle
from tldw_Server_API.tests.Web_Scraping.phase5_queue_fakes import Executor, job, page, settle, turn

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


async def test_priority_order_stable_and_start_idempotent():
    executor = Executor()
    q = ProcessLocalJobQueue(execute=executor, max_concurrent=1)
    priorities = [JobPriority.LOW, JobPriority.HIGH, JobPriority.NORMAL, JobPriority.CRITICAL, JobPriority.HIGH]
    futures = [await q.add_job(job(str(i), priority=p)) for i, p in enumerate(priorities)]
    await q.start()
    await q.start()
    for expected in ("3", "1", "4", "2", "0"):
        attempt = await executor.started.get()
        assert attempt.payload.job_id == expected
        attempt.finish()
    await asyncio.gather(*futures)
    await settle(q)
    assert q.get_status()["live_jobs"] == 0
    assert (await q.stop()).state == "stopped"


async def test_detached_waiter_does_not_cancel_work():
    executor = Executor()
    q = ProcessLocalJobQueue(execute=executor)
    await q.start()
    future = await q.add_job(job())
    attempt = await executor.started.get()
    future.cancel()
    attempt.finish()
    await settle(q)
    assert q.get_job("job").status is JobStatus.COMPLETED
    assert attempt.cancellations == 0


async def test_polling_timeout_is_detached_but_execution_timeout_terminalizes():
    executor = Executor()
    q = ProcessLocalJobQueue(execute=executor)
    await q.start()
    future = await q.add_job(job("poll"))
    attempt = await executor.started.get()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(future, 0)
    attempt.finish()
    await settle(q)
    timed = await q.add_job(job("execution"))
    second = await executor.started.get()
    second.result.set_exception(TimeoutError("execution timed out"))
    with pytest.raises(TimeoutError):
        await timed
    assert q.get_job("execution").status is JobStatus.FAILED
    assert q.get_job("poll").status is JobStatus.COMPLETED
    assert len(executor.attempts) == 2
    second.drained.set_result(None)
    await settle(q)
    await q.stop()


@pytest.mark.parametrize("action", ["cancel", "stop", "result"])
async def test_draining_job_holds_live_payload_and_execution_capacity(action):
    executor = Executor()
    q = ProcessLocalJobQueue(execute=executor, max_concurrent=1, limits=QueueLimits(max_pending=2, max_live=2))
    await q.start()
    future = await q.add_job(job("a"))
    attempt = await executor.started.get()
    following = await q.add_job(job("b"))
    if action == "cancel":
        await q.cancel_job("a")
    elif action == "stop":
        report = await q.stop()
        assert report.state == "stopping" and report.draining_jobs == 1
    else:
        attempt.result.set_result(page())
        await future
    assert q.get_status()["payload_bytes"] > 0
    assert q.get_status()["live_jobs"] >= 1
    assert executor.started.empty()
    if action != "stop":
        with pytest.raises(QueueRejected):
            await q.add_job(job("c"))
    if not attempt.result.done():
        attempt.result.set_result(page())
    attempt.drained.set_result(None)
    if action != "stop":
        next_attempt = await executor.started.get()
        next_attempt.finish()
    await asyncio.gather(future, following, return_exceptions=True)
    await settle(q)
    assert q.get_status()["live_jobs"] == q.get_status()["payload_bytes"] == 0
    assert (await q.stop()).state == "stopped"


async def test_full_queue_retry_stop_interleaving():
    executor = Executor()
    q = ProcessLocalJobQueue(execute=executor, max_concurrent=2, limits=QueueLimits(max_pending=2, max_live=3))
    await q.start()
    futures = [await q.add_job(job(str(i))) for i in range(2)]
    attempts = [await executor.started.get() for _ in range(2)]
    counts = [0, 0]
    for i, future in enumerate(futures):
        future.add_done_callback(lambda _f, i=i: counts.__setitem__(i, counts[i] + 1))
    for attempt in attempts:
        attempt.result.set_result(page(error="retry me", extraction_successful=False))
    await turn()
    with pytest.raises(QueueRejected, match="queue_overloaded"):
        await q.add_job(job("competing"))
    report = await q.stop()
    assert report.state == "stopping" and report.draining_jobs == 2
    for attempt in attempts:
        attempt.drained.set_result(None)
    await asyncio.gather(*futures, return_exceptions=True)
    await settle(q)
    assert counts == [1, 1]
    assert len(executor.attempts) == 2
    assert q.get_status()["pending_reservations"] == 0
    assert not q._heap and not q._entries and not q._tasks


async def test_retry_only_after_drain_reuses_payload_and_reservation():
    executor = Executor()
    q = ProcessLocalJobQueue(execute=executor, max_concurrent=1, limits=QueueLimits(max_pending=1, max_live=2))
    await q.start()
    future = await q.add_job(job())
    first = await executor.started.get()
    original_size = q.get_status()["payload_bytes"]
    first.result.set_result(page(error="failed", extraction_successful=False))
    await turn()
    assert executor.started.empty() and not future.done()
    with pytest.raises(QueueRejected):
        await q.add_job(job("other"))
    first.drained.set_result(None)
    second = await executor.started.get()
    assert second.payload is first.payload
    assert q.get_status()["payload_bytes"] == original_size
    second.finish()
    assert await future == page().article
    await settle(q)
    assert q.get_job("job").retries == 1
    await q.stop()


async def test_default_retry_count_and_failed_result_convention():
    executor = Executor()
    q = ProcessLocalJobQueue(execute=executor)
    await q.start()
    future = await q.add_job(job())
    failed = page(extraction_successful=False)
    for _ in range(4):
        attempt = await executor.started.get()
        attempt.finish(failed)
    with pytest.raises(Exception, match="Source extraction failed"):
        await future
    await settle(q)
    retained = q.get_job("job")
    assert retained.status is JobStatus.FAILED and retained.retries == 3
    assert retained.result == failed.article
    await q.stop()


@pytest.mark.parametrize("winner", ["cancel", "result"])
async def test_cancel_result_race_terminalizes_once(winner):
    executor = Executor()
    q = ProcessLocalJobQueue(execute=executor)
    await q.start()
    future = await q.add_job(job())
    attempt = await executor.started.get()
    if winner == "cancel":
        await q.cancel_job("job")
        attempt.result.set_result(page())
        with pytest.raises(Exception, match="Job cancelled"):
            await future
    else:
        attempt.result.set_result(page())
        await future
        await q.cancel_job("job")
    before = q.get_job("job").completed_at
    await q.cancel_job("job")
    attempt.drained.set_result(None)
    await settle(q)
    assert q.get_job("job").completed_at == before
    assert q.get_job("job").status is (JobStatus.CANCELLED if winner == "cancel" else JobStatus.COMPLETED)
    await q.stop()


async def test_submit_page_subscription_is_atomic_and_cancellation_owns_only_one_child():
    executor = Executor()
    q = ProcessLocalJobQueue(execute=executor, completed_retention=0)
    controls = CrawlContext(CrawlLimits(), clock=lambda: 0)
    await q.start()
    handle = await q.submit_page(job("owned"), controls=controls)
    other = await q.add_job(job("other"))
    a, b = await executor.started.get(), await executor.started.get()
    handle.cancel()
    await turn()
    with pytest.raises(asyncio.CancelledError):
        await handle.result
    assert a.controls is controls and b.controls is None
    assert a.cancellations == 1 and b.cancellations == 0
    assert not handle.drained.done()
    a.finish()
    b.finish()
    await handle.drained
    await other
    await settle(q)
    assert q.get_job("owned") is None
    await q.stop()


async def test_fast_page_delivery_with_zero_history():
    executor = Executor()

    def immediate(payload, controls):
        handle = executor(payload, controls)
        executor.attempts[-1].finish()
        return handle

    q = ProcessLocalJobQueue(execute=immediate, completed_retention=0)
    await q.start()
    handle = await q.submit_page(job(), controls=CrawlContext(CrawlLimits(), clock=lambda: 0))
    assert await handle.result == page()
    await handle.drained
    assert q.get_job("job") is None
    assert executor.attempts[0].controls.snapshot().page_attempts == 1
    await settle(q)
    await q.stop()


@pytest.mark.parametrize("stop", ["cancel", "deadline", "budget"])
async def test_parent_stop_before_attempt_never_executes_or_retries(stop):
    now = [0]
    controls = CrawlContext(CrawlLimits(page_attempts=1, execution_seconds=1), clock=lambda: now[0])
    executor = Executor()
    q = ProcessLocalJobQueue(execute=executor)
    handle = await q.submit_page(job(), controls=controls)
    if stop == "cancel":
        controls.cancel()
    elif stop == "deadline":
        now[0] = 1
    else:
        controls.reserve("page_attempts").commit()
    await q.start()
    with pytest.raises(asyncio.CancelledError if stop == "cancel" else ParentStop):
        await handle.result
    await handle.drained
    assert not executor.attempts
    await settle(q)
    await q.stop()


async def test_retry_checks_parent_and_charges_every_attempt():
    executor = Executor()
    controls = CrawlContext(CrawlLimits(page_attempts=1), clock=lambda: 0)
    q = ProcessLocalJobQueue(execute=executor)
    await q.start()
    handle = await q.submit_page(job(), controls=controls)
    attempt = await executor.started.get()
    attempt.finish(page(error="failure"))
    with pytest.raises(ParentStop, match="budget_exhausted"):
        await handle.result
    await handle.drained
    assert len(executor.attempts) == 1 and controls.snapshot().page_attempts == 1
    await settle(q)
    await q.stop()


@pytest.mark.parametrize("attempt_limit", [1, 2])
async def test_canonical_executor_is_the_only_attempt_accounting_owner(attempt_limit):
    executor = Executor()
    controls = CrawlContext(CrawlLimits(page_attempts=attempt_limit), clock=lambda: 0)

    def start_page(payload, parent):
        handle = executor(payload, parent)
        value = page(error="retry") if len(executor.attempts) < attempt_limit else page()
        executor.attempts[-1].finish(value)
        return handle

    q = ProcessLocalJobQueue(execute=start_page)
    await q.start()
    handle = await q.submit_page(job(), controls=controls)
    delivered = await asyncio.gather(handle.result, return_exceptions=True)
    await handle.drained
    await settle(q)
    await q.stop()
    assert delivered == [page()]
    assert len(executor.attempts) == attempt_limit
    assert controls.snapshot().page_attempts == attempt_limit


async def test_submit_stop_order_rejects_later_and_settles_prior():
    q = ProcessLocalJobQueue(execute=Executor())
    future = await q.add_job(job())
    assert (await q.stop()).state == "stopped"
    with pytest.raises(Exception, match="shutdown"):
        await future
    with pytest.raises(QueueRejected, match="queue_stopping"):
        await q.add_job(job("later"))
    await q.start()
    assert q.get_status()["state"] == "stopped"


async def test_submitter_context_dropped_at_terminal():
    marker = contextvars.ContextVar("marker", default="worker")
    seen = []
    executor = Executor()

    def execute(payload, controls):
        seen.append(marker.get())
        return executor(payload, controls)

    q = ProcessLocalJobQueue(execute=execute)
    await q.start()
    marker.set("submitter")
    future = await q.add_job(job())
    marker.set("later")
    attempt = await executor.started.get()
    attempt.finish()
    await future
    await settle(q)
    assert seen == ["submitter"] and q.get_job("job").context is None
    await q.stop()


async def test_owned_cancel_drain_can_finish_without_executor_logical_result():
    executor = Executor()
    q = ProcessLocalJobQueue(execute=executor)
    await q.start()
    handle = await q.submit_page(job(), controls=CrawlContext(CrawlLimits(), clock=lambda: 0))
    attempt = await executor.started.get()
    handle.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handle.result
    assert not handle.drained.done()
    attempt.drained.set_result(None)
    await asyncio.wait_for(asyncio.shield(handle.drained), 1)
    assert not attempt.result.done()
    assert q.get_status()["live_jobs"] == q.get_status()["payload_bytes"] == 0
    attempt.result.set_exception(RuntimeError("late private failure"))
    await settle(q)
    await q.stop()


async def test_detached_page_subscription_keeps_its_execution_owned():
    executor = Executor()
    q = ProcessLocalJobQueue(execute=executor)
    await q.start()
    handle = await q.submit_page(job(), controls=CrawlContext(CrawlLimits(), clock=lambda: 0))
    attempt = await executor.started.get()
    handle.result.cancel()
    handle.drained.cancel()
    attempt.finish()
    await settle(q)
    assert q.get_job("job").status is JobStatus.COMPLETED
    assert attempt.cancellations == 0
    await q.stop()


@pytest.mark.parametrize(
    "error", [ParentStop("deadline_exhausted"), ParentStop("budget_exhausted"), RuntimeError("refused")]
)
async def test_execution_exception_preserved_without_retry(error):
    executor = Executor()
    q = ProcessLocalJobQueue(execute=executor)
    await q.start()
    handle = await q.submit_page(job(), controls=CrawlContext(CrawlLimits(), clock=lambda: 0))
    attempt = await executor.started.get()
    attempt.result.set_exception(error)
    with pytest.raises(type(error)) as caught:
        await handle.result
    assert caught.value is error
    assert q.get_job("job").status is JobStatus.FAILED
    assert q.get_job("job").retries == 0
    attempt.drained.set_result(None)
    await handle.drained
    await settle(q)
    await q.stop()


async def test_stale_owned_cancel_cannot_target_reused_id():
    executor = Executor()
    q = ProcessLocalJobQueue(execute=executor)
    await q.start()
    handle = await q.submit_page(job(), controls=CrawlContext(CrawlLimits(), clock=lambda: 0))
    first = await executor.started.get()
    first.finish()
    await handle.result
    await handle.drained
    await settle(q)
    second = await q.add_job(job())
    attempt = await executor.started.get()
    handle.cancel()
    await turn()
    assert q.get_job("job").status is JobStatus.IN_PROGRESS
    assert attempt.cancellations == 0
    attempt.finish()
    await second
    await settle(q)
    await q.stop()


async def test_failing_cancel_callback_does_not_block_other_owned_cancellations():
    from loguru import logger

    executor = Executor()

    def execute(payload, controls):
        raw = executor(payload, controls)
        if payload.job_id == "a":

            def fail():
                raise RuntimeError("private-credential")

            return type(raw)(raw.result, raw.drained, fail)
        return raw

    q = ProcessLocalJobQueue(execute=execute)
    await q.start()
    futures = [await q.add_job(job(name)) for name in ("a", "b")]
    attempts = [await executor.started.get() for _ in range(2)]
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), format="{message}")
    try:
        await q.stop()
    finally:
        logger.remove(sink)
    assert messages == ["queue_cancel_callback_failed\n"]
    assert attempts[1].cancellations == 1
    for attempt in attempts:
        attempt.finish()
    await asyncio.gather(*futures, return_exceptions=True)
    await settle(q)
    assert (await q.stop()).state == "stopped"


async def test_failure_delivery_does_not_retain_controls_via_traceback_after_drain():
    loop = asyncio.get_running_loop()
    result, drained = loop.create_future(), loop.create_future()
    references = []

    def execute(payload, controls):
        references.extend((weakref.ref(payload), weakref.ref(controls)))
        return WorkHandle(result, drained, lambda: None)

    q = ProcessLocalJobQueue(execute=execute)
    controls = CrawlContext(CrawlLimits(), clock=lambda: 0)
    await q.start()
    handle = await q.submit_page(job(metadata={"custom_cookies": {"session": "secret"}}), controls=controls)
    del controls
    await turn()
    result.set_exception(RuntimeError("execution failed"))
    delivered = await asyncio.gather(handle.result, return_exceptions=True)
    assert isinstance(delivered[0], RuntimeError)
    gc.collect()
    assert all(reference() is not None for reference in references)
    drained.set_result(None)
    await handle.drained
    await settle(q)
    gc.collect()
    assert all(reference() is None for reference in references)
    await q.stop()


@pytest.mark.parametrize("chain", ["explicit_cause", "implicit_context", "cycle", "nested_group"])
async def test_exception_graph_releases_payload_and_controls_after_drain(chain):
    references, originals = [], []

    def failed_acquisition(payload, controls):
        try:
            raise ValueError("private provider cause")
        except ValueError as error:
            return error

    def execute(payload, controls):
        references.extend((weakref.ref(payload), weakref.ref(controls)))
        inner = failed_acquisition(payload, controls)
        if chain == "nested_group":
            nested = builtins.ExceptionGroup("nested failure", [inner])
            outer = builtins.ExceptionGroup("outer failure", [nested])
            originals.extend((outer, inner, nested))
            raise outer
        outer = RuntimeError("outer failure")
        originals.extend((outer, inner))
        if chain == "cycle":
            outer.__cause__ = inner
            inner.__context__ = outer
            raise outer
        try:
            raise inner
        except ValueError:
            if chain == "explicit_cause":
                raise outer from inner
            raise outer  # noqa: B904 - deliberately exercise implicit exception context

    q = ProcessLocalJobQueue(execute=execute)
    controls = CrawlContext(CrawlLimits(), clock=lambda: 0)
    await q.start()
    handle = await q.submit_page(job(metadata={"custom_cookies": {"session": "private-cookie"}}), controls=controls)
    del controls
    delivered = (await asyncio.gather(handle.result, return_exceptions=True))[0]
    await handle.drained
    await settle(q)
    await q.stop()
    gc.collect()
    assert delivered is originals[0]
    assert type(delivered) is (builtins.ExceptionGroup if chain == "nested_group" else RuntimeError)
    assert str(delivered) == ("outer failure (1 sub-exception)" if chain == "nested_group" else "outer failure")
    assert str(originals[1]) == "private provider cause"
    if chain == "nested_group":
        assert delivered.exceptions[0] is originals[2]
        assert originals[2].exceptions[0] is originals[1]
    elif chain == "implicit_context":
        assert delivered.__context__ is originals[1]
    else:
        assert delivered.__cause__ is originals[1]
        if chain == "cycle":
            assert originals[1].__context__ is delivered
    assert q.get_status()["live_jobs"] == q.get_status()["payload_bytes"] == 0
    assert all(reference() is None for reference in references)
    assert all(error.__traceback__ is None for error in originals)


async def test_initial_cancel_requested_preserves_legacy_no_execution_behavior():
    executor = Executor()
    q = ProcessLocalJobQueue(execute=executor)
    future = await q.add_job(job(cancel_requested=True))
    await q.start()
    try:
        with pytest.raises(Exception, match="Job cancelled"):
            await asyncio.wait_for(asyncio.shield(future), 1)
        assert executor.started.empty()
        assert q.get_job("job").status is JobStatus.CANCELLED
        assert q.get_status()["live_jobs"] == 0
    finally:
        for attempt in executor.attempts:
            attempt.finish()
        await q.stop()
        await settle(q)
