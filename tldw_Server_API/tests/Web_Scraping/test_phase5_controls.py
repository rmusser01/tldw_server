"""Atomic parent accounting and owned logical/physical completion controls."""

import asyncio
import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest
from hypothesis import given
from hypothesis import strategies as st
from loguru import logger

from tldw_Server_API.app.core.Web_Scraping.contracts.execution import CrawlOutcome
from tldw_Server_API.app.core.Web_Scraping.crawl.models import CrawlLimits
from tldw_Server_API.app.core.Web_Scraping.orchestration.crawl_context import CrawlContext
from tldw_Server_API.app.core.Web_Scraping.runtime.controls import ParentStop, WorkHandle

pytestmark = pytest.mark.unit

COUNTERS = (
    "page_attempts",
    "network_requests",
    "browser_attempts",
    "active_probes",
    "transfer_bytes",
    "discovered_candidates",
    "discovered_url_bytes",
)


@pytest.fixture
def ctx():
    return CrawlContext(CrawlLimits(), clock=lambda: 0.0)


@pytest.mark.parametrize("counter", COUNTERS)
def test_started_work_not_refunded(ctx, counter):
    lease = ctx.reserve(counter)
    lease.commit()
    lease.commit()
    lease.close()
    lease.close()
    assert getattr(ctx.snapshot(), counter) == 1


@pytest.mark.parametrize("counter", COUNTERS)
def test_prestart_rollback_once(ctx, counter):
    lease = ctx.reserve(counter, 2)
    lease.close()
    lease.close()
    with pytest.raises(RuntimeError):
        lease.commit()
    assert getattr(ctx.snapshot(), counter) == 0


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.5, float("inf"), float("nan"), "1"])
def test_invalid_reservation_amount(ctx, bad):
    with pytest.raises(ValueError):
        ctx.reserve("page_attempts", bad)


def test_unknown_counter_rejected(ctx):
    with pytest.raises(ValueError):
        ctx.reserve("frontier_entries")


@pytest.mark.parametrize("counter", COUNTERS)
def test_exact_boundary_then_overflow_latches(counter):
    ctx = CrawlContext(replace(CrawlLimits(), **{counter: 2}), clock=lambda: 0)
    lease = ctx.reserve(counter, 2)
    lease.commit()
    with pytest.raises(ParentStop, match="budget_exhausted"):
        ctx.reserve(counter)
    with pytest.raises(ParentStop, match="budget_exhausted"):
        ctx.check()
    assert getattr(ctx.snapshot(), counter) == 2


def test_byte_overflow_not_retained_and_stop_latched():
    ctx = CrawlContext(CrawlLimits(transfer_bytes=3), clock=lambda: 0)
    ctx.charge_bytes(3)
    with pytest.raises(ParentStop):
        ctx.charge_bytes(1)
    assert ctx.snapshot().transfer_bytes == 3
    with pytest.raises(ParentStop):
        ctx.reserve("page_attempts")


def test_timeout_capped_and_deadline_is_inclusive():
    now = [10.0]
    ctx = CrawlContext(CrawlLimits(execution_seconds=2), clock=lambda: now[0])
    now[0] = 11
    assert ctx.cap_timeout(None) == 1
    assert ctx.cap_timeout(0.5) == 0.5
    assert ctx.cap_timeout(10) == 1
    assert ctx.snapshot().remaining_seconds == 1
    now[0] = 12
    with pytest.raises(ParentStop, match="deadline_exhausted"):
        ctx.check()
    assert ctx.snapshot().remaining_seconds == 0


@pytest.mark.parametrize("bad", [True, 0, -1, float("inf"), float("nan")])
def test_invalid_timeout_rejected(ctx, bad):
    with pytest.raises(ValueError):
        ctx.cap_timeout(bad)


def test_cancel_prevents_start_but_allows_unused_rollback(ctx):
    lease = ctx.reserve("page_attempts")
    ctx.cancel()
    with pytest.raises(asyncio.CancelledError):
        lease.commit()
    lease.close()
    assert ctx.snapshot().page_attempts == 0
    with pytest.raises(asyncio.CancelledError):
        ctx.charge_bytes(1)


def test_terminal_publication_immutable(ctx):
    completed = CrawlOutcome("completed", "frontier_exhausted", False)
    assert ctx.publish(completed) is completed
    ctx.cancel()
    assert ctx.publish(CrawlOutcome("failed", "execution_failed", False)) is completed


def test_cancel_before_publish_honored(ctx):
    ctx.cancel()
    assert ctx.publish(CrawlOutcome("partial", "budget_exhausted", True)) == CrawlOutcome(
        "cancelled", "cancelled", True
    )


def test_budget_stop_cannot_publish_success():
    ctx = CrawlContext(CrawlLimits(transfer_bytes=1), clock=lambda: 0)
    with pytest.raises(ParentStop):
        ctx.charge_bytes(2)
    assert ctx.publish(CrawlOutcome("completed", "frontier_exhausted", False)) == CrawlOutcome(
        "failed", "budget_exhausted", False
    )


def test_successful_result_then_cancel_before_completed_publication(ctx):
    ctx.note_successful_result()
    ctx.cancel()
    assert ctx.publish(CrawlOutcome("completed", "frontier_exhausted", False)) == CrawlOutcome(
        "cancelled", "cancelled", True
    )


@pytest.mark.parametrize("stop", ["cancelled", "budget_exhausted", "deadline_exhausted"])
@pytest.mark.parametrize("successful", [False, True])
def test_recorded_presence_when_stop_replaces_completed_candidate(stop, successful):
    now = [0.0]
    ctx = CrawlContext(CrawlLimits(transfer_bytes=1, execution_seconds=1), clock=lambda: now[0])
    # Failed page attempts consume budget but never mark successful result presence.
    lease = ctx.reserve("page_attempts")
    lease.commit()
    if successful:
        ctx.note_successful_result()
    if stop == "cancelled":
        ctx.cancel()
    elif stop == "budget_exhausted":
        with pytest.raises(ParentStop, match=stop):
            ctx.charge_bytes(2)
    else:
        now[0] = 1.0
        with pytest.raises(ParentStop, match=stop):
            ctx.check()
    state = "cancelled" if stop == "cancelled" else ("partial" if successful else "failed")
    assert ctx.publish(CrawlOutcome("completed", "page_cap_reached", False)) == CrawlOutcome(state, stop, successful)


def test_successful_presence_is_not_a_counter_or_partial_completed_outcome(ctx):
    before = ctx.snapshot()
    ctx.note_successful_result()
    ctx.note_successful_result()
    assert ctx.snapshot() == before
    completed = CrawlOutcome("completed", "page_cap_reached", False)
    assert ctx.publish(completed) is completed
    ctx.cancel()
    assert ctx.publish(completed) is completed


@pytest.mark.parametrize("stop", ["cancelled", "budget_exhausted", "deadline_exhausted", "published"])
def test_late_success_mark_rejected_without_manufacturing_partial_presence(stop):
    now = [0.0]
    ctx = CrawlContext(CrawlLimits(transfer_bytes=1, execution_seconds=1), clock=lambda: now[0])
    completed = CrawlOutcome("completed", "frontier_exhausted", False)
    if stop == "cancelled":
        ctx.cancel()
        error = asyncio.CancelledError
        expected = CrawlOutcome("cancelled", "cancelled", False)
    elif stop == "budget_exhausted":
        with pytest.raises(ParentStop):
            ctx.charge_bytes(2)
        error = ParentStop
        expected = CrawlOutcome("failed", stop, False)
    elif stop == "deadline_exhausted":
        now[0] = 1.0
        error = ParentStop
        expected = CrawlOutcome("failed", stop, False)
    else:
        ctx.publish(completed)
        error = RuntimeError
        expected = completed
    with pytest.raises(error):
        ctx.note_successful_result()
    assert ctx.publish(completed) == expected


def test_success_mark_cancel_race_records_only_accepted_presence():
    for _ in range(20):
        ctx = CrawlContext(CrawlLimits(), clock=lambda: 0)
        barrier = Barrier(2)

        def mark(ctx=ctx, barrier=barrier):
            barrier.wait()
            try:
                ctx.note_successful_result()
                return True
            except asyncio.CancelledError:
                return False

        def cancel(ctx=ctx, barrier=barrier):
            barrier.wait()
            ctx.cancel()

        with ThreadPoolExecutor(max_workers=2) as pool:
            accepted = pool.submit(mark)
            pool.submit(cancel).result()
        assert ctx.publish(CrawlOutcome("completed", "frontier_exhausted", False)) == CrawlOutcome(
            "cancelled", "cancelled", accepted.result()
        )


def test_parent_stop_reasons_are_only_deadline_and_budget():
    for reason in ("cancelled", "execution_failed", "secret"):
        with pytest.raises(ValueError):
            ParentStop(reason)
    assert ParentStop("budget_exhausted").reason == "budget_exhausted"


def test_concurrent_reserve_boundary():
    ctx = CrawlContext(CrawlLimits(page_attempts=5), clock=lambda: 0)
    barrier = Barrier(16)

    def reserve(_):
        barrier.wait()
        try:
            return ctx.reserve("page_attempts")
        except ParentStop:
            return None

    with ThreadPoolExecutor(max_workers=16) as pool:
        leases = list(pool.map(reserve, range(16)))
    assert sum(lease is not None for lease in leases) == 5
    for lease in leases:
        if lease is not None:
            lease.close()
    assert ctx.snapshot().page_attempts == 0
    with pytest.raises(ParentStop):
        ctx.check()


@given(st.lists(st.booleans(), max_size=100))
def test_reservation_close_invariant(started):
    ctx = CrawlContext(CrawlLimits(), clock=lambda: 0)
    leases = [ctx.reserve("page_attempts") for _ in started]
    for lease, commit in zip(leases, started):
        if commit:
            lease.commit()
        lease.close()
        lease.close()
    assert ctx.snapshot().page_attempts == sum(started)


@pytest.mark.asyncio
async def test_owned_handle_cancellation_does_not_claim_physical_drain(ctx):
    loop = asyncio.get_running_loop()
    result, drained = loop.create_future(), loop.create_future()
    calls = []
    handle = WorkHandle(result, drained, lambda: calls.append("owned"))
    ctx.attach(handle)
    ctx.cancel()
    ctx.cancel()
    assert calls == ["owned"]
    assert not result.done()
    assert not drained.done()
    result.cancel()
    assert not drained.done()
    drained.set_result(None)


@pytest.mark.asyncio
async def test_attach_after_cancel_calls_owned_cancellation(ctx):
    loop = asyncio.get_running_loop()
    calls = []
    ctx.cancel()
    ctx.attach(WorkHandle(loop.create_future(), loop.create_future(), lambda: calls.append(1)))
    assert calls == [1]


def test_cancel_callback_runs_outside_lock_and_snapshot_is_immutable(ctx):
    loop = asyncio.new_event_loop()
    try:
        snapshot = ctx.snapshot()
        ctx.attach(WorkHandle(loop.create_future(), loop.create_future(), lambda: ctx.snapshot()))
        ctx.cancel()
        assert snapshot.page_attempts == 0
        assert snapshot.accounting_complete is True
    finally:
        loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "secondary", [RuntimeError("private-callback-detail"), asyncio.CancelledError("private-detail")]
)
async def test_cancel_fanout_reports_secondary_failure_without_masking_caller_cancel(ctx, secondary):
    loop = asyncio.get_running_loop()
    calls, messages = [], []

    def failing_cancel():
        calls.append("first")
        raise secondary

    first = WorkHandle(loop.create_future(), loop.create_future(), failing_cancel)
    second = WorkHandle(loop.create_future(), loop.create_future(), lambda: calls.append("second"))
    ctx.attach(first)
    ctx.attach(second)
    original = asyncio.CancelledError("original caller cancellation")
    sink = logger.add(lambda message: messages.append(str(message)), format="{message}")
    try:
        with pytest.raises(asyncio.CancelledError) as caught:
            try:
                raise original
            except asyncio.CancelledError:
                ctx.cancel()
                raise
        ctx.cancel()
    finally:
        logger.remove(sink)
    assert caught.value is original
    assert calls == ["first", "second"]
    assert messages == ["crawl_cancel_callback_failed count=1\n"]
    assert not first.drained.done() and not second.drained.done()
    assert not first.result.done() and not second.result.done()


@pytest.mark.asyncio
@pytest.mark.parametrize("late_attach", [False, True])
async def test_cancel_retains_owned_handle_until_actual_physical_drain(ctx, late_attach):
    loop = asyncio.get_running_loop()
    if late_attach:
        ctx.cancel()
    drained = loop.create_future()
    handle = WorkHandle(loop.create_future(), drained, lambda: None)
    retained = weakref.ref(handle)
    ctx.attach(handle)
    ctx.cancel()
    del handle
    gc.collect()
    assert retained() is not None
    assert not drained.done()
    drained.set_result(None)
    # Existing ownership pruning occurs on attach, only after actual drain.
    ctx.attach(WorkHandle(loop.create_future(), loop.create_future(), lambda: None))
    gc.collect()
    assert retained() is None


def test_commit_close_race_never_refunds_started_work(ctx):
    for _ in range(20):
        lease = ctx.reserve("page_attempts")
        before = ctx.snapshot().page_attempts
        barrier = Barrier(2)

        def commit(barrier=barrier, lease=lease):
            barrier.wait()
            try:
                lease.commit()
                return True
            except RuntimeError:
                return False

        def close(barrier=barrier, lease=lease):
            barrier.wait()
            lease.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            started = pool.submit(commit)
            pool.submit(close).result()
        assert ctx.snapshot().page_attempts == before - (not started.result())


def test_cancel_publish_race_has_one_immutable_winner():
    for _ in range(20):
        ctx = CrawlContext(CrawlLimits(), clock=lambda: 0)
        completed = CrawlOutcome("completed", "frontier_exhausted", False)
        barrier = Barrier(2)

        def publish(barrier=barrier, ctx=ctx, completed=completed):
            barrier.wait()
            return ctx.publish(completed)

        def cancel(barrier=barrier, ctx=ctx):
            barrier.wait()
            ctx.cancel()

        with ThreadPoolExecutor(max_workers=2) as pool:
            published = pool.submit(publish)
            pool.submit(cancel).result()
        winner = published.result()
        assert winner in (completed, CrawlOutcome("cancelled", "cancelled", False))
        assert ctx.publish(completed) is winner


@pytest.mark.asyncio
async def test_logically_complete_handle_is_owned_until_physical_drain(ctx):
    loop = asyncio.get_running_loop()
    result, drained = loop.create_future(), loop.create_future()
    result.set_result("logical result")
    calls = []
    ctx.attach(WorkHandle(result, drained, lambda: calls.append(1)))
    ctx.cancel()
    assert calls == [1]
    assert result.result() == "logical result"
    assert not drained.done()
    drained.set_result(None)


@pytest.mark.asyncio
async def test_physical_drain_promptly_releases_credentials_and_callbacks_without_next_attach(ctx):
    class Secret:
        pass

    secret = Secret()
    retained = weakref.ref(secret)
    loop = asyncio.get_running_loop()
    result, drained = loop.create_future(), loop.create_future()
    result.set_result("done")
    handle = WorkHandle(result, drained, lambda bound=secret: bound)
    ctx.attach(handle)
    del secret, handle
    assert retained() is not None
    drained.set_result(None)
    await asyncio.sleep(0)
    assert retained() is None


@given(st.lists(st.sampled_from(["reserve", "commit", "close"]), max_size=100))
def test_repeated_reservation_operations_match_model(actions):
    ctx = CrawlContext(CrawlLimits(), clock=lambda: 0)
    lease = None
    state = "none"
    charged = 0
    for action in actions:
        if action == "reserve":
            if lease is not None:
                lease.close()
                charged -= state == "reserved"
            lease = ctx.reserve("network_requests")
            state = "reserved"
            charged += 1
        elif lease is not None and action == "close":
            lease.close()
            charged -= state == "reserved"
            if state == "reserved":
                state = "closed"
        elif lease is not None:
            if state == "closed":
                with pytest.raises(RuntimeError):
                    lease.commit()
            else:
                lease.commit()
                state = "committed"
        assert ctx.snapshot().network_requests == charged
