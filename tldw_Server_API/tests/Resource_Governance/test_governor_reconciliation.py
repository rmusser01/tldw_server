"""Regression and property coverage for bounded governor settlement."""

import asyncio
import itertools
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import (
    LedgerEntry,
    ResourceDailyLedger,
)
from tldw_Server_API.app.core.Resource_Governance import MemoryResourceGovernor, RGRequest, daily_caps

pytestmark = pytest.mark.rate_limit
INTEGER_CEILING = (1 << 63) - 1


@pytest.fixture
async def ledger(tmp_path, monkeypatch):
    """Use an owned SQLite pool without changing the process-wide pool."""
    config = get_settings().model_copy(
        update={
            "AUTH_MODE": "single_user",
            "DATABASE_URL": f"sqlite:///{tmp_path / 'reconciliation.db'}",
        }
    )
    pool = DatabasePool(settings=config)
    result = ResourceDailyLedger(db_pool=pool)
    await result.initialize()
    monkeypatch.setattr(daily_caps, "_daily_ledger", result)
    try:
        yield result
    finally:
        await pool.close()


@pytest.fixture
def operation_ids():
    """Keep property examples independent inside their shared ledger fixture."""
    return itertools.count()


def _entry(units=10, **kwargs):
    values = {
        "entity_scope": "user",
        "entity_value": "reconcile",
        "category": "minutes",
        "units": units,
        "op_id": "reservation",
        "occurred_at": datetime.now(timezone.utc),
    }
    values.update(kwargs)
    return LedgerEntry(**values)


def _governor(*, daily=True, token_window=100, daily_cap=100):
    limits = {"daily_cap": daily_cap} if daily else {}
    return MemoryResourceGovernor(
        policies={
            "p": {
                "requests": {"rpm": 100, **limits},
                "tokens": {"per_min": token_window, **limits},
                "minutes": limits,
                "cost_units": limits,
                "streams": {"max_concurrent": 1},
                "jobs": {"max_concurrent": 1},
                "scopes": ["global", "user"],
            }
        },
        time_source=lambda: 0.0,
    )


async def _reserve(gov, categories, *, op_id="reservation", entity="user:reconcile"):
    decision, handle = await gov.reserve(
        RGRequest(
            entity=entity,
            categories={category: {"units": units} for category, units in categories.items()},
            tags={"policy_id": "p"},
        ),
        op_id=op_id,
    )
    assert decision.allowed and handle is not None
    return handle


@pytest.fixture
async def duplicate_reserve_results(ledger, monkeypatch):
    """Overlap same-operation calls while daily I/O holds back publication."""
    gov = _governor()
    request = RGRequest(entity="user:reconcile", categories={"tokens": {"units": 20}}, tags={"policy_id": "p"})
    consumed = asyncio.Event()
    publish = asyncio.Event()
    duplicate_started = asyncio.Event()
    consume = daily_caps.consume_daily_cap

    async def paused_consume(**kwargs):
        result = await consume(**kwargs)
        consumed.set()
        await publish.wait()
        return result

    async def duplicate():
        duplicate_started.set()
        return await gov.reserve(request, op_id="shared")

    monkeypatch.setattr("tldw_Server_API.app.core.Resource_Governance.governor.consume_daily_cap", paused_consume)
    first = asyncio.create_task(gov.reserve(request, op_id="shared"))
    tasks = [first]
    try:
        await asyncio.wait_for(consumed.wait(), timeout=5)
        tasks.append(asyncio.create_task(duplicate()))
        await asyncio.wait_for(duplicate_started.wait(), timeout=5)
        publish.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
        return gov, results
    finally:
        publish.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_concurrent_duplicate_reserve_replays_same_handle_and_decision(duplicate_reserve_results):
    gov, ((first_decision, first_handle), (duplicate_decision, duplicate_handle)) = duplicate_reserve_results

    assert first_decision.allowed and first_handle is not None
    assert duplicate_handle == first_handle
    assert duplicate_decision is first_decision
    assert (await gov.peek_with_policy("user:reconcile", ["tokens"], "p"))["tokens"]["remaining"] == 80


@pytest.mark.asyncio
async def test_late_duplicate_reserve_release_cannot_erase_completed_usage(ledger, duplicate_reserve_results):
    gov, ((_, first_handle), (_, duplicate_handle)) = duplicate_reserve_results

    await gov.commit(first_handle, actuals={"tokens": 10})
    assert await ledger.total_for_day("user", "reconcile", "tokens") == 10
    await gov.release(duplicate_handle)

    assert await ledger.total_for_day("user", "reconcile", "tokens") == 10


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cancel", "error"])
async def test_reserve_owner_failure_releases_operation_lock_for_waiter(ledger, monkeypatch, failure):
    gov = _governor()
    owner_entered = asyncio.Event()
    fail_owner = asyncio.Event()
    waiter_started = asyncio.Event()
    consume = daily_caps.consume_daily_cap
    calls = 0
    request = RGRequest(entity="user:reconcile", categories={"tokens": {"units": 20}}, tags={"policy_id": "p"})

    async def fail_first_consume(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            owner_entered.set()
            await fail_owner.wait()
            raise RuntimeError("reservation failure")
        return await consume(**kwargs)

    async def waiter_reserve():
        waiter_started.set()
        return await gov.reserve(request, op_id="shared")

    monkeypatch.setattr("tldw_Server_API.app.core.Resource_Governance.governor.consume_daily_cap", fail_first_consume)
    owner = asyncio.create_task(gov.reserve(request, op_id="shared"))
    tasks = [owner]
    try:
        await asyncio.wait_for(owner_entered.wait(), timeout=5)
        waiter = asyncio.create_task(waiter_reserve())
        tasks.append(waiter)
        await asyncio.wait_for(waiter_started.wait(), timeout=5)
        if failure == "cancel":
            owner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await owner
        else:
            fail_owner.set()
            with pytest.raises(RuntimeError, match="reservation failure"):
                await owner
        decision, handle = await asyncio.wait_for(waiter, timeout=5)
        assert decision.allowed and handle is not None
        replay_decision, replay_handle = await asyncio.wait_for(gov.reserve(request, op_id="shared"), timeout=5)
        assert replay_decision is decision and replay_handle == handle
        assert not gov._reserve_locks
    finally:
        fail_owner.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_reserve_waiter_does_not_remove_owner_operation_lock(ledger, monkeypatch):
    gov = _governor()
    owner_entered = asyncio.Event()
    publish = asyncio.Event()
    consume = daily_caps.consume_daily_cap
    request = RGRequest(entity="user:reconcile", categories={"tokens": {"units": 20}}, tags={"policy_id": "p"})

    async def paused_consume(**kwargs):
        result = await consume(**kwargs)
        owner_entered.set()
        await publish.wait()
        return result

    async def start_duplicate(started):
        started.set()
        return await gov.reserve(request, op_id="shared")

    monkeypatch.setattr("tldw_Server_API.app.core.Resource_Governance.governor.consume_daily_cap", paused_consume)
    owner = asyncio.create_task(gov.reserve(request, op_id="shared"))
    tasks = [owner]
    try:
        await asyncio.wait_for(owner_entered.wait(), timeout=5)
        waiter_started = asyncio.Event()
        cancelled_waiter = asyncio.create_task(start_duplicate(waiter_started))
        tasks.append(cancelled_waiter)
        await asyncio.wait_for(waiter_started.wait(), timeout=5)
        cancelled_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_waiter
        assert len(gov._reserve_locks) == 1

        later_started = asyncio.Event()
        later = asyncio.create_task(start_duplicate(later_started))
        tasks.append(later)
        await asyncio.wait_for(later_started.wait(), timeout=5)
        publish.set()
        first_result, later_result = await asyncio.wait_for(asyncio.gather(owner, later), timeout=5)
        assert later_result[0] is first_result[0] and later_result[1] == first_result[1]
        assert not gov._reserve_locks
    finally:
        publish.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_reserve_operation_lock_entries_do_not_accumulate(ledger):
    gov = _governor()
    for index in range(12):
        handle = await _reserve(gov, {"minutes": 1}, op_id=f"operation-{index}")
        await gov.release(handle)

    assert not gov._reserve_locks


@pytest.mark.asyncio
async def test_queued_duplicate_reserve_replays_after_slow_daily_io(ledger, monkeypatch):
    gov = _governor()
    clock = [0.0]
    gov._time = lambda: clock[0]
    consumed = asyncio.Event()
    publish = asyncio.Event()
    duplicate_started = asyncio.Event()
    consume = daily_caps.consume_daily_cap
    request = RGRequest(entity="user:reconcile", categories={"tokens": {"units": 20}}, tags={"policy_id": "p"})

    async def paused_consume(**kwargs):
        result = await consume(**kwargs)
        consumed.set()
        await publish.wait()
        return result

    async def duplicate():
        duplicate_started.set()
        return await gov.reserve(request, op_id="shared")

    monkeypatch.setattr("tldw_Server_API.app.core.Resource_Governance.governor.consume_daily_cap", paused_consume)
    owner = asyncio.create_task(gov.reserve(request, op_id="shared"))
    tasks = [owner]
    try:
        await asyncio.wait_for(consumed.wait(), timeout=5)
        later = asyncio.create_task(duplicate())
        tasks.append(later)
        await asyncio.wait_for(duplicate_started.wait(), timeout=5)
        clock[0] = 200.0
        publish.set()
        first_result, duplicate_result = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
        assert duplicate_result[0] is first_result[0] and duplicate_result[1] == first_result[1]
        await gov.commit(first_result[1], actuals={"tokens": 10})
        await gov.release(duplicate_result[1])
        assert await ledger.total_for_day("user", "reconcile", "tokens") == 10
        assert not gov._reserve_locks
    finally:
        publish.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_denied_reserve_result_ttl_starts_at_publication(ledger, monkeypatch):
    gov = _governor(daily_cap=10)
    clock = [0.0]
    gov._time = lambda: clock[0]
    check = gov.check
    calls = 0
    request = RGRequest(entity="user:reconcile", categories={"minutes": {"units": 20}}, tags={"policy_id": "p"})

    async def slow_check(req):
        nonlocal calls
        calls += 1
        result = await check(req)
        clock[0] += 200.0
        return result

    monkeypatch.setattr(gov, "check", slow_check)
    first_decision, first_handle = await gov.reserve(request, op_id="denied")
    repeated_decision, repeated_handle = await gov.reserve(request, op_id="denied")

    assert not first_decision.allowed and first_handle is None
    assert repeated_decision is first_decision and repeated_handle is None
    assert calls == 1
    assert not gov._reserve_locks


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["expired-cache", "new-governor"])
@pytest.mark.parametrize("finalization", ["release", "commit"])
async def test_durable_duplicate_reserve_cannot_own_prior_completed_charge(ledger, boundary, finalization):
    gov = _governor()
    clock = [0.0]
    gov._time = lambda: clock[0]
    original = await _reserve(gov, {"tokens": 20}, op_id="shared")
    await gov.commit(original, actuals={"tokens": 10})

    if boundary == "expired-cache":
        clock[0] = 200.0
    else:
        gov = _governor()
    duplicate = await _reserve(gov, {"tokens": 20}, op_id="shared")
    assert duplicate != original
    if finalization == "release":
        await gov.release(duplicate)
    else:
        await gov.commit(duplicate, actuals={"tokens": 1})

    assert await ledger.total_for_day("user", "reconcile", "tokens") == 10


@pytest.mark.asyncio
async def test_other_governor_duplicate_release_preserves_original_owner(ledger):
    gov = _governor()
    original = await _reserve(gov, {"tokens": 20}, op_id="shared")
    other = _governor()
    duplicate = await _reserve(other, {"tokens": 20}, op_id="shared")

    await other.release(duplicate)
    assert await ledger.total_for_day("user", "reconcile", "tokens") == 20
    await gov.commit(original, actuals={"tokens": 10})

    assert await ledger.total_for_day("user", "reconcile", "tokens") == 10


@pytest.mark.asyncio
async def test_daily_consume_reports_durable_insertion_ownership(ledger):
    kwargs = {
        "entity_scope": "user",
        "entity_value": "reconcile",
        "category": "tokens",
        "daily_cap": 100,
        "units": 20,
        "op_id": "shared",
    }

    allowed, _, first_details = await daily_caps.consume_daily_cap(**kwargs)
    repeated_allowed, _, repeated_details = await daily_caps.consume_daily_cap(**kwargs)

    assert allowed and repeated_allowed
    assert first_details["daily_inserted"] is True
    assert repeated_details["daily_inserted"] is False


@pytest.mark.asyncio
async def test_partial_denial_does_not_roll_back_prior_durable_duplicate(ledger, monkeypatch):
    original_gov = _governor(daily_cap=100)
    original = await _reserve(original_gov, {"minutes": 20}, op_id="shared")
    await original_gov.commit(original, actuals={"minutes": 10})
    duplicate_gov = _governor(daily_cap=100)
    consume = daily_caps.consume_daily_cap

    async def raced_consume(**kwargs):
        if kwargs["category"] == "cost_units":
            await ledger.add(_entry(100, category="cost_units", op_id="racer"))
        return await consume(**kwargs)

    monkeypatch.setattr("tldw_Server_API.app.core.Resource_Governance.governor.consume_daily_cap", raced_consume)
    request = RGRequest(
        entity="user:reconcile",
        categories={"minutes": {"units": 20}, "cost_units": {"units": 1}},
        tags={"policy_id": "p"},
    )

    decision, handle = await duplicate_gov.reserve(request, op_id="shared")

    assert not decision.allowed and handle is None
    assert await ledger.total_for_day("user", "reconcile", "minutes") == 10


@pytest.mark.asyncio
async def test_release_refunds_every_window_and_releases_leases():
    gov = _governor(daily=False)
    handle = await _reserve(gov, {"requests": 8, "tokens": 20, "streams": 1, "jobs": 1})

    await gov.release(handle)

    decision = await gov.check(
        RGRequest(
            entity="user:reconcile",
            categories={
                "requests": {"units": 100},
                "tokens": {"units": 100},
                "streams": {"units": 1},
                "jobs": {"units": 1},
            },
            tags={"policy_id": "p"},
        )
    )
    assert decision.allowed


@pytest.mark.asyncio
async def test_release_reconciles_all_daily_categories_to_zero(ledger):
    gov = _governor()
    categories = {"requests": 8, "tokens": 20, "minutes": 10, "cost_units": 5}
    handle = await _reserve(gov, categories)

    await gov.release(handle)
    await gov.release(handle)
    await gov.commit(handle, actuals=categories, op_id="late-callback")

    for category in categories:
        assert await ledger.total_for_day("user", "reconcile", category) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "actuals, expected",
    [(None, (20, 10)), ({}, (20, 10)), ({"tokens": 3}, (3, 10)), ({"tokens": -2, "minutes": 200}, (0, 10))],
)
async def test_commit_bounds_actuals_and_preserves_omitted_categories(ledger, actuals, expected):
    gov = _governor()
    handle = await _reserve(gov, {"tokens": 20, "minutes": 10})

    await gov.commit(handle, actuals=actuals, op_id="callback")
    await gov.commit(handle, actuals={"tokens": 0, "minutes": 0}, op_id="callback")
    await gov.commit(handle, actuals={"tokens": 0, "minutes": 0}, op_id="other-callback")
    await gov.release(handle)

    assert (
        await ledger.total_for_day("user", "reconcile", "tokens"),
        await ledger.total_for_day("user", "reconcile", "minutes"),
    ) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("actual, expected", [(None, 500), (300, 300), (0, 0), (1000, 500)])
async def test_daily_actuals_are_bounded_by_original_not_window_clamped_units(ledger, actual, expected):
    gov = _governor(token_window=100, daily_cap=1000)
    handle = await _reserve(gov, {"tokens": 500})

    await gov.commit(handle, actuals={} if actual is None else {"tokens": actual})

    assert await ledger.total_for_day("user", "reconcile", "tokens") == expected


@pytest.mark.asyncio
async def test_commit_keeps_other_reservations_charged(ledger):
    gov = _governor()
    first = await _reserve(gov, {"minutes": 10}, op_id="first")
    await _reserve(gov, {"minutes": 15}, op_id="second")

    await gov.commit(first, actuals={"minutes": 2})

    assert await ledger.total_for_day("user", "reconcile", "minutes") == 17


@pytest.mark.asyncio
async def test_unknown_finalization_does_not_adjust_ledger(ledger):
    gov = _governor()
    await _reserve(gov, {"minutes": 10})

    await gov.release("unknown")
    await gov.commit("unknown", actuals={"minutes": 0}, op_id="unknown-callback")

    assert await ledger.total_for_day("user", "reconcile", "minutes") == 10


@pytest.mark.asyncio
async def test_later_daily_denial_rolls_back_earlier_categories(ledger, monkeypatch):
    gov = _governor(daily_cap=10)
    consume = daily_caps.consume_daily_cap

    async def raced_consume(**kwargs):
        if kwargs["category"] == "cost_units":
            await ledger.add(_entry(10, category="cost_units", op_id="concurrent-writer"))
        return await consume(**kwargs)

    monkeypatch.setattr("tldw_Server_API.app.core.Resource_Governance.governor.consume_daily_cap", raced_consume)
    request = RGRequest(
        entity="user:reconcile",
        categories={"minutes": {"units": 5}, "tokens": {"units": 5}, "cost_units": {"units": 1}},
        tags={"policy_id": "p"},
    )

    decision, handle = await gov.reserve(request, op_id="denied")
    repeated, repeated_handle = await gov.reserve(request, op_id="denied")

    assert not decision.allowed and handle is None
    assert repeated is decision and repeated_handle is None
    assert await ledger.total_for_day("user", "reconcile", "minutes") == 0
    assert await ledger.total_for_day("user", "reconcile", "tokens") == 0
    assert await ledger.total_for_day("user", "reconcile", "cost_units") == 10


@pytest.mark.asyncio
async def test_ledger_downward_adjustment_is_idempotent_and_rejects_increases(ledger):
    entry = _entry()
    await ledger.add(entry)

    assert await ledger.adjust_downward(replace(entry, units=3)) is True
    assert await ledger.adjust_downward(replace(entry, units=3)) is True
    with pytest.raises(ValueError, match="increase"):
        await ledger.adjust_downward(replace(entry, units=4))

    assert await ledger.total_for_day("user", "reconcile", "minutes") == 3


@pytest.mark.asyncio
async def test_ledger_downward_adjustment_never_inserts_unknown_identity(ledger):
    assert await ledger.adjust_downward(_entry(0)) is False
    assert await ledger.total_for_day("user", "reconcile", "minutes") == 0


@pytest.mark.asyncio
async def test_ledger_downward_adjustment_rejects_negative_units(ledger):
    entry = _entry()
    await ledger.add(entry)

    with pytest.raises(ValueError, match="non-negative"):
        await ledger.adjust_downward(replace(entry, units=-1))

    assert await ledger.total_for_day("user", "reconcile", "minutes") == 10


@pytest.mark.asyncio
async def test_ledger_adjustment_uses_full_identity_including_day(ledger):
    entry = _entry(10)
    yesterday = entry.occurred_at - timedelta(days=1)
    others = [
        replace(entry, occurred_at=yesterday),
        replace(entry, entity_scope="team"),
        replace(entry, entity_value="other"),
        replace(entry, category="tokens"),
        replace(entry, op_id="other"),
    ]
    for row in [entry, *others]:
        await ledger.add(row)

    assert await ledger.adjust_downward(replace(entry, units=2)) is True
    assert await ledger.total_for_day("user", "reconcile", "minutes") == 12
    for row in others:
        day = row.occurred_at.date().isoformat()
        expected = 12 if row.op_id == "other" else 10
        assert await ledger.total_for_day(row.entity_scope, row.entity_value, row.category, day_utc=day) == expected


@pytest.mark.asyncio
async def test_release_preserves_reservation_day_across_midnight(ledger, monkeypatch):
    before = datetime(2026, 10, 2, 23, 59, tzinfo=timezone.utc)
    clock = [before]

    class ClockDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0]

    monkeypatch.setattr(daily_caps, "datetime", ClockDateTime)
    gov = _governor()
    handle = await _reserve(gov, {"minutes": 10})
    clock[0] = before + timedelta(minutes=2)
    await ledger.add(_entry(7, op_id="p:reservation:minutes", occurred_at=clock[0]))

    await gov.release(handle)

    assert await ledger.total_for_day("user", "reconcile", "minutes", day_utc="2026-10-02") == 0
    assert await ledger.total_for_day("user", "reconcile", "minutes", day_utc="2026-10-03") == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["release", "commit"])
async def test_ledger_outage_keeps_conservative_charge(ledger, monkeypatch, method):
    gov = _governor()
    handle = await _reserve(gov, {"tokens": 20})

    async def unavailable():
        return None

    monkeypatch.setattr(daily_caps, "_get_ledger", unavailable)
    if method == "release":
        await gov.release(handle)
    else:
        await gov.commit(handle, actuals={"tokens": 2})

    assert await ledger.total_for_day("user", "reconcile", "tokens") == 20


@pytest.mark.asyncio
async def test_reconciliation_write_error_is_noncritical_and_conservative(ledger, monkeypatch):
    gov = _governor()
    handle = await _reserve(gov, {"tokens": 20})

    async def failing_adjustment(*args, **kwargs):
        raise TransactionError("daily cap adjustment", "database unavailable")

    monkeypatch.setattr(ledger, "adjust_downward", failing_adjustment, raising=False)
    await gov.commit(handle, actuals={"tokens": 2})

    assert await ledger.total_for_day("user", "reconcile", "tokens") == 20


@pytest.mark.asyncio
async def test_concurrent_duplicate_callbacks_refund_window_only_once(ledger, monkeypatch):
    gov = _governor()
    handle = await _reserve(gov, {"tokens": 20})
    await _reserve(gov, {"tokens": 30}, op_id="other")
    adjustment_started = asyncio.Event()
    adjustment_continue = asyncio.Event()
    original = getattr(ledger, "adjust_downward", None)

    async def slow_adjustment(*args, **kwargs):
        adjustment_started.set()
        await adjustment_continue.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(ledger, "adjust_downward", slow_adjustment, raising=False)
    first = asyncio.create_task(gov.commit(handle, actuals={"tokens": 5}))
    try:
        await asyncio.wait_for(adjustment_started.wait(), timeout=1)
        await gov.commit(handle, actuals={"tokens": 0}, op_id="duplicate")
    finally:
        adjustment_continue.set()
        await first

    assert (await gov.peek_with_policy("user:reconcile", ["tokens"], "p"))["tokens"]["remaining"] == 65
    assert await ledger.total_for_day("user", "reconcile", "tokens") == 35


@pytest.mark.asyncio
async def test_daily_settlement_near_integer_ceiling_is_exact(ledger):
    gov = _governor(daily_cap=INTEGER_CEILING)
    handle = await _reserve(gov, {"minutes": INTEGER_CEILING})

    await gov.commit(handle, actuals={"minutes": INTEGER_CEILING - 1})

    assert await ledger.total_for_day("user", "reconcile", "minutes") == INTEGER_CEILING - 1
    decision, second = await gov.reserve(
        RGRequest(entity="user:reconcile", categories={"minutes": {"units": 1}}, tags={"policy_id": "p"})
    )
    assert decision.allowed and second is not None
    await gov.release(second)
    assert await ledger.total_for_day("user", "reconcile", "minutes") == INTEGER_CEILING - 1


@settings(max_examples=35, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    reserved=st.integers(min_value=1, max_value=INTEGER_CEILING),
    actual=st.integers(min_value=-10, max_value=INTEGER_CEILING + 10),
    release=st.booleans(),
    repeats=st.integers(min_value=1, max_value=5),
)
@pytest.mark.asyncio
async def test_property_finalization_is_bounded_and_first_callback_wins(
    ledger, operation_ids, reserved, actual, release, repeats
):
    entity_value = f"property-{next(operation_ids)}"
    gov = _governor(daily_cap=INTEGER_CEILING)
    handle = await _reserve(gov, {"minutes": reserved, "cost_units": 7}, entity=f"user:{entity_value}")
    if release:
        await gov.release(handle)
    else:
        await gov.commit(handle, actuals={"minutes": actual})
    for index in range(repeats):
        await gov.release(handle)
        await gov.commit(handle, actuals={"minutes": reserved}, op_id=f"duplicate-{index}")
        await gov.commit("unknown", actuals={"minutes": 0})

    assert await ledger.total_for_day("user", entity_value, "minutes") == (
        0 if release else max(0, min(actual, reserved))
    )
    assert await ledger.total_for_day("user", entity_value, "cost_units") == (0 if release else 7)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    reserved=st.integers(min_value=1, max_value=INTEGER_CEILING),
    actual=st.integers(min_value=0, max_value=INTEGER_CEILING),
)
@pytest.mark.asyncio
async def test_property_ledger_adjustment_is_monotonic(ledger, operation_ids, reserved, actual):
    entry = _entry(reserved, entity_value=f"ledger-property-{next(operation_ids)}")
    await ledger.add(entry)
    if actual > reserved:
        with pytest.raises(ValueError, match="increase"):
            await ledger.adjust_downward(replace(entry, units=actual))
        expected = reserved
    else:
        await ledger.adjust_downward(replace(entry, units=actual))
        await ledger.adjust_downward(replace(entry, units=actual))
        expected = actual
    assert await ledger.total_for_day("user", entry.entity_value, "minutes") == expected


@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    units=st.integers(min_value=1, max_value=100),
    categories=st.permutations(["minutes", "cost_units", "tokens"]),
)
@pytest.mark.asyncio
async def test_property_partial_denial_rolls_back_every_prior_category(
    ledger, operation_ids, monkeypatch, units, categories
):
    entity_value = f"denial-property-{next(operation_ids)}"
    gov = _governor(daily_cap=100)
    consume = daily_caps.consume_daily_cap

    async def raced_consume(**kwargs):
        if kwargs["category"] == categories[-1]:
            await ledger.add(_entry(100, entity_value=entity_value, category=categories[-1], op_id="racer"))
        return await consume(**kwargs)

    monkeypatch.setattr("tldw_Server_API.app.core.Resource_Governance.governor.consume_daily_cap", raced_consume)
    request = RGRequest(
        entity=f"user:{entity_value}",
        categories={category: {"units": units} for category in categories},
        tags={"policy_id": "p"},
    )

    decision, handle = await gov.reserve(request, op_id="denied")

    assert not decision.allowed and handle is None
    for category in categories[:-1]:
        assert await ledger.total_for_day("user", entity_value, category) == 0
    assert await ledger.total_for_day("user", entity_value, categories[-1]) == 100


@pytest.mark.asyncio
async def test_concurrent_ledger_adjustments_cannot_restore_consumption(ledger):
    entry = _entry()
    await ledger.add(entry)

    results = await asyncio.gather(
        ledger.adjust_downward(replace(entry, units=5)),
        ledger.adjust_downward(replace(entry, units=1)),
        ledger.adjust_downward(replace(entry, units=1)),
        return_exceptions=True,
    )

    assert all(result is True or isinstance(result, ValueError) for result in results)
    assert await ledger.total_for_day("user", "reconcile", "minutes") == 1
    replay = await ledger.consume_if_within_cap(entry, daily_cap=10)
    assert replay.allowed and not replay.inserted
    assert await ledger.total_for_day("user", "reconcile", "minutes") == 1


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["expired-cache", "new-governor"])
@pytest.mark.parametrize("finalization", ["release", "commit"])
async def test_postgres_duplicate_daily_settlement_ownership(
    isolated_test_environment, monkeypatch, boundary, finalization
):
    """Durable PostgreSQL replay cannot acquire another handle's refund rights."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    pool = await get_db_pool()
    pg_ledger = ResourceDailyLedger(db_pool=pool)
    await pg_ledger.initialize()
    monkeypatch.setattr(daily_caps, "_daily_ledger", pg_ledger)
    gov = _governor()
    clock = [0.0]
    gov._time = lambda: clock[0]
    original = await _reserve(gov, {"tokens": 20}, op_id="shared")
    await gov.commit(original, actuals={"tokens": 10})

    if boundary == "expired-cache":
        clock[0] = 200.0
    else:
        gov = _governor()
    duplicate = await _reserve(gov, {"tokens": 20}, op_id="shared")
    assert duplicate != original
    if finalization == "release":
        await gov.release(duplicate)
    else:
        await gov.commit(duplicate, actuals={"tokens": 1})

    assert await pg_ledger.total_for_day("user", "reconcile", "tokens") == 10


@pytest.mark.integration
@pytest.mark.asyncio
async def test_postgres_downward_adjustment_parity(isolated_test_environment):
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    pool = await get_db_pool()
    pg_ledger = ResourceDailyLedger(db_pool=pool)
    entry = _entry(INTEGER_CEILING)
    await pg_ledger.add(entry)

    assert await pg_ledger.adjust_downward(replace(entry, units=INTEGER_CEILING - 1)) is True
    assert await pg_ledger.adjust_downward(replace(entry, units=INTEGER_CEILING - 1)) is True
    with pytest.raises(ValueError, match="increase"):
        await pg_ledger.adjust_downward(entry)
    assert await pg_ledger.total_for_day("user", "reconcile", "minutes") == INTEGER_CEILING - 1
    assert await pg_ledger.adjust_downward(replace(entry, units=0)) is True
    replay = await pg_ledger.consume_if_within_cap(entry, daily_cap=INTEGER_CEILING)
    assert replay.allowed and not replay.inserted
    assert await pg_ledger.total_for_day("user", "reconcile", "minutes") == 0
