import asyncio
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest
from loguru import logger

from tldw_Server_API.app.core.Jobs import worker_sdk as worker_sdk_module
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    LeaseHorizonResult,
    NoTransitionReason,
    PreparedDispositionOrigin,
    PreparedDispositionResult,
    PreparedJobDisposition,
)
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK

DELIVERY_ID = "de305d54-75b4-431b-adb2-eb6b9e546014"
ATTEMPT_ID = "123e4567-e89b-42d3-a456-426614174000"
LEASED_UNTIL = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
RENEWED_UNTIL = LEASED_UNTIL + timedelta(minutes=5)


def _job() -> dict[str, object]:
    return {
        "id": 17,
        "uuid": "bffb6b56-0db3-4fe8-b2f8-a77d6131bf6d",
        "domain": "admin_webhooks",
        "queue": "delivery",
        "job_type": "admin_webhook_delivery",
        "payload": {"delivery_id": DELIVERY_ID},
        "worker_id": "worker-1",
        "lease_id": "lease-1",
        "leased_until": LEASED_UNTIL,
    }


def _complete(*, token: str = "a" * 64) -> PreparedJobDisposition:
    return PreparedJobDisposition.complete(
        token=token,
        delivery_id=DELIVERY_ID,
        attempt_id=ATTEMPT_ID,
    )


def _infrastructure_defer(*, token: str = "b" * 64) -> PreparedJobDisposition:
    return PreparedJobDisposition.infrastructure_defer(
        token=token,
        delivery_id=DELIVERY_ID,
        reason_code="worker_error",
    )


def _applied_result(*, already_applied: bool = False) -> PreparedDispositionResult:
    return PreparedDispositionResult.applied(
        state="completed",
        metadata={"delivery_id": DELIVERY_ID},
        already_applied=already_applied,
    )


class PreparedManager:
    def __init__(self) -> None:
        self.jobs = [_job()]
        self.acquire_calls: list[dict[str, object]] = []
        self.apply_calls: list[object] = []
        self.horizon_calls: list[object] = []
        self.fallback_calls: list[str] = []
        self.apply_result: object = _applied_result()
        self.horizon_result: object = LeaseHorizonResult.applied(
            leased_until=RENEWED_UNTIL
        )

    def acquire_next_job(self, **kwargs):
        self.acquire_calls.append(kwargs)
        return self.jobs.pop(0) if self.jobs else None

    def apply_prepared_disposition(self, command):
        self.apply_calls.append(command)
        if isinstance(self.apply_result, BaseException):
            raise self.apply_result
        return self.apply_result

    def ensure_lease_horizon(self, command):
        self.horizon_calls.append(command)
        if isinstance(self.horizon_result, BaseException):
            raise self.horizon_result
        return self.horizon_result

    def complete_job(self, *_args, **_kwargs):
        self.fallback_calls.append("complete_job")
        return False

    def fail_job(self, *_args, **_kwargs):
        self.fallback_calls.append("fail_job")
        return False

    def release_job(self, *_args, **_kwargs):
        self.fallback_calls.append("release_job")
        return False

    def finalize_cancelled(self, *_args, **_kwargs):
        self.fallback_calls.append("finalize_cancelled")
        return False

    def terminalize_job_from_worker(self, *_args, **_kwargs):
        self.fallback_calls.append("terminalize_job_from_worker")
        return "CONFLICT"


def _sdk(manager: PreparedManager, **config_overrides) -> WorkerSDK:
    config = WorkerConfig(
        domain="admin_webhooks",
        queue="delivery",
        worker_id="worker-1",
        lease_seconds=4,
        renew_threshold_seconds=3,
        renew_jitter_seconds=0,
        **config_overrides,
    )
    return WorkerSDK(manager, config)


async def _allow_acquire() -> bool:
    return True


def _closed_error_disposition(
    _job_row: dict[str, object],
    _error_class: type[BaseException],
) -> PreparedJobDisposition:
    return _infrastructure_defer()


@pytest.mark.asyncio
@pytest.mark.parametrize("guard_failure", [False, RuntimeError], ids=["false", "exception"])
async def test_pre_acquire_guard_fails_closed_without_acquiring(guard_failure):
    manager = PreparedManager()
    sdk = _sdk(manager)

    async def guard() -> bool:
        sdk.stop()
        if guard_failure is RuntimeError:
            raise RuntimeError("guard detail must stay private")
        return False

    async def handler(_job_row, _context):
        pytest.fail("handler must not run when the pre-acquire guard is closed")

    await sdk.run_prepared(
        handler=handler,
        pre_acquire_guard=guard,
        handler_error_disposition=_closed_error_disposition,
    )

    assert manager.acquire_calls == []
    assert manager.apply_calls == []


@pytest.mark.asyncio
async def test_pre_acquire_guard_runs_before_every_acquisition_attempt():
    manager = PreparedManager()
    sdk = _sdk(manager)
    guard_calls = 0

    async def guard() -> bool:
        nonlocal guard_calls
        guard_calls += 1
        if guard_calls == 2:
            sdk.stop()
            return False
        return True

    async def handler(_job_row, _context):
        return _complete()

    await sdk.run_prepared(
        handler=handler,
        pre_acquire_guard=guard,
        handler_error_disposition=_closed_error_disposition,
    )

    assert guard_calls == 2
    assert len(manager.acquire_calls) == 1
    assert len(manager.apply_calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "disposition",
    [
        _complete(token="1" * 64),
        PreparedJobDisposition.retry(
            token="2" * 64,
            delivery_id=DELIVERY_ID,
            attempt_id=ATTEMPT_ID,
            delay_seconds=30,
            not_before_at=LEASED_UNTIL,
            reason_code="remote_retry",
        ),
        PreparedJobDisposition.fail(
            token="3" * 64,
            delivery_id=DELIVERY_ID,
            attempt_id=ATTEMPT_ID,
            reason_code="terminal_failure",
        ),
        PreparedJobDisposition.cancel(
            token="4" * 64,
            delivery_id=DELIVERY_ID,
            reason_code="disabled",
        ),
        _infrastructure_defer(token="5" * 64),
        PreparedJobDisposition.recovery_defer_until(
            token="6" * 64,
            delivery_id=DELIVERY_ID,
            not_before_at=LEASED_UNTIL,
            reason_code="stale_attempt",
        ),
    ],
    ids=["complete", "retry", "fail", "cancel", "infrastructure", "recovery"],
)
async def test_handler_dispositions_use_one_typed_apply_and_no_legacy_fallback(
    disposition,
):
    manager = PreparedManager()
    sdk = _sdk(manager)

    async def handler(_job_row, _context):
        sdk.stop()
        return disposition

    await sdk.run_prepared(
        handler=handler,
        pre_acquire_guard=_allow_acquire,
        handler_error_disposition=_closed_error_disposition,
    )

    assert len(manager.apply_calls) == 1
    command = manager.apply_calls[0]
    assert command.job_id == 17
    assert command.domain == "admin_webhooks"
    assert command.queue == "delivery"
    assert command.job_type == "admin_webhook_delivery"
    assert command.expected_payload == {"delivery_id": DELIVERY_ID}
    assert command.worker_id == "worker-1"
    assert command.lease_id == "lease-1"
    assert command.disposition is disposition
    assert manager.fallback_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("already_applied", [False, True], ids=["new", "idempotent"])
async def test_authnz_applied_result_invokes_only_exact_applied_callback(
    already_applied,
):
    manager = PreparedManager()
    disposition = _complete()
    result = _applied_result(already_applied=already_applied)
    manager.apply_result = result
    sdk = _sdk(manager)
    applied: list[tuple[object, object, object]] = []
    rejected: list[tuple[object, object, object]] = []

    async def handler(_job_row, _context):
        return disposition

    async def on_applied(job_row, exact_disposition, exact_result):
        applied.append((job_row, exact_disposition, exact_result))
        sdk.stop()

    async def on_rejected(job_row, exact_disposition, exact_result):
        rejected.append((job_row, exact_disposition, exact_result))

    await sdk.run_prepared(
        handler=handler,
        pre_acquire_guard=_allow_acquire,
        handler_error_disposition=_closed_error_disposition,
        on_disposition_applied=on_applied,
        on_disposition_rejected=on_rejected,
    )

    assert len(applied) == 1
    assert applied[0][0] == _job()
    assert applied[0][1] is disposition
    assert applied[0][2] is result
    assert rejected == []
    assert len(manager.apply_calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "disposition",
    [
        _infrastructure_defer(),
        PreparedJobDisposition.recovery_defer_until(
            token="c" * 64,
            delivery_id=DELIVERY_ID,
            not_before_at=LEASED_UNTIL,
            reason_code="stale_attempt",
        ),
    ],
    ids=["infrastructure", "recovery"],
)
async def test_non_authnz_applied_result_never_invokes_ack_callback(disposition):
    manager = PreparedManager()
    sdk = _sdk(manager)
    callbacks: list[str] = []

    async def handler(_job_row, _context):
        sdk.stop()
        return disposition

    async def on_applied(_job_row, _disposition, _result):
        callbacks.append("applied")

    async def on_rejected(_job_row, _disposition, _result):
        callbacks.append("rejected")

    await sdk.run_prepared(
        handler=handler,
        pre_acquire_guard=_allow_acquire,
        handler_error_disposition=_closed_error_disposition,
        on_disposition_applied=on_applied,
        on_disposition_rejected=on_rejected,
    )

    assert callbacks == []
    assert len(manager.apply_calls) == 1


@pytest.mark.asyncio
async def test_typed_non_applied_result_invokes_only_rejected_callback():
    manager = PreparedManager()
    disposition = _complete()
    result = PreparedDispositionResult.conflict(state="processing")
    manager.apply_result = result
    sdk = _sdk(manager)
    applied: list[object] = []
    rejected: list[tuple[object, object, object]] = []

    async def handler(_job_row, _context):
        return disposition

    async def on_applied(*args):
        applied.append(args)

    async def on_rejected(job_row, exact_disposition, exact_result):
        rejected.append((job_row, exact_disposition, exact_result))
        sdk.stop()

    await sdk.run_prepared(
        handler=handler,
        pre_acquire_guard=_allow_acquire,
        handler_error_disposition=_closed_error_disposition,
        on_disposition_applied=on_applied,
        on_disposition_rejected=on_rejected,
    )

    assert applied == []
    assert len(rejected) == 1
    assert rejected[0][1] is disposition
    assert rejected[0][2] is result
    assert len(manager.apply_calls) == 1


class HandlerSecretError(RuntimeError):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_result", [None, {"wrong": "shape"}], ids=["none", "dict"])
async def test_malformed_handler_result_uses_only_type_error_class(handler_result):
    manager = PreparedManager()
    sdk = _sdk(manager)
    factory_calls: list[tuple[object, object]] = []

    async def handler(_job_row, _context):
        sdk.stop()
        return handler_result

    def error_disposition(job_row, error_class):
        factory_calls.append((job_row, error_class))
        return _infrastructure_defer()

    await sdk.run_prepared(
        handler=handler,
        pre_acquire_guard=_allow_acquire,
        handler_error_disposition=error_disposition,
    )

    assert factory_calls == [(_job(), TypeError)]
    assert len(manager.apply_calls) == 1
    disposition = manager.apply_calls[0].disposition
    assert disposition.origin is PreparedDispositionOrigin.INFRASTRUCTURE
    assert disposition.not_before_at is None
    assert manager.fallback_calls == []


@pytest.mark.asyncio
async def test_handler_exception_factory_receives_only_exception_class_and_job():
    manager = PreparedManager()
    derived_schedule = LEASED_UNTIL + timedelta(seconds=30)
    manager.apply_result = PreparedDispositionResult.applied(
        state="queued",
        metadata={"delivery_id": DELIVERY_ID},
        already_applied=False,
        not_before_at=derived_schedule,
    )
    sdk = _sdk(manager)
    factory_calls: list[tuple[object, object]] = []
    logs: list[str] = []
    sink = logger.add(logs.append, format="{message}|{extra}", level="DEBUG")

    async def handler(_job_row, _context):
        sdk.stop()
        raise HandlerSecretError("handler-secret-9f4d")

    def error_disposition(job_row, error_class):
        factory_calls.append((job_row, error_class))
        return _infrastructure_defer()

    try:
        await sdk.run_prepared(
            handler=handler,
            pre_acquire_guard=_allow_acquire,
            handler_error_disposition=error_disposition,
        )
    finally:
        logger.remove(sink)

    assert factory_calls == [(_job(), HandlerSecretError)]
    assert len(manager.apply_calls) == 1
    assert manager.apply_calls[0].disposition.not_before_at is None
    assert manager.apply_result.not_before_at == derived_schedule
    assert "handler-secret-9f4d" not in "".join(logs)
    assert manager.fallback_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("factory_failure", ["raises", "malformed"])
async def test_error_factory_failure_leaves_job_for_lease_recovery(factory_failure):
    manager = PreparedManager()
    sdk = _sdk(manager)
    logs: list[str] = []
    sink = logger.add(logs.append, format="{message}|{extra}", level="DEBUG")
    factory_calls = 0

    async def handler(_job_row, _context):
        sdk.stop()
        raise HandlerSecretError("handler-secret-2c8a")

    def error_disposition(_job_row, _error_class):
        nonlocal factory_calls
        factory_calls += 1
        if factory_failure == "raises":
            raise ValueError("factory-secret-71ab")
        return {"malformed": "factory-secret-71ab"}

    try:
        await sdk.run_prepared(
            handler=handler,
            pre_acquire_guard=_allow_acquire,
            handler_error_disposition=error_disposition,
        )
    finally:
        logger.remove(sink)

    rendered_logs = "".join(logs)
    assert factory_calls == 1
    assert manager.apply_calls == []
    assert manager.fallback_calls == []
    assert "handler-secret-2c8a" not in rendered_logs
    assert "factory-secret-71ab" not in rendered_logs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "apply_failure",
    [RuntimeError("apply-secret-455c"), {"malformed": "apply-secret-455c"}],
    ids=["raises", "malformed-result"],
)
async def test_apply_failure_invokes_no_callback_or_second_transition(apply_failure):
    manager = PreparedManager()
    manager.apply_result = apply_failure
    sdk = _sdk(manager)
    callbacks: list[str] = []
    logs: list[str] = []
    sink = logger.add(logs.append, format="{message}|{extra}", level="DEBUG")

    async def handler(_job_row, _context):
        sdk.stop()
        return _complete()

    async def on_applied(*_args):
        callbacks.append("applied")

    async def on_rejected(*_args):
        callbacks.append("rejected")

    try:
        await sdk.run_prepared(
            handler=handler,
            pre_acquire_guard=_allow_acquire,
            handler_error_disposition=_closed_error_disposition,
            on_disposition_applied=on_applied,
            on_disposition_rejected=on_rejected,
        )
    finally:
        logger.remove(sink)

    assert len(manager.apply_calls) == 1
    assert callbacks == []
    assert manager.fallback_calls == []
    assert "apply-secret-455c" not in "".join(logs)


@pytest.mark.asyncio
@pytest.mark.parametrize("callback_behavior", ["error", "timeout", "cancel"])
async def test_callback_failure_cannot_apply_a_second_transition(callback_behavior):
    manager = PreparedManager()
    sdk = _sdk(
        manager,
        completion_callback_timeout_seconds=0.01,
        completion_callback_max_detached_tasks=1,
    )

    async def handler(_job_row, _context):
        sdk.stop()
        return _complete()

    async def callback(_job_row, _disposition, _result):
        if callback_behavior == "error":
            raise RuntimeError("callback unavailable")
        if callback_behavior == "cancel":
            raise asyncio.CancelledError
        await asyncio.Event().wait()

    run = sdk.run_prepared(
        handler=handler,
        pre_acquire_guard=_allow_acquire,
        handler_error_disposition=_closed_error_disposition,
        on_disposition_applied=callback,
    )
    if callback_behavior == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await run
    else:
        await run
        await asyncio.sleep(0)

    assert len(manager.apply_calls) == 1
    assert manager.fallback_calls == []


@pytest.mark.asyncio
async def test_context_snapshot_is_read_only_and_horizon_updates_authoritative_lease():
    manager = PreparedManager()
    sdk = _sdk(manager)
    observed = []

    async def never_renew(_seconds):
        await asyncio.Event().wait()

    sdk._sleep = never_renew

    async def handler(_job_row, context):
        initial = context.snapshot()
        assert initial.worker_id == "worker-1"
        assert initial.lease_id == "lease-1"
        assert initial.leased_until == LEASED_UNTIL
        assert initial.renewal_lost is False
        with pytest.raises(FrozenInstanceError):
            initial.renewal_lost = True
        ensured = await context.ensure_lease_horizon(120)
        observed.append((ensured, context.snapshot(), context.renewal_lost))
        sdk.stop()
        return _infrastructure_defer()

    await sdk.run_prepared(
        handler=handler,
        pre_acquire_guard=_allow_acquire,
        handler_error_disposition=_closed_error_disposition,
    )

    assert observed == [
        (
            True,
            worker_sdk_module.WorkerLeaseSnapshot(
                worker_id="worker-1",
                lease_id="lease-1",
                leased_until=RENEWED_UNTIL,
                renewal_lost=False,
            ),
            False,
        )
    ]
    assert len(manager.horizon_calls) == 1
    assert manager.horizon_calls[0].minimum_seconds == 120


@pytest.mark.asyncio
async def test_auto_renew_updates_observable_snapshot_from_typed_horizon_result():
    manager = PreparedManager()
    sdk = _sdk(manager)
    sdk._max_iters = 1
    sdk._sleep = lambda _seconds: asyncio.sleep(0)
    observed = []

    async def handler(_job_row, context):
        while not manager.horizon_calls:
            await asyncio.sleep(0)
        observed.append(context.snapshot())
        sdk.stop()
        return _infrastructure_defer()

    await sdk.run_prepared(
        handler=handler,
        pre_acquire_guard=_allow_acquire,
        handler_error_disposition=_closed_error_disposition,
    )

    assert observed[0].leased_until == RENEWED_UNTIL
    assert observed[0].renewal_lost is False
    assert len(manager.horizon_calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("renewal_failure", ["false", "exception"])
async def test_auto_renew_loss_is_sticky_visible_and_stops(renewal_failure):
    manager = PreparedManager()
    if renewal_failure == "false":
        manager.horizon_result = LeaseHorizonResult.no_transition(
            NoTransitionReason.STALE_LEASE,
            leased_until=LEASED_UNTIL,
        )
    else:
        manager.horizon_result = RuntimeError("renewal backend private detail")
    sdk = _sdk(manager)
    sdk._sleep = lambda _seconds: asyncio.sleep(0)
    observed = []

    async def handler(_job_row, context):
        while not manager.horizon_calls:
            await asyncio.sleep(0)
        for _ in range(3):
            await asyncio.sleep(0)
        observed.append((context.renewal_lost, context.snapshot()))
        sdk.stop()
        return _infrastructure_defer()

    await sdk.run_prepared(
        handler=handler,
        pre_acquire_guard=_allow_acquire,
        handler_error_disposition=_closed_error_disposition,
    )

    assert len(manager.horizon_calls) == 1
    assert observed[0][0] is True
    assert observed[0][1].renewal_lost is True


@pytest.mark.asyncio
@pytest.mark.parametrize("horizon_failure", ["stale", "backend"])
async def test_manual_horizon_failure_returns_false_and_remains_visible(
    horizon_failure,
):
    manager = PreparedManager()
    if horizon_failure == "stale":
        manager.horizon_result = LeaseHorizonResult.no_transition(
            NoTransitionReason.STALE_LEASE,
            leased_until=LEASED_UNTIL,
        )
    else:
        manager.horizon_result = RuntimeError("horizon backend private detail")
    sdk = _sdk(manager)

    async def never_renew(_seconds):
        await asyncio.Event().wait()

    sdk._sleep = never_renew
    observed = []

    async def handler(_job_row, context):
        first = await context.ensure_lease_horizon(90)
        second = context.renewal_lost
        observed.append((first, second, context.snapshot()))
        sdk.stop()
        return _infrastructure_defer()

    await sdk.run_prepared(
        handler=handler,
        pre_acquire_guard=_allow_acquire,
        handler_error_disposition=_closed_error_disposition,
    )

    assert observed[0][0] is False
    assert observed[0][1] is True
    assert observed[0][2].renewal_lost is True


@pytest.mark.asyncio
async def test_handler_cancellation_cancels_and_awaits_renewal_then_reraises():
    manager = PreparedManager()
    sdk = _sdk(manager)
    renewal_started = asyncio.Event()
    renewal_cancelled = asyncio.Event()
    handler_started = asyncio.Event()

    async def blocking_sleep(_seconds):
        renewal_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            renewal_cancelled.set()

    sdk._sleep = blocking_sleep

    async def handler(_job_row, _context):
        await renewal_started.wait()
        handler_started.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(
        sdk.run_prepared(
            handler=handler,
            pre_acquire_guard=_allow_acquire,
            handler_error_disposition=_closed_error_disposition,
        )
    )
    await asyncio.wait_for(handler_started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert renewal_cancelled.is_set()
    assert manager.apply_calls == []
    assert manager.fallback_calls == []
