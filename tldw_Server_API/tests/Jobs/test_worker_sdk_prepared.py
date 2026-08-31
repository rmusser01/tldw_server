import asyncio
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest
from loguru import logger

from tldw_Server_API.app.core.Jobs import worker_sdk as worker_sdk_module
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    ApplyPreparedDispositionCommand,
    EnsureLeaseHorizonCommand,
    LeaseHorizonResult,
    NoTransitionReason,
    OperationOutcome,
    PreparedDispositionOrigin,
    PreparedDispositionResult,
    PreparedJobDisposition,
)
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK

DELIVERY_ID = "de305d54-75b4-431b-adb2-eb6b9e546014"
ATTEMPT_ID = "123e4567-e89b-42d3-a456-426614174000"
LEASED_UNTIL = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
NOW = LEASED_UNTIL - timedelta(seconds=30)
RENEWED_UNTIL = LEASED_UNTIL + timedelta(minutes=5)
_DEFAULT_HORIZON_RESULT = object()
_MISSING_GUARANTEE = object()


class _IntSubclass(int):
    pass


def _raw_horizon_result(
    *,
    outcome: object = OperationOutcome.APPLIED,
    ensured: object = True,
    guaranteed_seconds: object = _MISSING_GUARANTEE,
    leased_until: object = RENEWED_UNTIL,
    no_transition_reason: object = None,
) -> LeaseHorizonResult:
    """Build malformed typed evidence without weakening production invariants."""

    result = object.__new__(LeaseHorizonResult)
    object.__setattr__(result, "outcome", outcome)
    object.__setattr__(result, "ensured", ensured)
    object.__setattr__(result, "leased_until", leased_until)
    object.__setattr__(result, "no_transition_reason", no_transition_reason)
    if guaranteed_seconds is not _MISSING_GUARANTEE:
        object.__setattr__(result, "guaranteed_seconds", guaranteed_seconds)
    return result


def _job(*, leased_until: object = LEASED_UNTIL) -> dict[str, object]:
    return {
        "id": 17,
        "uuid": "bffb6b56-0db3-4fe8-b2f8-a77d6131bf6d",
        "domain": "admin_webhooks",
        "queue": "delivery",
        "job_type": "admin_webhook_delivery",
        "payload": {"delivery_id": DELIVERY_ID},
        "worker_id": "worker-1",
        "lease_id": "lease-1",
        "leased_until": leased_until,
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
        self.expected_lease_seconds = 4
        self.acquire_calls: list[dict[str, object]] = []
        self.apply_calls: list[object] = []
        self.horizon_calls: list[object] = []
        self.events: list[tuple[str, object]] = []
        self.fallback_calls: list[str] = []
        self.apply_result: object = _applied_result()
        self.horizon_result: object = _DEFAULT_HORIZON_RESULT
        self.horizon_results: list[object] = []
        self.before_horizon_cap_read = None
        self.after_horizon_result = None

    def acquire_next_job(self, **kwargs):
        assert kwargs == {
            "domain": "admin_webhooks",
            "queue": "delivery",
            "lease_seconds": self.expected_lease_seconds,
            "worker_id": "worker-1",
            "owner_user_id": None,
            "job_type": None,
        }
        self.acquire_calls.append(kwargs)
        return self.jobs.pop(0) if self.jobs else None

    def apply_prepared_disposition(self, command):
        assert isinstance(command, ApplyPreparedDispositionCommand)
        assert command.job_id == 17
        assert command.domain == "admin_webhooks"
        assert command.queue == "delivery"
        assert command.job_type == "admin_webhook_delivery"
        assert command.expected_payload == {"delivery_id": DELIVERY_ID}
        assert command.worker_id == "worker-1"
        assert command.lease_id == "lease-1"
        assert isinstance(command.disposition, PreparedJobDisposition)
        self.apply_calls.append(command)
        if isinstance(self.apply_result, BaseException):
            raise self.apply_result
        return self.apply_result

    def ensure_lease_horizon(self, command):
        assert isinstance(command, EnsureLeaseHorizonCommand)
        assert command.job_id == 17
        assert command.domain == "admin_webhooks"
        assert command.queue == "delivery"
        assert command.job_type == "admin_webhook_delivery"
        assert command.expected_payload == {"delivery_id": DELIVERY_ID}
        assert command.worker_id == "worker-1"
        assert command.lease_id == "lease-1"
        assert command.minimum_seconds > 0
        self.horizon_calls.append(command)
        self.events.append(("ensure", command.minimum_seconds))
        if self.before_horizon_cap_read is not None:
            self.before_horizon_cap_read()
        manager_cap = max(
            1,
            int(worker_sdk_module.os.getenv("JOBS_LEASE_MAX_SECONDS", "3600") or "3600"),
        )
        result = (
            self.horizon_results.pop(0)
            if self.horizon_results
            else self.horizon_result
        )
        if result is _DEFAULT_HORIZON_RESULT:
            result = LeaseHorizonResult.applied(
                leased_until=RENEWED_UNTIL,
                guaranteed_seconds=min(command.minimum_seconds, manager_cap),
            )
        if self.after_horizon_result is not None:
            self.after_horizon_result()
        if isinstance(result, BaseException):
            raise result
        return result

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
    config_values = {
        "domain": "admin_webhooks",
        "queue": "delivery",
        "worker_id": "worker-1",
        "lease_seconds": 4,
        "renew_threshold_seconds": 3,
        "renew_jitter_seconds": 0,
    }
    config_values.update(config_overrides)
    manager.expected_lease_seconds = config_values["lease_seconds"]
    sdk = WorkerSDK(manager, WorkerConfig(**config_values))
    sdk._monotonic = lambda: 0.0
    return sdk


async def _allow_acquire() -> bool:
    return True


def _closed_error_disposition(
    _job_row: dict[str, object],
    _error_class: type[BaseException],
) -> PreparedJobDisposition:
    return _infrastructure_defer()


async def _run_once(
    sdk: WorkerSDK,
    handler,
    *,
    error_disposition=_closed_error_disposition,
    on_applied=None,
    on_rejected=None,
) -> None:
    async def one_job(job_row, context):
        try:
            return await handler(job_row, context)
        finally:
            sdk.stop()

    await sdk.run_prepared(
        handler=one_job,
        pre_acquire_guard=_allow_acquire,
        handler_error_disposition=error_disposition,
        on_disposition_applied=on_applied,
        on_disposition_rejected=on_rejected,
    )


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
async def test_pre_acquire_guard_cancellation_propagates_without_acquiring():
    manager = PreparedManager()
    sdk = _sdk(manager)

    async def cancelled_guard() -> bool:
        raise asyncio.CancelledError

    async def handler(_job_row, _context):
        pytest.fail("handler must not run after guard cancellation")

    with pytest.raises(asyncio.CancelledError):
        await sdk.run_prepared(
            handler=handler,
            pre_acquire_guard=cancelled_guard,
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
async def test_error_factory_mutation_cannot_change_cas_facts_or_callback_job():
    manager = PreparedManager()
    sdk = _sdk(manager)
    factory_jobs = []
    callback_jobs = []
    logs: list[str] = []
    sink = logger.add(logs.append, format="{message}|{extra}", level="DEBUG")

    async def handler(_job_row, _context):
        raise HandlerSecretError("handler-secret-cas")

    def mutate_factory(job_row, error_class):
        assert error_class is HandlerSecretError
        factory_jobs.append(job_row)
        job_row["id"] = 991
        job_row["domain"] = "factory-secret-domain"
        job_row["queue"] = "factory-secret-queue"
        job_row["job_type"] = "factory-secret-type"
        job_row["payload"]["delivery_id"] = (
            "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        )
        job_row["lease_id"] = "factory-secret-lease"
        sdk.cfg.worker_id = "factory-secret-worker"
        return _complete()

    async def on_applied(job_row, _disposition, _result):
        callback_jobs.append(job_row)

    try:
        await _run_once(
            sdk,
            handler,
            error_disposition=mutate_factory,
            on_applied=on_applied,
        )
    finally:
        logger.remove(sink)

    assert len(manager.apply_calls) == 1
    command = manager.apply_calls[0]
    assert command.job_id == 17
    assert command.domain == "admin_webhooks"
    assert command.queue == "delivery"
    assert command.job_type == "admin_webhook_delivery"
    assert command.expected_payload == {"delivery_id": DELIVERY_ID}
    assert command.worker_id == "worker-1"
    assert command.lease_id == "lease-1"
    assert callback_jobs == [_job()]
    assert callback_jobs[0] is not factory_jobs[0]
    assert "factory-secret" not in "".join(logs)
    assert len(manager.apply_calls) == 1
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
@pytest.mark.parametrize(
    "initial_horizon",
    [
        None,
        "not-a-timestamp",
        NOW - timedelta(seconds=1),
        NOW + timedelta(seconds=2),
        NOW + timedelta(minutes=10),
    ],
    ids=["missing", "malformed", "expired", "inside-threshold", "apparently-safe"],
)
async def test_first_renewal_ensures_before_sleep_regardless_of_absolute_evidence(
    initial_horizon,
):
    manager = PreparedManager()
    manager.jobs = [_job(leased_until=initial_horizon)]
    sdk = _sdk(manager)
    sdk._max_iters = 1

    async def record_sleep(seconds):
        manager.events.append(("sleep", seconds))
        await asyncio.sleep(0)

    sdk._sleep = record_sleep

    async def handler(_job_row, _context):
        while not manager.horizon_calls:
            await asyncio.sleep(0)
        return _infrastructure_defer()

    await _run_once(sdk, handler)

    assert manager.events[0][0] == "ensure"
    assert len(manager.horizon_calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "worker_clock_offset",
    [-60, 60],
    ids=["worker-behind", "worker-ahead"],
)
async def test_relative_renewal_interval_ignores_worker_clock_skew(
    monkeypatch,
    worker_clock_offset,
):
    monkeypatch.setenv("JOBS_LEASE_MAX_SECONDS", "5")
    manager = PreparedManager()
    database_now = LEASED_UNTIL
    capped_until = database_now + timedelta(seconds=5)
    manager.jobs = [_job(leased_until=capped_until)]
    manager.horizon_result = LeaseHorizonResult.applied(
        leased_until=capped_until,
        guaranteed_seconds=5,
    )
    sdk = _sdk(
        manager,
        lease_seconds=30,
        renew_threshold_seconds=10,
        renew_jitter_seconds=0,
    )
    sdk._max_iters = 2
    sdk._utcnow = lambda: database_now + timedelta(seconds=worker_clock_offset)
    sleep_calls = []
    observed_renewal_loss = []

    async def record_sleep(seconds):
        sleep_calls.append(seconds)
        manager.events.append(("sleep", seconds))
        await asyncio.sleep(0)

    sdk._sleep = record_sleep

    async def handler(_job_row, context):
        while len(manager.horizon_calls) < 2 and not context.renewal_lost:
            await asyncio.sleep(0)
        observed_renewal_loss.append(context.renewal_lost)
        return _infrastructure_defer()

    await _run_once(sdk, handler)

    assert manager.events == [
        ("ensure", 5),
        ("sleep", 2.5),
        ("ensure", 5),
    ]
    assert len(manager.horizon_calls) == 2
    assert sleep_calls == [2.5]
    assert observed_renewal_loss == [False]


@pytest.mark.asyncio
async def test_manager_cap_aba_race_schedules_from_returned_guarantee(
    monkeypatch,
):
    monkeypatch.setenv("JOBS_LEASE_MAX_SECONDS", "30")
    manager = PreparedManager()
    manager.before_horizon_cap_read = lambda: monkeypatch.setenv(
        "JOBS_LEASE_MAX_SECONDS", "5"
    )
    manager.after_horizon_result = lambda: monkeypatch.setenv(
        "JOBS_LEASE_MAX_SECONDS", "30"
    )
    sdk = _sdk(
        manager,
        lease_seconds=30,
        renew_threshold_seconds=10,
        renew_jitter_seconds=0,
    )
    sdk._max_iters = 2
    ticks = iter([100.0, 101.0, 200.0, 201.0])
    sdk._monotonic = ticks.__next__
    sleep_calls = []

    async def record_sleep(seconds):
        sleep_calls.append(seconds)
        manager.events.append(("sleep", seconds))
        await asyncio.sleep(0)

    sdk._sleep = record_sleep

    async def handler(_job_row, _context):
        while len(manager.horizon_calls) < 2:
            await asyncio.sleep(0)
        return _infrastructure_defer()

    await _run_once(sdk, handler)

    assert [call.minimum_seconds for call in manager.horizon_calls] == [30, 30]
    assert manager.events == [
        ("ensure", 30),
        ("sleep", 2.0),
        ("ensure", 30),
    ]
    assert sleep_calls == [2.0]
    assert worker_sdk_module.os.getenv("JOBS_LEASE_MAX_SECONDS") == "30"


@pytest.mark.asyncio
async def test_elapsed_ensure_that_consumes_guarantee_fails_closed(monkeypatch):
    monkeypatch.setenv("JOBS_LEASE_MAX_SECONDS", "30")
    manager = PreparedManager()
    manager.horizon_result = _raw_horizon_result(guaranteed_seconds=5)
    sdk = _sdk(
        manager,
        lease_seconds=30,
        renew_threshold_seconds=10,
        renew_jitter_seconds=0,
    )
    sdk._max_iters = 2
    ticks = iter([100.0, 105.0])
    sdk._monotonic = ticks.__next__
    sleep_calls = []
    observed = []
    records: list[dict[str, object]] = []

    async def record_sleep(seconds):
        sleep_calls.append(seconds)
        await asyncio.sleep(0)

    sdk._sleep = record_sleep

    async def handler(_job_row, context):
        while len(manager.horizon_calls) < 2 and not context.renewal_lost:
            await asyncio.sleep(0)
        observed.append(context.renewal_lost)
        return _infrastructure_defer()

    sink = logger.add(
        lambda message: records.append(dict(message.record)),
        level="WARNING",
    )
    try:
        await _run_once(sdk, handler)
    finally:
        logger.remove(sink)

    assert len(manager.horizon_calls) == 1
    assert sleep_calls == []
    assert observed == [True]
    assert len(records) == 1
    assert records[0]["message"] == "Jobs prepared renewal scheduling failed"
    assert records[0]["extra"]["error_type"] == "ValueError"
    assert records[0]["exception"] is None


@pytest.mark.asyncio
async def test_one_second_effective_lease_uses_positive_interval(monkeypatch):
    monkeypatch.setenv("JOBS_LEASE_MAX_SECONDS", "1")
    manager = PreparedManager()
    sdk = _sdk(
        manager,
        lease_seconds=30,
        renew_threshold_seconds=10,
        renew_jitter_seconds=0,
    )
    sdk._max_iters = 2
    sleep_calls = []

    async def record_sleep(seconds):
        sleep_calls.append(seconds)
        await asyncio.sleep(0)

    sdk._sleep = record_sleep

    async def handler(_job_row, _context):
        while len(manager.horizon_calls) < 2:
            await asyncio.sleep(0)
        return _infrastructure_defer()

    await _run_once(sdk, handler)

    assert [call.minimum_seconds for call in manager.horizon_calls] == [1, 1]
    assert sleep_calls == [0.5]


@pytest.mark.asyncio
async def test_prepared_renewal_reloads_current_cap_before_each_ensure(monkeypatch):
    monkeypatch.setenv("JOBS_LEASE_MAX_SECONDS", "30")
    manager = PreparedManager()
    sdk = _sdk(
        manager,
        lease_seconds=30,
        renew_threshold_seconds=10,
        renew_jitter_seconds=0,
    )
    sdk._max_iters = 3
    sleep_calls = []

    async def record_sleep(seconds):
        sleep_calls.append(seconds)
        if len(sleep_calls) == 1:
            monkeypatch.setenv("JOBS_LEASE_MAX_SECONDS", "5")
        await asyncio.sleep(0)

    sdk._sleep = record_sleep

    async def handler(_job_row, _context):
        while len(manager.horizon_calls) < 3:
            await asyncio.sleep(0)
        return _infrastructure_defer()

    await _run_once(sdk, handler)

    assert [call.minimum_seconds for call in manager.horizon_calls] == [30, 5, 5]
    assert sleep_calls == [20.0, 2.5]


@pytest.mark.asyncio
async def test_prepared_renewal_jitter_only_moves_deadline_earlier(monkeypatch):
    monkeypatch.setenv("JOBS_LEASE_MAX_SECONDS", "30")
    manager = PreparedManager()
    manager.jobs = [_job(leased_until=NOW + timedelta(seconds=30))]
    sdk = _sdk(
        manager,
        lease_seconds=30,
        renew_threshold_seconds=10,
        renew_jitter_seconds=5,
    )
    sdk._max_iters = 2
    sleep_calls = []

    monkeypatch.setattr(
        worker_sdk_module.secrets,
        "randbelow",
        lambda upper: upper - 1,
    )

    async def record_sleep(seconds):
        sleep_calls.append(seconds)
        await asyncio.sleep(0)

    sdk._sleep = record_sleep

    async def handler(_job_row, _context):
        while len(manager.horizon_calls) < 2:
            await asyncio.sleep(0)
        return _infrastructure_defer()

    await _run_once(sdk, handler)

    assert manager.events == [
        ("ensure", 30),
        ("ensure", 30),
    ]
    assert sleep_calls == [15.0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config_overrides", "lease_cap", "expected_error_type"),
    [
        ({"lease_seconds": "lease-config-secret"}, "3600", "ValueError"),
        ({}, "lease-cap-secret", "ValueError"),
        (
            {"renew_threshold_seconds": "threshold-config-secret"},
            "3600",
            "ValueError",
        ),
        (
            {"renew_jitter_seconds": "jitter-config-secret"},
            "3600",
            "ValueError",
        ),
        ({"renew_threshold_seconds": 10**400}, "3600", "OverflowError"),
    ],
    ids=["lease", "cap", "threshold", "jitter", "huge-threshold"],
)
async def test_invalid_prepared_renewal_configuration_fails_closed(
    monkeypatch,
    config_overrides,
    lease_cap,
    expected_error_type,
):
    monkeypatch.setenv("JOBS_LEASE_MAX_SECONDS", lease_cap)
    manager = PreparedManager()
    sdk = _sdk(manager, **config_overrides)
    sdk._max_iters = 1
    sleep_calls = []
    records: list[dict[str, object]] = []
    context = worker_sdk_module.WorkerExecutionContext(
        manager,
        _job(),
        worker_id="worker-1",
    )

    async def record_sleep(seconds):
        sleep_calls.append(seconds)
        await asyncio.sleep(0)

    sdk._sleep = record_sleep
    sink = logger.add(
        lambda message: records.append(dict(message.record)),
        level="WARNING",
    )
    raised = None
    try:
        try:
            await sdk._auto_renew_prepared(context, asyncio.Event())
        except Exception as exc:  # noqa: BLE001 - assertion captures boundary leakage
            raised = type(exc)
    finally:
        logger.remove(sink)

    assert raised is None
    assert context.renewal_lost is True
    assert context.snapshot().renewal_lost is True
    assert manager.horizon_calls == []
    assert sleep_calls == []
    assert len(records) == 1
    record = records[0]
    assert record["message"] == "Jobs prepared renewal configuration failed"
    assert record["extra"]["error_type"] == expected_error_type
    assert record["exception"] is None
    rendered_record = repr(record)
    assert "config-secret" not in rendered_record
    assert "cap-secret" not in rendered_record
    assert "invalid literal" not in rendered_record


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_kind", "expected_error_type"),
    [
        ("huge-guarantee", "OverflowError"),
        ("backward-monotonic", "ValueError"),
        ("nonfinite-monotonic-start", "ValueError"),
        ("nonfinite-monotonic-finish", "ValueError"),
        ("jitter-error", "RuntimeError"),
        ("jitter-negative", "ValueError"),
        ("jitter-above-bound", "ValueError"),
        ("jitter-wrong-type", "ValueError"),
        ("sleep-error", "RuntimeError"),
    ],
)
async def test_prepared_renewal_arithmetic_and_jitter_fail_closed_once(
    monkeypatch,
    failure_kind,
    expected_error_type,
):
    manager = PreparedManager()
    sdk = _sdk(manager, renew_jitter_seconds=1)
    sdk._max_iters = 2
    context = worker_sdk_module.WorkerExecutionContext(
        manager,
        _job(),
        worker_id="worker-1",
    )
    sleep_calls = []
    records: list[dict[str, object]] = []

    if failure_kind == "huge-guarantee":
        manager.horizon_result = _raw_horizon_result(
            guaranteed_seconds=10**400
        )
    elif failure_kind == "backward-monotonic":
        ticks = iter([100.0, 99.0])
        sdk._monotonic = ticks.__next__
    elif failure_kind == "nonfinite-monotonic-start":
        sdk._monotonic = lambda: float("nan")
    elif failure_kind == "nonfinite-monotonic-finish":
        ticks = iter([100.0, float("inf")])
        sdk._monotonic = ticks.__next__
    elif failure_kind == "jitter-error":

        def fail_jitter(_upper_bound):
            raise RuntimeError("jitter-private-secret")

        monkeypatch.setattr(worker_sdk_module.secrets, "randbelow", fail_jitter)
    elif failure_kind == "jitter-negative":
        monkeypatch.setattr(worker_sdk_module.secrets, "randbelow", lambda _upper: -1)
    elif failure_kind == "jitter-above-bound":
        monkeypatch.setattr(
            worker_sdk_module.secrets,
            "randbelow",
            lambda upper: upper,
        )
    elif failure_kind == "jitter-wrong-type":
        monkeypatch.setattr(worker_sdk_module.secrets, "randbelow", lambda _upper: True)

    async def record_sleep(seconds):
        sleep_calls.append(seconds)
        if failure_kind == "sleep-error":
            raise RuntimeError("sleep-private-secret")
        await asyncio.sleep(0)

    sdk._sleep = record_sleep
    sink = logger.add(
        lambda message: records.append(dict(message.record)),
        level="WARNING",
    )
    raised = None
    try:
        try:
            await sdk._auto_renew_prepared(context, asyncio.Event())
        except Exception as exc:  # noqa: BLE001 - assertion captures boundary leakage
            raised = type(exc)
    finally:
        logger.remove(sink)

    assert raised is None
    assert context.renewal_lost is True
    assert context.snapshot().renewal_lost is True
    expected_ensure_calls = 0 if failure_kind == "nonfinite-monotonic-start" else 1
    assert len(manager.horizon_calls) == expected_ensure_calls
    assert len(sleep_calls) == (1 if failure_kind == "sleep-error" else 0)
    assert len(records) == 1
    record = records[0]
    assert record["message"] == "Jobs prepared renewal scheduling failed"
    assert record["extra"]["error_type"] == expected_error_type
    assert record["exception"] is None
    rendered_record = repr(record)
    assert "jitter-private-secret" not in rendered_record
    assert "sleep-private-secret" not in rendered_record
    assert "cannot convert" not in rendered_record


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformed_result",
    [
        object(),
        _raw_horizon_result(
            guaranteed_seconds=90,
            leased_until=None,
        ),
    ],
    ids=["wrong-object", "applied-without-deadline"],
)
async def test_malformed_horizon_result_fails_closed_and_stays_visible(
    malformed_result,
):
    manager = PreparedManager()
    manager.horizon_result = malformed_result
    sdk = _sdk(manager)

    async def never_renew(_seconds):
        await asyncio.Event().wait()

    sdk._sleep = never_renew
    observed = []

    async def handler(_job_row, context):
        observed.append(
            (
                await context.ensure_lease_horizon(90),
                context.renewal_lost,
                context.snapshot(),
            )
        )
        return _infrastructure_defer()

    await _run_once(sdk, handler)

    assert observed[0][0] is False
    assert observed[0][1] is True
    assert observed[0][2].renewal_lost is True
    assert len(manager.apply_calls) == 1
    assert manager.fallback_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "horizon_result",
    [
        _raw_horizon_result(
            outcome=OperationOutcome.NO_TRANSITION,
            ensured=False,
            leased_until=None,
            no_transition_reason=NoTransitionReason.MISSING,
            guaranteed_seconds=None,
        ),
        _raw_horizon_result(
            outcome=OperationOutcome.NO_TRANSITION,
            ensured=False,
            leased_until=LEASED_UNTIL,
            no_transition_reason=NoTransitionReason.STALE_LEASE,
            guaranteed_seconds=None,
        ),
        _raw_horizon_result(
            outcome=OperationOutcome.BACKEND_CONFLICT,
            ensured=False,
            leased_until=None,
            no_transition_reason=None,
            guaranteed_seconds=None,
        ),
        _raw_horizon_result(
            outcome=OperationOutcome.BACKEND_CONFLICT,
            ensured=False,
            leased_until=LEASED_UNTIL,
            no_transition_reason=None,
            guaranteed_seconds=None,
        ),
    ],
    ids=[
        "no-transition-without-deadline",
        "no-transition-with-deadline",
        "conflict-without-deadline",
        "conflict-with-deadline",
    ],
)
async def test_private_horizon_helper_accepts_valid_non_applied_matrix_as_loss(
    horizon_result,
):
    manager = PreparedManager()
    manager.horizon_result = horizon_result
    context = worker_sdk_module.WorkerExecutionContext(
        manager,
        _job(),
        worker_id="worker-1",
    )
    records: list[dict[str, object]] = []
    sink = logger.add(
        lambda message: records.append(dict(message.record)),
        level="WARNING",
    )
    try:
        guaranteed_seconds = await context._ensure_lease_horizon_typed(30)
    finally:
        logger.remove(sink)

    assert guaranteed_seconds is None
    assert context.renewal_lost is True
    assert records == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "horizon_result",
    [
        _raw_horizon_result(
            outcome=OperationOutcome.BACKEND_ERROR,
            ensured=False,
            leased_until=None,
            guaranteed_seconds=None,
        ),
        _raw_horizon_result(
            outcome="applied",
            ensured=True,
            guaranteed_seconds=30,
        ),
        _raw_horizon_result(ensured=1, guaranteed_seconds=30),
        _raw_horizon_result(
            guaranteed_seconds=30,
            no_transition_reason=NoTransitionReason.STALE_LEASE,
        ),
        _raw_horizon_result(
            outcome=OperationOutcome.NO_TRANSITION,
            ensured=True,
            leased_until=None,
            no_transition_reason=NoTransitionReason.MISSING,
            guaranteed_seconds=None,
        ),
        _raw_horizon_result(
            outcome=OperationOutcome.NO_TRANSITION,
            ensured=False,
            leased_until=None,
            no_transition_reason=None,
            guaranteed_seconds=None,
        ),
        _raw_horizon_result(
            outcome=OperationOutcome.NO_TRANSITION,
            ensured=False,
            leased_until=None,
            no_transition_reason="forged-private-secret",
            guaranteed_seconds=None,
        ),
        _raw_horizon_result(
            outcome=OperationOutcome.NO_TRANSITION,
            ensured=False,
            leased_until=None,
            no_transition_reason=NoTransitionReason.MISSING,
            guaranteed_seconds=30,
        ),
        _raw_horizon_result(
            outcome=OperationOutcome.BACKEND_CONFLICT,
            ensured=True,
            leased_until=None,
            guaranteed_seconds=None,
        ),
        _raw_horizon_result(
            outcome=OperationOutcome.BACKEND_CONFLICT,
            ensured=False,
            leased_until=None,
            no_transition_reason=NoTransitionReason.MISSING,
            guaranteed_seconds=None,
        ),
        _raw_horizon_result(
            outcome=OperationOutcome.BACKEND_CONFLICT,
            ensured=False,
            leased_until=None,
            guaranteed_seconds=30,
        ),
        _raw_horizon_result(
            outcome=OperationOutcome.BACKEND_CONFLICT,
            ensured=False,
            leased_until=datetime(2026, 8, 28),
            guaranteed_seconds=None,
        ),
    ],
    ids=[
        "unsupported-outcome",
        "string-outcome",
        "non-bool-ensured",
        "applied-with-reason",
        "no-transition-ensured",
        "no-transition-missing-reason",
        "no-transition-string-reason",
        "no-transition-with-guarantee",
        "conflict-ensured",
        "conflict-with-reason",
        "conflict-with-guarantee",
        "conflict-naive-deadline",
    ],
)
async def test_private_horizon_helper_revalidates_forged_state_matrix(
    horizon_result,
):
    manager = PreparedManager()
    manager.horizon_result = horizon_result
    context = worker_sdk_module.WorkerExecutionContext(
        manager,
        _job(),
        worker_id="worker-1",
    )
    records: list[dict[str, object]] = []
    sink = logger.add(
        lambda message: records.append(dict(message.record)),
        level="WARNING",
    )
    try:
        guaranteed_seconds = await context._ensure_lease_horizon_typed(30)
    finally:
        logger.remove(sink)

    assert guaranteed_seconds is None
    assert context.renewal_lost is True
    assert len(records) == 1
    record = records[0]
    assert record["message"] == "Jobs prepared lease horizon failed"
    assert record["extra"]["error_type"] == "ValueError"
    assert record["exception"] is None
    assert "forged-private-secret" not in repr(record)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("horizon_result", "expected_error_type"),
    [
        (object(), "TypeError"),
        (_raw_horizon_result(), "TypeError"),
        (_raw_horizon_result(guaranteed_seconds=0), "ValueError"),
        (_raw_horizon_result(guaranteed_seconds=-1), "ValueError"),
        (_raw_horizon_result(guaranteed_seconds=True), "ValueError"),
        (_raw_horizon_result(guaranteed_seconds=1.5), "ValueError"),
        (_raw_horizon_result(guaranteed_seconds=_IntSubclass(5)), "ValueError"),
        (
            _raw_horizon_result(guaranteed_seconds="guarantee-private-detail"),
            "ValueError",
        ),
        (
            _raw_horizon_result(guaranteed_seconds=5, leased_until=None),
            "ValueError",
        ),
        (RuntimeError("manager-private-detail"), "RuntimeError"),
    ],
    ids=[
        "wrong-object",
        "missing-guarantee",
        "zero-guarantee",
        "negative-guarantee",
        "bool-guarantee",
        "float-guarantee",
        "int-subclass-guarantee",
        "string-guarantee",
        "missing-deadline",
        "manager-exception",
    ],
)
async def test_private_typed_horizon_helper_logs_class_only_and_fails_closed(
    horizon_result,
    expected_error_type,
):
    manager = PreparedManager()
    manager.horizon_result = horizon_result
    context = worker_sdk_module.WorkerExecutionContext(
        manager,
        _job(),
        worker_id="worker-1",
    )
    records: list[dict[str, object]] = []
    sink = logger.add(
        lambda message: records.append(dict(message.record)),
        level="WARNING",
    )
    try:
        guaranteed_seconds = await context._ensure_lease_horizon_typed(30)
    finally:
        logger.remove(sink)

    assert guaranteed_seconds is None
    assert context.renewal_lost is True
    assert len(records) == 1
    record = records[0]
    assert record["message"] == "Jobs prepared lease horizon failed"
    assert record["extra"]["error_type"] == expected_error_type
    assert record["exception"] is None
    rendered_record = repr(record)
    assert "guarantee-private-detail" not in rendered_record
    assert "manager-private-detail" not in rendered_record


@pytest.mark.asyncio
async def test_context_rejects_applied_horizon_shorter_than_requested_minimum():
    manager = PreparedManager()
    manager.horizon_result = LeaseHorizonResult.applied(
        leased_until=RENEWED_UNTIL,
        guaranteed_seconds=30,
    )
    context = worker_sdk_module.WorkerExecutionContext(
        manager,
        _job(),
        worker_id="worker-1",
    )

    ensured = await context.ensure_lease_horizon(60)

    assert ensured is False
    assert context.renewal_lost is True
    assert manager.horizon_calls[0].minimum_seconds == 60


@pytest.mark.asyncio
async def test_horizon_failure_remains_sticky_after_later_success():
    manager = PreparedManager()
    manager.horizon_results = [
        LeaseHorizonResult.no_transition(
            NoTransitionReason.STALE_LEASE,
            leased_until=LEASED_UNTIL,
        ),
        LeaseHorizonResult.applied(
            leased_until=RENEWED_UNTIL,
            guaranteed_seconds=90,
        ),
    ]
    context = worker_sdk_module.WorkerExecutionContext(
        manager,
        _job(),
        worker_id="worker-1",
    )

    first = await context.ensure_lease_horizon(90)
    second = await context.ensure_lease_horizon(90)
    snapshot = context.snapshot()

    assert (first, second) == (False, True)
    assert snapshot.leased_until == RENEWED_UNTIL
    assert snapshot.renewal_lost is True
    assert len(manager.horizon_calls) == 2


@pytest.mark.asyncio
async def test_context_snapshot_is_read_only_and_horizon_updates_authoritative_lease():
    manager = PreparedManager()
    context = worker_sdk_module.WorkerExecutionContext(
        manager,
        _job(),
        worker_id="worker-1",
    )
    initial = context.snapshot()

    assert initial.worker_id == "worker-1"
    assert initial.lease_id == "lease-1"
    assert initial.leased_until == LEASED_UNTIL
    assert initial.renewal_lost is False
    with pytest.raises(FrozenInstanceError):
        initial.renewal_lost = True
    ensured = await context.ensure_lease_horizon(120)

    assert ensured is True
    assert context.snapshot() == worker_sdk_module.WorkerLeaseSnapshot(
        worker_id="worker-1",
        lease_id="lease-1",
        leased_until=RENEWED_UNTIL,
        renewal_lost=False,
    )
    assert context.renewal_lost is False
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


@pytest.mark.asyncio
async def test_outer_cancellation_during_resistant_renewal_cleanup_propagates():
    manager = PreparedManager()
    sdk = _sdk(manager)
    renewal_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    guard_calls = 0

    async def guard() -> bool:
        nonlocal guard_calls
        guard_calls += 1
        if guard_calls > 1:
            sdk.stop()
            return False
        return True

    async def cancellation_resistant_sleep(_seconds):
        renewal_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_started.set()
            while not release_cleanup.is_set():
                try:
                    await release_cleanup.wait()
                except asyncio.CancelledError:
                    continue

    sdk._sleep = cancellation_resistant_sleep

    async def handler(_job_row, _context):
        await renewal_started.wait()
        return _complete()

    task = asyncio.create_task(
        sdk.run_prepared(
            handler=handler,
            pre_acquire_guard=guard,
            handler_error_disposition=_closed_error_disposition,
        )
    )
    await asyncio.wait_for(cleanup_started.wait(), timeout=1)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    release_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert guard_calls == 1
    assert len(manager.acquire_calls) == 1
    assert len(manager.apply_calls) == 1
    assert manager.fallback_calls == []


@pytest.mark.asyncio
async def test_same_turn_child_and_outer_cancellation_preserves_outer_cancel():
    manager = PreparedManager()
    sdk = _sdk(manager)
    renewal_started = asyncio.Event()
    child_cleanup_started = asyncio.Event()
    release_child_cleanup = asyncio.Event()
    child_cleanup_complete = asyncio.Event()
    guard_calls = 0
    baseline_tasks = set(asyncio.all_tasks())

    async def guard() -> bool:
        nonlocal guard_calls
        guard_calls += 1
        if guard_calls > 1:
            sdk.stop()
            return False
        return True

    async def cancellation_reraising_sleep(_seconds):
        renewal_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            child_cleanup_started.set()
            await release_child_cleanup.wait()
            child_cleanup_complete.set()
            raise

    sdk._sleep = cancellation_reraising_sleep

    async def handler(_job_row, _context):
        await renewal_started.wait()
        return _complete()

    task = asyncio.create_task(
        sdk.run_prepared(
            handler=handler,
            pre_acquire_guard=guard,
            handler_error_disposition=_closed_error_disposition,
        )
    )
    await asyncio.wait_for(child_cleanup_started.wait(), timeout=1)
    loop = asyncio.get_running_loop()
    loop.call_soon(release_child_cleanup.set)
    loop.call_soon(task.cancel)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    await asyncio.sleep(0)
    leaked_tasks = [
        pending
        for pending in asyncio.all_tasks() - baseline_tasks
        if not pending.done()
    ]
    assert child_cleanup_complete.is_set()
    assert leaked_tasks == []
    assert guard_calls == 1
    assert len(manager.acquire_calls) == 1
    assert len(manager.apply_calls) == 1
    assert manager.fallback_calls == []
