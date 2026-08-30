from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from tldw_Server_API.app.core.Admin_Webhooks import domain
from tldw_Server_API.app.core.Admin_Webhooks import executor as executor_module
from tldw_Server_API.app.core.Admin_Webhooks.config import (
    AdminWebhookMode,
    AdminWebhookSettings,
    WebhookRouteSelection,
)
from tldw_Server_API.app.core.Admin_Webhooks.crypto import WebhookKeyRing
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    AttemptState,
    DeliveryKind,
    DeliveryReasonCode,
    DeliveryState,
    EventSourceKind,
)
from tldw_Server_API.app.core.Admin_Webhooks.executor import (
    AttemptExecutionResult,
    AttemptOutcome,
    AttemptReasonCode,
    DeliveryAttemptExecutor,
)
from tldw_Server_API.app.core.Admin_Webhooks.reconciler import JobsDeliveryQueue
from tldw_Server_API.app.core.Admin_Webhooks.worker import (
    AdminWebhookPreparedHandler,
    WorkerCrashPoint,
)
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
    EventInsert,
    RegistrationInsert,
    RegistrationPatch,
    RegistrationTarget,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    ApplyPreparedDispositionCommand,
    OperationOutcome,
    PreparedDispositionKind,
    PreparedDispositionOrigin,
)
from tldw_Server_API.app.core.Security.egress import URLPolicyResult
from tldw_Server_API.app.core.Security.http_hop import StatusOnlyHTTPHopResponse
from tldw_Server_API.tests.Admin_Webhooks.test_event_expansion import (
    NOW,
    canonical_uuid4,
)


def _token(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class TokenSource:
    def __init__(self, label: str) -> None:
        self.label = label
        self.index = 0

    def __call__(self) -> str:
        value = _token(f"{self.label}:{self.index}")
        self.index += 1
        return value


class MutableClock:
    def __init__(self, current: datetime = NOW) -> None:
        self.current = current

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: int) -> None:
        self.current += timedelta(seconds=seconds)


class FakeContext:
    def __init__(self, acquired: dict, order: list[str] | None = None) -> None:
        self.acquired = acquired
        self.order = order if order is not None else []
        self.ensure_calls: list[int] = []
        self.ensure_result = True

    async def ensure_lease_horizon(self, seconds: int) -> bool:
        self.order.append("ensure")
        self.ensure_calls.append(seconds)
        return self.ensure_result

    def snapshot(self):
        return SimpleNamespace(
            worker_id=self.acquired["worker_id"],
            lease_id=self.acquired["lease_id"],
            leased_until=self.acquired["leased_until"],
            renewal_lost=False,
        )


class FakeExecutor:
    def __init__(
        self,
        result: AttemptExecutionResult,
        *,
        order: list[str] | None = None,
        before_result=None,
    ) -> None:
        self.result = result
        self.order = order if order is not None else []
        self.before_result = before_result
        self.requests = []

    async def execute(self, request):
        self.order.append("execute")
        self.requests.append(request)
        if self.before_result is not None:
            await self.before_result()
        return self.result


class DeterministicExecutorClock:
    def __init__(self, worker_clock: MutableClock) -> None:
        self.worker_clock = worker_clock
        self.monotonic_value = 100.0

    def utc_now(self) -> datetime:
        return self.worker_clock()

    def monotonic(self) -> float:
        self.monotonic_value += 0.001
        return self.monotonic_value


class RetryingStatusEgress:
    def __init__(self) -> None:
        self.requests: list[object] = []

    async def __call__(self, request: object) -> StatusOnlyHTTPHopResponse:
        self.requests.append(request)
        return StatusOnlyHTTPHopResponse(
            status_code=503,
            latency_ms=12,
            retry_after_seconds=None,
        )


class SimulatedCrash(BaseException):
    pass


class OneShotCrash:
    def __init__(self, point: WorkerCrashPoint) -> None:
        self.point = point
        self.armed = True

    def __call__(self, point: WorkerCrashPoint) -> None:
        if self.armed and point is self.point:
            self.armed = False
            raise SimulatedCrash(point.value)


@dataclass
class WorkerFixture:
    repository: AdminWebhookRepository
    pool: DatabasePool
    manager: JobManager
    ring: WebhookKeyRing
    clock: MutableClock


@pytest_asyncio.fixture
async def worker_fixture(tmp_path: Path) -> WorkerFixture:
    pool = DatabasePool(
        Settings(
            AUTH_MODE="single_user",
            DATABASE_URL=f"sqlite:///{tmp_path / 'worker-auth.db'}",
        )
    )
    await pool.initialize()
    encoded = base64.b64encode(b"w" * 32).decode("ascii")
    fixture = WorkerFixture(
        repository=AdminWebhookRepository(pool),
        pool=pool,
        manager=JobManager(tmp_path / "worker-jobs.db"),
        ring=WebhookKeyRing({"worker-key": encoded}, primary_id="worker-key"),
        clock=MutableClock(),
    )
    try:
        yield fixture
    finally:
        await pool.close()


@pytest.fixture(autouse=True)
def allow_synthetic_worker_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    def allow(_url: str) -> URLPolicyResult:
        return URLPolicyResult(True, resolved_ips=("93.184.216.34",))

    monkeypatch.setattr(domain, "evaluate_platform_webhook_url_policy", allow)
    monkeypatch.setattr(
        executor_module,
        "evaluate_platform_webhook_url_policy",
        allow,
    )


def _settings() -> AdminWebhookSettings:
    return AdminWebhookSettings(
        mode=AdminWebhookMode.ON,
        route_selection=WebhookRouteSelection.CANONICAL,
        registration_limit=100,
        active_limit=25,
        allow_http_dev=False,
        idempotency_ttl_seconds=86_400,
        rollback_window_days=7,
    )


async def _seed_acquired(
    fixture: WorkerFixture,
    label: str,
    *,
    expires_in: int = 3600,
    kind: DeliveryKind = DeliveryKind.AUTOMATIC,
) -> tuple[int, str, dict]:
    repository = fixture.repository
    ring = fixture.ring
    event_id = canonical_uuid4(f"worker-{label}-event")
    delivery_id = canonical_uuid4(f"worker-{label}-delivery")
    event_type = f"worker.{label}"
    async with repository.transaction() as tx:
        webhook_id = await tx.allocate_registration_id()
        target = ring.encrypt_text(
            purpose="registration.target",
            identity={"registration_id": webhook_id, "target_version": 1},
            plaintext="https://hooks.example.com/delivery",
        )
        secret = ring.encrypt_text(
            purpose="registration.secret",
            identity={"registration_id": webhook_id, "secret_version": 1},
            plaintext="whsec_" + "a" * 64,
        )
        await tx.insert_registration(
            RegistrationInsert(
                id=webhook_id,
                description=label,
                target=RegistrationTarget(
                    protected=target,
                    hostname="hooks.example.com",
                    display="https://hooks.example.com",
                ),
                event_types=(event_type,),
                active=kind is DeliveryKind.AUTOMATIC,
                timeout_seconds=10,
                secret=secret,
                secret_rotation_required=False,
                actor_user_id=7,
                now=fixture.clock() - timedelta(minutes=1),
            )
        )
        body = b'{"id":1}'
        captured = await tx.capture_event_and_expand(
            EventInsert(
                id=event_id,
                event_type=event_type,
                api_version="2026-07-01",
                source_kind=EventSourceKind.COMMAND,
                aggregate_type=None,
                aggregate_id=None,
                aggregate_version=None,
                source_command_id=f"worker-{label}-command",
                source_component="authnz",
                source_request_id=None,
                body=ring.encrypt_event_body(
                    event_id=event_id,
                    api_version="2026-07-01",
                    body=body,
                ),
                body_size_bytes=len(body),
                created_at=fixture.clock(),
            ),
            lambda: delivery_id,
            fixture.clock() + timedelta(hours=72),
        )
        if kind is DeliveryKind.AUTOMATIC:
            assert len(captured.deliveries) == 1
        else:
            assert captured.deliveries == ()
            await tx.insert_delivery(
                delivery_id,
                event_id=event_id,
                webhook_id=webhook_id,
                kind=kind,
                expires_at=fixture.clock() + timedelta(hours=72),
                now=fixture.clock(),
            )
            await tx._execute(
                "UPDATE admin_webhook_registrations SET active = ? WHERE id = ?",
                (True, webhook_id),
            )
        if expires_in != 72 * 60 * 60:
            await tx._execute(
                "UPDATE admin_webhook_deliveries SET expires_at = ? WHERE id = ?",
                (fixture.clock() + timedelta(seconds=expires_in), delivery_id),
            )
        claim_token = _token(f"worker-{label}-claim")
        claim = await tx.claim_pending_delivery(
            claim_token,
            fixture.clock() + timedelta(seconds=60),
            fixture.clock(),
        )
        assert claim is not None

    admitted = JobsDeliveryQueue(fixture.manager).admit_delivery_job(
        delivery_id,
        fixture.clock() + timedelta(seconds=expires_in),
    )
    assert admitted.record is not None
    async with repository.transaction() as tx:
        attached = await tx.attach_jobs_job(
            delivery_id,
            claim_token,
            admitted.record.jobs_job_id,
            fixture.clock(),
        )
        assert attached is not None
    acquired = fixture.manager.acquire_next_job(
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        lease_seconds=120,
        worker_id=f"worker-{label}",
    )
    assert acquired is not None
    return webhook_id, delivery_id, acquired


def _handler(
    fixture: WorkerFixture,
    executor: FakeExecutor,
    *,
    crash_hook=None,
    metrics=None,
) -> AdminWebhookPreparedHandler:
    return AdminWebhookPreparedHandler(
        repository=fixture.repository,
        key_ring=fixture.ring,
        settings=_settings(),
        executor=executor,
        token_factory=TokenSource("worker-disposition"),
        attempt_id_factory=lambda: canonical_uuid4(
            f"worker-attempt-{len(executor.requests) + 1}"
        ),
        clock=fixture.clock,
        crash_hook=crash_hook,
        metrics=metrics,
    )


async def _apply(
    fixture: WorkerFixture,
    acquired: dict,
    disposition,
):
    return fixture.manager.apply_prepared_disposition(
        ApplyPreparedDispositionCommand(
            job_id=int(acquired["id"]),
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            expected_payload=acquired["payload"],
            worker_id=acquired["worker_id"],
            lease_id=acquired["lease_id"],
            disposition=disposition,
        )
    )


def _success() -> AttemptExecutionResult:
    return AttemptExecutionResult(
        outcome=AttemptOutcome.SUCCESS,
        status_code=204,
        latency_ms=8,
        reason_code=None,
        retry_delay_seconds=None,
    )


def _retryable() -> AttemptExecutionResult:
    return AttemptExecutionResult(
        outcome=AttemptOutcome.RETRYABLE,
        status_code=503,
        latency_ms=12,
        reason_code=AttemptReasonCode.HTTP_SERVER_ERROR,
        retry_delay_seconds=60,
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_handler_orders_horizon_reservation_execution_and_acknowledgement(
    worker_fixture: WorkerFixture,
) -> None:
    _, delivery_id, acquired = await _seed_acquired(worker_fixture, "success")
    order: list[str] = []
    executor = FakeExecutor(_success(), order=order)
    handler = _handler(worker_fixture, executor)
    context = FakeContext(acquired, order)

    disposition = await handler(acquired, context)

    assert order == ["ensure", "execute"]
    assert context.ensure_calls == [40]
    assert disposition.kind is PreparedDispositionKind.COMPLETE
    assert disposition.attempt_id is not None
    assert disposition.delivery_id == delivery_id
    assert len(executor.requests) == 1
    request = executor.requests[0]
    assert request.body == b'{"id":1}'
    assert request.signing_secret == "whsec_" + "a" * 64
    assert request.attempt_number == 1

    applied = await _apply(worker_fixture, acquired, disposition)
    assert applied.outcome is OperationOutcome.APPLIED
    await handler.on_disposition_applied(acquired, disposition, applied)
    bundle = await worker_fixture.repository.get_delivery_bundle(delivery_id)
    assert bundle is not None
    assert bundle.delivery.delivery.state is DeliveryState.SUCCEEDED
    assert bundle.delivery.jobs_disposition_applied is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_worker_observes_closed_executor_outcome_only_after_commit(
    worker_fixture: WorkerFixture,
) -> None:
    _, delivery_id, acquired = await _seed_acquired(worker_fixture, "metrics")
    executor = FakeExecutor(
        AttemptExecutionResult(
            outcome=AttemptOutcome.FAILED,
            status_code=None,
            latency_ms=5,
            reason_code=AttemptReasonCode.HTTP_HOP_DNS_ADDRESS_DENIED,
            retry_delay_seconds=None,
        )
    )
    observations: list[dict[str, object]] = []

    class Metrics:
        def attempt_committed(self, **values: object) -> None:
            observations.append(values)

    handler = AdminWebhookPreparedHandler(
        repository=worker_fixture.repository,
        key_ring=worker_fixture.ring,
        settings=_settings(),
        executor=executor,
        token_factory=TokenSource("worker-metrics"),
        attempt_id_factory=lambda: canonical_uuid4("worker-metrics-attempt"),
        clock=worker_fixture.clock,
        metrics=Metrics(),
    )

    disposition = await handler(acquired, FakeContext(acquired))

    bundle = await worker_fixture.repository.get_delivery_bundle(delivery_id)
    assert bundle is not None
    assert bundle.delivery.delivery.state is DeliveryState.DEAD
    assert observations == [
        {
            "state": DeliveryState.DEAD,
            "kind": DeliveryKind.AUTOMATIC,
            "reason_code": DeliveryReasonCode.HTTP_HOP_DNS_ADDRESS_DENIED,
            "delivery_reason_code": (
                DeliveryReasonCode.HTTP_HOP_DNS_ADDRESS_DENIED
            ),
            "status_code": None,
            "latency_ms": 5,
        }
    ]
    assert disposition.kind is PreparedDispositionKind.FAIL


@pytest.mark.asyncio
@pytest.mark.unit
async def test_pre_reservation_key_and_lease_failures_defer_without_attempt(
    worker_fixture: WorkerFixture,
) -> None:
    webhook_id, delivery_id, acquired = await _seed_acquired(
        worker_fixture,
        "infrastructure",
    )
    executor = FakeExecutor(_success())
    missing_key_ring = WebhookKeyRing(
        {"other-key": base64.b64encode(b"z" * 32).decode("ascii")},
        primary_id="other-key",
    )
    handler = AdminWebhookPreparedHandler(
        repository=worker_fixture.repository,
        key_ring=missing_key_ring,
        settings=_settings(),
        executor=executor,
        token_factory=TokenSource("missing-key"),
        attempt_id_factory=lambda: canonical_uuid4("missing-key-attempt"),
        clock=worker_fixture.clock,
    )

    key_failure = await handler(acquired, FakeContext(acquired))
    assert key_failure.origin is PreparedDispositionOrigin.INFRASTRUCTURE
    assert key_failure.not_before_at is None
    assert await worker_fixture.repository.list_delivery_attempts(
        webhook_id, delivery_id
    ) == ()

    lease_context = FakeContext(acquired)
    lease_context.ensure_result = False
    lease_failure = await _handler(worker_fixture, executor)(
        acquired,
        lease_context,
    )
    assert lease_failure.origin is PreparedDispositionOrigin.INFRASTRUCTURE
    assert lease_failure.not_before_at is None
    assert executor.requests == []
    assert await worker_fixture.repository.list_delivery_attempts(
        webhook_id, delivery_id
    ) == ()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_pending_retry_replays_without_io_and_lost_ack_continues_current_lease(
    worker_fixture: WorkerFixture,
) -> None:
    webhook_id, delivery_id, acquired = await _seed_acquired(
        worker_fixture,
        "retry-replay",
    )
    executor = FakeExecutor(_retryable())
    handler = _handler(worker_fixture, executor)
    first = await handler(acquired, FakeContext(acquired))
    assert first.kind is PreparedDispositionKind.RETRY
    assert len(executor.requests) == 1

    replay = await handler(acquired, FakeContext(acquired))
    assert replay == first
    assert len(executor.requests) == 1

    applied = await _apply(worker_fixture, acquired, first)
    assert applied.state == "queued"
    connection = worker_fixture.manager._connect()
    try:
        connection.execute(
            "UPDATE jobs SET available_at=NULL WHERE id=?",
            (acquired["id"],),
        )
        connection.commit()
    finally:
        connection.close()
    reacquired = worker_fixture.manager.acquire_next_job(
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        lease_seconds=120,
        worker_id="worker-reacquired",
    )
    assert reacquired is not None
    second = await handler(reacquired, FakeContext(reacquired))

    assert second.kind is PreparedDispositionKind.RETRY
    assert second.token != first.token
    assert len(executor.requests) == 2
    attempts = await worker_fixture.repository.list_delivery_attempts(
        webhook_id,
        delivery_id,
    )
    assert [attempt.attempt_number for attempt in attempts] == [1, 2]
    bundle = await worker_fixture.repository.get_delivery_bundle(delivery_id)
    assert bundle is not None
    assert bundle.delivery.pending_jobs_disposition_token == second.token


@pytest.mark.asyncio
@pytest.mark.unit
async def test_real_executor_attempt_four_exhausts_budget_without_fifth_request(
    worker_fixture: WorkerFixture,
) -> None:
    webhook_id, delivery_id, acquired = await _seed_acquired(
        worker_fixture,
        "real-executor-hard-cap",
    )
    egress = RetryingStatusEgress()
    executor = DeliveryAttemptExecutor(
        egress=egress,
        clock=DeterministicExecutorClock(worker_fixture.clock),
    )
    attempt_ids = iter(
        canonical_uuid4(f"real-executor-attempt-{number}")
        for number in range(1, 5)
    )
    handler = AdminWebhookPreparedHandler(
        repository=worker_fixture.repository,
        key_ring=worker_fixture.ring,
        settings=_settings(),
        executor=executor,
        token_factory=TokenSource("real-executor-disposition"),
        attempt_id_factory=lambda: next(attempt_ids),
        clock=worker_fixture.clock,
    )

    for attempt_number in range(1, 5):
        disposition = await handler(acquired, FakeContext(acquired))
        if attempt_number < 4:
            assert disposition.kind is PreparedDispositionKind.RETRY
            assert disposition.delay_seconds == (60, 300, 1_800)[
                attempt_number - 1
            ]
        else:
            assert disposition.kind is PreparedDispositionKind.FAIL
            assert (
                disposition.reason_code
                == DeliveryReasonCode.ATTEMPT_BUDGET_EXHAUSTED.value
            )
        applied = await _apply(worker_fixture, acquired, disposition)
        assert applied.outcome is OperationOutcome.APPLIED
        await handler.on_disposition_applied(acquired, disposition, applied)
        if attempt_number < 4:
            assert disposition.delay_seconds is not None
            worker_fixture.clock.advance(disposition.delay_seconds)
            assert worker_fixture.manager.reschedule_jobs(
                domain="admin_webhooks",
                queue="delivery",
                job_type="admin_webhook_delivery",
                status="queued",
                set_now=True,
            ) == 1
            acquired = worker_fixture.manager.acquire_next_job(
                domain="admin_webhooks",
                queue="delivery",
                job_type="admin_webhook_delivery",
                lease_seconds=120,
                worker_id=f"worker-real-executor-{attempt_number + 1}",
            )
            assert acquired is not None

    assert len(egress.requests) == 4
    attempts = await worker_fixture.repository.list_delivery_attempts(
        webhook_id,
        delivery_id,
    )
    assert [attempt.attempt_number for attempt in attempts] == [1, 2, 3, 4]
    assert attempts[-1].state is AttemptState.FAILED
    assert (
        attempts[-1].reason_code
        is DeliveryReasonCode.ATTEMPT_BUDGET_EXHAUSTED
    )
    bundle = await worker_fixture.repository.get_delivery_bundle(delivery_id)
    assert bundle is not None
    assert bundle.delivery.delivery.state is DeliveryState.DEAD
    assert (
        bundle.delivery.delivery.reason_code
        is DeliveryReasonCode.ATTEMPT_BUDGET_EXHAUSTED
    )
    assert worker_fixture.manager.acquire_next_job(
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        lease_seconds=120,
        worker_id="worker-real-executor-5",
    ) is None
    assert len(egress.requests) == 4


@pytest.mark.asyncio
@pytest.mark.unit
async def test_processing_attempt_defers_until_persisted_timeout_plus_ninety(
    worker_fixture: WorkerFixture,
) -> None:
    webhook_id, delivery_id, acquired = await _seed_acquired(
        worker_fixture,
        "not-stale",
    )
    executor = FakeExecutor(_success())
    crashing = _handler(
        worker_fixture,
        executor,
        crash_hook=OneShotCrash(WorkerCrashPoint.AFTER_RESERVATION_COMMIT_BEFORE_IO),
    )
    with pytest.raises(SimulatedCrash):
        await crashing(acquired, FakeContext(acquired))

    recovery_executor = FakeExecutor(_success())
    recovery = _handler(worker_fixture, recovery_executor)
    deferred = await recovery(acquired, FakeContext(acquired))
    attempts = await worker_fixture.repository.list_delivery_attempts(
        webhook_id,
        delivery_id,
    )

    assert deferred.origin is PreparedDispositionOrigin.RECOVERY
    assert deferred.not_before_at == attempts[0].started_at + timedelta(seconds=100)
    assert attempts[0].state is AttemptState.PROCESSING
    assert recovery_executor.requests == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_stale_attempt_closes_unknown_and_prepares_retry_without_http(
    worker_fixture: WorkerFixture,
) -> None:
    webhook_id, delivery_id, acquired = await _seed_acquired(
        worker_fixture,
        "stale",
    )
    executor = FakeExecutor(_success())
    crashing = _handler(
        worker_fixture,
        executor,
        crash_hook=OneShotCrash(WorkerCrashPoint.AFTER_RESERVATION_COMMIT_BEFORE_IO),
    )
    with pytest.raises(SimulatedCrash):
        await crashing(acquired, FakeContext(acquired))
    worker_fixture.clock.advance(100)

    recovery_executor = FakeExecutor(_success())
    disposition = await _handler(worker_fixture, recovery_executor)(
        acquired,
        FakeContext(acquired),
    )
    attempts = await worker_fixture.repository.list_delivery_attempts(
        webhook_id,
        delivery_id,
    )

    assert disposition.kind is PreparedDispositionKind.RETRY
    assert disposition.attempt_id == attempts[0].id
    assert disposition.not_before_at == worker_fixture.clock() + timedelta(seconds=60)
    assert attempts[0].state is AttemptState.OUTCOME_UNKNOWN
    assert recovery_executor.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expires_in", "expected_reason"),
    ((20, DeliveryReasonCode.DELIVERY_EXPIRED),),
)
@pytest.mark.parametrize("kind", (DeliveryKind.AUTOMATIC, DeliveryKind.MANUAL))
@pytest.mark.unit
async def test_required_horizon_terminalizes_without_an_attempt(
    worker_fixture: WorkerFixture,
    expires_in: int,
    expected_reason: DeliveryReasonCode,
    kind: DeliveryKind,
) -> None:
    webhook_id, delivery_id, acquired = await _seed_acquired(
        worker_fixture,
        f"horizon-{kind.value}",
        expires_in=expires_in,
        kind=kind,
    )
    executor = FakeExecutor(_success())
    observations: list[dict[str, object]] = []

    class Metrics:
        def delivery_committed(self, **values: object) -> None:
            observations.append(values)

    handler = _handler(worker_fixture, executor, metrics=Metrics())
    disposition = await handler(
        acquired,
        FakeContext(acquired),
    )
    replay = await handler(acquired, FakeContext(acquired))

    assert disposition.kind is PreparedDispositionKind.FAIL
    assert disposition.attempt_id is None
    assert disposition.reason_code == expected_reason.value
    assert executor.requests == []
    assert replay.token == disposition.token
    assert observations == [
        {
            "state": DeliveryState.DEAD,
            "kind": kind,
            "reason_code": expected_reason,
            "status_code": None,
        }
    ]
    assert await worker_fixture.repository.list_delivery_attempts(
        webhook_id, delivery_id
    ) == ()


async def _mutate_registration(
    fixture: WorkerFixture,
    webhook_id: int,
    mutation: str,
) -> None:
    current = await fixture.repository.get_protected_registration(
        webhook_id,
        include_deleted=True,
    )
    assert current is not None
    async with fixture.repository.transaction() as tx:
        if mutation == "delete":
            await tx.soft_delete_registration(
                webhook_id,
                expected_revision=current.registration.revision,
                actor_user_id=7,
                at=fixture.clock(),
            )
            return
        if mutation == "disable":
            patch = RegistrationPatch(active=False)
        elif mutation == "rotate":
            next_version = current.registration.secret_version + 1
            patch = RegistrationPatch(
                secret=fixture.ring.encrypt_text(
                    purpose="registration.secret",
                    identity={
                        "registration_id": webhook_id,
                        "secret_version": next_version,
                    },
                    plaintext="whsec_" + "b" * 64,
                )
            )
        else:
            patch = RegistrationPatch(timeout_seconds=11)
        await tx.patch_registration(
            webhook_id,
            expected_revision=current.registration.revision,
            patch=patch,
            actor_user_id=7,
            at=fixture.clock(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_trigger", ("lifecycle", "expiry", "budget"))
@pytest.mark.unit
async def test_nonstale_processing_precedes_every_no_attempt_terminal_trigger(
    worker_fixture: WorkerFixture,
    terminal_trigger: str,
) -> None:
    webhook_id, delivery_id, acquired = await _seed_acquired(
        worker_fixture,
        f"processing-precedence-{terminal_trigger}",
    )
    initial_executor = FakeExecutor(_success())
    crashing = _handler(
        worker_fixture,
        initial_executor,
        crash_hook=OneShotCrash(WorkerCrashPoint.AFTER_RESERVATION_COMMIT_BEFORE_IO),
    )
    with pytest.raises(SimulatedCrash):
        await crashing(acquired, FakeContext(acquired))

    if terminal_trigger == "lifecycle":
        await _mutate_registration(worker_fixture, webhook_id, "disable")
    else:
        async with worker_fixture.repository.transaction() as tx:
            if terminal_trigger == "expiry":
                await tx._execute(
                    "UPDATE admin_webhook_deliveries SET expires_at = ? WHERE id = ?",
                    (worker_fixture.clock() - timedelta(seconds=1), delivery_id),
                )
            else:
                await tx._execute(
                    "UPDATE admin_webhook_deliveries SET attempt_count = 4 WHERE id = ?",
                    (delivery_id,),
                )

    before = await worker_fixture.repository.get_delivery_bundle(delivery_id)
    attempts_before = await worker_fixture.repository.list_delivery_attempts(
        webhook_id,
        delivery_id,
    )
    recovery_executor = FakeExecutor(_success())
    disposition = await _handler(worker_fixture, recovery_executor)(
        acquired,
        FakeContext(acquired),
    )

    assert disposition.origin is PreparedDispositionOrigin.RECOVERY
    assert disposition.reason_code == "attempt_not_stale"
    assert disposition.not_before_at == attempts_before[0].started_at + timedelta(
        seconds=100
    )
    assert await worker_fixture.repository.get_delivery_bundle(delivery_id) == before
    assert await worker_fixture.repository.list_delivery_attempts(
        webhook_id,
        delivery_id,
    ) == attempts_before
    assert initial_executor.requests == []
    assert recovery_executor.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    (
        ("delete", DeliveryReasonCode.CANCELED_DELETED),
        ("disable", DeliveryReasonCode.CANCELED_DISABLED),
        ("rotate", DeliveryReasonCode.CANCELED_SECRET_ROTATION),
        ("config", DeliveryReasonCode.SUPERSEDED_CONFIG),
    ),
)
@pytest.mark.unit
async def test_lifecycle_winner_before_reservation_sends_nothing(
    worker_fixture: WorkerFixture,
    mutation: str,
    expected_reason: DeliveryReasonCode,
) -> None:
    webhook_id, delivery_id, acquired = await _seed_acquired(
        worker_fixture,
        f"before-{mutation}",
    )
    await _mutate_registration(worker_fixture, webhook_id, mutation)
    executor = FakeExecutor(_success())

    disposition = await _handler(worker_fixture, executor)(
        acquired,
        FakeContext(acquired),
    )

    assert disposition.kind is PreparedDispositionKind.CANCEL
    assert disposition.reason_code == expected_reason.value
    assert executor.requests == []
    assert await worker_fixture.repository.list_delivery_attempts(
        webhook_id, delivery_id
    ) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "expected_kind", "attempt_state"),
    (
        (_success(), PreparedDispositionKind.COMPLETE, AttemptState.SUCCEEDED),
        (
            AttemptExecutionResult(
                outcome=AttemptOutcome.FAILED,
                status_code=400,
                latency_ms=3,
                reason_code=AttemptReasonCode.HTTP_CLIENT_ERROR,
                retry_delay_seconds=None,
            ),
            PreparedDispositionKind.FAIL,
            AttemptState.FAILED,
        ),
        (_retryable(), PreparedDispositionKind.CANCEL, AttemptState.RETRYABLE),
    ),
)
@pytest.mark.parametrize(
    ("mutation", "lifecycle_reason", "lifecycle_state"),
    (
        ("delete", DeliveryReasonCode.CANCELED_DELETED, DeliveryState.CANCELED),
        ("disable", DeliveryReasonCode.CANCELED_DISABLED, DeliveryState.CANCELED),
        (
            "rotate",
            DeliveryReasonCode.CANCELED_SECRET_ROTATION,
            DeliveryState.CANCELED,
        ),
        ("config", DeliveryReasonCode.SUPERSEDED_CONFIG, DeliveryState.SUPERSEDED),
    ),
)
@pytest.mark.unit
async def test_post_reservation_config_race_preserves_real_attempt_evidence(
    worker_fixture: WorkerFixture,
    result: AttemptExecutionResult,
    expected_kind: PreparedDispositionKind,
    attempt_state: AttemptState,
    mutation: str,
    lifecycle_reason: DeliveryReasonCode,
    lifecycle_state: DeliveryState,
) -> None:
    webhook_id, delivery_id, acquired = await _seed_acquired(
        worker_fixture,
        f"after-{mutation}-{result.outcome.value}",
    )

    async def mutate() -> None:
        await _mutate_registration(worker_fixture, webhook_id, mutation)

    executor = FakeExecutor(result, before_result=mutate)
    disposition = await _handler(worker_fixture, executor)(
        acquired,
        FakeContext(acquired),
    )
    bundle = await worker_fixture.repository.get_delivery_bundle(delivery_id)
    attempts = await worker_fixture.repository.list_delivery_attempts(
        webhook_id,
        delivery_id,
    )

    assert disposition.kind is expected_kind
    assert disposition.not_before_at is None
    assert bundle is not None
    assert bundle.delivery.completed_after_config_change is True
    assert attempts[0].state is attempt_state
    if result.outcome is AttemptOutcome.RETRYABLE:
        assert bundle.delivery.delivery.state is lifecycle_state
        assert attempts[0].reason_code is DeliveryReasonCode.HTTP_SERVER_ERROR
        assert attempts[0].requested_retry_delay_seconds == 60
        assert bundle.delivery.delivery.reason_code is lifecycle_reason
    elif result.outcome is AttemptOutcome.SUCCESS:
        assert bundle.delivery.delivery.state is DeliveryState.SUCCEEDED
        assert bundle.delivery.delivery.reason_code is None
    else:
        assert bundle.delivery.delivery.state is DeliveryState.DEAD
        assert bundle.delivery.delivery.reason_code is DeliveryReasonCode.HTTP_CLIENT_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "point",
    (
        WorkerCrashPoint.BEFORE_RESERVATION_COMMIT,
        WorkerCrashPoint.AFTER_RESERVATION_COMMIT_BEFORE_IO,
        WorkerCrashPoint.AFTER_RECEIVER_RESULT_BEFORE_OUTCOME_COMMIT,
        WorkerCrashPoint.AFTER_OUTCOME_COMMIT_BEFORE_JOBS_APPLY,
    ),
)
@pytest.mark.unit
async def test_pre_apply_crash_boundaries_never_duplicate_receiver_io(
    worker_fixture: WorkerFixture,
    point: WorkerCrashPoint,
) -> None:
    webhook_id, delivery_id, acquired = await _seed_acquired(
        worker_fixture,
        f"crash-{point.value}",
    )
    executor = FakeExecutor(_success())
    handler = _handler(worker_fixture, executor, crash_hook=OneShotCrash(point))
    with pytest.raises(SimulatedCrash):
        await handler(acquired, FakeContext(acquired))

    bundle = await worker_fixture.repository.get_delivery_bundle(delivery_id)
    assert bundle is not None
    if point is WorkerCrashPoint.BEFORE_RESERVATION_COMMIT:
        assert await worker_fixture.repository.list_delivery_attempts(
            webhook_id, delivery_id
        ) == ()
        recovered = await handler(acquired, FakeContext(acquired))
        assert recovered.kind is PreparedDispositionKind.COMPLETE
        assert len(executor.requests) == 1
    elif point is WorkerCrashPoint.AFTER_OUTCOME_COMMIT_BEFORE_JOBS_APPLY:
        recovered = await handler(acquired, FakeContext(acquired))
        assert recovered.kind is PreparedDispositionKind.COMPLETE
        assert len(executor.requests) == 1
    else:
        worker_fixture.clock.advance(100)
        recovered = await handler(acquired, FakeContext(acquired))
        assert recovered.kind is PreparedDispositionKind.RETRY
        assert len(executor.requests) == (
            0
            if point is WorkerCrashPoint.AFTER_RESERVATION_COMMIT_BEFORE_IO
            else 1
        )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_callback_crashes_are_idempotent_around_exact_acknowledgement(
    worker_fixture: WorkerFixture,
) -> None:
    _, delivery_id, acquired = await _seed_acquired(worker_fixture, "callback-crash")
    executor = FakeExecutor(_success())
    before_ack = _handler(
        worker_fixture,
        executor,
        crash_hook=OneShotCrash(WorkerCrashPoint.AFTER_JOBS_APPLY_BEFORE_AUTHNZ_ACK),
    )
    disposition = await before_ack(acquired, FakeContext(acquired))
    applied = await _apply(worker_fixture, acquired, disposition)
    with pytest.raises(SimulatedCrash):
        await before_ack.on_disposition_applied(acquired, disposition, applied)
    stranded = await worker_fixture.repository.get_delivery_bundle(delivery_id)
    assert stranded is not None and not stranded.delivery.jobs_disposition_applied

    after_ack = _handler(
        worker_fixture,
        executor,
        crash_hook=OneShotCrash(WorkerCrashPoint.AFTER_AUTHNZ_ACK_BEFORE_RETURN),
    )
    with pytest.raises(SimulatedCrash):
        await after_ack.on_disposition_applied(acquired, disposition, applied)
    acknowledged = await worker_fixture.repository.get_delivery_bundle(delivery_id)
    assert acknowledged is not None and acknowledged.delivery.jobs_disposition_applied
    await after_ack.on_disposition_applied(acquired, disposition, applied)
    assert len(executor.requests) == 1
