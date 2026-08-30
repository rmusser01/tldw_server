from __future__ import annotations

import asyncio
import base64
import hashlib
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import asyncpg
import pytest

from tldw_Server_API.app.core.Admin_Webhooks import domain
from tldw_Server_API.app.core.Admin_Webhooks.config import (
    AdminWebhookMode,
    AdminWebhookSettings,
    WebhookRouteSelection,
)
from tldw_Server_API.app.core.Admin_Webhooks.crypto import WebhookKeyRing
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    AttemptState,
    DeliveryReasonCode,
    DeliveryState,
    JobsDispositionKind,
)
from tldw_Server_API.app.core.Admin_Webhooks.executor import (
    AttemptExecutionRequest,
    AttemptExecutionResult,
    AttemptOutcome,
    AttemptReasonCode,
)
from tldw_Server_API.app.core.Admin_Webhooks.reconciler import (
    AdminWebhookReconciler,
    EnqueueCrashPoint,
    JobsDeliveryQueue,
)
from tldw_Server_API.app.core.Admin_Webhooks.worker import (
    AdminWebhookPreparedHandler,
    WorkerCrashPoint,
)
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
    ensure_admin_webhook_canonical_tables_pg,
)
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
    AttemptCompletion,
    RegistrationInsert,
    RegistrationPatch,
    RegistrationTarget,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    ApplyPreparedDispositionCommand,
    PreparedDispositionKind,
    PreparedDispositionOrigin,
    PreparedDispositionResult,
    PreparedJobDisposition,
)
from tldw_Server_API.app.core.Jobs.pg_migrations import (
    ensure_job_counters_pg,
    ensure_jobs_tables_pg,
)
from tldw_Server_API.app.core.Jobs.worker_sdk import (
    WorkerConfig,
    WorkerExecutionContext,
    WorkerSDK,
)
from tldw_Server_API.app.core.Security.egress import URLPolicyResult
from tldw_Server_API.tests.Admin_Webhooks.test_event_expansion import (
    NOW,
    canonical_uuid4,
    event_insert,
    key_ring,
    opaque_token,
    seed_registration,
)

pytest_plugins = ("tldw_Server_API.tests.AuthNZ.conftest",)

BACKEND_PAIRS = (
    ("sqlite", "sqlite"),
    ("sqlite", "postgres"),
    ("postgres", "sqlite"),
    ("postgres", "postgres"),
)


@pytest.fixture
def matrix_jobs_pg_dsn(pg_temp_db, monkeypatch: pytest.MonkeyPatch) -> str:
    """Initialize the isolated Jobs database without loading Jobs conftest."""

    dsn = str(pg_temp_db["dsn"])
    ensure_jobs_tables_pg(dsn)
    ensure_job_counters_pg(dsn)
    JobManager.set_acquire_gate(False)
    monkeypatch.setenv("JOBS_DISABLE_LEASE_ENFORCEMENT", "true")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "false")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")
    return dsn


@pytest.fixture(autouse=True)
def allow_matrix_worker_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        domain,
        "evaluate_platform_webhook_url_policy",
        lambda _url: URLPolicyResult(True, resolved_ips=("93.184.216.34",)),
    )


class SimulatedCrash(BaseException):
    pass


class MutableClock:
    def __init__(self, now: datetime) -> None:
        self.current = now

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: int) -> None:
        self.current += timedelta(seconds=seconds)


class TokenSource:
    def __init__(self, label: str) -> None:
        self.label = label
        self.index = 0

    def __call__(self) -> str:
        token = hashlib.sha256(
            f"{self.label}:{self.index}".encode("ascii")
        ).hexdigest()
        self.index += 1
        return token


class AttemptIdSource:
    def __init__(self, label: str) -> None:
        self.label = label
        self.index = 0

    def __call__(self) -> str:
        attempt_id = canonical_uuid4(f"{self.label}:{self.index}")
        self.index += 1
        return attempt_id


class MatrixWorkerContext:
    def __init__(self, acquired: dict) -> None:
        self.acquired = acquired
        self.ensure_calls: list[int] = []

    async def ensure_lease_horizon(self, seconds: int) -> bool:
        self.ensure_calls.append(seconds)
        return True

    def snapshot(self):
        return SimpleNamespace(
            worker_id=self.acquired["worker_id"],
            lease_id=self.acquired["lease_id"],
            leased_until=self.acquired["leased_until"],
            renewal_lost=False,
        )


class MatrixExecutor:
    def __init__(
        self,
        result_factory: Callable[[AttemptExecutionRequest], AttemptExecutionResult],
        *,
        before_result: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.result_factory = result_factory
        self.before_result = before_result
        self.requests: list[AttemptExecutionRequest] = []

    async def execute(
        self,
        request: AttemptExecutionRequest,
    ) -> AttemptExecutionResult:
        self.requests.append(request)
        if self.before_result is not None:
            await self.before_result()
        return self.result_factory(request)


class WorkerOneShotCrash:
    def __init__(self, point: WorkerCrashPoint) -> None:
        self.point = point
        self.armed = True

    def __call__(self, point: WorkerCrashPoint) -> None:
        if self.armed and point is self.point:
            self.armed = False
            raise SimulatedCrash(point.value)


class OneShotCrash:
    def __init__(self, point: EnqueueCrashPoint) -> None:
        self.point = point
        self.armed = True

    def __call__(self, point: EnqueueCrashPoint) -> None:
        if self.armed and point is self.point:
            self.armed = False
            raise SimulatedCrash(point.value)


class TerminalizeAfterClaim:
    def __init__(
        self,
        repository: AdminWebhookRepository,
        webhook_id: int,
        clock: MutableClock,
    ) -> None:
        self.repository = repository
        self.webhook_id = webhook_id
        self.clock = clock

    async def __call__(self) -> None:
        await _terminalize(
            self.repository,
            self.webhook_id,
            reason=DeliveryReasonCode.CANCELED_DISABLED,
            now=self.clock(),
        )


class CrashAfterAdmissionManager:
    def __init__(self, manager: JobManager) -> None:
        self.manager = manager
        self.armed = True

    def admit_job(self, **kwargs):
        result = self.manager.admit_job(**kwargs)
        if self.armed:
            self.armed = False
            raise SimulatedCrash("after_jobs_create_before_response")
        return result

    def find_job_by_identity(self, command):
        return self.manager.find_job_by_identity(command)

    def get_job(self, job_id: int):
        return self.manager.get_job(job_id)

    def apply_prepared_disposition(self, command):
        return self.manager.apply_prepared_disposition(command)


class RecordingPreparedManager:
    def __init__(self, manager: JobManager) -> None:
        self.manager = manager
        self.apply_commands: list[ApplyPreparedDispositionCommand] = []
        self.apply_results: list[PreparedDispositionResult] = []
        self.after_apply: Callable[[], None] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.manager, name)

    def apply_prepared_disposition(
        self,
        command: ApplyPreparedDispositionCommand,
    ) -> PreparedDispositionResult:
        result = self.manager.apply_prepared_disposition(command)
        self.apply_commands.append(command)
        self.apply_results.append(result)
        if self.after_apply is not None:
            after_apply = self.after_apply
            self.after_apply = None
            after_apply()
        return result


async def _run_prepared_once(
    manager: RecordingPreparedManager,
    handler: AdminWebhookPreparedHandler,
    *,
    worker_id: str,
    captures: dict[str, Any],
    on_applied: Callable[
        [dict[str, Any], PreparedJobDisposition, PreparedDispositionResult],
        Awaitable[None],
    ]
    | None,
    stop_after_apply: bool = False,
) -> None:
    sdk = WorkerSDK(
        manager,
        WorkerConfig(
            domain="admin_webhooks",
            queue="delivery",
            worker_id=worker_id,
            lease_seconds=120,
            renew_jitter_seconds=0,
            renew_threshold_seconds=20,
            backoff_base_seconds=1,
            backoff_max_seconds=1,
            completion_callback_timeout_seconds=5.0,
        ),
    )

    async def one_job(
        job: dict[str, Any],
        context: WorkerExecutionContext,
    ) -> PreparedJobDisposition:
        captures["job"] = job
        try:
            disposition = await handler(job, context)
            captures["disposition"] = disposition
            return disposition
        finally:
            if not stop_after_apply:
                # stop() takes effect only after this disposition and callback path.
                sdk.stop()

    async def allow_acquire() -> bool:
        return True

    if stop_after_apply:
        assert manager.after_apply is None
        manager.after_apply = sdk.stop
    try:
        await asyncio.wait_for(
            sdk.run_prepared(
                handler=one_job,
                pre_acquire_guard=allow_acquire,
                handler_error_disposition=handler.handler_error_disposition,
                on_disposition_applied=on_applied,
            ),
            timeout=10.0,
        )
    finally:
        manager.after_apply = None


def _jobs_retry_evidence(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row.get("retry_count") or 0),
        row.get("failure_streak_code"),
        int(row.get("failure_streak_count") or 0),
        row.get("quarantined_at"),
    )


def _no_crash(_point: EnqueueCrashPoint) -> None:
    return None


class CountingQueue:
    def __init__(self, queue: JobsDeliveryQueue) -> None:
        self.queue = queue
        self.admit_calls: list[str] = []
        self.find_calls: list[str] = []
        self.cancel_calls: list[str] = []
        self.cancel_tokens: list[str] = []

    def admit_delivery_job(self, delivery_id: str, expires_at: datetime):
        self.admit_calls.append(delivery_id)
        return self.queue.admit_delivery_job(delivery_id, expires_at)

    def find_delivery_job_by_identity(self, delivery_id: str):
        self.find_calls.append(delivery_id)
        return self.queue.find_delivery_job_by_identity(delivery_id)

    def get_delivery_job(self, jobs_job_id: str):
        return self.queue.get_delivery_job(jobs_job_id)

    def apply_queued_cancel(
        self,
        jobs_job_id: str,
        delivery_id: str,
        disposition_token: str,
        reason_code: DeliveryReasonCode,
    ):
        self.cancel_calls.append(delivery_id)
        self.cancel_tokens.append(disposition_token)
        return self.queue.apply_queued_cancel(
            jobs_job_id,
            delivery_id,
            disposition_token,
            reason_code,
        )

    def reset_counts(self) -> None:
        self.admit_calls.clear()
        self.find_calls.clear()
        self.cancel_calls.clear()
        self.cancel_tokens.clear()


@asynccontextmanager
async def _auth_repository(
    backend: str,
    *,
    tmp_path: Path,
    test_db_pool,
):
    if backend == "sqlite":
        pool = DatabasePool(
            Settings(
                AUTH_MODE="single_user",
                DATABASE_URL=f"sqlite:///{tmp_path / 'matrix-auth.db'}",
            )
        )
        await pool.initialize()
        try:
            yield AdminWebhookRepository(pool)
        finally:
            await pool.close()
        return

    assert await ensure_admin_webhook_canonical_tables_pg(test_db_pool)
    connection = await asyncpg.connect(test_db_pool.settings.DATABASE_URL)
    try:
        await connection.execute(
            """
            TRUNCATE TABLE
                admin_webhook_delivery_attempts,
                admin_webhook_deliveries,
                admin_webhook_events,
                admin_webhook_idempotency,
                admin_webhook_runtime_heartbeats,
                admin_webhook_registrations,
                admin_webhook_sequences,
                admin_webhook_migration_state
            RESTART IDENTITY CASCADE
            """
        )
    finally:
        await connection.close()
    await test_db_pool.execute(
        "INSERT INTO admin_webhook_sequences (name, next_value) VALUES (?, ?)",
        "registration",
        1,
    )
    await test_db_pool.execute(
        """
        INSERT INTO admin_webhook_migration_state (
            singleton_id, schema_version, state_revision, phase
        ) VALUES (?, ?, ?, ?)
        """,
        1,
        1,
        1,
        "migration_pending",
    )
    yield AdminWebhookRepository(test_db_pool)


def _jobs_manager(
    backend: str,
    *,
    tmp_path: Path,
    jobs_pg_dsn: str,
) -> JobManager:
    if backend == "postgres":
        return JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    return JobManager(tmp_path / "matrix-jobs.db")


async def _seed_delivery(
    repository: AdminWebhookRepository,
    label: str,
    *,
    now: datetime,
    expires_at: datetime | None = None,
) -> tuple[int, str]:
    event_type = f"enqueue.matrix.{label}"
    webhook_id = await seed_registration(repository, event_types=(event_type,))
    delivery_id = canonical_uuid4(f"matrix-{label}-delivery")
    effective_expiry = expires_at or now + timedelta(hours=72)
    created_at = effective_expiry - timedelta(hours=72)
    async with repository.transaction() as tx:
        captured = await tx.capture_event_and_expand(
            event_insert(
                event_id=canonical_uuid4(f"matrix-{label}-event"),
                source_identity=f"matrix-{label}-command",
                event_type=event_type,
                created_at=created_at,
            ),
            lambda: delivery_id,
            effective_expiry,
        )
    assert len(captured.deliveries) == 1
    return webhook_id, delivery_id


def _worker_settings() -> AdminWebhookSettings:
    return AdminWebhookSettings(
        mode=AdminWebhookMode.ON,
        route_selection=WebhookRouteSelection.CANONICAL,
        registration_limit=100,
        active_limit=25,
        allow_http_dev=False,
        idempotency_ttl_seconds=86_400,
        rollback_window_days=7,
    )


async def _seed_worker_delivery(
    repository: AdminWebhookRepository,
    ring: WebhookKeyRing,
    label: str,
    *,
    now: datetime,
) -> tuple[int, str]:
    event_type = f"worker.matrix.{hashlib.sha256(label.encode('ascii')).hexdigest()[:16]}"
    event_id = canonical_uuid4(f"{label}-event")
    delivery_id = canonical_uuid4(f"{label}-delivery")
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
                active=True,
                timeout_seconds=10,
                secret=secret,
                secret_rotation_required=False,
                actor_user_id=7,
                now=now - timedelta(minutes=1),
            )
        )
        captured = await tx.capture_event_and_expand(
            event_insert(
                event_id=event_id,
                source_identity=f"{label}-command",
                event_type=event_type,
                created_at=now,
            ),
            lambda: delivery_id,
            now + timedelta(hours=72),
        )
    assert len(captured.deliveries) == 1
    return webhook_id, delivery_id


async def _disable_worker_registration(
    repository: AdminWebhookRepository,
    webhook_id: int,
    *,
    now: datetime,
) -> None:
    current = await repository.get_protected_registration(
        webhook_id,
        include_deleted=True,
    )
    assert current is not None
    async with repository.transaction() as tx:
        await tx.patch_registration(
            webhook_id,
            expected_revision=current.registration.revision,
            patch=RegistrationPatch(active=False),
            actor_user_id=7,
            at=now,
        )


def _worker_handler(
    repository: AdminWebhookRepository,
    ring: WebhookKeyRing,
    clock: MutableClock,
    executor: MatrixExecutor,
    label: str,
    *,
    crash_hook: Callable[[WorkerCrashPoint], None] | None = None,
) -> AdminWebhookPreparedHandler:
    return AdminWebhookPreparedHandler(
        repository=repository,
        key_ring=ring,
        settings=_worker_settings(),
        executor=executor,
        token_factory=TokenSource(f"{label}-disposition"),
        attempt_id_factory=AttemptIdSource(f"{label}-attempt"),
        clock=clock,
        crash_hook=crash_hook,
    )


def _matrix_result(
    outcome: str,
    *,
    retry_delay_seconds: int = 1_800,
) -> AttemptExecutionResult:
    if outcome == "complete":
        return AttemptExecutionResult(
            outcome=AttemptOutcome.SUCCESS,
            status_code=204,
            latency_ms=5,
            reason_code=None,
            retry_delay_seconds=None,
        )
    if outcome in {"retry", "cancel"}:
        return AttemptExecutionResult(
            outcome=AttemptOutcome.RETRYABLE,
            status_code=503,
            latency_ms=5,
            reason_code=AttemptReasonCode.HTTP_SERVER_ERROR,
            retry_delay_seconds=retry_delay_seconds,
        )
    return AttemptExecutionResult(
        outcome=AttemptOutcome.FAILED,
        status_code=400,
        latency_ms=5,
        reason_code=AttemptReasonCode.HTTP_CLIENT_ERROR,
        retry_delay_seconds=None,
    )


def _apply_worker_disposition(
    manager: JobManager,
    acquired: dict,
    disposition: PreparedJobDisposition,
) -> PreparedDispositionResult:
    return manager.apply_prepared_disposition(
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


async def _claim(
    repository: AdminWebhookRepository,
    delivery_id: str,
    *,
    token: str,
    now: datetime,
    ttl_seconds: int = 60,
) -> None:
    async with repository.transaction() as tx:
        claim = await tx.claim_pending_delivery(
            token,
            now + timedelta(seconds=ttl_seconds),
            now,
        )
    assert claim is not None and claim.delivery.delivery.id == delivery_id


async def _terminalize(
    repository: AdminWebhookRepository,
    webhook_id: int,
    *,
    reason: DeliveryReasonCode,
    now: datetime,
) -> None:
    async with repository.transaction() as tx:
        pending = await tx.cancel_registration_work(
            webhook_id,
            (2, 2),
            reason,
            lambda: opaque_token(f"matrix-terminal-{reason.value}"),
            now,
        )
    assert pending == ()


def _reconciler(
    repository: AdminWebhookRepository,
    queue,
    clock: MutableClock,
    tokens: TokenSource,
    *,
    crash_hook=lambda _point: None,
    after_claim_commit_hook: Callable[[], Awaitable[None]] | None = None,
) -> AdminWebhookReconciler:
    return AdminWebhookReconciler(
        repository=repository,
        queue=queue,
        token_factory=tokens,
        clock=clock,
        claim_ttl_seconds=60,
        failure_observer=lambda _failure: None,
        crash_hook=crash_hook,
        after_claim_commit_hook=after_claim_commit_hook,
    )


async def _assert_single_queued_identity(
    repository: AdminWebhookRepository,
    manager: JobManager,
    webhook_id: int,
    delivery_id: str,
) -> None:
    bundle = await repository.get_delivery_bundle(delivery_id)
    assert bundle is not None
    assert bundle.delivery.delivery.state is DeliveryState.QUEUED
    assert bundle.delivery.enqueue_claim_token is None
    assert bundle.delivery.enqueue_claim_expires_at is None
    assert bundle.delivery.jobs_job_id is not None
    assert await repository.list_delivery_attempts(webhook_id, delivery_id) == ()
    history = await repository.list_delivery_history(webhook_id, limit=10)
    assert history.total == 1
    assert history.items[0].kind.value == "automatic"

    rows = manager.list_jobs(
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        limit=100,
    )
    matches = [
        row for row in rows if row.get("payload") == {"delivery_id": delivery_id}
    ]
    assert len(matches) == 1
    assert str(matches[0]["id"]) == bundle.delivery.jobs_job_id
    assert (
        matches[0]["idempotency_key"]
        == f"admin-webhook-delivery:{delivery_id}"
    )


@pytest.mark.parametrize(
    ("auth_backend", "jobs_backend"),
    BACKEND_PAIRS,
    ids=("sqlite-sqlite", "sqlite-postgres", "postgres-sqlite", "postgres-postgres"),
)
@pytest.mark.integration
async def test_enqueue_six_crash_boundaries_converge_across_backend_matrix(
    auth_backend: str,
    jobs_backend: str,
    tmp_path: Path,
    test_db_pool,
    matrix_jobs_pg_dsn: str,
) -> None:
    manager = _jobs_manager(
        jobs_backend,
        tmp_path=tmp_path,
        jobs_pg_dsn=matrix_jobs_pg_dsn,
    )
    crash_cases = (
        ("before_claim_commit", EnqueueCrashPoint.BEFORE_CLAIM_COMMIT, False),
        ("after_claim_commit", EnqueueCrashPoint.AFTER_CLAIM_COMMIT, False),
        ("after_jobs_create_response", None, True),
        ("before_authnz_attach", EnqueueCrashPoint.BEFORE_AUTHNZ_ATTACH, False),
        ("after_authnz_attach", EnqueueCrashPoint.BEFORE_ATTACH_COMMIT, False),
        ("after_queued_commit", EnqueueCrashPoint.AFTER_QUEUED_COMMIT, False),
    )
    async with _auth_repository(
        auth_backend,
        tmp_path=tmp_path,
        test_db_pool=test_db_pool,
    ) as repository:
        for index, (label, crash_point, manager_crash) in enumerate(crash_cases):
            clock = MutableClock(NOW + timedelta(minutes=index * 5))
            webhook_id, delivery_id = await _seed_delivery(
                repository,
                f"{auth_backend}-{jobs_backend}-{label}",
                now=clock(),
            )
            tokens = TokenSource(f"{auth_backend}-{jobs_backend}-{label}")
            if manager_crash:
                first_queue = JobsDeliveryQueue(CrashAfterAdmissionManager(manager))
                crash_hook = _no_crash
            else:
                first_queue = JobsDeliveryQueue(manager)
                assert crash_point is not None
                crash_hook = OneShotCrash(crash_point)
            first = _reconciler(
                repository,
                first_queue,
                clock,
                tokens,
                crash_hook=crash_hook,
            )

            with pytest.raises(SimulatedCrash):
                await first.reconcile_enqueue_once()

            clock.advance(61)
            recovery = _reconciler(
                repository,
                JobsDeliveryQueue(manager),
                clock,
                tokens,
            )
            await recovery.reconcile_enqueue_once()
            await _assert_single_queued_identity(
                repository,
                manager,
                webhook_id,
                delivery_id,
            )


@pytest.mark.parametrize(
    ("auth_backend", "jobs_backend"),
    BACKEND_PAIRS,
    ids=("sqlite-sqlite", "sqlite-postgres", "postgres-sqlite", "postgres-postgres"),
)
@pytest.mark.parametrize(
    "claim_live_at_expiry",
    (True, False),
    ids=("live-claim", "expired-claim-concurrent-reclaim"),
)
@pytest.mark.integration
async def test_before_attach_crash_then_expiry_preserves_exact_cancel_recovery(
    auth_backend: str,
    jobs_backend: str,
    claim_live_at_expiry: bool,
    tmp_path: Path,
    test_db_pool,
    matrix_jobs_pg_dsn: str,
) -> None:
    manager = _jobs_manager(
        jobs_backend,
        tmp_path=tmp_path,
        jobs_pg_dsn=matrix_jobs_pg_dsn,
    )
    queue = CountingQueue(JobsDeliveryQueue(manager))
    clock = MutableClock(NOW)
    expiry_offset = 30 if claim_live_at_expiry else 120
    label = (
        f"{auth_backend}-{jobs_backend}-before-attach-expiry-"
        f"{'live' if claim_live_at_expiry else 'expired'}"
    )
    async with _auth_repository(
        auth_backend,
        tmp_path=tmp_path,
        test_db_pool=test_db_pool,
    ) as repository:
        webhook_id, delivery_id = await _seed_delivery(
            repository,
            label,
            now=clock(),
            expires_at=NOW + timedelta(seconds=expiry_offset),
        )
        with pytest.raises(SimulatedCrash):
            await _reconciler(
                repository,
                queue,
                clock,
                TokenSource(f"{label}-initial"),
                crash_hook=OneShotCrash(EnqueueCrashPoint.BEFORE_AUTHNZ_ATTACH),
            ).reconcile_enqueue_once()

        jobs_rows = manager.list_jobs(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            limit=100,
        )
        matching_jobs = [
            row
            for row in jobs_rows
            if row.get("payload") == {"delivery_id": delivery_id}
        ]
        assert len(matching_jobs) == 1
        assert matching_jobs[0]["status"] == "queued"

        clock.current = NOW + timedelta(seconds=expiry_offset)
        recovery_tokens = TokenSource(f"{label}-recovery")
        if claim_live_at_expiry:
            expiry = await repository.expire_due_deliveries(
                now=clock(),
                batch_size=100,
                token_factory=TokenSource(f"{label}-blind-expiry"),
            )
            assert expiry.expired == 0
            clock.current = NOW + timedelta(seconds=61)
            recovered_count = await _reconciler(
                repository,
                queue,
                clock,
                recovery_tokens,
            ).reconcile_enqueue_once()
        else:
            reclaimed = asyncio.Event()
            release_reconciler = asyncio.Event()

            async def pause_after_reclaim() -> None:
                reclaimed.set()
                await release_reconciler.wait()

            recovery_task = asyncio.create_task(
                _reconciler(
                    repository,
                    queue,
                    clock,
                    recovery_tokens,
                    after_claim_commit_hook=pause_after_reclaim,
                ).reconcile_enqueue_once()
            )
            await asyncio.wait_for(reclaimed.wait(), timeout=10)
            expiry = await repository.expire_due_deliveries(
                now=clock(),
                batch_size=100,
                token_factory=TokenSource(f"{label}-blind-expiry"),
            )
            assert expiry.expired == 0
            release_reconciler.set()
            recovered_count = await asyncio.wait_for(recovery_task, timeout=10)

        assert recovered_count == 1
        recovered = await repository.get_delivery_bundle(delivery_id)
        assert recovered is not None
        assert recovered.delivery.delivery.state is DeliveryState.DEAD
        assert recovered.delivery.delivery.reason_code is DeliveryReasonCode.DELIVERY_EXPIRED
        assert recovered.delivery.jobs_job_id == str(matching_jobs[0]["id"])
        assert recovered.delivery.enqueue_claim_token is None
        assert recovered.delivery.pending_jobs_disposition is JobsDispositionKind.CANCEL
        assert recovered.delivery.pending_jobs_disposition_token is not None
        assert recovered.delivery.jobs_disposition_applied
        assert queue.admit_calls == [delivery_id]
        assert queue.find_calls == [delivery_id]
        assert queue.cancel_tokens == [
            recovered.delivery.pending_jobs_disposition_token
        ]
        assert manager.get_job(int(matching_jobs[0]["id"]))["status"] == "cancelled"
        assert await repository.list_delivery_attempts(webhook_id, delivery_id) == ()


@pytest.mark.parametrize(
    ("auth_backend", "jobs_backend"),
    BACKEND_PAIRS,
    ids=("sqlite-sqlite", "sqlite-postgres", "postgres-sqlite", "postgres-postgres"),
)
@pytest.mark.integration
async def test_enqueue_revalidates_terminal_work_before_admission_across_backend_matrix(
    auth_backend: str,
    jobs_backend: str,
    tmp_path: Path,
    test_db_pool,
    matrix_jobs_pg_dsn: str,
) -> None:
    manager = _jobs_manager(
        jobs_backend,
        tmp_path=tmp_path,
        jobs_pg_dsn=matrix_jobs_pg_dsn,
    )
    queue = CountingQueue(JobsDeliveryQueue(manager))
    clock = MutableClock(NOW)
    async with _auth_repository(
        auth_backend,
        tmp_path=tmp_path,
        test_db_pool=test_db_pool,
    ) as repository:
        webhook_id, delivery_id = await _seed_delivery(
            repository,
            f"{auth_backend}-{jobs_backend}-post-claim-terminal",
            now=clock(),
        )
        reconciler = _reconciler(
            repository,
            queue,
            clock,
            TokenSource(f"{auth_backend}-{jobs_backend}-post-claim-terminal"),
            after_claim_commit_hook=TerminalizeAfterClaim(
                repository,
                webhook_id,
                clock,
            ),
        )

        assert await reconciler.reconcile_enqueue_once() == 1

        terminal = await repository.get_delivery_bundle(delivery_id)
        assert terminal is not None
        assert terminal.delivery.delivery.state is DeliveryState.CANCELED
        assert terminal.delivery.jobs_job_id is None
        assert terminal.delivery.enqueue_claim_token is None
        assert queue.admit_calls == []
        assert queue.find_calls == [delivery_id]


@pytest.mark.parametrize(
    ("auth_backend", "jobs_backend"),
    BACKEND_PAIRS,
    ids=("sqlite-sqlite", "sqlite-postgres", "postgres-sqlite", "postgres-postgres"),
)
@pytest.mark.parametrize(
    ("crash_point", "expected_status"),
    (
        (EnqueueCrashPoint.AFTER_ORPHAN_PREPARE, "queued"),
        (EnqueueCrashPoint.AFTER_JOBS_CANCEL, "cancelled"),
    ),
)
@pytest.mark.integration
async def test_terminal_orphan_crashes_recover_with_exact_claim_and_disposition(
    auth_backend: str,
    jobs_backend: str,
    crash_point: EnqueueCrashPoint,
    expected_status: str,
    tmp_path: Path,
    test_db_pool,
    matrix_jobs_pg_dsn: str,
) -> None:
    manager = _jobs_manager(
        jobs_backend,
        tmp_path=tmp_path,
        jobs_pg_dsn=matrix_jobs_pg_dsn,
    )
    queue = CountingQueue(JobsDeliveryQueue(manager))
    clock = MutableClock(NOW)
    async with _auth_repository(
        auth_backend,
        tmp_path=tmp_path,
        test_db_pool=test_db_pool,
    ) as repository:
        webhook_id, delivery_id = await _seed_delivery(
            repository,
            f"{auth_backend}-{jobs_backend}-{crash_point.value}",
            now=clock(),
        )
        await _claim(
            repository,
            delivery_id,
            token=opaque_token(f"{crash_point.value}-initial-claim"),
            now=clock(),
        )
        admitted = queue.queue.admit_delivery_job(
            delivery_id,
            clock() + timedelta(hours=72),
        )
        assert admitted.record is not None
        await _terminalize(
            repository,
            webhook_id,
            reason=DeliveryReasonCode.CANCELED_SECRET_ROTATION,
            now=clock(),
        )
        clock.advance(61)

        with pytest.raises(SimulatedCrash):
            await _reconciler(
                repository,
                queue,
                clock,
                TokenSource(f"{crash_point.value}-first"),
                crash_hook=OneShotCrash(crash_point),
            ).reconcile_enqueue_once()

        stranded = await repository.get_delivery_bundle(delivery_id)
        assert stranded is not None
        assert stranded.delivery.delivery.state is DeliveryState.CANCELED
        assert stranded.delivery.jobs_job_id == admitted.record.jobs_job_id
        assert stranded.delivery.enqueue_claim_token is not None
        assert stranded.delivery.pending_jobs_disposition is JobsDispositionKind.CANCEL
        assert stranded.delivery.pending_jobs_disposition_token is not None
        assert not stranded.delivery.jobs_disposition_applied
        assert manager.get_job(int(admitted.record.jobs_job_id))["status"] == expected_status
        disposition_token = stranded.delivery.pending_jobs_disposition_token
        assert queue.cancel_tokens == (
            []
            if crash_point is EnqueueCrashPoint.AFTER_ORPHAN_PREPARE
            else [disposition_token]
        )

        clock.advance(61)
        assert (
            await _reconciler(
                repository,
                queue,
                clock,
                TokenSource(f"{crash_point.value}-recovery"),
            ).reconcile_enqueue_once()
            == 1
        )
        recovered = await repository.get_delivery_bundle(delivery_id)
        assert recovered is not None
        assert recovered.delivery.jobs_job_id == admitted.record.jobs_job_id
        assert recovered.delivery.enqueue_claim_token is None
        assert recovered.delivery.pending_jobs_disposition_token == disposition_token
        assert recovered.delivery.jobs_disposition_applied
        assert manager.get_job(int(admitted.record.jobs_job_id))["status"] == "cancelled"
        assert queue.admit_calls == []
        assert queue.cancel_tokens == (
            [disposition_token]
            if crash_point is EnqueueCrashPoint.AFTER_ORPHAN_PREPARE
            else [disposition_token, disposition_token]
        )


@pytest.mark.parametrize(
    ("auth_backend", "jobs_backend"),
    BACKEND_PAIRS,
    ids=("sqlite-sqlite", "sqlite-postgres", "postgres-sqlite", "postgres-postgres"),
)
@pytest.mark.integration
async def test_enqueue_foreign_claim_cancellation_and_expiry_matrix(
    auth_backend: str,
    jobs_backend: str,
    tmp_path: Path,
    test_db_pool,
    matrix_jobs_pg_dsn: str,
) -> None:
    manager = _jobs_manager(
        jobs_backend,
        tmp_path=tmp_path,
        jobs_pg_dsn=matrix_jobs_pg_dsn,
    )
    queue = CountingQueue(JobsDeliveryQueue(manager))
    clock = MutableClock(NOW)
    async with _auth_repository(
        auth_backend,
        tmp_path=tmp_path,
        test_db_pool=test_db_pool,
    ) as repository:
        webhook_id, foreign_id = await _seed_delivery(
            repository,
            f"{auth_backend}-{jobs_backend}-foreign",
            now=clock(),
        )
        await _claim(
            repository,
            foreign_id,
            token=opaque_token("matrix-foreign-claim"),
            now=clock(),
        )
        foreign_reconciler = _reconciler(
            repository,
            queue,
            clock,
            TokenSource("matrix-foreign"),
        )
        assert await foreign_reconciler.reconcile_enqueue_once() == 0
        assert queue.admit_calls == []
        clock.advance(61)
        assert await foreign_reconciler.reconcile_enqueue_once() == 1
        await _assert_single_queued_identity(
            repository,
            manager,
            webhook_id,
            foreign_id,
        )
        foreign = await repository.get_delivery_bundle(foreign_id)
        assert foreign is not None and foreign.delivery.jobs_job_id is not None
        assert manager.cancel_job(
            int(foreign.delivery.jobs_job_id),
            reason="matrix_test_cleanup",
            expected_domain="admin_webhooks",
            expected_job_type="admin_webhook_delivery",
        )

        queue.reset_counts()
        missing_webhook, missing_id = await _seed_delivery(
            repository,
            f"{auth_backend}-{jobs_backend}-cancel-missing",
            now=clock(),
        )
        await _claim(
            repository,
            missing_id,
            token=opaque_token("matrix-cancel-missing-claim"),
            now=clock(),
        )
        await _terminalize(
            repository,
            missing_webhook,
            reason=DeliveryReasonCode.CANCELED_DISABLED,
            now=clock() + timedelta(seconds=1),
        )
        clock.advance(61)
        assert (
            await _reconciler(
                repository,
                queue,
                clock,
                TokenSource("matrix-cancel-missing"),
            ).reconcile_enqueue_once()
            == 1
        )
        missing = await repository.get_delivery_bundle(missing_id)
        assert missing is not None
        assert missing.delivery.delivery.state is DeliveryState.CANCELED
        assert (
            missing.delivery.delivery.reason_code
            is DeliveryReasonCode.CANCELED_DISABLED
        )
        assert missing.delivery.enqueue_claim_token is None
        assert missing.delivery.jobs_job_id is None
        assert queue.admit_calls == []
        assert queue.find_calls == [missing_id]

        queue.reset_counts()
        queued_webhook, queued_id = await _seed_delivery(
            repository,
            f"{auth_backend}-{jobs_backend}-cancel-queued",
            now=clock(),
        )
        queued_claim = opaque_token("matrix-cancel-queued-claim")
        await _claim(
            repository,
            queued_id,
            token=queued_claim,
            now=clock(),
        )
        queued_job = queue.queue.admit_delivery_job(
            queued_id,
            clock() + timedelta(hours=72),
        )
        assert queued_job.record is not None
        await _terminalize(
            repository,
            queued_webhook,
            reason=DeliveryReasonCode.CANCELED_SECRET_ROTATION,
            now=clock() + timedelta(seconds=1),
        )
        clock.advance(61)
        assert (
            await _reconciler(
                repository,
                queue,
                clock,
                TokenSource("matrix-cancel-queued"),
            ).reconcile_enqueue_once()
            == 1
        )
        queued = await repository.get_delivery_bundle(queued_id)
        assert queued is not None
        assert queued.delivery.delivery.state is DeliveryState.CANCELED
        assert queued.delivery.jobs_job_id == queued_job.record.jobs_job_id
        assert queued.delivery.pending_jobs_disposition is JobsDispositionKind.CANCEL
        assert queued.delivery.jobs_disposition_applied is True
        assert queued.delivery.enqueue_claim_token is None
        assert manager.get_job(int(queued_job.record.jobs_job_id))["status"] == "cancelled"
        assert queue.admit_calls == []
        assert queue.find_calls == [queued_id]
        assert queue.cancel_calls == [queued_id]

        queue.reset_counts()
        processing_webhook, processing_id = await _seed_delivery(
            repository,
            f"{auth_backend}-{jobs_backend}-cancel-processing",
            now=clock(),
        )
        await _claim(
            repository,
            processing_id,
            token=opaque_token("matrix-cancel-processing-claim"),
            now=clock(),
        )
        processing_job = queue.queue.admit_delivery_job(
            processing_id,
            clock() + timedelta(hours=72),
        )
        assert processing_job.record is not None
        acquired = manager.acquire_next_job(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            lease_seconds=120,
            worker_id="matrix-worker",
        )
        assert acquired is not None
        assert str(acquired["id"]) == processing_job.record.jobs_job_id
        await _terminalize(
            repository,
            processing_webhook,
            reason=DeliveryReasonCode.SUPERSEDED_CONFIG,
            now=clock() + timedelta(seconds=1),
        )
        clock.advance(61)
        assert (
            await _reconciler(
                repository,
                queue,
                clock,
                TokenSource("matrix-cancel-processing"),
            ).reconcile_enqueue_once()
            == 1
        )
        processing = await repository.get_delivery_bundle(processing_id)
        assert processing is not None
        assert processing.delivery.delivery.state is DeliveryState.SUPERSEDED
        assert processing.delivery.jobs_job_id == processing_job.record.jobs_job_id
        assert (
            processing.delivery.pending_jobs_disposition
            is JobsDispositionKind.CANCEL
        )
        assert processing.delivery.jobs_disposition_applied is False
        assert processing.delivery.enqueue_claim_token is not None
        assert manager.get_job(int(processing_job.record.jobs_job_id))["status"] == "processing"
        assert await repository.list_delivery_attempts(
            processing_webhook,
            processing_id,
        ) == ()
        assert queue.admit_calls == []
        assert queue.find_calls == [processing_id]
        assert queue.cancel_calls == [processing_id]

        queue.reset_counts()
        _, before_create_id = await _seed_delivery(
            repository,
            f"{auth_backend}-{jobs_backend}-expiry-before-create",
            now=clock(),
            expires_at=clock(),
        )
        assert (
            await _reconciler(
                repository,
                queue,
                clock,
                TokenSource("matrix-expiry-before-create"),
            ).reconcile_enqueue_once()
            == 1
        )
        before_create = await repository.get_delivery_bundle(before_create_id)
        assert before_create is not None
        assert before_create.delivery.delivery.state is DeliveryState.DEAD
        assert (
            before_create.delivery.delivery.reason_code
            is DeliveryReasonCode.DELIVERY_EXPIRED
        )
        assert before_create.delivery.jobs_job_id is None
        assert queue.admit_calls == []
        assert queue.find_calls == [before_create_id]

        queue.reset_counts()
        _, after_create_id = await _seed_delivery(
            repository,
            f"{auth_backend}-{jobs_backend}-expiry-after-create",
            now=clock(),
            expires_at=clock() + timedelta(seconds=30),
        )
        await _claim(
            repository,
            after_create_id,
            token=opaque_token("matrix-expiry-after-create-claim"),
            now=clock(),
        )
        after_create_job = queue.queue.admit_delivery_job(
            after_create_id,
            clock() + timedelta(seconds=30),
        )
        assert after_create_job.record is not None
        clock.advance(61)
        assert (
            await _reconciler(
                repository,
                queue,
                clock,
                TokenSource("matrix-expiry-after-create"),
            ).reconcile_enqueue_once()
            == 2
        )
        after_create = await repository.get_delivery_bundle(after_create_id)
        assert after_create is not None
        assert after_create.delivery.delivery.state is DeliveryState.DEAD
        assert (
            after_create.delivery.delivery.reason_code
            is DeliveryReasonCode.DELIVERY_EXPIRED
        )
        assert after_create.delivery.jobs_job_id == after_create_job.record.jobs_job_id
        assert after_create.delivery.jobs_disposition_applied is True
        assert (
            manager.get_job(int(after_create_job.record.jobs_job_id))["status"]
            == "cancelled"
        )
        assert queue.admit_calls == []
        assert queue.find_calls == [after_create_id, processing_id]
        assert queue.cancel_calls == [after_create_id, processing_id]


def _prepared_from_pending(pending) -> PreparedJobDisposition:
    kwargs = {
        "token": pending.token,
        "delivery_id": pending.delivery_id,
        "reason_code": (
            pending.reason_code.value if pending.reason_code is not None else None
        ),
    }
    if pending.kind is JobsDispositionKind.COMPLETE:
        return PreparedJobDisposition.complete(
            token=pending.token,
            delivery_id=pending.delivery_id,
            attempt_id=pending.attempt_id,
        )
    if pending.kind is JobsDispositionKind.RETRY:
        return PreparedJobDisposition.retry(
            **kwargs,
            attempt_id=pending.attempt_id,
            delay_seconds=pending.delay_seconds,
            not_before_at=pending.not_before_at,
        )
    if pending.kind is JobsDispositionKind.FAIL:
        return PreparedJobDisposition.fail(
            **kwargs,
            attempt_id=pending.attempt_id,
        )
    return PreparedJobDisposition.cancel(
        **kwargs,
        attempt_id=pending.attempt_id,
    )


@pytest.mark.parametrize(
    ("auth_backend", "jobs_backend"),
    BACKEND_PAIRS,
    ids=("sqlite-sqlite", "sqlite-postgres", "postgres-sqlite", "postgres-postgres"),
)
@pytest.mark.parametrize(
    ("kind", "attempt_state", "delivery_state", "reason", "delay", "jobs_state"),
    (
        (
            JobsDispositionKind.COMPLETE,
            AttemptState.SUCCEEDED,
            DeliveryState.SUCCEEDED,
            None,
            None,
            "completed",
        ),
        (
            JobsDispositionKind.RETRY,
            AttemptState.RETRYABLE,
            DeliveryState.RETRY_WAIT,
            DeliveryReasonCode.OUTCOME_UNKNOWN,
            60,
            "queued",
        ),
        (
            JobsDispositionKind.FAIL,
            AttemptState.FAILED,
            DeliveryState.DEAD,
            DeliveryReasonCode.ATTEMPT_BUDGET_EXHAUSTED,
            None,
            "failed",
        ),
        (
            JobsDispositionKind.CANCEL,
            AttemptState.CANCELED,
            DeliveryState.CANCELED,
            DeliveryReasonCode.CANCELED_DISABLED,
            None,
            "cancelled",
        ),
    ),
)
@pytest.mark.integration
async def test_authnz_disposition_lost_ack_reconciles_across_backend_matrix(
    auth_backend: str,
    jobs_backend: str,
    kind: JobsDispositionKind,
    attempt_state: AttemptState,
    delivery_state: DeliveryState,
    reason: DeliveryReasonCode | None,
    delay: int | None,
    jobs_state: str,
    tmp_path: Path,
    test_db_pool,
    matrix_jobs_pg_dsn: str,
) -> None:
    manager = _jobs_manager(
        jobs_backend,
        tmp_path=tmp_path,
        jobs_pg_dsn=matrix_jobs_pg_dsn,
    )
    queue = CountingQueue(JobsDeliveryQueue(manager))
    clock = MutableClock(NOW)
    label = f"task8-{auth_backend}-{jobs_backend}-{kind.value}"
    async with _auth_repository(
        auth_backend,
        tmp_path=tmp_path,
        test_db_pool=test_db_pool,
    ) as repository:
        webhook_id, delivery_id = await _seed_delivery(
            repository,
            label,
            now=clock(),
        )
        assert await _reconciler(
            repository,
            queue,
            clock,
            TokenSource(label),
        ).reconcile_enqueue_once() == 1
        bundle = await repository.get_delivery_bundle(delivery_id)
        assert bundle is not None and bundle.delivery.jobs_job_id is not None
        acquired = manager.acquire_next_job(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            lease_seconds=120,
            worker_id=f"{label}-worker",
        )
        assert acquired is not None
        attempt_id = canonical_uuid4(f"{label}-attempt")
        disposition_token = opaque_token(f"{label}-disposition")
        async with repository.transaction() as tx:
            reserved = await tx.reserve_jobs_attempt(
                delivery_id,
                bundle.delivery.jobs_job_id,
                acquired["lease_id"],
                attempt_id,
                10,
                clock(),
                clock() + timedelta(seconds=40),
                expected_delivery_config_version=(
                    bundle.delivery.delivery.delivery_config_version
                ),
                expected_secret_version=bundle.delivery.delivery.secret_version,
                disposition_token=opaque_token(f"{label}-unused"),
            )
            assert reserved is not None and reserved.reserved
            pending = await tx.finish_attempt_and_prepare_disposition(
                acquired["lease_id"],
                AttemptCompletion(
                    attempt_state=attempt_state,
                    delivery_state=delivery_state,
                    disposition=kind,
                    status_code=204 if kind is JobsDispositionKind.COMPLETE else 503,
                    latency_ms=5,
                    reason_code=reason,
                    requested_retry_delay_seconds=delay,
                    finished_at=clock() + timedelta(seconds=1),
                    completed_after_config_change=False,
                ),
                disposition_token,
                clock() + timedelta(seconds=60) if delay is not None else None,
                delivery_id=delivery_id,
                attempt_id=attempt_id,
                jobs_job_id=bundle.delivery.jobs_job_id,
            )
        assert pending is not None
        disposition = _prepared_from_pending(pending)
        applied = manager.apply_prepared_disposition(
            ApplyPreparedDispositionCommand(
                job_id=int(acquired["id"]),
                domain="admin_webhooks",
                queue="delivery",
                job_type="admin_webhook_delivery",
                expected_payload={"delivery_id": delivery_id},
                worker_id=acquired["worker_id"],
                lease_id=acquired["lease_id"],
                disposition=disposition,
            )
        )
        assert applied.state == jobs_state
        stranded = await repository.get_delivery_bundle(delivery_id)
        assert stranded is not None and not stranded.delivery.jobs_disposition_applied

        repaired = await _reconciler(
            repository,
            queue,
            clock,
            TokenSource(f"{label}-repair"),
        ).reconcile_pending_dispositions_once()

        assert repaired == 1
        recovered = await repository.get_delivery_bundle(delivery_id)
        assert recovered is not None and recovered.delivery.jobs_disposition_applied
        assert await repository.list_delivery_attempts(webhook_id, delivery_id)
        assert queue.admit_calls == [delivery_id]


@pytest.mark.parametrize(
    ("auth_backend", "jobs_backend"),
    BACKEND_PAIRS,
    ids=("sqlite-sqlite", "sqlite-postgres", "postgres-sqlite", "postgres-postgres"),
)
@pytest.mark.parametrize("origin", ("infrastructure", "recovery"))
@pytest.mark.integration
async def test_no_ack_defer_marker_is_historical_across_backend_matrix(
    auth_backend: str,
    jobs_backend: str,
    origin: str,
    tmp_path: Path,
    test_db_pool,
    matrix_jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_run_calls = 0
    original_run_prepared = WorkerSDK.run_prepared

    async def tracked_run_prepared(self, **kwargs) -> None:
        nonlocal sdk_run_calls
        sdk_run_calls += 1
        await original_run_prepared(self, **kwargs)

    monkeypatch.setattr(WorkerSDK, "run_prepared", tracked_run_prepared)
    manager = _jobs_manager(
        jobs_backend,
        tmp_path=tmp_path,
        jobs_pg_dsn=matrix_jobs_pg_dsn,
    )
    recording_manager = RecordingPreparedManager(manager)
    queue = JobsDeliveryQueue(manager)
    clock = MutableClock(
        datetime.now(timezone.utc)
        - (timedelta(seconds=99) if origin == "recovery" else timedelta())
    )
    ring = key_ring()
    label = f"task8-{auth_backend}-{jobs_backend}-{origin}"
    async with _auth_repository(
        auth_backend,
        tmp_path=tmp_path,
        test_db_pool=test_db_pool,
    ) as repository:
        webhook_id, delivery_id = await _seed_worker_delivery(
            repository,
            ring,
            label,
            now=clock(),
        )
        assert await _reconciler(
            repository,
            queue,
            clock,
            TokenSource(label),
        ).reconcile_enqueue_once() == 1

        stranded_lease_id: str | None = None
        if origin == "recovery":
            stranded = manager.acquire_next_job(
                domain="admin_webhooks",
                queue="delivery",
                job_type="admin_webhook_delivery",
                lease_seconds=1,
                worker_id=f"{label}-stranded-worker",
            )
            assert stranded is not None
            stranded_lease_id = str(stranded["lease_id"])
            bundle = await repository.get_delivery_bundle(delivery_id)
            assert bundle is not None and bundle.delivery.jobs_job_id is not None
            async with repository.transaction() as tx:
                reservation = await tx.reserve_jobs_attempt(
                    delivery_id,
                    bundle.delivery.jobs_job_id,
                    stranded_lease_id,
                    canonical_uuid4(f"{label}-stranded-attempt"),
                    10,
                    clock(),
                    clock() + timedelta(seconds=40),
                    expected_delivery_config_version=(
                        bundle.delivery.delivery.delivery_config_version
                    ),
                    expected_secret_version=bundle.delivery.delivery.secret_version,
                    disposition_token=opaque_token(f"{label}-unused-terminal"),
                )
            assert reservation is not None and reservation.reserved
            await asyncio.sleep(1.25)

        initial_ring = (
            WebhookKeyRing(
                {"other-key": base64.b64encode(b"z" * 32).decode("ascii")},
                primary_id="other-key",
            )
            if origin == "infrastructure"
            else ring
        )
        initial_executor = MatrixExecutor(lambda _request: _matrix_result("complete"))
        initial_handler = _worker_handler(
            repository,
            initial_ring,
            clock,
            initial_executor,
            f"{label}-initial",
        )
        callback_tokens: list[str] = []

        async def initial_callback(
            job: dict[str, Any],
            disposition: PreparedJobDisposition,
            result: PreparedDispositionResult,
        ) -> None:
            callback_tokens.append(disposition.token)
            await initial_handler.on_disposition_applied(job, disposition, result)

        initial_capture: dict[str, Any] = {}
        await _run_prepared_once(
            recording_manager,
            initial_handler,
            worker_id=f"{label}-worker",
            captures=initial_capture,
            on_applied=initial_callback,
            stop_after_apply=True,
        )
        assert sdk_run_calls == 1
        assert callback_tokens == []
        assert len(recording_manager.apply_commands) == 1
        initial_job = initial_capture["job"]
        disposition = initial_capture["disposition"]
        assert disposition.origin is (
            PreparedDispositionOrigin.INFRASTRUCTURE
            if origin == "infrastructure"
            else PreparedDispositionOrigin.RECOVERY
        )
        assert recording_manager.apply_results[0].state == "queued"
        if stranded_lease_id is not None:
            assert initial_job["lease_id"] != stranded_lease_id

        persisted = manager.get_job(int(initial_job["id"]))
        assert persisted is not None
        historical_evidence = _jobs_retry_evidence(persisted)
        historical_schedule = persisted["available_at"]
        historical_result = persisted["result"]
        historical_fingerprint = persisted["prepared_disposition_fingerprint"]
        assert historical_schedule is not None
        historical_record = queue.get_delivery_job(str(initial_job["id"]))
        assert historical_record is not None and historical_record.marker is not None
        historical_marker = historical_record.marker
        assert historical_marker.token == disposition.token
        assert historical_marker.origin is disposition.origin
        assert historical_marker.original_not_before_at is not None
        bundle = await repository.get_delivery_bundle(delivery_id)
        assert bundle is not None
        assert bundle.delivery.pending_jobs_disposition is None
        assert await _reconciler(
            repository,
            queue,
            clock,
            TokenSource(f"{label}-repair"),
        ).reconcile_pending_dispositions_once() == 0

        due_at = max(
            historical_marker.applied_at,
            historical_marker.original_not_before_at,
        )
        await asyncio.sleep(
            max(0.0, (due_at - datetime.now(timezone.utc)).total_seconds()) + 0.1
        )
        if origin == "recovery":
            clock.current = due_at + timedelta(microseconds=1)
        due_job = manager.get_job(int(initial_job["id"]))
        assert due_job is not None
        assert due_job["available_at"] == historical_schedule
        assert due_job["result"] == historical_result
        assert (
            due_job["prepared_disposition_fingerprint"]
            == historical_fingerprint
        )
        assert _jobs_retry_evidence(due_job) == historical_evidence

        recovery_executor = MatrixExecutor(lambda _request: _matrix_result("complete"))
        recovery_handler = _worker_handler(
            repository,
            ring,
            clock,
            recovery_executor,
            f"{label}-replacement",
        )

        async def recovery_callback(
            job: dict[str, Any],
            current_disposition: PreparedJobDisposition,
            result: PreparedDispositionResult,
        ) -> None:
            callback_tokens.append(current_disposition.token)
            await recovery_handler.on_disposition_applied(
                job,
                current_disposition,
                result,
            )

        recovery_capture: dict[str, Any] = {}
        await _run_prepared_once(
            recording_manager,
            recovery_handler,
            worker_id=f"{label}-replacement-worker",
            captures=recovery_capture,
            on_applied=recovery_callback,
        )

        assert sdk_run_calls == 2
        reacquired = recovery_capture["job"]
        current_disposition = recovery_capture["disposition"]
        reacquired_record = JobsDeliveryQueue.acquired_delivery_job(reacquired)
        assert reacquired_record.marker == historical_marker
        assert reacquired["lease_id"] != initial_job["lease_id"]
        assert _jobs_retry_evidence(reacquired) == historical_evidence
        assert [
            command.disposition.token
            for command in recording_manager.apply_commands
        ].count(disposition.token) == 1
        assert len(recording_manager.apply_commands) == 2
        assert recording_manager.apply_commands[1].lease_id == reacquired["lease_id"]
        assert callback_tokens == [current_disposition.token]
        assert disposition.token not in callback_tokens

        final_job = manager.get_job(int(initial_job["id"]))
        assert final_job is not None
        assert final_job["worker_id"] is None
        assert final_job["lease_id"] is None
        assert final_job["leased_until"] is None
        attempts = await repository.list_delivery_attempts(webhook_id, delivery_id)
        assert all(attempt.state is not AttemptState.PROCESSING for attempt in attempts)
        if origin == "infrastructure":
            assert len(initial_executor.requests) == 0
            assert len(recovery_executor.requests) == 1
            assert current_disposition.kind is PreparedDispositionKind.COMPLETE
            assert final_job["status"] == "completed"
        else:
            assert len(initial_executor.requests) == 0
            assert len(recovery_executor.requests) == 0
            assert current_disposition.kind is PreparedDispositionKind.RETRY
            assert final_job["status"] == "queued"


@pytest.mark.parametrize(
    ("auth_backend", "jobs_backend"),
    BACKEND_PAIRS,
    ids=("sqlite-sqlite", "sqlite-postgres", "postgres-sqlite", "postgres-postgres"),
)
@pytest.mark.parametrize("historical_origin", ("retry", "infrastructure", "recovery"))
@pytest.mark.integration
async def test_queued_cancel_replaces_only_an_exact_historical_marker(
    auth_backend: str,
    jobs_backend: str,
    historical_origin: str,
    tmp_path: Path,
    test_db_pool,
    matrix_jobs_pg_dsn: str,
) -> None:
    manager = _jobs_manager(
        jobs_backend,
        tmp_path=tmp_path,
        jobs_pg_dsn=matrix_jobs_pg_dsn,
    )
    queue = CountingQueue(JobsDeliveryQueue(manager))
    clock = MutableClock(NOW)
    label = f"queued-cancel-{auth_backend}-{jobs_backend}-{historical_origin}"
    async with _auth_repository(
        auth_backend,
        tmp_path=tmp_path,
        test_db_pool=test_db_pool,
    ) as repository:
        webhook_id, delivery_id = await _seed_delivery(
            repository,
            label,
            now=clock(),
        )
        assert await _reconciler(
            repository,
            queue,
            clock,
            TokenSource(label),
        ).reconcile_enqueue_once() == 1
        bundle = await repository.get_delivery_bundle(delivery_id)
        assert bundle is not None and bundle.delivery.jobs_job_id is not None
        acquired = manager.acquire_next_job(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            lease_seconds=120,
            worker_id=f"{label}-worker",
        )
        assert acquired is not None
        historical_token = opaque_token(f"{label}-historical")
        if historical_origin == "retry":
            historical = PreparedJobDisposition.retry(
                token=historical_token,
                delivery_id=delivery_id,
                attempt_id=canonical_uuid4(f"{label}-historical-attempt"),
                delay_seconds=60,
                not_before_at=clock() + timedelta(seconds=60),
                reason_code="http_server_error",
            )
        elif historical_origin == "infrastructure":
            historical = PreparedJobDisposition.infrastructure_defer(
                token=historical_token,
                delivery_id=delivery_id,
                reason_code="worker_infrastructure_unavailable",
            )
        else:
            historical = PreparedJobDisposition.recovery_defer_until(
                token=historical_token,
                delivery_id=delivery_id,
                not_before_at=clock() + timedelta(seconds=60),
                reason_code="attempt_not_stale",
            )
        applied = manager.apply_prepared_disposition(
            ApplyPreparedDispositionCommand(
                job_id=int(acquired["id"]),
                domain="admin_webhooks",
                queue="delivery",
                job_type="admin_webhook_delivery",
                expected_payload={"delivery_id": delivery_id},
                worker_id=acquired["worker_id"],
                lease_id=acquired["lease_id"],
                disposition=historical,
            )
        )
        assert applied.state == "queued"
        before_cancel = manager.get_job(int(acquired["id"]))

        cancel_token = opaque_token(f"{label}-cancel")
        async with repository.transaction() as tx:
            pending = await tx.cancel_registration_work(
                webhook_id,
                (2, 2),
                DeliveryReasonCode.CANCELED_DISABLED,
                lambda: cancel_token,
                clock(),
            )
        assert len(pending) == 1
        assert pending[0].token == cancel_token
        assert pending[0].token != historical_token

        repaired = await _reconciler(
            repository,
            queue,
            clock,
            TokenSource(f"{label}-repair"),
        ).reconcile_pending_dispositions_once()

        assert repaired == 1
        recovered = await repository.get_delivery_bundle(delivery_id)
        assert recovered is not None
        assert recovered.delivery.pending_jobs_disposition_token == cancel_token
        assert recovered.delivery.jobs_disposition_applied
        after_cancel = manager.get_job(int(acquired["id"]))
        assert after_cancel["status"] == "cancelled"
        assert after_cancel["retry_count"] == before_cancel["retry_count"]
        record = queue.queue.get_delivery_job(str(acquired["id"]))
        assert record is not None and record.marker is not None
        assert record.marker.kind is PreparedDispositionKind.CANCEL
        assert record.marker.origin is PreparedDispositionOrigin.AUTHNZ
        assert record.marker.token == cancel_token
        assert queue.cancel_tokens == [cancel_token]


_WORKER_CRASH_POINTS = (
    WorkerCrashPoint.BEFORE_RESERVATION_COMMIT,
    WorkerCrashPoint.AFTER_RESERVATION_COMMIT_BEFORE_IO,
    WorkerCrashPoint.AFTER_RECEIVER_RESULT_BEFORE_OUTCOME_COMMIT,
    WorkerCrashPoint.AFTER_OUTCOME_COMMIT_BEFORE_JOBS_APPLY,
    WorkerCrashPoint.AFTER_JOBS_APPLY_BEFORE_AUTHNZ_ACK,
    WorkerCrashPoint.AFTER_AUTHNZ_ACK_BEFORE_RETURN,
)


@pytest.mark.parametrize(
    ("auth_backend", "jobs_backend"),
    BACKEND_PAIRS,
    ids=("sqlite-sqlite", "sqlite-postgres", "postgres-sqlite", "postgres-postgres"),
)
@pytest.mark.integration
async def test_worker_authnz_outcome_crash_cross_product_across_backend_matrix(
    auth_backend: str,
    jobs_backend: str,
    tmp_path: Path,
    test_db_pool,
    matrix_jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk_run_calls = 0
    original_run_prepared = WorkerSDK.run_prepared

    async def tracked_run_prepared(self, **kwargs) -> None:
        nonlocal sdk_run_calls
        sdk_run_calls += 1
        await original_run_prepared(self, **kwargs)

    monkeypatch.setattr(WorkerSDK, "run_prepared", tracked_run_prepared)
    manager = _jobs_manager(
        jobs_backend,
        tmp_path=tmp_path,
        jobs_pg_dsn=matrix_jobs_pg_dsn,
    )
    queue = CountingQueue(JobsDeliveryQueue(manager))
    ring = key_ring()
    async with _auth_repository(
        auth_backend,
        tmp_path=tmp_path,
        test_db_pool=test_db_pool,
    ) as repository:
        for outcome in ("complete", "retry", "fail", "cancel"):
            for crash_point in _WORKER_CRASH_POINTS:
                clock = MutableClock(datetime.now(timezone.utc))
                label = (
                    f"worker-crash-{auth_backend}-{jobs_backend}-{outcome}-"
                    f"{crash_point.value}"
                )
                webhook_id, delivery_id = await _seed_worker_delivery(
                    repository,
                    ring,
                    label,
                    now=clock(),
                )
                assert await _reconciler(
                    repository,
                    queue,
                    clock,
                    TokenSource(f"{label}-enqueue"),
                ).reconcile_enqueue_once() == 1
                recording_manager = RecordingPreparedManager(manager)
                callback_boundary = crash_point in {
                    WorkerCrashPoint.AFTER_JOBS_APPLY_BEFORE_AUTHNZ_ACK,
                    WorkerCrashPoint.AFTER_AUTHNZ_ACK_BEFORE_RETURN,
                }
                acquired: dict[str, Any] | None = None
                if not callback_boundary:
                    acquired = manager.acquire_next_job(
                        domain="admin_webhooks",
                        queue="delivery",
                        job_type="admin_webhook_delivery",
                        lease_seconds=120,
                        worker_id=f"{label}-worker",
                    )
                    assert acquired is not None

                async def disable_after_io(
                    current_webhook_id: int = webhook_id,
                    current_clock: MutableClock = clock,
                ) -> None:
                    await _disable_worker_registration(
                        repository,
                        current_webhook_id,
                        now=current_clock(),
                    )

                result = _matrix_result(
                    outcome,
                    retry_delay_seconds=(
                        1
                        if outcome == "retry"
                        and crash_point
                        is WorkerCrashPoint.AFTER_JOBS_APPLY_BEFORE_AUTHNZ_ACK
                        else 1_800
                    ),
                )
                executor = MatrixExecutor(
                    lambda _request, result=result: result,
                    before_result=disable_after_io if outcome == "cancel" else None,
                )
                crashing = _worker_handler(
                    repository,
                    ring,
                    clock,
                    executor,
                    label,
                    crash_hook=WorkerOneShotCrash(crash_point),
                )
                crashed = False
                try:
                    if callback_boundary:
                        capture: dict[str, Any] = {}
                        await _run_prepared_once(
                            recording_manager,
                            crashing,
                            worker_id=f"{label}-worker",
                            captures=capture,
                            on_applied=crashing.on_disposition_applied,
                        )
                    else:
                        assert acquired is not None
                        disposition = await crashing(
                            acquired,
                            MatrixWorkerContext(acquired),
                        )
                        applied = _apply_worker_disposition(
                            recording_manager,
                            acquired,
                            disposition,
                        )
                        await crashing.on_disposition_applied(
                            acquired,
                            disposition,
                            applied,
                        )
                except SimulatedCrash:
                    crashed = True
                assert crashed, f"crash hook was not reached for {label}"
                if callback_boundary:
                    acquired = capture["job"]
                    disposition = capture["disposition"]
                    applied = recording_manager.apply_results[-1]
                assert acquired is not None
                if (
                    outcome == "cancel"
                    and crash_point
                    is WorkerCrashPoint.AFTER_RESERVATION_COMMIT_BEFORE_IO
                ):
                    await _disable_worker_registration(
                        repository,
                        webhook_id,
                        now=clock(),
                    )

                recovery = _worker_handler(
                    repository,
                    ring,
                    clock,
                    executor,
                    label,
                )
                replacement_executor: MatrixExecutor | None = None
                retry_boundary_five = (
                    outcome == "retry"
                    and crash_point
                    is WorkerCrashPoint.AFTER_JOBS_APPLY_BEFORE_AUTHNZ_ACK
                )
                if crash_point is WorkerCrashPoint.BEFORE_RESERVATION_COMMIT:
                    disposition = await recovery(
                        acquired,
                        MatrixWorkerContext(acquired),
                    )
                    applied = _apply_worker_disposition(
                        recording_manager,
                        acquired,
                        disposition,
                    )
                    await recovery.on_disposition_applied(
                        acquired,
                        disposition,
                        applied,
                    )
                elif crash_point in {
                    WorkerCrashPoint.AFTER_RESERVATION_COMMIT_BEFORE_IO,
                    WorkerCrashPoint.AFTER_RECEIVER_RESULT_BEFORE_OUTCOME_COMMIT,
                }:
                    deferred = await recovery(
                        acquired,
                        MatrixWorkerContext(acquired),
                    )
                    assert deferred.origin is PreparedDispositionOrigin.RECOVERY
                    attempts = await repository.list_delivery_attempts(
                        webhook_id,
                        delivery_id,
                    )
                    assert len(attempts) == 1
                    clock.current = attempts[0].started_at + timedelta(seconds=100)
                    disposition = await recovery(
                        acquired,
                        MatrixWorkerContext(acquired),
                    )
                    applied = _apply_worker_disposition(
                        recording_manager,
                        acquired,
                        disposition,
                    )
                    await recovery.on_disposition_applied(
                        acquired,
                        disposition,
                        applied,
                    )
                elif crash_point is WorkerCrashPoint.AFTER_OUTCOME_COMMIT_BEFORE_JOBS_APPLY:
                    disposition = await recovery(
                        acquired,
                        MatrixWorkerContext(acquired),
                    )
                    applied = _apply_worker_disposition(
                        recording_manager,
                        acquired,
                        disposition,
                    )
                    await recovery.on_disposition_applied(
                        acquired,
                        disposition,
                        applied,
                    )
                elif (
                    crash_point
                    is WorkerCrashPoint.AFTER_JOBS_APPLY_BEFORE_AUTHNZ_ACK
                ):
                    stranded = await repository.get_delivery_bundle(delivery_id)
                    assert stranded is not None
                    assert not stranded.delivery.jobs_disposition_applied
                    assert (
                        stranded.delivery.pending_jobs_disposition_token
                        == disposition.token
                    )
                    assert len(recording_manager.apply_commands) == 1
                    if retry_boundary_five:
                        old_job = manager.get_job(int(acquired["id"]))
                        assert old_job is not None
                        old_schedule = old_job["available_at"]
                        old_result = old_job["result"]
                        old_fingerprint = old_job[
                            "prepared_disposition_fingerprint"
                        ]
                        old_evidence = _jobs_retry_evidence(old_job)
                        assert old_job["status"] == "queued"
                        assert old_job["worker_id"] is None
                        assert old_job["lease_id"] is None
                        assert old_job["leased_until"] is None
                        old_record = queue.queue.get_delivery_job(
                            str(acquired["id"])
                        )
                        assert old_record is not None and old_record.marker is not None
                        old_marker = old_record.marker
                        assert old_marker.token == disposition.token
                        assert old_marker.kind is PreparedDispositionKind.RETRY
                        assert old_marker.original_not_before_at is not None
                        due_at = max(
                            old_marker.applied_at,
                            old_marker.original_not_before_at,
                        )
                        await asyncio.sleep(
                            max(
                                0.0,
                                (due_at - datetime.now(timezone.utc)).total_seconds(),
                            )
                            + 0.1
                        )
                        clock.current = max(
                            clock.current,
                            due_at + timedelta(microseconds=1),
                        )
                        due_job = manager.get_job(int(acquired["id"]))
                        assert due_job is not None
                        assert due_job["available_at"] == old_schedule
                        assert due_job["result"] == old_result
                        assert (
                            due_job["prepared_disposition_fingerprint"]
                            == old_fingerprint
                        )
                        assert _jobs_retry_evidence(due_job) == old_evidence

                        replacement_executor = MatrixExecutor(
                            lambda _request: _matrix_result("complete")
                        )
                        replacement_handler = _worker_handler(
                            repository,
                            ring,
                            clock,
                            replacement_executor,
                            f"{label}-replacement",
                        )
                        replacement_capture: dict[str, Any] = {}
                        await _run_prepared_once(
                            recording_manager,
                            replacement_handler,
                            worker_id=f"{label}-replacement-worker",
                            captures=replacement_capture,
                            on_applied=replacement_handler.on_disposition_applied,
                        )
                        reacquired = replacement_capture["job"]
                        reacquired_record = JobsDeliveryQueue.acquired_delivery_job(
                            reacquired
                        )
                        assert reacquired_record.marker == old_marker
                        assert reacquired["lease_id"] != acquired["lease_id"]
                        assert _jobs_retry_evidence(reacquired) == old_evidence
                        assert [
                            command.disposition.token
                            for command in recording_manager.apply_commands
                        ].count(disposition.token) == 1
                        assert len(recording_manager.apply_commands) == 2
                        assert (
                            recording_manager.apply_commands[1].lease_id
                            == reacquired["lease_id"]
                        )
                        recovered_job = manager.get_job(int(acquired["id"]))
                        assert recovered_job is not None
                        assert _jobs_retry_evidence(recovered_job) == old_evidence
                        assert recovered_job["status"] == "completed"
                        assert recovered_job["worker_id"] is None
                        assert recovered_job["lease_id"] is None
                        assert recovered_job["leased_until"] is None
                    else:
                        reconciler = _reconciler(
                            repository,
                            queue,
                            clock,
                            TokenSource(f"{label}-repair"),
                        )
                        assert (
                            await reconciler.reconcile_pending_dispositions_once()
                            == 1
                        )
                        assert (
                            await reconciler.reconcile_pending_dispositions_once()
                            == 0
                        )
                        await recovery.on_disposition_applied(
                            acquired,
                            disposition,
                            applied,
                        )
                        assert len(recording_manager.apply_commands) == 1
                else:
                    durable = await repository.get_delivery_bundle(delivery_id)
                    assert durable is not None
                    assert durable.delivery.jobs_disposition_applied
                    assert (
                        durable.delivery.pending_jobs_disposition_token
                        == disposition.token
                    )
                    await recovery.on_disposition_applied(
                        acquired,
                        disposition,
                        applied,
                    )
                    repaired = await _reconciler(
                        repository,
                        queue,
                        clock,
                        TokenSource(f"{label}-repair"),
                    ).reconcile_pending_dispositions_once()
                    assert repaired == 0
                    assert len(recording_manager.apply_commands) == 1

                expected_io = (
                    0
                    if crash_point
                    is WorkerCrashPoint.AFTER_RESERVATION_COMMIT_BEFORE_IO
                    else 2
                    if retry_boundary_five
                    else 1
                )
                actual_io = len(executor.requests) + (
                    len(replacement_executor.requests)
                    if replacement_executor is not None
                    else 0
                )
                assert actual_io == expected_io
                attempts = await repository.list_delivery_attempts(
                    webhook_id,
                    delivery_id,
                )
                assert len(attempts) == (2 if retry_boundary_five else 1)
                if retry_boundary_five:
                    assert [attempt.state for attempt in attempts] == [
                        AttemptState.RETRYABLE,
                        AttemptState.SUCCEEDED,
                    ]
                    assert [attempt.attempt_number for attempt in attempts] == [1, 2]
                elif crash_point in {
                    WorkerCrashPoint.AFTER_RESERVATION_COMMIT_BEFORE_IO,
                    WorkerCrashPoint.AFTER_RECEIVER_RESULT_BEFORE_OUTCOME_COMMIT,
                }:
                    assert attempts[0].state is AttemptState.OUTCOME_UNKNOWN
                else:
                    assert attempts[0].state is {
                        "complete": AttemptState.SUCCEEDED,
                        "retry": AttemptState.RETRYABLE,
                        "fail": AttemptState.FAILED,
                        "cancel": AttemptState.RETRYABLE,
                    }[outcome]
                final = await repository.get_delivery_bundle(delivery_id)
                assert final is not None and final.delivery.jobs_disposition_applied
                if outcome == "cancel" and not retry_boundary_five:
                    assert final.delivery.delivery.state is DeliveryState.CANCELED
                if retry_boundary_five:
                    assert final.delivery.delivery.state is DeliveryState.SUCCEEDED
                    assert all(
                        attempt.state is not AttemptState.PROCESSING
                        for attempt in attempts
                    )
                expected_jobs_state = {
                    DeliveryState.SUCCEEDED: "completed",
                    DeliveryState.RETRY_WAIT: "queued",
                    DeliveryState.DEAD: "failed",
                    DeliveryState.CANCELED: "cancelled",
                    DeliveryState.SUPERSEDED: "cancelled",
                }[final.delivery.delivery.state]
                assert manager.get_job(int(acquired["id"]))["status"] == expected_jobs_state
        assert sdk_run_calls == 9


@pytest.mark.parametrize(
    ("auth_backend", "jobs_backend"),
    BACKEND_PAIRS,
    ids=("sqlite-sqlite", "sqlite-postgres", "postgres-sqlite", "postgres-postgres"),
)
@pytest.mark.integration
async def test_worker_hard_cap_is_four_receiver_calls_across_backend_matrix(
    auth_backend: str,
    jobs_backend: str,
    tmp_path: Path,
    test_db_pool,
    matrix_jobs_pg_dsn: str,
) -> None:
    manager = _jobs_manager(
        jobs_backend,
        tmp_path=tmp_path,
        jobs_pg_dsn=matrix_jobs_pg_dsn,
    )
    queue = CountingQueue(JobsDeliveryQueue(manager))
    clock = MutableClock(datetime.now(timezone.utc))
    ring = key_ring()
    label = f"hard-cap-{auth_backend}-{jobs_backend}"

    def classify(request: AttemptExecutionRequest) -> AttemptExecutionResult:
        if request.attempt_number == 4:
            return AttemptExecutionResult(
                outcome=AttemptOutcome.FAILED,
                status_code=503,
                latency_ms=5,
                reason_code=AttemptReasonCode.ATTEMPT_BUDGET_EXHAUSTED,
                retry_delay_seconds=None,
            )
        return AttemptExecutionResult(
            outcome=AttemptOutcome.RETRYABLE,
            status_code=503,
            latency_ms=5,
            reason_code=AttemptReasonCode.HTTP_SERVER_ERROR,
            retry_delay_seconds=(60, 300, 1_800)[request.attempt_number - 1],
        )

    executor = MatrixExecutor(classify)
    async with _auth_repository(
        auth_backend,
        tmp_path=tmp_path,
        test_db_pool=test_db_pool,
    ) as repository:
        webhook_id, delivery_id = await _seed_worker_delivery(
            repository,
            ring,
            label,
            now=clock(),
        )
        assert await _reconciler(
            repository,
            queue,
            clock,
            TokenSource(f"{label}-enqueue"),
        ).reconcile_enqueue_once() == 1
        acquired = manager.acquire_next_job(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            lease_seconds=120,
            worker_id=f"{label}-worker-1",
        )
        assert acquired is not None
        handler = _worker_handler(
            repository,
            ring,
            clock,
            executor,
            label,
        )

        for attempt_number in range(1, 5):
            disposition = await handler(
                acquired,
                MatrixWorkerContext(acquired),
            )
            assert disposition.kind is (
                PreparedDispositionKind.FAIL
                if attempt_number == 4
                else PreparedDispositionKind.RETRY
            )
            applied = _apply_worker_disposition(manager, acquired, disposition)
            await handler.on_disposition_applied(acquired, disposition, applied)
            if attempt_number < 4:
                assert disposition.delay_seconds is not None
                clock.advance(disposition.delay_seconds)
                assert manager.reschedule_jobs(
                    domain="admin_webhooks",
                    queue="delivery",
                    job_type="admin_webhook_delivery",
                    status="queued",
                    set_now=True,
                ) == 1
                acquired = manager.acquire_next_job(
                    domain="admin_webhooks",
                    queue="delivery",
                    job_type="admin_webhook_delivery",
                    lease_seconds=120,
                    worker_id=f"{label}-worker-{attempt_number + 1}",
                )
                assert acquired is not None

        assert len(executor.requests) == 4
        assert [request.attempt_number for request in executor.requests] == [1, 2, 3, 4]
        attempts = await repository.list_delivery_attempts(webhook_id, delivery_id)
        assert [attempt.attempt_number for attempt in attempts] == [1, 2, 3, 4]
        assert attempts[-1].state is AttemptState.FAILED
        assert (
            attempts[-1].reason_code
            is DeliveryReasonCode.ATTEMPT_BUDGET_EXHAUSTED
        )
        final = await repository.get_delivery_bundle(delivery_id)
        assert final is not None
        assert final.delivery.delivery.state is DeliveryState.DEAD
        assert (
            final.delivery.delivery.reason_code
            is DeliveryReasonCode.ATTEMPT_BUDGET_EXHAUSTED
        )
        assert final.delivery.jobs_disposition_applied
        assert manager.acquire_next_job(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            lease_seconds=120,
            worker_id=f"{label}-worker-5",
        ) is None
        assert len(executor.requests) == 4


@pytest.mark.parametrize(
    ("auth_backend", "jobs_backend"),
    BACKEND_PAIRS,
    ids=("sqlite-sqlite", "sqlite-postgres", "postgres-sqlite", "postgres-postgres"),
)
@pytest.mark.integration
async def test_exact_late_writer_cannot_replace_stale_recovery_across_backend_matrix(
    auth_backend: str,
    jobs_backend: str,
    tmp_path: Path,
    test_db_pool,
    matrix_jobs_pg_dsn: str,
) -> None:
    manager = _jobs_manager(
        jobs_backend,
        tmp_path=tmp_path,
        jobs_pg_dsn=matrix_jobs_pg_dsn,
    )
    queue = CountingQueue(JobsDeliveryQueue(manager))
    clock = MutableClock(datetime.now(timezone.utc))
    ring = key_ring()
    label = f"late-writer-{auth_backend}-{jobs_backend}"
    executor = MatrixExecutor(lambda _request: _matrix_result("complete"))
    async with _auth_repository(
        auth_backend,
        tmp_path=tmp_path,
        test_db_pool=test_db_pool,
    ) as repository:
        webhook_id, delivery_id = await _seed_worker_delivery(
            repository,
            ring,
            label,
            now=clock(),
        )
        assert await _reconciler(
            repository,
            queue,
            clock,
            TokenSource(f"{label}-enqueue"),
        ).reconcile_enqueue_once() == 1
        acquired = manager.acquire_next_job(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            lease_seconds=120,
            worker_id=f"{label}-worker",
        )
        assert acquired is not None
        crashing = _worker_handler(
            repository,
            ring,
            clock,
            executor,
            label,
            crash_hook=WorkerOneShotCrash(
                WorkerCrashPoint.AFTER_RESERVATION_COMMIT_BEFORE_IO
            ),
        )
        with pytest.raises(SimulatedCrash):
            await crashing(acquired, MatrixWorkerContext(acquired))
        attempts = await repository.list_delivery_attempts(webhook_id, delivery_id)
        assert len(attempts) == 1
        attempt = attempts[0]
        clock.current = attempt.started_at + timedelta(seconds=100)
        recovery = _worker_handler(
            repository,
            ring,
            clock,
            executor,
            label,
        )
        disposition = await recovery(acquired, MatrixWorkerContext(acquired))
        assert disposition.kind is PreparedDispositionKind.RETRY
        recovered_before = await repository.get_delivery_bundle(delivery_id)
        attempts_before = await repository.list_delivery_attempts(
            webhook_id,
            delivery_id,
        )
        assert recovered_before is not None

        async with repository.transaction() as tx:
            late = await tx.finish_attempt_and_prepare_disposition(
                acquired["lease_id"],
                AttemptCompletion(
                    attempt_state=AttemptState.SUCCEEDED,
                    delivery_state=DeliveryState.SUCCEEDED,
                    disposition=JobsDispositionKind.COMPLETE,
                    status_code=204,
                    latency_ms=5,
                    reason_code=None,
                    requested_retry_delay_seconds=None,
                    finished_at=clock() + timedelta(seconds=1),
                    completed_after_config_change=False,
                ),
                opaque_token(f"{label}-late-disposition"),
                None,
                delivery_id=delivery_id,
                attempt_id=attempt.id,
                jobs_job_id=str(acquired["id"]),
            )
        assert late is None
        assert await repository.get_delivery_bundle(delivery_id) == recovered_before
        assert await repository.list_delivery_attempts(
            webhook_id,
            delivery_id,
        ) == attempts_before
        assert executor.requests == []
