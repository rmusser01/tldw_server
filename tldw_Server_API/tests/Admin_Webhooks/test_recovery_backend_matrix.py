from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

import asyncpg
import pytest

from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    DeliveryReasonCode,
    DeliveryState,
    JobsDispositionKind,
)
from tldw_Server_API.app.core.Admin_Webhooks.reconciler import (
    AdminWebhookReconciler,
    EnqueueCrashPoint,
    JobsDeliveryQueue,
)
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
    ensure_admin_webhook_canonical_tables_pg,
)
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.tests.Admin_Webhooks.test_event_expansion import (
    NOW,
    canonical_uuid4,
    event_insert,
    opaque_token,
    seed_registration,
)

pytest_plugins = (
    "tldw_Server_API.tests.AuthNZ.conftest",
    "tldw_Server_API.tests.Jobs.conftest",
)
pytestmark = pytest.mark.integration

BACKEND_PAIRS = (
    ("sqlite", "sqlite"),
    ("sqlite", "postgres"),
    ("postgres", "sqlite"),
    ("postgres", "postgres"),
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
async def test_enqueue_six_crash_boundaries_converge_across_backend_matrix(
    auth_backend: str,
    jobs_backend: str,
    tmp_path: Path,
    test_db_pool,
    jobs_pg_dsn: str,
) -> None:
    manager = _jobs_manager(
        jobs_backend,
        tmp_path=tmp_path,
        jobs_pg_dsn=jobs_pg_dsn,
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
async def test_enqueue_revalidates_terminal_work_before_admission_across_backend_matrix(
    auth_backend: str,
    jobs_backend: str,
    tmp_path: Path,
    test_db_pool,
    jobs_pg_dsn: str,
) -> None:
    manager = _jobs_manager(
        jobs_backend,
        tmp_path=tmp_path,
        jobs_pg_dsn=jobs_pg_dsn,
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
async def test_terminal_orphan_crashes_recover_with_exact_claim_and_disposition(
    auth_backend: str,
    jobs_backend: str,
    crash_point: EnqueueCrashPoint,
    expected_status: str,
    tmp_path: Path,
    test_db_pool,
    jobs_pg_dsn: str,
) -> None:
    manager = _jobs_manager(
        jobs_backend,
        tmp_path=tmp_path,
        jobs_pg_dsn=jobs_pg_dsn,
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
async def test_enqueue_foreign_claim_cancellation_and_expiry_matrix(
    auth_backend: str,
    jobs_backend: str,
    tmp_path: Path,
    test_db_pool,
    jobs_pg_dsn: str,
) -> None:
    manager = _jobs_manager(
        jobs_backend,
        tmp_path=tmp_path,
        jobs_pg_dsn=jobs_pg_dsn,
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
