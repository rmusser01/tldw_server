"""Supervised runtime for canonical admin-webhook delivery."""

from __future__ import annotations

import asyncio
import os
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, NoReturn
from uuid import uuid4

from loguru import logger

from tldw_Server_API.app.core.Admin_Webhooks.config import (
    AdminWebhookMode,
    AdminWebhookSettings,
    WebhookRouteSelection,
)
from tldw_Server_API.app.core.Admin_Webhooks.crypto import load_webhook_key_ring
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    DeliveryRuntimeComponent,
    DeliveryRuntimeReasonCode,
)
from tldw_Server_API.app.core.Admin_Webhooks.executor import DeliveryAttemptExecutor
from tldw_Server_API.app.core.Admin_Webhooks.observability import (
    AdminWebhookDeliveryCapability,
    AdminWebhookMetrics,
    JobManagerJobsCapabilityProbe,
    JobsCapabilityStatus,
)
from tldw_Server_API.app.core.Admin_Webhooks.reconciler import (
    AdminWebhookReconciler,
    JobsDeliveryQueue,
)
from tldw_Server_API.app.core.Admin_Webhooks.worker import (
    AdminWebhookPreparedHandler,
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
    RuntimeHeartbeatWrite,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    ADMIN_WEBHOOK_DELIVERY_DOMAIN,
    ADMIN_WEBHOOK_DELIVERY_JOB_TYPE,
    ADMIN_WEBHOOK_DELIVERY_QUEUE,
)
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK

_RESTART_DELAY_SECONDS = 1
_Loop = Callable[[asyncio.Event, object], Awaitable[None]]


class _UnavailableJobsCapabilityProbe:
    async def status(self) -> JobsCapabilityStatus:
        return JobsCapabilityStatus(
            database_ready=False,
            queue_ready=False,
            job_type_ready=False,
            backend="unavailable",
        )


class _UnavailableJobsQueue:
    @staticmethod
    def _raise() -> NoReturn:
        raise RuntimeError("Jobs delivery queue is unavailable")

    def admit_delivery_job(self, *_args: object, **_kwargs: object) -> NoReturn:
        self._raise()

    def find_delivery_job_by_identity(
        self,
        *_args: object,
        **_kwargs: object,
    ) -> NoReturn:
        self._raise()

    def get_delivery_job(self, *_args: object, **_kwargs: object) -> NoReturn:
        self._raise()

    def apply_queued_cancel(self, *_args: object, **_kwargs: object) -> NoReturn:
        self._raise()


@dataclass(frozen=True)
class _RuntimeComponents:
    settings: AdminWebhookSettings
    worker_repository: AdminWebhookRepository
    reconciler_repository: AdminWebhookRepository
    retention_repository: AdminWebhookRepository
    capability: AdminWebhookDeliveryCapability
    worker_sdk: WorkerSDK | None
    worker_handler: AdminWebhookPreparedHandler | None
    reconciler: AdminWebhookReconciler
    metrics: AdminWebhookMetrics
    worker_instance_id: str
    reconciler_instance_id: str
    retention_instance_id: str
    token_factory: Callable[[], str]
    clock: Callable[[], datetime]


async def _wait_interruptibly(stop_event: asyncio.Event, seconds: float) -> None:
    if seconds <= 0:
        await asyncio.sleep(0)
        return
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except TimeoutError:
        return


async def _supervise_loop(
    name: str,
    loop: _Loop,
    stop_event: asyncio.Event,
    components: object,
) -> None:
    while not stop_event.is_set():
        try:
            await loop(stop_event, components)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - one component must not stop its peers
            logger.exception("Canonical admin-webhook {} loop failed", name)
        if not stop_event.is_set():
            await _wait_interruptibly(stop_event, _RESTART_DELAY_SECONDS)


async def _build_runtime_components() -> _RuntimeComponents:
    settings = AdminWebhookSettings.from_environment(os.environ)
    if settings.mode is not AdminWebhookMode.ON or settings.route_selection is not WebhookRouteSelection.CANONICAL:
        raise RuntimeError("canonical admin-webhook runtime mode is not enabled")
    pool = await get_db_pool()
    health_repository = AdminWebhookRepository(pool)
    worker_repository = AdminWebhookRepository(pool)
    reconciler_repository = AdminWebhookRepository(pool)
    retention_repository = AdminWebhookRepository(pool)
    manager: JobManager | None
    try:
        manager = await asyncio.to_thread(JobManager)
    except Exception as exc:  # noqa: BLE001 - runtime remains observable
        logger.bind(error_type=type(exc).__name__).warning("Canonical admin-webhook Jobs preflight is unavailable")
        manager = None
    if manager is None:
        queue = _UnavailableJobsQueue()
        jobs_probe = _UnavailableJobsCapabilityProbe()
        jobs_backend = "unavailable"
    else:
        queue = JobsDeliveryQueue(manager)
        jobs_probe = JobManagerJobsCapabilityProbe(manager)
        jobs_backend = str(getattr(manager, "backend", "unavailable"))
    key_ring_result = load_webhook_key_ring()
    metrics = AdminWebhookMetrics()

    def clock() -> datetime:
        return datetime.now(timezone.utc)

    def token_factory() -> str:
        return secrets.token_hex(32)

    capability = AdminWebhookDeliveryCapability(
        repository=health_repository,
        key_ring_result=key_ring_result,
        jobs_probe=jobs_probe,
        heartbeat_freshness_seconds=(settings.delivery_heartbeat_freshness_seconds),
        metrics=metrics,
    )
    reconciler = AdminWebhookReconciler(
        repository=reconciler_repository,
        queue=queue,
        token_factory=token_factory,
        clock=clock,
        claim_ttl_seconds=settings.delivery_claim_ttl_seconds,
        failure_observer=lambda failure: metrics.enqueue_failure(
            failure,
            backend=jobs_backend,
        ),
        success_observer=lambda success: metrics.enqueue_success(
            success,
            backend=jobs_backend,
        ),
    )
    worker_id = str(uuid4())
    worker_sdk: WorkerSDK | None = None
    worker_handler: AdminWebhookPreparedHandler | None = None
    if key_ring_result.ring is not None and manager is not None:
        worker_handler = AdminWebhookPreparedHandler(
            repository=worker_repository,
            key_ring=key_ring_result.ring,
            settings=settings,
            executor=DeliveryAttemptExecutor(
                allow_http_dev=settings.allow_http_dev,
            ),
            token_factory=token_factory,
            attempt_id_factory=lambda: str(uuid4()),
            clock=clock,
            metrics=metrics,
        )
        worker_sdk = WorkerSDK(
            manager,
            WorkerConfig(
                domain=ADMIN_WEBHOOK_DELIVERY_DOMAIN,
                queue=ADMIN_WEBHOOK_DELIVERY_QUEUE,
                worker_id=worker_id,
                lease_seconds=120,
                renew_jitter_seconds=5,
                renew_threshold_seconds=20,
                backoff_base_seconds=settings.delivery_loop_interval_seconds,
                backoff_max_seconds=30,
            ),
        )
    return _RuntimeComponents(
        settings=settings,
        worker_repository=worker_repository,
        reconciler_repository=reconciler_repository,
        retention_repository=retention_repository,
        capability=capability,
        worker_sdk=worker_sdk,
        worker_handler=worker_handler,
        reconciler=reconciler,
        metrics=metrics,
        worker_instance_id=worker_id,
        reconciler_instance_id=str(uuid4()),
        retention_instance_id=str(uuid4()),
        token_factory=token_factory,
        clock=clock,
    )


async def _write_heartbeat(
    repository: object,
    *,
    component: DeliveryRuntimeComponent,
    instance_id: str,
    ready: bool,
    reason_code: DeliveryRuntimeReasonCode | None,
    now: datetime,
) -> bool:
    try:
        await repository.upsert_runtime_heartbeat(
            RuntimeHeartbeatWrite(
                component=component,
                instance_id=instance_id,
                ready=ready,
                reason_code=reason_code,
                heartbeat_at=now,
                last_success_at=now if ready else None,
            )
        )
    except Exception:  # noqa: BLE001 - heartbeat failure cannot stop recovery
        return False
    return True


async def _worker_pre_acquire(components: object) -> bool:
    now = components.clock()
    try:
        status = await components.capability.status(now)
        ready = bool(status.acquisition_ready)
        reason = status.acquisition_reason_code
    except Exception:  # noqa: BLE001 - acquisition is fail-closed
        status = None
        ready = False
        reason = DeliveryRuntimeReasonCode.DATABASE_UNAVAILABLE
    heartbeat_written = await _write_heartbeat(
        components.worker_repository,
        component=DeliveryRuntimeComponent.WORKER,
        instance_id=components.worker_instance_id,
        ready=ready,
        reason_code=None if ready else reason,
        now=now,
    )
    return ready and heartbeat_written


async def _run_worker_loop(
    stop_event: asyncio.Event,
    components: object,
) -> None:
    try:
        if components.worker_sdk is None or components.worker_handler is None:
            while not stop_event.is_set():
                await _worker_pre_acquire(components)
                await _wait_interruptibly(
                    stop_event,
                    components.settings.delivery_loop_interval_seconds,
                )
            return
        run_task = asyncio.create_task(
            components.worker_sdk.run_prepared(
                handler=components.worker_handler,
                pre_acquire_guard=lambda: _worker_pre_acquire(components),
                handler_error_disposition=(components.worker_handler.handler_error_disposition),
                job_type=ADMIN_WEBHOOK_DELIVERY_JOB_TYPE,
                on_disposition_applied=(components.worker_handler.on_disposition_applied),
            ),
            name="admin_webhook_delivery_prepared_worker",
        )
        stop_task = asyncio.create_task(
            stop_event.wait(),
            name="admin_webhook_delivery_worker_stop",
        )
        try:
            done, _pending = await asyncio.wait(
                {run_task, stop_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if run_task in done:
                await run_task
        finally:
            components.worker_sdk.stop()
            if not stop_task.done():
                stop_task.cancel()
            await asyncio.gather(stop_task, return_exceptions=True)
            if not run_task.done():
                await run_task
    finally:
        await _write_heartbeat(
            components.worker_repository,
            component=DeliveryRuntimeComponent.WORKER,
            instance_id=components.worker_instance_id,
            ready=False,
            reason_code=DeliveryRuntimeReasonCode.WORKER_UNAVAILABLE,
            now=components.clock(),
        )


async def _run_reconciler_loop(
    stop_event: asyncio.Event,
    components: object,
) -> None:
    try:
        while not stop_event.is_set():
            failed = False
            for operation in (
                components.reconciler.reconcile_enqueue_once,
                components.reconciler.reconcile_pending_dispositions_once,
                components.reconciler.recover_stale_test_attempts_once,
            ):
                try:
                    await operation()
                except Exception:  # noqa: BLE001 - stages recover independently
                    failed = True
            try:
                expiry = await components.reconciler.reconcile_expired_once()
                components.metrics.expiries_committed(expiry.expired)
            except Exception:  # noqa: BLE001 - expiry cannot stop other repair
                failed = True
            await _write_heartbeat(
                components.reconciler_repository,
                component=DeliveryRuntimeComponent.RECONCILER,
                instance_id=components.reconciler_instance_id,
                ready=not failed,
                reason_code=(DeliveryRuntimeReasonCode.RECONCILER_UNAVAILABLE if failed else None),
                now=components.clock(),
            )
            await _wait_interruptibly(
                stop_event,
                components.settings.delivery_loop_interval_seconds,
            )
    finally:
        await _write_heartbeat(
            components.reconciler_repository,
            component=DeliveryRuntimeComponent.RECONCILER,
            instance_id=components.reconciler_instance_id,
            ready=False,
            reason_code=DeliveryRuntimeReasonCode.RECONCILER_UNAVAILABLE,
            now=components.clock(),
        )


async def _run_retention_loop(
    stop_event: asyncio.Event,
    components: object,
) -> None:
    try:
        while not stop_event.is_set():
            now = components.clock()
            try:
                result = await components.retention_repository.purge_retained_rows(
                    now,
                    now - timedelta(days=components.settings.delivery_retention_days),
                    200,
                )
                components.metrics.retention_committed(result)
                ready = True
                reason = None
            except Exception:  # noqa: BLE001 - retention is independently degraded
                ready = False
                reason = DeliveryRuntimeReasonCode.RETENTION_UNAVAILABLE
            await _write_heartbeat(
                components.retention_repository,
                component=DeliveryRuntimeComponent.RETENTION,
                instance_id=components.retention_instance_id,
                ready=ready,
                reason_code=reason,
                now=now,
            )
            await _wait_interruptibly(
                stop_event,
                components.settings.delivery_loop_interval_seconds,
            )
    finally:
        await _write_heartbeat(
            components.retention_repository,
            component=DeliveryRuntimeComponent.RETENTION,
            instance_id=components.retention_instance_id,
            ready=False,
            reason_code=DeliveryRuntimeReasonCode.RETENTION_UNAVAILABLE,
            now=components.clock(),
        )


async def run_admin_webhook_delivery_runtime(stop_event: Any) -> None:
    """Run and await the three independent canonical delivery loops."""

    if not isinstance(stop_event, asyncio.Event):
        raise TypeError("stop event must be an asyncio.Event")
    components = await _build_runtime_components()
    loops = (
        ("worker", _run_worker_loop),
        ("reconciler", _run_reconciler_loop),
        ("retention", _run_retention_loop),
    )
    tasks = [
        asyncio.create_task(
            _supervise_loop(name, loop, stop_event, components),
            name=f"admin_webhook_delivery_{name}_loop",
        )
        for name, loop in loops
    ]
    try:
        await stop_event.wait()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


__all__ = ["run_admin_webhook_delivery_runtime"]
