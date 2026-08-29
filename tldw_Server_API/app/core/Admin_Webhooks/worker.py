"""Lease-aware prepared worker for exactly one canonical webhook attempt."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol

from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
    AttemptCompletion,
    DeliveryBundle,
    PendingJobsDisposition,
)
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    OperationOutcome,
    PreparedDispositionKind,
    PreparedDispositionOrigin,
    PreparedDispositionResult,
    PreparedJobDisposition,
    admin_webhook_disposition_marker_matches,
    canonical_admin_webhook_delivery_id,
)

from .config import AdminWebhookSettings
from .crypto import WebhookKeyRing
from .delivery import registration_work_lifecycle_reason
from .domain import (
    AttemptState,
    DeliveryReasonCode,
    DeliveryState,
    JobsDispositionKind,
    validate_webhook_target,
)
from .executor import (
    AttemptExecutionRequest,
    AttemptExecutionResult,
    AttemptOutcome,
    DeliveryAttemptExecutor,
)
from .reconciler import (
    JobsDeliveryConflictError,
    JobsDeliveryQueue,
    _prepared_from_pending,
)

_SIGNING_SECRET = re.compile(r"whsec_[0-9a-f]{64}\Z")


class WorkerCrashPoint(str, Enum):
    """Deterministic boundaries for Task 8 crash recovery tests."""

    BEFORE_RESERVATION_COMMIT = "before_reservation_commit"
    AFTER_RESERVATION_COMMIT_BEFORE_IO = "after_reservation_commit_before_io"
    AFTER_RECEIVER_RESULT_BEFORE_OUTCOME_COMMIT = (
        "after_receiver_result_before_outcome_commit"
    )
    AFTER_OUTCOME_COMMIT_BEFORE_JOBS_APPLY = (
        "after_outcome_commit_before_jobs_apply"
    )
    AFTER_JOBS_APPLY_BEFORE_AUTHNZ_ACK = "after_jobs_apply_before_authnz_ack"
    AFTER_AUTHNZ_ACK_BEFORE_RETURN = "after_authnz_ack_before_return"


class _ExecutionContext(Protocol):
    async def ensure_lease_horizon(self, seconds: int) -> bool: ...

    def snapshot(self) -> Any: ...


class _Executor(Protocol):
    async def execute(
        self,
        request: AttemptExecutionRequest,
    ) -> AttemptExecutionResult: ...


def _aware_utc(value: datetime, *, field: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _pending_from_bundle(bundle: DeliveryBundle) -> PendingJobsDisposition | None:
    delivery = bundle.delivery
    if delivery.pending_jobs_disposition is None or delivery.jobs_disposition_applied:
        return None
    if delivery.jobs_job_id is None or delivery.pending_jobs_disposition_token is None:
        raise ValueError("pending disposition coordinates are invalid")
    return PendingJobsDisposition(
        delivery_id=delivery.delivery.id,
        jobs_job_id=delivery.jobs_job_id,
        attempt_id=delivery.current_attempt_id,
        kind=delivery.pending_jobs_disposition,
        delay_seconds=delivery.pending_jobs_disposition_delay_seconds,
        token=delivery.pending_jobs_disposition_token,
        not_before_at=delivery.pending_jobs_disposition_not_before_at,
        reason_code=delivery.delivery.reason_code,
    )


class AdminWebhookPreparedHandler:
    """Prepare one exact Jobs disposition without finalizing Jobs directly."""

    def __init__(
        self,
        *,
        repository: AdminWebhookRepository,
        key_ring: WebhookKeyRing,
        settings: AdminWebhookSettings,
        executor: _Executor | DeliveryAttemptExecutor,
        token_factory: Callable[[], str],
        attempt_id_factory: Callable[[], str],
        clock: Callable[[], datetime],
        crash_hook: Callable[[WorkerCrashPoint], None] | None = None,
    ) -> None:
        if not isinstance(repository, AdminWebhookRepository):
            raise TypeError("repository is invalid")
        if not isinstance(key_ring, WebhookKeyRing):
            raise TypeError("key ring is invalid")
        if not isinstance(settings, AdminWebhookSettings):
            raise TypeError("settings are invalid")
        if not callable(getattr(executor, "execute", None)):
            raise TypeError("executor is invalid")
        for value, field in (
            (token_factory, "token factory"),
            (attempt_id_factory, "attempt ID factory"),
            (clock, "clock"),
        ):
            if not callable(value):
                raise TypeError(f"{field} is invalid")
        if crash_hook is not None and not callable(crash_hook):
            raise TypeError("crash hook is invalid")
        self._repository = repository
        self._key_ring = key_ring
        self._settings = settings
        self._executor = executor
        self._token_factory = token_factory
        self._attempt_id_factory = attempt_id_factory
        self._clock = clock
        self._crash_hook = crash_hook

    def _crash(self, point: WorkerCrashPoint) -> None:
        if self._crash_hook is not None:
            self._crash_hook(point)

    def _token(self) -> str:
        return self._token_factory()

    def _infrastructure_defer(
        self,
        delivery_id: str,
        *,
        reason_code: str = "worker_infrastructure_unavailable",
    ) -> PreparedJobDisposition:
        return PreparedJobDisposition.infrastructure_defer(
            token=self._token(),
            delivery_id=delivery_id,
            reason_code=reason_code,
        )

    async def _prepare_no_attempt(
        self,
        bundle: DeliveryBundle,
        jobs_job_id: str,
        reason: DeliveryReasonCode,
        now: datetime,
    ) -> PreparedJobDisposition:
        async with self._repository.transaction() as tx:
            pending = await tx.prepare_no_attempt_terminal(
                bundle.delivery.delivery.id,
                jobs_job_id,
                reason,
                self._token(),
                now,
                expected_delivery_config_version=(
                    bundle.delivery.delivery.delivery_config_version
                ),
                expected_secret_version=bundle.delivery.delivery.secret_version,
            )
        if pending is None:
            return self._infrastructure_defer(
                bundle.delivery.delivery.id,
                reason_code="delivery_state_conflict",
            )
        return _prepared_from_pending(pending)

    def _decrypt_material(
        self,
        bundle: DeliveryBundle,
    ) -> tuple[Any, str, bytes]:
        registration = bundle.registration.registration
        target_url = self._key_ring.decrypt_text(
            purpose="registration.target",
            identity={
                "registration_id": registration.id,
                "target_version": registration.target_version,
            },
            protected=bundle.registration.target,
        )
        target = validate_webhook_target(
            target_url,
            allow_http_dev=self._settings.allow_http_dev,
        )
        if target.hostname != registration.target_hostname:
            raise ValueError("target hostname does not match registration")
        secret = self._key_ring.decrypt_text(
            purpose="registration.secret",
            identity={
                "registration_id": registration.id,
                "secret_version": registration.secret_version,
            },
            protected=bundle.registration.secret,
        )
        if _SIGNING_SECRET.fullmatch(secret) is None:
            raise ValueError("signing secret is invalid")
        body = self._key_ring.decrypt_event_body(
            event_id=bundle.event.event.id,
            api_version=bundle.event.event.api_version,
            protected=bundle.event.body,
        )
        return target, secret, body

    async def __call__(
        self,
        job: dict[str, Any],
        context: _ExecutionContext,
    ) -> PreparedJobDisposition:
        try:
            delivery_id = canonical_admin_webhook_delivery_id(job.get("payload"))
        except ValueError:
            raise ValueError("canonical delivery identity is unavailable") from None
        try:
            jobs_record = JobsDeliveryQueue.acquired_delivery_job(job)
            snapshot = context.snapshot()
            lease_id = str(job["lease_id"])
            if (
                jobs_record.delivery_id != delivery_id
                or str(job["id"]) != jobs_record.jobs_job_id
                or getattr(snapshot, "lease_id", None) != lease_id
                or getattr(snapshot, "worker_id", None) != job.get("worker_id")
            ):
                raise JobsDeliveryConflictError()
            bundle = await self._repository.get_delivery_bundle(delivery_id)
        except Exception:  # noqa: BLE001 - pre-reservation isolation boundary.
            return self._infrastructure_defer(
                delivery_id,
                reason_code="jobs_identity_conflict",
            )
        if bundle is None or bundle.delivery.jobs_job_id != jobs_record.jobs_job_id:
            return self._infrastructure_defer(
                delivery_id,
                reason_code="delivery_state_conflict",
            )

        pending = _pending_from_bundle(bundle)
        marker = jobs_record.marker
        if pending is not None:
            disposition = _prepared_from_pending(pending)
            if marker is None or marker.origin in {
                PreparedDispositionOrigin.INFRASTRUCTURE,
                PreparedDispositionOrigin.RECOVERY,
            }:
                return disposition
            if not admin_webhook_disposition_marker_matches(marker, disposition):
                return self._infrastructure_defer(
                    delivery_id,
                    reason_code="jobs_identity_conflict",
                )
            if disposition.kind is not PreparedDispositionKind.RETRY:
                return self._infrastructure_defer(
                    delivery_id,
                    reason_code="jobs_identity_conflict",
                )
            async with self._repository.transaction() as tx:
                acknowledged = await tx.acknowledge_jobs_disposition(
                    delivery_id,
                    disposition.token,
                    jobs_record.status,
                )
            if not acknowledged:
                return self._infrastructure_defer(
                    delivery_id,
                    reason_code="delivery_state_conflict",
                )
            bundle = await self._repository.get_delivery_bundle(delivery_id)
            if bundle is None:
                return self._infrastructure_defer(delivery_id)

        delivery = bundle.delivery.delivery
        registration = bundle.registration.registration
        now = _aware_utc(self._clock(), field="clock value")
        if delivery.state is DeliveryState.PROCESSING:
            attempt = await self._repository.get_current_delivery_attempt(delivery_id)
            if (
                attempt is None
                or attempt.state is not AttemptState.PROCESSING
                or attempt.request_timeout_seconds is None
            ):
                return self._infrastructure_defer(
                    delivery_id,
                    reason_code="delivery_state_conflict",
                )
            stale_at = attempt.started_at + timedelta(
                seconds=(
                    attempt.request_timeout_seconds
                    + self._settings.delivery_stale_attempt_margin_seconds
                )
            )
            if now < stale_at:
                return PreparedJobDisposition.recovery_defer_until(
                    token=self._token(),
                    delivery_id=delivery_id,
                    not_before_at=stale_at,
                    reason_code="attempt_not_stale",
                )
            async with self._repository.transaction() as tx:
                recovered = await tx.recover_stale_attempt_and_prepare_disposition(
                    delivery_id,
                    attempt.id,
                    jobs_record.jobs_job_id,
                    now,
                    self._token(),
                )
            if recovered is None:
                return PreparedJobDisposition.recovery_defer_until(
                    token=self._token(),
                    delivery_id=delivery_id,
                    not_before_at=stale_at,
                    reason_code="attempt_recovery_conflict",
                )
            return _prepared_from_pending(recovered)
        lifecycle_reason = registration_work_lifecycle_reason(
            delivery,
            registration,
        )
        if lifecycle_reason is not None:
            return await self._prepare_no_attempt(
                bundle,
                jobs_record.jobs_job_id,
                lifecycle_reason,
                now,
            )
        if delivery.expires_at <= now:
            return await self._prepare_no_attempt(
                bundle,
                jobs_record.jobs_job_id,
                DeliveryReasonCode.DELIVERY_EXPIRED,
                now,
            )
        if delivery.attempt_count >= self._settings.delivery_max_attempts:
            return await self._prepare_no_attempt(
                bundle,
                jobs_record.jobs_job_id,
                DeliveryReasonCode.ATTEMPT_BUDGET_EXHAUSTED,
                now,
            )
        if delivery.state not in {DeliveryState.QUEUED, DeliveryState.RETRY_WAIT}:
            return self._infrastructure_defer(
                delivery_id,
                reason_code="delivery_state_conflict",
            )

        try:
            target, signing_secret, body = self._decrypt_material(bundle)
        except Exception:  # noqa: BLE001 - protected material isolation boundary.
            return self._infrastructure_defer(delivery_id)
        required_seconds = (
            registration.timeout_seconds
            + self._settings.delivery_commit_margin_seconds
        )
        try:
            horizon_ok = await context.ensure_lease_horizon(required_seconds)
        except Exception:  # noqa: BLE001 - context is an external boundary.
            horizon_ok = False
        if not horizon_ok:
            return self._infrastructure_defer(
                delivery_id,
                reason_code="lease_horizon_unavailable",
            )
        reservation_now = _aware_utc(self._clock(), field="clock value")
        required_horizon = reservation_now + timedelta(seconds=required_seconds)
        attempt_id = self._attempt_id_factory()
        async with self._repository.transaction() as tx:
            reservation = await tx.reserve_jobs_attempt(
                delivery_id,
                jobs_record.jobs_job_id,
                lease_id,
                attempt_id,
                registration.timeout_seconds,
                reservation_now,
                required_horizon,
                expected_delivery_config_version=delivery.delivery_config_version,
                expected_secret_version=delivery.secret_version,
                disposition_token=self._token(),
            )
            if reservation is not None and reservation.reserved:
                self._crash(WorkerCrashPoint.BEFORE_RESERVATION_COMMIT)
        if reservation is None:
            return self._infrastructure_defer(
                delivery_id,
                reason_code="delivery_state_conflict",
            )
        if not reservation.reserved:
            if reservation.pending_disposition is None:
                return self._infrastructure_defer(
                    delivery_id,
                    reason_code="delivery_state_conflict",
                )
            return _prepared_from_pending(reservation.pending_disposition)
        if reservation.attempt is None:
            return self._infrastructure_defer(delivery_id)
        self._crash(WorkerCrashPoint.AFTER_RESERVATION_COMMIT_BEFORE_IO)

        request = AttemptExecutionRequest(
            target=target,
            body=body,
            signing_secret=signing_secret,
            timeout_seconds=registration.timeout_seconds,
            event_type=bundle.event.event.event_type,
            event_id=bundle.event.event.id,
            delivery_id=delivery_id,
            attempt_number=reservation.attempt.attempt_number,
            secret_version=registration.secret_version,
            kind=delivery.kind,
        )
        result = await self._executor.execute(request)
        if not isinstance(result, AttemptExecutionResult):
            raise TypeError("attempt executor returned an invalid result")
        self._crash(WorkerCrashPoint.AFTER_RECEIVER_RESULT_BEFORE_OUTCOME_COMMIT)
        finished_at = _aware_utc(self._clock(), field="clock value")
        reason = (
            DeliveryReasonCode(result.reason_code.value)
            if result.reason_code is not None
            else None
        )
        if result.outcome is AttemptOutcome.SUCCESS:
            completion = AttemptCompletion(
                attempt_state=AttemptState.SUCCEEDED,
                delivery_state=DeliveryState.SUCCEEDED,
                disposition=JobsDispositionKind.COMPLETE,
                status_code=result.status_code,
                latency_ms=result.latency_ms,
                reason_code=None,
                requested_retry_delay_seconds=None,
                finished_at=finished_at,
                completed_after_config_change=False,
            )
            not_before_at = None
        elif result.outcome is AttemptOutcome.RETRYABLE:
            if result.retry_delay_seconds is None or reason is None:
                raise ValueError("retryable attempt result is incomplete")
            completion = AttemptCompletion(
                attempt_state=AttemptState.RETRYABLE,
                delivery_state=DeliveryState.RETRY_WAIT,
                disposition=JobsDispositionKind.RETRY,
                status_code=result.status_code,
                latency_ms=result.latency_ms,
                reason_code=reason,
                requested_retry_delay_seconds=result.retry_delay_seconds,
                finished_at=finished_at,
                completed_after_config_change=False,
            )
            not_before_at = finished_at + timedelta(
                seconds=result.retry_delay_seconds
            )
        else:
            if reason is None:
                raise ValueError("failed attempt result is incomplete")
            completion = AttemptCompletion(
                attempt_state=AttemptState.FAILED,
                delivery_state=DeliveryState.DEAD,
                disposition=JobsDispositionKind.FAIL,
                status_code=result.status_code,
                latency_ms=result.latency_ms,
                reason_code=reason,
                requested_retry_delay_seconds=None,
                finished_at=finished_at,
                completed_after_config_change=False,
            )
            not_before_at = None
        disposition_token = self._token()
        async with self._repository.transaction() as tx:
            pending = await tx.finish_attempt_and_prepare_disposition(
                lease_id,
                completion,
                disposition_token,
                not_before_at,
                delivery_id=delivery_id,
                attempt_id=reservation.attempt.id,
                jobs_job_id=jobs_record.jobs_job_id,
            )
        if pending is None:
            stale_at = reservation.attempt.started_at + timedelta(
                seconds=(
                    registration.timeout_seconds
                    + self._settings.delivery_stale_attempt_margin_seconds
                )
            )
            return PreparedJobDisposition.recovery_defer_until(
                token=self._token(),
                delivery_id=delivery_id,
                not_before_at=stale_at,
                reason_code="attempt_result_conflict",
            )
        self._crash(WorkerCrashPoint.AFTER_OUTCOME_COMMIT_BEFORE_JOBS_APPLY)
        return _prepared_from_pending(pending)

    async def on_disposition_applied(
        self,
        job: dict[str, Any],
        disposition: PreparedJobDisposition,
        result: PreparedDispositionResult,
    ) -> None:
        """Acknowledge only one exact applied AuthNZ disposition token."""

        if disposition.origin is not PreparedDispositionOrigin.AUTHNZ:
            return
        if (
            result.outcome is not OperationOutcome.APPLIED
            or result.state is None
            or result.metadata is None
        ):
            return
        try:
            delivery_id = canonical_admin_webhook_delivery_id(job.get("payload"))
        except ValueError:
            return
        if delivery_id != disposition.delivery_id:
            return
        expected = {
            "token": disposition.token,
            "kind": disposition.kind.value,
            "origin": disposition.origin.value,
            "delivery_id": disposition.delivery_id,
        }
        if disposition.attempt_id is not None:
            expected["attempt_id"] = disposition.attempt_id
        if any(result.metadata.get(key) != value for key, value in expected.items()):
            return
        self._crash(WorkerCrashPoint.AFTER_JOBS_APPLY_BEFORE_AUTHNZ_ACK)
        async with self._repository.transaction() as tx:
            acknowledged = await tx.acknowledge_jobs_disposition(
                delivery_id,
                disposition.token,
                result.state,
            )
        if acknowledged:
            self._crash(WorkerCrashPoint.AFTER_AUTHNZ_ACK_BEFORE_RETURN)

    def handler_error_disposition(
        self,
        job: dict[str, Any],
        _error_class: type[BaseException],
    ) -> PreparedJobDisposition:
        """Return one secret-free timestamp-free handler isolation defer."""

        delivery_id = canonical_admin_webhook_delivery_id(job.get("payload"))
        return self._infrastructure_defer(delivery_id)


__all__ = ["AdminWebhookPreparedHandler", "WorkerCrashPoint"]
