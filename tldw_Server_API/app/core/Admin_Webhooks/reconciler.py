"""Recoverable AuthNZ-to-Jobs admission for canonical webhook deliveries."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol

from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
    EnqueueClaim,
    StoredWebhookDelivery,
)
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    ADMIN_WEBHOOK_DELIVERY_DOMAIN,
    ADMIN_WEBHOOK_DELIVERY_JOB_TYPE,
    ADMIN_WEBHOOK_DELIVERY_MAX_RETRIES,
    ADMIN_WEBHOOK_DELIVERY_PRIORITY,
    ADMIN_WEBHOOK_DELIVERY_QUARANTINE_THRESHOLD,
    ADMIN_WEBHOOK_DELIVERY_QUEUE,
    AdmissionRejectionReason,
    AdmissionResult,
    ApplyPreparedDispositionCommand,
    ExpiredLeasePolicy,
    FindJobByIdentityCommand,
    JobIdentityLookupResult,
    JobIdentityLookupState,
    NoTransitionReason,
    OperationOutcome,
    PreparedDispositionResult,
    PreparedJobDisposition,
    canonical_admin_webhook_delivery_id,
    canonical_admin_webhook_idempotency_key,
    canonical_admin_webhook_row_matches,
)

from .domain import DeliveryReasonCode, DeliveryState

_MAX_ENQUEUE_BATCH = 100


class EnqueueFailureKind(str, Enum):
    """Closed low-cardinality enqueue failures for a future metrics adapter."""

    ADMISSION_REJECTED = "admission_rejected"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    IDENTITY_CONFLICT = "identity_conflict"


class EnqueueCrashPoint(str, Enum):
    """Deterministic handshake boundaries used by crash recovery tests."""

    BEFORE_CLAIM_COMMIT = "before_claim_commit"
    AFTER_CLAIM_COMMIT = "after_claim_commit"
    BEFORE_AUTHNZ_ATTACH = "before_authnz_attach"
    BEFORE_ATTACH_COMMIT = "before_attach_commit"
    AFTER_QUEUED_COMMIT = "after_queued_commit"


class JobsDeliveryConflictError(RuntimeError):
    """A Jobs read cannot prove one exact canonical delivery identity."""

    def __init__(self) -> None:
        super().__init__("canonical Jobs delivery identity conflict")


@dataclass(frozen=True)
class JobsDeliveryRecord:
    """Bounded Jobs facts needed by enqueue recovery."""

    jobs_job_id: str
    delivery_id: str
    status: str
    archived: bool

    def __post_init__(self) -> None:
        try:
            parsed_id = int(self.jobs_job_id)
        except (TypeError, ValueError):
            raise ValueError("Jobs delivery record ID is invalid") from None
        if parsed_id <= 0 or str(parsed_id) != self.jobs_job_id:
            raise ValueError("Jobs delivery record ID is invalid")
        canonical_admin_webhook_delivery_id({"delivery_id": self.delivery_id})
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("Jobs delivery record status is invalid")
        if not isinstance(self.archived, bool):
            raise ValueError("Jobs delivery record archive state is invalid")


@dataclass(frozen=True)
class JobsDeliveryAdmission:
    """Typed canonical admission result without backend exception text."""

    outcome: OperationOutcome
    record: JobsDeliveryRecord | None = None
    no_transition_reason: NoTransitionReason | None = None
    admission_rejection_reason: AdmissionRejectionReason | None = None

    def __post_init__(self) -> None:
        successful = (
            self.outcome is OperationOutcome.APPLIED
            and self.no_transition_reason is None
        ) or (
            self.outcome is OperationOutcome.NO_TRANSITION
            and self.no_transition_reason is NoTransitionReason.IDEMPOTENT_EXISTING
        )
        rejected = self.outcome is OperationOutcome.ADMISSION_REJECTED
        valid = (
            successful
            and self.record is not None
            and self.admission_rejection_reason is None
        ) or (
            rejected
            and self.record is None
            and self.no_transition_reason is None
            and self.admission_rejection_reason is not None
        ) or (
            self.outcome
            in {
                OperationOutcome.BACKEND_CONFLICT,
                OperationOutcome.BACKEND_SCHEMA_ERROR,
                OperationOutcome.BACKEND_ERROR,
            }
            and self.record is None
            and self.no_transition_reason is None
            and self.admission_rejection_reason is None
        )
        if not valid:
            raise ValueError("Jobs delivery admission shape is invalid")


class _JobManager(Protocol):
    def admit_job(self, **kwargs: Any) -> AdmissionResult: ...

    def find_job_by_identity(
        self,
        command: FindJobByIdentityCommand,
    ) -> JobIdentityLookupResult: ...

    def get_job(self, job_id: int) -> dict[str, Any] | None: ...

    def apply_prepared_disposition(
        self,
        command: ApplyPreparedDispositionCommand,
    ) -> PreparedDispositionResult: ...


class _DeliveryQueue(Protocol):
    def admit_delivery_job(
        self,
        delivery_id: str,
        expires_at: datetime,
    ) -> JobsDeliveryAdmission: ...

    def find_delivery_job_by_identity(
        self,
        delivery_id: str,
    ) -> JobsDeliveryRecord | None: ...

    def get_delivery_job(self, jobs_job_id: str) -> JobsDeliveryRecord | None: ...

    def apply_queued_cancel(
        self,
        jobs_job_id: str,
        delivery_id: str,
        disposition_token: str,
        reason_code: DeliveryReasonCode,
    ) -> PreparedDispositionResult: ...


def _aware_utc(value: datetime, *, field: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _positive_job_id(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise JobsDeliveryConflictError() from None
    if parsed <= 0 or str(parsed) != value:
        raise JobsDeliveryConflictError()
    return parsed


class JobsDeliveryQueue:
    """Narrow adapter over supported typed JobManager operations."""

    def __init__(self, manager: _JobManager) -> None:
        self._manager = manager

    @staticmethod
    def _payload(delivery_id: str) -> dict[str, str]:
        payload = {"delivery_id": delivery_id}
        canonical_admin_webhook_delivery_id(payload)
        return payload

    @staticmethod
    def _record(
        row: dict[str, Any],
        *,
        expected_payload: dict[str, Any],
        archived: bool,
    ) -> JobsDeliveryRecord:
        if not canonical_admin_webhook_row_matches(
            row,
            expected_payload=expected_payload,
            archived=archived,
        ):
            raise JobsDeliveryConflictError()
        raw_id = row.get("id")
        if isinstance(raw_id, bool) or not isinstance(raw_id, int) or raw_id <= 0:
            raise JobsDeliveryConflictError()
        status = row.get("status")
        if not isinstance(status, str) or not status:
            raise JobsDeliveryConflictError()
        try:
            delivery_id = canonical_admin_webhook_delivery_id(expected_payload)
        except ValueError:
            raise JobsDeliveryConflictError() from None
        return JobsDeliveryRecord(str(raw_id), delivery_id, status, archived)

    def admit_delivery_job(
        self,
        delivery_id: str,
        expires_at: datetime,
    ) -> JobsDeliveryAdmission:
        _aware_utc(expires_at, field="delivery expiry")
        payload = self._payload(delivery_id)
        result = self._manager.admit_job(
            domain=ADMIN_WEBHOOK_DELIVERY_DOMAIN,
            queue=ADMIN_WEBHOOK_DELIVERY_QUEUE,
            job_type=ADMIN_WEBHOOK_DELIVERY_JOB_TYPE,
            payload=payload,
            owner_user_id=None,
            project_id=None,
            batch_group=None,
            priority=ADMIN_WEBHOOK_DELIVERY_PRIORITY,
            max_retries=ADMIN_WEBHOOK_DELIVERY_MAX_RETRIES,
            available_at=None,
            idempotency_key=canonical_admin_webhook_idempotency_key(delivery_id),
            request_id=None,
            trace_id=None,
            expired_lease_policy=ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT,
            quarantine_threshold=ADMIN_WEBHOOK_DELIVERY_QUARANTINE_THRESHOLD,
        )
        successful = result.outcome is OperationOutcome.APPLIED or (
            result.outcome is OperationOutcome.NO_TRANSITION
            and result.no_transition_reason is NoTransitionReason.IDEMPOTENT_EXISTING
        )
        if successful:
            if result.row is None:
                return JobsDeliveryAdmission(OperationOutcome.BACKEND_CONFLICT)
            try:
                record = self._record(
                    result.row,
                    expected_payload=payload,
                    archived=False,
                )
            except JobsDeliveryConflictError:
                return JobsDeliveryAdmission(OperationOutcome.BACKEND_CONFLICT)
            return JobsDeliveryAdmission(
                outcome=result.outcome,
                record=record,
                no_transition_reason=result.no_transition_reason,
            )
        if result.outcome is OperationOutcome.ADMISSION_REJECTED:
            if result.admission_rejection_reason is None:
                return JobsDeliveryAdmission(OperationOutcome.BACKEND_CONFLICT)
            return JobsDeliveryAdmission(
                outcome=result.outcome,
                admission_rejection_reason=result.admission_rejection_reason,
            )
        if result.outcome in {
            OperationOutcome.BACKEND_CONFLICT,
            OperationOutcome.BACKEND_SCHEMA_ERROR,
            OperationOutcome.BACKEND_ERROR,
        }:
            return JobsDeliveryAdmission(result.outcome)
        return JobsDeliveryAdmission(OperationOutcome.BACKEND_CONFLICT)

    def find_delivery_job_by_identity(
        self,
        delivery_id: str,
    ) -> JobsDeliveryRecord | None:
        payload = self._payload(delivery_id)
        result = self._manager.find_job_by_identity(
            FindJobByIdentityCommand(
                domain=ADMIN_WEBHOOK_DELIVERY_DOMAIN,
                queue=ADMIN_WEBHOOK_DELIVERY_QUEUE,
                job_type=ADMIN_WEBHOOK_DELIVERY_JOB_TYPE,
                idempotency_key=canonical_admin_webhook_idempotency_key(delivery_id),
                expected_payload=payload,
            )
        )
        if result.state is JobIdentityLookupState.MISSING:
            return None
        if result.state is JobIdentityLookupState.CONFLICT or result.row is None:
            raise JobsDeliveryConflictError()
        archived = result.state is JobIdentityLookupState.ARCHIVED
        if not archived and result.state is not JobIdentityLookupState.ACTIVE:
            raise JobsDeliveryConflictError()
        return self._record(
            result.row,
            expected_payload=payload,
            archived=archived,
        )

    def get_delivery_job(self, jobs_job_id: str) -> JobsDeliveryRecord | None:
        row = self._manager.get_job(_positive_job_id(jobs_job_id))
        if row is None:
            return None
        payload = row.get("payload")
        try:
            delivery_id = canonical_admin_webhook_delivery_id(payload)
        except ValueError:
            raise JobsDeliveryConflictError() from None
        return self._record(
            row,
            expected_payload={"delivery_id": delivery_id},
            archived=False,
        )

    def apply_queued_cancel(
        self,
        jobs_job_id: str,
        delivery_id: str,
        disposition_token: str,
        reason_code: DeliveryReasonCode,
    ) -> PreparedDispositionResult:
        if not isinstance(reason_code, DeliveryReasonCode):
            raise ValueError("delivery cancellation reason is invalid")
        payload = self._payload(delivery_id)
        return self._manager.apply_prepared_disposition(
            ApplyPreparedDispositionCommand(
                job_id=_positive_job_id(jobs_job_id),
                domain=ADMIN_WEBHOOK_DELIVERY_DOMAIN,
                queue=ADMIN_WEBHOOK_DELIVERY_QUEUE,
                job_type=ADMIN_WEBHOOK_DELIVERY_JOB_TYPE,
                expected_payload=payload,
                disposition=PreparedJobDisposition.cancel(
                    token=disposition_token,
                    delivery_id=delivery_id,
                    reason_code=reason_code.value,
                ),
            )
        )


class AdminWebhookReconciler:
    """Run one bounded, independently scheduled enqueue recovery iteration."""

    def __init__(
        self,
        *,
        repository: AdminWebhookRepository,
        queue: _DeliveryQueue,
        token_factory: Callable[[], str],
        clock: Callable[[], datetime],
        claim_ttl_seconds: int,
        failure_observer: Callable[[EnqueueFailureKind], None],
        crash_hook: Callable[[EnqueueCrashPoint], None] | None = None,
    ) -> None:
        if (
            isinstance(claim_ttl_seconds, bool)
            or not isinstance(claim_ttl_seconds, int)
            or claim_ttl_seconds <= 0
        ):
            raise ValueError("enqueue claim TTL must be a positive integer")
        for value, field in (
            (token_factory, "token factory"),
            (clock, "clock"),
            (failure_observer, "failure observer"),
        ):
            if not callable(value):
                raise TypeError(f"{field} is invalid")
        if crash_hook is not None and not callable(crash_hook):
            raise TypeError("crash hook is invalid")
        self._repository = repository
        self._queue = queue
        self._token_factory = token_factory
        self._clock = clock
        self._claim_ttl = timedelta(seconds=claim_ttl_seconds)
        self._failure_observer = failure_observer
        self._crash_hook = crash_hook

    def _crash(self, point: EnqueueCrashPoint) -> None:
        if self._crash_hook is not None:
            self._crash_hook(point)

    def _observe(self, failure: EnqueueFailureKind) -> None:
        try:
            self._failure_observer(failure)
        except Exception:  # noqa: BLE001 - observation cannot affect state repair.
            return

    async def _fresh_owned_delivery(
        self,
        delivery_id: str,
        claim_token: str,
    ) -> StoredWebhookDelivery | None:
        bundle = await self._repository.get_delivery_bundle(delivery_id)
        if bundle is None or bundle.delivery.enqueue_claim_token != claim_token:
            return None
        return bundle.delivery

    async def _release_transient(
        self,
        claim: EnqueueClaim,
        failure: EnqueueFailureKind,
    ) -> None:
        now = _aware_utc(self._clock(), field="clock value")
        async with self._repository.transaction() as tx:
            released = await tx.release_enqueue_claim(
                claim.delivery.delivery.id,
                claim.claim_token,
                now,
            )
        self._observe(failure)
        if released is None:
            fresh = await self._fresh_owned_delivery(
                claim.delivery.delivery.id,
                claim.claim_token,
            )
            if fresh is not None:
                await self._recover_terminal(fresh, claim.claim_token)

    async def _fail_permanent(self, claim: EnqueueClaim) -> None:
        now = _aware_utc(self._clock(), field="clock value")
        async with self._repository.transaction() as tx:
            failed = await tx.fail_enqueue_claim(
                claim.delivery.delivery.id,
                claim.claim_token,
                now,
            )
        self._observe(EnqueueFailureKind.IDENTITY_CONFLICT)
        if failed is None:
            fresh = await self._fresh_owned_delivery(
                claim.delivery.delivery.id,
                claim.claim_token,
            )
            if fresh is not None:
                await self._recover_terminal(fresh, claim.claim_token)

    async def _acknowledge_cancel(
        self,
        delivery_id: str,
        disposition_token: str,
        jobs_job_id: str,
    ) -> None:
        try:
            observed = self._queue.get_delivery_job(jobs_job_id)
        except JobsDeliveryConflictError:
            self._observe(EnqueueFailureKind.IDENTITY_CONFLICT)
            return
        except Exception:  # noqa: BLE001 - backend failures keep disposition pending.
            self._observe(EnqueueFailureKind.BACKEND_UNAVAILABLE)
            return
        if (
            observed is None
            or observed.jobs_job_id != jobs_job_id
            or observed.delivery_id != delivery_id
            or observed.status != "cancelled"
        ):
            return
        async with self._repository.transaction() as tx:
            await tx.acknowledge_jobs_disposition(
                delivery_id,
                disposition_token,
                "cancelled",
            )

    async def _recover_terminal(
        self,
        delivery: StoredWebhookDelivery,
        claim_token: str,
    ) -> None:
        now = _aware_utc(self._clock(), field="clock value")
        if delivery.enqueue_claim_token != claim_token or not (
            delivery.delivery.state in DeliveryState.terminal_states()
            or delivery.delivery.expires_at <= now
        ):
            return
        try:
            record = self._queue.find_delivery_job_by_identity(
                delivery.delivery.id
            )
        except JobsDeliveryConflictError:
            self._observe(EnqueueFailureKind.IDENTITY_CONFLICT)
            if delivery.delivery.state in DeliveryState.terminal_states():
                async with self._repository.transaction() as tx:
                    await tx.retire_terminal_enqueue_claim(
                        delivery.delivery.id,
                        claim_token,
                        now,
                    )
            else:
                async with self._repository.transaction() as tx:
                    await tx.fail_enqueue_claim(
                        delivery.delivery.id,
                        claim_token,
                        now,
                    )
            return
        except Exception:  # noqa: BLE001 - backend failures release only live claims.
            self._observe(EnqueueFailureKind.BACKEND_UNAVAILABLE)
            if delivery.delivery.state not in DeliveryState.terminal_states():
                async with self._repository.transaction() as tx:
                    await tx.release_enqueue_claim(
                        delivery.delivery.id,
                        claim_token,
                        now,
                    )
            return

        if record is None:
            async with self._repository.transaction() as tx:
                await tx.retire_terminal_enqueue_claim(
                    delivery.delivery.id,
                    claim_token,
                    now,
                )
            return

        if record.delivery_id != delivery.delivery.id:
            self._observe(EnqueueFailureKind.IDENTITY_CONFLICT)
            if delivery.delivery.state in DeliveryState.terminal_states():
                async with self._repository.transaction() as tx:
                    await tx.retire_terminal_enqueue_claim(
                        delivery.delivery.id,
                        claim_token,
                        now,
                    )
            else:
                async with self._repository.transaction() as tx:
                    await tx.fail_enqueue_claim(
                        delivery.delivery.id,
                        claim_token,
                        now,
                    )
            return

        disposition_token = self._token_factory()
        async with self._repository.transaction() as tx:
            retired = await tx.retire_terminal_enqueue_claim(
                delivery.delivery.id,
                claim_token,
                now,
                jobs_job_id=record.jobs_job_id,
                disposition_token=disposition_token,
            )
        if retired is None or retired.delivery.reason_code is None:
            return
        try:
            result = self._queue.apply_queued_cancel(
                record.jobs_job_id,
                delivery.delivery.id,
                disposition_token,
                retired.delivery.reason_code,
            )
        except JobsDeliveryConflictError:
            self._observe(EnqueueFailureKind.IDENTITY_CONFLICT)
            return
        except Exception:  # noqa: BLE001 - backend failures keep disposition pending.
            self._observe(EnqueueFailureKind.BACKEND_UNAVAILABLE)
            return
        if (
            result.outcome is OperationOutcome.APPLIED
            and result.state == "cancelled"
        ):
            await self._acknowledge_cancel(
                delivery.delivery.id,
                disposition_token,
                record.jobs_job_id,
            )

    async def _process_claim(self, claim: EnqueueClaim) -> bool:
        now = _aware_utc(self._clock(), field="clock value")
        if (
            claim.delivery.delivery.state in DeliveryState.terminal_states()
            or claim.delivery.delivery.expires_at <= now
        ):
            await self._recover_terminal(claim.delivery, claim.claim_token)
            return True

        try:
            admission = self._queue.admit_delivery_job(
                claim.delivery.delivery.id,
                claim.delivery.delivery.expires_at,
            )
        except JobsDeliveryConflictError:
            await self._fail_permanent(claim)
            return True
        except Exception:  # noqa: BLE001 - admission backend failures are transient.
            await self._release_transient(
                claim,
                EnqueueFailureKind.BACKEND_UNAVAILABLE,
            )
            return False

        if admission.outcome is OperationOutcome.ADMISSION_REJECTED:
            await self._release_transient(
                claim,
                EnqueueFailureKind.ADMISSION_REJECTED,
            )
            return False
        if admission.outcome is OperationOutcome.BACKEND_ERROR:
            await self._release_transient(
                claim,
                EnqueueFailureKind.BACKEND_UNAVAILABLE,
            )
            return False
        if admission.outcome in {
            OperationOutcome.BACKEND_CONFLICT,
            OperationOutcome.BACKEND_SCHEMA_ERROR,
        }:
            await self._fail_permanent(claim)
            return True
        if (
            admission.record is None
            or admission.record.delivery_id != claim.delivery.delivery.id
        ):
            await self._fail_permanent(claim)
            return True

        self._crash(EnqueueCrashPoint.BEFORE_AUTHNZ_ATTACH)
        attach_now = _aware_utc(self._clock(), field="clock value")
        async with self._repository.transaction() as tx:
            attached = await tx.attach_jobs_job(
                claim.delivery.delivery.id,
                claim.claim_token,
                admission.record.jobs_job_id,
                attach_now,
            )
            if attached is not None:
                self._crash(EnqueueCrashPoint.BEFORE_ATTACH_COMMIT)
        if attached is not None:
            self._crash(EnqueueCrashPoint.AFTER_QUEUED_COMMIT)
            return True

        fresh = await self._fresh_owned_delivery(
            claim.delivery.delivery.id,
            claim.claim_token,
        )
        if fresh is not None:
            await self._recover_terminal(fresh, claim.claim_token)
        return True

    async def reconcile_enqueue_once(self) -> int:
        """Process at most one hundred ordered enqueue candidates and yield."""

        processed = 0
        while processed < _MAX_ENQUEUE_BATCH:
            now = _aware_utc(self._clock(), field="clock value")
            claim_token = self._token_factory()
            async with self._repository.transaction() as tx:
                claim = await tx.claim_pending_delivery(
                    claim_token,
                    now + self._claim_ttl,
                    now,
                )
                if claim is not None:
                    self._crash(EnqueueCrashPoint.BEFORE_CLAIM_COMMIT)
            if claim is None:
                break
            processed += 1
            self._crash(EnqueueCrashPoint.AFTER_CLAIM_COMMIT)
            if not await self._process_claim(claim):
                break
        await asyncio.sleep(0)
        return processed
