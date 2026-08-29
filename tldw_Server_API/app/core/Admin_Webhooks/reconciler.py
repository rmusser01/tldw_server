"""Recoverable AuthNZ-to-Jobs admission for canonical webhook deliveries."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol

from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
    EnqueueClaim,
    PendingJobsDisposition,
    StoredWebhookDelivery,
)
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    ADMIN_WEBHOOK_DELIVERY_DOMAIN,
    ADMIN_WEBHOOK_DELIVERY_JOB_TYPE,
    ADMIN_WEBHOOK_DELIVERY_MAX_RETRIES,
    ADMIN_WEBHOOK_DELIVERY_PRIORITY,
    ADMIN_WEBHOOK_DELIVERY_QUARANTINE_THRESHOLD,
    ADMIN_WEBHOOK_DELIVERY_QUEUE,
    AdminWebhookDispositionMarker,
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
    admin_webhook_disposition_marker_matches,
    canonical_admin_webhook_delivery_id,
    canonical_admin_webhook_idempotency_key,
    canonical_admin_webhook_row_matches,
    project_admin_webhook_disposition_marker,
)

from .domain import DeliveryReasonCode, DeliveryState, JobsDispositionKind

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
    AFTER_ORPHAN_PREPARE = "after_orphan_prepare"
    AFTER_JOBS_CANCEL = "after_jobs_cancel"


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
    marker: AdminWebhookDispositionMarker | None = None

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
    def _operation_row(row: dict[str, Any]) -> dict[str, Any]:
        """Decode the SQLite operation payload without another Jobs read."""

        normalized = dict(row)
        raw_payload = normalized.get("payload")
        if isinstance(raw_payload, str):
            try:
                normalized["payload"] = json.loads(raw_payload)
            except (TypeError, ValueError):
                raise JobsDeliveryConflictError() from None
        return normalized

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
        marker = project_admin_webhook_disposition_marker(
            row,
            expected_payload=expected_payload,
            archived=archived,
        )
        if row.get("result") is not None and marker is None:
            raise JobsDeliveryConflictError()
        return JobsDeliveryRecord(
            str(raw_id),
            delivery_id,
            status,
            archived,
            marker,
        )

    @staticmethod
    def acquired_delivery_job(row: dict[str, Any]) -> JobsDeliveryRecord:
        """Validate and project one acquired canonical Jobs row."""

        payload = row.get("payload")
        try:
            delivery_id = canonical_admin_webhook_delivery_id(payload)
        except ValueError:
            raise JobsDeliveryConflictError() from None
        record = JobsDeliveryQueue._record(
            row,
            expected_payload={"delivery_id": delivery_id},
            archived=False,
        )
        if record.status != "processing":
            raise JobsDeliveryConflictError()
        return record

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
                    self._operation_row(result.row),
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


def _prepared_from_pending(
    pending: PendingJobsDisposition,
) -> PreparedJobDisposition:
    reason_code = (
        pending.reason_code.value if pending.reason_code is not None else None
    )
    if pending.kind is not JobsDispositionKind.COMPLETE and reason_code is None:
        raise ValueError("pending disposition reason is unavailable")
    if pending.kind is JobsDispositionKind.COMPLETE:
        if pending.attempt_id is None:
            raise ValueError("complete disposition attempt is unavailable")
        return PreparedJobDisposition.complete(
            token=pending.token,
            delivery_id=pending.delivery_id,
            attempt_id=pending.attempt_id,
        )
    if pending.kind is JobsDispositionKind.RETRY:
        if (
            pending.attempt_id is None
            or pending.delay_seconds is None
            or pending.not_before_at is None
        ):
            raise ValueError("retry disposition coordinates are unavailable")
        return PreparedJobDisposition.retry(
            token=pending.token,
            delivery_id=pending.delivery_id,
            attempt_id=pending.attempt_id,
            delay_seconds=pending.delay_seconds,
            not_before_at=pending.not_before_at,
            reason_code=reason_code,
        )
    if pending.kind is JobsDispositionKind.FAIL:
        return PreparedJobDisposition.fail(
            token=pending.token,
            delivery_id=pending.delivery_id,
            attempt_id=pending.attempt_id,
            reason_code=reason_code,
        )
    if pending.kind is JobsDispositionKind.CANCEL:
        return PreparedJobDisposition.cancel(
            token=pending.token,
            delivery_id=pending.delivery_id,
            attempt_id=pending.attempt_id,
            reason_code=reason_code,
        )
    raise ValueError("AuthNZ cannot persist a defer disposition")


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
        after_claim_commit_hook: Callable[[], Awaitable[None]] | None = None,
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
        if after_claim_commit_hook is not None and not callable(
            after_claim_commit_hook
        ):
            raise TypeError("after-claim commit hook is invalid")
        self._repository = repository
        self._queue = queue
        self._token_factory = token_factory
        self._clock = clock
        self._claim_ttl = timedelta(seconds=claim_ttl_seconds)
        self._failure_observer = failure_observer
        self._crash_hook = crash_hook
        self._after_claim_commit_hook = after_claim_commit_hook

    def _crash(self, point: EnqueueCrashPoint) -> None:
        if self._crash_hook is not None:
            self._crash_hook(point)

    def _observe(self, failure: EnqueueFailureKind) -> None:
        try:
            self._failure_observer(failure)
        except Exception:  # noqa: BLE001 - observation cannot affect state repair.
            return

    async def _apply_terminal_cancel(
        self,
        delivery: StoredWebhookDelivery,
        claim_token: str,
    ) -> None:
        """Apply and acknowledge one already-persisted terminal orphan cancel."""

        jobs_job_id = delivery.jobs_job_id
        disposition_token = delivery.pending_jobs_disposition_token
        reason_code = delivery.delivery.reason_code
        if (
            jobs_job_id is None
            or disposition_token is None
            or reason_code is None
            or delivery.pending_jobs_disposition is None
            or delivery.jobs_disposition_applied
        ):
            return
        self._crash(EnqueueCrashPoint.AFTER_ORPHAN_PREPARE)
        try:
            result = self._queue.apply_queued_cancel(
                jobs_job_id,
                delivery.delivery.id,
                disposition_token,
                reason_code,
            )
        except JobsDeliveryConflictError:
            self._observe(EnqueueFailureKind.IDENTITY_CONFLICT)
            return
        except Exception:  # noqa: BLE001 - backend failures retain the recovery coordinate.
            self._observe(EnqueueFailureKind.BACKEND_UNAVAILABLE)
            return
        if (
            result.outcome is not OperationOutcome.APPLIED
            or result.state != "cancelled"
        ):
            return
        self._crash(EnqueueCrashPoint.AFTER_JOBS_CANCEL)
        async with self._repository.transaction() as tx:
            await tx.acknowledge_terminal_enqueue_cancel(
                delivery.delivery.id,
                claim_token,
                disposition_token,
            )

    async def _process_claim(self, claim: EnqueueClaim) -> bool:
        """Reread the exact claim under lock before any Jobs-side mutation."""

        delivery_id = claim.delivery.delivery.id
        terminal_cancel: StoredWebhookDelivery | None = None
        failure: EnqueueFailureKind | None = None
        continue_batch = True
        queued = False
        async with self._repository.transaction() as tx:
            current = await tx.lock_owned_enqueue_claim(
                delivery_id,
                claim.claim_token,
            )
            if current is None:
                return True
            now = _aware_utc(self._clock(), field="clock value")
            if (
                current.enqueue_claim_expires_at is None
                or current.enqueue_claim_expires_at <= now
            ):
                return True
            terminal = (
                current.delivery.state in DeliveryState.terminal_states()
                or current.delivery.expires_at <= now
            )
            if terminal:
                try:
                    record = self._queue.find_delivery_job_by_identity(delivery_id)
                except JobsDeliveryConflictError:
                    failure = EnqueueFailureKind.IDENTITY_CONFLICT
                except Exception:  # noqa: BLE001 - ambiguous lookup retains the claim.
                    failure = EnqueueFailureKind.BACKEND_UNAVAILABLE
                else:
                    if record is None:
                        await tx.retire_terminal_enqueue_claim(
                            delivery_id,
                            claim.claim_token,
                            now,
                        )
                    elif record.delivery_id != delivery_id:
                        failure = EnqueueFailureKind.IDENTITY_CONFLICT
                    else:
                        disposition_token = (
                            current.pending_jobs_disposition_token
                            if (
                                current.pending_jobs_disposition is not None
                                and not current.jobs_disposition_applied
                                and current.pending_jobs_disposition_token is not None
                            )
                            else self._token_factory()
                        )
                        terminal_cancel = await tx.retire_terminal_enqueue_claim(
                            delivery_id,
                            claim.claim_token,
                            now,
                            jobs_job_id=record.jobs_job_id,
                            disposition_token=disposition_token,
                        )
                continue_batch = failure is None
            else:
                try:
                    admission = self._queue.admit_delivery_job(
                        delivery_id,
                        current.delivery.expires_at,
                    )
                except JobsDeliveryConflictError:
                    await tx.fail_enqueue_claim(
                        delivery_id,
                        claim.claim_token,
                        now,
                    )
                    failure = EnqueueFailureKind.IDENTITY_CONFLICT
                except Exception:  # noqa: BLE001 - ambiguous admission retains the claim.
                    failure = EnqueueFailureKind.BACKEND_UNAVAILABLE
                    continue_batch = False
                else:
                    if admission.outcome is OperationOutcome.ADMISSION_REJECTED:
                        await tx.release_enqueue_claim(
                            delivery_id,
                            claim.claim_token,
                            now,
                        )
                        failure = EnqueueFailureKind.ADMISSION_REJECTED
                        continue_batch = False
                    elif admission.outcome is OperationOutcome.BACKEND_ERROR:
                        failure = EnqueueFailureKind.BACKEND_UNAVAILABLE
                        continue_batch = False
                    elif admission.outcome in {
                        OperationOutcome.BACKEND_CONFLICT,
                        OperationOutcome.BACKEND_SCHEMA_ERROR,
                    } or (
                        admission.record is None
                        or admission.record.delivery_id != delivery_id
                    ):
                        await tx.fail_enqueue_claim(
                            delivery_id,
                            claim.claim_token,
                            now,
                        )
                        failure = EnqueueFailureKind.IDENTITY_CONFLICT
                    else:
                        self._crash(EnqueueCrashPoint.BEFORE_AUTHNZ_ATTACH)
                        attach_now = _aware_utc(self._clock(), field="clock value")
                        attached = await tx.attach_jobs_job(
                            delivery_id,
                            claim.claim_token,
                            admission.record.jobs_job_id,
                            attach_now,
                        )
                        if attached is not None:
                            queued = True
                            self._crash(EnqueueCrashPoint.BEFORE_ATTACH_COMMIT)
                        elif current.delivery.expires_at <= attach_now:
                            terminal_cancel = await tx.retire_terminal_enqueue_claim(
                                delivery_id,
                                claim.claim_token,
                                attach_now,
                                jobs_job_id=admission.record.jobs_job_id,
                                disposition_token=self._token_factory(),
                            )
        if failure is not None:
            self._observe(failure)
            return continue_batch
        if terminal_cancel is not None:
            await self._apply_terminal_cancel(terminal_cancel, claim.claim_token)
            return True
        if queued:
            self._crash(EnqueueCrashPoint.AFTER_QUEUED_COMMIT)
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
            if self._after_claim_commit_hook is not None:
                await self._after_claim_commit_hook()
            if not await self._process_claim(claim):
                break
        await asyncio.sleep(0)
        return processed

    async def reconcile_pending_dispositions_once(
        self,
        *,
        limit: int = _MAX_ENQUEUE_BATCH,
    ) -> int:
        """Repair one bounded ordered page using lookup-only Jobs evidence."""

        pending_page = await self._repository.list_pending_jobs_dispositions(
            limit=limit
        )
        repaired = 0
        for pending in pending_page:
            try:
                disposition = _prepared_from_pending(pending)
                record = self._queue.get_delivery_job(pending.jobs_job_id)
                if record is None:
                    record = self._queue.find_delivery_job_by_identity(
                        pending.delivery_id
                    )
            except JobsDeliveryConflictError:
                self._observe(EnqueueFailureKind.IDENTITY_CONFLICT)
                continue
            except Exception:  # noqa: BLE001 - ambiguous reads retain recovery.
                self._observe(EnqueueFailureKind.BACKEND_UNAVAILABLE)
                continue
            if record is None or record.delivery_id != pending.delivery_id:
                continue
            if record.marker is not None and admin_webhook_disposition_marker_matches(
                record.marker,
                disposition,
            ):
                async with self._repository.transaction() as tx:
                    acknowledged = await tx.acknowledge_jobs_disposition(
                        pending.delivery_id,
                        pending.token,
                        record.status,
                    )
                repaired += int(acknowledged)
                continue
            if (
                pending.kind is not JobsDispositionKind.CANCEL
                or record.marker is not None
                or record.status != "queued"
            ):
                continue
            try:
                result = self._queue.apply_queued_cancel(
                    pending.jobs_job_id,
                    pending.delivery_id,
                    pending.token,
                    pending.reason_code,
                )
            except JobsDeliveryConflictError:
                self._observe(EnqueueFailureKind.IDENTITY_CONFLICT)
                continue
            except Exception:  # noqa: BLE001 - ambiguous apply retains recovery.
                self._observe(EnqueueFailureKind.BACKEND_UNAVAILABLE)
                continue
            if result.outcome is not OperationOutcome.APPLIED:
                continue
            async with self._repository.transaction() as tx:
                acknowledged = await tx.acknowledge_jobs_disposition(
                    pending.delivery_id,
                    pending.token,
                    result.state or "",
                )
            repaired += int(acknowledged)
        await asyncio.sleep(0)
        return repaired
