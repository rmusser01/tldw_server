"""Bounded provider-independent reconciliation for suggestion runs."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from .suggestion_jobs import (
    PublicationReceiptError,
    SuggestionCancellationCoordinator,
    validate_publication_receipt,
)
from .suggestion_observability import (
    SuggestionErrorCode,
    SuggestionEventName,
    record_event,
    record_run_error,
)

_FAILURE_GUIDANCE = {
    "notes_graph_capabilities_changed_before_provider": "retry_generation",
    "notes_graph_fingerprint_stale": "refresh_note",
    "notes_graph_fts_not_ready": "contact_administrator",
    "notes_graph_provider_retry_policy_unsupported": "configure_provider",
    "notes_graph_provider_unavailable": "retry_generation",
    "notes_graph_source_too_large": "refresh_note",
    "notes_graph_suggestion_no_valid_items": "retry_generation",
    "notes_graph_suggestion_suppression_limit": "refresh_note",
}
_TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled", "quarantined"}
_TEMPORARY_AUTHORITY_ERRORS = (ConnectionError, OSError, RuntimeError, TimeoutError)


def _utc(value: datetime | str) -> datetime:
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        parsed = value
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def missing_job_reference_at(
    run: Any,
    *,
    cancellation_created_at: datetime | str | None = None,
) -> datetime:
    """Return the best persisted timestamp for the current missing-Job grace."""

    state = run.state.value
    if state == "cancelling" and cancellation_created_at is not None:
        return _utc(cancellation_created_at)
    if state in {"running", "cancelling"} and getattr(run, "started_at", None):
        return _utc(run.started_at)
    return _utc(run.created_at)


def classify_job_observation(
    *,
    run: Any,
    job: dict[str, Any] | None,
    now: datetime,
    missing_since: datetime | str | None = None,
) -> str | None:
    """Return one closed store observation, or None while recovery remains possible."""

    state = run.state.value
    age = _utc(now) - (_utc(missing_since) if missing_since is not None else missing_job_reference_at(run))
    if job is None:
        if state in {"admitting", "queued", "running", "cancelling"}:
            return "definitively_missing" if age >= timedelta(minutes=10) else None
        if state == "publishing":
            return "publication_receipt_missing" if age > timedelta(days=30) else None
        return "definitively_missing"
    status = job.get("status")
    if status not in {"completed", "failed", "cancelled", "quarantined"}:
        return None
    if state == "publishing" and status == "completed":
        try:
            validate_publication_receipt(
                job=job,
                run=run,
                owner_user_id=run.owner_user_id,
            )
        except PublicationReceiptError:
            return "publication_receipt_mismatch"
        return "terminal_succeeded"
    if status == "completed":
        return "terminal_succeeded"
    if status == "cancelled":
        return "terminal_cancelled"
    return "terminal_failed"


@dataclass(frozen=True, slots=True)
class MaintenanceScope:
    store: Any
    dataset_id: str
    decision_service: Any | None = None

    def __post_init__(self) -> None:
        if self.decision_service is not None:
            return
        note_db = getattr(self.store, "_db", None)
        owner = getattr(self.store, "owner_user_id", None)
        if note_db is None or not isinstance(owner, str):
            return
        from .suggestion_service import build_suggestion_decision_service

        decisions = build_suggestion_decision_service(
            note_db=note_db,
            owner_user_id=owner,
            dataset_id=self.dataset_id,
        )
        object.__setattr__(self, "decision_service", decisions)


@dataclass(frozen=True, slots=True)
class MaintenancePassResult:
    claimed: int
    reconciled: int
    released: int
    cleaned: int


class SuggestionMaintenance:
    """Claim at most 100 runs and reconcile each after a separate Jobs lookup."""

    def __init__(self, *, jobs: Any, scopes: Iterable[MaintenanceScope]) -> None:
        self._jobs = jobs
        self._scopes = tuple(scopes)

    @staticmethod
    def _release(scope: MaintenanceScope, run: Any, now: datetime) -> None:
        scope.store.release_run_maintenance_lease(
            dataset_id=scope.dataset_id,
            run_id=run.id,
            expected_state=run.state.value,
            expected_revision=run.revision,
            maintenance_lease_token=run.maintenance_lease_token,
            now=now,
        )

    def _reconcile(
        self,
        *,
        scope: MaintenanceScope,
        run: Any,
        job: dict[str, Any] | None,
        observation: str,
        now: datetime,
    ) -> Any:
        if run.state.value == "publishing" and observation == "terminal_succeeded":
            return scope.store.activate_staged_run_from_maintenance(
                dataset_id=scope.dataset_id,
                run_id=run.id,
                expected_state="publishing",
                expected_revision=run.revision,
                maintenance_lease_token=run.maintenance_lease_token,
                observed_job_id=str(job["uuid"]),
                observed_completion_token=str(job["completion_token"]),
                observed_result_digest=str(job["result"]["result_digest"]),
                now=now,
            )
        error_code = None
        guidance_key = None
        if observation == "terminal_failed":
            candidate = str((job or {}).get("error_code") or "")
            error_code = candidate if candidate in _FAILURE_GUIDANCE else "notes_graph_provider_unavailable"
            guidance_key = _FAILURE_GUIDANCE[error_code]
        return scope.store.reconcile_run_after_job_lookup(
            dataset_id=scope.dataset_id,
            run_id=run.id,
            expected_state=run.state.value,
            expected_revision=run.revision,
            maintenance_lease_token=run.maintenance_lease_token,
            observation=observation,
            error_code=error_code,
            guidance_key=guidance_key,
            now=now,
        )

    @staticmethod
    def _job_matches_run(run: Any, job: dict[str, Any]) -> bool:
        return (
            str(job.get("uuid") or "") == run.job_id
            and str(job.get("owner_user_id") or "") == run.owner_user_id
            and job.get("domain") == "notes"
            and job.get("queue") == "graph-suggestions"
            and job.get("job_type") == "note_graph_suggestions"
        )

    def _resume_cancellation(
        self,
        *,
        scope: MaintenanceScope,
        run: Any,
        context: Any,
        job: dict[str, Any] | None,
        now: datetime,
    ) -> tuple[dict[str, Any] | None, bool]:
        result = SuggestionCancellationCoordinator(
            store=scope.store,
            jobs=self._jobs,
            owner_user_id=run.owner_user_id,
        ).resume(
            dataset_id=scope.dataset_id,
            operation_id=context.operation_id,
            now=now,
            job=job,
            expected_run=run,
        )
        job = result.job
        accepted = result.accepted

        if job is not None and job.get("status") not in _TERMINAL_JOB_STATUSES:
            job = self._jobs.get_job_or_archived_by_uuid(
                run.job_id,
                domain="notes",
                owner_user_id=run.owner_user_id,
            )
            if job is not None and not self._job_matches_run(run, job):
                job = None
            if not accepted and job is not None and job.get("status") in _TERMINAL_JOB_STATUSES:
                scope.store.complete_run_cancellation_receipt(
                    dataset_id=scope.dataset_id,
                    run_id=run.id,
                    operation_id=context.operation_id,
                    expected_state=run.state.value,
                    expected_revision=run.revision,
                    now=now,
                )
                accepted = True
        return job, accepted

    @staticmethod
    def _record_reconciliation(run: Any, reconciled: Any) -> None:
        state = reconciled.state.value
        error_code = None
        if state == "succeeded":
            record_event(
                SuggestionEventName.PUBLISHED,
                run_id=run.id,
                job_id=run.job_id,
                count=int(reconciled.suggestion_count),
            )
        elif state == "cancelled":
            record_event(
                SuggestionEventName.CANCELLED,
                run_id=run.id,
                job_id=run.job_id,
            )
        elif state == "stale":
            error_code = SuggestionErrorCode.FINGERPRINT_STALE
            record_event(
                SuggestionEventName.STALE,
                run_id=run.id,
                job_id=run.job_id,
                error_code=error_code,
            )
        elif state == "failed":
            try:
                error_code = SuggestionErrorCode(reconciled.error_code)
            except ValueError:
                error_code = SuggestionErrorCode.PROVIDER_UNAVAILABLE
            record_event(
                SuggestionEventName.FAILED,
                run_id=run.id,
                job_id=run.job_id,
                error_code=error_code,
            )
        if error_code is not None:
            record_run_error(error_code)
        record_event(
            SuggestionEventName.RECONCILED,
            run_id=run.id,
            job_id=run.job_id,
            error_code=error_code,
        )

    def run_pass(
        self,
        *,
        now: datetime,
        limit: int = 100,
        on_claimed: Callable[[int], None] | None = None,
    ) -> MaintenancePassResult:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("notes_graph_maintenance_limit_invalid")
        remaining = limit
        claimed = reconciled = released = 0
        for scope in self._scopes:
            if remaining == 0:
                break
            runs = scope.store.claim_runs_for_maintenance(
                dataset_id=scope.dataset_id,
                limit=remaining,
                now=now,
            )
            claimed += len(runs)
            remaining -= len(runs)
            if runs and on_claimed is not None:
                on_claimed(len(runs))
            for run in runs:
                cancellation_context = None
                try:
                    if run.state.value == "cancelling":
                        cancellation_context = scope.store.get_run_cancellation_maintenance_context(
                            dataset_id=scope.dataset_id,
                            run_id=run.id,
                        )
                    job = (
                        self._jobs.get_job_or_archived_by_uuid(
                            run.job_id,
                            domain="notes",
                            owner_user_id=run.owner_user_id,
                        )
                        if run.job_id
                        else None
                    )
                    if job is not None and not self._job_matches_run(run, job):
                        job = None
                    if cancellation_context is not None and cancellation_context.state == "in_progress":
                        job, _cancellation_accepted = self._resume_cancellation(
                            scope=scope,
                            run=run,
                            context=cancellation_context,
                            job=job,
                            now=now,
                        )
                except _TEMPORARY_AUTHORITY_ERRORS:
                    self._release(scope, run, now)
                    released += 1
                    continue
                observation = classify_job_observation(
                    run=run,
                    job=job,
                    now=now,
                    missing_since=missing_job_reference_at(
                        run,
                        cancellation_created_at=(
                            cancellation_context.created_at if cancellation_context is not None else None
                        ),
                    ),
                )
                if observation is None:
                    self._release(scope, run, now)
                    released += 1
                    continue
                reconciled_run = self._reconcile(
                    scope=scope,
                    run=run,
                    job=job,
                    observation=observation,
                    now=now,
                )
                reconciled += 1
                self._record_reconciliation(run, reconciled_run)

        acceptance_remaining = remaining
        for scope in self._scopes:
            if acceptance_remaining == 0:
                break
            if scope.decision_service is None:
                continue
            reconciliation_kwargs = {
                "dataset_id": scope.dataset_id,
                "limit": acceptance_remaining,
                "now": now,
            }
            if on_claimed is not None:
                reconciliation_kwargs["on_claimed"] = on_claimed
            decisions = scope.decision_service.reconcile_expired(**reconciliation_kwargs)
            claimed += len(decisions)
            reconciled += len(decisions)
            acceptance_remaining -= min(acceptance_remaining, len(decisions))

        cleaned = 0
        cleanup_remaining = max(0, limit - claimed)
        for scope in self._scopes:
            if cleanup_remaining == 0:
                break
            counts = scope.store.cleanup_retention(
                dataset_id=scope.dataset_id,
                now=now,
                limit=cleanup_remaining,
            )
            count = sum(int(value) for value in counts.values())
            cleaned += count
            cleanup_remaining -= min(cleanup_remaining, count)
        return MaintenancePassResult(claimed, reconciled, released, cleaned)


async def run_maintenance_loop(
    maintenance: SuggestionMaintenance,
    stop_event: asyncio.Event,
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> None:
    """Run once at startup and no more frequently than every 60 seconds."""

    while not stop_event.is_set():
        result = maintenance.run_pass(now=now())
        if inspect.isawaitable(result):
            await result
        if stop_event.is_set():
            break
        await sleep(60.0)


__all__ = [
    "MaintenancePassResult",
    "MaintenanceScope",
    "SuggestionMaintenance",
    "classify_job_observation",
    "missing_job_reference_at",
    "run_maintenance_loop",
]
