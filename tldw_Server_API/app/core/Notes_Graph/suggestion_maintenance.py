"""Bounded provider-independent reconciliation for suggestion runs."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from .suggestion_jobs import PublicationReceiptError, validate_publication_receipt
from .suggestion_observability import SuggestionEventName, record_event

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


def _utc(value: datetime | str) -> datetime:
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        parsed = value
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def classify_job_observation(*, run: Any, job: dict[str, Any] | None, now: datetime) -> str | None:
    """Return one closed store observation, or None while recovery remains possible."""

    state = run.state.value
    age = _utc(now) - _utc(run.created_at)
    if job is None:
        if state == "admitting":
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
    ) -> None:
        if run.state.value == "publishing" and observation == "terminal_succeeded":
            scope.store.activate_staged_run(
                dataset_id=scope.dataset_id,
                run_id=run.id,
                expected_state="publishing",
                expected_revision=run.revision,
                observed_job_id=str(job["uuid"]),
                observed_completion_token=str(job["completion_token"]),
                observed_result_digest=str(job["result"]["result_digest"]),
                now=now,
            )
            return
        error_code = None
        guidance_key = None
        if observation == "terminal_failed":
            candidate = str((job or {}).get("error_code") or "")
            error_code = candidate if candidate in _FAILURE_GUIDANCE else "notes_graph_provider_unavailable"
            guidance_key = _FAILURE_GUIDANCE[error_code]
        scope.store.reconcile_run_after_job_lookup(
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

    def run_pass(self, *, now: datetime, limit: int = 100) -> MaintenancePassResult:
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
            for run in runs:
                try:
                    job = (
                        self._jobs.get_job_or_archived_by_uuid(
                            run.job_id,
                            domain="notes",
                            owner_user_id=run.owner_user_id,
                        )
                        if run.job_id
                        else None
                    )
                except (ConnectionError, OSError, RuntimeError, TimeoutError):
                    self._release(scope, run, now)
                    released += 1
                    continue
                observation = classify_job_observation(run=run, job=job, now=now)
                if observation is None:
                    self._release(scope, run, now)
                    released += 1
                    continue
                self._reconcile(
                    scope=scope,
                    run=run,
                    job=job,
                    observation=observation,
                    now=now,
                )
                reconciled += 1
                record_event(
                    SuggestionEventName.RECONCILED,
                    run_id=run.id,
                    job_id=run.job_id,
                )

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
        maintenance.run_pass(now=now())
        if stop_event.is_set():
            break
        await sleep(60.0)


__all__ = [
    "MaintenancePassResult",
    "MaintenanceScope",
    "SuggestionMaintenance",
    "classify_job_observation",
    "run_maintenance_loop",
]
