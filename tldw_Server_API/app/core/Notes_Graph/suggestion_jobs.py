"""Content-free Jobs admission and receipt-bound publication."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

from tldw_Server_API.app.core.Jobs.manager import (
    OWNER_SCOPE_ACTIVE_LIMIT_EXCEEDED,
    OWNER_SCOPE_ADMISSION_LIMIT_EXCEEDED,
    OwnerScopeAdmissionPolicy,
)

from .suggestion_observability import SuggestionEventName, record_event

JOB_DOMAIN = "notes"
JOB_QUEUE = "graph-suggestions"
JOB_TYPE = "note_graph_suggestions"
JOB_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "dataset_id",
        "source_note_id",
        "source_fingerprint",
        "provider",
        "model",
        "capability_revision",
        "prompt_contract_version",
    }
)
JOB_RESULT_KEYS = frozenset(
    {
        "run_id",
        "result_digest",
        "candidate_count",
        "evidence_count",
        "validated_count",
        "dropped_count",
        "input_tokens",
        "output_tokens",
    }
)
_OWNER_ACTIVE_JOB_STATUSES = ("queued", "processing")


class PublicationReceiptError(RuntimeError):
    """Stable failure for an absent or mismatched publication authority."""


@dataclass(frozen=True, slots=True)
class SuggestionAdmission:
    disposition: str
    run: Any
    job: dict[str, Any] | None
    replay_envelope: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SuggestionCancellationCommand:
    """One durable cancellation continuation and its Jobs observation."""

    cancellation: Any
    job: dict[str, Any] | None
    accepted: bool


_JOB_NOT_SUPPLIED = object()


def completion_placeholder(run_id: str, job_uuid: str) -> str:
    value = hashlib.sha256(f"{run_id}:{job_uuid}".encode("ascii")).hexdigest()
    return f"placeholder_{value}"


def _payload(run: Any, dataset_id: str) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "run_id": run.id,
        "dataset_id": dataset_id,
        "source_note_id": run.source_note_id,
        "source_fingerprint": run.source_fingerprint,
        "provider": run.provider,
        "model": run.model,
        "capability_revision": run.capability_revision,
        "prompt_contract_version": run.prompt_contract_version,
    }
    if set(payload) != JOB_PAYLOAD_KEYS:
        raise RuntimeError("notes_graph_job_payload_contract_invalid")
    return payload


class SuggestionAdmissionService:
    """Coordinate one store admission and one separate Jobs enqueue."""

    def __init__(self, *, store: Any, jobs: Any, owner_user_id: str) -> None:
        self._store = store
        self._jobs = jobs
        self._owner_user_id = owner_user_id

    def _existing_job(self, run: Any) -> dict[str, Any] | None:
        if not run.job_id:
            return None
        return self._jobs.get_job_or_archived_by_uuid(
            run.job_id,
            domain=JOB_DOMAIN,
            owner_user_id=self._owner_user_id,
        )

    def _job_by_run_id(self, run_id: str) -> dict[str, Any] | None:
        return self._jobs.get_job_or_archived_by_idempotency_key(
            idempotency_key=run_id,
            domain=JOB_DOMAIN,
            queue=JOB_QUEUE,
            job_type=JOB_TYPE,
            owner_user_id=self._owner_user_id,
        )

    def _bind_job(
        self,
        *,
        run: Any,
        job: dict[str, Any],
        dataset_id: str,
        now: datetime,
    ) -> Any:
        if job.get("payload") != _payload(run, dataset_id):
            raise RuntimeError("notes_graph_admission_job_contract_invalid")
        queued = self._store.bind_admitted_run(
            dataset_id=dataset_id,
            run_id=run.id,
            expected_state="admitting",
            expected_revision=run.revision,
            job_id=str(job["uuid"]),
            completion_token=completion_placeholder(run.id, str(job["uuid"])),
            replay_envelope={"run_id": run.id, "state": "queued"},
            now=now,
        )
        record_event(
            SuggestionEventName.RUN_ADMITTED,
            run_id=run.id,
            job_id=str(job["uuid"]),
        )
        return queued

    def replay(
        self,
        *,
        dataset_id: str,
        source_note_id: str,
        requested_provider: str | None,
        requested_model: str | None,
        prompt_contract_version: str,
        idempotency_key: str,
    ) -> SuggestionAdmission | None:
        """Resolve an exact terminal receipt before current provider/source checks."""

        replay = self._store.get_run_admission_replay(
            dataset_id=dataset_id,
            source_note_id=source_note_id,
            requested_provider=requested_provider,
            requested_model=requested_model,
            prompt_contract_version=prompt_contract_version,
            idempotency_key=idempotency_key,
        )
        if replay is None:
            return None
        return SuggestionAdmission(
            replay.disposition,
            None,
            None,
            replay.replay_envelope,
        )

    def admit(
        self,
        *,
        dataset_id: str,
        source_note_id: str,
        source_fingerprint: str,
        provider: str,
        model: str,
        requested_provider: str | None = None,
        requested_model: str | None = None,
        capability_revision: str,
        prompt_contract_version: str,
        idempotency_key: str,
        now: datetime,
        validate_before_enqueue: Callable[[Any], None] | None = None,
    ) -> SuggestionAdmission:
        admission = self._store.admit_run(
            dataset_id=dataset_id,
            source_note_id=source_note_id,
            source_fingerprint=source_fingerprint,
            provider=provider,
            model=model,
            requested_provider=requested_provider,
            requested_model=requested_model,
            capability_revision=capability_revision,
            prompt_contract_version=prompt_contract_version,
            idempotency_key=idempotency_key,
            now=now,
        )
        if admission.run is None:
            envelope = admission.replay_envelope or {}
            return SuggestionAdmission(
                admission.disposition,
                None,
                None,
                envelope,
            )
        run = admission.run
        existing = self._existing_job(run)
        if existing is not None:
            return SuggestionAdmission(admission.disposition, run, existing)
        if run.state.value != "admitting":
            raise RuntimeError("notes_graph_admission_job_missing")

        recovered = self._job_by_run_id(run.id)
        if recovered is not None:
            queued = self._bind_job(
                run=run,
                job=recovered,
                dataset_id=dataset_id,
                now=now,
            )
            return SuggestionAdmission(admission.disposition, queued, recovered)

        if validate_before_enqueue is not None:
            try:
                validate_before_enqueue(run)
            except Exception:
                self._store.fail_admission(
                    dataset_id=dataset_id,
                    run_id=run.id,
                    expected_state="admitting",
                    expected_revision=run.revision,
                    error_code="notes_graph_capabilities_changed_before_queue",
                    guidance_key="retry_generation",
                    now=now,
                )
                raise

        try:
            job = self._jobs.create_job(
                domain=JOB_DOMAIN,
                queue=JOB_QUEUE,
                job_type=JOB_TYPE,
                payload=_payload(run, dataset_id),
                owner_user_id=self._owner_user_id,
                max_retries=0,
                idempotency_key=run.id,
                owner_scope_admission=OwnerScopeAdmissionPolicy(
                    active_statuses=_OWNER_ACTIVE_JOB_STATUSES,
                    active_limit=1,
                    admission_limit=20,
                    created_after=now - timedelta(hours=1),
                ),
            )
        except ValueError as exc:
            code = {
                OWNER_SCOPE_ACTIVE_LIMIT_EXCEEDED: "notes_graph_owner_active_run_conflict",
                OWNER_SCOPE_ADMISSION_LIMIT_EXCEEDED: "notes_graph_admission_rate_limited",
            }.get(str(exc))
            if code is None:
                raise
            if admission.disposition == "created":
                self._store.fail_admission(
                    dataset_id=dataset_id,
                    run_id=run.id,
                    expected_state="admitting",
                    expected_revision=run.revision,
                    error_code="notes_graph_admission_failed",
                    guidance_key="retry_generation",
                    now=now,
                )
            raise RuntimeError(code) from exc
        normalized_job = self._jobs.get_job_or_archived_by_uuid(
            str(job["uuid"]),
            domain=JOB_DOMAIN,
            owner_user_id=self._owner_user_id,
        )
        if normalized_job is None:
            raise RuntimeError("notes_graph_admission_job_missing")
        queued = self._bind_job(
            run=run,
            job=normalized_job,
            dataset_id=dataset_id,
            now=now,
        )
        return SuggestionAdmission(admission.disposition, queued, normalized_job)


class SuggestionCancellationCoordinator:
    """Continue one receipt-backed cancellation with Jobs calls outside ChaCha."""

    def __init__(self, *, store: Any, jobs: Any, owner_user_id: str) -> None:
        self._store = store
        self._jobs = jobs
        self._owner_user_id = owner_user_id

    @staticmethod
    def _job_matches(run: Any, job: dict[str, Any]) -> bool:
        return (
            str(job.get("uuid") or "") == run.job_id
            and str(job.get("owner_user_id") or "") == run.owner_user_id
            and job.get("domain") == JOB_DOMAIN
            and job.get("queue") == JOB_QUEUE
            and job.get("job_type") == JOB_TYPE
        )

    @staticmethod
    def _run_matches(expected: Any, actual: Any) -> bool:
        if expected is None:
            return True
        fields = ("id", "revision", "job_id", "maintenance_lease_token")
        return all(getattr(expected, field, None) == getattr(actual, field, None) for field in fields) and (
            getattr(expected.state, "value", expected.state)
            == getattr(actual.state, "value", actual.state)
        )

    def cancel(
        self,
        *,
        dataset_id: str,
        run_id: str,
        expected_source_note_id: str,
        expected_state: str | None,
        expected_revision: int,
        idempotency_key: str,
        now: datetime,
        reason: str = "user_cancelled",
    ) -> SuggestionCancellationCommand:
        admitted = self._store.admit_run_cancellation(
            dataset_id=dataset_id,
            run_id=run_id,
            expected_source_note_id=expected_source_note_id,
            expected_state=expected_state,
            expected_revision=expected_revision,
            reason=reason,
            idempotency_key=idempotency_key,
            now=now,
        )
        if admitted.disposition == "terminal_replay":
            return SuggestionCancellationCommand(admitted, None, True)
        return self.resume(
            dataset_id=dataset_id,
            operation_id=admitted.operation_id,
            now=now,
        )

    def resume(
        self,
        *,
        dataset_id: str,
        operation_id: str,
        now: datetime,
        job: dict[str, Any] | None | object = _JOB_NOT_SUPPLIED,
        expected_run: Any | None = None,
    ) -> SuggestionCancellationCommand:
        continuation = self._store.get_run_cancellation_continuation(
            dataset_id=dataset_id,
            operation_id=operation_id,
        )
        if continuation.disposition == "terminal_replay":
            return SuggestionCancellationCommand(continuation, None, True)
        run = continuation.run
        if run is None or not self._run_matches(expected_run, run):
            return SuggestionCancellationCommand(continuation, None, False)
        observed = job
        if observed is _JOB_NOT_SUPPLIED:
            observed = (
                self._jobs.get_job_or_archived_by_uuid(
                    run.job_id,
                    domain=JOB_DOMAIN,
                    owner_user_id=self._owner_user_id,
                )
                if run.job_id
                else None
            )
        if observed is not None and not self._job_matches(run, observed):
            observed = None

        terminal = {"completed", "failed", "cancelled", "quarantined"}
        accepted = run.job_id is None or (
            observed is not None and observed.get("status") in terminal
        )
        if observed is not None and not accepted:
            accepted = self._jobs.cancel_job(
                int(observed["id"]),
                reason="requested",
                expected_uuid=run.job_id,
                expected_domain=JOB_DOMAIN,
                expected_job_type=JOB_TYPE,
                cascade_dependents=False,
            )
        if accepted:
            continuation = self._store.complete_run_cancellation_receipt(
                dataset_id=dataset_id,
                run_id=run.id,
                operation_id=operation_id,
                expected_state=getattr(run.state, "value", run.state),
                expected_revision=run.revision,
                now=now,
            )
        return SuggestionCancellationCommand(continuation, observed, accepted)


def validate_publication_receipt(*, job: dict[str, Any], run: Any, owner_user_id: str) -> None:
    result = job.get("result")
    immutable = (
        job.get("uuid") == run.job_id
        and job.get("owner_user_id") == owner_user_id == run.owner_user_id
        and job.get("domain") == JOB_DOMAIN
        and job.get("queue") == JOB_QUEUE
        and job.get("job_type") == JOB_TYPE
        and job.get("status") == "completed"
        and job.get("completion_token") == run.expected_completion_token
        and isinstance(result, dict)
        and set(result) == JOB_RESULT_KEYS
        and result.get("run_id") == run.id
        and result.get("result_digest") == run.result_digest
    )
    if not immutable:
        raise PublicationReceiptError("notes_graph_publication_receipt_mismatch")


class SuggestionPublisher:
    """Publish staged rows only after an exact active/archive Jobs lookup."""

    def __init__(self, *, jobs: Any, store_factory: Any) -> None:
        self._jobs = jobs
        self._store_factory = store_factory

    def publish(
        self,
        *,
        run: Any,
        job_uuid: str,
        owner_user_id: str,
        dataset_id: str,
        now: datetime,
    ) -> Any | None:
        job = self._jobs.get_job_or_archived_by_uuid(
            job_uuid,
            domain=JOB_DOMAIN,
            owner_user_id=owner_user_id,
        )
        if job is None or job.get("status") not in {
            "completed",
            "failed",
            "cancelled",
            "quarantined",
        }:
            raise PublicationReceiptError("notes_graph_publication_receipt_pending")
        validate_publication_receipt(job=job, run=run, owner_user_id=owner_user_id)
        store = self._store_factory(owner_user_id)
        try:
            published = store.activate_staged_run(
                dataset_id=dataset_id,
                run_id=run.id,
                expected_state="publishing",
                expected_revision=run.revision,
                observed_job_id=job_uuid,
                observed_completion_token=str(job["completion_token"]),
                observed_result_digest=str(job["result"]["result_digest"]),
                now=now,
            )
        except RuntimeError as exc:
            if str(exc) == "notes_graph_run_conflict":
                return None
            raise
        record_event(
            SuggestionEventName.PUBLISHED,
            run_id=run.id,
            job_id=job_uuid,
            count=int(job["result"]["candidate_count"]),
        )
        return published


__all__ = [
    "JOB_DOMAIN",
    "JOB_PAYLOAD_KEYS",
    "JOB_QUEUE",
    "JOB_RESULT_KEYS",
    "JOB_TYPE",
    "PublicationReceiptError",
    "SuggestionAdmission",
    "SuggestionAdmissionService",
    "SuggestionCancellationCommand",
    "SuggestionCancellationCoordinator",
    "SuggestionPublisher",
    "completion_placeholder",
    "validate_publication_receipt",
]
