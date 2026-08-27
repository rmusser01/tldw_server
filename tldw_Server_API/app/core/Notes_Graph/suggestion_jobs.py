"""Content-free Jobs admission and receipt-bound publication."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

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
_ACTIVE_JOB_STATUSES = ("queued", "processing")


class PublicationReceiptError(RuntimeError):
    """Stable failure for an absent or mismatched publication authority."""


@dataclass(frozen=True, slots=True)
class SuggestionAdmission:
    disposition: str
    run: Any
    job: dict[str, Any] | None


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

    def _enforce_owner_limits(self, *, now: datetime) -> None:
        for status in _ACTIVE_JOB_STATUSES:
            if self._jobs.count_jobs(
                domain=JOB_DOMAIN,
                queue=JOB_QUEUE,
                job_type=JOB_TYPE,
                owner_user_id=self._owner_user_id,
                status=status,
            ):
                raise RuntimeError("notes_graph_owner_active_run_conflict")
        recent = self._jobs.list_jobs(
            domain=JOB_DOMAIN,
            queue=JOB_QUEUE,
            job_type=JOB_TYPE,
            owner_user_id=self._owner_user_id,
            created_after=now - timedelta(hours=1),
            limit=20,
        )
        if len(recent) >= 20:
            raise RuntimeError("notes_graph_admission_rate_limited")

    def admit(
        self,
        *,
        dataset_id: str,
        source_note_id: str,
        source_fingerprint: str,
        provider: str,
        model: str,
        capability_revision: str,
        prompt_contract_version: str,
        idempotency_key: str,
        now: datetime,
    ) -> SuggestionAdmission:
        admission = self._store.admit_run(
            dataset_id=dataset_id,
            source_note_id=source_note_id,
            source_fingerprint=source_fingerprint,
            provider=provider,
            model=model,
            capability_revision=capability_revision,
            prompt_contract_version=prompt_contract_version,
            idempotency_key=idempotency_key,
            now=now,
        )
        if admission.run is None:
            envelope = admission.replay_envelope or {}
            run_id = str(envelope.get("run_id") or "")
            matching = next(
                (
                    job
                    for job in self._jobs.list_jobs(
                        domain=JOB_DOMAIN,
                        queue=JOB_QUEUE,
                        job_type=JOB_TYPE,
                        owner_user_id=self._owner_user_id,
                        limit=100,
                    )
                    if job.get("idempotency_key") == run_id
                ),
                None,
            )
            replay_run = SimpleNamespace(id=run_id, state=envelope.get("state")) if run_id else None
            return SuggestionAdmission(admission.disposition, replay_run, matching)
        run = admission.run
        existing = self._existing_job(run)
        if existing is not None:
            return SuggestionAdmission(admission.disposition, run, existing)
        if run.state.value != "admitting":
            raise RuntimeError("notes_graph_admission_job_missing")

        try:
            self._enforce_owner_limits(now=now)
        except RuntimeError:
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
            raise

        job = self._jobs.create_job(
            domain=JOB_DOMAIN,
            queue=JOB_QUEUE,
            job_type=JOB_TYPE,
            payload=_payload(run, dataset_id),
            owner_user_id=self._owner_user_id,
            max_retries=0,
            idempotency_key=run.id,
        )
        normalized_job = self._jobs.get_job_or_archived_by_uuid(
            str(job["uuid"]),
            domain=JOB_DOMAIN,
            owner_user_id=self._owner_user_id,
        )
        if normalized_job is None:
            raise RuntimeError("notes_graph_admission_job_missing")
        queued = self._store.bind_admitted_run(
            dataset_id=dataset_id,
            run_id=run.id,
            expected_state="admitting",
            expected_revision=run.revision,
            job_id=str(normalized_job["uuid"]),
            completion_token=completion_placeholder(
                run.id,
                str(normalized_job["uuid"]),
            ),
            replay_envelope={"run_id": run.id, "state": "queued"},
            now=now,
        )
        record_event(
            SuggestionEventName.RUN_ADMITTED,
            run_id=run.id,
            job_id=str(normalized_job["uuid"]),
        )
        return SuggestionAdmission(admission.disposition, queued, normalized_job)


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
    "SuggestionPublisher",
    "completion_placeholder",
    "validate_publication_receipt",
]
