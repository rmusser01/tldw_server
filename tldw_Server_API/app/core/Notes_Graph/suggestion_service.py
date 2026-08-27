"""One-attempt Notes graph suggestion worker orchestration."""

from __future__ import annotations

import hashlib
import inspect
import json
from datetime import datetime, timezone
from typing import Any, Callable

from .suggestion_content import content_fingerprint
from .suggestion_generation import build_generation_request, generate_suggestions_once
from .suggestion_jobs import (
    JOB_DOMAIN,
    JOB_PAYLOAD_KEYS,
    JOB_QUEUE,
    JOB_RESULT_KEYS,
    JOB_TYPE,
)
from .suggestion_observability import (
    SuggestionEventName,
    record_candidate_counts,
    record_event,
    record_validation_counts,
)
from .suggestion_retrieval import SuggestionRetriever


class SuggestionWorkerError(RuntimeError):
    retryable = False

    def __init__(self, code: str) -> None:
        self.failure_code = code
        super().__init__(code)


class SuggestionWorkerCancelled(SuggestionWorkerError):
    def __init__(self) -> None:
        super().__init__("notes_graph_generation_cancelled")


def _default_retrieve(*, store: Any, dataset_id: str, source_note_id: str) -> Any:
    return SuggestionRetriever(store).retrieve(
        dataset_id=dataset_id,
        source_note_id=source_note_id,
    )


def _default_prepare(*, store: Any, dataset_id: str, retrieval: Any) -> Any:
    source = store.load_source_note(
        dataset_id=dataset_id,
        note_id=retrieval.source_note_id,
    )
    return build_generation_request(
        retrieval=retrieval,
        source_title=source.title,
        source_content=source.content,
    )


def _default_freshness_check(
    *,
    store: Any,
    dataset_id: str,
    running: Any,
    generated: Any,
) -> None:
    source = store.load_source_note(dataset_id=dataset_id, note_id=running.source_note_id)
    if content_fingerprint(source.title, source.content) != running.source_fingerprint:
        raise SuggestionWorkerError("notes_graph_fingerprint_stale")
    for relationship in generated.relationships:
        target = store.load_source_note(
            dataset_id=dataset_id,
            note_id=relationship.target_note_id,
        )
        if content_fingerprint(target.title, target.content) != relationship.target_fingerprint:
            raise SuggestionWorkerError("notes_graph_fingerprint_stale")


def _evidence(reference: Any, *, side: str, ordinal: int) -> dict[str, object]:
    return {
        "side": side,
        "ordinal": ordinal,
        "note_id": reference.note_id,
        "field": reference.field,
        "content_fingerprint": reference.fingerprint,
        "start_offset": reference.start_offset,
        "end_offset": reference.end_offset,
    }


def _candidate_id(run_id: str, kind: str, ordinal: int) -> str:
    digest = hashlib.sha256(f"{run_id}:{kind}:{ordinal}".encode("ascii")).hexdigest()
    return f"suggestion_{digest}"


def _stage_candidates(run_id: str, generated: Any) -> tuple[dict[str, Any], ...]:
    candidates: list[dict[str, Any]] = []
    for ordinal, item in enumerate(generated.relationships):
        evidence = tuple(
            _evidence(reference, side="source", ordinal=index) for index, reference in enumerate(item.source_evidence)
        ) + tuple(
            _evidence(reference, side="target", ordinal=index) for index, reference in enumerate(item.target_evidence)
        )
        candidates.append(
            {
                "id": _candidate_id(run_id, "related_note", ordinal),
                "kind": "related_note",
                "target_note_id": item.target_note_id,
                "target_fingerprint": item.target_fingerprint,
                "match_strength": item.match_strength.split()[0].lower(),
                "rationale": item.rationale,
                "evidence": evidence,
            }
        )
    offset = len(candidates)
    for ordinal, item in enumerate(generated.tags):
        candidates.append(
            {
                "id": _candidate_id(run_id, "tag", offset + ordinal),
                "kind": "tag",
                "normalized_tag": item.normalized_tag,
                "display_tag": item.display_tag,
                "keyword_sync_id": item.existing_tag_id,
                "match_strength": item.match_strength.split()[0].lower(),
                "rationale": item.rationale,
                "evidence": tuple(
                    _evidence(reference, side="source", ordinal=index)
                    for index, reference in enumerate(item.source_evidence)
                ),
            }
        )
    return tuple(candidates)


async def _resolve(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


class SuggestionWorker:
    """Validate one leased Job, invoke once, and stage before completion."""

    def __init__(
        self,
        *,
        store_factory: Callable[[str], Any],
        resolve_capability: Callable[..., Any],
        cancellation_requested: Callable[[dict[str, Any]], Any],
        retrieve: Callable[..., Any] = _default_retrieve,
        prepare: Callable[..., Any] = _default_prepare,
        generate: Callable[..., Any] = generate_suggestions_once,
        freshness_check: Callable[..., Any] = _default_freshness_check,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._store_factory = store_factory
        self._resolve_capability = resolve_capability
        self._cancellation_requested = cancellation_requested
        self._retrieve = retrieve
        self._prepare = prepare
        self._generate = generate
        self._freshness_check = freshness_check
        self._now = now

    @staticmethod
    def _validate_job(job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        owner = job.get("owner_user_id")
        payload = job.get("payload")
        if (
            not isinstance(owner, str)
            or not owner
            or job.get("domain") != JOB_DOMAIN
            or job.get("queue") != JOB_QUEUE
            or job.get("job_type") != JOB_TYPE
            or not isinstance(payload, dict)
            or set(payload) != JOB_PAYLOAD_KEYS
            or "owner_user_id" in payload
        ):
            raise SuggestionWorkerError("notes_graph_job_contract_invalid")
        return owner, payload

    @staticmethod
    def _validate_run_binding(
        *,
        run: Any,
        payload: dict[str, Any],
        job_uuid: str,
    ) -> None:
        immutable_fields = (
            "source_note_id",
            "source_fingerprint",
            "provider",
            "model",
            "capability_revision",
            "prompt_contract_version",
        )
        state = getattr(run.state, "value", run.state)
        if (
            state != "queued"
            or run.id != payload["run_id"]
            or run.job_id != job_uuid
            or any(getattr(run, field) != payload[field] for field in immutable_fields)
        ):
            raise SuggestionWorkerError("notes_graph_job_contract_invalid")

    async def handle(self, job: dict[str, Any]) -> dict[str, Any]:
        owner, payload = self._validate_job(job)
        lease_id = str(job.get("lease_id") or "")
        job_uuid = str(job.get("uuid") or "")
        store = self._store_factory(owner)
        queued = store.get_run(
            dataset_id=str(payload["dataset_id"]),
            run_id=str(payload["run_id"]),
        )
        self._validate_run_binding(run=queued, payload=payload, job_uuid=job_uuid)
        running = store.start_run(
            dataset_id=str(payload["dataset_id"]),
            run_id=str(payload["run_id"]),
            expected_state="queued",
            expected_revision=queued.revision,
            expected_job_id=job_uuid,
            acquired_completion_token=lease_id,
            now=self._now(),
        )
        retrieval = await _resolve(
            self._retrieve(
                store=store,
                dataset_id=str(payload["dataset_id"]),
                source_note_id=str(payload["source_note_id"]),
            )
        )
        record_event(
            SuggestionEventName.SHORTLIST_COMPLETED,
            run_id=running.id,
            job_id=job_uuid,
            count=len(getattr(retrieval, "candidates", ())),
        )
        prepared = await _resolve(
            self._prepare(
                store=store,
                dataset_id=str(payload["dataset_id"]),
                retrieval=retrieval,
            )
        )
        capabilities, provider = await _resolve(
            self._resolve_capability(
                provider=str(payload["provider"]),
                model=str(payload["model"]),
            )
        )
        if capabilities.revision != payload["capability_revision"] or not capabilities.generation_available:
            raise SuggestionWorkerError("notes_graph_capabilities_changed_before_provider")
        if await _resolve(self._cancellation_requested(job)):
            raise SuggestionWorkerCancelled()
        record_event(
            SuggestionEventName.PROVIDER_STARTED,
            run_id=running.id,
            job_id=job_uuid,
        )
        generated = await _resolve(self._generate(prepared=prepared, provider=provider))
        record_event(
            SuggestionEventName.PROVIDER_COMPLETED,
            run_id=running.id,
            job_id=job_uuid,
        )
        if await _resolve(self._cancellation_requested(job)):
            raise SuggestionWorkerCancelled()
        await _resolve(
            self._freshness_check(
                store=store,
                dataset_id=str(payload["dataset_id"]),
                running=running,
                generated=generated,
            )
        )
        candidates = _stage_candidates(running.id, generated)
        encoded = json.dumps(
            candidates,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        result_digest = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
        counts = dict(generated.validation_counts)
        received = int(counts.get("relationship_items_received", 0)) + int(counts.get("tag_items_received", 0))
        dropped = max(0, received - len(candidates))
        store.stage_suggestions(
            dataset_id=str(payload["dataset_id"]),
            run_id=running.id,
            expected_state="running",
            expected_revision=running.revision,
            expected_job_id=job_uuid,
            expected_completion_token=lease_id,
            result_digest=result_digest,
            candidates=candidates,
            invalid_item_count=dropped,
            now=self._now(),
        )
        evidence_count = sum(len(candidate["evidence"]) for candidate in candidates)
        record_event(
            SuggestionEventName.STAGED,
            run_id=running.id,
            job_id=job_uuid,
            count=len(candidates),
        )
        record_candidate_counts(candidates=len(candidates), evidence=evidence_count)
        record_validation_counts(validated=len(candidates), dropped=dropped)
        result = {
            "run_id": running.id,
            "result_digest": result_digest,
            "candidate_count": len(candidates),
            "evidence_count": evidence_count,
            "validated_count": len(candidates),
            "dropped_count": dropped,
            "input_tokens": 0,
            "output_tokens": 0,
        }
        if set(result) != JOB_RESULT_KEYS:
            raise SuggestionWorkerError("notes_graph_job_result_contract_invalid")
        return result


__all__ = [
    "SuggestionWorker",
    "SuggestionWorkerCancelled",
    "SuggestionWorkerError",
]
