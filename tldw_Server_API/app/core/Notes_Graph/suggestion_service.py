"""One-attempt Notes graph suggestion worker orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from datetime import datetime, timezone
from typing import Any, Callable

from .suggestion_content import content_fingerprint
from .suggestion_generation import (
    SuggestionGenerationError,
    build_generation_request,
    generate_suggestions_once,
)
from .suggestion_jobs import (
    JOB_DOMAIN,
    JOB_PAYLOAD_KEYS,
    JOB_QUEUE,
    JOB_RESULT_KEYS,
    JOB_TYPE,
)
from .suggestion_observability import (
    SuggestionErrorCode,
    SuggestionEventName,
    record_candidate_counts,
    record_event,
    record_provider_usage,
    record_queue_latency,
    record_run_duration,
    record_run_error,
    record_validation_counts,
)
from .suggestion_retrieval import SuggestionRetriever


def build_suggestion_decision_service(
    *,
    note_db: Any,
    owner_user_id: str,
    dataset_id: str,
) -> Any | None:
    """Build owner/dataset-bound decision coordinators when canonical Sync is active."""

    from tldw_Server_API.app.core.Sync.v2.notes_link_coordinator import NotesLinkCoordinator
    from tldw_Server_API.app.core.Sync.v2.notes_organization_coordinator import (
        NotesOrganizationCoordinator,
    )
    from tldw_Server_API.app.core.Sync.v2.server_origin import (
        get_active_server_origin_sync_service_for_user,
    )

    from .suggestion_decisions import SuggestionDecisionService

    sync = get_active_server_origin_sync_service_for_user(owner_user_id)
    if sync is None:
        return None
    dataset = sync.store.get_dataset(dataset_id)
    if dataset is None or dataset.owner_user_id != owner_user_id:
        return None
    return SuggestionDecisionService(
        store=note_db.note_graph_suggestion_store,
        link_coordinator=NotesLinkCoordinator(sync, note_db, owner_user_id, dataset),
        organization_coordinator=NotesOrganizationCoordinator(sync, note_db, owner_user_id),
    )


class SuggestionWorkerError(RuntimeError):
    retryable = False

    def __init__(self, code: SuggestionErrorCode | str) -> None:
        try:
            self.error_code = SuggestionErrorCode(code)
        except ValueError as exc:
            raise ValueError("unknown suggestion worker error code") from exc
        self.failure_code = self.error_code.value
        super().__init__(self.failure_code)


class SuggestionWorkerCancelled(SuggestionWorkerError):
    def __init__(self) -> None:
        super().__init__(SuggestionErrorCode.GENERATION_CANCELLED)


_GENERATION_ERROR_CODES = {
    "notes_graph_provider_call_failed": SuggestionErrorCode.PROVIDER_UNAVAILABLE,
    "notes_graph_provider_call_policy_unsupported": SuggestionErrorCode.PROVIDER_RETRY_POLICY_UNSUPPORTED,
    "notes_graph_suggestion_input_too_large": SuggestionErrorCode.SOURCE_TOO_LARGE,
    "notes_graph_suggestion_stale_evidence": SuggestionErrorCode.FINGERPRINT_STALE,
    "notes_graph_suggestion_no_valid_items": SuggestionErrorCode.SUGGESTION_NO_VALID_ITEMS,
    "notes_graph_suggestion_invalid_model_output": SuggestionErrorCode.SUGGESTION_NO_VALID_ITEMS,
    "notes_graph_suggestion_unknown_reference": SuggestionErrorCode.SUGGESTION_NO_VALID_ITEMS,
}


def _error_code(exc: Exception) -> SuggestionErrorCode:
    if isinstance(exc, SuggestionWorkerError):
        return exc.error_code
    if isinstance(exc, SuggestionGenerationError):
        return _GENERATION_ERROR_CODES.get(
            exc.code,
            SuggestionErrorCode.SUGGESTION_NO_VALID_ITEMS,
        )
    value = str(exc)
    aliases = {
        "notes_graph_run_conflict": SuggestionErrorCode.RUN_CONFLICT,
        "notes_graph_suggestion_conflict": SuggestionErrorCode.RUN_CONFLICT,
        "notes_graph_fts_not_ready": SuggestionErrorCode.FTS_NOT_READY,
        "notes_graph_source_too_large": SuggestionErrorCode.SOURCE_TOO_LARGE,
    }
    return aliases.get(value, SuggestionErrorCode.PROVIDER_UNAVAILABLE)


def _utc(value: datetime | str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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


async def _invoke(
    callback: Callable[..., Any],
    /,
    *args: Any,
    sync_cleanup: Callable[[], None] | None = None,
    **kwargs: Any,
) -> Any:
    if inspect.iscoroutinefunction(callback):
        return await callback(*args, **kwargs)

    def invoke_sync() -> Any:
        try:
            return callback(*args, **kwargs)
        finally:
            if sync_cleanup is not None:
                sync_cleanup()

    value = await asyncio.to_thread(invoke_sync)
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
        sync_cleanup: Callable[[], None] | None = None,
    ) -> None:
        self._store_factory = store_factory
        self._resolve_capability = resolve_capability
        self._cancellation_requested = cancellation_requested
        self._retrieve = retrieve
        self._prepare = prepare
        self._generate = generate
        self._freshness_check = freshness_check
        self._now = now
        self._sync_cleanup = sync_cleanup

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
        store = await _invoke(self._store_factory, owner)
        queued = await _invoke(
            store.get_run,
            dataset_id=str(payload["dataset_id"]),
            run_id=str(payload["run_id"]),
            sync_cleanup=self._sync_cleanup,
        )
        self._validate_run_binding(run=queued, payload=payload, job_uuid=job_uuid)
        started_at = self._now()
        running = await _invoke(
            store.start_run,
            dataset_id=str(payload["dataset_id"]),
            run_id=str(payload["run_id"]),
            expected_state="queued",
            expected_revision=queued.revision,
            expected_job_id=job_uuid,
            acquired_completion_token=lease_id,
            now=started_at,
            sync_cleanup=self._sync_cleanup,
        )
        queued_at = job.get("created_at") or getattr(queued, "created_at", started_at)
        record_queue_latency(max(0.0, (_utc(started_at) - _utc(queued_at)).total_seconds()))
        try:
            retrieval = await _invoke(
                self._retrieve,
                store=store,
                dataset_id=str(payload["dataset_id"]),
                source_note_id=str(payload["source_note_id"]),
                sync_cleanup=self._sync_cleanup,
            )
            record_event(
                SuggestionEventName.SHORTLIST_COMPLETED,
                run_id=running.id,
                job_id=job_uuid,
                count=len(getattr(retrieval, "candidates", ())),
            )
            prepared = await _invoke(
                self._prepare,
                store=store,
                dataset_id=str(payload["dataset_id"]),
                retrieval=retrieval,
                sync_cleanup=self._sync_cleanup,
            )
            if await _invoke(self._cancellation_requested, job):
                raise SuggestionWorkerCancelled()
            resolved_provider = await _invoke(
                self._resolve_capability,
                provider=str(payload["provider"]),
                model=str(payload["model"]),
            )
            if isinstance(resolved_provider, tuple):
                capabilities, provider = resolved_provider
            else:
                capabilities = resolved_provider.capabilities
                provider = resolved_provider.provider
            if capabilities.revision != payload["capability_revision"] or not capabilities.generation_available:
                raise SuggestionWorkerError(SuggestionErrorCode.CAPABILITIES_CHANGED)
            record_event(
                SuggestionEventName.PROVIDER_STARTED,
                run_id=running.id,
                job_id=job_uuid,
            )
            generated = await _invoke(
                self._generate,
                prepared=prepared,
                provider=provider,
            )
            record_event(
                SuggestionEventName.PROVIDER_COMPLETED,
                run_id=running.id,
                job_id=job_uuid,
            )
            input_tokens = int(getattr(generated, "input_tokens", 0))
            output_tokens = int(getattr(generated, "output_tokens", 0))
            record_provider_usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            if await _invoke(self._cancellation_requested, job):
                raise SuggestionWorkerCancelled()
            await _invoke(
                self._freshness_check,
                store=store,
                dataset_id=str(payload["dataset_id"]),
                running=running,
                generated=generated,
                sync_cleanup=self._sync_cleanup,
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
            if dropped:
                record_event(
                    SuggestionEventName.VALIDATION_REJECTED,
                    run_id=running.id,
                    job_id=job_uuid,
                    count=dropped,
                )
            await _invoke(
                store.stage_suggestions,
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
                sync_cleanup=self._sync_cleanup,
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
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            if set(result) != JOB_RESULT_KEYS:
                raise SuggestionWorkerError(SuggestionErrorCode.JOB_RESULT_CONTRACT_INVALID)
            record_run_duration(max(0.0, (_utc(self._now()) - _utc(started_at)).total_seconds()))
            return result
        except Exception as exc:
            code = _error_code(exc)
            if code == SuggestionErrorCode.GENERATION_CANCELLED:
                event = SuggestionEventName.CANCELLED
            elif code == SuggestionErrorCode.FINGERPRINT_STALE:
                event = SuggestionEventName.STALE
            else:
                event = SuggestionEventName.FAILED
            record_event(
                event,
                run_id=running.id,
                job_id=job_uuid,
                error_code=code,
            )
            record_run_error(code)
            record_run_duration(max(0.0, (_utc(self._now()) - _utc(started_at)).total_seconds()))
            if isinstance(exc, SuggestionWorkerError) and exc.error_code == code:
                raise
            raise SuggestionWorkerError(code) from None


__all__ = [
    "SuggestionWorker",
    "SuggestionWorkerCancelled",
    "SuggestionWorkerError",
    "build_suggestion_decision_service",
]
