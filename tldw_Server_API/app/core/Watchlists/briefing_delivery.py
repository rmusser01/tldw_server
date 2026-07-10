"""Idempotent post-artifact delivery for Watchlists briefing occurrences."""

from __future__ import annotations

import asyncio
import html
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.Notifications import NotificationsService
from tldw_Server_API.app.core.Watchlists.audio_artifact_projection import build_audio_projection
from tldw_Server_API.app.services.outputs_service import _resolve_output_path_for_user

_EXTERNAL_ADAPTERS = ("email", "chatbook")


class BriefingArtifactsNotReadyError(RuntimeError):
    """Raised when delivery is invoked before selected artifacts are durable."""


@dataclass(frozen=True)
class BriefingDeliveryResult:
    """Compact delivery result returned to Scheduler and API callers."""

    occurrence_id: int
    delivery_status: str
    adapters: dict[str, str]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def external_delivery_adapters(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return enabled external adapters; Reports is artifact storage, not delivery."""
    delivery = contract.get("delivery")
    if not isinstance(delivery, Mapping):
        return {}
    return {
        adapter: dict(config)
        for adapter in _EXTERNAL_ADAPTERS
        if isinstance((config := delivery.get(adapter)), Mapping) and bool(config.get("enabled"))
    }


def _read_stages(occurrence: Any) -> dict[str, dict[str, Any]]:
    raw = _json_object(getattr(occurrence, "stages_json", None))
    return {str(name): dict(stage) for name, stage in raw.items() if isinstance(stage, Mapping)}


def _adapter_outcome(stages: Mapping[str, Mapping[str, Any]], adapter: str) -> str | None:
    stage = stages.get(f"deliver:{adapter}")
    outcome = stage.get("outcome") if isinstance(stage, Mapping) else None
    return str(outcome) if outcome else None


def _attempt_count(stage: Mapping[str, Any] | None) -> int:
    try:
        return max(0, int((stage or {}).get("attempt_count") or 0))
    except (TypeError, ValueError):
        return 0


def _artifact_version(stages: Mapping[str, Mapping[str, Any]]) -> int:
    try:
        return max(1, int(stages.get("persist_text", {}).get("output_version") or 1))
    except (TypeError, ValueError):
        return 1


def _claim_attempt(
    watchlists_db: Any,
    occurrence: Any,
    *,
    adapter: str,
    stages: Mapping[str, Mapping[str, Any]],
    requested_stage: str | None = None,
    allow_retry: bool = False,
) -> Any | None:
    claim = getattr(watchlists_db, "claim_briefing_attempt", None)
    if not callable(claim):
        return None
    return claim(
        occurrence_id=int(occurrence.id),
        artifact_version=_artifact_version(stages),
        adapter=adapter,
        requested_stage=requested_stage,
        allow_retry=allow_retry,
    )


def _transition_attempt(
    watchlists_db: Any,
    attempt: Any | None,
    *,
    expected_states: set[str],
    state: str,
    **fields: Any,
) -> Any | None:
    transition = getattr(watchlists_db, "transition_briefing_attempt", None)
    if attempt is None or not callable(transition):
        return attempt
    return transition(
        int(attempt.id),
        expected_states=expected_states,
        state=state,
        **fields,
    )


def reconcile_successful_delivery_attempt(
    *,
    watchlists_db: Any,
    occurrence: Any,
    adapter: str,
) -> Any | None:
    """Project an authoritative successful ledger row and return the occurrence."""
    get_latest = getattr(watchlists_db, "get_latest_briefing_attempt", None)
    if not callable(get_latest):
        return None
    stages = _read_stages(occurrence)
    attempt = get_latest(
        occurrence_id=int(occurrence.id),
        artifact_version=_artifact_version(stages),
        adapter=adapter,
    )
    if attempt is None or str(attempt.state) != "successful":
        return None
    stage_name = f"deliver:{adapter}"
    stage = {
        **stages.get(stage_name, {}),
        "status": "ready",
        "code": getattr(attempt, "code", None) or "delivery_acknowledged",
        "retryable": False,
        "finished_at": getattr(attempt, "updated_at", None) or _utcnow_iso(),
        "outcome": "successful",
        "attempt_count": int(attempt.attempt),
    }
    finalize = getattr(watchlists_db, "finalize_briefing_attempt", None)
    if callable(finalize):
        return finalize(
            int(attempt.id),
            expected_states={"successful"},
            state="successful",
            stage_updates={stage_name: stage},
            code=stage["code"],
        )
    stages[stage_name] = stage
    return watchlists_db.update_briefing_occurrence(int(occurrence.id), stages=stages)


def _aggregate_status(adapters: Mapping[str, Mapping[str, Any]], stages: Mapping[str, Mapping[str, Any]]) -> str:
    if not adapters:
        return "not_configured"
    outcomes = [_adapter_outcome(stages, adapter) for adapter in adapters]
    if any(outcome == "sending" for outcome in outcomes):
        return "delivering"
    if any(outcome == "unknown" for outcome in outcomes):
        return "unknown"
    if all(outcome == "successful" for outcome in outcomes):
        return "delivered"
    if any(outcome in {"successful", "partial"} for outcome in outcomes):
        return "partially_delivered"
    if any(outcome == "failed" for outcome in outcomes):
        return "failed"
    return "waiting_for_artifacts"


def _aggregate_stage(delivery_status: str) -> dict[str, Any]:
    """Build the persisted aggregate delivery stage."""
    return {
        "status": {
            "not_configured": "skipped",
            "waiting_for_artifacts": "not_started",
            "delivering": "running",
            "delivered": "ready",
            "partially_delivered": "failed",
            "failed": "failed",
            "unknown": "failed",
        }[delivery_status],
        "code": None if delivery_status == "delivered" else delivery_status,
        "retryable": delivery_status in {"failed", "partially_delivered"},
        "finished_at": (
            _utcnow_iso()
            if delivery_status in {"not_configured", "delivered", "partially_delivered", "failed", "unknown"}
            else None
        ),
    }


def _save_stages(
    watchlists_db: Any,
    occurrence: Any,
    stages: dict[str, dict[str, Any]],
    *,
    changed_stage: str | None = None,
) -> Any:
    merge = getattr(watchlists_db, "merge_briefing_occurrence_stages", None)
    if callable(merge):
        if changed_stage is not None:
            occurrence = merge(
                int(occurrence.id),
                stage_updates={changed_stage: stages[changed_stage]},
            )
        else:
            occurrence = watchlists_db.get_briefing_occurrence(int(occurrence.id))
        stages.clear()
        stages.update(_read_stages(occurrence))
    delivery_status = _aggregate_status(
        external_delivery_adapters(_json_object(occurrence.contract_json)),
        stages,
    )
    stages["deliver"] = _aggregate_stage(delivery_status)
    if callable(merge):
        occurrence = merge(
            int(occurrence.id),
            stage_updates={"deliver": stages["deliver"]},
            delivery_status=delivery_status,
        )
        stages.clear()
        stages.update(_read_stages(occurrence))
        return occurrence
    return watchlists_db.update_briefing_occurrence(
        int(occurrence.id), stages=stages, delivery_status=delivery_status
    )


def _safe_result_details(details: Any) -> dict[str, Any]:
    if not isinstance(details, Mapping):
        return {}
    safe_keys = {
        "deliveries",
        "document_id",
        "error_type",
        "invalid_recipient_count",
        "provider",
        "provider_id",
        "reason",
        "recipient_count",
    }
    return {str(key): value for key, value in details.items() if key in safe_keys}


def _provider_timed_out(status: str, details: Mapping[str, Any]) -> bool:
    if "timeout" in str(details.get("reason") or "").lower():
        return True
    if "timeout" in str(details.get("error_type") or "").lower():
        return True
    deliveries = details.get("deliveries")
    return isinstance(deliveries, list) and any(
        "timeout" in str(entry.get("error_type") or "").lower() for entry in deliveries if isinstance(entry, Mapping)
    )


def _result_outcome(adapter: str, result: Any) -> tuple[str, str, dict[str, Any]]:
    status = str(getattr(result, "status", "failed") or "failed").lower()
    details = _safe_result_details(getattr(result, "details", None))
    if _provider_timed_out(status, details):
        return "unknown", "delivery_outcome_unknown", details
    if status in ({"sent"} if adapter == "email" else {"stored"}):
        return "successful", "delivery_acknowledged", details
    if status == "partial":
        return "partial", "delivery_partially_acknowledged", details
    return "failed", str(details.get("reason") or f"{adapter}_delivery_failed"), details


def _supports_idempotency_key(callable_: Any) -> bool:
    try:
        parameters = inspect.signature(callable_).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "idempotency_key" or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _output_content(user_id: int, output: Any) -> str:
    metadata = _json_object(getattr(output, "metadata_json", None))
    compatibility_content = metadata.get("content")
    if isinstance(compatibility_content, str):
        return compatibility_content
    storage_path = str(getattr(output, "storage_path", None) or "").strip()
    if not storage_path:
        raise BriefingArtifactsNotReadyError("briefing_text_artifact_missing")
    path: Path = _resolve_output_path_for_user(user_id, storage_path)
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise BriefingArtifactsNotReadyError("briefing_text_artifact_unavailable") from exc


def _persist_output_result(
    collections_db: Any,
    output: Any,
    *,
    adapter: str,
    outcome: str,
    code: str,
    details: Mapping[str, Any],
) -> Any:
    metadata = _json_object(getattr(output, "metadata_json", None))
    deliveries = metadata.get("briefing_deliveries")
    if not isinstance(deliveries, dict):
        deliveries = {}
    deliveries[adapter] = {
        "outcome": outcome,
        "code": code,
        "details": dict(details),
        "finished_at": _utcnow_iso(),
    }
    metadata["briefing_deliveries"] = deliveries
    return collections_db.update_output_artifact_metadata(
        int(output.id),
        metadata_json=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    )


async def _send_adapter(
    *,
    adapter: str,
    config: Mapping[str, Any],
    notifications: Any,
    output: Any,
    content: str,
    occurrence: Any,
) -> Any:
    idempotency_key = f"watchlists-briefing-delivery:{occurrence.user_id}:{occurrence.id}:{adapter}"
    if adapter == "email":
        method = notifications.deliver_email
        kwargs: dict[str, Any] = {
            "subject": str(config.get("subject") or getattr(output, "title", None) or "Watchlist briefing"),
            "html_body": f"<pre>{html.escape(content)}</pre>",
            "text_body": content,
            "recipients": list(config.get("recipients") or []),
            "attachments": None,
            "fallback_to_user_email": False,
        }
        if bool(config.get("attach_file", True)):
            kwargs["attachments"] = [
                {
                    "filename": Path(str(getattr(output, "storage_path", None) or "briefing.md")).name,
                    "content": content.encode("utf-8"),
                }
            ]
        if _supports_idempotency_key(method):
            kwargs["idempotency_key"] = idempotency_key
        return await method(**kwargs)

    method = notifications.deliver_chatbook
    kwargs = {
        "title": str(config.get("title") or getattr(output, "title", None) or "Watchlist briefing"),
        "content": content,
        "description": config.get("description"),
        "metadata": {
            **dict(config.get("metadata") or {}),
            "job_id": int(occurrence.job_id),
            "run_id": int(occurrence.run_id),
            "output_id": int(output.id),
            "delivery_idempotency_key": idempotency_key,
        },
        "provider": config.get("provider") or "watchlists",
        "model": config.get("model") or "watchlists",
        "conversation_id": config.get("conversation_id"),
    }
    if _supports_idempotency_key(method):
        kwargs["idempotency_key"] = idempotency_key
    result = method(**kwargs)
    return await result if inspect.isawaitable(result) else result


async def schedule_briefing_delivery(
    *,
    occurrence: Any,
    audio_task_id: str | None,
    scheduler: Any | None = None,
    watchlists_db: Any | None = None,
    requested_adapters: set[str] | None = None,
    confirmed_unknown_adapters: set[str] | None = None,
) -> str:
    """Submit one stable delivery task, dependent on selected audio when present."""
    if scheduler is None:
        from tldw_Server_API.app.core.Scheduler import get_global_scheduler

        scheduler = await get_global_scheduler()
    task_key = f"watchlists-briefing-delivery:{occurrence.user_id}:{occurrence.id}"
    stages = _read_stages(occurrence)
    requested = sorted(requested_adapters or set())
    configured = external_delivery_adapters(_json_object(occurrence.contract_json))
    adapters_to_claim = requested or sorted(configured)
    confirmed = set(confirmed_unknown_adapters or set())
    for adapter in requested:
        reconciled = (
            reconcile_successful_delivery_attempt(
                watchlists_db=watchlists_db,
                occurrence=occurrence,
                adapter=adapter,
            )
            if watchlists_db is not None
            else None
        )
        if reconciled is not None:
            raise BriefingArtifactsNotReadyError("delivery_already_successful")
    if watchlists_db is not None:
        get_latest = getattr(watchlists_db, "get_latest_briefing_attempt", None)
        transition_attempt = getattr(watchlists_db, "transition_briefing_attempt", None)
        if callable(get_latest) and callable(transition_attempt):
            for adapter in confirmed:
                previous = get_latest(
                    occurrence_id=int(occurrence.id),
                    artifact_version=_artifact_version(stages),
                    adapter=adapter,
                )
                if previous is not None and str(previous.state) == "sending":
                    stage_name = f"deliver:{adapter}"
                    unknown_stage = {
                        **stages.get(stage_name, {}),
                        "status": "failed",
                        "code": "delivery_outcome_unknown",
                        "retryable": False,
                        "finished_at": _utcnow_iso(),
                        "outcome": "unknown",
                        "attempt_count": int(previous.attempt),
                    }
                    finalize = getattr(watchlists_db, "finalize_briefing_attempt", None)
                    if callable(finalize):
                        occurrence = finalize(
                            int(previous.id),
                            expected_states={"sending"},
                            state="unknown",
                            stage_updates={stage_name: unknown_stage},
                            delivery_status="unknown",
                            code="delivery_outcome_unknown",
                        ) or watchlists_db.get_briefing_occurrence(int(occurrence.id))
                        stages = _read_stages(occurrence)
                    else:
                        transition_attempt(
                            int(previous.id),
                            expected_states={"sending"},
                            state="unknown",
                            code="delivery_outcome_unknown",
                        )
    attempts = {
        adapter: _claim_attempt(
            watchlists_db,
            occurrence,
            adapter=adapter,
            stages=stages,
            requested_stage=f"deliver:{adapter}",
            allow_retry=adapter in requested,
        )
        for adapter in adapters_to_claim
        if watchlists_db is not None
    }
    if requested:
        attempt_keys = [
            f"{adapter}-{getattr(attempts.get(adapter), 'attempt', _attempt_count(stages.get(f'deliver:{adapter}')))}"
            for adapter in requested
        ]
        task_key = f"{task_key}:retry:{','.join(attempt_keys)}"
    elif audio_task_id and occurrence.delivery_task_id:
        previous_dependency = stages.get("deliver", {}).get("audio_dependency_task_id")
        if str(previous_dependency or "") != str(audio_task_id):
            task_key = f"{task_key}:audio:{audio_task_id}"
    try:
        task_id = await scheduler.submit(
            "watchlists_deliver_briefing",
            payload={
                "user_id": int(occurrence.user_id),
                "occurrence_id": int(occurrence.id),
                "audio_dependency_task_id": str(audio_task_id) if audio_task_id else None,
                "requested_adapters": requested,
                "confirmed_unknown_adapters": sorted(confirmed_unknown_adapters or set()),
            },
            queue_name="watchlists",
            depends_on=[str(audio_task_id)] if audio_task_id else None,
            idempotency_key=task_key,
            metadata={
                "source": "watchlists_briefing_delivery",
                "user_id": str(occurrence.user_id),
                "briefing_occurrence_id": int(occurrence.id),
            },
            max_retries=0,
        )
    except BaseException:
        for attempt in attempts.values():
            _transition_attempt(
                watchlists_db,
                attempt,
                expected_states={"intent"},
                state="failed",
                code="delivery_enqueue_failed",
            )
        raise
    for attempt in attempts.values():
        bind_attempt = getattr(watchlists_db, "bind_briefing_attempt_scheduler_task", None)
        if attempt is not None and callable(bind_attempt):
            attempt = bind_attempt(
                int(attempt.id),
                scheduler_task_id=str(task_id),
                request_id=str(task_id),
            )
        _transition_attempt(
            watchlists_db,
            attempt,
            expected_states={"intent"},
            state="queued",
            scheduler_task_id=str(task_id),
        )
    if watchlists_db is not None:
        watchlists_db.update_briefing_occurrence(
            int(occurrence.id),
            delivery_task_id=str(task_id),
        )
    return str(task_id)


def _workflow_artifact_metadata(artifact: Any) -> dict[str, Any]:
    if isinstance(artifact, Mapping):
        return _json_object(artifact.get("metadata_json") or artifact.get("metadata"))
    return _json_object(getattr(artifact, "metadata_json", None) or getattr(artifact, "metadata", None))


def _persist_audio_terminal_failure(
    *,
    watchlists_db: Any,
    occurrence: Any,
    attempt: Any,
    stages: dict[str, dict[str, Any]],
    state: str,
    code: str,
    workflow_run_id: str,
) -> bool:
    target_stage = (
        "generate_audio"
        if code == "audio_final_artifact_missing"
        else str(getattr(attempt, "requested_stage", None) or "generate_audio")
    )
    if target_stage not in {"compose_audio_script", "persist_audio_script", "generate_audio", "persist_audio"}:
        target_stage = "generate_audio"
    now = _utcnow_iso()
    stages[target_stage] = {
        **stages.get(target_stage, {}),
        "status": "cancelled" if state == "cancelled" else "failed",
        "code": code,
        "retryable": True,
        "finished_at": now,
        "workflow_run_id": workflow_run_id,
        "attempt_count": int(getattr(attempt, "attempt", 1)),
    }
    finalize = getattr(watchlists_db, "finalize_briefing_attempt", None)
    if callable(finalize):
        return finalize(
            int(attempt.id),
            expected_states={"intent", "queued", "sending"},
            state=state,
            stage_updates={target_stage: stages[target_stage]},
            artifact_status="cancelled" if state == "cancelled" else "failed",
            workflow_run_id=workflow_run_id,
            code=code,
        ) is not None
    transitioned = watchlists_db.transition_briefing_attempt(
        int(attempt.id),
        expected_states={"intent", "queued", "sending"},
        state=state,
        workflow_run_id=workflow_run_id,
        code=code,
    )
    if transitioned is None:
        return False
    watchlists_db.update_briefing_occurrence(
        int(occurrence.id),
        stages=stages,
        artifact_status="cancelled" if state == "cancelled" else "failed",
    )
    return True


def record_audio_workflow_terminal(
    *,
    user_id: int,
    tenant_id: str,
    workflow_run_id: str,
    status: str,
    metadata: Mapping[str, Any],
    workflow_db: Any,
    workflow_run: Any | None = None,
    watchlists_db: Any | None = None,
) -> None:
    """Validate and durably project one terminal Watchlists audio Workflow."""
    from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase

    watchlists_db = watchlists_db or WatchlistsDatabase.for_user(user_id)
    try:
        occurrence_id = int(metadata["briefing_occurrence_id"])
        attempt_id = int(metadata["briefing_attempt_id"])
        expected_job_id = int(metadata["watchlist_job_id"])
        expected_run_id = int(metadata["watchlist_run_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BriefingArtifactsNotReadyError("audio_workflow_identity_missing") from exc
    request_id = str(metadata.get("audio_request_id") or "").strip()
    occurrence = watchlists_db.get_briefing_occurrence(occurrence_id)
    attempt = watchlists_db.get_briefing_attempt(attempt_id)
    workflow_run = workflow_run or workflow_db.get_run(workflow_run_id)
    get_latest_attempt = getattr(watchlists_db, "get_latest_briefing_attempt", None)
    latest_attempt = (
        get_latest_attempt(
            occurrence_id=occurrence_id,
            artifact_version=int(attempt.artifact_version),
            adapter="audio",
        )
        if callable(get_latest_attempt)
        else attempt
    )
    if (
        str(metadata.get("source")) != "watchlist_audio_briefing"
        or str(occurrence.user_id) != str(user_id)
        or int(occurrence.job_id) != expected_job_id
        or int(occurrence.run_id) != expected_run_id
        or int(attempt.occurrence_id) != occurrence_id
        or str(attempt.adapter) != "audio"
        or int(attempt.artifact_version) != _artifact_version(_read_stages(occurrence))
        or workflow_run is None
        or (
            getattr(workflow_run, "run_id", None) is not None
            and str(workflow_run.run_id) != str(workflow_run_id)
        )
        or (
            getattr(workflow_run, "user_id", None) is not None
            and str(workflow_run.user_id) != str(user_id)
        )
        or (
            getattr(workflow_run, "tenant_id", None) is not None
            and str(workflow_run.tenant_id) != str(tenant_id)
        )
        or (getattr(attempt, "request_id", None) and str(attempt.request_id) != request_id)
    ):
        raise BriefingArtifactsNotReadyError("audio_workflow_identity_mismatch")
    if latest_attempt is None or int(latest_attempt.id) != int(attempt.id):
        return
    stages = _read_stages(occurrence)
    terminal_status = str(status).lower()
    if terminal_status in {"failed", "cancelled", "canceled"}:
        state = "cancelled" if terminal_status in {"cancelled", "canceled"} else "failed"
        _persist_audio_terminal_failure(
            watchlists_db=watchlists_db,
            occurrence=occurrence,
            attempt=attempt,
            stages=stages,
            state=state,
            code="audio_workflow_cancelled" if state == "cancelled" else "audio_workflow_failed",
            workflow_run_id=workflow_run_id,
        )
        return
    if terminal_status != "succeeded":
        raise BriefingArtifactsNotReadyError("audio_workflow_not_terminal")

    artifacts = list(workflow_db.list_artifacts_for_run(workflow_run_id) or [])
    projection = build_audio_projection(
        run_id=int(occurrence.run_id),
        task_id=getattr(attempt, "scheduler_task_id", None),
        audio_request_id=request_id,
        workflow_run=workflow_run,
        artifacts=artifacts,
    )
    final_artifact = projection.get("final_artifact")
    final_artifact_id = final_artifact.get("artifact_id") if isinstance(final_artifact, Mapping) else None
    matching_final = False
    for artifact in artifacts:
        artifact_id = artifact.get("artifact_id") if isinstance(artifact, Mapping) else getattr(artifact, "artifact_id", None)
        if str(artifact_id or "") != str(final_artifact_id or ""):
            continue
        artifact_metadata = _workflow_artifact_metadata(artifact)
        matching_final = (
            bool(artifact_metadata.get("final_artifact"))
            and str(artifact_metadata.get("source")) == "watchlist_audio_briefing"
            and str(artifact_metadata.get("watchlist_job_id")) == str(occurrence.job_id)
            and str(artifact_metadata.get("watchlist_run_id")) == str(occurrence.run_id)
            and str(artifact_metadata.get("briefing_occurrence_id")) == str(occurrence.id)
            and str(artifact_metadata.get("briefing_attempt_id")) == str(attempt.id)
            and str(artifact_metadata.get("audio_request_id")) == request_id
        )
        break
    if projection.get("status") != "completed" or not final_artifact_id or not matching_final:
        applied = _persist_audio_terminal_failure(
            watchlists_db=watchlists_db,
            occurrence=occurrence,
            attempt=attempt,
            stages=stages,
            state="failed",
            code="audio_final_artifact_missing",
            workflow_run_id=workflow_run_id,
        )
        if applied:
            raise BriefingArtifactsNotReadyError("audio_final_artifact_missing")
        return

    now = _utcnow_iso()
    for name in ("compose_audio_script", "persist_audio_script", "generate_audio", "persist_audio"):
        stages[name] = {
            **stages.get(name, {}),
            "status": "ready",
            "code": None,
            "retryable": False,
            "finished_at": now,
            "audio_request_id": request_id,
            "workflow_run_id": workflow_run_id,
            "artifact_id": str(final_artifact_id) if name == "persist_audio" else stages.get(name, {}).get("artifact_id"),
            "attempt_count": int(attempt.attempt),
        }
    text_ready = stages.get("persist_text", {}).get("status") == "ready"
    finalize = getattr(watchlists_db, "finalize_briefing_attempt", None)
    if callable(finalize):
        finalize(
            int(attempt.id),
            expected_states={"intent", "queued", "sending"},
            state="successful",
            stage_updates={name: stages[name] for name in ("compose_audio_script", "persist_audio_script", "generate_audio", "persist_audio")},
            artifact_status="ready" if text_ready else "failed",
            workflow_run_id=workflow_run_id,
            artifact_id=str(final_artifact_id),
        )
        return
    transitioned = watchlists_db.transition_briefing_attempt(
        int(attempt.id),
        expected_states={"intent", "queued", "sending"},
        state="successful",
        workflow_run_id=workflow_run_id,
        artifact_id=str(final_artifact_id),
    )
    if transitioned is not None:
        watchlists_db.update_briefing_occurrence(
            int(occurrence.id),
            stages=stages,
            artifact_status="ready" if text_ready else "failed",
        )


def assert_audio_dependency_ready(
    *,
    user_id: int,
    occurrence_id: int,
    audio_task_id: str,
) -> None:
    """Require a matching terminal audio attempt without mutating occurrence state."""
    from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase

    watchlists_db = WatchlistsDatabase.for_user(user_id)
    occurrence = watchlists_db.get_briefing_occurrence(occurrence_id)
    if not occurrence.audio_task_id or str(occurrence.audio_task_id) != str(audio_task_id):
        raise BriefingArtifactsNotReadyError("audio_dependency_mismatch")
    contract = _json_object(occurrence.contract_json)
    if not bool(contract.get("audio", {}).get("enabled")):
        raise BriefingArtifactsNotReadyError("audio_not_selected")
    stages = _read_stages(occurrence)
    attempt = watchlists_db.get_latest_briefing_attempt(
        occurrence_id=int(occurrence.id),
        artifact_version=_artifact_version(stages),
        adapter="audio",
    )
    if (
        attempt is None
        or str(attempt.state) != "successful"
        or str(attempt.scheduler_task_id or "") != str(audio_task_id)
        or any(stages.get(name, {}).get("status") != "ready" for name in ("compose_audio_script", "persist_audio_script", "generate_audio", "persist_audio"))
    ):
        raise BriefingArtifactsNotReadyError("audio_dependency_not_terminal")


async def deliver_briefing_occurrence(
    *,
    occurrence_id: int,
    watchlists_db: Any,
    collections_db: Any,
    notifications: Any | None = None,
    requested_adapters: set[str] | None = None,
    confirmed_unknown_adapters: set[str] | None = None,
) -> BriefingDeliveryResult:
    """Deliver configured adapters once and persist every adapter transition."""
    occurrence = watchlists_db.get_briefing_occurrence(int(occurrence_id))
    contract = _json_object(occurrence.contract_json)
    adapters = external_delivery_adapters(contract)
    stages = _read_stages(occurrence)
    if not adapters:
        occurrence = _save_stages(watchlists_db, occurrence, stages)
        return BriefingDeliveryResult(int(occurrence.id), "not_configured", {})
    if str(occurrence.artifact_status) != "ready":
        raise BriefingArtifactsNotReadyError("briefing_artifacts_not_ready")
    if occurrence.output_id is None:
        raise BriefingArtifactsNotReadyError("briefing_text_artifact_missing")

    output = collections_db.get_output_artifact(int(occurrence.output_id))
    if int(getattr(output, "run_id", occurrence.run_id) or occurrence.run_id) != int(occurrence.run_id):
        raise BriefingArtifactsNotReadyError("briefing_output_ownership_mismatch")
    content = _output_content(int(occurrence.user_id), output)
    notifications = notifications or NotificationsService(user_id=int(occurrence.user_id))
    requested = set(requested_adapters or set())
    confirmed = set(confirmed_unknown_adapters or set())

    for adapter, config in adapters.items():
        reconciled = reconcile_successful_delivery_attempt(
            watchlists_db=watchlists_db,
            occurrence=occurrence,
            adapter=adapter,
        )
        if reconciled is not None:
            occurrence = reconciled
            stages = _read_stages(occurrence)
            continue
        previous_outcome = _adapter_outcome(stages, adapter)
        if previous_outcome == "successful":
            continue
        if previous_outcome == "sending":
            previous = stages.get(f"deliver:{adapter}", {})
            stages[f"deliver:{adapter}"] = {
                **previous,
                "status": "failed",
                "code": "delivery_outcome_unknown",
                "retryable": False,
                "finished_at": _utcnow_iso(),
                "outcome": "unknown",
            }
            get_latest = getattr(watchlists_db, "get_latest_briefing_attempt", None)
            reconciled_unknown = False
            if callable(get_latest):
                sending_attempt = get_latest(
                    occurrence_id=int(occurrence.id),
                    artifact_version=_artifact_version(stages),
                    adapter=adapter,
                )
                finalize = getattr(watchlists_db, "finalize_briefing_attempt", None)
                if sending_attempt is not None and callable(finalize):
                    occurrence = finalize(
                        int(sending_attempt.id),
                        expected_states={"sending"},
                        state="unknown",
                        stage_updates={f"deliver:{adapter}": stages[f"deliver:{adapter}"]},
                        delivery_status="unknown",
                        code="delivery_outcome_unknown",
                    ) or watchlists_db.get_briefing_occurrence(int(occurrence.id))
                    stages = _read_stages(occurrence)
                    reconciled_unknown = True
            if not reconciled_unknown:
                occurrence = _save_stages(
                    watchlists_db, occurrence, stages, changed_stage=f"deliver:{adapter}"
                )
            continue
        if previous_outcome == "unknown" and adapter not in confirmed:
            continue
        if previous_outcome == "partial" and adapter not in requested:
            continue
        if requested and adapter not in requested:
            continue

        attempt = _claim_attempt(
            watchlists_db,
            occurrence,
            adapter=adapter,
            stages=stages,
            requested_stage=f"deliver:{adapter}",
            allow_retry=adapter in requested,
        )
        if attempt is not None:
            attempt_count = int(attempt.attempt)
            attempt_state = str(attempt.state)
            if attempt_state not in {"intent", "queued"}:
                continue
            claimed = _transition_attempt(
                watchlists_db,
                attempt,
                expected_states={"intent", "queued"},
                state="sending",
            )
            if claimed is None:
                continue
            attempt = claimed
        else:
            attempt_count = _attempt_count(stages.get(f"deliver:{adapter}")) + 1
        started_at = _utcnow_iso()
        stages[f"deliver:{adapter}"] = {
            "status": "running",
            "code": None,
            "retryable": False,
            "started_at": started_at,
            "finished_at": None,
            "outcome": "sending",
            "attempt_count": attempt_count,
        }
        occurrence = _save_stages(
            watchlists_db, occurrence, stages, changed_stage=f"deliver:{adapter}"
        )
        try:
            provider_result = await _send_adapter(
                adapter=adapter,
                config=config,
                notifications=notifications,
                output=output,
                content=content,
                occurrence=occurrence,
            )
            outcome, code, details = _result_outcome(adapter, provider_result)
        except asyncio.CancelledError:
            outcome, code, details = "unknown", "delivery_outcome_unknown", {"error_type": "CancelledError"}
            terminal_stage = {
                "status": "failed",
                "code": code,
                "retryable": False,
                "started_at": started_at,
                "finished_at": _utcnow_iso(),
                "outcome": outcome,
                "attempt_count": attempt_count,
            }
            stages[f"deliver:{adapter}"] = terminal_stage
            delivery_status = _aggregate_status(adapters, stages)
            finalize = getattr(watchlists_db, "finalize_briefing_attempt", None)
            if attempt is not None and callable(finalize):
                finalize(
                    int(attempt.id),
                    expected_states={"sending"},
                    state="unknown",
                    stage_updates={
                        f"deliver:{adapter}": terminal_stage,
                        "deliver": _aggregate_stage(delivery_status),
                    },
                    delivery_status=delivery_status,
                    code=code,
                )
            else:
                _transition_attempt(
                    watchlists_db,
                    attempt,
                    expected_states={"sending"},
                    state="unknown",
                    code=code,
                )
                _save_stages(
                    watchlists_db, occurrence, stages, changed_stage=f"deliver:{adapter}"
                )
            raise
        except (asyncio.TimeoutError, TimeoutError):
            outcome, code, details = "unknown", "delivery_outcome_unknown", {"error_type": "TimeoutError"}
        except Exception as exc:  # noqa: BLE001 - post-dispatch exceptions are ambiguous
            outcome, code, details = "unknown", "delivery_outcome_unknown", {"error_type": type(exc).__name__}

        terminal_stage = {
            "status": "ready" if outcome == "successful" else "failed",
            "code": code,
            "retryable": outcome in {"failed", "partial"},
            "started_at": started_at,
            "finished_at": _utcnow_iso(),
            "outcome": outcome,
            "attempt_count": attempt_count,
        }
        stages[f"deliver:{adapter}"] = terminal_stage
        delivery_status = _aggregate_status(adapters, stages)
        finalize = getattr(watchlists_db, "finalize_briefing_attempt", None)
        if attempt is not None and callable(finalize):
            occurrence = finalize(
                int(attempt.id),
                expected_states={"sending"},
                state=outcome,
                stage_updates={
                    f"deliver:{adapter}": terminal_stage,
                    "deliver": _aggregate_stage(delivery_status),
                },
                delivery_status=delivery_status,
                code=code,
                artifact_id=str(details.get("document_id")) if details.get("document_id") is not None else None,
            ) or watchlists_db.get_briefing_occurrence(int(occurrence.id))
            stages = _read_stages(occurrence)
        else:
            _transition_attempt(
                watchlists_db,
                attempt,
                expected_states={"sending"},
                state=outcome,
                code=code,
                artifact_id=str(details.get("document_id")) if details.get("document_id") is not None else None,
            )
            occurrence = _save_stages(
                watchlists_db, occurrence, stages, changed_stage=f"deliver:{adapter}"
            )
        output = _persist_output_result(
            collections_db,
            output,
            adapter=adapter,
            outcome=outcome,
            code=code,
            details=details,
        )
        if adapter == "chatbook" and outcome == "successful" and details.get("document_id") is not None:
            output = collections_db.update_output_artifact_metadata(
                int(output.id),
                chatbook_path=f"generated_document:{details['document_id']}",
            )

    occurrence = _save_stages(watchlists_db, occurrence, stages)
    return BriefingDeliveryResult(
        occurrence_id=int(occurrence.id),
        delivery_status=str(occurrence.delivery_status),
        adapters={adapter: _adapter_outcome(stages, adapter) or "not_started" for adapter in adapters},
    )


async def deliver_briefing_for_user(
    *,
    user_id: int,
    occurrence_id: int,
    requested_adapters: set[str] | None = None,
    confirmed_unknown_adapters: set[str] | None = None,
) -> dict[str, Any]:
    """Resolve owned repositories and deliver one occurrence from Scheduler."""
    from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
    from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase

    result = await deliver_briefing_occurrence(
        occurrence_id=occurrence_id,
        watchlists_db=WatchlistsDatabase.for_user(user_id),
        collections_db=CollectionsDatabase.for_user(user_id),
        requested_adapters=requested_adapters,
        confirmed_unknown_adapters=confirmed_unknown_adapters,
    )
    return {
        "occurrence_id": result.occurrence_id,
        "delivery_status": result.delivery_status,
        "adapters": result.adapters,
    }


__all__ = [
    "BriefingArtifactsNotReadyError",
    "BriefingDeliveryResult",
    "deliver_briefing_for_user",
    "deliver_briefing_occurrence",
    "external_delivery_adapters",
    "reconcile_successful_delivery_attempt",
    "assert_audio_dependency_ready",
    "record_audio_workflow_terminal",
    "schedule_briefing_delivery",
]
