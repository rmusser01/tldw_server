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


def _save_stages(watchlists_db: Any, occurrence: Any, stages: dict[str, dict[str, Any]]) -> Any:
    delivery_status = _aggregate_status(
        external_delivery_adapters(_json_object(occurrence.contract_json)),
        stages,
    )
    aggregate_stage_status = {
        "not_configured": "skipped",
        "waiting_for_artifacts": "not_started",
        "delivering": "running",
        "delivered": "ready",
        "partially_delivered": "failed",
        "failed": "failed",
        "unknown": "failed",
    }[delivery_status]
    stages["deliver"] = {
        "status": aggregate_stage_status,
        "code": None if delivery_status == "delivered" else delivery_status,
        "retryable": delivery_status in {"failed", "partially_delivered"},
        "finished_at": (
            _utcnow_iso()
            if delivery_status in {"not_configured", "delivered", "partially_delivered", "failed", "unknown"}
            else None
        ),
    }
    return watchlists_db.update_briefing_occurrence(
        int(occurrence.id),
        stages=stages,
        delivery_status=delivery_status,
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
    if requested:
        attempt_keys = [f"{adapter}-{_attempt_count(stages.get(f'deliver:{adapter}'))}" for adapter in requested]
        task_key = f"{task_key}:retry:{','.join(attempt_keys)}"
    elif audio_task_id and occurrence.delivery_task_id:
        previous_dependency = stages.get("deliver", {}).get("audio_dependency_task_id")
        if str(previous_dependency or "") != str(audio_task_id):
            task_key = f"{task_key}:audio:{audio_task_id}"
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
    if watchlists_db is not None:
        watchlists_db.update_briefing_occurrence(
            int(occurrence.id),
            delivery_task_id=str(task_id),
        )
    return str(task_id)


def mark_audio_dependency_ready(
    *,
    user_id: int,
    occurrence_id: int,
    audio_task_id: str,
) -> None:
    """Project a successfully completed Scheduler audio dependency into occurrence state."""
    from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase

    watchlists_db = WatchlistsDatabase.for_user(user_id)
    occurrence = watchlists_db.get_briefing_occurrence(occurrence_id)
    if not occurrence.audio_task_id or str(occurrence.audio_task_id) != str(audio_task_id):
        raise BriefingArtifactsNotReadyError("audio_dependency_mismatch")
    contract = _json_object(occurrence.contract_json)
    if not bool(contract.get("audio", {}).get("enabled")):
        raise BriefingArtifactsNotReadyError("audio_not_selected")
    stages = _read_stages(occurrence)
    now = _utcnow_iso()
    for name in ("compose_audio_script", "persist_audio_script", "generate_audio", "persist_audio"):
        stage = stages.get(name, {})
        stages[name] = {
            **stage,
            "status": "ready",
            "code": None,
            "retryable": False,
            "started_at": stage.get("started_at"),
            "finished_at": now,
        }
    watchlists_db.update_briefing_occurrence(
        occurrence_id,
        stages=stages,
        artifact_status="ready",
    )


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
        previous_outcome = _adapter_outcome(stages, adapter)
        if previous_outcome == "successful":
            continue
        if previous_outcome == "unknown" and adapter not in confirmed:
            continue
        if previous_outcome == "partial" and adapter not in requested:
            continue
        if requested and adapter not in requested:
            continue

        started_at = _utcnow_iso()
        attempt_count = _attempt_count(stages.get(f"deliver:{adapter}")) + 1
        stages[f"deliver:{adapter}"] = {
            "status": "running",
            "code": None,
            "retryable": False,
            "started_at": started_at,
            "finished_at": None,
            "outcome": "sending",
            "attempt_count": attempt_count,
        }
        occurrence = _save_stages(watchlists_db, occurrence, stages)
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
        except (asyncio.TimeoutError, TimeoutError):
            outcome, code, details = "unknown", "delivery_outcome_unknown", {"error_type": "TimeoutError"}
        except Exception as exc:  # noqa: BLE001 - provider boundary persists a safe failure code
            outcome, code, details = "failed", f"{adapter}_delivery_failed", {"error_type": type(exc).__name__}

        stages[f"deliver:{adapter}"] = {
            "status": "ready" if outcome == "successful" else "failed",
            "code": code,
            "retryable": outcome in {"failed", "partial"},
            "started_at": started_at,
            "finished_at": _utcnow_iso(),
            "outcome": outcome,
            "attempt_count": attempt_count,
        }
        occurrence = _save_stages(watchlists_db, occurrence, stages)
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
    "mark_audio_dependency_ready",
    "schedule_briefing_delivery",
]
