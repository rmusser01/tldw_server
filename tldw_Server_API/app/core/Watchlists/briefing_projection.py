"""Effective read model for durable Watchlists briefing occurrences."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.services.outputs_service import _resolve_output_path_for_user

_AUDIO_STAGES = (
    "compose_audio_script",
    "persist_audio_script",
    "generate_audio",
    "persist_audio",
)
_VALID_STAGE_STATUSES = {"not_started", "queued", "running", "ready", "failed", "skipped", "cancelled"}


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


def _stages(occurrence: Any) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    raw = _json_object(getattr(occurrence, "stages_json", None))
    stages: dict[str, dict[str, Any]] = {}
    for name, value in raw.items():
        if not isinstance(value, Mapping):
            continue
        status = str(value.get("status") or "not_started")
        stages[str(name)] = {
            "status": status if status in _VALID_STAGE_STATUSES else "not_started",
            "code": str(value["code"]) if value.get("code") is not None else None,
            "retryable": bool(value.get("retryable")),
            "started_at": value.get("started_at"),
            "finished_at": value.get("finished_at"),
            "outcome": value.get("outcome"),
            **{
                key: value[key]
                for key in (
                    "artifact_id",
                    "artifact_version",
                    "attempt_count",
                    "audio_request_id",
                    "scheduler_task_id",
                    "task_id",
                    "workflow_run_id",
                )
                if value.get(key) is not None
            },
        }
    return stages, raw


def _output(collections_db: Any, occurrence: Any) -> tuple[dict[str, Any] | None, str | None]:
    if occurrence.output_id is None:
        return None, None
    try:
        row = collections_db.get_output_artifact(int(occurrence.output_id))
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None, "briefing_text_artifact_missing"
    metadata = _json_object(getattr(row, "metadata_json", None))
    if (
        (getattr(row, "user_id", None) is not None and str(row.user_id) != str(occurrence.user_id))
        or int(getattr(row, "run_id", 0) or 0) != int(occurrence.run_id)
        or int(getattr(row, "job_id", 0) or 0) != int(occurrence.job_id)
        or str(metadata.get("occurrence_id")) != str(occurrence.id)
    ):
        return None, "briefing_output_ownership_mismatch"
    storage_path = str(getattr(row, "storage_path", None) or "").strip()
    if not storage_path:
        return None, "briefing_text_artifact_missing"
    try:
        if not _resolve_output_path_for_user(int(occurrence.user_id), storage_path).is_file():
            return None, "briefing_text_artifact_missing"
    except (OSError, RuntimeError, TypeError, ValueError):
        return None, "briefing_text_artifact_missing"
    public_keys = {
        "ai_generated_speech",
        "candidate_count",
        "checked_at",
        "included_count",
        "no_material_updates",
        "omitted_count",
        "outcome_noun",
        "program_format",
        "provenance",
        "selected_count",
        "show_identity",
        "show_name",
        "source_counts",
        "speech_disclosure",
    }
    return {
        "id": int(row.id),
        "run_id": int(row.run_id or occurrence.run_id),
        "job_id": int(row.job_id or occurrence.job_id),
        "type": row.type,
        "format": row.format,
        "title": getattr(row, "title", None),
        "created_at": row.created_at,
        "download_url": f"/api/v1/watchlists/outputs/{int(row.id)}/download",
        "metadata": {key: metadata[key] for key in public_keys if key in metadata},
    }, None


def _with_audio(
    stages: dict[str, dict[str, Any]],
    audio: Mapping[str, Any] | None,
    artifact_status: str,
) -> tuple[dict[str, dict[str, Any]], str]:
    if audio is None:
        return stages, artifact_status
    status = str(audio.get("status") or "unknown").lower()
    if status == "completed" and audio.get("final_artifact"):
        for name in _AUDIO_STAGES:
            stages[name] = {**stages.get(name, {}), "status": "ready", "code": None, "retryable": False}
        return stages, artifact_status
    if status in {"failed", "dead"}:
        for name in _AUDIO_STAGES:
            existing = stages.get(name, {})
            if existing.get("status") not in {"ready", "skipped"}:
                stages[name] = {
                    **existing,
                    "status": "failed",
                    "code": str(audio.get("fallback_reason") or "audio_generation_failed"),
                    "retryable": True,
                }
                break
        return stages, "failed"
    if status in {"cancelled", "canceled"}:
        for name in _AUDIO_STAGES:
            existing = stages.get(name, {})
            if existing.get("status") not in {"ready", "skipped"}:
                stages[name] = {**existing, "status": "cancelled", "retryable": True}
        return stages, "cancelled"
    return stages, artifact_status


def _aggregate_artifact_status(
    stages: Mapping[str, Mapping[str, Any]],
    *,
    audio_enabled: bool,
) -> str:
    required = ["persist_text"]
    if audio_enabled:
        required.extend(_AUDIO_STAGES)
    statuses = [str(stages.get(name, {}).get("status") or "not_started") for name in required]
    if any(status == "failed" for status in statuses):
        return "failed"
    if any(status == "cancelled" for status in statuses):
        return "cancelled"
    if statuses and all(status in {"ready", "skipped"} for status in statuses):
        return "ready"
    return "running"


def build_briefing_projection(
    *,
    occurrence: Any,
    watchlists_db: Any,
    collections_db: Any,
    audio: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine authoritative occurrence state with current output/audio projections."""
    contract = _json_object(occurrence.contract_json)
    stages, raw_stages = _stages(occurrence)
    stages, _artifact_status = _with_audio(stages, audio, str(occurrence.artifact_status))
    output, output_error = _output(collections_db, occurrence)
    if output_error:
        failed_stage = {
            **stages.get("persist_text", {}),
            "status": "failed",
            "code": output_error,
            "retryable": True,
        }
        stages["persist_text"] = failed_stage
    artifact_status = _aggregate_artifact_status(
        stages,
        audio_enabled=bool(contract.get("audio", {}).get("enabled")),
    )
    select = raw_stages.get("select") if isinstance(raw_stages.get("select"), Mapping) else {}
    candidate_count = int(
        select.get("candidate_count") or int(occurrence.selected_count or 0) + int(occurrence.omitted_count or 0)
    )
    delivery_stages = [stage for name, stage in stages.items() if name.startswith("deliver:")]
    delivery_status = str(occurrence.delivery_status)
    job = watchlists_db.get_job(int(occurrence.job_id))
    return {
        "occurrence_id": int(occurrence.id),
        "run_id": int(occurrence.run_id),
        "job_id": int(occurrence.job_id),
        "artifact_status": artifact_status,
        "delivery_status": delivery_status,
        "stages": stages,
        "output": output,
        "audio": dict(audio) if audio is not None else None,
        "editorial": dict(contract.get("editorial") or {}),
        "selection": {
            "candidate_count": candidate_count,
            "included_count": int(occurrence.selected_count or 0),
            "omitted_count": int(occurrence.omitted_count or 0),
        },
        "next_run_at": getattr(job, "next_run_at", None),
        "recovery": {
            "can_open_report": output is not None,
            "can_retry_text": any(
                stages.get(name, {}).get("status") == "failed" for name in ("render_text", "persist_text")
            ),
            "can_retry_audio": any(stages.get(name, {}).get("status") == "failed" for name in _AUDIO_STAGES),
            "can_regenerate_audio": bool(contract.get("audio", {}).get("enabled")),
            "can_retry_delivery": delivery_status in {"failed", "partially_delivered", "unknown"}
            or any(stage.get("status") == "failed" for stage in delivery_stages),
            "requires_unknown_delivery_confirmation": delivery_status == "unknown"
            or any(stage.get("outcome") == "unknown" for stage in delivery_stages),
        },
    }


__all__ = ["build_briefing_projection"]
