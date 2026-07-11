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
_PROGRAM_FORMATS = {
    "concise_briefing",
    "solo_update",
    "host_discussion",
    "sportscast",
    "culture_roundtable",
    "custom",
}


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


def _bounded_text(value: Any, limit: int) -> str | None:
    text = str(value or "").strip()
    return text[:limit] if text else None


def _api_download_url(value: Any) -> str | None:
    path = _bounded_text(value, 512)
    return path if path and path.startswith("/api/v1/") else None


def _public_artifact(value: Any) -> dict[str, Any] | None:
    artifact = _json_object(value)
    download_url = _api_download_url(artifact.get("download_url"))
    if not artifact or not download_url:
        return None
    return {
        key: projected
        for key, projected in {
            "artifact_id": artifact.get("artifact_id"),
            "type": _bounded_text(artifact.get("type"), 64),
            "download_url": download_url,
            "size_bytes": artifact.get("size_bytes") if isinstance(artifact.get("size_bytes"), int) else None,
            "mime_type": _bounded_text(artifact.get("mime_type"), 128),
            "title": _bounded_text(artifact.get("title"), 256),
            "speaker_id": _bounded_text(artifact.get("speaker_id"), 64),
            "voice": _bounded_text(artifact.get("voice"), 128),
        }.items()
        if projected is not None
    }


def _public_audio(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    script_artifact = _public_artifact(value.get("script_artifact"))
    final_artifact = _public_artifact(value.get("final_artifact"))
    download_url = _api_download_url(value.get("download_url"))
    if download_url is None and final_artifact is not None:
        download_url = final_artifact.get("download_url")
    return {
        key: projected
        for key, projected in {
            "run_id": value.get("run_id"),
            "task_id": _bounded_text(value.get("task_id"), 128),
            "queue_name": _bounded_text(value.get("queue_name"), 128),
            "status": _bounded_text(value.get("status"), 32) or "unknown",
            "download_url": download_url,
            "artifact_id": value.get("artifact_id"),
            "size_bytes": value.get("size_bytes") if isinstance(value.get("size_bytes"), int) else None,
            "mime_type": _bounded_text(value.get("mime_type"), 128),
            "script_artifact": script_artifact,
            "final_artifact": final_artifact,
            "audio_request_id": _bounded_text(value.get("audio_request_id"), 128),
            "workflow_run_id": value.get("workflow_run_id"),
            "stale": bool(value.get("stale")),
            "superseded_by": _bounded_text(value.get("superseded_by"), 128),
        }.items()
        if projected is not None
    }


def _public_editorial(contract: Mapping[str, Any]) -> dict[str, Any]:
    editorial = _json_object(contract.get("editorial"))
    audio = _json_object(contract.get("audio"))
    text = _json_object(contract.get("text"))
    show_name = _bounded_text(editorial.get("show_name"), 128)
    premise = _bounded_text(editorial.get("premise"), 280)
    program_format = str(editorial.get("program_format") or "concise_briefing")
    if program_format not in _PROGRAM_FORMATS:
        program_format = "concise_briefing"
    outcome_noun = "episode" if editorial.get("outcome_noun") == "episode" else "briefing"
    cast_value = _json_object(audio.get("cast"))
    speakers: list[dict[str, Any]] = []
    raw_speakers = cast_value.get("speakers")
    if isinstance(raw_speakers, list):
        for item in raw_speakers[:4]:
            speaker = _json_object(item)
            label = _bounded_text(speaker.get("label"), 128)
            if not label:
                continue
            speakers.append(
                {
                    "label": label,
                    "role": _bounded_text(speaker.get("role"), 128),
                    "voice": _bounded_text(speaker.get("voice") or speaker.get("synthetic_voice"), 128),
                    "synthetic": bool(speaker.get("synthetic", True)),
                }
            )
    target_minutes = audio.get("target_minutes")
    try:
        target_minutes = max(1, min(60, int(target_minutes)))
    except (TypeError, ValueError):
        target_minutes = None
    return {
        "program_format": program_format,
        "outcome_noun": outcome_noun,
        "show_name": show_name,
        "show_identity": {"name": show_name, "premise": premise},
        "show_notes": bool(text.get("show_notes")),
        "target_minutes": target_minutes,
        "cast": {
            "speaker_count": len(speakers),
            "speakers": speakers,
        } if speakers else None,
    }


def _public_delivery(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    delivery = _json_object(contract.get("delivery"))
    summaries: dict[str, dict[str, Any]] = {}
    email = _json_object(delivery.get("email"))
    recipients = email.get("recipients")
    recipient_count = min(1000, len(recipients)) if isinstance(recipients, list) else 0
    if bool(email.get("enabled")) or recipient_count:
        summaries["email"] = {
            "adapter": "email",
            "recipient_count": recipient_count,
            "masked_label": f"{recipient_count} recipient" if recipient_count == 1 else f"{recipient_count} recipients",
        }
    chatbook = _json_object(delivery.get("chatbook"))
    if bool(chatbook.get("enabled")):
        summaries["chatbook"] = {
            "adapter": "chatbook",
            "recipient_count": 1,
            "masked_label": "Chatbook",
        }
    return summaries


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
    public_metadata: dict[str, Any] = {}
    for key in (
        "candidate_count",
        "included_count",
        "omitted_count",
        "selected_count",
    ):
        if isinstance(metadata.get(key), int):
            public_metadata[key] = max(0, int(metadata[key]))
    for key in ("ai_generated_speech", "no_material_updates"):
        if isinstance(metadata.get(key), bool):
            public_metadata[key] = metadata[key]
    for key, limit in (
        ("checked_at", 64),
        ("outcome_noun", 16),
        ("program_format", 64),
        ("show_name", 128),
        ("speech_disclosure", 256),
    ):
        value = _bounded_text(metadata.get(key), limit)
        if value is not None:
            public_metadata[key] = value
    provenance = metadata.get("provenance")
    if isinstance(provenance, list):
        public_metadata["provenance"] = [
            {"source_id": item.get("source_id")}
            for item in provenance[:1000]
            if isinstance(item, Mapping) and item.get("source_id") is not None
        ]
    source_counts = metadata.get("source_counts")
    if isinstance(source_counts, Mapping):
        public_metadata["source_counts"] = {
            str(key)[:64]: max(0, int(value))
            for key, value in list(source_counts.items())[:100]
            if isinstance(value, int)
        }
    return {
        "id": int(row.id),
        "run_id": int(row.run_id or occurrence.run_id),
        "job_id": int(row.job_id or occurrence.job_id),
        "type": row.type,
        "format": row.format,
        "title": _bounded_text(getattr(row, "title", None), 256),
        "created_at": row.created_at,
        "download_url": f"/api/v1/watchlists/outputs/{int(row.id)}/download",
        "metadata": public_metadata,
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
        failed_candidates = _AUDIO_STAGES
        if audio.get("script_artifact"):
            for ready_name in ("compose_audio_script", "persist_audio_script"):
                stages[ready_name] = {
                    **stages.get(ready_name, {}),
                    "status": "ready",
                    "code": None,
                    "retryable": False,
                }
            failed_candidates = ("generate_audio", "persist_audio")
        for name in failed_candidates:
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
    editorial = _public_editorial(contract)
    return {
        "occurrence_id": int(occurrence.id),
        "run_id": int(occurrence.run_id),
        "job_id": int(occurrence.job_id),
        "artifact_status": artifact_status,
        "delivery_status": delivery_status,
        "stages": stages,
        "output": output,
        "audio": _public_audio(audio),
        "editorial": editorial,
        "delivery": _public_delivery(contract),
        "selection": {
            "candidate_count": candidate_count,
            "included_count": int(occurrence.selected_count or 0),
            "omitted_count": int(occurrence.omitted_count or 0),
        },
        "next_run_at": getattr(job, "next_run_at", None),
        "timezone": str(getattr(job, "schedule_timezone", None) or "UTC"),
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
