"""Projection helpers for durable Watchlists audio briefing artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from urllib.parse import quote

from loguru import logger

_PROJECTION_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    UnicodeError,
    ValueError,
    json.JSONDecodeError,
)

_CORRELATION_KEYS = ("source", "watchlist_job_id", "watchlist_run_id", "audio_request_id")


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from dict-like and object-like rows without raising."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _json_object(value: Any) -> dict[str, Any]:
    """Coerce a JSON object payload to a dict, returning empty dict on bad input."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except _PROJECTION_NONCRITICAL_EXCEPTIONS:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _json_dumps(value: dict[str, Any]) -> str:
    """Serialize metadata deterministically for idempotent update comparisons."""
    return json.dumps(value, sort_keys=True)


def _same_json_object(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Compare JSON-compatible dicts using the same ordering used for persistence."""
    return _json_dumps(left) == _json_dumps(right)


def _first_non_empty_string(*values: Any) -> str | None:
    """Return the first scalar value that can be represented as non-empty text."""
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list, tuple, set)):
            text = str(value).strip()
            if text:
                return text
    return None


def normalize_audio_status(status: Any, *, task_id: Any = None) -> str:
    """Normalize Workflow/Scheduler status names into Watchlists audio status names."""
    raw = str(status or "").strip().lower()
    if raw in {"succeeded", "success", "done"}:
        return "completed"
    if raw in {"queued", "submitted", "scheduled"}:
        return "queued"
    if raw in {"running", "in_progress", "processing", "started"}:
        return "running"
    if raw in {"failed", "error", "errored"}:
        return "failed"
    if raw in {"cancelled", "canceled"}:
        return "cancelled"
    if raw in {"completed", "pending", "unknown"}:
        return raw
    return "pending" if task_id else "unknown"


def artifact_download_url(artifact_id: Any, *, target_user_id: int | None = None) -> str | None:  # noqa: ARG001
    """Build the canonical Workflows artifact download URL for a known artifact ID."""
    if artifact_id is None:
        return None
    artifact_id_str = str(artifact_id).strip()
    if not artifact_id_str:
        return None
    return f"/api/v1/workflows/artifacts/{quote(artifact_id_str, safe='')}/download"


def _artifact_metadata(artifact: Any) -> dict[str, Any]:
    """Read artifact metadata from current and compatibility artifact row fields."""
    metadata = _json_object(_get_value(artifact, "metadata_json"))
    if not metadata:
        metadata = _json_object(_get_value(artifact, "metadata"))
    return metadata


def _artifact_id(artifact: Any) -> Any:
    """Return the artifact identifier across Workflow artifact row shapes."""
    return _get_value(artifact, "artifact_id") or _get_value(artifact, "id")


def _scrub_artifact_metadata(value: Any) -> Any:
    """Remove raw artifact URI fields from metadata before mirroring to Watchlists."""
    if isinstance(value, dict):
        return {key: _scrub_artifact_metadata(item) for key, item in value.items() if key != "uri"}
    if isinstance(value, list):
        return [_scrub_artifact_metadata(item) for item in value]
    return value


def summarize_audio_artifact(
    artifact: Any,
    *,
    metadata: dict[str, Any] | None = None,
    fallback_title: str,
    mime_type: str | None = None,
) -> dict[str, Any]:
    """Return a mirrored-safe artifact summary without raw file URIs."""
    art_meta = _scrub_artifact_metadata(dict(metadata or _artifact_metadata(artifact)))
    artifact_id = _artifact_id(artifact)
    title = _first_non_empty_string(
        art_meta.get("title"),
        art_meta.get("label"),
        art_meta.get("name"),
        fallback_title,
    )
    summary: dict[str, Any] = {
        "artifact_id": artifact_id,
        "type": _get_value(artifact, "type"),
        "download_url": artifact_download_url(artifact_id),
        "size_bytes": _get_value(artifact, "size_bytes"),
        "mime_type": mime_type or _get_value(artifact, "mime_type"),
        "metadata": art_meta,
    }
    if title:
        summary["title"] = title
    speaker_id = _first_non_empty_string(
        art_meta.get("speaker_id"),
        art_meta.get("speakerId"),
        art_meta.get("voice_marker"),
    )
    if speaker_id:
        summary["speaker_id"] = speaker_id
    voice = _first_non_empty_string(art_meta.get("voice"), art_meta.get("tts_voice"))
    if voice:
        summary["voice"] = voice
    return {key: value for key, value in summary.items() if value is not None}


def _created_at_rank(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return 0.0
        normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        try:
            return datetime.fromisoformat(normalized).timestamp()
        except _PROJECTION_NONCRITICAL_EXCEPTIONS:
            try:
                return float(raw)
            except _PROJECTION_NONCRITICAL_EXCEPTIONS:
                return 0.0
    return 0.0


def _artifact_id_rank(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        digits = "".join(char for char in value if char.isdigit())
        if digits:
            try:
                return int(digits)
            except _PROJECTION_NONCRITICAL_EXCEPTIONS:
                return 0
    return 0


def _artifact_matches_run(metadata: dict[str, Any], *, run_id: int, audio_request_id: str | None) -> bool:
    metadata_run_id = metadata.get("watchlist_run_id")
    if metadata_run_id is not None and str(metadata_run_id) != str(run_id):
        return False
    source = metadata.get("source")
    if source is not None and source != "watchlist_audio_briefing":
        return False
    if audio_request_id:
        metadata_request_id = _first_non_empty_string(metadata.get("audio_request_id"))
        if metadata_request_id != audio_request_id:
            return False
    return True


def extract_workflow_run_metadata(workflow_run: Any) -> dict[str, Any]:
    """Extract Watchlists correlation metadata from current and compatibility fields."""
    candidates: list[dict[str, Any]] = []
    candidates.append(_json_object(_get_value(workflow_run, "metadata_json")))

    definition_snapshot = _json_object(_get_value(workflow_run, "definition_snapshot_json"))
    definition_metadata = definition_snapshot.get("metadata") if isinstance(definition_snapshot, dict) else None
    candidates.append(definition_metadata if isinstance(definition_metadata, dict) else {})

    candidates.append(_json_object(_get_value(workflow_run, "inputs_json")))

    merged: dict[str, Any] = {}
    for candidate in candidates:
        extracted = {key: candidate[key] for key in _CORRELATION_KEYS if candidate.get(key) is not None}
        for key, value in extracted.items():
            merged.setdefault(key, value)
        for optional_key in ("fallback_reason", "audio_fallback_reason", "fallback_error"):
            if candidate.get(optional_key) is not None:
                merged.setdefault(optional_key, candidate[optional_key])
    return merged


def _workflow_run_id(workflow_run: Any) -> Any:
    return _get_value(workflow_run, "run_id") or _get_value(workflow_run, "id")


def build_audio_projection(
    *,
    run_id: int,
    task_id: Any,
    audio_request_id: str | None,
    workflow_run: Any,
    artifacts: list[Any],
) -> dict[str, Any]:
    """Build a compact durable Watchlists audio graph from Workflow artifacts."""
    workflow_metadata = extract_workflow_run_metadata(workflow_run)
    active_request_id = audio_request_id or _first_non_empty_string(workflow_metadata.get("audio_request_id"))
    status = normalize_audio_status(_get_value(workflow_run, "status"), task_id=task_id)

    script_candidates: list[tuple[tuple[int, float, int, int], dict[str, Any]]] = []
    speaker_artifacts: list[dict[str, Any]] = []
    audio_candidates: list[tuple[tuple[int, float, int, int], dict[str, Any]]] = []
    fallback_reason = _first_non_empty_string(
        workflow_metadata.get("fallback_reason"),
        workflow_metadata.get("audio_fallback_reason"),
        workflow_metadata.get("fallback_error"),
    )

    for idx, artifact in enumerate(artifacts or []):
        metadata = _artifact_metadata(artifact)
        if not _artifact_matches_run(metadata, run_id=run_id, audio_request_id=active_request_id):
            continue

        artifact_id = _artifact_id(artifact)
        request_rank = 1 if active_request_id and metadata.get("audio_request_id") == active_request_id else 0
        rank_suffix = (
            _created_at_rank(_get_value(artifact, "created_at")),
            idx,
            _artifact_id_rank(artifact_id),
        )
        artifact_type = _get_value(artifact, "type")

        if (
            metadata.get("script_artifact")
            or metadata.get("audio_briefing_script")
            or artifact_type in {"audio_script", "briefing_script", "script"}
        ):
            script_candidates.append(
                (
                    (request_rank, *rank_suffix),
                    summarize_audio_artifact(
                        artifact,
                        metadata=metadata,
                        fallback_title="Briefing script",
                        mime_type=_get_value(artifact, "mime_type") or "text/markdown",
                    ),
                )
            )
            continue

        if artifact_type == "tts_audio" or metadata.get("multi_voice"):
            is_speaker_artifact = bool(metadata.get("speaker_artifact") or metadata.get("speaker_id"))
            final_hint = bool(metadata.get("final_artifact") or metadata.get("is_final") or metadata.get("final"))
            background_hint = bool(metadata.get("background_mixed") or metadata.get("mixed"))
            fallback_hint = bool(
                metadata.get("fallback_artifact") or metadata.get("single_voice_fallback") or metadata.get("fallback")
            )
            if fallback_reason is None:
                fallback_reason = _first_non_empty_string(metadata.get("fallback_reason"), metadata.get("fallback_error"))

            if is_speaker_artifact:
                speaker_artifacts.append(
                    summarize_audio_artifact(
                        artifact,
                        metadata=metadata,
                        fallback_title=f"Speaker {len(speaker_artifacts) + 1}",
                        mime_type=_get_value(artifact, "mime_type") or "audio/mpeg",
                    )
                )
                if not (final_hint or background_hint or fallback_hint):
                    continue

            candidate_priority = 30 if background_hint else 20 if final_hint else 10 if fallback_hint else 0
            audio_candidates.append(
                (
                    (request_rank, candidate_priority, *rank_suffix),
                    summarize_audio_artifact(
                        artifact,
                        metadata=metadata,
                        fallback_title="Final audio" if candidate_priority else "Audio artifact",
                        mime_type=_get_value(artifact, "mime_type") or "audio/mpeg",
                    ),
                )
            )

    script_artifact = max(script_candidates, key=lambda item: item[0])[1] if script_candidates else None
    final_artifact = max(audio_candidates, key=lambda item: item[0])[1] if audio_candidates else None

    if final_artifact and status not in {"failed", "cancelled"}:
        status = "completed"

    projection: dict[str, Any] = {
        "run_id": run_id,
        "task_id": str(task_id) if task_id is not None else None,
        "status": status,
        "workflow_run_id": _workflow_run_id(workflow_run),
        "audio_request_id": active_request_id,
        "script_artifact": script_artifact,
        "speaker_artifacts": speaker_artifacts,
        "final_artifact": final_artifact,
        "fallback_reason": fallback_reason,
    }
    if final_artifact:
        projection.update(
            {
                "artifact_id": final_artifact.get("artifact_id"),
                "download_url": final_artifact.get("download_url"),
                "size_bytes": final_artifact.get("size_bytes"),
                "mime_type": final_artifact.get("mime_type"),
            }
        )
    return projection


def merge_audio_projection_metadata(existing: dict[str, Any], projection: dict[str, Any]) -> dict[str, Any]:
    """Merge a durable audio projection into run/output metadata without dropping unrelated fields."""
    merged = dict(existing or {})
    merged["audio"] = dict(projection)
    merged["audio_briefing_status"] = projection.get("status")
    if projection.get("audio_request_id"):
        merged["audio_request_id"] = projection.get("audio_request_id")
    else:
        merged.pop("audio_request_id", None)
    if projection.get("task_id"):
        merged["audio_briefing_task_id"] = projection.get("task_id")
    return merged


def mark_audio_projection_stale(existing: dict[str, Any], *, superseded_by: str | None) -> dict[str, Any]:
    """Move the active mirrored audio graph aside so retries cannot present it as current."""
    updated = dict(existing or {})
    active_audio = updated.pop("audio", None)
    if isinstance(active_audio, dict):
        previous_audio = dict(active_audio)
        previous_audio["stale"] = True
        if superseded_by:
            previous_audio["superseded_by"] = superseded_by
        updated["previous_audio"] = previous_audio
    return updated


def _run_stats(run: Any) -> dict[str, Any]:
    return _json_object(_get_value(run, "stats_json"))


def get_mirrored_audio_projection(run: Any) -> dict[str, Any] | None:
    """Return the previously mirrored audio graph from Watchlists run stats."""
    audio = _run_stats(run).get("audio")
    return audio if isinstance(audio, dict) else None


def find_matching_workflow_run(
    workflow_db: Any,
    *,
    tenant_id: str,
    user_id: str,
    job_id: int | str | None = None,
    run_id: int,
    audio_request_id: str | None,
) -> Any | None:
    """Find the Workflow run that belongs to a Watchlists run/request."""
    idempotency_key = _watchlist_audio_idempotency_key(
        user_id=user_id,
        job_id=job_id,
        run_id=run_id,
        audio_request_id=audio_request_id,
    )
    if idempotency_key:
        try:
            lookup = getattr(workflow_db, "get_run_by_idempotency", None)
            lookup_run = lookup(tenant_id, user_id, idempotency_key) if callable(lookup) else None
        except _PROJECTION_NONCRITICAL_EXCEPTIONS:
            lookup_run = None
        if lookup_run is not None:
            metadata = extract_workflow_run_metadata(lookup_run)
            stored_key = _first_non_empty_string(_get_value(lookup_run, "idempotency_key"))
            candidate_request_id = _first_non_empty_string(metadata.get("audio_request_id"))
            if stored_key == idempotency_key:
                return lookup_run
            if str(metadata.get("watchlist_run_id")) == str(run_id) and candidate_request_id == audio_request_id:
                return lookup_run

    fallback_run = None
    page_size = 50
    offset = 0
    while True:
        try:
            runs = workflow_db.list_runs(tenant_id=tenant_id, user_id=user_id, limit=page_size, offset=offset)
        except TypeError:
            runs = workflow_db.list_runs(limit=page_size, offset=offset)
        if not runs:
            break
        for workflow_run in runs:
            metadata = extract_workflow_run_metadata(workflow_run)
            if str(metadata.get("watchlist_run_id")) != str(run_id):
                continue
            candidate_request_id = _first_non_empty_string(metadata.get("audio_request_id"))
            if audio_request_id and candidate_request_id == audio_request_id:
                return workflow_run
            if audio_request_id:
                continue
            if fallback_run is None:
                fallback_run = workflow_run
        if len(runs) < page_size:
            break
        offset += page_size
    return fallback_run


def _watchlist_audio_idempotency_key(
    *,
    user_id: str,
    job_id: int | str | None,
    run_id: int,
    audio_request_id: str | None,
) -> str | None:
    """Build the Watchlists audio Workflow idempotency key when all parts are known."""
    request_id = _first_non_empty_string(audio_request_id)
    job_id_text = _first_non_empty_string(job_id)
    run_id_text = _first_non_empty_string(run_id)
    user_id_text = _first_non_empty_string(user_id)
    if not request_id or not job_id_text or not run_id_text or not user_id_text:
        return None
    return f"watchlist-audio-briefing:{user_id_text}:{job_id_text}:{run_id_text}:{request_id}"


def find_canonical_watchlist_output(
    collections_db: Any,
    run_id: int,
    audio_request_id: str | None = None,
) -> Any | None:
    """Find the base Watchlists output whose metadata should mirror audio state."""
    try:
        rows, _ = collections_db.list_output_artifacts(run_id=run_id, limit=50, offset=0)
    except _PROJECTION_NONCRITICAL_EXCEPTIONS:
        return None
    fallback_row = None
    for row in rows or []:
        metadata = _json_object(_get_value(row, "metadata_json"))
        row_type = str(_get_value(row, "type") or "").lower()
        if row_type in {"tts_audio", "audio", "audio_briefing"}:
            continue
        candidate_request_id = _first_non_empty_string(
            metadata.get("audio_request_id"),
            metadata.get("audio", {}).get("audio_request_id") if isinstance(metadata.get("audio"), dict) else None,
        )
        if audio_request_id and candidate_request_id == audio_request_id:
            return row
        if audio_request_id and candidate_request_id and candidate_request_id != audio_request_id:
            continue
        if fallback_row is None:
            fallback_row = row
    return fallback_row


def mirror_audio_projection(
    run_db: Any,
    collections_db: Any,
    run: Any,
    projection: dict[str, Any],
    *,
    user_id: int,  # noqa: ARG001
) -> bool:
    """Mirror a canonical Workflow audio projection into Watchlists run/output metadata."""
    try:
        run_id = _get_value(run, "id")
        run_stats = _run_stats(run)
        merged_run_stats = merge_audio_projection_metadata(run_stats, projection)
        if not _same_json_object(run_stats, merged_run_stats):
            run_db.update_run(run_id, stats_json=_json_dumps(merged_run_stats))

        output = find_canonical_watchlist_output(
            collections_db,
            int(projection.get("run_id") or run_id),
            audio_request_id=_first_non_empty_string(projection.get("audio_request_id")),
        )
        if output is not None:
            output_metadata = _json_object(_get_value(output, "metadata_json"))
            merged_output_metadata = merge_audio_projection_metadata(output_metadata, projection)
            if not _same_json_object(output_metadata, merged_output_metadata):
                collections_db.update_output_artifact_metadata(
                    _get_value(output, "id"),
                    metadata_json=_json_dumps(merged_output_metadata),
                )
        return True
    except _PROJECTION_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning("Watchlists audio projection mirror failed (error_type={})", type(exc).__name__)
        return False
