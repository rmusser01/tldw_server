"""Read-computed status projection for Research Workspace sources."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db import api as media_db_api
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError


_ACTIVE_JOB_STATUSES = frozenset({"queued", "processing", "running", "retrying"})
_FAILED_JOB_STATUSES = frozenset({"failed", "cancelled", "quarantined"})
_PROCESSING_STATES = frozenset({"queued", "ingesting", "extracting", "chunking", "indexing", "retrying"})
_PENDING_REASONS = frozenset({"vector_index_pending", "chunking_pending", "extraction_pending"})


def build_source_status_projection(
    *,
    workspace_id: str,
    sources: list[dict[str, Any]],
    media_db: Any | None = None,
    jobs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the source status response payload for a workspace.

    The projection is intentionally computed on read for this first contract:
    workspace membership is stored in ChaChaNotes, text/indexing readiness lives
    in the Media DB, and in-flight ingestion progress belongs to Jobs.
    """
    job_index = _build_job_index(jobs or [])
    statuses = [
        _build_source_status(source, media_db=media_db, job_index=job_index)
        for source in sources
    ]
    return {
        "workspace_id": workspace_id,
        "sources": statuses,
        "summary": _summarize_sources(statuses),
    }


def build_workspace_capability_projection(
    *,
    workspace: dict[str, Any],
    status_projection: dict[str, Any],
) -> dict[str, Any]:
    """Build conservative capability gates for Research Workspace clients."""
    summary = dict(status_projection.get("summary") or {})
    has_queryable_sources = int(summary.get("queryable") or 0) > 0
    access_level = _normalize_access_level(workspace.get("access_level"))
    can_modify_workspace = access_level in {"owner", "editor"}

    return {
        "workspace_id": workspace["id"],
        "workspace_kind": "research_workspace",
        "access_level": access_level,
        "source_summary": summary,
        "workspace_services": {
            "migration": {
                "state": "available",
                "reason_code": None,
                "management_surface": "research_workspace_import",
            },
            "sharing": {
                "state": "private",
                "reason_code": None,
                "management_surface": "shared_workspaces",
            },
            "mcp": {
                "state": "not_configured",
                "reason_code": "no_workspace_mcp_binding",
                "management_surface": "mcp_hub",
            },
            "acp": {
                "state": "not_configured",
                "reason_code": "no_workspace_acp_binding",
                "management_surface": "acp_workspace",
            },
            "sandbox": {
                "state": "not_configured",
                "reason_code": "no_workspace_sandbox_binding",
                "management_surface": "sandbox_settings",
            },
            "provider": {
                "state": "unknown",
                "reason_code": "provider_not_evaluated",
                "management_surface": "model_settings",
            },
        },
        "allowed_actions": {
            "add_sources": _allowed(can_modify_workspace, "insufficient_access"),
            "inspect_sources": _allowed(
                int(summary.get("total") or 0) > 0,
                "no_sources",
            ),
            "ask_grounded_questions": _allowed(
                has_queryable_sources,
                "no_queryable_sources",
            ),
            "export_workspace": _allowed(can_modify_workspace, "insufficient_access"),
            "manage_tools": _allowed(can_modify_workspace, "insufficient_access"),
            "run_mcp_tools": _allowed(False, "mcp_not_configured"),
            "use_acp_agents": _allowed(False, "acp_not_configured"),
            "use_sandbox": _allowed(False, "sandbox_not_configured"),
        },
    }


def _build_source_status(
    source: dict[str, Any],
    *,
    media_db: Any | None,
    job_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    matched_job = _match_job_for_source(source, job_index)
    if matched_job and _is_active_job(matched_job):
        return _status_from_active_job(source, matched_job)

    if matched_job and _is_failed_job(matched_job):
        return _status_from_failed_job(source, matched_job)

    media_id = _coerce_int(source.get("media_id"))
    if media_id is None or media_id <= 0:
        return _base_status(
            source,
            state="missing_media",
            reason="media_id_missing",
            progress_percent=0.0,
            progress_message="Workspace source does not have a media item yet.",
        )

    media = _get_media(media_db, media_id)
    if media is None:
        return _base_status(
            source,
            state="missing_media",
            reason="media_not_found",
            progress_percent=0.0,
            progress_message="Media item is missing or unavailable.",
        )

    return _status_from_media(source, media, media_db=media_db, media_id=media_id)


def _status_from_media(
    source: dict[str, Any],
    media: dict[str, Any],
    *,
    media_db: Any | None,
    media_id: int,
) -> dict[str, Any]:
    content = str(media.get("content") or "")
    text_ready = bool(content and not content.isspace())
    chunking_status = str(media.get("chunking_status") or "").strip().lower()
    vector_processing = media.get("vector_processing")
    vector_ready = _is_vector_ready(vector_processing)
    unvectorized_chunks = _has_unvectorized_chunks(media_db, media_id)
    vector_failed = _is_vector_failed(vector_processing) or _is_failure_status(chunking_status)

    readiness = {
        "metadata_ready": True,
        "text_extracted": text_ready,
        "fts_ready": text_ready,
        "vector_ready": vector_ready and not unvectorized_chunks,
        "citation_ready": text_ready,
        "summary_ready": bool(media.get("summary") or media.get("analysis") or media.get("analysis_content")),
        "tool_accessible": True,
    }

    if not text_ready:
        return _base_status(
            source,
            state="extracting",
            reason="extraction_pending",
            readiness=readiness,
            progress_percent=35.0,
            progress_message="Text extraction has not completed.",
        )

    if chunking_status and chunking_status not in {"completed", "complete", "done"} and not vector_failed:
        return _base_status(
            source,
            state="chunking",
            reason="chunking_pending",
            readiness=readiness,
            progress_percent=65.0,
            progress_message="Text is available while chunking continues.",
        )

    if readiness["vector_ready"]:
        return _base_status(
            source,
            state="queryable",
            reason="source_queryable",
            readiness=readiness,
            progress_percent=100.0,
            progress_message="Ready for grounded questions.",
        )

    if vector_failed:
        return _base_status(
            source,
            state="partially_queryable",
            reason="vector_index_failed",
            readiness=readiness,
            progress_percent=75.0,
            progress_message="Text search is available, but vector indexing failed.",
        )

    return _base_status(
        source,
        state="partially_queryable",
        reason="vector_index_pending",
        readiness=readiness,
        progress_percent=75.0,
        progress_message="Text search is available while vector indexing continues.",
    )


def _status_from_active_job(source: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    state = _state_from_job(job)
    return _base_status(
        source,
        state=state,
        reason=f"job_{state}",
        progress_percent=_coerce_float(job.get("progress_percent")),
        progress_message=job.get("progress_message") or "Ingestion job is running.",
        job=_job_payload(job),
    )


def _status_from_failed_job(source: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    raw_error = job.get("error_message") or _job_result_error(job)
    if raw_error:
        logger.warning(
            "Workspace source ingest job failed for source={} job_id={} error={}",
            source.get("id"),
            job.get("id") if job.get("id") is not None else job.get("uuid"),
            raw_error,
        )
    return _base_status(
        source,
        state="failed",
        reason="job_failed",
        progress_percent=_coerce_float(job.get("progress_percent")),
        progress_message="Ingestion job failed.",
        job=_job_payload(job),
    )


def _base_status(
    source: dict[str, Any],
    *,
    state: str,
    reason: str,
    readiness: dict[str, bool] | None = None,
    progress_percent: float | None = None,
    progress_message: str | None = None,
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": source["id"],
        "workspace_id": source["workspace_id"],
        "media_id": source.get("media_id"),
        "title": source.get("title") or "",
        "source_type": source.get("source_type") or "",
        "url": source.get("url"),
        "selected": bool(source.get("selected", True)),
        "state": state,
        "status_reason": reason,
        "readiness": readiness or _empty_readiness(),
        "progress_percent": progress_percent,
        "progress_message": progress_message,
        "job": job,
        "updated_at": str(source.get("added_at", "")),
    }


def _empty_readiness() -> dict[str, bool]:
    return {
        "metadata_ready": False,
        "text_extracted": False,
        "fts_ready": False,
        "vector_ready": False,
        "citation_ready": False,
        "summary_ready": False,
        "tool_accessible": False,
    }


def _summarize_sources(statuses: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "total": len(statuses),
        "selected": 0,
        "queryable": 0,
        "partially_queryable": 0,
        "processing": 0,
        "failed": 0,
        "missing": 0,
    }
    for status in statuses:
        state = str(status.get("state") or "")
        reason = str(status.get("status_reason") or "")
        if status.get("selected"):
            summary["selected"] += 1
        if state == "queryable":
            summary["queryable"] += 1
        if state == "partially_queryable":
            summary["partially_queryable"] += 1
        if state in _PROCESSING_STATES or reason in _PENDING_REASONS:
            summary["processing"] += 1
        if state in {"failed", "blocked_by_permissions"}:
            summary["failed"] += 1
        if state == "missing_media":
            summary["missing"] += 1
    return summary


def _get_media(media_db: Any | None, media_id: int) -> dict[str, Any] | None:
    if media_db is None:
        return None
    try:
        return media_db_api.get_media_by_id(media_db, media_id)
    except (AttributeError, DatabaseError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("Workspace source status media lookup failed for media_id={}: {}", media_id, exc)
        return None


def _has_unvectorized_chunks(media_db: Any | None, media_id: int) -> bool:
    if media_db is None:
        return False
    try:
        return media_db_api.has_unvectorized_chunks(media_db, media_id)
    except (AttributeError, DatabaseError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("Workspace source status chunk lookup failed for media_id={}: {}", media_id, exc)
        return False


def _is_vector_ready(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return int(value) == 1
    normalized = str(value or "").strip().lower()
    return normalized in {"1", "true", "ready", "completed", "complete", "done"}


def _is_vector_failed(value: Any) -> bool:
    if isinstance(value, int | float):
        return int(value) < 0
    normalized = str(value or "").strip().lower()
    return any(token in normalized for token in ("error", "failed", "failure"))


def _is_failure_status(value: str) -> bool:
    return any(token in value for token in ("error", "failed", "failure"))


def _allowed(allowed: bool, reason_code: str | None = None) -> dict[str, Any]:
    return {
        "allowed": allowed,
        "reason_code": None if allowed else reason_code,
    }


def _normalize_access_level(value: Any) -> str:
    """Normalize owned/shared workspace access labels for capability gating."""
    normalized = str(value or "").strip().lower()
    if normalized in {"owner", "editor", "viewer"}:
        return normalized
    if normalized == "full_edit":
        return "editor"
    if normalized == "view_chat_add":
        return "editor"
    if normalized == "view_chat":
        return "viewer"
    return "viewer"


def _build_job_index(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for job in jobs:
        for key in _job_match_keys(job):
            existing = index.get(key)
            if existing is None or _job_sort_value(job) > _job_sort_value(existing):
                index[key] = job
    return index


def _match_job_for_source(source: dict[str, Any], job_index: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for key in _source_match_keys(source):
        job = job_index.get(key)
        if job is not None:
            return job
    return None


def _source_match_keys(source: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    raw_id = source.get("id")
    if raw_id:
        keys.append(f"id:{str(raw_id).strip()}")
    media_id = _coerce_int(source.get("media_id"))
    if media_id is not None and media_id > 0:
        keys.append(f"media:{media_id}")
    for field in ("url", "title"):
        raw = source.get(field)
        if raw:
            keys.append(f"{field}:{str(raw).strip()}")
    return keys


def _job_match_keys(job: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    payload = _normalize_mapping(job.get("payload"))
    result = _normalize_mapping(job.get("result"))
    for container in (payload, result):
        media_id = _coerce_int(container.get("media_id"))
        if media_id is not None and media_id > 0:
            keys.add(f"media:{media_id}")
    for field in ("source_id", "workspace_source_id"):
        raw = payload.get(field) or result.get(field)
        if raw:
            keys.add(f"id:{str(raw).strip()}")
    for field in ("source", "url", "input_ref", "original_filename"):
        raw = payload.get(field) or result.get(field)
        if raw:
            keys.add(f"url:{str(raw).strip()}")
    for field in ("title",):
        raw = payload.get(field) or result.get(field)
        if raw:
            keys.add(f"title:{str(raw).strip()}")
    return keys


def _normalize_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _is_active_job(job: dict[str, Any]) -> bool:
    return str(job.get("status") or "").strip().lower() in _ACTIVE_JOB_STATUSES


def _is_failed_job(job: dict[str, Any]) -> bool:
    return str(job.get("status") or "").strip().lower() in _FAILED_JOB_STATUSES


def _state_from_job(job: dict[str, Any]) -> str:
    status = str(job.get("status") or "").strip().lower()
    if status == "queued":
        return "queued"
    if status == "retrying":
        return "retrying"
    message = str(job.get("progress_message") or "").strip().lower()
    if "index" in message or "embedding" in message or "vector" in message:
        return "indexing"
    if "chunk" in message:
        return "chunking"
    if "extract" in message or "transcrib" in message:
        return "extracting"
    return "ingesting"


def _job_payload(job: dict[str, Any]) -> dict[str, Any]:
    error_message = None if _is_failed_job(job) else job.get("error_message") or _job_result_error(job)
    return {
        "id": job.get("id"),
        "uuid": job.get("uuid"),
        "status": job.get("status"),
        "job_type": job.get("job_type"),
        "progress_percent": _coerce_float(job.get("progress_percent")),
        "progress_message": job.get("progress_message"),
        "error_message": error_message,
    }


def _job_result_error(job: dict[str, Any]) -> str | None:
    result = _normalize_mapping(job.get("result"))
    raw = result.get("error") or result.get("message")
    return str(raw) if raw else None


def _job_sort_value(job: dict[str, Any]) -> tuple[str, int]:
    created_at = str(job.get("created_at") or "")
    job_id = _coerce_int(job.get("id"))
    return created_at, job_id if job_id is not None else 0


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
