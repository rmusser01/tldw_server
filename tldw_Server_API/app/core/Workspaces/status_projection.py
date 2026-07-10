"""Read-computed status projection for Research Workspace sources."""

from __future__ import annotations

import json
from typing import Any

from tldw_Server_API.app.core.DB_Management.media_db import api as media_db_api
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.Workspaces.context import build_workspace_core_context


_ACTIVE_JOB_STATUSES = frozenset({"queued", "processing", "running", "retrying"})
_FAILED_JOB_STATUSES = frozenset({"failed", "cancelled", "quarantined"})
_PROCESSING_STATES = frozenset({"queued", "ingesting", "extracting", "chunking", "indexing", "retrying"})
_PENDING_REASONS = frozenset({"vector_index_pending", "chunking_pending", "extraction_pending"})
_WORKSPACE_SOURCE_JOB_TYPE = "workspace_source_ingest"
_ERROR_CODE_ALLOWED_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
)


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


def derive_workspace_source_media_status(
    media: dict[str, Any],
    *,
    media_db: Any | None = None,
    media_id: int | None = None,
) -> dict[str, Any]:
    """Derive source lifecycle state from an already-ingested media row."""
    resolved_media_id = media_id if media_id is not None else _coerce_int(media.get("id"))
    source = {
        "id": str(media.get("uuid") or resolved_media_id or ""),
        "workspace_id": "",
        "media_id": resolved_media_id,
        "title": str(media.get("title") or ""),
        "source_type": str(media.get("type") or media.get("media_type") or ""),
        "url": media.get("url"),
        "selected": True,
        "added_at": "",
    }
    return _status_from_media(
        source,
        media,
        media_db=media_db,
        media_id=int(resolved_media_id or 0),
    )


def build_workspace_capability_projection(
    *,
    workspace: dict[str, Any],
    status_projection: dict[str, Any],
    service_capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build conservative capability gates for Research Workspace clients."""
    summary = dict(status_projection.get("summary") or {})
    has_retrieval_capable_sources = _has_retrieval_capable_selected_sources(
        status_projection
    )
    workspace_services = {
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
    }
    service_actions: dict[str, dict[str, Any]] = {}
    if isinstance(service_capabilities, dict):
        dynamic_services = service_capabilities.get("workspace_services")
        if isinstance(dynamic_services, dict):
            workspace_services.update(
                {
                    key: value
                    for key, value in dynamic_services.items()
                    if isinstance(value, dict)
                }
            )
        dynamic_actions = service_capabilities.get("allowed_actions")
        if isinstance(dynamic_actions, dict):
            service_actions = {
                key: value
                for key, value in dynamic_actions.items()
                if isinstance(value, dict)
            }
    ask_action = _grounded_question_action(
        has_queryable_sources=has_retrieval_capable_sources,
        provider_service=workspace_services.get("provider") or {},
    )
    allowed_actions = {
        "add_sources": _allowed(True),
        "inspect_sources": _allowed(
            int(summary.get("total") or 0) > 0,
            "no_sources",
        ),
        "ask_grounded_questions": ask_action,
        "export_workspace": _allowed(True),
        "manage_tools": _allowed(True),
        "run_mcp_tools": _allowed(False, "mcp_not_configured"),
        "use_acp_agents": _allowed(False, "acp_not_configured"),
        "use_sandbox": _allowed(False, "sandbox_not_configured"),
    }
    allowed_actions.update(service_actions)
    allowed_actions["ask_grounded_questions"] = ask_action
    core_context = build_workspace_core_context(
        workspace=workspace,
        primary_root=workspace.get("primary_root") or workspace.get("project_root"),
        source_summary=summary,
        service_capabilities={
            "workspace_services": workspace_services,
            "allowed_actions": allowed_actions,
        },
        partial_errors=(
            service_capabilities.get("partial_errors")
            if isinstance(service_capabilities, dict)
            else []
        ),
    )

    return {
        "workspace_id": core_context["workspace_id"],
        "workspace_profile": core_context["workspace_profile"],
        "workspace_kind": core_context["workspace_kind"],
        "access_level": core_context["access_level"],
        "resolution": core_context["resolution"],
        "project_root": core_context["project_root"],
        "source_summary": core_context["source_summary"],
        "workspace_services": core_context["workspace_services"],
        "allowed_actions": core_context["allowed_actions"],
    }


def _has_retrieval_capable_selected_sources(status_projection: dict[str, Any]) -> bool:
    """Return whether selected sources can support a grounded retrieval path."""
    statuses = status_projection.get("sources")
    if isinstance(statuses, list):
        return any(
            isinstance(status, dict)
            and bool(status.get("selected"))
            and (
                str(status.get("state") or "").strip().lower() == "queryable"
                or _supports_text_search(status)
            )
            for status in statuses
        )

    summary = dict(status_projection.get("summary") or {})
    # Summary-only payloads cannot express the text_extracted/FTS/tool gates.
    # Treat partial queryability as a conservative fallback only when per-source
    # records are unavailable.
    return (
        int(summary.get("queryable") or 0) > 0
        or int(summary.get("partially_queryable") or 0) > 0
    )


def _build_source_status(
    source: dict[str, Any],
    *,
    media_db: Any | None,
    job_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    matched_job = _match_job_for_source(source, job_index)
    media_id = _coerce_int(source.get("media_id"))
    media = _get_media(media_db, media_id) if media_id is not None and media_id > 0 else None
    media_status = (
        _status_from_media(source, media, media_db=media_db, media_id=media_id)
        if media is not None and media_id is not None
        else None
    )
    if media_status is not None and _should_prefer_media_status_over_job(matched_job, media_status):
        if matched_job and _is_active_job(matched_job):
            return _attach_stale_job(media_status, matched_job)
        return media_status

    if matched_job and _is_active_job(matched_job):
        return _status_from_active_job(source, matched_job)

    if matched_job and _is_failed_job(matched_job):
        return _status_from_failed_job(source, matched_job)

    if media_id is None or media_id <= 0:
        return _base_status(
            source,
            state="missing_media",
            reason="media_id_missing",
            progress_percent=0.0,
            progress_message="Workspace source does not have a media item yet.",
        )

    if media is None:
        return _base_status(
            source,
            state="missing_media",
            reason="media_not_found",
            progress_percent=0.0,
            progress_message="Media item is missing or unavailable.",
        )

    return media_status or _status_from_media(source, media, media_db=media_db, media_id=media_id)


def _status_from_media(
    source: dict[str, Any],
    media: dict[str, Any],
    *,
    media_db: Any | None,
    media_id: int,
) -> dict[str, Any]:
    text_ready = _media_has_text(media)
    chunking_status = str(media.get("chunking_status") or "").strip().lower()
    vector_processing = media.get("vector_processing")
    vector_ready = _is_vector_ready(vector_processing)
    vector_failed = _is_vector_failed(vector_processing) or _is_failure_status(chunking_status)

    readiness = {
        "metadata_ready": True,
        "text_extracted": text_ready,
        "fts_ready": text_ready,
        "vector_ready": vector_ready,
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

    if readiness["vector_ready"] and not vector_failed:
        return _base_status(
            source,
            state="queryable",
            reason="source_queryable",
            readiness=readiness,
            progress_percent=100.0,
            progress_message="Ready for grounded questions.",
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
    return _base_status(
        source,
        state="failed",
        reason=_failed_job_reason(job),
        progress_percent=_coerce_float(job.get("progress_percent")),
        progress_message=job.get("error_message") or _job_result_error(job) or "Ingestion job failed.",
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
    resolved_readiness = readiness or _empty_readiness()
    return {
        "id": source["id"],
        "workspace_id": source["workspace_id"],
        "media_id": source.get("media_id"),
        "title": source.get("title") or "",
        "source_type": source.get("source_type") or "",
        "url": source.get("url"),
        "selected": bool(source.get("selected", True)),
        "review_state": source.get("review_state") or "unset",
        "review_state_updated_at": source.get("review_state_updated_at"),
        "reviewed_at": source.get("reviewed_at"),
        "reviewed_by_user_id": source.get("reviewed_by_user_id"),
        "state": state,
        "status_reason": reason,
        "readiness": resolved_readiness,
        "progress_percent": progress_percent,
        "progress_message": progress_message,
        "job": job,
        "next_action": _next_action_for_status(
            state=state,
            reason=reason,
            readiness=resolved_readiness,
        ),
        "retry_eligible": _retry_eligible_for_status(state=state, reason=reason),
        "stale": False,
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


def _supports_text_search(status: dict[str, Any]) -> bool:
    """Return whether a partial source still has enough text readiness for FTS."""
    readiness = status.get("readiness") or {}
    return (
        str(status.get("state") or "").strip().lower() == "partially_queryable"
        and bool(readiness.get("text_extracted"))
        and bool(readiness.get("fts_ready"))
        and bool(readiness.get("tool_accessible"))
    )


def _next_action_for_status(
    *,
    state: str,
    reason: str,
    readiness: dict[str, bool],
) -> str:
    """Map source state and readiness into the next user-facing recovery action."""
    if state == "queryable":
        return "ask_grounded_questions"
    if state == "partially_queryable":
        if reason == "vector_index_failed":
            return "retry_vector_indexing"
        if _supports_text_search({"state": state, "readiness": readiness}):
            return "vector_indexing_pending"
        return "refresh_source_status"
    if state == "queued":
        return "wait_for_ingestion_start"
    if state == "ingesting":
        return "wait_for_ingestion"
    if state == "extracting":
        return "wait_for_text_extraction"
    if state == "chunking":
        return "wait_for_chunking"
    if state == "indexing":
        return "wait_for_indexing"
    if state == "retrying":
        return "wait_for_retry"
    if state == "failed":
        return "retry_ingestion_or_readd_source"
    if state == "missing_media":
        return "restore_or_readd_media"
    if state == "blocked_by_permissions":
        return "check_source_permissions"
    return "refresh_source_status"


def _retry_eligible_for_status(*, state: str, reason: str) -> bool:
    """Return whether the UI should expose retry for a projected source status."""
    if state in {"failed", "missing_media", "blocked_by_permissions"}:
        return True
    return state == "partially_queryable" and reason == "vector_index_failed"


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
        return media_db_api.get_media_status_by_id(media_db, media_id)
    except (AttributeError, DatabaseError, RuntimeError, TypeError, ValueError):
        return None


def _media_has_text(media: dict[str, Any]) -> bool:
    if "has_content" in media:
        return _coerce_bool(media.get("has_content"))
    if "content_length" in media:
        return (_coerce_int(media.get("content_length")) or 0) > 0
    return bool(str(media.get("content") or "").strip())


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return int(value) != 0
    normalized = str(value or "").strip().lower()
    return normalized in {"1", "true", "yes", "y", "ready", "completed", "complete", "done"}


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


def _grounded_question_action(
    *,
    has_queryable_sources: bool,
    provider_service: dict[str, Any],
) -> dict[str, Any]:
    if not has_queryable_sources:
        return _allowed(False, "no_queryable_sources")
    provider_state = str(provider_service.get("state") or "").strip().lower()
    if provider_state in {"not_configured", "unknown", "blocked"}:
        return _allowed(
            False,
            str(provider_service.get("reason_code") or "provider_not_available"),
        )
    return _allowed(True)


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


def _source_match_keys(source: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    media_id = _coerce_int(source.get("media_id"))
    if media_id is not None and media_id > 0:
        keys.add(f"media:{media_id}")
    for field in ("id", "url", "title"):
        raw = source.get(field)
        if raw:
            keys.add(f"{field}:{str(raw).strip()}")
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


def _is_workspace_source_job(job: dict[str, Any] | None) -> bool:
    return str((job or {}).get("job_type") or "").strip().lower() == _WORKSPACE_SOURCE_JOB_TYPE


def _should_prefer_media_status_over_job(
    job: dict[str, Any] | None,
    media_status: dict[str, Any],
) -> bool:
    if job is None:
        return True
    if not _is_active_job(job):
        return False
    if str(media_status.get("state") or "") == "queryable":
        return True
    readiness = media_status.get("readiness") or {}
    return _is_workspace_source_job(job) and bool(readiness.get("text_extracted"))


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
    payload = {
        "id": job.get("id"),
        "uuid": job.get("uuid"),
        "status": job.get("status"),
        "job_type": job.get("job_type"),
        "progress_percent": _coerce_float(job.get("progress_percent")),
        "progress_message": job.get("progress_message"),
        "error_message": job.get("error_message") or _job_result_error(job),
    }
    error_code = _job_error_code(job)
    if error_code is not None:
        payload["error_code"] = error_code
    return payload


def _attach_stale_job(
    status: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    return {
        **status,
        "job": _job_payload(job),
        "stale": True,
    }


def _job_result_error(job: dict[str, Any]) -> str | None:
    result = _normalize_mapping(job.get("result"))
    raw = result.get("error") or result.get("message")
    return str(raw) if raw else None


def _failed_job_reason(job: dict[str, Any]) -> str:
    """Return the public source-status reason for a failed Jobs row.

    Workspace-source jobs can expose their sanitized persisted error code.
    Other job types stay collapsed to the generic `job_failed` reason.
    """
    if _is_workspace_source_job(job):
        return _job_error_code(job) or "job_failed"
    return "job_failed"


def _job_error_code(job: dict[str, Any]) -> str | None:
    """Return a safe, identifier-like Jobs error code for API projection.

    Only non-empty `[A-Za-z0-9_.-]` values are exposed, capped to 128
    characters, so free-form worker errors cannot become public reason codes.
    """
    raw = str(job.get("error_code") or "").strip()
    if not raw:
        return None
    if any(char not in _ERROR_CODE_ALLOWED_CHARS for char in raw):
        return None
    return raw[:128]


def _job_sort_value(job: dict[str, Any]) -> tuple[str, int]:
    created_at = str(job.get("created_at") or "")
    return created_at, _coerce_int(job.get("id")) or 0


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
