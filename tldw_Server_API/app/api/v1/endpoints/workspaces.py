"""Workspace lifecycle CRUD endpoints."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import PurePath, PureWindowsPath
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import try_get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import try_get_job_manager
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import (
    WORKSPACES_DELETE_RATE_LIMIT,
    WORKSPACES_READ_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.api.v1.schemas.workspace_schemas import (
    StatusResponse,
    WorkspaceArtifactCreateRequest,
    WorkspaceArtifactExportRequest,
    WorkspaceArtifactExportResponse,
    WorkspaceArtifactResponse,
    WorkspaceArtifactUpdateRequest,
    WorkspaceContextResponse,
    WorkspaceFileInventoryEntryKind,
    WorkspaceFileInventoryItemsResponse,
    WorkspaceFileInventoryScanRequest,
    WorkspaceFileInventoryStatusResponse,
    WorkspaceListResponse,
    WorkspaceNoteCreateRequest,
    WorkspaceNoteResponse,
    WorkspaceNoteUpdateRequest,
    WorkspacePatchRequest,
    WorkspacePrimaryRootAttachRequest,
    WorkspaceCapabilitiesResponse,
    WorkspaceResponse,
    WorkspaceRootResponse,
    WorkspaceRootsResponse,
    WorkspaceSourceCreateRequest,
    WorkspaceSourcePreviewResponse,
    WorkspaceSourceReorderRequest,
    WorkspaceSourceResponse,
    WorkspaceSourceSelectionRequest,
    WorkspaceSourceStatusListResponse,
    WorkspaceSourceUpdateRequest,
    WorkspaceUpsertRequest,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.media_db import api as media_db_api
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Workspaces.file_inventory_ignore import build_inventory_ignore_policy
from tldw_Server_API.app.core.Workspaces.file_inventory_jobs import (
    WorkspaceFileInventoryEnqueueError,
    enqueue_workspace_file_inventory_scan_job,
)
from tldw_Server_API.app.core.Workspaces.source_jobs import (
    WORKSPACE_SOURCE_JOB_DOMAIN,
    WORKSPACE_SOURCE_JOB_QUEUE,
    WORKSPACE_SOURCE_JOB_TYPE,
    enqueue_workspace_source_ingest_job,
)
from tldw_Server_API.app.core.Workspaces.service_capabilities import (
    collect_workspace_service_capabilities,
)
from tldw_Server_API.app.core.Workspaces.root_binding_service import (
    WorkspaceRootAttachRequest,
    WorkspaceRootServiceError,
    attach_primary_workspace_root,
)
from tldw_Server_API.app.core.Workspaces.models import normalize_project_root_state
from tldw_Server_API.app.core.Workspaces.status_projection import (
    build_source_status_projection,
    build_workspace_capability_projection,
)
from tldw_Server_API.app.core.Workspaces.workspace_artifact_exports import (
    export_workspace_artifact_version,
)
from tldw_Server_API.app.core.exceptions import WorkspaceArtifactExportStateError

router = APIRouter()

WORKSPACE_ACTIVE_JOB_STATUSES = {"queued", "processing", "running", "retrying"}


def _ws_to_response(ws: dict) -> WorkspaceResponse:
    """Convert a workspace DB row dict to a WorkspaceResponse schema."""
    return WorkspaceResponse(
        id=ws["id"],
        name=ws.get("name"),
        archived=bool(ws.get("archived", False)),
        study_materials_policy=str(ws.get("study_materials_policy") or "general"),
        workspace_profile=str(ws.get("workspace_profile") or "research"),
        deleted=bool(ws.get("deleted", False)),
        banner_title=ws.get("banner_title"),
        banner_subtitle=ws.get("banner_subtitle"),
        banner_color=ws.get("banner_color"),
        audio_provider=ws.get("audio_provider"),
        audio_model=ws.get("audio_model"),
        audio_voice=ws.get("audio_voice"),
        audio_speed=ws.get("audio_speed"),
        created_at=str(ws.get("created_at", "")),
        last_modified=str(ws.get("last_modified", "")),
        version=ws.get("version", 1),
    )


def _src_to_response(src: dict) -> WorkspaceSourceResponse:
    """Convert a workspace source DB row dict to a WorkspaceSourceResponse schema."""
    return WorkspaceSourceResponse(
        id=src["id"],
        workspace_id=src["workspace_id"],
        media_id=src["media_id"],
        title=src["title"],
        source_type=src["source_type"],
        url=src.get("url"),
        position=src.get("position", 0),
        selected=bool(src.get("selected", True)),
        added_at=str(src.get("added_at", "")),
        version=src.get("version", 1),
    )


def _root_to_response(root: dict[str, Any]) -> WorkspaceRootResponse:
    """Convert a project root DB row to a redacted public response."""
    return WorkspaceRootResponse(
        workspace_id=root.get("workspace_id"),
        root_id=root.get("root_id") or root.get("id"),
        backend=_root_backend(root.get("backend")),
        state=normalize_project_root_state(root.get("root_state") or root.get("state")),
        display_name=root.get("display_name"),
        path_hint=_root_path_hint(root),
        git_state=root.get("git_state"),
        file_inventory_state=root.get("file_inventory_state"),
        indexing_state=root.get("indexing_state"),
        sandbox_mount_state=root.get("sandbox_mount_state"),
        mcp_trust_state=root.get("mcp_trust_state"),
        is_primary=bool(root.get("is_primary", True)),
        version=root.get("version"),
        updated_at=str(root["updated_at"]) if root.get("updated_at") else None,
    )


def _workspace_roots_response(
    *,
    workspace_id: str,
    workspace: dict[str, Any],
    roots: list[dict[str, Any]],
) -> WorkspaceRootsResponse:
    primary_root = next((root for root in roots if bool(root.get("is_primary"))), None)
    return WorkspaceRootsResponse(
        workspace_id=workspace_id,
        workspace_profile=str(workspace.get("workspace_profile") or "research"),
        primary_root=_root_to_response(primary_root) if primary_root else None,
        roots=[_root_to_response(root) for root in roots],
    )


def _root_path_hint(root: dict[str, Any]) -> str | None:
    explicit_hint = root.get("path_hint")
    if explicit_hint:
        return _redacted_path_hint(explicit_hint)
    if root.get("sandbox_volume_id"):
        return str(root["sandbox_volume_id"])
    if root.get("display_name"):
        return _redacted_path_hint(root["display_name"])
    absolute_root = root.get("absolute_root")
    if absolute_root:
        return _basename_path_hint(absolute_root)
    return None


def _root_backend(value: Any) -> str | None:
    backend = str(value or "").strip().lower()
    if backend in {"host_local", "sandbox_volume"}:
        return backend
    return None


def _redacted_path_hint(value: Any) -> str:
    raw_value = str(value)
    windows_path = PureWindowsPath(raw_value)
    if raw_value.startswith(("/", "~", "\\\\")) or windows_path.is_absolute():
        if windows_path.is_absolute() or raw_value.startswith("\\\\"):
            return windows_path.name or "project_root"
        return PurePath(raw_value).name or "project_root"
    return raw_value


def _basename_path_hint(value: Any) -> str:
    raw_value = str(value).strip()
    if not raw_value:
        return "project_root"
    windows_path = PureWindowsPath(raw_value)
    windows_name = windows_path.name
    if windows_name and windows_name not in {".", "..", "/", "\\"}:
        return windows_name
    if windows_path.drive or windows_path.root:
        return "project_root"
    posix_name = PurePath(raw_value).name
    if posix_name and posix_name not in {".", "..", "/", "\\"}:
        return posix_name
    return "project_root"


def _utc_now_iso() -> str:
    """Return a UTC timestamp suitable for read-projection responses."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_preview_href(workspace_id: str, source_id: str) -> str:
    """Return the API-relative source preview URL for a workspace source."""
    return (
        f"/api/v1/workspaces/{quote(workspace_id, safe='')}"
        f"/sources/{quote(source_id, safe='')}/preview"
    )


def _find_source_in_workspace(
    sources: list[dict[str, Any]],
    source_id: str,
) -> dict[str, Any] | None:
    for source in sources:
        if str(source.get("id")) == source_id:
            return source
    return None


def _source_preview_summary(
    workspace_id: str,
    source_status: dict[str, Any],
) -> dict[str, Any]:
    readiness = source_status.get("readiness") or {}
    available = bool(
        readiness.get("citation_ready") or readiness.get("text_extracted")
    )
    return {
        "available": available,
        "detail_href": _source_preview_href(
            workspace_id,
            str(source_status.get("id") or ""),
        ),
        "snippet_count": None,
        "total_chars": None,
        "unavailable_reason": None
        if available
        else str(source_status.get("status_reason") or "source_unavailable"),
    }


def _context_source_payload(
    *,
    workspace_id: str,
    source: dict[str, Any],
    source_status: dict[str, Any],
) -> dict[str, Any]:
    base = _src_to_response(source).model_dump()
    source_status = {"id": source["id"], **source_status}
    return {
        **base,
        "state": source_status.get("state") or "missing_media",
        "status_reason": source_status.get("status_reason") or "unknown",
        "readiness": source_status.get("readiness") or {},
        "progress_percent": source_status.get("progress_percent"),
        "progress_message": source_status.get("progress_message"),
        "job": source_status.get("job"),
        "updated_at": source_status.get("updated_at") or "",
        "preview": _source_preview_summary(workspace_id, source_status),
    }


def _job_status_payload(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job.get("id"),
        "uuid": job.get("uuid"),
        "status": job.get("status"),
        "job_type": job.get("job_type"),
        "progress_percent": job.get("progress_percent"),
        "progress_message": job.get("progress_message"),
        "error_message": job.get("error_message"),
    }


def _workspace_file_inventory_job_status(job: dict[str, Any] | None) -> dict[str, Any] | None:
    if not job:
        return None
    payload = _job_status_payload(job)
    payload["error_message"] = None
    return payload


def _safe_get_job(jm: JobManager | None, job_id: Any) -> dict[str, Any] | None:
    if jm is None or job_id is None:
        return None
    try:
        return jm.get_job(int(job_id))
    except (AttributeError, TypeError, ValueError) as exc:
        logger.warning("Workspace file inventory job lookup failed for id {}: {}", job_id, exc)
        return None
    except Exception as exc:
        logger.warning("Workspace file inventory job lookup failed for id {}: {}", job_id, exc)
        return None


def _workspace_file_inventory_status_response(
    status_payload: dict[str, Any],
    *,
    job: dict[str, Any] | None = None,
) -> WorkspaceFileInventoryStatusResponse:
    return WorkspaceFileInventoryStatusResponse(
        workspace_id=str(status_payload.get("workspace_id") or ""),
        root_id=status_payload.get("root_id"),
        state=str(status_payload.get("state") or "not_started"),
        durable_state=status_payload.get("durable_state"),
        stale=bool(status_payload.get("stale", False)),
        last_scan_id=status_payload.get("scan_id"),
        last_scan_started_at=status_payload.get("started_at"),
        last_scan_completed_at=status_payload.get("completed_at"),
        root_version=status_payload.get("root_version"),
        scan_root_version=status_payload.get("scan_root_version"),
        ignore_policy_fingerprint=status_payload.get("ignore_policy_fingerprint"),
        root_snapshot_token=status_payload.get("root_snapshot_token"),
        counts=status_payload.get("counts") or {},
        diagnostics=status_payload.get("diagnostics") or [],
        job=_workspace_file_inventory_job_status(job),
        updated_at=status_payload.get("updated_at"),
    )


def _workspace_file_inventory_no_root_conflict() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "code": "workspace_project_root_missing",
            "message": "Workspace has no primary project root.",
        },
    )


def _art_to_response(art: dict) -> WorkspaceArtifactResponse:
    """Convert a workspace artifact DB row dict to a WorkspaceArtifactResponse schema."""
    version = art.get("version") or 1
    return WorkspaceArtifactResponse(
        id=art["id"],
        workspace_id=art["workspace_id"],
        artifact_type=art["artifact_type"],
        title=art["title"],
        status=art.get("status", "pending"),
        content=art.get("content"),
        content_type=art.get("content_type") or "text/markdown",
        preview_text=art.get("preview_text"),
        summary=art.get("summary"),
        review_state=art.get("review_state") or "draft",
        owner_scope=art.get("owner_scope") or "user",
        owner_id=art.get("owner_id"),
        project_id=art.get("project_id"),
        task_id=art.get("task_id"),
        source_collection_id=art.get("source_collection_id"),
        root_artifact_id=art.get("root_artifact_id") or art["id"],
        artifact_version_id=art.get("artifact_version_id") or f"{art['id']}:v{version}",
        previous_version_id=art.get("previous_version_id"),
        producer_metadata=art.get("producer_metadata") or {},
        source_lineage=art.get("source_lineage") or {},
        review_metadata=art.get("review_metadata") or {},
        version_metadata=art.get("version_metadata") or {},
        export_refs=art.get("export_refs") or [],
        redaction=art.get("redaction") or {"support_safe": True, "redacted": False},
        schema_version=art.get("schema_version") or 1,
        total_tokens=art.get("total_tokens"),
        total_cost_usd=art.get("total_cost_usd"),
        created_at=str(art.get("created_at", "")),
        completed_at=str(art["completed_at"]) if art.get("completed_at") else None,
        version=version,
    )


def _workspace_artifact_version_for_export(
    db: CharactersRAGDB,
    workspace_id: str,
    artifact_id: str,
    artifact_version_id: str | None,
) -> dict:
    """Fetch the current artifact or a version snapshot for export."""
    artifact = db.get_workspace_artifact(workspace_id, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Workspace artifact not found")
    if artifact_version_id is None:
        return artifact

    version = db.get_workspace_artifact_version(workspace_id, artifact_id, artifact_version_id)
    if version is None:
        raise HTTPException(status_code=404, detail="Workspace artifact version not found")
    merged = {**artifact, **version}
    merged["id"] = artifact_id
    merged["workspace_id"] = workspace_id
    merged["artifact_type"] = artifact.get("artifact_type")
    merged["content_type"] = artifact.get("content_type") or "text/markdown"
    merged["status"] = artifact.get("status")
    merged["owner_scope"] = artifact.get("owner_scope")
    merged["owner_id"] = artifact.get("owner_id")
    merged["project_id"] = artifact.get("project_id")
    merged["task_id"] = artifact.get("task_id")
    merged["source_collection_id"] = artifact.get("source_collection_id")
    merged["schema_version"] = artifact.get("schema_version") or 1
    return merged


def _note_to_response(note: dict) -> WorkspaceNoteResponse:
    """Convert a workspace note DB row dict to a WorkspaceNoteResponse schema."""
    return WorkspaceNoteResponse(
        id=note["id"],
        workspace_id=note["workspace_id"],
        title=note["title"],
        content=note["content"],
        keywords_json=note.get("keywords_json", "[]"),
        created_at=str(note.get("created_at", "")),
        last_modified=str(note.get("last_modified", "")),
        version=note.get("version", 1),
    )


def _require_workspace(db: CharactersRAGDB, workspace_id: str) -> dict:
    """Fetch a workspace or raise 404 if not found."""
    ws = db.get_workspace(workspace_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return ws


def _workspace_with_primary_root(
    db: CharactersRAGDB,
    workspace_id: str,
    workspace: dict,
) -> dict:
    """Attach the primary project root to a workspace row for read-model projection."""
    primary_root = db.get_workspace_primary_root(workspace_id)
    if primary_root is None:
        return workspace
    enriched = dict(workspace)
    enriched["primary_root"] = primary_root
    return enriched


def try_get_workspace_job_manager() -> JobManager | None:
    """Resolve the Jobs manager for workspace views without blocking workspace reads/writes."""
    try:
        return try_get_job_manager()
    except Exception as exc:
        logger.warning("Workspace Jobs manager unavailable: {}", exc)
        return None


def _dedupe_jobs_by_identity(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for job in jobs:
        key = job.get("id") or job.get("uuid")
        if key is None:
            key = (
                job.get("domain"),
                job.get("queue"),
                job.get("job_type"),
                job.get("created_at"),
                str(job.get("payload") or ""),
            )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(job)
    return deduped


def _safe_list_jobs(jm: JobManager, **kwargs: Any) -> list[dict[str, Any]]:
    try:
        return jm.list_jobs(**kwargs)
    except Exception as exc:
        logger.warning("Workspace job list failed for filters {}: {}", kwargs, exc)
        return []


def _list_recent_media_ingest_jobs(jm: JobManager | None, current_user: User) -> list[dict[str, Any]]:
    """Return recent media-ingest Jobs for status projection, failing open."""
    if jm is None:
        return []
    workspace_source_jobs = _safe_list_jobs(
        jm,
        domain=WORKSPACE_SOURCE_JOB_DOMAIN,
        queue=WORKSPACE_SOURCE_JOB_QUEUE,
        owner_user_id=str(current_user.id),
        job_type=WORKSPACE_SOURCE_JOB_TYPE,
        limit=500,
        sort_by="created_at",
        sort_order="desc",
    )
    legacy_media_jobs = _safe_list_jobs(
        jm,
        domain=WORKSPACE_SOURCE_JOB_DOMAIN,
        owner_user_id=str(current_user.id),
        limit=500,
        sort_by="created_at",
        sort_order="desc",
    )
    return _dedupe_jobs_by_identity(workspace_source_jobs + legacy_media_jobs)


def _normalize_job_mapping(value: Any) -> dict[str, Any]:
    """Normalize Jobs payload/result fields that may arrive as JSON strings."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _workspace_source_match_keys(sources: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for source in sources:
        media_id = source.get("media_id")
        if media_id is not None:
            keys.add(f"media:{media_id}")
        for field in ("id",):
            raw = source.get(field)
            if raw:
                keys.add(f"{field}:{str(raw).strip()}")
    return keys


def _job_match_keys(job: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    payload = _normalize_job_mapping(job.get("payload"))
    result = _normalize_job_mapping(job.get("result"))
    for container in (payload, result):
        media_id = container.get("media_id")
        if media_id is not None:
            keys.add(f"media:{media_id}")
    for field in ("source_id", "workspace_source_id"):
        raw = payload.get(field) or result.get(field)
        if raw:
            keys.add(f"id:{str(raw).strip()}")
    return keys


def _job_workspace_id(job: dict[str, Any]) -> str | None:
    payload = _normalize_job_mapping(job.get("payload"))
    result = _normalize_job_mapping(job.get("result"))
    raw = payload.get("workspace_id") or result.get("workspace_id")
    return str(raw).strip() if raw else None


def _active_workspace_jobs(
    jobs: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    workspace_id: str,
) -> list[dict[str, Any]]:
    source_keys = _workspace_source_match_keys(sources)
    if not source_keys:
        return []
    active_jobs: list[dict[str, Any]] = []
    for job in jobs:
        status_value = str(job.get("status") or "").strip().lower()
        if status_value not in WORKSPACE_ACTIVE_JOB_STATUSES:
            continue
        matched_workspace_id = _job_workspace_id(job)
        if matched_workspace_id != workspace_id:
            continue
        if _job_match_keys(job).intersection(source_keys):
            active_jobs.append(job)
    return active_jobs


def _safe_get_media(media_db: Any | None, media_id: int) -> dict[str, Any] | None:
    if media_db is None:
        return None
    try:
        return media_db_api.get_media_by_id(media_db, media_id)
    except (AttributeError, DatabaseError, RuntimeError, TypeError, ValueError):
        return None


def _preview_mode_for_unavailable(source_status: dict[str, Any]) -> str:
    state = str(source_status.get("state") or "")
    reason = str(source_status.get("status_reason") or "")
    if state == "missing_media" or reason in {"media_not_found", "media_id_missing", "media_db_unavailable"}:
        return "missing_media"
    if state == "failed" or "failed" in reason:
        return "failed"
    if state in {"queued", "ingesting", "extracting", "chunking", "indexing", "retrying"}:
        return "pending"
    return "empty"


def _content_excerpt_snippet(
    *,
    source_id: str,
    media_id: int | None,
    text_preview: str,
) -> dict[str, Any]:
    return {
        "id": "content:0",
        "source_id": source_id,
        "media_id": media_id,
        "kind": "content_excerpt",
        "text": text_preview,
        "start_char": 0,
        "end_char": len(text_preview),
        "chunk_index": None,
        "chunk_uuid": None,
        "chunk_type": None,
    }


def _chunk_preview_snippets(
    *,
    media_db: Any | None,
    source_id: str,
    media_id: int | None,
    chunk_limit: int,
) -> list[dict[str, Any]]:
    if media_db is None or media_id is None or chunk_limit <= 0:
        return []
    try:
        chunks = media_db_api.get_unvectorized_chunks_in_range(
            media_db,
            media_id,
            0,
            chunk_limit - 1,
        )
    except (AttributeError, DatabaseError, RuntimeError, TypeError, ValueError):
        return []
    snippets: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        text = str(chunk.get("chunk_text") or "")
        if not text.strip():
            continue
        chunk_uuid = chunk.get("uuid")
        chunk_index = chunk.get("chunk_index")
        snippet_id = str(chunk_uuid or f"chunk:{chunk_index if chunk_index is not None else index}")
        snippets.append(
            {
                "id": snippet_id,
                "source_id": source_id,
                "media_id": media_id,
                "kind": "chunk",
                "text": text,
                "start_char": chunk.get("start_char"),
                "end_char": chunk.get("end_char"),
                "chunk_index": chunk_index,
                "chunk_uuid": str(chunk_uuid) if chunk_uuid is not None else None,
                "chunk_type": chunk.get("chunk_type"),
            }
        )
    return snippets


def _source_preview_payload(
    *,
    workspace_id: str,
    source: dict[str, Any],
    source_status: dict[str, Any],
    media_db: Any | None,
    max_chars: int,
    chunk_limit: int,
) -> dict[str, Any]:
    media_id_raw = source.get("media_id")
    try:
        media_id = int(media_id_raw) if media_id_raw is not None else None
    except (TypeError, ValueError):
        media_id = None

    media = _safe_get_media(media_db, media_id) if media_id is not None else None
    content = str((media or {}).get("content") or "")
    if not content.strip():
        reason = (
            "media_db_unavailable"
            if media_db is None
            else str(source_status.get("status_reason") or "content_unavailable")
        )
        return {
            "workspace_id": workspace_id,
            "source_id": source["id"],
            "media_id": media_id,
            "title": source.get("title") or "",
            "source_type": source.get("source_type") or "",
            "url": source.get("url"),
            "state": source_status.get("state") or "missing_media",
            "status_reason": reason,
            "readiness": source_status.get("readiness") or {},
            "content_available": False,
            "preview_mode": _preview_mode_for_unavailable(
                {**source_status, "status_reason": reason}
            ),
            "unavailable_reason": reason,
            "text_preview": None,
            "text_total_chars": None,
            "text_truncated": False,
            "snippets": [],
            "generated_at": _utc_now_iso(),
        }

    text_preview = content[:max_chars]
    snippets = [
        _content_excerpt_snippet(
            source_id=str(source["id"]),
            media_id=media_id,
            text_preview=text_preview,
        )
    ]
    snippets.extend(
        _chunk_preview_snippets(
            media_db=media_db,
            source_id=str(source["id"]),
            media_id=media_id,
            chunk_limit=chunk_limit,
        )
    )
    return {
        "workspace_id": workspace_id,
        "source_id": source["id"],
        "media_id": media_id,
        "title": source.get("title") or "",
        "source_type": source.get("source_type") or "",
        "url": source.get("url"),
        "state": source_status.get("state") or "queryable",
        "status_reason": source_status.get("status_reason") or "source_queryable",
        "readiness": source_status.get("readiness") or {},
        "content_available": True,
        "preview_mode": "available",
        "unavailable_reason": None,
        "text_preview": text_preview,
        "text_total_chars": len(content),
        "text_truncated": len(content) > len(text_preview),
        "snippets": snippets,
        "generated_at": _utc_now_iso(),
    }


def _enqueue_workspace_source_ingest_job(
    *,
    jm: JobManager | None,
    current_user: User,
    workspace_id: str,
    src: dict[str, Any],
) -> None:
    """Submit a user-visible lifecycle job after the workspace source row exists."""
    enqueue_workspace_source_ingest_job(
        jm=jm,
        owner_user_id=current_user.id,
        workspace_id=workspace_id,
        src=src,
    )


# ── Workspace CRUD ──────────────────────────────────────────────

@router.get(
    "/",
    response_model=WorkspaceListResponse,
    dependencies=[Depends(WORKSPACES_READ_RATE_LIMIT)],
    summary="List workspaces",
)
async def list_workspaces(
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """List non-deleted workspaces for the current user."""
    try:
        items = db.list_workspaces()
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to fetch workspaces") from exc
    return WorkspaceListResponse(
        items=[_ws_to_response(w) for w in items],
        total=len(items),
    )


@router.get(
    "/{workspace_id}",
    response_model=WorkspaceResponse,
    dependencies=[Depends(WORKSPACES_READ_RATE_LIMIT)],
    summary="Get workspace",
)
async def get_workspace(
    workspace_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """Fetch a workspace by ID."""
    try:
        ws = _require_workspace(db, workspace_id)
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to fetch workspace") from exc
    return _ws_to_response(ws)


@router.put(
    "/{workspace_id}",
    response_model=WorkspaceResponse,
    dependencies=[Depends(WORKSPACES_WRITE_RATE_LIMIT)],
    summary="Create or update workspace",
)
async def upsert_workspace(
    workspace_id: str,
    body: WorkspaceUpsertRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """Create or update a workspace (idempotent)."""
    try:
        workspace_profile = (
            body.workspace_profile
            if "workspace_profile" in body.model_fields_set
            else None
        )
        ws = db.upsert_workspace(
            workspace_id,
            body.name,
            study_materials_policy=body.study_materials_policy,
            workspace_profile=workspace_profile,
        )
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to create or update workspace") from exc
    return _ws_to_response(ws)


@router.patch(
    "/{workspace_id}",
    response_model=WorkspaceResponse,
    dependencies=[Depends(WORKSPACES_WRITE_RATE_LIMIT)],
    summary="Update workspace",
)
async def patch_workspace(
    workspace_id: str,
    body: WorkspacePatchRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """Update workspace fields with optimistic locking."""
    updates = body.model_dump(exclude_unset=True, exclude={"version"})
    if not updates:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No fields to update",
        )
    try:
        ws = db.update_workspace(workspace_id, updates, body.version)
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to update workspace") from exc
    return _ws_to_response(ws)


@router.delete(
    "/{workspace_id}",
    status_code=204,
    dependencies=[Depends(WORKSPACES_DELETE_RATE_LIMIT)],
    summary="Delete workspace",
)
async def delete_workspace(
    workspace_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """Soft-delete a workspace and cascade soft-delete its conversations."""
    ws = _require_workspace(db, workspace_id)
    try:
        db.delete_workspace(workspace_id, ws["version"])
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to delete workspace") from exc


# ── Sources ─────────────────────────────────────────────────────

@router.get(
    "/{workspace_id}/sources",
    response_model=list[WorkspaceSourceResponse],
    dependencies=[Depends(WORKSPACES_READ_RATE_LIMIT)],
    summary="List workspace sources",
)
async def list_sources(
    workspace_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
) -> list[WorkspaceSourceResponse]:
    """List all sources belonging to a workspace."""
    _require_workspace(db, workspace_id)
    try:
        sources = db.list_workspace_sources(workspace_id)
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to fetch workspace sources") from exc
    return [_src_to_response(s) for s in sources]


@router.get(
    "/{workspace_id}/sources/status",
    response_model=WorkspaceSourceStatusListResponse,
    dependencies=[Depends(WORKSPACES_READ_RATE_LIMIT)],
    summary="Get workspace source ingestion and indexing status",
)
async def get_sources_status(
    workspace_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    media_db: Any | None = Depends(try_get_media_db_for_user),
    jm: JobManager | None = Depends(try_get_workspace_job_manager),
    current_user: User = Depends(get_request_user),
) -> WorkspaceSourceStatusListResponse:
    """Return read-computed ingestion, extraction, chunking, and indexing status."""
    _require_workspace(db, workspace_id)
    try:
        sources = db.list_workspace_sources(workspace_id)
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to fetch workspace source status") from exc
    jobs = _list_recent_media_ingest_jobs(jm, current_user)
    payload = build_source_status_projection(
        workspace_id=workspace_id,
        sources=sources,
        media_db=media_db,
        jobs=jobs,
    )
    return WorkspaceSourceStatusListResponse(**payload)


@router.get(
    "/{workspace_id}/capabilities",
    response_model=WorkspaceCapabilitiesResponse,
    dependencies=[Depends(WORKSPACES_READ_RATE_LIMIT)],
    summary="Get workspace capability gates",
)
async def get_workspace_capabilities(
    workspace_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    media_db: Any | None = Depends(try_get_media_db_for_user),
    jm: JobManager | None = Depends(try_get_workspace_job_manager),
    current_user: User = Depends(get_request_user),
) -> WorkspaceCapabilitiesResponse:
    """Return conservative UI capability gates for the workspace model."""
    workspace = _require_workspace(db, workspace_id)
    try:
        sources = db.list_workspace_sources(workspace_id)
        workspace_projection = _workspace_with_primary_root(db, workspace_id, workspace)
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to fetch workspace capabilities") from exc
    status_payload = build_source_status_projection(
        workspace_id=workspace_id,
        sources=sources,
        media_db=media_db,
        jobs=_list_recent_media_ingest_jobs(jm, current_user),
    )
    payload = build_workspace_capability_projection(
        workspace=workspace_projection,
        status_projection=status_payload,
        service_capabilities=await collect_workspace_service_capabilities(
            workspace_id=workspace_id,
            user_id=getattr(current_user, "id", None),
        ),
    )
    return WorkspaceCapabilitiesResponse(**payload)


@router.get(
    "/{workspace_id}/context",
    response_model=WorkspaceContextResponse,
    dependencies=[Depends(WORKSPACES_READ_RATE_LIMIT)],
    summary="Get workspace page context",
)
async def get_workspace_context(
    workspace_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    media_db: Any | None = Depends(try_get_media_db_for_user),
    jm: JobManager | None = Depends(try_get_workspace_job_manager),
    current_user: User = Depends(get_request_user),
) -> WorkspaceContextResponse:
    """Return the canonical read model for the workspace page shell."""
    workspace = _require_workspace(db, workspace_id)
    try:
        sources = db.list_workspace_sources(workspace_id)
        workspace_projection = _workspace_with_primary_root(db, workspace_id, workspace)
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to fetch workspace context") from exc

    jobs = _list_recent_media_ingest_jobs(jm, current_user)
    status_payload = build_source_status_projection(
        workspace_id=workspace_id,
        sources=sources,
        media_db=media_db,
        jobs=jobs,
    )
    capability_payload = build_workspace_capability_projection(
        workspace=workspace_projection,
        status_projection=status_payload,
        service_capabilities=await collect_workspace_service_capabilities(
            workspace_id=workspace_id,
            user_id=getattr(current_user, "id", None),
        ),
    )
    statuses_by_id = {
        str(source_status.get("id")): source_status
        for source_status in status_payload.get("sources", [])
    }
    partial_errors: list[dict[str, str]] = []
    if media_db is None and sources:
        partial_errors.append(
            {
                "scope": "sources",
                "code": "media_db_unavailable",
                "message": "Media database is unavailable; source readiness is conservative.",
            }
        )
    if jm is None and sources:
        partial_errors.append(
            {
                "scope": "jobs",
                "code": "jobs_unavailable",
                "message": "Jobs service is unavailable; in-flight ingestion progress may be incomplete.",
            }
        )

    context_sources = [
        _context_source_payload(
            workspace_id=workspace_id,
            source=source,
            source_status=statuses_by_id.get(str(source["id"]), {}),
        )
        for source in sources
    ]
    active_jobs = [
        _job_status_payload(job)
        for job in _active_workspace_jobs(jobs, sources, workspace_id)
    ]
    return WorkspaceContextResponse(
        workspace_id=workspace_id,
        workspace_profile=str(capability_payload.get("workspace_profile") or "research"),
        workspace_kind=str(capability_payload.get("workspace_kind") or "research_workspace"),
        schema_version=2,
        generated_at=_utc_now_iso(),
        workspace=_ws_to_response(workspace),
        resolution=capability_payload.get("resolution") or {},
        project_root=capability_payload.get("project_root") or {},
        sources={
            "items": context_sources,
            "summary": status_payload.get("summary") or {},
        },
        capabilities=WorkspaceCapabilitiesResponse(**capability_payload),
        services=capability_payload.get("workspace_services") or {},
        allowed_actions=capability_payload.get("allowed_actions") or {},
        active_jobs=active_jobs,
        partial_errors=partial_errors,
    )


@router.get(
    "/{workspace_id}/roots",
    response_model=WorkspaceRootsResponse,
    dependencies=[Depends(WORKSPACES_READ_RATE_LIMIT)],
    summary="List workspace project roots",
)
async def list_workspace_roots(
    workspace_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
) -> WorkspaceRootsResponse:
    """Return the read-only project root contract for a workspace."""
    _ = current_user
    try:
        workspace = _require_workspace(db, workspace_id)
        roots = db.list_workspace_project_roots(workspace_id)
    except HTTPException:
        raise
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to fetch workspace roots") from exc

    return _workspace_roots_response(workspace_id=workspace_id, workspace=workspace, roots=roots)


@router.put(
    "/{workspace_id}/roots/primary",
    response_model=WorkspaceRootsResponse,
    dependencies=[Depends(WORKSPACES_WRITE_RATE_LIMIT)],
    summary="Attach or replace the workspace primary project root",
)
async def attach_workspace_primary_root(
    workspace_id: str,
    body: WorkspacePrimaryRootAttachRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
) -> WorkspaceRootsResponse:
    """Attach or replace a workspace primary project root through the root-binding service."""
    try:
        _require_workspace(db, workspace_id)
        attach_primary_workspace_root(
            db=db,
            workspace_id=workspace_id,
            user_id=str(getattr(current_user, "id", "")),
            request=WorkspaceRootAttachRequest(**body.model_dump()),
        )
        workspace = _require_workspace(db, workspace_id)
        roots = db.list_workspace_project_roots(workspace_id)
    except WorkspaceRootServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    except HTTPException:
        raise
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(
            exc,
            default_detail="Failed to attach workspace primary root",
        ) from exc

    return _workspace_roots_response(workspace_id=workspace_id, workspace=workspace, roots=roots)


@router.post(
    "/{workspace_id}/file-inventory/scan",
    response_model=WorkspaceFileInventoryStatusResponse,
    dependencies=[Depends(WORKSPACES_WRITE_RATE_LIMIT)],
    summary="Queue a workspace project-root file inventory scan",
)
async def queue_workspace_file_inventory_scan(
    workspace_id: str,
    body: WorkspaceFileInventoryScanRequest,
    response: Response,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    jm: JobManager | None = Depends(try_get_workspace_job_manager),
    current_user: User = Depends(get_request_user),
) -> WorkspaceFileInventoryStatusResponse:
    """Queue a metadata-only inventory scan for the workspace primary root."""
    _require_workspace(db, workspace_id)
    root = db.get_workspace_primary_root(workspace_id)
    if root is None:
        raise _workspace_file_inventory_no_root_conflict()
    if body.expected_root_version is not None and int(root.get("version") or 0) != body.expected_root_version:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "root_version_mismatch",
                "message": "Workspace project root version does not match expected_root_version.",
            },
        )

    policy = build_inventory_ignore_policy()
    try:
        current_status = db.get_workspace_file_inventory_status(
            workspace_id,
            policy_fingerprint=policy.fingerprint,
        )
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(
            exc,
            default_detail="Failed to fetch workspace file inventory status",
        ) from exc
    if not body.force and current_status.get("state") == "current":
        response.status_code = status.HTTP_200_OK
        return _workspace_file_inventory_status_response(current_status)

    if jm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "jobs_unavailable",
                "message": "Jobs unavailable for workspace file inventory scans.",
            },
        )

    try:
        result = enqueue_workspace_file_inventory_scan_job(
            db=db,
            workspace_id=workspace_id,
            root_id=str(root["root_id"]),
            root_version=int(root["version"]),
            policy_fingerprint=policy.fingerprint,
            requested_by=str(getattr(current_user, "id", "")),
            owner_user_id=str(getattr(current_user, "id", "")),
            job_manager=jm,
        )
    except WorkspaceFileInventoryEnqueueError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": exc.error_code,
                "message": "Jobs unavailable for workspace file inventory scans.",
            },
        ) from exc
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(
            exc,
            input_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            default_detail="Failed to queue workspace file inventory scan",
        ) from exc

    response.status_code = status.HTTP_202_ACCEPTED
    return _workspace_file_inventory_status_response(
        result["status"],
        job=result.get("job"),
    )


@router.get(
    "/{workspace_id}/file-inventory/status",
    response_model=WorkspaceFileInventoryStatusResponse,
    dependencies=[Depends(WORKSPACES_READ_RATE_LIMIT)],
    summary="Get workspace project-root file inventory status",
)
async def get_workspace_file_inventory_status(
    workspace_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    jm: JobManager | None = Depends(try_get_workspace_job_manager),
    current_user: User = Depends(get_request_user),
) -> WorkspaceFileInventoryStatusResponse:
    """Return the latest durable file inventory scan status."""
    _ = current_user
    _require_workspace(db, workspace_id)
    try:
        status_payload = db.get_workspace_file_inventory_status(
            workspace_id,
            policy_fingerprint=build_inventory_ignore_policy().fingerprint,
        )
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(
            exc,
            default_detail="Failed to fetch workspace file inventory status",
        ) from exc
    job = _safe_get_job(jm, status_payload.get("job_id"))
    return _workspace_file_inventory_status_response(status_payload, job=job)


@router.get(
    "/{workspace_id}/file-inventory/items",
    response_model=WorkspaceFileInventoryItemsResponse,
    dependencies=[Depends(WORKSPACES_READ_RATE_LIMIT)],
    summary="List workspace project-root file inventory items",
)
async def list_workspace_file_inventory_items(
    workspace_id: str,
    prefix: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    cursor: str | None = Query(default=None),
    include_ignored: bool = Query(default=False),
    entry_kind: WorkspaceFileInventoryEntryKind | None = Query(default=None),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
) -> WorkspaceFileInventoryItemsResponse:
    """Return redacted, project-root-relative inventory items."""
    _ = current_user
    _require_workspace(db, workspace_id)
    root = db.get_workspace_primary_root(workspace_id)
    if root is None:
        raise _workspace_file_inventory_no_root_conflict()
    try:
        page = db.list_workspace_file_inventory_items(
            workspace_id,
            prefix=prefix,
            cursor=cursor,
            limit=limit,
            include_ignored=include_ignored,
            entry_kind=entry_kind,
        )
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(
            exc,
            input_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            default_detail="Failed to list workspace file inventory items",
        ) from exc
    return WorkspaceFileInventoryItemsResponse(
        workspace_id=workspace_id,
        root_id=root.get("root_id"),
        items=page.get("items") or [],
        next_cursor=page.get("next_cursor"),
        limit=limit,
    )


@router.get(
    "/{workspace_id}/sources/{source_id}/preview",
    response_model=WorkspaceSourcePreviewResponse,
    dependencies=[Depends(WORKSPACES_READ_RATE_LIMIT)],
    summary="Preview workspace source content and evidence",
)
async def get_source_preview(
    workspace_id: str,
    source_id: str,
    max_chars: int = Query(default=3000, ge=1, le=12000),
    chunk_limit: int = Query(default=3, ge=0, le=10),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    media_db: Any | None = Depends(try_get_media_db_for_user),
    jm: JobManager | None = Depends(try_get_workspace_job_manager),
    current_user: User = Depends(get_request_user),
) -> WorkspaceSourcePreviewResponse:
    """Return bounded captured text and chunk evidence for one workspace source."""
    _require_workspace(db, workspace_id)
    try:
        sources = db.list_workspace_sources(workspace_id)
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to fetch workspace source preview") from exc
    source = _find_source_in_workspace(sources, source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Workspace source not found")

    status_payload = build_source_status_projection(
        workspace_id=workspace_id,
        sources=[source],
        media_db=media_db,
        jobs=_list_recent_media_ingest_jobs(jm, current_user),
    )
    source_status = (status_payload.get("sources") or [{}])[0]
    payload = _source_preview_payload(
        workspace_id=workspace_id,
        source=source,
        source_status=source_status,
        media_db=media_db,
        max_chars=max_chars,
        chunk_limit=chunk_limit,
    )
    return WorkspaceSourcePreviewResponse(**payload)


@router.post(
    "/{workspace_id}/sources",
    response_model=WorkspaceSourceResponse,
    status_code=201,
    dependencies=[Depends(WORKSPACES_WRITE_RATE_LIMIT)],
    summary="Add source to workspace",
)
async def add_source(
    workspace_id: str,
    body: WorkspaceSourceCreateRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    jm: JobManager | None = Depends(try_get_workspace_job_manager),
    current_user: User = Depends(get_request_user),
) -> WorkspaceSourceResponse:
    """Add a media source to a workspace."""
    _require_workspace(db, workspace_id)
    try:
        src = db.add_workspace_source(workspace_id, body.model_dump())
        _enqueue_workspace_source_ingest_job(
            jm=jm,
            current_user=current_user,
            workspace_id=workspace_id,
            src=src,
        )
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to add workspace source") from exc
    return _src_to_response(src)


@router.put(
    "/{workspace_id}/sources/selection",
    response_model=StatusResponse,
    dependencies=[Depends(WORKSPACES_WRITE_RATE_LIMIT)],
    summary="Batch-update source selection",
)
async def update_source_selection(
    workspace_id: str,
    body: WorkspaceSourceSelectionRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
) -> StatusResponse:
    """Set which sources are selected in a workspace (batch operation)."""
    _require_workspace(db, workspace_id)
    try:
        db.update_workspace_source_selection(workspace_id, selected_ids=body.selected_ids)
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(
            exc, default_detail="Failed to update workspace source selection"
        ) from exc
    return StatusResponse()


@router.put(
    "/{workspace_id}/sources/reorder",
    response_model=StatusResponse,
    dependencies=[Depends(WORKSPACES_WRITE_RATE_LIMIT)],
    summary="Reorder workspace sources",
)
async def reorder_sources(
    workspace_id: str,
    body: WorkspaceSourceReorderRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
) -> StatusResponse:
    """Reorder workspace sources by providing an ordered list of IDs."""
    _require_workspace(db, workspace_id)
    try:
        db.reorder_workspace_sources(workspace_id, body.ordered_ids)
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to reorder workspace sources") from exc
    return StatusResponse()


@router.put(
    "/{workspace_id}/sources/{source_id}",
    response_model=WorkspaceSourceResponse,
    dependencies=[Depends(WORKSPACES_WRITE_RATE_LIMIT)],
    summary="Update workspace source",
)
async def update_source(
    workspace_id: str,
    source_id: str,
    body: WorkspaceSourceUpdateRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
) -> WorkspaceSourceResponse:
    """Update a workspace source with optimistic locking."""
    _require_workspace(db, workspace_id)
    updates = body.model_dump(exclude_unset=True, exclude={"version"})
    try:
        src = db.update_workspace_source(workspace_id, source_id, updates, expected_version=body.version)
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to update workspace source") from exc
    return _src_to_response(src)


@router.delete(
    "/{workspace_id}/sources/{source_id}",
    status_code=204,
    dependencies=[Depends(WORKSPACES_DELETE_RATE_LIMIT)],
    summary="Delete workspace source",
)
async def delete_source(
    workspace_id: str,
    source_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """Remove a source from a workspace."""
    _require_workspace(db, workspace_id)
    try:
        db.delete_workspace_source(workspace_id, source_id)
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to delete workspace source") from exc


# ── Artifacts ───────────────────────────────────────────────────

@router.get(
    "/{workspace_id}/artifacts",
    response_model=list[WorkspaceArtifactResponse],
    dependencies=[Depends(WORKSPACES_READ_RATE_LIMIT)],
    summary="List workspace artifacts",
)
async def list_artifacts(
    workspace_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
) -> list[WorkspaceArtifactResponse]:
    """List all artifacts belonging to a workspace."""
    _require_workspace(db, workspace_id)
    try:
        artifacts = db.list_workspace_artifacts(workspace_id)
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to fetch workspace artifacts") from exc
    return [_art_to_response(a) for a in artifacts]


@router.post(
    "/{workspace_id}/artifacts",
    response_model=WorkspaceArtifactResponse,
    status_code=201,
    dependencies=[Depends(WORKSPACES_WRITE_RATE_LIMIT)],
    summary="Add artifact to workspace",
)
async def add_artifact(
    workspace_id: str,
    body: WorkspaceArtifactCreateRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
) -> WorkspaceArtifactResponse:
    """Add an artifact to a workspace."""
    _require_workspace(db, workspace_id)
    try:
        art = db.add_workspace_artifact(workspace_id, body.model_dump())
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to add workspace artifact") from exc
    return _art_to_response(art)


@router.put(
    "/{workspace_id}/artifacts/{artifact_id}",
    response_model=WorkspaceArtifactResponse,
    dependencies=[Depends(WORKSPACES_WRITE_RATE_LIMIT)],
    summary="Update workspace artifact",
)
async def update_artifact(
    workspace_id: str,
    artifact_id: str,
    body: WorkspaceArtifactUpdateRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
) -> WorkspaceArtifactResponse:
    """Update a workspace artifact with optimistic locking."""
    _require_workspace(db, workspace_id)
    updates = body.model_dump(exclude_unset=True, exclude={"version"})
    try:
        art = db.update_workspace_artifact(workspace_id, artifact_id, updates, expected_version=body.version)
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to update workspace artifact") from exc
    return _art_to_response(art)


@router.post(
    "/{workspace_id}/artifacts/{artifact_id}/exports",
    response_model=WorkspaceArtifactExportResponse,
    dependencies=[Depends(WORKSPACES_WRITE_RATE_LIMIT)],
    summary="Export accepted workspace artifact version",
)
async def export_artifact(
    workspace_id: str,
    artifact_id: str,
    body: WorkspaceArtifactExportRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
) -> WorkspaceArtifactExportResponse:
    """Export an accepted workspace artifact version as Markdown, HTML, or JSON."""
    try:
        _require_workspace(db, workspace_id)
        artifact = _workspace_artifact_version_for_export(
            db,
            workspace_id,
            artifact_id,
            body.artifact_version_id,
        )
        payload = export_workspace_artifact_version(artifact, export_format=body.format)
        db.append_workspace_artifact_export_ref(workspace_id, artifact_id, payload["export_ref"])
    except WorkspaceArtifactExportStateError as exc:
        logger.warning(
            "Workspace artifact export rejected: workspace_id={} artifact_id={} artifact_version_id={} "
            "format={} reason={}",
            workspace_id,
            artifact_id,
            body.artifact_version_id,
            body.format,
            str(exc),
        )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        logger.exception(
            "Workspace artifact export failed: workspace_id={} artifact_id={} artifact_version_id={} format={}",
            workspace_id,
            artifact_id,
            body.artifact_version_id,
            body.format,
        )
        raise map_db_error_to_http(
            exc,
            default_detail="Failed to export workspace artifact",
            log_error=False,
        ) from exc
    return WorkspaceArtifactExportResponse(**payload)


@router.delete(
    "/{workspace_id}/artifacts/{artifact_id}",
    status_code=204,
    dependencies=[Depends(WORKSPACES_DELETE_RATE_LIMIT)],
    summary="Delete workspace artifact",
)
async def delete_artifact(
    workspace_id: str,
    artifact_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """Remove an artifact from a workspace."""
    _require_workspace(db, workspace_id)
    try:
        db.delete_workspace_artifact(workspace_id, artifact_id)
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to delete workspace artifact") from exc


# ── Notes ───────────────────────────────────────────────────────

@router.get(
    "/{workspace_id}/notes",
    response_model=list[WorkspaceNoteResponse],
    dependencies=[Depends(WORKSPACES_READ_RATE_LIMIT)],
    summary="List workspace notes",
)
async def list_notes(
    workspace_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
) -> list[WorkspaceNoteResponse]:
    """List all notes belonging to a workspace."""
    _require_workspace(db, workspace_id)
    try:
        notes = db.list_workspace_notes(workspace_id)
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to fetch workspace notes") from exc
    return [_note_to_response(n) for n in notes]


@router.post(
    "/{workspace_id}/notes",
    response_model=WorkspaceNoteResponse,
    status_code=201,
    dependencies=[Depends(WORKSPACES_WRITE_RATE_LIMIT)],
    summary="Add note to workspace",
)
async def add_note(
    workspace_id: str,
    body: WorkspaceNoteCreateRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
) -> WorkspaceNoteResponse:
    """Add a note to a workspace."""
    _require_workspace(db, workspace_id)
    try:
        note = db.add_workspace_note(workspace_id, body.model_dump())
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to add workspace note") from exc
    return _note_to_response(note)


@router.put(
    "/{workspace_id}/notes/{note_id}",
    response_model=WorkspaceNoteResponse,
    dependencies=[Depends(WORKSPACES_WRITE_RATE_LIMIT)],
    summary="Update workspace note",
)
async def update_note(
    workspace_id: str,
    note_id: int,
    body: WorkspaceNoteUpdateRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
) -> WorkspaceNoteResponse:
    """Update a workspace note with optimistic locking."""
    _require_workspace(db, workspace_id)
    updates = body.model_dump(exclude_unset=True, exclude={"version"})
    try:
        note = db.update_workspace_note(workspace_id, note_id, updates, expected_version=body.version)
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to update workspace note") from exc
    return _note_to_response(note)


@router.delete(
    "/{workspace_id}/notes/{note_id}",
    status_code=204,
    dependencies=[Depends(WORKSPACES_DELETE_RATE_LIMIT)],
    summary="Delete workspace note",
)
async def delete_note(
    workspace_id: str,
    note_id: int,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    current_user: User = Depends(get_request_user),
):
    """Remove a note from a workspace."""
    _require_workspace(db, workspace_id)
    try:
        db.delete_workspace_note(workspace_id, note_id)
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to delete workspace note") from exc
