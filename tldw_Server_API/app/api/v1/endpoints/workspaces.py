"""Workspace lifecycle CRUD endpoints."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import try_get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import get_job_manager
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.api.v1.schemas.workspace_schemas import (
    StatusResponse,
    WorkspaceArtifactCreateRequest,
    WorkspaceArtifactResponse,
    WorkspaceArtifactUpdateRequest,
    WorkspaceListResponse,
    WorkspaceNoteCreateRequest,
    WorkspaceNoteResponse,
    WorkspaceNoteUpdateRequest,
    WorkspacePatchRequest,
    WorkspaceCapabilitiesResponse,
    WorkspaceResponse,
    WorkspaceSourceCreateRequest,
    WorkspaceSourceReorderRequest,
    WorkspaceSourceResponse,
    WorkspaceSourceSelectionRequest,
    WorkspaceSourceStatusListResponse,
    WorkspaceSourceUpdateRequest,
    WorkspaceUpsertRequest,
)
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import (
    WORKSPACES_DELETE_RATE_LIMIT,
    WORKSPACES_READ_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Workspaces.status_projection import (
    build_source_status_projection,
    build_workspace_capability_projection,
)

router = APIRouter()

WORKSPACE_SOURCE_JOB_DOMAIN = "media_ingest"
WORKSPACE_SOURCE_JOB_QUEUE = "default"
WORKSPACE_SOURCE_JOB_TYPE = "workspace_source_ingest"
WORKSPACE_SOURCE_JOB_STAGES = ["ingestion", "extraction", "chunking", "indexing"]


def _ws_to_response(ws: dict) -> WorkspaceResponse:
    """Convert a workspace DB row dict to a WorkspaceResponse schema."""
    return WorkspaceResponse(
        id=ws["id"],
        name=ws.get("name"),
        archived=bool(ws.get("archived", False)),
        study_materials_policy=str(ws.get("study_materials_policy") or "general"),
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


def _art_to_response(art: dict) -> WorkspaceArtifactResponse:
    """Convert a workspace artifact DB row dict to a WorkspaceArtifactResponse schema."""
    return WorkspaceArtifactResponse(
        id=art["id"],
        workspace_id=art["workspace_id"],
        artifact_type=art["artifact_type"],
        title=art["title"],
        status=art.get("status", "pending"),
        content=art.get("content"),
        total_tokens=art.get("total_tokens"),
        total_cost_usd=art.get("total_cost_usd"),
        created_at=str(art.get("created_at", "")),
        completed_at=str(art["completed_at"]) if art.get("completed_at") else None,
        version=art.get("version", 1),
    )


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


def _find_workspace_source(db: CharactersRAGDB, workspace_id: str, source_id: str) -> dict[str, Any] | None:
    """Return an existing workspace source without requiring a public DB helper."""
    private_getter = getattr(db, "_get_workspace_source", None)
    if callable(private_getter):
        source = private_getter(workspace_id, source_id)
        return dict(source) if source else None
    for source in db.list_workspace_sources(workspace_id):
        if str(source.get("id")) == source_id:
            return source
    return None


def try_get_workspace_job_manager() -> JobManager | None:
    """Resolve the Jobs manager for workspace views without blocking workspace reads/writes."""
    try:
        return get_job_manager()
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
    except Exception:
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


def _enqueue_workspace_source_ingest_job(
    *,
    jm: JobManager | None,
    current_user: User,
    workspace_id: str,
    src: dict[str, Any],
) -> None:
    """Submit a source lifecycle job after the workspace source row exists."""
    if jm is None:
        return
    source_id = str(src["id"])
    media_id = int(src["media_id"])
    payload = {
        "workspace_id": workspace_id,
        "workspace_source_id": source_id,
        "source_id": source_id,
        "media_id": media_id,
        "source_type": str(src["source_type"]),
        "title": str(src["title"]),
        "url": src.get("url"),
        "requested_stages": WORKSPACE_SOURCE_JOB_STAGES,
    }
    try:
        jm.create_job(
            domain=WORKSPACE_SOURCE_JOB_DOMAIN,
            queue=WORKSPACE_SOURCE_JOB_QUEUE,
            job_type=WORKSPACE_SOURCE_JOB_TYPE,
            payload=payload,
            owner_user_id=str(current_user.id),
            idempotency_key=f"workspace-source:{workspace_id}:{source_id}:{media_id}",
            max_retries=3,
        )
    except Exception as exc:
        logger.warning(
            "Workspace source ingest job enqueue failed for workspace={} source={}: {}",
            workspace_id,
            source_id,
            exc,
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
    items = db.list_workspaces()
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
    ws = _require_workspace(db, workspace_id)
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
    ws = db.upsert_workspace(
        workspace_id,
        body.name,
        study_materials_policy=body.study_materials_policy,
    )
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
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
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
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


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
    return [_src_to_response(s) for s in db.list_workspace_sources(workspace_id)]


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
    payload = build_source_status_projection(
        workspace_id=workspace_id,
        sources=sources,
        media_db=media_db,
        jobs=_list_recent_media_ingest_jobs(jm, current_user),
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
    except (ConflictError, InputError, CharactersRAGDBError) as exc:
        raise map_db_error_to_http(exc, default_detail="Failed to fetch workspace capabilities") from exc
    status_payload = build_source_status_projection(
        workspace_id=workspace_id,
        sources=sources,
        media_db=media_db,
        jobs=_list_recent_media_ingest_jobs(jm, current_user),
    )
    payload = build_workspace_capability_projection(
        workspace=workspace,
        status_projection=status_payload,
    )
    return WorkspaceCapabilitiesResponse(**payload)


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
    src = _find_workspace_source(db, workspace_id, body.id)
    if src is None:
        try:
            src = db.add_workspace_source(workspace_id, body.model_dump())
        except (ConflictError, InputError, CharactersRAGDBError):
            src = _find_workspace_source(db, workspace_id, body.id)
            if src is None:
                raise
    _enqueue_workspace_source_ingest_job(
        jm=jm,
        current_user=current_user,
        workspace_id=workspace_id,
        src=src,
    )
    return _src_to_response(src)


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
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
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
    db.delete_workspace_source(workspace_id, source_id)


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
    db.update_workspace_source_selection(workspace_id, selected_ids=body.selected_ids)
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
    db.reorder_workspace_sources(workspace_id, body.ordered_ids)
    return StatusResponse()


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
    return [_art_to_response(a) for a in db.list_workspace_artifacts(workspace_id)]


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
    art = db.add_workspace_artifact(workspace_id, body.model_dump())
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
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return _art_to_response(art)


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
    db.delete_workspace_artifact(workspace_id, artifact_id)


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
    return [_note_to_response(n) for n in db.list_workspace_notes(workspace_id)]


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
    note = db.add_workspace_note(workspace_id, body.model_dump())
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
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
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
    db.delete_workspace_note(workspace_id, note_id)
