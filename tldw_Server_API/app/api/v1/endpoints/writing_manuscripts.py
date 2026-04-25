"""Manuscript management API endpoints (projects, parts, chapters, scenes)."""
from __future__ import annotations

import json
from typing import Any, NoReturn

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_rate_limiter_dep,
    rbac_rate_limit,
)
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.schemas.writing_manuscript_schemas import (
    ChapterSummary,
    ManuscriptAnalysisListResponse,
    ManuscriptAnalysisRequest,
    ManuscriptAnalysisResponse,
    ManuscriptCharacterCreate,
    ManuscriptCharacterResponse,
    ManuscriptCharacterUpdate,
    ManuscriptChapterCreate,
    ManuscriptChapterResponse,
    ManuscriptChapterUpdate,
    ManuscriptCitationCreate,
    ManuscriptCitationResponse,
    ManuscriptPartCreate,
    ManuscriptPartResponse,
    ManuscriptPartUpdate,
    ManuscriptPlotEventCreate,
    ManuscriptPlotEventResponse,
    ManuscriptPlotEventUpdate,
    ManuscriptPlotHoleCreate,
    ManuscriptPlotHoleResponse,
    ManuscriptPlotHoleUpdate,
    ManuscriptPlotLineCreate,
    ManuscriptPlotLineResponse,
    ManuscriptPlotLineUpdate,
    ManuscriptProjectCreate,
    ManuscriptProjectListResponse,
    ManuscriptProjectResponse,
    ManuscriptProjectUpdate,
    ManuscriptRelationshipCreate,
    ManuscriptRelationshipResponse,
    ManuscriptResearchRequest,
    ManuscriptResearchResponse,
    ManuscriptSceneCreate,
    ManuscriptSceneResponse,
    ManuscriptSceneUpdate,
    ManuscriptSearchResponse,
    ManuscriptSearchResult,
    ManuscriptStructureResponse,
    ManuscriptWorldInfoCreate,
    ManuscriptWorldInfoResponse,
    ManuscriptWorldInfoUpdate,
    PartSummary,
    ReorderRequest,
    SceneCharacterLink,
    SceneCharacterLinkResponse,
    SceneSummary,
    SceneWorldInfoLink,
    SceneWorldInfoLinkResponse,
    CHARACTER_ROLES,
    WORLD_INFO_KINDS,
)
from tldw_Server_API.app.core.AuthNZ.rate_limiter import RateLimiter
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.Chat.chat_service import is_model_known_for_provider
from tldw_Server_API.app.core.Chat.provider_manager import get_provider_manager
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.ManuscriptDB import ManuscriptDBHelper
from tldw_Server_API.app.core.LLM_Calls.provider_metadata import PROVIDER_CAPABILITIES

router = APIRouter()

# ---------------------------------------------------------------------------
# Exception tuple and helpers (mirrors writing.py)
# ---------------------------------------------------------------------------

_MANUSCRIPT_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    CharactersRAGDBError,
    ConflictError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    InputError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
)


async def _enforce_rate_limit(rate_limiter: RateLimiter, user_id: int, scope: str) -> None:
    """Enforce a rate limit for the given user and scope."""
    try:
        allowed, meta = await rate_limiter.check_user_rate_limit(int(user_id), scope)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        retry_after = 60
        logger.exception(
            "Rate limiter check failed for user_id={} scope={}",
            user_id,
            scope,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Rate limiter unavailable",
            headers={"Retry-After": str(retry_after)},
        ) from exc
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for {scope}",
            headers={"Retry-After": str(meta.get("retry_after", 60))},
        )
def _handle_db_errors(exc: Exception, entity_label: str) -> NoReturn:
    """Translate database exceptions into HTTP errors."""
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, InputError):
        logger.warning("Input error for {}: {}", entity_label, exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if isinstance(exc, ConflictError):
        message = str(exc)
        lowered = message.lower()
        # Check "not found" / "soft-deleted" first — the message may also contain
        # "version conflict" (e.g. "version conflict or not found"), and a pure 404
        # should not be reported as 409.
        if ("not found" in lowered or "soft-deleted" in lowered or "soft deleted" in lowered) and "version conflict" not in lowered:
            logger.debug("Entity not found for {}: {}", entity_label, exc)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"{entity_label} not found"
            ) from exc
        logger.warning("Conflict error for {}: {}", entity_label, exc)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=message) from exc
    if isinstance(exc, CharactersRAGDBError):
        logger.error("Database error for {}: {}", entity_label, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error while processing {entity_label}",
        ) from exc
    logger.exception("Unexpected error for {}: {}", entity_label, exc)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Unexpected error while processing {entity_label}",
    ) from exc


def _get_helper(db: CharactersRAGDB) -> ManuscriptDBHelper:
    """Construct a ManuscriptDBHelper from the per-user DB dependency."""
    return ManuscriptDBHelper(db)


def _field_present(payload: Any, field_name: str) -> bool:
    """Return True when a field was explicitly included in the request payload."""
    return field_name in payload.model_fields_set


def _require_non_empty_text(value: str | None, label: str) -> str:
    """Normalize required text fields and reject null/blank values."""
    if value is None:
        raise ValueError(f"{label} cannot be null")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} cannot be empty")
    return normalized


def _normalize_mapping_field(value: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize nullable mapping fields for storage."""
    return {} if value is None else value


# ===================================================================
# Projects
# ===================================================================


@router.get(
    "/projects",
    response_model=ManuscriptProjectListResponse,
    summary="List manuscript projects",
    tags=["manuscripts"],
)
async def list_projects(
    status_filter: str | None = Query(None, alias="status", description="Filter by status"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.list")),
) -> ManuscriptProjectListResponse:
    """List manuscript projects for the current user."""
    try:
        helper = _get_helper(db)
        projects, total = helper.list_projects(
            status_filter=status_filter, limit=limit, offset=offset
        )
        items = [ManuscriptProjectResponse(**p) for p in projects]
        return ManuscriptProjectListResponse(projects=items, total=total)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript projects")


@router.post(
    "/projects",
    response_model=ManuscriptProjectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a manuscript project",
    tags=["manuscripts"],
)
async def create_project(
    payload: ManuscriptProjectCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.create")),
) -> ManuscriptProjectResponse:
    """Create a new manuscript project."""
    try:
        title = _require_non_empty_text(payload.title, "title")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        helper = _get_helper(db)
        project_id = helper.create_project(
            title=title,
            subtitle=payload.subtitle,
            author=payload.author,
            genre=payload.genre,
            status=payload.status,
            synopsis=payload.synopsis,
            target_word_count=payload.target_word_count,
            settings=_normalize_mapping_field(payload.settings),
            project_id=payload.id,
        )
        project = helper.get_project(project_id)
        if not project:
            raise CharactersRAGDBError("Project created but could not be retrieved")
        return ManuscriptProjectResponse(**project)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript project")


@router.get(
    "/projects/{project_id}",
    response_model=ManuscriptProjectResponse,
    summary="Get a manuscript project",
    tags=["manuscripts"],
)
async def get_project(
    project_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.get")),
) -> ManuscriptProjectResponse:
    """Fetch a manuscript project by ID."""
    try:
        helper = _get_helper(db)
        project = helper.get_project(project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
            )
        return ManuscriptProjectResponse(**project)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript project")


@router.patch(
    "/projects/{project_id}",
    response_model=ManuscriptProjectResponse,
    summary="Update a manuscript project",
    tags=["manuscripts"],
)
async def update_project(
    project_id: str,
    payload: ManuscriptProjectUpdate,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.update")),
) -> ManuscriptProjectResponse:
    """Update a manuscript project with optimistic locking."""
    update_data: dict[str, Any] = {}
    if _field_present(payload, "title"):
        try:
            update_data["title"] = _require_non_empty_text(payload.title, "title")
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if _field_present(payload, "subtitle"):
        update_data["subtitle"] = payload.subtitle
    if _field_present(payload, "author"):
        update_data["author"] = payload.author
    if _field_present(payload, "genre"):
        update_data["genre"] = payload.genre
    if _field_present(payload, "status"):
        if payload.status is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="status cannot be null",
            )
        update_data["status"] = payload.status
    if _field_present(payload, "synopsis"):
        update_data["synopsis"] = payload.synopsis
    if _field_present(payload, "target_word_count"):
        update_data["target_word_count"] = payload.target_word_count
    if _field_present(payload, "settings"):
        update_data["settings"] = _normalize_mapping_field(payload.settings)
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided for update"
        )
    try:
        helper = _get_helper(db)
        helper.update_project(project_id, update_data, expected_version)
        project = helper.get_project(project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
            )
        return ManuscriptProjectResponse(**project)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript project")


@router.delete(
    "/projects/{project_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a manuscript project",
    tags=["manuscripts"],
)
async def delete_project(
    project_id: str,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.delete")),
) -> Response:
    """Soft-delete a manuscript project."""
    try:
        helper = _get_helper(db)
        helper.soft_delete_project(project_id, expected_version)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript project")


@router.get(
    "/projects/{project_id}/structure",
    response_model=ManuscriptStructureResponse,
    summary="Get full manuscript structure tree",
    tags=["manuscripts"],
)
async def get_project_structure(
    project_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.get")),
) -> ManuscriptStructureResponse:
    """Build the hierarchical structure of a manuscript project."""
    try:
        helper = _get_helper(db)
        # Verify project exists
        project = helper.get_project(project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
            )
        raw = helper.get_project_structure(project_id)

        def _scene_summary(s: dict[str, Any]) -> SceneSummary:
            return SceneSummary(
                id=s["id"],
                title=s["title"],
                sort_order=s["sort_order"],
                word_count=s.get("word_count", 0),
                status=s.get("status", "draft"),
                version=s.get("version", 1),
            )

        def _chapter_summary(c: dict[str, Any]) -> ChapterSummary:
            return ChapterSummary(
                id=c["id"],
                title=c["title"],
                sort_order=c["sort_order"],
                part_id=c.get("part_id"),
                word_count=c.get("word_count", 0),
                status=c.get("status", "draft"),
                version=c.get("version", 1),
                scenes=[_scene_summary(s) for s in c.get("scenes", [])],
            )

        def _part_summary(p: dict[str, Any]) -> PartSummary:
            return PartSummary(
                id=p["id"],
                title=p["title"],
                sort_order=p["sort_order"],
                word_count=p.get("word_count", 0),
                version=p.get("version", 1),
                chapters=[_chapter_summary(c) for c in p.get("chapters", [])],
            )

        return ManuscriptStructureResponse(
            project_id=project_id,
            parts=[_part_summary(p) for p in raw.get("parts", [])],
            unassigned_chapters=[
                _chapter_summary(c) for c in raw.get("unassigned_chapters", [])
            ],
        )
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript structure")


@router.post(
    "/projects/{project_id}/reorder",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Batch reorder parts, chapters, or scenes",
    tags=["manuscripts"],
)
async def reorder_entities(
    project_id: str,
    payload: ReorderRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.update")),
) -> Response:
    """Batch-update sort_order for parts, chapters, or scenes within a project."""
    # Map plural form from schema to singular form used by ManuscriptDBHelper
    entity_type_map = {"parts": "part", "chapters": "chapter", "scenes": "scene"}
    entity_type = entity_type_map[payload.entity_type]

    items = []
    for item in payload.items:
        entry: dict[str, Any] = {"id": item.id, "sort_order": item.sort_order, "version": item.version}
        if item.new_parent_id is not None and entity_type == "chapter":
            entry["part_id"] = item.new_parent_id
        items.append(entry)

    try:
        helper = _get_helper(db)
        helper.reorder_items(entity_type, items, project_id=project_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript reorder")


@router.get(
    "/projects/{project_id}/search",
    response_model=ManuscriptSearchResponse,
    summary="Full-text search across scenes in a project",
    tags=["manuscripts"],
)
async def search_project(
    project_id: str,
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.get")),
) -> ManuscriptSearchResponse:
    """Search scenes within a project using FTS5."""
    try:
        helper = _get_helper(db)
        rows = helper.search_scenes(project_id, q, limit=limit)
        results = [
            ManuscriptSearchResult(
                id=r["id"],
                title=r["title"],
                chapter_id=r["chapter_id"],
                word_count=r.get("word_count", 0),
                status=r.get("status", "draft"),
                snippet=r.get("snippet"),
            )
            for r in rows
        ]
        return ManuscriptSearchResponse(query=q, results=results)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript search")


# ===================================================================
# Parts
# ===================================================================


@router.post(
    "/projects/{project_id}/parts",
    response_model=ManuscriptPartResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a manuscript part",
    tags=["manuscripts"],
)
async def create_part(
    project_id: str,
    payload: ManuscriptPartCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.create")),
) -> ManuscriptPartResponse:
    """Create a new part within a project."""
    try:
        title = _require_non_empty_text(payload.title, "title")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        helper = _get_helper(db)
        part_id = helper.create_part(
            project_id=project_id,
            title=title,
            sort_order=payload.sort_order,
            synopsis=payload.synopsis,
            part_id=payload.id,
        )
        part = helper.get_part(part_id)
        if not part:
            raise CharactersRAGDBError("Part created but could not be retrieved")
        return ManuscriptPartResponse(**part)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript part")


@router.get(
    "/projects/{project_id}/parts",
    response_model=list[ManuscriptPartResponse],
    summary="List parts in a project",
    tags=["manuscripts"],
)
async def list_parts(
    project_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.list")),
) -> list[ManuscriptPartResponse]:
    """List all parts within a project."""
    try:
        helper = _get_helper(db)
        parts = helper.list_parts(project_id)
        return [ManuscriptPartResponse(**p) for p in parts]
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript parts")


@router.get(
    "/parts/{part_id}",
    response_model=ManuscriptPartResponse,
    summary="Get a manuscript part",
    tags=["manuscripts"],
)
async def get_part(
    part_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.get")),
) -> ManuscriptPartResponse:
    """Fetch a manuscript part by ID."""
    try:
        helper = _get_helper(db)
        part = helper.get_part(part_id)
        if not part:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Part not found"
            )
        return ManuscriptPartResponse(**part)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript part")


@router.patch(
    "/parts/{part_id}",
    response_model=ManuscriptPartResponse,
    summary="Update a manuscript part",
    tags=["manuscripts"],
)
async def update_part(
    part_id: str,
    payload: ManuscriptPartUpdate,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.update")),
) -> ManuscriptPartResponse:
    """Update a manuscript part with optimistic locking."""
    update_data: dict[str, Any] = {}
    if _field_present(payload, "title"):
        try:
            update_data["title"] = _require_non_empty_text(payload.title, "title")
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if _field_present(payload, "sort_order") and payload.sort_order is not None:
        update_data["sort_order"] = payload.sort_order
    if _field_present(payload, "synopsis"):
        update_data["synopsis"] = payload.synopsis
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided for update"
        )
    try:
        helper = _get_helper(db)
        helper.update_part(part_id, update_data, expected_version)
        part = helper.get_part(part_id)
        if not part:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Part not found"
            )
        return ManuscriptPartResponse(**part)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript part")


@router.delete(
    "/parts/{part_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a manuscript part",
    tags=["manuscripts"],
)
async def delete_part(
    part_id: str,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.delete")),
) -> Response:
    """Soft-delete a manuscript part."""
    try:
        helper = _get_helper(db)
        helper.soft_delete_part(part_id, expected_version)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript part")


# ===================================================================
# Chapters
# ===================================================================


@router.post(
    "/projects/{project_id}/chapters",
    response_model=ManuscriptChapterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a manuscript chapter",
    tags=["manuscripts"],
)
async def create_chapter(
    project_id: str,
    payload: ManuscriptChapterCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.create")),
) -> ManuscriptChapterResponse:
    """Create a new chapter within a project."""
    try:
        title = _require_non_empty_text(payload.title, "title")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        helper = _get_helper(db)
        chapter_id = helper.create_chapter(
            project_id=project_id,
            title=title,
            part_id=payload.part_id,
            sort_order=payload.sort_order,
            synopsis=payload.synopsis,
            status=payload.status,
            chapter_id=payload.id,
        )
        chapter = helper.get_chapter(chapter_id)
        if not chapter:
            raise CharactersRAGDBError("Chapter created but could not be retrieved")
        return ManuscriptChapterResponse(**chapter)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript chapter")


@router.get(
    "/projects/{project_id}/chapters",
    response_model=list[ManuscriptChapterResponse],
    summary="List chapters in a project",
    tags=["manuscripts"],
)
async def list_chapters(
    project_id: str,
    part_id: str | None = Query(None, description="Filter by part ID"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.list")),
) -> list[ManuscriptChapterResponse]:
    """List chapters within a project, optionally filtered by part."""
    try:
        helper = _get_helper(db)
        chapters = helper.list_chapters(project_id, part_id=part_id)
        return [ManuscriptChapterResponse(**c) for c in chapters]
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript chapters")


@router.get(
    "/chapters/{chapter_id}",
    response_model=ManuscriptChapterResponse,
    summary="Get a manuscript chapter",
    tags=["manuscripts"],
)
async def get_chapter(
    chapter_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.get")),
) -> ManuscriptChapterResponse:
    """Fetch a manuscript chapter by ID."""
    try:
        helper = _get_helper(db)
        chapter = helper.get_chapter(chapter_id)
        if not chapter:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Chapter not found"
            )
        return ManuscriptChapterResponse(**chapter)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript chapter")


@router.patch(
    "/chapters/{chapter_id}",
    response_model=ManuscriptChapterResponse,
    summary="Update a manuscript chapter",
    tags=["manuscripts"],
)
async def update_chapter(
    chapter_id: str,
    payload: ManuscriptChapterUpdate,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.update")),
) -> ManuscriptChapterResponse:
    """Update a manuscript chapter with optimistic locking."""
    update_data: dict[str, Any] = {}
    if _field_present(payload, "title"):
        try:
            update_data["title"] = _require_non_empty_text(payload.title, "title")
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if _field_present(payload, "part_id"):
        update_data["part_id"] = payload.part_id
    if _field_present(payload, "sort_order") and payload.sort_order is not None:
        update_data["sort_order"] = payload.sort_order
    if _field_present(payload, "synopsis"):
        update_data["synopsis"] = payload.synopsis
    if _field_present(payload, "status"):
        if payload.status is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="status cannot be null",
            )
        update_data["status"] = payload.status
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided for update"
        )
    try:
        helper = _get_helper(db)
        helper.update_chapter(chapter_id, update_data, expected_version)
        chapter = helper.get_chapter(chapter_id)
        if not chapter:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Chapter not found"
            )
        return ManuscriptChapterResponse(**chapter)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript chapter")


@router.delete(
    "/chapters/{chapter_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a manuscript chapter",
    tags=["manuscripts"],
)
async def delete_chapter(
    chapter_id: str,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.delete")),
) -> Response:
    """Soft-delete a manuscript chapter."""
    try:
        helper = _get_helper(db)
        helper.soft_delete_chapter(chapter_id, expected_version)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript chapter")


# ===================================================================
# Scenes
# ===================================================================


@router.post(
    "/chapters/{chapter_id}/scenes",
    response_model=ManuscriptSceneResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a manuscript scene",
    tags=["manuscripts"],
)
async def create_scene(
    chapter_id: str,
    payload: ManuscriptSceneCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.create")),
) -> ManuscriptSceneResponse:
    """Create a new scene within a chapter.

    The project_id is resolved from the chapter's parent project.
    """
    try:
        title = _require_non_empty_text(payload.title, "title")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        helper = _get_helper(db)
        # Resolve project_id from the chapter
        chapter = helper.get_chapter(chapter_id)
        if not chapter:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Chapter not found"
            )
        project_id = chapter["project_id"]

        content_json = None
        if payload.content is not None:
            content_json = json.dumps(payload.content)

        scene_id = helper.create_scene(
            chapter_id=chapter_id,
            project_id=project_id,
            title=title,
            content_json=content_json,
            content_plain=payload.content_plain,
            synopsis=payload.synopsis,
            sort_order=payload.sort_order,
            status=payload.status,
            scene_id=payload.id,
        )
        scene = helper.get_scene(scene_id)
        if not scene:
            raise CharactersRAGDBError("Scene created but could not be retrieved")
        return ManuscriptSceneResponse(**scene)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript scene")


@router.get(
    "/chapters/{chapter_id}/scenes",
    response_model=list[ManuscriptSceneResponse],
    summary="List scenes in a chapter",
    tags=["manuscripts"],
)
async def list_scenes(
    chapter_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.list")),
) -> list[ManuscriptSceneResponse]:
    """List all scenes within a chapter."""
    try:
        helper = _get_helper(db)
        scenes = helper.list_scenes(chapter_id)
        return [ManuscriptSceneResponse(**s) for s in scenes]
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript scenes")


@router.get(
    "/scenes/{scene_id}",
    response_model=ManuscriptSceneResponse,
    summary="Get a manuscript scene",
    tags=["manuscripts"],
)
async def get_scene(
    scene_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.get")),
) -> ManuscriptSceneResponse:
    """Fetch a manuscript scene by ID."""
    try:
        helper = _get_helper(db)
        scene = helper.get_scene(scene_id)
        if not scene:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Scene not found"
            )
        return ManuscriptSceneResponse(**scene)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript scene")


@router.patch(
    "/scenes/{scene_id}",
    response_model=ManuscriptSceneResponse,
    summary="Update a manuscript scene",
    tags=["manuscripts"],
)
async def update_scene(
    scene_id: str,
    payload: ManuscriptSceneUpdate,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.update")),
) -> ManuscriptSceneResponse:
    """Update a manuscript scene with optimistic locking.

    When ``content`` (TipTap JSON dict) is provided, it is serialised to
    ``content_json`` before storage.
    """
    update_data: dict[str, Any] = {}
    if _field_present(payload, "title"):
        try:
            update_data["title"] = _require_non_empty_text(payload.title, "title")
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if _field_present(payload, "content"):
        update_data["content_json"] = (
            json.dumps(payload.content) if payload.content is not None else None
        )
    elif _field_present(payload, "content_plain"):
        # Plain-text-only edit: clear stale rich content so they don't diverge
        update_data["content_json"] = None
    if _field_present(payload, "content_plain"):
        update_data["content_plain"] = payload.content_plain
    if _field_present(payload, "synopsis"):
        update_data["synopsis"] = payload.synopsis
    if _field_present(payload, "sort_order") and payload.sort_order is not None:
        update_data["sort_order"] = payload.sort_order
    if _field_present(payload, "status"):
        if payload.status is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="status cannot be null",
            )
        update_data["status"] = payload.status
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided for update"
        )
    try:
        helper = _get_helper(db)
        helper.update_scene(scene_id, update_data, expected_version)
        scene = helper.get_scene(scene_id)
        if not scene:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Scene not found"
            )
        if "content_json" in update_data:
            scene["content_json"] = update_data["content_json"]
        return ManuscriptSceneResponse(**scene)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript scene")


@router.delete(
    "/scenes/{scene_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a manuscript scene",
    tags=["manuscripts"],
)
async def delete_scene(
    scene_id: str,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.delete")),
) -> Response:
    """Soft-delete a manuscript scene."""
    try:
        helper = _get_helper(db)
        helper.soft_delete_scene(scene_id, expected_version)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript scene")


# ===================================================================
# Characters
# ===================================================================


@router.post(
    "/projects/{project_id}/characters",
    response_model=ManuscriptCharacterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a manuscript character",
    tags=["manuscripts"],
)
async def create_character(
    project_id: str,
    payload: ManuscriptCharacterCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.create")),
) -> ManuscriptCharacterResponse:
    """Create a new character within a project."""
    try:
        name = _require_non_empty_text(payload.name, "name")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        helper = _get_helper(db)
        character_id = helper.create_character(
            project_id=project_id,
            name=name,
            role=payload.role,
            cast_group=payload.cast_group,
            full_name=payload.full_name,
            age=payload.age,
            gender=payload.gender,
            appearance=payload.appearance,
            personality=payload.personality,
            backstory=payload.backstory,
            motivation=payload.motivation,
            arc_summary=payload.arc_summary,
            notes=payload.notes,
            custom_fields=_normalize_mapping_field(payload.custom_fields),
            sort_order=payload.sort_order,
            character_id=payload.id,
        )
        character = helper.get_character(character_id)
        if not character:
            raise CharactersRAGDBError("Character created but could not be retrieved")
        return ManuscriptCharacterResponse(**character)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript character")


@router.get(
    "/projects/{project_id}/characters",
    response_model=list[ManuscriptCharacterResponse],
    summary="List characters in a project",
    tags=["manuscripts"],
)
async def list_characters(
    project_id: str,
    role: CHARACTER_ROLES | None = Query(None, description="Filter by role"),
    cast_group: str | None = Query(None, description="Filter by cast group"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.list")),
) -> list[ManuscriptCharacterResponse]:
    """List all characters within a project, optionally filtered by role or cast group."""
    try:
        helper = _get_helper(db)
        characters = helper.list_characters(
            project_id, role_filter=role, cast_group_filter=cast_group,
        )
        return [ManuscriptCharacterResponse(**c) for c in characters]
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript characters")


@router.get(
    "/characters/{character_id}",
    response_model=ManuscriptCharacterResponse,
    summary="Get a manuscript character",
    tags=["manuscripts"],
)
async def get_character(
    character_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.get")),
) -> ManuscriptCharacterResponse:
    """Fetch a manuscript character by ID."""
    try:
        helper = _get_helper(db)
        character = helper.get_character(character_id)
        if not character:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Character not found"
            )
        return ManuscriptCharacterResponse(**character)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript character")


@router.patch(
    "/characters/{character_id}",
    response_model=ManuscriptCharacterResponse,
    summary="Update a manuscript character",
    tags=["manuscripts"],
)
async def update_character(
    character_id: str,
    payload: ManuscriptCharacterUpdate,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.update")),
) -> ManuscriptCharacterResponse:
    """Update a manuscript character with optimistic locking."""
    update_data: dict[str, Any] = {}
    if _field_present(payload, "name"):
        try:
            update_data["name"] = _require_non_empty_text(payload.name, "name")
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if _field_present(payload, "role") and payload.role is not None:
        update_data["role"] = payload.role
    if _field_present(payload, "cast_group"):
        update_data["cast_group"] = payload.cast_group
    if _field_present(payload, "full_name"):
        update_data["full_name"] = payload.full_name
    if _field_present(payload, "age"):
        update_data["age"] = payload.age
    if _field_present(payload, "gender"):
        update_data["gender"] = payload.gender
    if _field_present(payload, "appearance"):
        update_data["appearance"] = payload.appearance
    if _field_present(payload, "personality"):
        update_data["personality"] = payload.personality
    if _field_present(payload, "backstory"):
        update_data["backstory"] = payload.backstory
    if _field_present(payload, "motivation"):
        update_data["motivation"] = payload.motivation
    if _field_present(payload, "arc_summary"):
        update_data["arc_summary"] = payload.arc_summary
    if _field_present(payload, "notes"):
        update_data["notes"] = payload.notes
    if _field_present(payload, "custom_fields"):
        update_data["custom_fields"] = _normalize_mapping_field(payload.custom_fields)
    if _field_present(payload, "sort_order") and payload.sort_order is not None:
        update_data["sort_order"] = payload.sort_order
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided for update"
        )
    try:
        helper = _get_helper(db)
        helper.update_character(character_id, update_data, expected_version)
        character = helper.get_character(character_id)
        if not character:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Character not found"
            )
        return ManuscriptCharacterResponse(**character)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript character")


@router.delete(
    "/characters/{character_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a manuscript character",
    tags=["manuscripts"],
)
async def delete_character(
    character_id: str,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.delete")),
) -> Response:
    """Soft-delete a manuscript character."""
    try:
        helper = _get_helper(db)
        helper.soft_delete_character(character_id, expected_version)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript character")


# ===================================================================
# Character Relationships
# ===================================================================


@router.post(
    "/projects/{project_id}/characters/relationships",
    response_model=ManuscriptRelationshipResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a character relationship",
    tags=["manuscripts"],
)
async def create_relationship(
    project_id: str,
    payload: ManuscriptRelationshipCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.create")),
) -> ManuscriptRelationshipResponse:
    """Create a relationship between two characters."""
    try:
        helper = _get_helper(db)
        rel_id = helper.create_relationship(
            project_id=project_id,
            from_character_id=payload.from_character_id,
            to_character_id=payload.to_character_id,
            relationship_type=payload.relationship_type,
            description=payload.description,
            bidirectional=payload.bidirectional,
            relationship_id=payload.id,
        )
        rel = helper.get_relationship(rel_id)
        if not rel:
            raise CharactersRAGDBError("Relationship created but could not be retrieved")
        return ManuscriptRelationshipResponse(**rel)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript relationship")


@router.get(
    "/projects/{project_id}/characters/relationships",
    response_model=list[ManuscriptRelationshipResponse],
    summary="List character relationships",
    tags=["manuscripts"],
)
async def list_relationships(
    project_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.list")),
) -> list[ManuscriptRelationshipResponse]:
    """List all character relationships within a project."""
    try:
        helper = _get_helper(db)
        rels = helper.list_relationships(project_id)
        return [ManuscriptRelationshipResponse(**r) for r in rels]
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript relationships")


@router.delete(
    "/characters/relationships/{relationship_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a character relationship",
    tags=["manuscripts"],
)
async def delete_relationship(
    relationship_id: str,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.delete")),
) -> Response:
    """Soft-delete a character relationship."""
    try:
        helper = _get_helper(db)
        helper.soft_delete_relationship(relationship_id, expected_version)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript relationship")


# ===================================================================
# World Info
# ===================================================================


@router.post(
    "/projects/{project_id}/world-info",
    response_model=ManuscriptWorldInfoResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a world-info entry",
    tags=["manuscripts"],
)
async def create_world_info(
    project_id: str,
    payload: ManuscriptWorldInfoCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.create")),
) -> ManuscriptWorldInfoResponse:
    """Create a new world-info entry within a project."""
    try:
        name = _require_non_empty_text(payload.name, "name")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        helper = _get_helper(db)
        item_id = helper.create_world_info(
            project_id=project_id,
            kind=payload.kind,
            name=name,
            description=payload.description,
            parent_id=payload.parent_id,
            properties=_normalize_mapping_field(payload.properties),
            tags=payload.tags or [],
            sort_order=payload.sort_order,
            world_info_id=payload.id,
        )
        item = helper.get_world_info(item_id)
        if not item:
            raise CharactersRAGDBError("World info created but could not be retrieved")
        return ManuscriptWorldInfoResponse(**item)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript world info")


@router.get(
    "/projects/{project_id}/world-info",
    response_model=list[ManuscriptWorldInfoResponse],
    summary="List world-info entries in a project",
    tags=["manuscripts"],
)
async def list_world_info(
    project_id: str,
    kind: WORLD_INFO_KINDS | None = Query(None, description="Filter by kind"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.list")),
) -> list[ManuscriptWorldInfoResponse]:
    """List all world-info entries within a project."""
    try:
        helper = _get_helper(db)
        items = helper.list_world_info(project_id, kind_filter=kind)
        return [ManuscriptWorldInfoResponse(**i) for i in items]
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript world info")


@router.get(
    "/world-info/{item_id}",
    response_model=ManuscriptWorldInfoResponse,
    summary="Get a world-info entry",
    tags=["manuscripts"],
)
async def get_world_info(
    item_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.get")),
) -> ManuscriptWorldInfoResponse:
    """Fetch a world-info entry by ID."""
    try:
        helper = _get_helper(db)
        item = helper.get_world_info(item_id)
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="World info not found"
            )
        return ManuscriptWorldInfoResponse(**item)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript world info")


@router.patch(
    "/world-info/{item_id}",
    response_model=ManuscriptWorldInfoResponse,
    summary="Update a world-info entry",
    tags=["manuscripts"],
)
async def update_world_info(
    item_id: str,
    payload: ManuscriptWorldInfoUpdate,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.update")),
) -> ManuscriptWorldInfoResponse:
    """Update a world-info entry with optimistic locking."""
    update_data: dict[str, Any] = {}
    if _field_present(payload, "name"):
        try:
            update_data["name"] = _require_non_empty_text(payload.name, "name")
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if _field_present(payload, "description"):
        update_data["description"] = payload.description
    if _field_present(payload, "parent_id"):
        update_data["parent_id"] = payload.parent_id
    if _field_present(payload, "properties"):
        update_data["properties"] = _normalize_mapping_field(payload.properties)
    if _field_present(payload, "tags"):
        update_data["tags"] = [] if payload.tags is None else payload.tags
    if _field_present(payload, "sort_order") and payload.sort_order is not None:
        update_data["sort_order"] = payload.sort_order
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided for update"
        )
    try:
        helper = _get_helper(db)
        helper.update_world_info(item_id, update_data, expected_version)
        item = helper.get_world_info(item_id)
        if not item:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="World info not found"
            )
        return ManuscriptWorldInfoResponse(**item)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript world info")


@router.delete(
    "/world-info/{item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a world-info entry",
    tags=["manuscripts"],
)
async def delete_world_info(
    item_id: str,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.delete")),
) -> Response:
    """Soft-delete a world-info entry."""
    try:
        helper = _get_helper(db)
        helper.soft_delete_world_info(item_id, expected_version)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript world info")


# ===================================================================
# Plot Lines
# ===================================================================


@router.post(
    "/projects/{project_id}/plot-lines",
    response_model=ManuscriptPlotLineResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a plot line",
    tags=["manuscripts"],
)
async def create_plot_line(
    project_id: str,
    payload: ManuscriptPlotLineCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.create")),
) -> ManuscriptPlotLineResponse:
    """Create a new plot line within a project."""
    try:
        title = _require_non_empty_text(payload.title, "title")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        helper = _get_helper(db)
        plot_line_id = helper.create_plot_line(
            project_id=project_id,
            title=title,
            description=payload.description,
            status=payload.status,
            color=payload.color,
            sort_order=payload.sort_order,
            plot_line_id=payload.id,
        )
        line = helper.get_plot_line(plot_line_id)
        if not line:
            raise CharactersRAGDBError("Plot line created but could not be retrieved")
        return ManuscriptPlotLineResponse(**line)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript plot line")


@router.get(
    "/projects/{project_id}/plot-lines",
    response_model=list[ManuscriptPlotLineResponse],
    summary="List plot lines in a project",
    tags=["manuscripts"],
)
async def list_plot_lines(
    project_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.list")),
) -> list[ManuscriptPlotLineResponse]:
    """List all plot lines within a project."""
    try:
        helper = _get_helper(db)
        lines = helper.list_plot_lines(project_id)
        return [ManuscriptPlotLineResponse(**pl) for pl in lines]
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript plot lines")


@router.patch(
    "/plot-lines/{plot_line_id}",
    response_model=ManuscriptPlotLineResponse,
    summary="Update a plot line",
    tags=["manuscripts"],
)
async def update_plot_line(
    plot_line_id: str,
    payload: ManuscriptPlotLineUpdate,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.update")),
) -> ManuscriptPlotLineResponse:
    """Update a plot line with optimistic locking."""
    update_data: dict[str, Any] = {}
    if _field_present(payload, "title"):
        try:
            update_data["title"] = _require_non_empty_text(payload.title, "title")
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if _field_present(payload, "description"):
        update_data["description"] = payload.description
    if _field_present(payload, "status"):
        if payload.status is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="status cannot be null",
            )
        update_data["status"] = payload.status
    if _field_present(payload, "color"):
        update_data["color"] = payload.color
    if _field_present(payload, "sort_order") and payload.sort_order is not None:
        update_data["sort_order"] = payload.sort_order
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided for update"
        )
    try:
        helper = _get_helper(db)
        helper.update_plot_line(plot_line_id, update_data, expected_version)
        line = helper.get_plot_line(plot_line_id)
        if not line:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Plot line not found"
            )
        return ManuscriptPlotLineResponse(**line)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript plot line")


@router.delete(
    "/plot-lines/{plot_line_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a plot line",
    tags=["manuscripts"],
)
async def delete_plot_line(
    plot_line_id: str,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.delete")),
) -> Response:
    """Soft-delete a plot line."""
    try:
        helper = _get_helper(db)
        helper.soft_delete_plot_line(plot_line_id, expected_version)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript plot line")


# ===================================================================
# Plot Events
# ===================================================================


@router.post(
    "/plot-lines/{plot_line_id}/events",
    response_model=ManuscriptPlotEventResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a plot event",
    tags=["manuscripts"],
)
async def create_plot_event(
    plot_line_id: str,
    payload: ManuscriptPlotEventCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.create")),
) -> ManuscriptPlotEventResponse:
    """Create a new plot event within a plot line.

    The project_id is resolved from the plot line's parent project.
    """
    try:
        title = _require_non_empty_text(payload.title, "title")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        helper = _get_helper(db)
        # Resolve project_id from the plot line
        plot_line = helper.get_plot_line(plot_line_id)
        if not plot_line:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Plot line not found"
            )
        project_id = plot_line["project_id"]

        event_id = helper.create_plot_event(
            project_id=project_id,
            plot_line_id=plot_line_id,
            title=title,
            description=payload.description,
            scene_id=payload.scene_id,
            chapter_id=payload.chapter_id,
            event_type=payload.event_type,
            sort_order=payload.sort_order,
            event_id=payload.id,
        )
        event = helper.get_plot_event(event_id)
        if not event:
            raise CharactersRAGDBError("Plot event created but could not be retrieved")
        return ManuscriptPlotEventResponse(**event)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript plot event")


@router.get(
    "/plot-lines/{plot_line_id}/events",
    response_model=list[ManuscriptPlotEventResponse],
    summary="List plot events for a plot line",
    tags=["manuscripts"],
)
async def list_plot_events(
    plot_line_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.list")),
) -> list[ManuscriptPlotEventResponse]:
    """List all plot events for a plot line."""
    try:
        helper = _get_helper(db)
        events = helper.list_plot_events(plot_line_id)
        return [ManuscriptPlotEventResponse(**e) for e in events]
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript plot events")


@router.patch(
    "/plot-events/{plot_event_id}",
    response_model=ManuscriptPlotEventResponse,
    summary="Update a plot event",
    tags=["manuscripts"],
)
async def update_plot_event(
    plot_event_id: str,
    payload: ManuscriptPlotEventUpdate,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.update")),
) -> ManuscriptPlotEventResponse:
    """Update a plot event with optimistic locking."""
    update_data: dict[str, Any] = {}
    if _field_present(payload, "title"):
        try:
            update_data["title"] = _require_non_empty_text(payload.title, "title")
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if _field_present(payload, "description"):
        update_data["description"] = payload.description
    if _field_present(payload, "scene_id"):
        update_data["scene_id"] = payload.scene_id
    if _field_present(payload, "chapter_id"):
        update_data["chapter_id"] = payload.chapter_id
    if _field_present(payload, "event_type") and payload.event_type is not None:
        update_data["event_type"] = payload.event_type
    if _field_present(payload, "sort_order") and payload.sort_order is not None:
        update_data["sort_order"] = payload.sort_order
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided for update"
        )
    try:
        helper = _get_helper(db)
        helper.update_plot_event(plot_event_id, update_data, expected_version)
        event = helper.get_plot_event(plot_event_id)
        if not event:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Plot event not found"
            )
        return ManuscriptPlotEventResponse(**event)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript plot event")


@router.delete(
    "/plot-events/{plot_event_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a plot event",
    tags=["manuscripts"],
)
async def delete_plot_event(
    plot_event_id: str,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.delete")),
) -> Response:
    """Soft-delete a plot event."""
    try:
        helper = _get_helper(db)
        helper.soft_delete_plot_event(plot_event_id, expected_version)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript plot event")


# ===================================================================
# Plot Holes
# ===================================================================


@router.post(
    "/projects/{project_id}/plot-holes",
    response_model=ManuscriptPlotHoleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a plot hole",
    tags=["manuscripts"],
)
async def create_plot_hole(
    project_id: str,
    payload: ManuscriptPlotHoleCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.create")),
) -> ManuscriptPlotHoleResponse:
    """Create a new plot hole entry."""
    try:
        title = _require_non_empty_text(payload.title, "title")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    try:
        helper = _get_helper(db)
        hole_id = helper.create_plot_hole(
            project_id=project_id,
            title=title,
            description=payload.description,
            severity=payload.severity,
            scene_id=payload.scene_id,
            chapter_id=payload.chapter_id,
            plot_line_id=payload.plot_line_id,
            detected_by=payload.detected_by,
            plot_hole_id=payload.id,
        )
        hole = helper.get_plot_hole(hole_id)
        if not hole:
            raise CharactersRAGDBError("Plot hole created but could not be retrieved")
        return ManuscriptPlotHoleResponse(**hole)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript plot hole")


@router.get(
    "/projects/{project_id}/plot-holes",
    response_model=list[ManuscriptPlotHoleResponse],
    summary="List plot holes in a project",
    tags=["manuscripts"],
)
async def list_plot_holes(
    project_id: str,
    status_filter: str | None = Query(None, alias="status", description="Filter by status"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.list")),
) -> list[ManuscriptPlotHoleResponse]:
    """List all plot holes within a project."""
    try:
        helper = _get_helper(db)
        holes = helper.list_plot_holes(project_id, status_filter=status_filter)
        return [ManuscriptPlotHoleResponse(**h) for h in holes]
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript plot holes")


@router.patch(
    "/plot-holes/{hole_id}",
    response_model=ManuscriptPlotHoleResponse,
    summary="Update a plot hole",
    tags=["manuscripts"],
)
async def update_plot_hole(
    hole_id: str,
    payload: ManuscriptPlotHoleUpdate,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.update")),
) -> ManuscriptPlotHoleResponse:
    """Update a plot hole with optimistic locking."""
    update_data: dict[str, Any] = {}
    if _field_present(payload, "title"):
        try:
            update_data["title"] = _require_non_empty_text(payload.title, "title")
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if _field_present(payload, "description"):
        update_data["description"] = payload.description
    if _field_present(payload, "severity") and payload.severity is not None:
        update_data["severity"] = payload.severity
    if _field_present(payload, "status"):
        if payload.status is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="status cannot be null",
            )
        update_data["status"] = payload.status
    if _field_present(payload, "resolution"):
        update_data["resolution"] = payload.resolution
    if _field_present(payload, "scene_id"):
        update_data["scene_id"] = payload.scene_id
    if _field_present(payload, "chapter_id"):
        update_data["chapter_id"] = payload.chapter_id
    if _field_present(payload, "plot_line_id"):
        update_data["plot_line_id"] = payload.plot_line_id
    if _field_present(payload, "detected_by") and payload.detected_by is not None:
        update_data["detected_by"] = payload.detected_by
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No fields provided for update"
        )
    try:
        helper = _get_helper(db)
        helper.update_plot_hole(hole_id, update_data, expected_version)
        hole = helper.get_plot_hole(hole_id)
        if not hole:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Plot hole not found"
            )
        return ManuscriptPlotHoleResponse(**hole)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript plot hole")


@router.delete(
    "/plot-holes/{hole_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a plot hole",
    tags=["manuscripts"],
)
async def delete_plot_hole(
    hole_id: str,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.delete")),
) -> Response:
    """Soft-delete a plot hole."""
    try:
        helper = _get_helper(db)
        helper.soft_delete_plot_hole(hole_id, expected_version)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript plot hole")


# ===================================================================
# Scene-Character Linking
# ===================================================================


@router.post(
    "/scenes/{scene_id}/characters",
    response_model=list[SceneCharacterLinkResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Link a character to a scene",
    tags=["manuscripts"],
)
async def link_scene_character(
    scene_id: str,
    payload: SceneCharacterLink,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.create")),
) -> list[SceneCharacterLinkResponse]:
    """Link a character to a scene."""
    try:
        helper = _get_helper(db)
        helper.link_scene_character(scene_id, payload.character_id, is_pov=payload.is_pov)
        links = helper.list_scene_characters(scene_id)
        return [SceneCharacterLinkResponse(**lk) for lk in links]
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "scene-character link")


@router.get(
    "/scenes/{scene_id}/characters",
    response_model=list[SceneCharacterLinkResponse],
    summary="List characters linked to a scene",
    tags=["manuscripts"],
)
async def list_scene_characters(
    scene_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.list")),
) -> list[SceneCharacterLinkResponse]:
    """List all characters linked to a scene."""
    try:
        helper = _get_helper(db)
        links = helper.list_scene_characters(scene_id)
        return [SceneCharacterLinkResponse(**lk) for lk in links]
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "scene-character links")


@router.delete(
    "/scenes/{scene_id}/characters/{character_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Unlink a character from a scene",
    tags=["manuscripts"],
)
async def unlink_scene_character(
    scene_id: str,
    character_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.delete")),
) -> Response:
    """Remove the link between a character and a scene."""
    try:
        helper = _get_helper(db)
        helper.unlink_scene_character(scene_id, character_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "scene-character unlink")


# ===================================================================
# Scene-World Info Linking
# ===================================================================


@router.post(
    "/scenes/{scene_id}/world-info",
    response_model=list[SceneWorldInfoLinkResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Link a world-info entry to a scene",
    tags=["manuscripts"],
)
async def link_scene_world_info(
    scene_id: str,
    payload: SceneWorldInfoLink,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.create")),
) -> list[SceneWorldInfoLinkResponse]:
    """Link a world-info entry to a scene."""
    try:
        helper = _get_helper(db)
        helper.link_scene_world_info(scene_id, payload.world_info_id)
        links = helper.list_scene_world_info(scene_id)
        return [SceneWorldInfoLinkResponse(**lk) for lk in links]
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "scene-world-info link")


@router.get(
    "/scenes/{scene_id}/world-info",
    response_model=list[SceneWorldInfoLinkResponse],
    summary="List world-info entries linked to a scene",
    tags=["manuscripts"],
)
async def list_scene_world_info(
    scene_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.list")),
) -> list[SceneWorldInfoLinkResponse]:
    """List all world-info entries linked to a scene."""
    try:
        helper = _get_helper(db)
        links = helper.list_scene_world_info(scene_id)
        return [SceneWorldInfoLinkResponse(**lk) for lk in links]
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "scene-world-info links")


@router.delete(
    "/scenes/{scene_id}/world-info/{world_info_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Unlink a world-info entry from a scene",
    tags=["manuscripts"],
)
async def unlink_scene_world_info(
    scene_id: str,
    world_info_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.delete")),
) -> Response:
    """Remove the link between a world-info entry and a scene."""
    try:
        helper = _get_helper(db)
        helper.unlink_scene_world_info(scene_id, world_info_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "scene-world-info unlink")


# ===================================================================
# Citations
# ===================================================================


@router.post(
    "/scenes/{scene_id}/citations",
    response_model=ManuscriptCitationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a citation for a scene",
    tags=["manuscripts"],
)
async def create_citation(
    scene_id: str,
    payload: ManuscriptCitationCreate,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.create")),
) -> ManuscriptCitationResponse:
    """Create a citation linked to a scene.

    The project_id is resolved from the scene's parent project.
    """
    try:
        helper = _get_helper(db)
        # Resolve project_id from the scene
        scene = helper.get_scene(scene_id)
        if not scene:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Scene not found"
            )
        project_id = scene["project_id"]

        citation_id = helper.create_citation(
            project_id=project_id,
            scene_id=scene_id,
            source_type=payload.source_type,
            source_id=payload.source_id,
            source_title=payload.source_title,
            excerpt=payload.excerpt,
            query_used=payload.query_used,
            anchor_offset=payload.anchor_offset,
            citation_id=payload.id,
        )
        citation = helper.get_citation(citation_id)
        if not citation:
            raise CharactersRAGDBError("Citation created but could not be retrieved")
        return ManuscriptCitationResponse(**citation)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript citation")


@router.get(
    "/scenes/{scene_id}/citations",
    response_model=list[ManuscriptCitationResponse],
    summary="List citations for a scene",
    tags=["manuscripts"],
)
async def list_citations(
    scene_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.list")),
) -> list[ManuscriptCitationResponse]:
    """List all citations for a scene."""
    try:
        helper = _get_helper(db)
        citations = helper.list_citations(scene_id)
        return [ManuscriptCitationResponse(**c) for c in citations]
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript citations")


@router.delete(
    "/citations/{citation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Delete a citation",
    tags=["manuscripts"],
)
async def delete_citation(
    citation_id: str,
    expected_version: int = Header(..., description="Expected version for optimistic locking"),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.delete")),
) -> Response:
    """Soft-delete a citation."""
    try:
        helper = _get_helper(db)
        helper.soft_delete_citation(citation_id, expected_version)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript citation")


# ===================================================================
# Research (stub)
# ===================================================================


@router.post(
    "/scenes/{scene_id}/research",
    response_model=ManuscriptResearchResponse,
    summary="Search RAG sources for research",
    tags=["manuscripts"],
)
async def research_scene(
    scene_id: str,
    payload: ManuscriptResearchRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.get")),
) -> ManuscriptResearchResponse:
    """Search RAG sources relevant to a scene.

    This is a stub endpoint. Full RAG integration will be added in a follow-up.
    """
    try:
        helper = _get_helper(db)
        # Verify scene exists
        scene = helper.get_scene(scene_id)
        if not scene:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Scene not found"
            )
        # Stub: return empty results for now
        return ManuscriptResearchResponse(query=payload.query, results=[])
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "manuscript research")


# ===================================================================
# AI Analysis
# ===================================================================


_VALID_ANALYSIS_TYPES = {"pacing", "plot_holes", "consistency"}


def _normalize_analysis_override(value: str | None) -> str | None:
    """Normalize optional provider/model override strings."""
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _validate_analysis_overrides(
    *,
    provider: str | None,
    model: str | None,
) -> tuple[str | None, str | None]:
    """Validate analysis provider/model overrides against configured allowlists."""
    normalized_provider = _normalize_analysis_override(provider)
    normalized_model = _normalize_analysis_override(model)

    provider_manager = get_provider_manager()
    configured_providers = {
        str(entry).strip().lower()
        for entry in getattr(provider_manager, "providers", [])
        if str(entry).strip()
    }
    fallback_known_providers = {key.strip().lower() for key in PROVIDER_CAPABILITIES.keys()}
    allowed_providers = configured_providers or fallback_known_providers

    if normalized_provider:
        provider_key = normalized_provider.lower()
        if provider_key not in allowed_providers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown analysis provider override: {normalized_provider}",
            )
    else:
        provider_key = (
            str(getattr(provider_manager, "primary_provider", "")).strip().lower() or None
        )

    if normalized_model:
        if not provider_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model override requires a configured analysis provider",
            )
        model_known = is_model_known_for_provider(provider_key, normalized_model)
        if model_known is False:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown model override '{normalized_model}' for provider '{provider_key}'",
            )

    return provider_key, normalized_model


@router.post(
    "/scenes/{scene_id}/analyze",
    response_model=list[ManuscriptAnalysisResponse],
    summary="Run AI analysis on a scene",
    tags=["manuscripts"],
)
async def analyze_scene(
    scene_id: str,
    payload: ManuscriptAnalysisRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.analyze")),
) -> list[ManuscriptAnalysisResponse]:
    """Run one or more LLM-powered analyses on a single scene."""
    try:
        await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.manuscripts.analyze")
        provider_override, model_override = _validate_analysis_overrides(
            provider=payload.provider,
            model=payload.model,
        )
        helper = _get_helper(db)
        scene = helper.get_scene(scene_id)
        if not scene:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scene not found")

        from tldw_Server_API.app.core.Writing.manuscript_analysis import (
            analyze_pacing,
            analyze_consistency as _analyze_consistency,
            analyze_plot_holes as _analyze_plot_holes,
        )

        text = scene.get("content_plain", "") or ""
        results: list[ManuscriptAnalysisResponse] = []
        for analysis_type in payload.analysis_types:
            if analysis_type == "pacing":
                result = await analyze_pacing(text, provider=provider_override, model=model_override)
                score = result.get("pacing") if isinstance(result.get("pacing"), (int, float)) else None
            elif analysis_type == "plot_holes":
                result = await _analyze_plot_holes(text, provider=provider_override, model=model_override)
                score = None
            elif analysis_type == "consistency":
                result = await _analyze_consistency(text, provider=provider_override, model=model_override)
                score = result.get("overall_score") if isinstance(result.get("overall_score"), (int, float)) else None
            else:
                result = {"error": f"Unknown analysis type: {analysis_type}"}
                score = None

            aid = helper.create_analysis(
                scene["project_id"], "scene", scene_id, analysis_type,
                result, score=score, provider=provider_override, model=model_override,
            )
            analysis = helper.get_analysis(aid)
            if analysis:
                results.append(ManuscriptAnalysisResponse(**analysis))

        return results
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "scene analysis")


@router.post(
    "/chapters/{chapter_id}/analyze",
    response_model=list[ManuscriptAnalysisResponse],
    summary="Run AI analysis on a chapter",
    tags=["manuscripts"],
)
async def analyze_chapter(
    chapter_id: str,
    payload: ManuscriptAnalysisRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.analyze")),
) -> list[ManuscriptAnalysisResponse]:
    """Run LLM analysis across all scenes in a chapter."""
    try:
        await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.manuscripts.analyze")
        provider_override, model_override = _validate_analysis_overrides(
            provider=payload.provider,
            model=payload.model,
        )
        helper = _get_helper(db)
        chapter = helper.get_chapter(chapter_id)
        if not chapter:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chapter not found")

        # Gather all scene text
        scenes = helper.list_scenes(chapter_id)
        combined_text = "\n\n".join(s.get("content_plain", "") or "" for s in scenes)

        from tldw_Server_API.app.core.Writing.manuscript_analysis import (
            analyze_pacing,
            analyze_consistency as _analyze_consistency,
            analyze_plot_holes as _analyze_plot_holes,
        )

        results: list[ManuscriptAnalysisResponse] = []
        for analysis_type in payload.analysis_types:
            if analysis_type == "pacing":
                result = await analyze_pacing(combined_text, provider=provider_override, model=model_override)
                score = result.get("pacing") if isinstance(result.get("pacing"), (int, float)) else None
            elif analysis_type == "plot_holes":
                result = await _analyze_plot_holes(combined_text, provider=provider_override, model=model_override)
                score = None
            elif analysis_type == "consistency":
                result = await _analyze_consistency(combined_text, provider=provider_override, model=model_override)
                score = result.get("overall_score") if isinstance(result.get("overall_score"), (int, float)) else None
            else:
                result = {"error": f"Unknown analysis type: {analysis_type}"}
                score = None

            aid = helper.create_analysis(
                chapter["project_id"], "chapter", chapter_id, analysis_type,
                result, score=score, provider=provider_override, model=model_override,
            )
            analysis = helper.get_analysis(aid)
            if analysis:
                results.append(ManuscriptAnalysisResponse(**analysis))

        return results
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "chapter analysis")


def _gather_project_text(helper: ManuscriptDBHelper, project_id: str) -> str:
    """Collect all scene text across the whole project in a single query."""
    all_texts = helper.get_all_scene_texts(project_id)
    return "\n\n".join(all_texts)


def _gather_character_and_world_summaries(
    helper: ManuscriptDBHelper, project_id: str,
) -> tuple[str, str]:
    """Build short summaries of characters and world info for analysis prompts."""
    characters = helper.list_characters(project_id)
    char_summary = ", ".join(f"{c['name']} ({c['role']})" for c in characters)
    world_info = helper.list_world_info(project_id)
    world_summary = ", ".join(f"{w['name']} ({w['kind']})" for w in world_info)
    return char_summary, world_summary


@router.post(
    "/projects/{project_id}/analyze/plot-holes",
    response_model=list[ManuscriptAnalysisResponse],
    summary="AI plot hole detection",
    tags=["manuscripts"],
)
async def analyze_plot_holes_endpoint(
    project_id: str,
    payload: ManuscriptAnalysisRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.analyze")),
) -> list[ManuscriptAnalysisResponse]:
    """Detect plot holes and inconsistencies across an entire project."""
    try:
        await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.manuscripts.analyze")
        provider_override, model_override = _validate_analysis_overrides(
            provider=payload.provider,
            model=payload.model,
        )
        helper = _get_helper(db)
        proj = helper.get_project(project_id)
        if not proj:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

        combined_text = _gather_project_text(helper, project_id)
        char_summary, world_summary = _gather_character_and_world_summaries(helper, project_id)

        from tldw_Server_API.app.core.Writing.manuscript_analysis import analyze_plot_holes as _analyze_plot_holes

        result = await _analyze_plot_holes(
            combined_text, char_summary, world_summary,
            provider=provider_override, model=model_override,
        )

        aid = helper.create_analysis(
            project_id, "project", project_id, "plot_holes",
            result, provider=provider_override, model=model_override,
        )
        analysis = helper.get_analysis(aid)
        return [ManuscriptAnalysisResponse(**analysis)] if analysis else []
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "plot hole analysis")


@router.post(
    "/projects/{project_id}/analyze/consistency",
    response_model=list[ManuscriptAnalysisResponse],
    summary="Check character/world consistency",
    tags=["manuscripts"],
)
async def analyze_consistency_endpoint(
    project_id: str,
    payload: ManuscriptAnalysisRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.analyze")),
) -> list[ManuscriptAnalysisResponse]:
    """Check character and world-building consistency across an entire project."""
    try:
        await _enforce_rate_limit(rate_limiter, int(current_user.id), "writing.manuscripts.analyze")
        provider_override, model_override = _validate_analysis_overrides(
            provider=payload.provider,
            model=payload.model,
        )
        helper = _get_helper(db)
        proj = helper.get_project(project_id)
        if not proj:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

        combined_text = _gather_project_text(helper, project_id)
        char_summary, world_summary = _gather_character_and_world_summaries(helper, project_id)

        from tldw_Server_API.app.core.Writing.manuscript_analysis import analyze_consistency as _analyze_consistency

        result = await _analyze_consistency(
            combined_text, char_summary, world_summary,
            provider=provider_override, model=model_override,
        )
        score = result.get("overall_score") if isinstance(result.get("overall_score"), (int, float)) else None

        aid = helper.create_analysis(
            project_id, "project", project_id, "consistency",
            result, score=score, provider=provider_override, model=model_override,
        )
        analysis = helper.get_analysis(aid)
        return [ManuscriptAnalysisResponse(**analysis)] if analysis else []
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "consistency analysis")


@router.get(
    "/projects/{project_id}/analyses",
    response_model=ManuscriptAnalysisListResponse,
    summary="List cached analyses",
    tags=["manuscripts"],
)
async def list_analyses(
    project_id: str,
    scope_type: str | None = Query(None),
    analysis_type: str | None = Query(None),
    include_stale: bool = Query(False),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    _: None = Depends(rbac_rate_limit("writing.manuscripts.list")),
) -> ManuscriptAnalysisListResponse:
    """List cached AI analyses for a project, with optional filters."""
    try:
        helper = _get_helper(db)
        analyses = helper.list_analyses(
            project_id, scope_type=scope_type, analysis_type=analysis_type,
            include_stale=include_stale,
        )
        items = [ManuscriptAnalysisResponse(**a) for a in analyses]
        return ManuscriptAnalysisListResponse(analyses=items, total=len(items))
    except _MANUSCRIPT_NONCRITICAL_EXCEPTIONS as exc:
        _handle_db_errors(exc, "analyses")
