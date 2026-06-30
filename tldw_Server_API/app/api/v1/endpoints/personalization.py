# tldw_Server_API/app/api/v1/endpoints/personalization.py
# Personalization endpoints: opt-in, preferences, memories, explanations

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import (
    UsageEventLogger,
    get_personalization_db_for_user,
    get_usage_event_logger,
)
from tldw_Server_API.app.api.v1.schemas.pagination import PagePaginationMeta
from tldw_Server_API.app.api.v1.schemas.personalization import (
    DetailResponse,
    ExplanationListResponse,
    MemoryCreate,
    MemoryImportRequest,
    MemoryItem,
    MemoryListResponse,
    MemoryUpdate,
    MemoryValidateRequest,
    OptInRequest,
    PersonalizationProfile,
    PreferencesUpdate,
    PurgeResponse,
)
from tldw_Server_API.app.api.v1.utils.pagination import (
    build_offset_pagination_meta,
    build_page_pagination_meta,
)
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB, SemanticMemory
from tldw_Server_API.app.core.feature_flags import is_personalization_enabled

router = APIRouter()


def _memory_pagination(*, total: int, page: int, size: int) -> PagePaginationMeta:
    total_pages = (total + size - 1) // size if total > 0 else 0
    return build_page_pagination_meta(
        page=page,
        per_page=size,
        total=total,
        total_pages=total_pages,
    )


def _profile_from_dict(prof_dict: dict, db: PersonalizationDB, user_id: str) -> PersonalizationProfile:
    """Build a PersonalizationProfile from a DB profile dict, using DB values."""
    raw_updated = prof_dict.get("updated_at")
    if isinstance(raw_updated, str):
        updated_at = datetime.fromisoformat(raw_updated)
    elif isinstance(raw_updated, datetime):
        updated_at = raw_updated
    else:
        logger.warning("Profile updated_at missing or unparseable")
        updated_at = datetime.now(timezone.utc)
    return PersonalizationProfile(
        enabled=bool(prof_dict.get("enabled")),
        alpha=float(prof_dict.get("alpha", 0.2)),
        beta=float(prof_dict.get("beta", 0.6)),
        gamma=float(prof_dict.get("gamma", 0.2)),
        recency_half_life_days=int(prof_dict.get("recency_half_life_days", 14)),
        topic_count=db.topic_counts(user_id),
        memory_count=db.memory_counts(user_id),
        session_count=db.session_count(user_id),
        proactive_enabled=bool(prof_dict.get("proactive_enabled", 1)),
        proactive_frequency=str(prof_dict.get("proactive_frequency", "normal")),
        response_style=str(prof_dict.get("response_style", "balanced")),
        preferred_format=str(prof_dict.get("preferred_format", "auto")),
        companion_reflections_enabled=bool(prof_dict.get("companion_reflections_enabled", 1)),
        companion_daily_reflections_enabled=bool(prof_dict.get("companion_daily_reflections_enabled", 1)),
        companion_weekly_reflections_enabled=bool(prof_dict.get("companion_weekly_reflections_enabled", 1)),
        updated_at=updated_at,
    )


@router.post("/opt-in", response_model=PersonalizationProfile, tags=["personalization"], status_code=status.HTTP_200_OK)
async def personalization_opt_in(
    payload: OptInRequest = Body(...),
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> PersonalizationProfile:
    """Enable/disable personalization for current user."""
    if not is_personalization_enabled():
        raise HTTPException(status_code=404, detail="Personalization disabled")
    logger.info(f"Personalization opt-in called: enabled={payload.enabled}")
    try:
        prof_dict = db.update_profile(user_id=log.user_id, enabled=1 if payload.enabled else 0)
        prof = _profile_from_dict(prof_dict, db, log.user_id)
        log.log_event("personalization.opt-in", metadata={"enabled": prof.enabled})
        return prof
    except Exception as e:
        logger.warning("Opt-in failed")
        raise HTTPException(status_code=500, detail="Failed to update personalization profile") from e


@router.post("/purge", response_model=PurgeResponse, tags=["personalization"], status_code=status.HTTP_200_OK)
async def personalization_purge(
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> PurgeResponse:
    """Purge all personalization data for the user."""
    if not is_personalization_enabled():
        raise HTTPException(status_code=404, detail="Personalization disabled")
    logger.info("Personalization purge called")
    try:
        counts = db.purge_user(log.user_id)
        return PurgeResponse(
            status="ok",
            deleted_counts=counts,
            enabled=False,
            purged_at=datetime.now(timezone.utc),
        )
    except Exception as e:
        logger.warning("Purge failed")
        raise HTTPException(status_code=500, detail="Failed to purge personalization data") from e


@router.get("/profile", response_model=PersonalizationProfile, tags=["personalization"], status_code=status.HTTP_200_OK)
async def personalization_profile(
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> PersonalizationProfile:
    """Get current personalization profile."""
    if not is_personalization_enabled():
        raise HTTPException(status_code=404, detail="Personalization disabled")
    prof = db.get_or_create_profile(log.user_id)
    out = _profile_from_dict(prof, db, log.user_id)
    log.log_event("personalization.view")
    return out


@router.post("/preferences", response_model=PersonalizationProfile, tags=["personalization"], status_code=status.HTTP_200_OK)
async def personalization_preferences(
    update: PreferencesUpdate,
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> PersonalizationProfile:
    """Update personalization weights/preferences."""
    if not is_personalization_enabled():
        raise HTTPException(status_code=404, detail="Personalization disabled")
    fields = update.model_dump(exclude_none=True)
    # Map quiet_hours dict to individual columns
    if "quiet_hours" in fields:
        qh = fields.pop("quiet_hours")
        if isinstance(qh, dict):
            if "start" in qh:
                fields["quiet_hours_start"] = qh["start"]
            if "end" in qh:
                fields["quiet_hours_end"] = qh["end"]
    # Map proactive_types list to JSON string
    if "proactive_types" in fields:
        import json
        fields["proactive_types"] = json.dumps(fields["proactive_types"])
    # Map bool flags to int for SQLite storage
    for flag in (
        "proactive_enabled",
        "companion_reflections_enabled",
        "companion_daily_reflections_enabled",
        "companion_weekly_reflections_enabled",
    ):
        if flag in fields:
            fields[flag] = int(bool(fields[flag]))

    prof_dict = db.update_profile(log.user_id, **fields)
    return _profile_from_dict(prof_dict, db, log.user_id)


@router.get("/memories", response_model=MemoryListResponse, tags=["personalization"], status_code=status.HTTP_200_OK)
async def list_memories(
    memory_type: str | None = Query(None, alias="type", description="semantic|episodic"),
    q: str | None = Query(None),
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=200),
    include_hidden: bool = Query(False),
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> MemoryListResponse:
    if not is_personalization_enabled():
        raise HTTPException(status_code=404, detail="Personalization disabled")
    offset = (page - 1) * size
    # Only semantic implemented; episodic not yet
    if memory_type and memory_type != "semantic":
        return MemoryListResponse(
            items=[],
            total=0,
            page=page,
            size=size,
            pagination=_memory_pagination(total=0, page=page, size=size),
        )
    items, total = db.list_semantic_memories(
        log.user_id, q=q, limit=size, offset=offset, include_hidden=include_hidden,
    )
    log.log_event("personalization.memories.view", metadata={"count": len(items)})
    return MemoryListResponse(
        items=items,
        total=total,
        page=page,
        size=size,
        pagination=_memory_pagination(total=total, page=page, size=size),
    )


@router.get("/memories/export", tags=["personalization"], status_code=status.HTTP_200_OK)
async def export_memories(
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> dict:
    """Export all memories as JSON."""
    if not is_personalization_enabled():
        raise HTTPException(status_code=404, detail="Personalization disabled")
    items = db.export_all_memories(log.user_id)
    log.log_event("personalization.memories.export", metadata={"count": len(items)})
    return {"memories": items, "total": len(items)}


@router.get("/memories/{memory_id}", response_model=MemoryItem, tags=["personalization"], status_code=status.HTTP_200_OK)
async def get_memory(
    memory_id: str,
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> MemoryItem:
    """Get a specific memory by ID."""
    if not is_personalization_enabled():
        raise HTTPException(status_code=404, detail="Personalization disabled")
    mem = db.get_memory(memory_id, log.user_id)
    if mem is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return MemoryItem(**mem)


@router.post("/memories", response_model=MemoryItem, tags=["personalization"], status_code=status.HTTP_201_CREATED)
async def add_memory(
    item: MemoryCreate,
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> MemoryItem:
    """Add a semantic memory."""
    if not is_personalization_enabled():
        raise HTTPException(status_code=404, detail="Personalization disabled")
    mem = SemanticMemory(user_id=log.user_id, content=item.content, tags=item.tags, pinned=item.pinned)
    mid = db.add_semantic_memory(mem)
    return MemoryItem(id=mid, type="semantic", content=item.content, pinned=item.pinned, tags=item.tags)


@router.patch("/memories/{memory_id}", response_model=MemoryItem, tags=["personalization"], status_code=status.HTTP_200_OK)
async def update_memory(
    memory_id: str,
    update: MemoryUpdate,
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> MemoryItem:
    """Update a memory (content, pinned, hidden, tags)."""
    if not is_personalization_enabled():
        raise HTTPException(status_code=404, detail="Personalization disabled")
    fields = update.model_dump(exclude_none=True)
    result = db.update_memory(memory_id, log.user_id, **fields)
    if result is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return MemoryItem(**result)


@router.delete("/memories/{memory_id}", response_model=DetailResponse, tags=["personalization"], status_code=status.HTTP_200_OK)
async def delete_memory(
    memory_id: str,
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> DetailResponse:
    """Delete memory by id."""
    if not is_personalization_enabled():
        raise HTTPException(status_code=404, detail="Personalization disabled")
    ok = db.delete_memory(memory_id, log.user_id)
    return DetailResponse(detail="ok: deleted" if ok else "no-op: not found")


@router.post("/memories/validate", response_model=DetailResponse, tags=["personalization"], status_code=status.HTTP_200_OK)
async def validate_memories(
    payload: MemoryValidateRequest,
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> DetailResponse:
    """Batch validate memories (mark as last_validated)."""
    if not is_personalization_enabled():
        raise HTTPException(status_code=404, detail="Personalization disabled")
    count = db.validate_memories(log.user_id, payload.memory_ids)
    return DetailResponse(detail=f"ok: validated {count} memories")


@router.post("/memories/import", response_model=DetailResponse, tags=["personalization"], status_code=status.HTTP_201_CREATED)
async def import_memories(
    payload: MemoryImportRequest,
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> DetailResponse:
    """Import memories from JSON."""
    if not is_personalization_enabled():
        raise HTTPException(status_code=404, detail="Personalization disabled")
    count = db.bulk_add_memories(log.user_id, payload.memories)
    log.log_event("personalization.memories.import", metadata={"count": count})
    return DetailResponse(detail=f"ok: imported {count} memories")


@router.get("/explanations", response_model=ExplanationListResponse, tags=["personalization"], status_code=status.HTTP_200_OK)
async def list_explanations(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> ExplanationListResponse:
    """Return recent personalization explanations (scaffold: returns empty)."""
    return ExplanationListResponse(
        items=[],
        total=0,
        pagination=build_offset_pagination_meta(
            limit=limit,
            offset=offset,
            total=0,
            count=0,
        ),
    )
