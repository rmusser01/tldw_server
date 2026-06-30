from __future__ import annotations

"""Companion API endpoints for explicit activity, goals, and workspace data."""

import asyncio
import sqlite3
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status

from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import get_collections_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import rbac_rate_limit
from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import get_job_manager
from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import (
    UsageEventLogger,
    get_personalization_db_for_user,
    get_usage_event_logger,
)
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.schemas.companion import (
    CompanionActivityCreate,
    CompanionActivityDetail,
    CompanionActivityListResponse,
    CompanionActivityItem,
    CompanionCheckInCreate,
    CompanionConversationPromptsResponse,
    CompanionGoal,
    CompanionGoalCreate,
    CompanionGoalListResponse,
    CompanionGoalUpdate,
    CompanionKnowledgeDetail,
    CompanionKnowledgeListResponse,
    CompanionLifecycleResponse,
    CompanionPurgeRequest,
    CompanionReflectionDetail,
    CompanionRebuildRequest,
)
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.feature_flags import is_personalization_enabled
from tldw_Server_API.app.core.Personalization.companion_activity import build_manual_check_in_activity
from tldw_Server_API.app.core.Personalization.companion_context import load_companion_context
from tldw_Server_API.app.core.Personalization.companion_followups import (
    build_companion_conversation_prompts,
)
from tldw_Server_API.app.core.Personalization.companion_lifecycle import purge_companion_scope
from tldw_Server_API.app.core.Personalization.companion_reflection_jobs import (
    COMPANION_REBUILD_JOB_TYPE,
    COMPANION_REFLECTION_DOMAIN,
    companion_reflection_queue,
)


router = APIRouter()


def _ensure_personalization_enabled() -> None:
    """Raise when the companion module is unavailable."""
    if not is_personalization_enabled():
        raise HTTPException(status_code=404, detail="Personalization disabled")


def _ensure_companion_opt_in(db: PersonalizationDB, user_id: str) -> None:
    """Raise when the user has not explicitly enabled personalization."""
    profile = db.get_or_create_profile(user_id)
    if not bool(profile.get("enabled")):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Enable personalization before using companion.",
        )


def _dedupe_ids(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _extract_evidence_ids(
    evidence: list[dict[str, object]],
    *,
    kind: str,
    key: str,
) -> list[str]:
    values: list[str] = []
    for item in evidence:
        if str(item.get("kind") or "").strip() != kind:
            continue
        raw = item.get(key)
        if raw is None:
            continue
        values.append(str(raw))
    return _dedupe_ids(values)


def _reflection_prompt_payload(event: dict[str, object]) -> dict[str, object]:
    metadata = dict(event.get("metadata") or {})
    return {
        "id": str(event.get("id") or ""),
        "summary": str(metadata.get("summary") or ""),
        "theme_key": metadata.get("theme_key"),
        "signal_strength": metadata.get("signal_strength"),
        "follow_up_prompts": list(metadata.get("follow_up_prompts") or []),
    }


async def _load_activity_details(
    db: PersonalizationDB,
    *,
    user_id: str,
    event_ids: list[str],
) -> list[dict[str, object]]:
    loaded = await asyncio.gather(
        *(asyncio.to_thread(db.get_companion_activity_event, user_id, event_id) for event_id in event_ids)
    )
    return [item for item in loaded if item is not None]


async def _load_knowledge_details(
    db: PersonalizationDB,
    *,
    user_id: str,
    card_ids: list[str],
) -> list[dict[str, object]]:
    loaded = await asyncio.gather(
        *(asyncio.to_thread(db.get_companion_knowledge_card, user_id, card_id) for card_id in card_ids)
    )
    return [item for item in loaded if item is not None]


async def _load_goal_details(
    db: PersonalizationDB,
    *,
    user_id: str,
    goal_ids: list[str],
) -> list[dict[str, object]]:
    loaded = await asyncio.gather(
        *(asyncio.to_thread(db.get_companion_goal, goal_id, user_id) for goal_id in goal_ids)
    )
    return [item for item in loaded if item is not None]


@router.post(
    "/activity",
    response_model=CompanionActivityItem,
    tags=["companion"],
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rbac_rate_limit("companion.activity.create"))],
)
async def create_companion_activity(
    payload: CompanionActivityCreate = Body(...),
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> CompanionActivityItem:
    """Create one explicit companion activity event."""
    _ensure_personalization_enabled()
    await asyncio.to_thread(_ensure_companion_opt_in, db, log.user_id)
    dedupe_key = (
        payload.dedupe_key
        or f"{payload.event_type}:{payload.source_type}:{payload.source_id}"
    )
    try:
        event_id = await asyncio.to_thread(
            db.insert_companion_activity_event,
            user_id=log.user_id,
            event_type=payload.event_type,
            source_type=payload.source_type,
            source_id=payload.source_id,
            surface=payload.surface,
            dedupe_key=dedupe_key,
            tags=payload.tags,
            provenance=payload.provenance,
            metadata=payload.metadata,
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Companion activity already captured",
        ) from exc

    await asyncio.to_thread(
        log.log_event,
        "companion.activity.create",
        resource_id=event_id,
        tags=list(payload.tags or []),
        metadata={"event_type": payload.event_type, "surface": payload.surface},
    )
    return CompanionActivityItem(
        id=event_id,
        event_type=payload.event_type,
        source_type=payload.source_type,
        source_id=payload.source_id,
        surface=payload.surface,
        tags=list(payload.tags or []),
        provenance=dict(payload.provenance),
        metadata=dict(payload.metadata or {}),
        created_at=datetime.now(timezone.utc),
    )


@router.post(
    "/check-ins",
    response_model=CompanionActivityItem,
    tags=["companion"],
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rbac_rate_limit("companion.checkins.create"))],
)
async def create_companion_check_in(
    payload: CompanionCheckInCreate = Body(...),
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> CompanionActivityItem:
    """Create one manual companion check-in."""
    _ensure_personalization_enabled()
    await asyncio.to_thread(_ensure_companion_opt_in, db, log.user_id)
    created_at = datetime.now(timezone.utc)
    source_id = f"checkin-{uuid4().hex}"
    activity_payload = build_manual_check_in_activity(
        source_id=source_id,
        title=payload.title,
        summary=payload.summary,
        surface=payload.surface or "companion.workspace",
        tags=payload.tags,
        event_timestamp=created_at.isoformat(),
    )
    event_id = await asyncio.to_thread(
        db.insert_companion_activity_event,
        user_id=log.user_id,
        **activity_payload,
    )
    await asyncio.to_thread(
        log.log_event,
        "companion.checkins.create",
        resource_id=event_id,
        tags=list(activity_payload["tags"] or []),
        metadata={"source_id": source_id},
    )
    return CompanionActivityItem(
        id=event_id,
        event_type=activity_payload["event_type"],
        source_type=activity_payload["source_type"],
        source_id=activity_payload["source_id"],
        surface=activity_payload["surface"],
        tags=list(activity_payload["tags"] or []),
        provenance=dict(activity_payload["provenance"]),
        metadata=dict(activity_payload["metadata"] or {}),
        created_at=created_at,
    )


@router.get(
    "/activity",
    response_model=CompanionActivityListResponse,
    tags=["companion"],
    dependencies=[Depends(rbac_rate_limit("companion.activity.read"))],
)
async def list_companion_activity(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> CompanionActivityListResponse:
    """List persisted companion activity events for the current user."""
    _ensure_personalization_enabled()
    await asyncio.to_thread(_ensure_companion_opt_in, db, log.user_id)
    items, total = await asyncio.to_thread(
        db.list_companion_activity_events,
        log.user_id,
        limit,
        offset,
    )
    await asyncio.to_thread(log.log_event, "companion.activity.view", metadata={"count": len(items)})
    return CompanionActivityListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        pagination=build_offset_pagination_meta(
            total=total,
            offset=offset,
            limit=limit,
            count=len(items),
        ),
    )


@router.get(
    "/activity/{event_id}",
    response_model=CompanionActivityDetail,
    tags=["companion"],
    dependencies=[Depends(rbac_rate_limit("companion.activity.read"))],
)
async def get_companion_activity_detail(
    event_id: str,
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> CompanionActivityDetail:
    """Return one detailed companion activity event for the current user."""
    _ensure_personalization_enabled()
    await asyncio.to_thread(_ensure_companion_opt_in, db, log.user_id)
    item = await asyncio.to_thread(db.get_companion_activity_event, log.user_id, event_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Companion activity not found")
    await asyncio.to_thread(log.log_event, "companion.activity.detail", resource_id=event_id)
    return CompanionActivityDetail.model_validate(item)


@router.get(
    "/knowledge",
    response_model=CompanionKnowledgeListResponse,
    tags=["companion"],
    dependencies=[Depends(rbac_rate_limit("companion.knowledge.read"))],
)
async def list_companion_knowledge(
    status_filter: str | None = Query("active", alias="status"),
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> CompanionKnowledgeListResponse:
    """List derived companion knowledge cards for the current user."""
    _ensure_personalization_enabled()
    await asyncio.to_thread(_ensure_companion_opt_in, db, log.user_id)
    items = await asyncio.to_thread(db.list_companion_knowledge_cards, log.user_id, status_filter)
    await asyncio.to_thread(log.log_event, "companion.knowledge.view", metadata={"count": len(items)})
    return CompanionKnowledgeListResponse(items=items, total=len(items))


@router.get(
    "/knowledge/{card_id}",
    response_model=CompanionKnowledgeDetail,
    tags=["companion"],
    dependencies=[Depends(rbac_rate_limit("companion.knowledge.read"))],
)
async def get_companion_knowledge_detail(
    card_id: str,
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> CompanionKnowledgeDetail:
    """Return one companion knowledge card with resolved evidence rows."""
    _ensure_personalization_enabled()
    await asyncio.to_thread(_ensure_companion_opt_in, db, log.user_id)
    card = await asyncio.to_thread(db.get_companion_knowledge_card, log.user_id, card_id)
    if card is None:
        raise HTTPException(status_code=404, detail="Companion knowledge card not found")
    evidence = list(card.get("evidence") or [])
    event_ids = _dedupe_ids(
        [str(item.get("event_id")) for item in evidence if item.get("event_id") is not None]
        + [str(item.get("source_event_id")) for item in evidence if item.get("source_event_id") is not None]
    )
    goal_ids = _dedupe_ids(
        [str(item.get("goal_id")) for item in evidence if item.get("goal_id") is not None]
    )
    event_rows, goal_rows = await asyncio.gather(
        _load_activity_details(db, user_id=log.user_id, event_ids=event_ids),
        _load_goal_details(db, user_id=log.user_id, goal_ids=goal_ids),
    )
    await asyncio.to_thread(log.log_event, "companion.knowledge.detail", resource_id=card_id)
    return CompanionKnowledgeDetail.model_validate(
        {
            **card,
            "evidence_events": event_rows,
            "evidence_goals": goal_rows,
        }
    )


@router.get(
    "/reflections/{reflection_id}",
    response_model=CompanionReflectionDetail,
    tags=["companion"],
    dependencies=[Depends(rbac_rate_limit("companion.reflections.read"))],
)
async def get_companion_reflection_detail(
    reflection_id: str,
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> CompanionReflectionDetail:
    """Return one companion reflection entry with resolved provenance references."""
    _ensure_personalization_enabled()
    await asyncio.to_thread(_ensure_companion_opt_in, db, log.user_id)
    reflection = await asyncio.to_thread(db.get_companion_activity_event, log.user_id, reflection_id)
    if reflection is None or str(reflection.get("source_type") or "") != "companion_reflection":
        raise HTTPException(status_code=404, detail="Companion reflection not found")

    provenance = dict(reflection.get("provenance") or {})
    evidence = list((reflection.get("metadata") or {}).get("evidence") or [])
    event_ids = _dedupe_ids(
        list(provenance.get("source_event_ids") or [])
        + _extract_evidence_ids(evidence, kind="activity_event", key="source_event_id")
    )
    card_ids = _dedupe_ids(
        list(provenance.get("knowledge_card_ids") or [])
        + _extract_evidence_ids(evidence, kind="knowledge_card", key="card_id")
    )
    goal_ids = _dedupe_ids(
        list(provenance.get("goal_ids") or [])
        + _extract_evidence_ids(evidence, kind="goal", key="goal_id")
    )
    activity_rows, knowledge_rows, goal_rows = await asyncio.gather(
        _load_activity_details(db, user_id=log.user_id, event_ids=event_ids),
        _load_knowledge_details(db, user_id=log.user_id, card_ids=card_ids),
        _load_goal_details(db, user_id=log.user_id, goal_ids=goal_ids),
    )
    metadata = dict(reflection.get("metadata") or {})
    await asyncio.to_thread(log.log_event, "companion.reflections.detail", resource_id=reflection_id)
    return CompanionReflectionDetail(
        id=str(reflection["id"]),
        title=str(metadata.get("title") or "Reflection"),
        cadence=metadata.get("cadence"),
        summary=str(metadata.get("summary") or ""),
        evidence=evidence,
        delivery_decision=metadata.get("delivery_decision"),
        delivery_reason=metadata.get("delivery_reason"),
        theme_key=metadata.get("theme_key"),
        signal_strength=metadata.get("signal_strength"),
        follow_up_prompts=list(metadata.get("follow_up_prompts") or []),
        provenance=provenance,
        created_at=reflection["created_at"],
        activity_events=activity_rows,
        knowledge_cards=knowledge_rows,
        goals=goal_rows,
    )


@router.get(
    "/conversation-prompts",
    response_model=CompanionConversationPromptsResponse,
    tags=["companion"],
    dependencies=[Depends(rbac_rate_limit("companion.conversation.read"))],
)
async def get_companion_conversation_prompts(
    query: str = Query(..., min_length=1),
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> CompanionConversationPromptsResponse:
    """Return deterministic follow-up prompts for the companion conversation surface."""
    _ensure_personalization_enabled()
    await asyncio.to_thread(_ensure_companion_opt_in, db, log.user_id)
    ranked_context = await asyncio.to_thread(
        load_companion_context,
        user_id=log.user_id,
        query=query,
        include_candidates=True,
        db=db,
    )
    activity_rows, _total = await asyncio.to_thread(db.list_companion_activity_events, log.user_id, 50, 0)
    delivered_reflections: list[dict[str, object]] = []
    suppressed_reflections: list[dict[str, object]] = []
    for row in activity_rows:
        if str(row.get("source_type") or "").strip() != "companion_reflection":
            continue
        metadata = dict(row.get("metadata") or {})
        decision = str(metadata.get("delivery_decision") or "").strip().lower()
        payload = _reflection_prompt_payload(row)
        if decision == "suppressed":
            suppressed_reflections.append(payload)
        else:
            delivered_reflections.append(payload)
    prompt_payload = build_companion_conversation_prompts(
        query=query,
        delivered_reflections=delivered_reflections,
        suppressed_reflections=suppressed_reflections,
        context_cards=list(ranked_context.get("cards") or []),
        context_goals=list(ranked_context.get("goals") or []),
        context_activity=list(ranked_context.get("activity_rows") or []),
    )
    await asyncio.to_thread(
        log.log_event,
        "companion.conversation.prompts",
        metadata={
            "prompt_source_kind": prompt_payload.get("prompt_source_kind"),
            "prompt_count": len(prompt_payload.get("prompts") or []),
        },
    )
    return CompanionConversationPromptsResponse.model_validate(prompt_payload)


@router.get(
    "/goals",
    response_model=CompanionGoalListResponse,
    tags=["companion"],
    dependencies=[Depends(rbac_rate_limit("companion.goals.read"))],
)
async def list_companion_goals(
    status_filter: str | None = Query(None, alias="status"),
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> CompanionGoalListResponse:
    """List companion goals for the current user."""
    _ensure_personalization_enabled()
    await asyncio.to_thread(_ensure_companion_opt_in, db, log.user_id)
    items = await asyncio.to_thread(db.list_companion_goals, log.user_id, status_filter)
    await asyncio.to_thread(log.log_event, "companion.goals.view", metadata={"count": len(items)})
    return CompanionGoalListResponse(items=items, total=len(items))


@router.post(
    "/goals",
    response_model=CompanionGoal,
    tags=["companion"],
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rbac_rate_limit("companion.goals.create"))],
)
async def create_companion_goal(
    payload: CompanionGoalCreate = Body(...),
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> CompanionGoal:
    """Create a new companion goal."""
    _ensure_personalization_enabled()
    await asyncio.to_thread(_ensure_companion_opt_in, db, log.user_id)
    goal_id = await asyncio.to_thread(
        db.create_companion_goal,
        user_id=log.user_id,
        title=payload.title,
        description=payload.description,
        goal_type=payload.goal_type,
        config=payload.config,
        progress=payload.progress,
        status=payload.status,
    )
    goal = await asyncio.to_thread(db.update_companion_goal, goal_id, log.user_id)
    if goal is None:
        raise HTTPException(status_code=500, detail="Failed to load created goal")
    await asyncio.to_thread(
        log.log_event,
        "companion.goals.create",
        resource_id=goal_id,
        tags=[payload.goal_type],
        metadata={"status": goal["status"]},
    )
    return CompanionGoal.model_validate(goal)


@router.patch(
    "/goals/{goal_id}",
    response_model=CompanionGoal,
    tags=["companion"],
    dependencies=[Depends(rbac_rate_limit("companion.goals.update"))],
)
async def update_companion_goal(
    goal_id: str,
    payload: CompanionGoalUpdate = Body(...),
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> CompanionGoal:
    """Update a companion goal."""
    _ensure_personalization_enabled()
    await asyncio.to_thread(_ensure_companion_opt_in, db, log.user_id)
    fields = payload.model_dump(exclude_unset=True)
    invalid_null_fields = sorted(
        key for key in ("title", "config", "progress", "status") if key in fields and fields[key] is None
    )
    if invalid_null_fields:
        raise HTTPException(
            status_code=422,
            detail=f"Fields cannot be null: {', '.join(invalid_null_fields)}",
        )
    goal = await asyncio.to_thread(db.update_companion_goal, goal_id, log.user_id, **fields)
    if goal is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    await asyncio.to_thread(
        log.log_event,
        "companion.goals.update",
        resource_id=goal_id,
        tags=[goal["goal_type"], goal["status"]],
        metadata={"updated_fields": sorted(fields.keys())},
    )
    return CompanionGoal.model_validate(goal)


@router.post(
    "/purge",
    response_model=CompanionLifecycleResponse,
    tags=["companion"],
    dependencies=[Depends(rbac_rate_limit("companion.lifecycle.purge"))],
)
async def purge_companion_data(
    payload: CompanionPurgeRequest = Body(...),
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    collections_db: CollectionsDatabase = Depends(get_collections_db_for_user),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> CompanionLifecycleResponse:
    """Purge one rebuildable slice of companion state while preserving explicit activity by default."""
    _ensure_personalization_enabled()
    await asyncio.to_thread(_ensure_companion_opt_in, db, log.user_id)
    result = await asyncio.to_thread(
        purge_companion_scope,
        user_id=log.user_id,
        scope=payload.scope,
        personalization_db=db,
        collections_db=collections_db,
    )
    await asyncio.to_thread(
        log.log_event,
        "companion.lifecycle.purge",
        tags=[payload.scope],
        metadata={"deleted_counts": result["deleted_counts"]},
    )
    return CompanionLifecycleResponse.model_validate(result)


@router.post(
    "/rebuild",
    response_model=CompanionLifecycleResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["companion"],
    dependencies=[Depends(rbac_rate_limit("companion.lifecycle.rebuild"))],
)
async def rebuild_companion_data(
    payload: CompanionRebuildRequest = Body(...),
    db: PersonalizationDB = Depends(get_personalization_db_for_user),
    jm: JobManager = Depends(get_job_manager),
    log: UsageEventLogger = Depends(get_usage_event_logger),
) -> CompanionLifecycleResponse:
    """Queue a scoped companion rebuild job."""
    _ensure_personalization_enabled()
    await asyncio.to_thread(_ensure_companion_opt_in, db, log.user_id)
    job = await asyncio.to_thread(
        jm.create_job,
        domain=COMPANION_REFLECTION_DOMAIN,
        queue=companion_reflection_queue(),
        job_type=COMPANION_REBUILD_JOB_TYPE,
        payload={"scope": payload.scope, "user_id": log.user_id},
        owner_user_id=str(log.user_id),
        priority=5,
        max_retries=1,
    )
    job_id = job.get("id")
    await asyncio.to_thread(
        log.log_event,
        "companion.lifecycle.rebuild",
        resource_id=None if job_id is None else str(job_id),
        tags=[payload.scope],
        metadata={"job_uuid": job.get("uuid")},
    )
    return CompanionLifecycleResponse(
        status=str(job.get("status") or "queued"),
        scope=payload.scope,
        job_id=None if job_id is None else int(job_id),
        job_uuid=job.get("uuid"),
    )
