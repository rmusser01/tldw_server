"""REST API routes for note-backed task operations."""

# Endpoint boundaries intentionally funnel all service/database failures through
# the single sanitized mapper below.
# ruff: noqa: BLE001

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger
from starlette.concurrency import run_in_threadpool

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RateLimiter,
    User,
    get_rate_limiter_dep,
    get_request_user,
    rbac_rate_limit,
)
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints.notes_sync_errors import (
    NOTES_SYNC_EXCEPTIONS,
    notes_sync_http_error,
)
from tldw_Server_API.app.api.v1.schemas.notes_tasks_schemas import (
    TaskActivityListResponse,
    TaskActivityPatchRequest,
    TaskActivityResponse,
    TaskActivityStateResponse,
    TaskCreateRequest,
    TaskListResponse,
    TaskMetadata,
    TaskNoteSummaryResponse,
    TaskProjectionDriftListResponse,
    TaskProjectionDriftResolveRequest,
    TaskProjectionDriftResponse,
    TaskProjectionResponse,
    TaskReconciliationSummaryResponse,
    TaskResponse,
    TaskStatusBatchRequest,
    TaskStatusBatchResponse,
    TaskStatusValue,
    TaskUpdateRequest,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Notes_Tasks import NotesTaskService, ReconciliationResult, TaskActor
from tldw_Server_API.app.core.Notes_Tasks.service import (
    ReconciliationBatchResult,
    TaskStoreScope,
    resolve_task_compatibility_scope,
)

router = APIRouter()


def get_notes_task_service() -> NotesTaskService:
    """Provide the stateless notes task service for endpoint dependency overrides."""
    return NotesTaskService()


def _actor(current_user: User) -> TaskActor:
    return TaskActor(actor_type="user", actor_id=str(current_user.id))


async def _check_rate_limit(rate_limiter: RateLimiter, current_user: User, action: str) -> None:
    try:
        user_id = int(current_user.id)
    except (TypeError, ValueError) as exc:
        logger.warning("Invalid user id for notes task rate limit: {}", current_user.id)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for {action}",
            headers={"Retry-After": "60"},
        ) from exc
    try:
        allowed, meta = await rate_limiter.check_user_rate_limit(user_id, action)
    except Exception as exc:
        logger.warning("Rate limiter failed for notes task action {} and user {}: {}", action, current_user.id, exc)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for {action}",
            headers={"Retry-After": "60"},
        ) from exc
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for {action}",
            headers={"Retry-After": str(meta.get("retry_after", 60))},
        )


def _handle_task_error(exc: Exception) -> None:
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, NOTES_SYNC_EXCEPTIONS):
        raise notes_sync_http_error(exc) from exc
    if isinstance(exc, InputError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if isinstance(exc, ConflictError):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if isinstance(exc, CharactersRAGDBError):
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Task operation failed") from exc


def _metadata_from_request(metadata: TaskMetadata | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    return metadata.as_compact_dict()


def _projection_response(projection: dict[str, Any] | None) -> TaskProjectionResponse | None:
    if projection is None:
        return None
    return TaskProjectionResponse(
        note_id=str(projection["note_id"]),
        note_version=int(projection["note_version"]),
        line_number=int(projection["line_number"]),
        start_offset=int(projection["start_offset"]),
        end_offset=int(projection["end_offset"]),
        raw_line=str(projection.get("raw_line") or ""),
        has_child_content=bool(projection.get("has_child_content")),
        projection_status=projection["projection_status"],
    )


def _note_summary(db: CharactersRAGDB, note_id: str) -> TaskNoteSummaryResponse | None:
    note = db.get_note_by_id(note_id)
    if not note:
        return None
    return TaskNoteSummaryResponse(
        id=str(note["id"]),
        title=str(note["title"]),
        version=int(note["version"]),
    )


def _timestamp_response(value: Any) -> str | None:
    """Normalize backend timestamp values at the existing string response boundary."""
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    return str(isoformat() if callable(isoformat) else value)


def _drift_response(drift: dict[str, Any]) -> TaskProjectionDriftResponse:
    """Build a content-free drift response from authorized storage claims."""

    created_at = _timestamp_response(drift.get("created_at"))
    updated_at = _timestamp_response(drift.get("updated_at"))
    if created_at is None or updated_at is None:
        raise CharactersRAGDBError("Projection drift timestamps are unavailable.")
    return TaskProjectionDriftResponse(
        id=str(drift["id"]),
        note_id=str(drift["note_id"]),
        task_id=str(drift["task_id"]),
        marker_base_revision=int(drift["marker_base_revision"]),
        marker_base_hash=str(drift["marker_base_hash"]),
        note_head_cursor=drift.get("note_head_cursor"),
        note_head_hash=drift.get("note_head_hash"),
        task_head_cursor=drift.get("task_head_cursor"),
        task_head_hash=drift.get("task_head_hash"),
        reason_code=str(drift["reason_code"]),
        status=drift["status"],
        lifecycle_revision=1 if drift["status"] == "open" else 2,
        created_at=created_at,
        updated_at=updated_at,
        resolved_at=_timestamp_response(drift.get("resolved_at")),
    )


def _task_response(
    db: CharactersRAGDB,
    task: dict[str, Any],
    *,
    scope: TaskStoreScope,
    include_projection: bool = True,
) -> TaskResponse:
    """Build one backend-neutral task response within an authenticated scope."""
    projection = db.get_task_projection(
        owner_user_id=scope.owner_user_id,
        dataset_id=scope.dataset_id,
        task_id=str(task["id"]),
    ) if include_projection else None
    return TaskResponse(
        id=str(task["id"]),
        note_id=str(task["note_id"]),
        text=str(task["text"]),
        status=task["status"],
        metadata=dict(task.get("metadata_json") or {}),
        projection_status=task["projection_status"],
        version=int(task["version"]),
        created_at=_timestamp_response(task.get("created_at")),
        updated_at=_timestamp_response(task.get("updated_at")),
        completed_at=_timestamp_response(task.get("completed_at")),
        note=_note_summary(db, str(task["note_id"])),
        projection=_projection_response(projection),
    )


def _reconciliation_response(
    result: ReconciliationResult | ReconciliationBatchResult | None,
    *,
    fallback_state: dict[str, Any] | None = None,
) -> TaskReconciliationSummaryResponse:
    if isinstance(result, ReconciliationBatchResult):
        return TaskReconciliationSummaryResponse(
            status=result.status,
            processed_notes=result.processed_notes,
            remaining_stale_notes=result.remaining_stale_notes,
        )
    if isinstance(result, ReconciliationResult):
        return TaskReconciliationSummaryResponse(
            status="clean" if result.warning_count == 0 else "warnings",
            note_id=result.note_id,
            note_version=result.note_version,
            parsed_count=result.parsed_count,
            created_count=result.created_count,
            updated_count=result.updated_count,
            unlinked_count=result.unlinked_count,
            ambiguous_count=result.ambiguous_count,
            warning_count=result.warning_count,
        )
    if fallback_state is not None:
        return TaskReconciliationSummaryResponse(
            status=fallback_state["status"],
            note_id=str(fallback_state["note_id"]),
            note_version=int(fallback_state["note_version"]),
            parsed_count=int(fallback_state["item_count"]),
            warning_count=int(fallback_state["warning_count"]),
        )
    return TaskReconciliationSummaryResponse(status="clean")


def _stale_reconciliation_response(
    db: CharactersRAGDB,
    *,
    scope: TaskStoreScope,
    note_id: str | None = None,
) -> TaskReconciliationSummaryResponse:
    remaining = db.count_candidate_notes_for_task_discovery(
        owner_user_id=scope.owner_user_id,
        dataset_id=scope.dataset_id,
        note_id=note_id,
    )
    if remaining:
        note = db.get_note_by_id(note_id) if note_id is not None else None
        return TaskReconciliationSummaryResponse(
            status="incomplete",
            note_id=str(note_id) if note_id is not None else None,
            note_version=int(note["version"]) if note is not None else None,
            processed_notes=0,
            remaining_stale_notes=remaining,
        )
    if note_id is not None:
        state_row = db.get_reconciliation_state(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            note_id=note_id,
        )
        return _reconciliation_response(None, fallback_state=state_row)
    return TaskReconciliationSummaryResponse(status="clean", processed_notes=0, remaining_stale_notes=0)


@router.get("/tasks", response_model=TaskListResponse, tags=["notes"])
async def list_tasks(
    status_filter: TaskStatusValue | None = Query(None, alias="status"),
    projection_status: str | None = Query(None, pattern="^(live|unlinked|ambiguous|deleted)$"),
    limit: int = Query(100, ge=1, le=500),
    reconcile_limit: int = Query(25, ge=0, le=100),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("notes.read")),
) -> TaskListResponse:
    await _check_rate_limit(rate_limiter, current_user, "notes.read")
    try:
        scope = resolve_task_compatibility_scope(
            db, authenticated_owner_user_id=str(current_user.id)
        )
        tasks = db.list_tasks(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            status=status_filter,
            projection_status=projection_status,
            limit=limit,
        )
        return TaskListResponse(
            tasks=[_task_response(db, task, scope=scope) for task in tasks],
            reconciliation=_stale_reconciliation_response(db, scope=scope),
        )
    except Exception as exc:
        _handle_task_error(exc)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Task list failed")


@router.post("/tasks/status", response_model=TaskStatusBatchResponse, tags=["notes"])
async def set_task_status(
    request: TaskStatusBatchRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    task_service: NotesTaskService = Depends(get_notes_task_service),
    _: None = Depends(rbac_rate_limit("notes.update")),
) -> TaskStatusBatchResponse:
    await _check_rate_limit(rate_limiter, current_user, "notes.update")
    try:
        tasks = []
        with db.transaction():
            for item in request.updates:
                tasks.append(
                    task_service.update_task(
                        db=db,
                        owner_user_id=str(current_user.id),
                        task_id=item.task_id,
                        expected_task_version=item.expected_task_version,
                        expected_note_version=item.expected_note_version,
                        status=item.status,
                        actor=_actor(current_user),
                        record_only=item.record_only,
                    )
                )
        scope = resolve_task_compatibility_scope(
            db, authenticated_owner_user_id=str(current_user.id)
        )
        return TaskStatusBatchResponse(tasks=[_task_response(db, task, scope=scope) for task in tasks])
    except Exception as exc:
        _handle_task_error(exc)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Task status update failed")


@router.get("/tasks/activity", response_model=TaskActivityListResponse, tags=["notes"])
async def list_task_activity(
    limit: int = Query(50, ge=1, le=200),
    note_id: str | None = Query(None, min_length=1),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("notes.read")),
) -> TaskActivityListResponse:
    await _check_rate_limit(rate_limiter, current_user, "notes.read")
    try:
        user_id = str(current_user.id)
        scope = resolve_task_compatibility_scope(db, authenticated_owner_user_id=user_id)
        events = db.list_recent_unread_task_activity(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            user_id=user_id,
            note_id=note_id,
            actor_type="agent",
            limit=limit,
        )
        return TaskActivityListResponse(events=[_activity_response(event, None) for event in events])
    except Exception as exc:
        _handle_task_error(exc)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Task activity list failed")


@router.patch("/tasks/activity/{event_id}", response_model=TaskActivityStateResponse, tags=["notes"])
async def update_task_activity_state(
    event_id: str,
    request: TaskActivityPatchRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("notes.update")),
) -> TaskActivityStateResponse:
    await _check_rate_limit(rate_limiter, current_user, "notes.update")
    try:
        user_id = str(current_user.id)
        scope = resolve_task_compatibility_scope(db, authenticated_owner_user_id=user_id)
        state_row = (
            db.mark_task_activity_dismissed(
                owner_user_id=scope.owner_user_id,
                dataset_id=scope.dataset_id,
                event_id=event_id,
                user_id=user_id,
            )
            if request.dismissed
            else db.mark_task_activity_read(
                owner_user_id=scope.owner_user_id,
                dataset_id=scope.dataset_id,
                event_id=event_id,
                user_id=user_id,
            )
        )
        return TaskActivityStateResponse(
            event_id=str(state_row["event_id"]),
            user_id=str(state_row["user_id"]),
            read_at=_timestamp_response(state_row.get("read_at")),
            dismissed_at=_timestamp_response(state_row.get("dismissed_at")),
        )
    except Exception as exc:
        _handle_task_error(exc)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Task activity update failed")


@router.get(
    "/tasks/drifts",
    response_model=TaskProjectionDriftListResponse,
    tags=["notes"],
)
async def list_task_projection_drifts(
    note_id: str = Query(..., min_length=1, max_length=128),
    task_id: str | None = Query(None, min_length=1, max_length=128),
    drift_status: str = Query("open", alias="status", pattern="^(open|resolved|dismissed)$"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("notes.read")),
) -> TaskProjectionDriftListResponse:
    """List bounded content-free projection drift claims for one note."""

    await _check_rate_limit(rate_limiter, current_user, "notes.read")
    try:
        scope = resolve_task_compatibility_scope(
            db,
            authenticated_owner_user_id=str(current_user.id),
        )
        drifts = db.task_store.list_task_projection_drifts(
            owner_user_id=scope.owner_user_id,
            dataset_id=scope.dataset_id,
            note_id=note_id,
            task_id=task_id,
            status=drift_status,
            limit=limit,
            offset=offset,
        )
        return TaskProjectionDriftListResponse(
            drifts=[_drift_response(drift) for drift in drifts]
        )
    except Exception as exc:
        _handle_task_error(exc)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Projection drift list failed",
    )


@router.post(
    "/tasks/{task_id}/drifts/{drift_id}/resolve",
    response_model=TaskProjectionDriftResponse,
    tags=["notes"],
)
async def resolve_task_projection_drift(
    task_id: str,
    drift_id: str,
    request: TaskProjectionDriftResolveRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    task_service: NotesTaskService = Depends(get_notes_task_service),
    _: None = Depends(rbac_rate_limit("notes.update")),
) -> TaskProjectionDriftResponse:
    """Resolve or dismiss one drift only against exact current claims."""

    await _check_rate_limit(rate_limiter, current_user, "notes.update")
    try:
        drift = task_service.resolve_projection_drift(
            db=db,
            owner_user_id=str(current_user.id),
            note_id=request.note_id,
            task_id=task_id,
            drift_id=drift_id,
            action=request.action,
            expected_lifecycle_revision=request.expected_lifecycle_revision,
            expected_note_head_cursor=request.expected_note_head_cursor,
            expected_note_head_hash=request.expected_note_head_hash,
            expected_task_head_cursor=request.expected_task_head_cursor,
            expected_task_head_hash=request.expected_task_head_hash,
            actor=_actor(current_user),
        )
        return _drift_response(drift)
    except Exception as exc:
        _handle_task_error(exc)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Projection drift resolution failed",
    )


@router.get("/tasks/{task_id}", response_model=TaskResponse, tags=["notes"])
async def get_task(
    task_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("notes.read")),
) -> TaskResponse:
    await _check_rate_limit(rate_limiter, current_user, "notes.read")
    scope = resolve_task_compatibility_scope(
        db, authenticated_owner_user_id=str(current_user.id)
    )
    task = db.get_task(
        owner_user_id=scope.owner_user_id,
        dataset_id=scope.dataset_id,
        task_id=task_id,
    )
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    return _task_response(db, task, scope=scope)


@router.patch("/tasks/{task_id}", response_model=TaskResponse, tags=["notes"])
async def update_task(
    task_id: str,
    request: TaskUpdateRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    task_service: NotesTaskService = Depends(get_notes_task_service),
    _: None = Depends(rbac_rate_limit("notes.update")),
) -> TaskResponse:
    await _check_rate_limit(rate_limiter, current_user, "notes.update")
    try:
        task = task_service.update_task(
            db=db,
            owner_user_id=str(current_user.id),
            task_id=task_id,
            expected_task_version=request.expected_task_version,
            expected_note_version=request.expected_note_version,
            text=request.text,
            metadata=_metadata_from_request(request.metadata) if request.metadata is not None else None,
            actor=_actor(current_user),
            record_only=request.record_only,
        )
        scope = resolve_task_compatibility_scope(
            db, authenticated_owner_user_id=str(current_user.id)
        )
        return _task_response(db, task, scope=scope)
    except Exception as exc:
        _handle_task_error(exc)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Task update failed")


@router.delete("/tasks/{task_id}", response_model=TaskResponse, tags=["notes"])
async def delete_task(
    task_id: str,
    expected_task_version: int = Query(..., ge=1),
    expected_note_version: int | None = Query(None, ge=1),
    record_only: bool = Query(False),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    task_service: NotesTaskService = Depends(get_notes_task_service),
    _: None = Depends(rbac_rate_limit("notes.delete")),
) -> TaskResponse:
    await _check_rate_limit(rate_limiter, current_user, "notes.delete")
    try:
        task = task_service.delete_task(
            db=db,
            owner_user_id=str(current_user.id),
            task_id=task_id,
            expected_task_version=expected_task_version,
            expected_note_version=expected_note_version,
            record_only=record_only,
            actor=_actor(current_user),
        )
        scope = resolve_task_compatibility_scope(
            db, authenticated_owner_user_id=str(current_user.id)
        )
        return _task_response(db, task, scope=scope)
    except Exception as exc:
        _handle_task_error(exc)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Task delete failed")


def _list_note_tasks_response(
    db: CharactersRAGDB,
    *,
    owner_user_id: str,
    note_id: str,
    limit: int,
) -> TaskListResponse:
    """Build a note-scoped task list without blocking the async request loop."""
    scope = resolve_task_compatibility_scope(
        db, authenticated_owner_user_id=owner_user_id
    )
    if db.get_note_by_id(note_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")
    tasks = db.list_tasks(
        owner_user_id=scope.owner_user_id,
        dataset_id=scope.dataset_id,
        note_id=note_id,
        limit=limit,
    )
    return TaskListResponse(
        tasks=[_task_response(db, task, scope=scope) for task in tasks],
        reconciliation=_stale_reconciliation_response(db, scope=scope, note_id=note_id),
    )


@router.get("/{note_id}/tasks", response_model=TaskListResponse, tags=["notes"])
async def list_note_tasks(
    note_id: str,
    limit: int = Query(100, ge=1, le=500),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("notes.read")),
) -> TaskListResponse:
    await _check_rate_limit(rate_limiter, current_user, "notes.read")
    try:
        return await run_in_threadpool(
            _list_note_tasks_response,
            db,
            owner_user_id=str(current_user.id),
            note_id=note_id,
            limit=limit,
        )
    except Exception as exc:
        _handle_task_error(exc)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Note task list failed")


@router.post("/{note_id}/tasks", response_model=TaskResponse, status_code=status.HTTP_201_CREATED, tags=["notes"])
async def create_note_task(
    note_id: str,
    request: TaskCreateRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    task_service: NotesTaskService = Depends(get_notes_task_service),
    _: None = Depends(rbac_rate_limit("notes.update")),
) -> TaskResponse:
    await _check_rate_limit(rate_limiter, current_user, "notes.update")
    try:
        task = task_service.create_task_for_note(
            db=db,
            owner_user_id=str(current_user.id),
            note_id=note_id,
            text=request.text,
            status=request.status,
            metadata=_metadata_from_request(request.metadata),
            expected_note_version=request.expected_note_version,
            actor=_actor(current_user),
        )
        scope = resolve_task_compatibility_scope(
            db, authenticated_owner_user_id=str(current_user.id)
        )
        return _task_response(db, task, scope=scope)
    except Exception as exc:
        _handle_task_error(exc)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Task create failed")


@router.post("/{note_id}/tasks/reconcile", response_model=TaskReconciliationSummaryResponse, tags=["notes"])
async def reconcile_note_tasks(
    note_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    task_service: NotesTaskService = Depends(get_notes_task_service),
    _: None = Depends(rbac_rate_limit("notes.update")),
) -> TaskReconciliationSummaryResponse:
    await _check_rate_limit(rate_limiter, current_user, "notes.update")
    try:
        result = task_service.reconcile_note_current(
            db=db,
            owner_user_id=str(current_user.id),
            note_id=note_id,
            actor=_actor(current_user),
        )
        return _reconciliation_response(result)
    except Exception as exc:
        _handle_task_error(exc)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Task reconciliation failed")


def _activity_response(event: dict[str, Any], state_row: dict[str, Any] | None) -> TaskActivityResponse:
    created_at = _timestamp_response(event.get("created_at"))
    if created_at is None:
        raise ValueError("Task activity created_at is required.")
    return TaskActivityResponse(
        id=str(event["id"]),
        task_id=event.get("task_id"),
        note_id=event.get("note_id"),
        event_type=str(event["event_type"]),
        actor_type=str(event["actor_type"]),
        actor_id=event.get("actor_id"),
        tool_name=event.get("tool_name"),
        policy_mode=event.get("policy_mode"),
        approval_id=event.get("approval_id"),
        old_value=event.get("old_value_json"),
        new_value=event.get("new_value_json"),
        created_at=created_at,
        read_at=_timestamp_response(state_row.get("read_at")) if state_row else None,
        dismissed_at=_timestamp_response(state_row.get("dismissed_at")) if state_row else None,
    )
