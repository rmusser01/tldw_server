"""REST API routes for note-backed task operations."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RateLimiter,
    User,
    get_rate_limiter_dep,
    get_request_user,
    rbac_rate_limit,
)
from tldw_Server_API.app.api.v1.schemas.notes_tasks_schemas import (
    TaskActivityListResponse,
    TaskActivityPatchRequest,
    TaskActivityResponse,
    TaskActivityStateResponse,
    TaskCreateRequest,
    TaskDeleteRequest,
    TaskListResponse,
    TaskMetadata,
    TaskNoteSummaryResponse,
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
from tldw_Server_API.app.core.Notes_Tasks.service import ReconciliationBatchResult

router = APIRouter()
_TASK_SERVICE = NotesTaskService()


def _actor(current_user: User) -> TaskActor:
    return TaskActor(actor_type="user", actor_id=str(current_user.id))


async def _check_rate_limit(rate_limiter: RateLimiter, current_user: User, action: str) -> None:
    try:
        allowed, meta = await rate_limiter.check_user_rate_limit(int(current_user.id), action)
    except Exception:
        allowed, meta = True, {}
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for {action}",
            headers={"Retry-After": str(meta.get("retry_after", 60))},
        )


def _handle_task_error(exc: Exception) -> None:
    if isinstance(exc, HTTPException):
        raise exc
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


def _task_response(db: CharactersRAGDB, task: dict[str, Any], *, include_projection: bool = True) -> TaskResponse:
    projection = db.task_store._fetch_projection(str(task["id"])) if include_projection else None
    return TaskResponse(
        id=str(task["id"]),
        note_id=str(task["note_id"]),
        text=str(task["text"]),
        status=task["status"],
        metadata=dict(task.get("metadata_json") or {}),
        projection_status=task["projection_status"],
        version=int(task["version"]),
        created_at=task.get("created_at"),
        updated_at=task.get("updated_at"),
        completed_at=task.get("completed_at"),
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
        reconciliation = _TASK_SERVICE.reconcile_stale_notes(
            db=db,
            limit=reconcile_limit,
            actor=_actor(current_user),
        )
        tasks = db.list_tasks(
            status=status_filter,
            projection_status=projection_status,
            limit=limit,
        )
        return TaskListResponse(
            tasks=[_task_response(db, task) for task in tasks],
            reconciliation=_reconciliation_response(reconciliation),
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
    _: None = Depends(rbac_rate_limit("notes.update")),
) -> TaskStatusBatchResponse:
    await _check_rate_limit(rate_limiter, current_user, "notes.update")
    try:
        tasks = [
            _TASK_SERVICE.update_task(
                db=db,
                task_id=item.task_id,
                expected_task_version=item.expected_task_version,
                expected_note_version=item.expected_note_version,
                status=item.status,
                actor=_actor(current_user),
                record_only=item.record_only,
            )
            for item in request.updates
        ]
        return TaskStatusBatchResponse(tasks=[_task_response(db, task) for task in tasks])
    except Exception as exc:
        _handle_task_error(exc)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Task status update failed")


@router.get("/tasks/activity", response_model=TaskActivityListResponse, tags=["notes"])
async def list_task_activity(
    limit: int = Query(50, ge=1, le=200),
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("notes.read")),
) -> TaskActivityListResponse:
    await _check_rate_limit(rate_limiter, current_user, "notes.read")
    try:
        events = list(reversed(db.list_task_activity(limit=200)))
        visible: list[TaskActivityResponse] = []
        user_id = str(current_user.id)
        for event in events:
            if event.get("actor_type") != "agent":
                continue
            state_row = db.get_task_activity_read_state(str(event["id"]), user_id=user_id)
            if state_row is not None and (state_row.get("read_at") or state_row.get("dismissed_at")):
                continue
            visible.append(_activity_response(event, state_row))
            if len(visible) >= limit:
                break
        return TaskActivityListResponse(events=visible)
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
        state_row = (
            db.mark_task_activity_dismissed(event_id, user_id=user_id)
            if request.dismissed
            else db.mark_task_activity_read(event_id, user_id=user_id)
        )
        return TaskActivityStateResponse(
            event_id=str(state_row["event_id"]),
            user_id=str(state_row["user_id"]),
            read_at=state_row.get("read_at"),
            dismissed_at=state_row.get("dismissed_at"),
        )
    except Exception as exc:
        _handle_task_error(exc)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Task activity update failed")


@router.get("/tasks/{task_id}", response_model=TaskResponse, tags=["notes"])
async def get_task(
    task_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("notes.read")),
) -> TaskResponse:
    await _check_rate_limit(rate_limiter, current_user, "notes.read")
    task = db.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    return _task_response(db, task)


@router.patch("/tasks/{task_id}", response_model=TaskResponse, tags=["notes"])
async def update_task(
    task_id: str,
    request: TaskUpdateRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("notes.update")),
) -> TaskResponse:
    await _check_rate_limit(rate_limiter, current_user, "notes.update")
    try:
        task = _TASK_SERVICE.update_task(
            db=db,
            task_id=task_id,
            expected_task_version=request.expected_task_version,
            expected_note_version=request.expected_note_version,
            text=request.text,
            metadata=_metadata_from_request(request.metadata) if request.metadata is not None else None,
            actor=_actor(current_user),
            record_only=request.record_only,
        )
        return _task_response(db, task)
    except Exception as exc:
        _handle_task_error(exc)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Task update failed")


@router.delete("/tasks/{task_id}", response_model=TaskResponse, tags=["notes"])
async def delete_task(
    task_id: str,
    request: TaskDeleteRequest,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("notes.delete")),
) -> TaskResponse:
    await _check_rate_limit(rate_limiter, current_user, "notes.delete")
    try:
        task = _TASK_SERVICE.delete_task(
            db=db,
            task_id=task_id,
            expected_task_version=request.expected_task_version,
            expected_note_version=request.expected_note_version,
            record_only=request.record_only,
            actor=_actor(current_user),
        )
        return _task_response(db, task)
    except Exception as exc:
        _handle_task_error(exc)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Task delete failed")


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
        result = _TASK_SERVICE.ensure_note_reconciled(db=db, note_id=note_id, actor=_actor(current_user))
        tasks = db.list_tasks(note_id=note_id, limit=limit)
        state_row = db.get_reconciliation_state(note_id)
        return TaskListResponse(
            tasks=[_task_response(db, task) for task in tasks],
            reconciliation=_reconciliation_response(result, fallback_state=state_row),
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
    _: None = Depends(rbac_rate_limit("notes.update")),
) -> TaskResponse:
    await _check_rate_limit(rate_limiter, current_user, "notes.update")
    try:
        task = _TASK_SERVICE.create_task_for_note(
            db=db,
            note_id=note_id,
            text=request.text,
            status=request.status,
            metadata=_metadata_from_request(request.metadata),
            expected_note_version=request.expected_note_version,
            actor=_actor(current_user),
        )
        return _task_response(db, task)
    except Exception as exc:
        _handle_task_error(exc)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Task create failed")


@router.post("/{note_id}/tasks/reconcile", response_model=TaskReconciliationSummaryResponse, tags=["notes"])
async def reconcile_note_tasks(
    note_id: str,
    db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    current_user: User = Depends(get_request_user),
    _: None = Depends(rbac_rate_limit("notes.update")),
) -> TaskReconciliationSummaryResponse:
    await _check_rate_limit(rate_limiter, current_user, "notes.update")
    try:
        result = _TASK_SERVICE.reconcile_note_current(db=db, note_id=note_id, actor=_actor(current_user))
        return _reconciliation_response(result)
    except Exception as exc:
        _handle_task_error(exc)
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Task reconciliation failed")


def _activity_response(event: dict[str, Any], state_row: dict[str, Any] | None) -> TaskActivityResponse:
    return TaskActivityResponse(
        id=str(event["id"]),
        task_id=event.get("task_id"),
        note_id=event.get("note_id"),
        event_type=str(event["event_type"]),
        actor_type=str(event["actor_type"]),
        actor_id=event.get("actor_id"),
        old_value=event.get("old_value_json"),
        new_value=event.get("new_value_json"),
        created_at=str(event["created_at"]),
        read_at=state_row.get("read_at") if state_row else None,
        dismissed_at=state_row.get("dismissed_at") if state_row else None,
    )
