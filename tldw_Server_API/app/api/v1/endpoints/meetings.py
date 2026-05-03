from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.Meetings_DB_Deps import (
    get_meetings_db_for_user,
    get_meetings_db_for_websocket,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_request_user, User
from tldw_Server_API.app.api.v1.schemas.meetings_schemas import (
    MeetingArtifactCreate,
    MeetingArtifactResponse,
    MeetingFinalizeRequest,
    MeetingFinalizeResponse,
    MeetingHealthResponse,
    MeetingSessionCreate,
    MeetingSessionResponse,
    MeetingSessionStatus,
    MeetingSessionStatusUpdate,
    MeetingShareRequest,
    MeetingShareResponse,
    MeetingTemplateCreate,
    MeetingTemplateResponse,
    MeetingTemplateScope,
)
from tldw_Server_API.app.core.DB_Management.Meetings_DB import MeetingsDatabase
from tldw_Server_API.app.core.Meetings.artifact_service import MeetingArtifactService
from tldw_Server_API.app.core.Meetings.events_service import MeetingEventsService
from tldw_Server_API.app.core.Meetings.integration_service import MeetingIntegrationService
from tldw_Server_API.app.core.Meetings.session_service import MeetingSessionService
from tldw_Server_API.app.core.Meetings.stream_adapter import build_meeting_event, to_sse_frame
from tldw_Server_API.app.core.Meetings.template_service import MeetingTemplateService

router = APIRouter(prefix="/meetings", tags=["meetings"])


@router.get("/health", response_model=MeetingHealthResponse, openapi_extra={"security": []})
async def meetings_health() -> MeetingHealthResponse:
    return MeetingHealthResponse()


def _to_session_response(row: dict[str, Any]) -> MeetingSessionResponse:
    return MeetingSessionResponse(
        id=str(row.get("id") or ""),
        title=str(row.get("title") or ""),
        meeting_type=str(row.get("meeting_type") or ""),
        status=row.get("status") or "scheduled",
        source_type=row.get("source_type") or "upload",
        language=row.get("language"),
        template_id=row.get("template_id"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _to_template_response(row: dict[str, Any]) -> MeetingTemplateResponse:
    return MeetingTemplateResponse(
        id=str(row.get("id") or ""),
        name=str(row.get("name") or ""),
        scope=row.get("scope") or "personal",
        enabled=bool(row.get("enabled")),
        is_default=bool(row.get("is_default")),
        version=int(row.get("version") or 1),
        schema_json=row.get("schema_json") or {},
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _to_artifact_response(row: dict[str, Any]) -> MeetingArtifactResponse:
    return MeetingArtifactResponse(
        id=str(row.get("id") or ""),
        session_id=str(row.get("session_id") or ""),
        kind=row.get("kind") or "summary",
        format=str(row.get("format") or "json"),
        payload_json=row.get("payload_json") or {},
        version=int(row.get("version") or 1),
        created_at=row.get("created_at"),
    )


def _to_share_response(row: dict[str, Any]) -> MeetingShareResponse:
    return MeetingShareResponse(
        dispatch_id=int(row.get("id") or 0),
        session_id=str(row.get("session_id") or ""),
        integration_type=str(row.get("integration_type") or "webhook"),
        status="queued",
    )


@router.post(
    "/sessions",
    response_model=MeetingSessionResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(check_rate_limit)],
)
def create_session(
    payload: MeetingSessionCreate,
    meetings_db: MeetingsDatabase = Depends(get_meetings_db_for_user),
) -> MeetingSessionResponse:
    service = MeetingSessionService(db=meetings_db)
    events_service = MeetingEventsService(db=meetings_db)
    row = service.create_session(
        title=payload.title,
        meeting_type=payload.meeting_type,
        source_type=payload.source_type,
        language=payload.language,
        template_id=payload.template_id,
        metadata=payload.metadata,
    )
    events_service.emit(
        session_id=str(row.get("id") or ""),
        event_type="session.created",
        data={"status": row.get("status"), "title": row.get("title")},
    )
    return _to_session_response(row)


@router.get(
    "/sessions",
    response_model=list[MeetingSessionResponse],
    dependencies=[Depends(check_rate_limit)],
)
def list_sessions(
    status_filter: MeetingSessionStatus | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    meetings_db: MeetingsDatabase = Depends(get_meetings_db_for_user),
) -> list[MeetingSessionResponse]:
    service = MeetingSessionService(db=meetings_db)
    rows = service.list_sessions(status=status_filter, limit=limit, offset=offset)
    return [_to_session_response(row) for row in rows]


@router.get(
    "/sessions/{session_id}",
    response_model=MeetingSessionResponse,
    dependencies=[Depends(check_rate_limit)],
)
def get_session(
    session_id: str,
    meetings_db: MeetingsDatabase = Depends(get_meetings_db_for_user),
) -> MeetingSessionResponse:
    service = MeetingSessionService(db=meetings_db)
    try:
        row = service.get_session(session_id=session_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meeting session not found") from exc
    return _to_session_response(row)


@router.post(
    "/sessions/{session_id}/status",
    response_model=MeetingSessionResponse,
    dependencies=[Depends(check_rate_limit)],
)
def transition_session_status(
    session_id: str,
    payload: MeetingSessionStatusUpdate,
    meetings_db: MeetingsDatabase = Depends(get_meetings_db_for_user),
) -> MeetingSessionResponse:
    service = MeetingSessionService(db=meetings_db)
    events_service = MeetingEventsService(db=meetings_db)
    try:
        row = service.transition(session_id=session_id, to_status=payload.status)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meeting session not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    events_service.emit(
        session_id=session_id,
        event_type="session.status",
        data={"status": row.get("status")},
    )
    return _to_session_response(row)


@router.post(
    "/templates",
    response_model=MeetingTemplateResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(check_rate_limit)],
)
def create_template(
    payload: MeetingTemplateCreate,
    meetings_db: MeetingsDatabase = Depends(get_meetings_db_for_user),
    current_user: User = Depends(get_request_user),
) -> MeetingTemplateResponse:
    if payload.scope == "builtin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Builtin templates are immutable",
        )
    if payload.scope in {"org", "team"} and not (
        bool(current_user.is_admin) or "team_lead" in set(current_user.roles or [])
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient privileges for requested template scope",
        )
    service = MeetingTemplateService(db=meetings_db)
    try:
        row = service.create_template(
            name=payload.name,
            scope=payload.scope,
            schema_json=payload.schema_json,
            enabled=payload.enabled,
            is_default=payload.is_default,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return _to_template_response(row)


@router.get(
    "/templates",
    response_model=list[MeetingTemplateResponse],
    dependencies=[Depends(check_rate_limit)],
)
def list_templates(
    scope: MeetingTemplateScope | None = Query(default=None),
    include_disabled: bool = Query(default=False),
    meetings_db: MeetingsDatabase = Depends(get_meetings_db_for_user),
) -> list[MeetingTemplateResponse]:
    service = MeetingTemplateService(db=meetings_db)
    rows = service.list_templates(scope=scope, include_disabled=include_disabled)
    return [_to_template_response(row) for row in rows]


@router.get(
    "/templates/{template_id}",
    response_model=MeetingTemplateResponse,
    dependencies=[Depends(check_rate_limit)],
)
def get_template(
    template_id: str,
    meetings_db: MeetingsDatabase = Depends(get_meetings_db_for_user),
) -> MeetingTemplateResponse:
    service = MeetingTemplateService(db=meetings_db)
    try:
        row = service.get_template(template_id=template_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meeting template not found") from exc
    return _to_template_response(row)


@router.post(
    "/sessions/{session_id}/artifacts",
    response_model=MeetingArtifactResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(check_rate_limit)],
)
def create_artifact(
    session_id: str,
    payload: MeetingArtifactCreate,
    meetings_db: MeetingsDatabase = Depends(get_meetings_db_for_user),
) -> MeetingArtifactResponse:
    service = MeetingArtifactService(db=meetings_db)
    events_service = MeetingEventsService(db=meetings_db)
    try:
        row = service.create_artifact(
            session_id=session_id,
            kind=payload.kind,
            format=payload.format,
            payload_json=payload.payload_json,
            version=payload.version,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meeting session not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    events_service.emit(
        session_id=session_id,
        event_type="artifact.ready",
        data={"artifact_id": row.get("id"), "kind": row.get("kind")},
    )
    return _to_artifact_response(row)


@router.get(
    "/sessions/{session_id}/artifacts",
    response_model=list[MeetingArtifactResponse],
    dependencies=[Depends(check_rate_limit)],
)
def list_artifacts(
    session_id: str,
    meetings_db: MeetingsDatabase = Depends(get_meetings_db_for_user),
) -> list[MeetingArtifactResponse]:
    service = MeetingArtifactService(db=meetings_db)
    rows = service.list_artifacts(session_id=session_id)
    return [_to_artifact_response(row) for row in rows]


@router.post(
    "/sessions/{session_id}/commit",
    response_model=MeetingFinalizeResponse,
    dependencies=[Depends(check_rate_limit)],
)
def finalize_session(
    session_id: str,
    payload: MeetingFinalizeRequest,
    meetings_db: MeetingsDatabase = Depends(get_meetings_db_for_user),
) -> MeetingFinalizeResponse:
    session_service = MeetingSessionService(db=meetings_db)
    artifact_service = MeetingArtifactService(db=meetings_db)
    events_service = MeetingEventsService(db=meetings_db)

    try:
        session_service.get_session(session_id=session_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meeting session not found") from exc

    try:
        artifacts = artifact_service.generate_final_artifacts(
            session_id=session_id,
            transcript_text=payload.transcript_text,
            include=list(payload.include) if payload.include else None,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meeting session not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    meetings_db.update_session_status(session_id=session_id, status="completed")
    events_service.emit(session_id=session_id, event_type="session.status", data={"status": "completed"})
    for artifact in artifacts:
        events_service.emit(
            session_id=session_id,
            event_type="artifact.ready",
            data={"artifact_id": artifact.get("id"), "kind": artifact.get("kind")},
        )

    return MeetingFinalizeResponse(
        session_id=session_id,
        artifacts=[_to_artifact_response(row) for row in artifacts],
    )


def _queue_share_dispatch(
    *,
    session_id: str,
    integration_type: str,
    payload: MeetingShareRequest,
    meetings_db: MeetingsDatabase,
) -> MeetingShareResponse:
    integration_service = MeetingIntegrationService(db=meetings_db)
    events_service = MeetingEventsService(db=meetings_db)

    try:
        dispatch_row = integration_service.queue_dispatch(
            session_id=session_id,
            integration_type=integration_type,
            webhook_url=payload.webhook_url,
            artifact_ids=list(payload.artifact_ids or []),
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meeting session not found") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    events_service.emit(
        session_id=session_id,
        event_type="integration.queued",
        data={
            "dispatch_id": dispatch_row.get("id"),
            "integration_type": dispatch_row.get("integration_type"),
        },
    )
    return _to_share_response(dispatch_row)


@router.post(
    "/sessions/{session_id}/share/slack",
    response_model=MeetingShareResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(check_rate_limit)],
)
def share_session_to_slack(
    session_id: str,
    payload: MeetingShareRequest,
    meetings_db: MeetingsDatabase = Depends(get_meetings_db_for_user),
) -> MeetingShareResponse:
    return _queue_share_dispatch(
        session_id=session_id,
        integration_type="slack",
        payload=payload,
        meetings_db=meetings_db,
    )


@router.post(
    "/sessions/{session_id}/share/webhook",
    response_model=MeetingShareResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(check_rate_limit)],
)
def share_session_to_webhook(
    session_id: str,
    payload: MeetingShareRequest,
    meetings_db: MeetingsDatabase = Depends(get_meetings_db_for_user),
) -> MeetingShareResponse:
    return _queue_share_dispatch(
        session_id=session_id,
        integration_type="webhook",
        payload=payload,
        meetings_db=meetings_db,
    )


@router.get(
    "/sessions/{session_id}/events",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Meeting session event stream.",
            "content": {
                "text/event-stream": {},
            },
        },
    },
    dependencies=[Depends(check_rate_limit)],
)
async def stream_session_events(
    session_id: str,
    meetings_db: MeetingsDatabase = Depends(get_meetings_db_for_user),
) -> StreamingResponse:
    session_service = MeetingSessionService(db=meetings_db)
    events_service = MeetingEventsService(db=meetings_db)

    try:
        session_row = await asyncio.to_thread(session_service.get_session, session_id=session_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meeting session not found") from exc

    events = await asyncio.to_thread(events_service.recent, session_id=session_id, limit=100)
    if not events:
        events = [events_service.snapshot_for_session(session_row)]

    async def _event_stream():
        for event in events:
            yield to_sse_frame(event)
        yield to_sse_frame(
            build_meeting_event(
                event_type="stream.complete",
                session_id=session_id,
                data={"count": len(events)},
            )
        )

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


def _resolve_ws_event_persistence(msg_type: str, message: dict[str, Any]) -> tuple[str, bool]:
    is_final = bool(message.get("final")) or bool(message.get("is_final"))
    if msg_type in {"transcript.final", "final", "segment.final"} or is_final:
        return "transcript.final", True
    if msg_type in {"transcript.partial", "partial", "transcript.interim", "interim", ""}:
        return "transcript.partial", False
    if msg_type.startswith("transcript.") and msg_type != "transcript.partial":
        return msg_type, True
    return "transcript.partial", False


@router.websocket("/sessions/{session_id}/stream")
async def stream_session_ws(
    websocket: WebSocket,
    session_id: str,
    meetings_db: MeetingsDatabase = Depends(get_meetings_db_for_websocket),
) -> None:
    session_service = MeetingSessionService(db=meetings_db)
    events_service = MeetingEventsService(db=meetings_db)

    await websocket.accept()
    try:
        session_row = await asyncio.to_thread(session_service.get_session, session_id=session_id)
    except KeyError:
        await websocket.send_json({"type": "error", "detail": "Meeting session not found", "session_id": session_id})
        await websocket.close(code=4404)
        return

    await websocket.send_json(events_service.snapshot_for_session(session_row))

    while True:
        try:
            message = await websocket.receive_json()
        except WebSocketDisconnect:
            break
        except Exception as exc:
            logger.warning("Meetings websocket received invalid JSON frame")
            await websocket.send_json({"type": "error", "detail": "invalid_message", "session_id": session_id})
            continue

        if not isinstance(message, dict):
            await websocket.send_json({"type": "error", "detail": "invalid_message", "session_id": session_id})
            continue

        msg_type = str(message.get("type") or "").strip().lower()
        if msg_type == "ping":
            await websocket.send_json({"type": "pong", "session_id": session_id})
            continue
        if msg_type in {"close", "disconnect"}:
            await websocket.close(code=1000)
            break

        event_type, persist_event = _resolve_ws_event_persistence(msg_type, message)
        if persist_event:
            event = await asyncio.to_thread(
                events_service.emit,
                session_id=session_id,
                event_type=event_type,
                data=message,
            )
        else:
            event = build_meeting_event(
                event_type=event_type,
                session_id=session_id,
                data=message,
            )
        await websocket.send_json(event)
