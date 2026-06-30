"""API routes for deep research session lifecycle management."""

from __future__ import annotations

import asyncio
import contextlib
import os

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query
from fastapi.responses import StreamingResponse

from tldw_Server_API.app.api.v1.schemas.research_runs_schemas import (
    ResearchArtifactResponse,
    ResearchCheckpointPatchApproveRequest,
    ResearchRunCreateRequest,
    ResearchRunDeleteResponse,
    ResearchRunListItemResponse,
    ResearchRunResponse,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User
from tldw_Server_API.app.core.Research.streaming import (
    diff_stream_events,
    initial_stream_events,
    load_research_stream_state,
    persisted_event_to_stream_event,
    synthetic_terminal_payload,
)
from tldw_Server_API.app.core.Research.service import ResearchService
from tldw_Server_API.app.core.Streaming.streams import SSEStream
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.Workflows.research_wait_bridge import (
    resume_workflows_waiting_on_research_checkpoint,
)

router = APIRouter(prefix="/research", tags=["research-runs"])
_RESEARCH_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


def get_research_service() -> ResearchService:
    """Return the default deep research service."""
    return ResearchService(research_db_path=None, outputs_dir=None, job_manager=None)


def _raise_research_http_error(exc: Exception) -> None:
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from None
    if isinstance(exc, KeyError):
        raise HTTPException(status_code=404, detail=f"research_not_found:{exc.args[0]}") from None
    raise exc


def _job_manager_for_stream(service: ResearchService) -> object | None:
    factory = getattr(service, "get_job_manager", None)
    if not callable(factory):
        return None
    try:
        return factory()
    except Exception:
        return None


@router.post("/runs", response_model=ResearchRunResponse, summary="Create a deep research run")
async def create_research_run(
    payload: ResearchRunCreateRequest = Body(...),
    current_user: User = Depends(get_request_user),
    service: ResearchService = Depends(get_research_service),
) -> ResearchRunResponse:
    try:
        session = service.create_session(
            owner_user_id=str(current_user.id),
            query=payload.query,
            source_policy=payload.source_policy,
            autonomy_mode=payload.autonomy_mode,
            limits_json=payload.limits_json,
            provider_overrides=payload.provider_overrides,
            chat_handoff=payload.chat_handoff.model_dump() if payload.chat_handoff is not None else None,
            follow_up=payload.follow_up.model_dump() if payload.follow_up is not None else None,
        )
    except (KeyError, ValueError) as exc:
        _raise_research_http_error(exc)
    return ResearchRunResponse.model_validate(session)


@router.get("/runs", response_model=list[ResearchRunListItemResponse], summary="List deep research runs")
async def list_research_runs(
    limit: int = Query(25, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None, min_length=1),
    session_id: str | None = Query(None, min_length=1),
    current_user: User = Depends(get_request_user),
    service: ResearchService = Depends(get_research_service),
) -> list[ResearchRunListItemResponse]:
    sessions = service.list_sessions(
        owner_user_id=str(current_user.id),
        limit=limit,
        offset=offset,
        status=status,
        session_id=session_id,
    )
    return [ResearchRunListItemResponse.model_validate(session) for session in sessions]


@router.get("/runs/{session_id}", response_model=ResearchRunResponse, summary="Get a deep research run")
async def get_research_run(
    session_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
    service: ResearchService = Depends(get_research_service),
) -> ResearchRunResponse:
    try:
        session = service.get_session(
            owner_user_id=str(current_user.id),
            session_id=session_id,
        )
    except (KeyError, ValueError) as exc:
        _raise_research_http_error(exc)
    return ResearchRunResponse.model_validate(session)


@router.delete("/runs/{session_id}", response_model=ResearchRunDeleteResponse, summary="Delete a deep research run")
async def delete_research_run(
    session_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
    service: ResearchService = Depends(get_research_service),
) -> ResearchRunDeleteResponse:
    try:
        deleted = service.delete_run(
            owner_user_id=str(current_user.id),
            session_id=session_id,
        )
    except (KeyError, ValueError) as exc:
        _raise_research_http_error(exc)
    return ResearchRunDeleteResponse(deleted=bool(deleted))


@router.get(
    "/runs/{session_id}/events/stream",
    summary="Stream live deep research run events",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-sent events stream of research run updates",
            "content": {"text/event-stream": {}},
        },
    },
)
async def stream_research_run_events(
    session_id: str = Path(..., min_length=1),
    after_id: int = Query(0, ge=0),
    current_user: User = Depends(get_request_user),
    service: ResearchService = Depends(get_research_service),
) -> StreamingResponse:
    job_manager = _job_manager_for_stream(service)
    try:
        initial_state = load_research_stream_state(
            service=service,
            owner_user_id=str(current_user.id),
            session_id=session_id,
            job_manager=job_manager,
        )
    except (KeyError, ValueError) as exc:
        _raise_research_http_error(exc)

    poll_interval = float(os.getenv("RESEARCH_RUNS_SSE_POLL_INTERVAL", "1.0") or "1.0")
    max_duration_s: float | None = None
    try:
        if is_test_mode():
            max_duration_s = float(os.getenv("RESEARCH_RUNS_SSE_TEST_MAX_SECONDS", "1.0") or "1.0")
    except (OSError, ValueError, TypeError):
        max_duration_s = 1.0

    stream = SSEStream(
        heartbeat_interval_s=poll_interval,
        heartbeat_mode="data",
        max_duration_s=max_duration_s,
        labels={"component": "research", "endpoint": "research_run_events_sse"},
    )

    async def _producer() -> None:
        state = initial_state
        list_run_events_after = getattr(service, "list_run_events_after", None)
        if not callable(list_run_events_after):
            for event in initial_stream_events(state):
                await stream.send_event(event.event, event.data, event_id=event.event_id)
            if state.snapshot.run.status in _RESEARCH_TERMINAL_STATUSES:
                await stream.done()
                return

            while True:
                await asyncio.sleep(poll_interval)
                try:
                    current = load_research_stream_state(
                        service=service,
                        owner_user_id=str(current_user.id),
                        session_id=session_id,
                        job_manager=job_manager,
                    )
                except (KeyError, ValueError) as exc:
                    await stream.error(
                        "research_stream_error",
                        str(exc),
                        close=True,
                    )
                    return

                for event in diff_stream_events(previous=state, current=current):
                    await stream.send_event(event.event, event.data, event_id=event.event_id)
                state = current
                if state.snapshot.run.status in _RESEARCH_TERMINAL_STATUSES:
                    await stream.done()
                    return

        snapshot_event = initial_stream_events(state)[0]
        await stream.send_event(
            snapshot_event.event,
            snapshot_event.data,
            event_id=snapshot_event.event_id,
        )

        cursor = int(after_id)
        replayed_terminal = False
        replay_rows = list_run_events_after(
            owner_user_id=str(current_user.id),
            session_id=session_id,
            after_id=cursor,
        )
        for event_row in replay_rows:
            stream_event = persisted_event_to_stream_event(event_row, replayed=True)
            await stream.send_event(
                stream_event.event,
                stream_event.data,
                event_id=stream_event.event_id,
            )
            cursor = max(cursor, int(event_row.id))
            if event_row.event_type == "terminal":
                replayed_terminal = True

        if state.snapshot.run.status in _RESEARCH_TERMINAL_STATUSES:
            if not replayed_terminal:
                synthetic_terminal = synthetic_terminal_payload(state.snapshot)
                await stream.send_event(
                    "terminal",
                    synthetic_terminal,
                    event_id=(
                        str(state.snapshot.latest_event_id)
                        if state.snapshot.latest_event_id > 0
                        else None
                    ),
                )
            await stream.done()
            return

        while True:
            await asyncio.sleep(poll_interval)
            try:
                current = load_research_stream_state(
                    service=service,
                    owner_user_id=str(current_user.id),
                    session_id=session_id,
                    job_manager=job_manager,
                )
            except (KeyError, ValueError) as exc:
                await stream.error(
                    "research_stream_error",
                    str(exc),
                    close=True,
                )
                return

            live_rows = list_run_events_after(
                owner_user_id=str(current_user.id),
                session_id=session_id,
                after_id=cursor,
            )
            for event_row in live_rows:
                stream_event = persisted_event_to_stream_event(event_row, replayed=False)
                await stream.send_event(
                    stream_event.event,
                    stream_event.data,
                    event_id=stream_event.event_id,
                )
                cursor = max(cursor, int(event_row.id))
                if event_row.event_type == "terminal":
                    await stream.done()
                    return
            state = current
            if (
                state.snapshot.run.status in _RESEARCH_TERMINAL_STATUSES
                and state.snapshot.latest_event_id <= cursor
            ):
                synthetic_terminal = synthetic_terminal_payload(state.snapshot)
                await stream.send_event(
                    "terminal",
                    synthetic_terminal,
                    event_id=(
                        str(state.snapshot.latest_event_id)
                        if state.snapshot.latest_event_id > 0
                        else None
                    ),
                )
                await stream.done()
                return

    async def _gen():
        producer = asyncio.create_task(_producer())
        try:
            async for line in stream.iter_sse():
                yield line
        finally:
            if not producer.done():
                with contextlib.suppress(asyncio.CancelledError, RuntimeError, OSError):
                    producer.cancel()
                with contextlib.suppress(asyncio.CancelledError, RuntimeError, OSError):
                    await producer

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/runs/{session_id}/pause", response_model=ResearchRunResponse, summary="Pause a deep research run")
async def pause_research_run(
    session_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
    service: ResearchService = Depends(get_research_service),
) -> ResearchRunResponse:
    try:
        session = service.pause_run(
            owner_user_id=str(current_user.id),
            session_id=session_id,
        )
    except (KeyError, ValueError) as exc:
        _raise_research_http_error(exc)
    return ResearchRunResponse.model_validate(session)


@router.post("/runs/{session_id}/resume", response_model=ResearchRunResponse, summary="Resume a deep research run")
async def resume_research_run(
    session_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
    service: ResearchService = Depends(get_research_service),
) -> ResearchRunResponse:
    try:
        session = service.resume_run(
            owner_user_id=str(current_user.id),
            session_id=session_id,
        )
    except (KeyError, ValueError) as exc:
        _raise_research_http_error(exc)
    return ResearchRunResponse.model_validate(session)


@router.post("/runs/{session_id}/cancel", response_model=ResearchRunResponse, summary="Cancel a deep research run")
async def cancel_research_run(
    session_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
    service: ResearchService = Depends(get_research_service),
) -> ResearchRunResponse:
    try:
        session = service.cancel_run(
            owner_user_id=str(current_user.id),
            session_id=session_id,
        )
    except (KeyError, ValueError) as exc:
        _raise_research_http_error(exc)
    return ResearchRunResponse.model_validate(session)


@router.get("/runs/{session_id}/bundle", summary="Get the final deep research bundle")
async def get_research_bundle(
    session_id: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
    service: ResearchService = Depends(get_research_service),
) -> dict:
    try:
        return service.get_bundle(
            owner_user_id=str(current_user.id),
            session_id=session_id,
        )
    except (KeyError, ValueError) as exc:
        _raise_research_http_error(exc)


@router.get(
    "/runs/{session_id}/artifacts/{artifact_name}",
    response_model=ResearchArtifactResponse,
    summary="Get an allowlisted deep research artifact",
)
async def get_research_artifact(
    session_id: str = Path(..., min_length=1),
    artifact_name: str = Path(..., min_length=1),
    current_user: User = Depends(get_request_user),
    service: ResearchService = Depends(get_research_service),
) -> ResearchArtifactResponse:
    try:
        artifact = service.get_artifact(
            owner_user_id=str(current_user.id),
            session_id=session_id,
            artifact_name=artifact_name,
        )
    except (KeyError, ValueError) as exc:
        _raise_research_http_error(exc)
    return ResearchArtifactResponse.model_validate(artifact)


@router.post(
    "/runs/{session_id}/checkpoints/{checkpoint_id}/patch-and-approve",
    response_model=ResearchRunResponse,
    summary="Patch and approve a deep research checkpoint",
)
async def patch_and_approve_research_checkpoint(
    session_id: str = Path(..., min_length=1),
    checkpoint_id: str = Path(..., min_length=1),
    payload: ResearchCheckpointPatchApproveRequest = Body(default_factory=ResearchCheckpointPatchApproveRequest),
    current_user: User = Depends(get_request_user),
    service: ResearchService = Depends(get_research_service),
) -> ResearchRunResponse:
    try:
        session = service.approve_checkpoint(
            owner_user_id=str(current_user.id),
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            patch_payload=payload.patch_payload,
        )
    except (KeyError, ValueError) as exc:
        _raise_research_http_error(exc)
    with contextlib.suppress(Exception):
        asyncio.create_task(
            resume_workflows_waiting_on_research_checkpoint(
                research_run_id=session_id,
                checkpoint_id=checkpoint_id,
            )
        )
        await asyncio.sleep(0)
    return ResearchRunResponse.model_validate(session)
