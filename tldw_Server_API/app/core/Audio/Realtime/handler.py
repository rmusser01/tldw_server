"""WebSocket handler for JSON-only OpenAI-shaped realtime speech events."""

from __future__ import annotations

import asyncio
import contextlib
import json
import inspect
from collections.abc import Callable
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from tldw_Server_API.app.core.Audio.Realtime.auth import RealtimeRouteKind, authenticate_realtime_websocket
from tldw_Server_API.app.core.Audio.Realtime.constants import (
    REALTIME_INTERNAL_ERROR_CLOSE_CODE,
    REALTIME_MAX_JSON_FRAME_BYTES,
    REALTIME_PAYLOAD_TOO_LARGE_CLOSE_CODE,
)
from tldw_Server_API.app.core.Audio.Realtime.models import (
    ClientCommand,
    CreateResponseCommand,
    RealtimeErrorEvent,
    RealtimeLimits,
    RealtimeServerEvent,
)
from tldw_Server_API.app.core.Audio.Realtime.protocol import parse_client_event, to_openai_server_event
from tldw_Server_API.app.core.Audio.Realtime.session import RealtimeSession

PipelineFactory = Callable[..., Any]
PersistenceFactory = Callable[[], Any]


async def handle_realtime_websocket(
    websocket: WebSocket,
    route_kind: RealtimeRouteKind,
    pipeline_factory: PipelineFactory,
    persistence_factory: PersistenceFactory,
) -> None:
    """Run an authenticated realtime speech WebSocket session."""

    authenticated, user_id = await authenticate_realtime_websocket(websocket, route_kind)
    if not authenticated:
        return

    try:
        websocket_state = getattr(websocket, "state", None)
        principal = getattr(websocket_state, "auth_principal", None)
        session = RealtimeSession(
            pipeline=_call_pipeline_factory(pipeline_factory, principal=principal, user_id=user_id),
            persistence_adapter=persistence_factory(),
        )
        limits = RealtimeLimits()
        await websocket.accept()
        await _send_session_start(websocket, session)
        await _run_realtime_loop(websocket, session, limits)
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Realtime WebSocket fatal error: {exc}")
        try:
            await websocket.close(code=REALTIME_INTERNAL_ERROR_CLOSE_CODE)
        except Exception as close_exc:  # noqa: BLE001
            logger.debug(f"Failed to close realtime websocket after fatal error: {close_exc}")


async def _run_realtime_loop(websocket: WebSocket, session: RealtimeSession, limits: RealtimeLimits) -> None:
    outbound_events: asyncio.Queue[RealtimeServerEvent | BaseException] = asyncio.Queue()
    receive_task = asyncio.create_task(websocket.receive())
    generation_task: asyncio.Task[None] | None = None
    outbound_task: asyncio.Task[RealtimeServerEvent | BaseException] | None = None

    try:
        while True:
            if (
                generation_task is not None
                and outbound_task is None
                and not (generation_task.done() and outbound_events.empty())
            ):
                outbound_task = asyncio.create_task(outbound_events.get())

            wait_for = [receive_task]
            if outbound_task is not None:
                wait_for.append(outbound_task)

            done, _pending = await asyncio.wait(wait_for, return_when=asyncio.FIRST_COMPLETED)

            if receive_task in done:
                message = receive_task.result()
                receive_task = asyncio.create_task(websocket.receive())
                command_or_error = await _parse_realtime_message(websocket, message, limits)
                if command_or_error is None:
                    return
                if isinstance(command_or_error, RealtimeErrorEvent):
                    await _send_event(websocket, command_or_error)
                elif isinstance(command_or_error, CreateResponseCommand):
                    if generation_task is not None:
                        await _send_event(
                            websocket,
                            RealtimeErrorEvent(
                                code="invalid_request",
                                message="response.create is not allowed while another response is active",
                                event_id=command_or_error.event_id,
                            ),
                        )
                    else:
                        generation_task = asyncio.create_task(
                            _enqueue_session_events(session, command_or_error, outbound_events)
                        )
                        session.set_active_task(generation_task)
                else:
                    async for event in session.handle_command(command_or_error):
                        await _send_event(websocket, event)

            if outbound_task is not None and outbound_task in done:
                outbound = outbound_task.result()
                outbound_task = None
                if isinstance(outbound, BaseException):
                    raise outbound
                await _send_event(websocket, outbound)

            if generation_task is not None and generation_task.done() and outbound_events.empty():
                if outbound_task is not None:
                    outbound_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await outbound_task
                    outbound_task = None
                await _observe_generation_task(generation_task)
                if session.active_task is generation_task:
                    session.set_active_task(None)
                generation_task = None
    finally:
        for task in (receive_task, outbound_task, generation_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task


async def _parse_realtime_message(
    websocket: WebSocket,
    message: dict[str, Any],
    limits: RealtimeLimits,
) -> ClientCommand | RealtimeErrorEvent | None:
    if message.get("type") == "websocket.disconnect":
        return None
    if "bytes" in message and message.get("bytes") is not None:
        return RealtimeErrorEvent(
            code="invalid_event",
            message="Binary WebSocket frames are not supported; send JSON text frames.",
        )
    text = message.get("text")
    if not isinstance(text, str):
        return RealtimeErrorEvent(code="invalid_event", message="Realtime events must be JSON text frames.")
    if len(text.encode("utf-8")) > REALTIME_MAX_JSON_FRAME_BYTES:
        await websocket.close(code=REALTIME_PAYLOAD_TOO_LARGE_CLOSE_CODE)
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return RealtimeErrorEvent(code="invalid_event", message="Realtime event frame must be valid JSON.")
    if not isinstance(payload, dict):
        return RealtimeErrorEvent(code="invalid_event", message="Realtime event frame must be a JSON object.")
    return parse_client_event(payload, limits)


async def _enqueue_session_events(
    session: RealtimeSession,
    command: ClientCommand,
    queue: asyncio.Queue[RealtimeServerEvent | BaseException],
) -> None:
    try:
        async for event in session.handle_command(command):
            await queue.put(event)
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        await queue.put(exc)


async def _observe_generation_task(task: asyncio.Task[None]) -> None:
    try:
        await task
    except asyncio.CancelledError:
        return


async def _send_session_start(websocket: WebSocket, session: RealtimeSession) -> None:
    for event in session.drain_pending_events():
        await _send_event(websocket, event)
    await websocket.send_json({"type": "rate_limits.updated", "event_id": None, "rate_limits": []})


async def _send_event(websocket: WebSocket, event: Any) -> None:
    await websocket.send_json(to_openai_server_event(event))


def _call_pipeline_factory(
    pipeline_factory: PipelineFactory,
    *,
    principal: Any | None,
    user_id: int | None,
) -> Any:
    signature = inspect.signature(pipeline_factory)
    if not signature.parameters:
        return pipeline_factory()

    kwargs: dict[str, Any] = {}
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs or "principal" in signature.parameters:
        kwargs["principal"] = principal
    if accepts_kwargs or "user_id" in signature.parameters:
        kwargs["user_id"] = user_id
    return pipeline_factory(**kwargs)
