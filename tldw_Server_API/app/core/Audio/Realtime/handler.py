"""WebSocket handler for JSON-only OpenAI-shaped realtime speech events."""

from __future__ import annotations

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
    RealtimeErrorEvent,
    RealtimeLimits,
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

        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
            if "bytes" in message and message.get("bytes") is not None:
                await _send_event(
                    websocket,
                    RealtimeErrorEvent(
                        code="invalid_event",
                        message="Binary WebSocket frames are not supported; send JSON text frames.",
                    ),
                )
                continue
            text = message.get("text")
            if not isinstance(text, str):
                await _send_event(
                    websocket,
                    RealtimeErrorEvent(code="invalid_event", message="Realtime events must be JSON text frames."),
                )
                continue
            if len(text.encode("utf-8")) > REALTIME_MAX_JSON_FRAME_BYTES:
                await websocket.close(code=REALTIME_PAYLOAD_TOO_LARGE_CLOSE_CODE)
                return
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                await _send_event(
                    websocket,
                    RealtimeErrorEvent(code="invalid_event", message="Realtime event frame must be valid JSON."),
                )
                continue
            if not isinstance(payload, dict):
                await _send_event(
                    websocket,
                    RealtimeErrorEvent(code="invalid_event", message="Realtime event frame must be a JSON object."),
                )
                continue

            command_or_error = parse_client_event(payload, limits)
            if isinstance(command_or_error, RealtimeErrorEvent):
                await _send_event(websocket, command_or_error)
                continue

            async for event in session.handle_command(command_or_error):
                await _send_event(websocket, event)
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Realtime WebSocket fatal error: {exc}")
        try:
            await websocket.close(code=REALTIME_INTERNAL_ERROR_CLOSE_CODE)
        except Exception as close_exc:  # noqa: BLE001
            logger.debug(f"Failed to close realtime websocket after fatal error: {close_exc}")


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
