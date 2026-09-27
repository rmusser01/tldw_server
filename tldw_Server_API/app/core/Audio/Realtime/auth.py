"""Authentication adapter for realtime speech WebSocket routes."""

from __future__ import annotations

from typing import Literal

from fastapi import WebSocket

from tldw_Server_API.app.core.Audio.streaming_service import _audio_ws_authenticate

RealtimeRouteKind = Literal["native", "openai_compat"]


def _route_path(route_kind: RealtimeRouteKind) -> str:
    if route_kind == "openai_compat":
        return "/v1/realtime"
    if route_kind == "native":
        return "/api/v1/audio/realtime"
    raise ValueError(f"Unsupported realtime route kind: {route_kind!r}")


async def authenticate_realtime_websocket(
    websocket: WebSocket,
    route_kind: RealtimeRouteKind,
) -> tuple[bool, int | None]:
    """Authenticate realtime WebSockets without consuming protocol events."""

    return await _audio_ws_authenticate(
        websocket,
        None,
        endpoint_id="audio.realtime",
        ws_path=_route_path(route_kind),
        allow_initial_auth_message=False,
    )
