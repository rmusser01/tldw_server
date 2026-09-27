"""OpenAI-compatible realtime WebSocket endpoints."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket

from tldw_Server_API.app.api.v1.endpoints.audio.audio_realtime import (
    _default_realtime_persistence_factory,
    _default_realtime_pipeline_factory,
)
from tldw_Server_API.app.core.Audio.Realtime.handler import handle_realtime_websocket

router = APIRouter(tags=["OpenAI Realtime Compatibility"])

DEFAULT_REALTIME_PIPELINE_FACTORY = _default_realtime_pipeline_factory
DEFAULT_REALTIME_PERSISTENCE_FACTORY = _default_realtime_persistence_factory


@router.websocket("/realtime")
async def websocket_realtime_compat(websocket: WebSocket) -> None:
    """Handle OpenAI-compatible realtime speech WebSocket sessions."""

    await handle_realtime_websocket(
        websocket,
        "openai_compat",
        DEFAULT_REALTIME_PIPELINE_FACTORY,
        DEFAULT_REALTIME_PERSISTENCE_FACTORY,
    )
