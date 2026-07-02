"""Native realtime speech WebSocket endpoints."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, WebSocket

from tldw_Server_API.app.core.Audio.Realtime.capabilities import build_realtime_capabilities
from tldw_Server_API.app.core.Audio.Realtime.default_pipeline import build_default_realtime_pipeline
from tldw_Server_API.app.core.Audio.Realtime.handler import handle_realtime_websocket
from tldw_Server_API.app.core.Audio.Realtime.persistence import NoopRealtimePersistenceAdapter
from tldw_Server_API.app.core.Audio.Realtime.pipeline import RealtimePipeline

router = APIRouter(tags=["Audio Realtime"])
ws_router = APIRouter(tags=["Audio Realtime"])


def _default_realtime_pipeline_factory(principal: Any | None = None, user_id: int | None = None) -> RealtimePipeline:
    return build_default_realtime_pipeline(principal=principal, user_id=user_id)


def _default_realtime_persistence_factory() -> NoopRealtimePersistenceAdapter:
    return NoopRealtimePersistenceAdapter()


DEFAULT_REALTIME_PIPELINE_FACTORY = _default_realtime_pipeline_factory
DEFAULT_REALTIME_PERSISTENCE_FACTORY = _default_realtime_persistence_factory


@router.get("/realtime/capabilities")
async def get_realtime_capabilities() -> dict[str, Any]:
    """Return native and OpenAI-compatible realtime capability metadata."""

    return asdict(build_realtime_capabilities())


@ws_router.websocket("/realtime")
async def websocket_realtime(websocket: WebSocket) -> None:
    """Handle native realtime speech WebSocket sessions."""

    await handle_realtime_websocket(
        websocket,
        "native",
        DEFAULT_REALTIME_PIPELINE_FACTORY,
        DEFAULT_REALTIME_PERSISTENCE_FACTORY,
    )
