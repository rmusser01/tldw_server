"""Native realtime speech WebSocket endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, WebSocket

from tldw_Server_API.app.core.Audio.Realtime.capabilities import build_realtime_capabilities
from tldw_Server_API.app.core.Audio.Realtime.handler import handle_realtime_websocket
from tldw_Server_API.app.core.Audio.Realtime.persistence import NoopRealtimePersistenceAdapter
from tldw_Server_API.app.core.Audio.Realtime.pipeline import RealtimePipeline

router = APIRouter(tags=["Audio Realtime"])
ws_router = APIRouter(tags=["Audio Realtime"])


class _UnavailableRealtimePipeline:
    async def transcribe_pcm16(self, audio: bytes, *, sample_rate_hz: int, language: str | None) -> str:  # noqa: ARG002
        raise NotImplementedError("Realtime STT/LLM/TTS pipeline is provided by Stage 4")

    async def stream_turn(self, transcript: str, *, config: Any):  # noqa: ANN401, ARG002
        if False:
            yield None
        raise NotImplementedError("Realtime STT/LLM/TTS pipeline is provided by Stage 4")


def _default_realtime_pipeline_factory() -> RealtimePipeline:
    return _UnavailableRealtimePipeline()


def _default_realtime_persistence_factory() -> NoopRealtimePersistenceAdapter:
    return NoopRealtimePersistenceAdapter()


DEFAULT_REALTIME_PIPELINE_FACTORY = _default_realtime_pipeline_factory
DEFAULT_REALTIME_PERSISTENCE_FACTORY = _default_realtime_persistence_factory


@router.get("/realtime/capabilities")
async def get_realtime_capabilities() -> dict[str, Any]:
    """Return native and OpenAI-compatible realtime capability metadata."""

    return build_realtime_capabilities().__dict__


@ws_router.websocket("/realtime")
async def websocket_realtime(websocket: WebSocket) -> None:
    """Handle native realtime speech WebSocket sessions."""

    await handle_realtime_websocket(
        websocket,
        "native",
        DEFAULT_REALTIME_PIPELINE_FACTORY,
        DEFAULT_REALTIME_PERSISTENCE_FACTORY,
    )
