"""Audio endpoints package.

Exports only what callers still read from the package: the routers (lazy), the
lazy ``websocket_*`` wrappers, ``audio_tts`` and the names tests use as
``dependency_overrides`` keys (``get_tts_service``, ``create_speech``,
``check_rate_limit``, ``get_usage_event_logger``). Patch everything else at its
home module. Heavy endpoint aggregation (``audio.audio``) is intentionally not
imported at module import time.
"""

import importlib

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    check_rate_limit,
)
from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import (
    get_usage_event_logger,
)

from . import audio_tts as audio_tts

# Dependency-override keys.
create_speech = audio_tts.create_speech
get_tts_service = audio_tts.get_tts_service


async def websocket_audio_chat_stream(*args, **kwargs):
    from .audio_streaming import websocket_audio_chat_stream as _impl

    return await _impl(*args, **kwargs)


async def websocket_tts(*args, **kwargs):
    from .audio_streaming import websocket_tts as _impl

    return await _impl(*args, **kwargs)


async def websocket_tts_realtime(*args, **kwargs):
    from .audio_streaming import websocket_tts_realtime as _impl

    return await _impl(*args, **kwargs)


async def websocket_transcribe(*args, **kwargs):
    from .audio_streaming import websocket_transcribe as _impl

    return await _impl(*args, **kwargs)


_LAZY_ROUTERS = {"router", "ws_router"}


def __getattr__(name: str):
    """Lazily forward ``router`` / ``ws_router`` to the aggregate audio module."""
    if name in _LAZY_ROUTERS:
        try:
            aggregate = importlib.import_module(f"{__name__}.audio")
        except Exception as exc:  # pragma: no cover - defensive import fallback
            raise AttributeError(name) from exc
        return getattr(aggregate, name)
    raise AttributeError(name)


__all__ = [
    "audio_tts",
    "check_rate_limit",
    "create_speech",
    "get_tts_service",
    "get_usage_event_logger",
    "websocket_audio_chat_stream",
    "websocket_transcribe",
    "websocket_tts",
    "websocket_tts_realtime",
]
