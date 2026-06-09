"""Audio endpoints package.

Explicit, lightweight exports for endpoint helpers and shim patch points.
Heavy endpoint aggregation (``audio.audio``) is intentionally not imported
at module import time.
"""

import asyncio as asyncio
import importlib
import os

import soundfile as sf
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
    get_chacha_db_for_user_id,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    check_rate_limit,
)
from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import (
    get_usage_event_logger,
)
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import get_api_keys
from tldw_Server_API.app.core.Audio.streaming_service import (
    _audio_ws_authenticate,
    _stream_tts_to_websocket,
)
from tldw_Server_API.app.core.Audio.tokenizer_service import (
    _get_qwen3_tokenizer_settings,
    _load_qwen3_tokenizer,
)
from tldw_Server_API.app.core.Audio.tts_service import (
    _sanitize_speech_request,
    _tts_fallback_resolver,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import resolve_byok_credentials
from tldw_Server_API.app.core.Chat.chat_helpers import (
    get_or_create_character_context,
    get_or_create_conversation,
)
from tldw_Server_API.app.core.Chat.chat_service import (
    perform_chat_api_call_async as chat_api_call_async,
)
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry
from tldw_Server_API.app.core.Storage.generated_file_helpers import (
    save_and_register_tts_audio,
)
from tldw_Server_API.app.core.Usage.audio_quota import (
    active_streams_count,
    add_daily_minutes,
    bytes_to_seconds,
    can_start_job,
    can_start_stream,
    check_daily_minutes_allow,
    finish_job,
    finish_stream,
    get_daily_minutes_used,
    get_job_heartbeat_interval_seconds,
    get_limits_for_user,
    get_user_tier,
    heartbeat_jobs,
    heartbeat_stream,
    increment_jobs_started,
)
from tldw_Server_API.app.core.config import load_comprehensive_config

from . import audio_tts as audio_tts

# Endpoint helpers that tests patch at package scope.
create_speech = audio_tts.create_speech
get_tts_service = audio_tts.get_tts_service


def create_voice_conversion(*args, **kwargs):
    """Lazy wrapper for the Chatterbox voice conversion endpoint."""
    from .audio_voice_conversion import create_voice_conversion as _impl

    return _impl(*args, **kwargs)


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


async def _resolve_tts_byok(*args, **kwargs):
    """Lazy wrapper preserving the historical package-level patch point."""
    from .audio import _resolve_tts_byok as _impl

    return await _impl(*args, **kwargs)


def UnifiedStreamingTranscriber(*args, **kwargs):
    """Lazy constructor wrapper to avoid importing streaming backends eagerly."""
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import (
        UnifiedStreamingTranscriber as _impl,
    )

    return _impl(*args, **kwargs)


def SileroTurnDetector(*args, **kwargs):
    """Lazy constructor wrapper to avoid importing streaming backends eagerly."""
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import (
        SileroTurnDetector as _impl,
    )

    return _impl(*args, **kwargs)


def _get_failopen_cap_minutes() -> float:
    """Return per-connection fail-open cap in minutes for streaming quotas.

    Resolution order:
      1) Env var AUDIO_FAILOPEN_CAP_MINUTES (>0)
      2) Config [Audio-Quota] failopen_cap_minutes (>0)
      3) Config [Audio] failopen_cap_minutes (>0)
      4) Default 5.0
    """
    v = os.getenv("AUDIO_FAILOPEN_CAP_MINUTES")
    if v is not None:
        try:
            f = float(v)
            if f > 0:
                return f
        except (ValueError, TypeError):
            logger.debug("AUDIO_FAILOPEN_CAP_MINUTES parse failed")
    try:
        cfg = load_comprehensive_config()
        if cfg is not None:
            if cfg.has_section("Audio-Quota"):
                try:
                    f = float(cfg.get("Audio-Quota", "failopen_cap_minutes", fallback=""))
                    if f > 0:
                        return f
                except (ValueError, TypeError):
                    logger.debug("[Audio-Quota].failopen_cap_minutes parse failed")
            if cfg.has_section("Audio"):
                try:
                    f = float(cfg.get("Audio", "failopen_cap_minutes", fallback=""))
                    if f > 0:
                        return f
                except (ValueError, TypeError):
                    logger.debug("[Audio].failopen_cap_minutes parse failed")
    except Exception:
        logger.debug("Config read for failopen cap failed")
    return 5.0


_LEGACY_AUDIO_ATTRS = {
    "router",
    "ws_router",
    "create_speech_metadata",
    "list_tts_providers",
    "list_tts_voices",
    "get_tts_provider_model_info",
    "reset_tts_metrics",
    "encode_audio_tokenizer",
    "decode_audio_tokenizer",
    "create_transcription",
    "create_translation",
    "segment_transcript",
    "audio_chat_turn",
    "streaming_status",
    "streaming_limits",
    "test_streaming",
    "get_tts_health",
    "get_stt_health",
    "upload_voice",
    "encode_voice_reference",
    "list_voices",
    "get_voice_details",
    "delete_voice",
    "preview_voice",
    "create_voice_conversion",
    "CHAT_HISTORY_MAX_MESSAGES",
}


def __getattr__(name: str):
    """Backwards-compatible lazy attribute forwarding to aggregate audio module.

    Historically callers imported symbols from
    ``tldw_Server_API.app.api.v1.endpoints.audio`` when audio lived in a single
    module. Keep that behavior while preserving lazy imports.
    """
    if name in _LEGACY_AUDIO_ATTRS:
        try:
            aggregate = importlib.import_module(f"{__name__}.audio")
        except Exception as exc:  # pragma: no cover - defensive import fallback
            raise AttributeError(name) from exc
        if hasattr(aggregate, name):
            return getattr(aggregate, name)
    raise AttributeError(name)


__all__ = sorted(
    name
    for name in globals()
    if not name.startswith("_")
    or name
    in {
        "_audio_ws_authenticate",
        "_get_qwen3_tokenizer_settings",
        "_load_qwen3_tokenizer",
        "_resolve_tts_byok",
        "_sanitize_speech_request",
        "_stream_tts_to_websocket",
        "_tts_fallback_resolver",
    }
)
