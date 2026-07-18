# audio.py
# Description: Aggregate audio endpoints and WebSocket routes.
import asyncio as asyncio
import importlib
import os
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from loguru import logger
from starlette import status

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit
from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import get_usage_event_logger
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import get_api_keys
from tldw_Server_API.app.core.Audio.streaming_service import (
    CHAT_HISTORY_MAX_MESSAGES,
    _audio_ws_authenticate,
    _stream_tts_to_websocket,
)
from tldw_Server_API.app.core.Chat.chat_service import (
    perform_chat_api_call_async as chat_api_call_async,
)
from tldw_Server_API.app.core.Metrics.metrics_manager import (
    get_metrics_registry,
)

from . import (
    audio_health,
    audio_history,
    audio_presets,
    audio_tokenizer,
    audio_transcriptions,
    audio_tts,
    audio_voice_conversion,
    audio_voices,
)

router = APIRouter(
    tags=["Audio"],
    responses={
        404: {"description": "Not found"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)

# Include HTTP routers
router.include_router(audio_tts.router)
router.include_router(audio_history.router)
router.include_router(audio_tokenizer.router)
router.include_router(audio_transcriptions.router)
router.include_router(audio_health.router)
router.include_router(audio_voices.router)
router.include_router(audio_presets.router)
router.include_router(audio_voice_conversion.router)

_AUDIO_STREAMING_MODULE = f"{__package__}.audio_streaming"


def _load_audio_streaming() -> Any:
    return importlib.import_module(_AUDIO_STREAMING_MODULE)


def _streaming_attr(name: str) -> Any:
    return getattr(_load_audio_streaming(), name)


def _mount_streaming_routes() -> APIRouter:
    try:
        streaming_module = _load_audio_streaming()
        router.include_router(streaming_module.router)
        return streaming_module.ws_router
    except Exception:
        logger.warning("Audio streaming routes unavailable; skipping import")
        return APIRouter()


# Expose WebSocket router
ws_router = _mount_streaming_routes()

# Re-export selected endpoint callables for tests/backwards-compat imports
create_speech = audio_tts.create_speech
create_speech_metadata = audio_tts.create_speech_metadata
list_tts_providers = audio_tts.list_tts_providers
list_tts_voices = audio_tts.list_tts_voices
reset_tts_metrics = audio_tts.reset_tts_metrics
encode_audio_tokenizer = audio_tokenizer.encode_audio_tokenizer
decode_audio_tokenizer = audio_tokenizer.decode_audio_tokenizer
create_transcription = audio_transcriptions.create_transcription
create_translation = audio_transcriptions.create_translation
segment_transcript = audio_transcriptions.segment_transcript
get_tts_health = audio_health.get_tts_health
get_stt_health = audio_health.get_stt_health
upload_voice = audio_voices.upload_voice
encode_voice_reference = audio_voices.encode_voice_reference
list_voices = audio_voices.list_voices
get_voice_details = audio_voices.get_voice_details
delete_voice = audio_voices.delete_voice
preview_voice = audio_voices.preview_voice
create_fish_s2_reference = audio_voices.create_fish_s2_reference
list_fish_s2_references = audio_voices.list_fish_s2_references
delete_fish_s2_reference = audio_voices.delete_fish_s2_reference

# Dependency helpers (for FastAPI overrides in tests)
get_tts_service = audio_tts.get_tts_service
get_usage_event_logger = get_usage_event_logger
check_rate_limit = check_rate_limit

# Shared helper re-exports used in tests
from tldw_Server_API.app.core.Audio.tts_service import (
    _tts_fallback_resolver,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    resolve_byok_credentials as _default_resolve_byok_credentials,
)
from tldw_Server_API.app.core.config import load_comprehensive_config as _load_comprehensive_config
from tldw_Server_API.app.core.exceptions import TTSPublicHTTPException

# Re-export config loader for tests to monkeypatch
load_comprehensive_config = _load_comprehensive_config
resolve_byok_credentials = _default_resolve_byok_credentials


_TTS_CREDENTIAL_ERROR_MESSAGES = {
    "invalid_provider_credentials": "The selected provider credentials are invalid.",
    "credential_store_unavailable": "Provider credential storage is temporarily unavailable.",
    "credential_scope_revoked": "The selected provider credential scope is no longer available.",
    "provider_disabled": "The selected provider is disabled by administrator policy.",
    "model_not_allowed": "The selected model is not allowed for this provider.",
}


def _tts_credential_http_exception(exc: ByokResolutionError) -> HTTPException:
    """Map typed TTS credential failures to bounded public responses."""
    code = getattr(exc, "policy_code", exc.code)
    message = _TTS_CREDENTIAL_ERROR_MESSAGES.get(code)
    if message is None:
        code = "invalid_provider_credentials"
        message = _TTS_CREDENTIAL_ERROR_MESSAGES[code]
    return TTSPublicHTTPException(
        status_code=(
            status.HTTP_403_FORBIDDEN
            if code in {"provider_disabled", "model_not_allowed", "credential_scope_revoked"}
            else status.HTTP_503_SERVICE_UNAVAILABLE
        ),
        detail={"error_code": code, "message": message},
    )


def _resolve_tts_byok_resolver():
    local_resolver = resolve_byok_credentials
    package_resolver = _default_resolve_byok_credentials
    try:
        from tldw_Server_API.app.api.v1.endpoints import audio as _audio_pkg

        package_resolver = getattr(_audio_pkg, "resolve_byok_credentials", _default_resolve_byok_credentials)
    except (AttributeError, ImportError):
        logger.debug("Falling back to default BYOK resolver after audio package resolver lookup failed")
    if local_resolver is not _default_resolve_byok_credentials:
        return local_resolver
    if package_resolver is not _default_resolve_byok_credentials:
        return package_resolver
    try:
        from tldw_Server_API.app.core.Audio import tts_service as core_tts_service

        core_resolver = getattr(core_tts_service, "resolve_byok_credentials", _default_resolve_byok_credentials)
        if core_resolver is not _default_resolve_byok_credentials:
            return core_resolver
    except (AttributeError, ImportError):
        logger.debug("Falling back to default BYOK resolver after core resolver lookup failed")
    return _default_resolve_byok_credentials


async def _resolve_tts_byok(
    *,
    provider_hint: Optional[str],
    current_user,
    request,
    model: Optional[str] = None,
    force_oauth_refresh: bool = False,
    rejected_credentials: Optional[ResolvedByokCredentials] = None,
):
    """Delegate TTS credential resolution to the single core implementation."""
    from tldw_Server_API.app.core.Audio import tts_service as core_tts_service

    kwargs = {
        "provider_hint": provider_hint,
        "current_user": current_user,
        "request": request,
        "force_oauth_refresh": force_oauth_refresh,
    }
    if rejected_credentials is not None:
        kwargs["rejected_credentials"] = rejected_credentials
    if model is not None:
        kwargs["model"] = model
    resolver = _resolve_tts_byok_resolver()
    if resolver is not _default_resolve_byok_credentials:
        kwargs["credential_resolver"] = resolver
    try:
        return await core_tts_service._resolve_tts_byok(**kwargs)
    except ByokResolutionError as exc:
        raise _tts_credential_http_exception(exc) from None


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
        try:
            from tldw_Server_API.app.api.v1.endpoints import audio as _audio_pkg

            cfg_loader = getattr(_audio_pkg, "load_comprehensive_config", load_comprehensive_config)
        except Exception:
            cfg_loader = load_comprehensive_config
        cfg = cfg_loader()
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


async def audio_chat_turn(*args, **kwargs):
    return await _streaming_attr("audio_chat_turn")(*args, **kwargs)


async def streaming_status(*args, **kwargs):
    return await _streaming_attr("streaming_status")(*args, **kwargs)


async def streaming_limits(*args, **kwargs):
    return await _streaming_attr("streaming_limits")(*args, **kwargs)


async def test_streaming(*args, **kwargs):
    return await _streaming_attr("test_streaming")(*args, **kwargs)


async def websocket_audio_chat_stream(*args, **kwargs):
    return await _streaming_attr("websocket_audio_chat_stream")(*args, **kwargs)


async def websocket_tts(*args, **kwargs):
    return await _streaming_attr("websocket_tts")(*args, **kwargs)


async def websocket_tts_realtime(*args, **kwargs):
    return await _streaming_attr("websocket_tts_realtime")(*args, **kwargs)


async def websocket_transcribe(*args, **kwargs):
    return await _streaming_attr("websocket_transcribe")(*args, **kwargs)


def UnifiedStreamingTranscriber(*args, **kwargs):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import (
        UnifiedStreamingTranscriber as _impl,
    )

    return _impl(*args, **kwargs)


def SileroTurnDetector(*args, **kwargs):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import (
        SileroTurnDetector as _impl,
    )

    return _impl(*args, **kwargs)

# Re-export quota helpers for tests/monkeypatching
from tldw_Server_API.app.core.Usage.audio_quota import (
    add_daily_minutes as add_daily_minutes,
)
from tldw_Server_API.app.core.Usage.audio_quota import (
    bytes_to_seconds as bytes_to_seconds,
)
from tldw_Server_API.app.core.Usage.audio_quota import (
    can_start_stream as can_start_stream,
)
from tldw_Server_API.app.core.Usage.audio_quota import (
    check_daily_minutes_allow as check_daily_minutes_allow,
)
from tldw_Server_API.app.core.Usage.audio_quota import (
    consume_daily_minutes as consume_daily_minutes,
)
from tldw_Server_API.app.core.Usage.audio_quota import (
    finish_stream as finish_stream,
)

# Optional helpers for status/limits and TTL heartbeat
try:
    from tldw_Server_API.app.core.Usage.audio_quota import (
        active_streams_count as active_streams_count,
    )
    from tldw_Server_API.app.core.Usage.audio_quota import (
        get_daily_minutes_used as get_daily_minutes_used,
    )
    from tldw_Server_API.app.core.Usage.audio_quota import (
        get_job_heartbeat_interval_seconds as get_job_heartbeat_interval_seconds,
    )
    from tldw_Server_API.app.core.Usage.audio_quota import (
        get_user_tier as get_user_tier,
    )
    from tldw_Server_API.app.core.Usage.audio_quota import (
        heartbeat_jobs as heartbeat_jobs,
    )
    from tldw_Server_API.app.core.Usage.audio_quota import (
        heartbeat_stream as heartbeat_stream,
    )
except ImportError:
    logger.debug("audio_quota optional helpers not available")

# Expose job quota helpers at module scope for tests to monkeypatch
try:
    from tldw_Server_API.app.core.Usage.audio_quota import (
        can_start_job as can_start_job,
    )
    from tldw_Server_API.app.core.Usage.audio_quota import (
        finish_job as finish_job,
    )
    from tldw_Server_API.app.core.Usage.audio_quota import (
        get_limits_for_user as get_limits_for_user,
    )
    from tldw_Server_API.app.core.Usage.audio_quota import (
        increment_jobs_started as increment_jobs_started,
    )
except ImportError:
    logger.debug("audio_quota job helpers not available")
