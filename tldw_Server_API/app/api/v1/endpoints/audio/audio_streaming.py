# audio_streaming.py
# Description: Audio streaming endpoints (HTTP + WebSocket) and non-streaming chat.
import asyncio
import base64
from collections import deque
import configparser
import contextlib
import importlib
import json
import os
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from loguru import logger
from starlette import status

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import require_token_scope
from tldw_Server_API.app.api.v1.API_Deps.billing_deps import resolve_org_id_for_principal
from tldw_Server_API.app.core.Resource_Governance import cost_units
from tldw_Server_API.app.core.Billing.enforcement import (
    LimitCategory,
    enforcement_enabled,
    get_billing_enforcer,
)
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
    get_chacha_db_for_user,
    get_chacha_db_for_user_id,
)
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import _resolve_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import UsageEventLogger, get_usage_event_logger
from tldw_Server_API.app.api.v1.endpoints.audio.audio_tts import get_tts_service
from tldw_Server_API.app.api.v1.schemas.audio_schemas import (
    OpenAISpeechRequest,
    SpeechChatRequest,
    SpeechChatResponse,
    StreamingLimitsResponse,
    StreamingStatusResponse,
    StreamingTestResponse,
)
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import DEFAULT_LLM_PROVIDER, get_api_keys
from tldw_Server_API.app.core.Audio.error_payloads import _maybe_debug_details
from tldw_Server_API.app.core.Audio.streaming_exceptions import QuotaExceeded
from tldw_Server_API.app.core.Audio.transcription_service import _map_openai_audio_model_to_whisper
from tldw_Server_API.app.core.Audio.quota_helpers import EXPECTED_DB_EXC, EXPECTED_REDIS_EXC, _get_failopen_cap_minutes
from tldw_Server_API.app.core.Usage.audio_quota import (
    active_streams_count,
    add_daily_minutes,
    bytes_to_seconds,
    can_start_stream,
    check_daily_minutes_allow,
    finish_job,
    finish_stream,
    get_daily_minutes_used,
    get_limits_for_user,
    get_user_tier,
    heartbeat_stream,
    increment_jobs_started,
)
from tldw_Server_API.app.core.Audio.streaming_service import (
    CHAT_HISTORY_MAX_MESSAGES,
    _audio_ws_authenticate,
    _stream_tts_to_websocket,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import resolve_byok_credentials
from tldw_Server_API.app.core.AuthNZ.settings import is_multi_user_mode
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async as chat_api_call_async
from tldw_Server_API.app.core.Chat.chat_helpers import (
    get_or_create_character_context,
    get_or_create_conversation,
)
from tldw_Server_API.app.core.config import load_comprehensive_config
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.legacy_transcripts import (
    upsert_transcript,
)
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
    ensure_app_config,
    normalize_provider,
    resolve_provider_api_key_from_config,
)
from tldw_Server_API.app.core.Logging.log_context import ensure_request_id, get_ps_logger
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry, increment_counter
from tldw_Server_API.app.core.Metrics.stt_metrics import (
    emit_stt_error_total,
    emit_stt_redaction_total,
    emit_stt_request_total,
    emit_stt_run_write_total,
    emit_stt_session_end_total,
    emit_stt_session_start_total,
    observe_stt_final_latency_seconds,
)
from tldw_Server_API.app.core.Streaming.phrase_chunker import PhraseChunker
from tldw_Server_API.app.core.Streaming import speech_chat_service
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.app.core.TTS.realtime_session import RealtimeSessionConfig
from tldw_Server_API.app.core.TTS.tts_request_resolution import (
    resolve_tts_request_defaults,
)
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.model_utils import normalize_model_and_variant
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_policy import (
    RedactingWebSocketProxy,
    apply_transcript_payload_policy,
    apply_transcript_text_policy,
    get_websocket_auth_principal,
    resolve_effective_stt_policy,
)
from tldw_Server_API.app.services.app_lifecycle import assert_may_start_work

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import (
        UnifiedStreamingConfig as _UnifiedStreamingConfig,
    )


_AUDIO_STREAMING_UNIFIED_MODULE = (
    "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified"
)


def _load_audio_streaming_unified_module():
    return importlib.import_module(_AUDIO_STREAMING_UNIFIED_MODULE)


def _load_audio_streaming_unified_attr(name: str) -> Any:
    return getattr(_load_audio_streaming_unified_module(), name)


def UnifiedStreamingTranscriber(*args, **kwargs):
    return _load_audio_streaming_unified_attr("UnifiedStreamingTranscriber")(*args, **kwargs)


def SileroTurnDetector(*args, **kwargs):
    return _load_audio_streaming_unified_attr("SileroTurnDetector")(*args, **kwargs)


def _new_unified_streaming_config(**kwargs):
    return _load_audio_streaming_unified_attr("UnifiedStreamingConfig")(**kwargs)


def _new_ws_control_session():
    config_factory = _load_audio_streaming_unified_attr("_get_ws_control_protocol_config")
    session_cls = _load_audio_streaming_unified_attr("WSControlSession")
    return session_cls(config_factory())


def _estimate_stream_audio_seconds(audio_bytes: bytes, sample_rate: int) -> float:
    estimator = _load_audio_streaming_unified_attr("_estimate_audio_seconds")
    return float(estimator(audio_bytes, sample_rate))


def _drop_oldest_stream_audio(
    paused_audio_chunks: "deque[tuple[bytes, float]]",
    dropped_seconds: float,
) -> None:
    dropper = _load_audio_streaming_unified_attr("_drop_oldest_buffered_audio")
    dropper(paused_audio_chunks, dropped_seconds)


def _build_transcript_diagnostics_payload(
    *,
    auto_commit: bool,
    vad_status: str,
    diarization_status: str,
    diarization_details: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    builder = _load_audio_streaming_unified_attr("_build_transcript_diagnostics")
    return builder(
        auto_commit=auto_commit,
        vad_status=vad_status,
        diarization_status=diarization_status,
        diarization_details=diarization_details,
    )


async def _handle_unified_websocket(*args, **kwargs):
    handler = _load_audio_streaming_unified_attr("handle_unified_websocket")
    return await handler(*args, **kwargs)


async def handle_unified_websocket(*args, **kwargs):
    """Compatibility shim kept for tests and local monkeypatch seams."""
    return await _handle_unified_websocket(*args, **kwargs)

_AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    IndexError,
    KeyError,
    LookupError,
    NameError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    configparser.Error,
    json.JSONDecodeError,
    HTTPException,
    WebSocketDisconnect,
    QuotaExceeded,
    *EXPECTED_DB_EXC,
    *EXPECTED_REDIS_EXC,
)

router = APIRouter(
    tags=["Audio"],
    responses={
        404: {"description": "Not found"},
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
    },
)

def _audio_shim_attr(name: str):
    defaults: dict[str, Any] = {
        "asyncio": asyncio,
        "_audio_ws_authenticate": _audio_ws_authenticate,
        "get_metrics_registry": get_metrics_registry,
        "get_api_keys": get_api_keys,
        "chat_api_call_async": chat_api_call_async,
        "UnifiedStreamingTranscriber": UnifiedStreamingTranscriber,
        "SileroTurnDetector": SileroTurnDetector,
        "get_tts_service": get_tts_service,
        "get_chacha_db_for_user_id": get_chacha_db_for_user_id,
        "get_or_create_character_context": get_or_create_character_context,
        "get_or_create_conversation": get_or_create_conversation,
        "can_start_stream": can_start_stream,
        "finish_stream": finish_stream,
        "check_daily_minutes_allow": check_daily_minutes_allow,
        "add_daily_minutes": add_daily_minutes,
        "bytes_to_seconds": bytes_to_seconds,
        "heartbeat_stream": heartbeat_stream,
        "active_streams_count": active_streams_count,
        "get_daily_minutes_used": get_daily_minutes_used,
        "get_user_tier": get_user_tier,
        "get_limits_for_user": get_limits_for_user,
        "increment_jobs_started": increment_jobs_started,
        "finish_job": finish_job,
    }
    default_value = defaults.get(name)
    package_candidate: Any = None
    module_candidate: Any = None

    def _is_test_override(value: Any) -> bool:
        try:
            module_name = str(getattr(value, "__module__", "") or "")
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
            module_name = ""
        if module_name.startswith("tldw_Server_API.tests") or module_name.startswith("tests."):
            return True
        if module_name == "__main__":
            return True
        try:
            class_module = str(getattr(getattr(value, "__class__", object), "__module__", "") or "")
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
            class_module = ""
        return class_module.startswith("tldw_Server_API.tests") or class_module.startswith("tests.")

    try:
        from tldw_Server_API.app.api.v1.endpoints import audio as audio_pkg_shim

        if hasattr(audio_pkg_shim, name):
            package_candidate = getattr(audio_pkg_shim, name)
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        logger.debug("audio_streaming shim package lookup failed for {}", name, exc_info=True)
    except Exception:
        logger.debug("audio_streaming shim package lookup raised unexpected error for {}", name, exc_info=True)
    try:
        from tldw_Server_API.app.api.v1.endpoints.audio import audio as audio_module_shim

        if hasattr(audio_module_shim, name):
            module_candidate = getattr(audio_module_shim, name)
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        logger.debug("audio_streaming shim module lookup failed for {}", name, exc_info=True)
    except Exception:
        logger.debug("audio_streaming shim module lookup raised unexpected error for {}", name, exc_info=True)

    if package_candidate is not None and module_candidate is not None:
        if package_candidate is module_candidate:
            return package_candidate
        package_is_test_override = _is_test_override(package_candidate)
        module_is_test_override = _is_test_override(module_candidate)
        if package_is_test_override != module_is_test_override:
            return package_candidate if package_is_test_override else module_candidate
        if default_value is not None:
            package_changed = package_candidate is not default_value
            module_changed = module_candidate is not default_value
            if package_changed != module_changed:
                return package_candidate if package_changed else module_candidate
            if package_changed and module_changed:
                # Prefer package-level shim when both are intentionally overridden.
                return package_candidate
        # Preserve historical behavior when no clear override signal is present.
        return module_candidate

    if package_candidate is not None:
        return package_candidate
    if module_candidate is not None:
        return module_candidate
    if name in defaults:
        return defaults[name]
    raise NameError(name)


def _shim_asyncio():
    try:
        return _audio_shim_attr("asyncio")
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        return asyncio


async def _shim_audio_ws_authenticate(*args, **kwargs):
    try:
        fn = _audio_shim_attr("_audio_ws_authenticate")
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        fn = _audio_ws_authenticate
    return await fn(*args, **kwargs)


def _shim_get_metrics_registry():
    try:
        fn = _audio_shim_attr("get_metrics_registry")
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        fn = get_metrics_registry
    return fn()


def _audio_ws_compat_error_type_enabled() -> bool:
    """
    Return True when Audio WS payloads should include the legacy `error_type`
    alias alongside canonical `code`.
    """
    raw = str(os.getenv("AUDIO_WS_COMPAT_ERROR_TYPE", "1")).strip().lower()
    return is_truthy(raw)


def _audio_ws_error_payload(
    *,
    code: str,
    message: str,
    request_id: Optional[str] = None,
    data: Optional[dict[str, Any]] = None,
    exc: Optional[Exception] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "error", "code": code, "message": message}
    if request_id:
        payload["request_id"] = request_id
    if data is not None:
        payload["data"] = data
    details = _maybe_debug_details(exc)
    if details:
        payload["details"] = details
    if extra:
        payload.update(extra)
    if _audio_ws_compat_error_type_enabled():
        payload["error_type"] = code
    return payload


def _audio_ws_quota_error_payload(
    *,
    quota: str,
    message: str,
    request_id: Optional[str] = None,
) -> dict[str, Any]:
    payload = _audio_ws_error_payload(
        code="quota_exceeded",
        message=message,
        request_id=request_id,
        data={"quota": quota},
    )
    if _audio_ws_compat_error_type_enabled():
        payload["quota"] = quota
    return payload


async def _guard_audio_ws_work_start(
    websocket: WebSocket,
    *,
    kind: str,
    outer_stream: Any = None,
    request_id: Optional[str] = None,
) -> bool:
    app = getattr(websocket, "app", None)
    if app is None:
        return True
    try:
        assert_may_start_work(app, kind)
        return True
    except HTTPException:
        payload = _audio_ws_error_payload(
            code="service_unavailable",
            message="Shutdown in progress",
            request_id=request_id,
            data={"kind": kind},
        )
        try:
            if outer_stream:
                await outer_stream.send_json(payload)
            else:
                await websocket.send_json(payload)
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
            pass
        try:
            await websocket.close(code=1013, reason="shutdown_draining")
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
            pass
        return False


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    try:
        lowered = str(value).strip().lower()
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        return default
    if lowered in {"1", "true", "yes", "on", "enabled"}:
        return True
    if lowered in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _coerce_positive_float(raw: Any, default: float, *, min_value: float = 0.0) -> float:
    try:
        value = float(raw)
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        return default
    if value < min_value:
        return default
    return value


def _coerce_positive_int(raw: Any) -> Optional[int]:
    try:
        value = int(raw)
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        return None
    if value <= 0:
        return None
    return value


def _stream_model_label(config: "_UnifiedStreamingConfig") -> str:
    model = str(getattr(config, "model", "parakeet") or "parakeet").strip().lower()
    variant = str(getattr(config, "model_variant", "standard") or "standard").strip().lower()
    return f"stream:{model}:{variant}"


def _stt_redaction_outcome_for_text(
    *,
    original_text: str,
    redacted_text: str,
    policy: Any,
) -> str:
    if not getattr(policy, "redact_pii", False):
        return "not_requested"
    return "applied" if redacted_text != original_text else "skipped"


def _stt_redaction_outcome_for_payload(
    *,
    original_payload: dict[str, Any],
    redacted_payload: dict[str, Any],
    policy: Any,
) -> str:
    if not getattr(policy, "redact_pii", False):
        return "not_requested"
    return "applied" if redacted_payload != original_payload else "skipped"


def _resolve_default_streaming_model() -> tuple[str, str, str]:
    """
    Resolve server-side streaming defaults from STT config.

    Returns:
        tuple[str, str, str]: (model, variant, whisper_model_size)
    """
    default_model_id = "parakeet-onnx"
    default_variant = "standard"
    default_whisper_model_size = "distil-large-v3"

    try:
        cfg = load_comprehensive_config()
        if cfg is not None and cfg.has_section("STT-Settings"):
            configured_default = cfg.get(
                "STT-Settings",
                "default_streaming_transcription_model",
                fallback=default_model_id,
            )
            default_model_id = str(configured_default).strip() or default_model_id
            configured_variant = cfg.get(
                "STT-Settings",
                "nemo_model_variant",
                fallback=default_variant,
            )
            default_variant = str(configured_variant).strip().lower() or default_variant
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(f"Could not read streaming STT defaults from config: {exc}")

    model = "parakeet"
    variant = default_variant
    whisper_model_size = default_whisper_model_size
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import (
            get_stt_provider_registry,
        )

        registry = get_stt_provider_registry()
        provider, resolved_model_id, resolved_variant = registry.resolve_provider_for_model(default_model_id)
        provider_name = str(provider or "").strip().lower()

        if provider_name in {"faster-whisper", "whisper"}:
            model = "whisper"
            candidate_model = str(resolved_model_id or "").strip()
            if candidate_model:
                whisper_model_size = candidate_model
        elif provider_name in {"parakeet", "canary", "qwen3-asr"}:
            model = provider_name
        else:
            logger.warning(
                "Unsupported streaming default model '{}'; falling back to parakeet-onnx".format(default_model_id)
            )
            model = "parakeet"
            resolved_variant = "onnx"

        if model == "parakeet":
            candidate_variant = str(resolved_variant or variant or "standard").strip().lower()
            if candidate_variant not in {"standard", "onnx", "mlx", "cuda"}:
                candidate_variant = "standard"
            variant = candidate_variant
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(
            "Could not resolve configured streaming model '{}'; falling back to parakeet-onnx. Error: {}"
            .format(default_model_id, exc)
        )
        model = "parakeet"
        variant = "onnx"

    return model, variant, whisper_model_size


def _resolve_audio_chat_streaming_model(
    *,
    raw_model: Any,
    variant_override: Any,
    current_model: str,
    current_variant: str,
    current_whisper_model_size: str,
    explicit_whisper_model_size: Any = None,
) -> tuple[str, str, str]:
    """Normalize client STT identifiers into the canonical streaming selector fields."""

    model = current_model
    variant = current_variant
    whisper_model_size = current_whisper_model_size

    raw_model_str = str(raw_model or "").strip()
    explicit_whisper_size = str(explicit_whisper_model_size or "").strip()
    if not raw_model_str:
        normalized_model, normalized_variant = normalize_model_and_variant(
            None,
            current_model=current_model,
            current_variant=current_variant,
            variant_override=variant_override,
        )
        return normalized_model, normalized_variant, whisper_model_size

    provider_name = ""
    resolved_model_id = raw_model_str
    resolved_variant = None
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import (
            get_stt_provider_registry,
        )

        registry = get_stt_provider_registry()
        provider_name, resolved_model_id, resolved_variant = registry.resolve_provider_for_model(raw_model_str)
        provider_name = str(provider_name or "").strip().lower()
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"Could not resolve chat streaming STT provider for '{raw_model_str}': {exc}")

    if provider_name in {"faster-whisper", "whisper"}:
        resolved_whisper_model = explicit_whisper_size or str(resolved_model_id or raw_model_str).strip()
        return "whisper", variant, _map_openai_audio_model_to_whisper(resolved_whisper_model)

    if provider_name == "parakeet":
        normalized_model, normalized_variant = normalize_model_and_variant(
            raw_model_str,
            current_model="parakeet",
            current_variant=current_variant,
            variant_override=variant_override or resolved_variant,
        )
        return normalized_model, normalized_variant, whisper_model_size

    if provider_name in {"canary", "qwen3-asr"}:
        return provider_name, variant, whisper_model_size

    normalized_model, normalized_variant = normalize_model_and_variant(
        raw_model_str,
        current_model=current_model,
        current_variant=current_variant,
        variant_override=variant_override,
    )
    return normalized_model, normalized_variant, whisper_model_size


def _compose_transcript_snapshot(full_text: str, latest_text: str) -> str:
    full = str(full_text or "").strip()
    latest = str(latest_text or "").strip()
    if full and latest:
        if full.endswith(latest) or latest in full:
            return full
        return f"{full} {latest}".strip()
    return full or latest


def _shim_get_api_keys():
    try:
        fn = _audio_shim_attr("get_api_keys")
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        fn = get_api_keys
    return fn()


async def _shim_chat_api_call_async(**kwargs):
    try:
        fn = _audio_shim_attr("chat_api_call_async")
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        fn = chat_api_call_async
    return await fn(**kwargs)


def _shim_transcriber_cls():
    try:
        return _audio_shim_attr("UnifiedStreamingTranscriber")
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        return UnifiedStreamingTranscriber


def _shim_silero_cls():
    try:
        return _audio_shim_attr("SileroTurnDetector")
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        return SileroTurnDetector


async def _shim_get_tts_service():
    try:
        fn = _audio_shim_attr("get_tts_service")
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        fn = get_tts_service
    return await fn()


async def _shim_get_chacha_db_for_user_id(user_id: int, client_id: Optional[str] = None):
    try:
        fn = _audio_shim_attr("get_chacha_db_for_user_id")
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        fn = get_chacha_db_for_user_id
    return await fn(user_id, client_id=client_id)


async def _shim_get_or_create_character_context(db: CharactersRAGDB, character_id: Optional[str], loop):
    try:
        fn = _audio_shim_attr("get_or_create_character_context")
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        fn = get_or_create_character_context
    return await fn(db, character_id, loop)


async def _shim_get_or_create_conversation(
    db: CharactersRAGDB,
    conversation_id: Optional[str],
    character_id: int,
    character_name: str,
    client_id: str,
    loop,
):
    try:
        fn = _audio_shim_attr("get_or_create_conversation")
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        fn = get_or_create_conversation
    return await fn(db, conversation_id, character_id, character_name, client_id, loop)


async def _can_start_stream(user_id: int):
    return await _audio_shim_attr("can_start_stream")(user_id)


async def _finish_stream(user_id: int):
    return await _audio_shim_attr("finish_stream")(user_id)


async def _check_daily_minutes_allow(user_id: int, minutes: float):
    return await _audio_shim_attr("check_daily_minutes_allow")(user_id, minutes)


async def _add_daily_minutes(user_id: int, minutes: float):
    return await _audio_shim_attr("add_daily_minutes")(user_id, minutes)


def _bytes_to_seconds(size_bytes: int, sample_rate: int) -> float:
    return _audio_shim_attr("bytes_to_seconds")(size_bytes, sample_rate)


async def _heartbeat_stream(user_id: int):
    return await _audio_shim_attr("heartbeat_stream")(user_id)


async def _active_streams_count(user_id: int):
    return await _audio_shim_attr("active_streams_count")(user_id)


async def _get_daily_minutes_used(user_id: int):
    return await _audio_shim_attr("get_daily_minutes_used")(user_id)


async def _get_user_tier(user_id: int):
    return await _audio_shim_attr("get_user_tier")(user_id)


async def _get_limits_for_user(user_id: int):
    return await _audio_shim_attr("get_limits_for_user")(user_id)


async def _increment_jobs_started(user_id: int):
    return await _audio_shim_attr("increment_jobs_started")(user_id)


async def _finish_job(user_id: int):
    return await _audio_shim_attr("finish_job")(user_id)


@router.post(
    "/chat",
    response_model=SpeechChatResponse,
    summary="Non-streaming Speech-to-Speech chat (STT → LLM → TTS)",
    dependencies=[
        Depends(
            require_token_scope(
                "any",
                require_if_present=True,
                endpoint_id="audio.chat",
                count_as="call",
            )
        )
    ],
)
async def audio_chat_turn(
    request_data: SpeechChatRequest,
    request: Request,
    tts_service: TTSServiceV2 = Depends(get_tts_service),
    current_user: User = Depends(get_request_user),
    chat_db: CharactersRAGDB = Depends(get_chacha_db_for_user),
    usage_log: UsageEventLogger = Depends(get_usage_event_logger),
) -> SpeechChatResponse:
    """
    Execute a single non-streaming speech chat turn:

      - Accept base64-encoded user audio.
      - Run STT to obtain a transcript.
      - Call the LLM with recent conversation history.
      - Persist user/assistant messages into ChaChaNotes.
      - Run TTS on the assistant reply and return base64-encoded audio.

    This endpoint focuses on the v1 non-streaming path; streaming speech chat
    is handled by a separate WebSocket endpoint in v2.
    """
    rid = ensure_request_id(request)
    try:
        usage_log.log_event(
            "audio.chat",
            tags=[str(getattr(current_user, "id", ""))],
            metadata={
                "session_id": request_data.session_id or "",
                "input_audio_format": request_data.input_audio_format,
            },
        )
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as e:  # noqa: BLE001
        logger.debug(f"usage_log audio.chat failed: error={e}; request_id={rid}")

    acquired_stream = False
    user_id_for_usage = int(getattr(current_user, "id", 0) or 0)
    try:
        # Per-user concurrent chat guard (reuses audio stream limits)
        can_start, reason = await _can_start_stream(user_id_for_usage)
        if not can_start:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=reason or "Concurrent audio chat limit reached",
            )

        acquired_stream = True

        try:
            return await speech_chat_service.run_speech_chat_turn(
                request_data=request_data,
                request=request,
                current_user=current_user,
                chat_db=chat_db,
                tts_service=tts_service,
            )
        finally:
            if acquired_stream:
                try:
                    await _finish_stream(user_id_for_usage)
                except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as e:
                    logger.debug(f"_finish_stream failed (audio chat): {e}")
    except HTTPException:
        raise
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as e:  # noqa: BLE001
        logger.error(f"Speech chat turn failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Speech chat pipeline failed",
        ) from e

######################################################################################################################

# Create a separate router for WebSocket endpoints to avoid authentication conflicts
ws_router = APIRouter()


@ws_router.websocket("/stream/transcribe")
async def websocket_transcribe(
    websocket: WebSocket, token: Optional[str] = Query(None)  # Get token from query parameter
):
    """
    Handle a WebSocket connection to perform real-time streaming audio transcription.

    Accepts a WebSocket and an optional query token. Authentication is supported via:
    - Multi-user: X-API-KEY header, Authorization: Bearer <JWT>, `token` query parameter (API key or JWT), or an initial auth message.
    - Single-user: API key via header, `token` query parameter, or an initial auth message; an IP allowlist may be enforced.
    Supported incoming message types: "auth" (for token-based auth), "config" (streaming configuration), "audio" (base64-encoded audio chunks), and "commit" (finalize current utterance).
    Outgoing message types include partial updates ("partial"), interim/final transcriptions ("transcription"), the final transcript ("full_transcript"), and structured error frames ("error").
    Per-user limits are enforced (concurrent streams and daily minute quotas); when a quota is exceeded the server sends an "error" with `code="quota_exceeded"` and closes the connection with code 4003 (or 1008 when `AUDIO_WS_QUOTA_CLOSE_1008=1`). A compatibility alias `error_type` is included when `AUDIO_WS_COMPAT_ERROR_TYPE=1` (default).
    A server-side default streaming configuration is used if the client does not provide one before audio arrives.
    Parameters:
        websocket (WebSocket): The active WebSocket connection.
        token (Optional[str]): Optional API key or JWT token supplied via the query string for both multi-user and single-user authentication.
    """
    # Create a lightweight WebSocketStream for uniform metrics on outer error paths
    _outer_stream = None
    try:
        from tldw_Server_API.app.core.Streaming.streams import WebSocketStream as _WSStream

        _outer_stream = _WSStream(
            websocket,
            heartbeat_interval_s=0,
            idle_timeout_s=0,
            compat_error_type=_audio_ws_compat_error_type_enabled(),
            labels={"component": "audio", "endpoint": "audio_unified_ws"},
        )
        await _outer_stream.start()
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        _outer_stream = None
    if _outer_stream is None:
        class _BareStream:
            def __init__(self, ws: WebSocket):
                self.ws = ws

            async def start(self) -> None:
                try:
                    already_accepted = False
                    try:
                        state = getattr(self.ws, "application_state", None)
                        if state is not None and str(state).upper().endswith("CONNECTED"):
                            already_accepted = True
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
                        already_accepted = False
                    if hasattr(self.ws, "accept") and not already_accepted:
                        await self.ws.accept()
                except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
                    logger.debug(f"_BareStream.start failed: {exc}")

            async def send_json(self, payload: dict[str, Any]) -> None:
                try:
                    await self.ws.send_json(payload)
                except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
                    logger.debug(f"_BareStream.send_json failed: {exc}")

            async def error(self, code: str, message: str, *, data: Optional[dict[str, Any]] = None) -> None:
                p = _audio_ws_error_payload(code=code, message=message, data=data)
                await self.send_json(p)

            async def done(self) -> None:
                await self.send_json({"type": "done"})

            def mark_activity(self) -> None:
                return

            async def stop(self) -> None:
                return

        _outer_stream = _BareStream(websocket)

    # Correlate via request id (header or generated)
    try:
        _hdrs = websocket.headers or {}
        request_id = (
            _hdrs.get("x-request-id")
            or _hdrs.get("X-Request-Id")
            or (websocket.query_params.get("request_id") if hasattr(websocket, "query_params") else None)
            or str(uuid4())
        )
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        request_id = str(uuid4())
    try:
        logger.info(f"Audio WS connected: request_id={request_id}")
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"Audio WS connection logging failed: {exc}")

    # Ops toggle for standardized close code on quota/rate limits (4003 → 1008)
    import os as _os

    def _policy_close_code() -> int:
        flag = str(_os.getenv("AUDIO_WS_QUOTA_CLOSE_1008", "0")).strip().lower()
        return 1008 if is_truthy(flag) else 4003

    # Authenticate (shared helper; parity with other audio WS endpoints)
    auth_ok, jwt_user_id = await _shim_audio_ws_authenticate(
        websocket,
        _outer_stream,
        endpoint_id="audio.stream.transcribe",
        ws_path="/api/v1/audio/stream/transcribe",
    )
    if not auth_ok:
        return
    if not await _guard_audio_ws_work_start(
        websocket,
        kind="audio.stream.transcribe",
        outer_stream=_outer_stream,
        request_id=request_id,
    ):
        return

    # Billing: check transcription minutes quota before streaming begins
    _ws_billing_org_id: int | None = None
    if enforcement_enabled():
        try:
            _ws_principal = get_websocket_auth_principal(websocket)
            if _ws_principal is not None:
                _ws_billing_org_id = await resolve_org_id_for_principal(_ws_principal)
                if _ws_billing_org_id is not None:
                    _enforcer = get_billing_enforcer()
                    _result = await _enforcer.check_limit(
                        _ws_billing_org_id,
                        LimitCategory.TRANSCRIPTION_MINUTES_MONTH,
                        requested_units=1,
                    )
                    if _result.should_block:
                        await _outer_stream.send_json({
                            "type": "error",
                            "code": "billing_limit_exceeded",
                            "message": _result.message or "Transcription minutes quota exceeded. Please upgrade your plan.",
                            "category": "transcription_minutes_month",
                            "current": _result.current,
                            "limit": _result.limit,
                            "upgrade_url": "/billing/plans",
                        })
                        await websocket.close(code=_policy_close_code(), reason="billing_limit_exceeded")
                        return
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as _billing_err:
            logger.debug(f"WS billing pre-check failed (fail-open): {_billing_err}")

    stt_metrics_provider = "other"
    stt_metrics_model = "other"
    stt_request_status = "ok"
    stt_session_close_reason = "client_disconnect"
    stt_session_started = False

    try:
        # Default configuration prefers explicit streaming model selection from
        # [STT-Settings].default_streaming_transcription_model.
        default_model, default_variant, default_whisper_model_size = _resolve_default_streaming_model()

        config = _new_unified_streaming_config(
            model=default_model,
            model_variant=default_variant,
            sample_rate=16000,
            chunk_duration=2.0,
            overlap_duration=0.5,
            enable_partial=True,
            partial_interval=0.5,
            language="en",  # Default language for Canary
            whisper_model_size=default_whisper_model_size,
        )
        stt_metrics_provider = getattr(config, "model", stt_metrics_provider)
        stt_metrics_model = _stream_model_label(config)

        logger.info(
            f"WebSocket authenticated, calling handle_unified_websocket with default config: model={config.model}, variant={config.model_variant}"
        )

        # Enforce per-user streaming quotas and daily minutes during streaming
        # Resolve user id for quotas (JWT in multi-user; fixed id in single-user)
        if is_multi_user_mode() and jwt_user_id is not None:
            user_id_for_usage = int(jwt_user_id)
        else:
            from tldw_Server_API.app.core.AuthNZ.settings import get_settings as _get_settings

            _s = _get_settings()
            user_id_for_usage = getattr(_s, "SINGLE_USER_FIXED_ID", 1)

        effective_stt_policy = await resolve_effective_stt_policy(
            principal=get_websocket_auth_principal(websocket),
            user_id=int(user_id_for_usage) if user_id_for_usage is not None else None,
            db=None,
        )

        class _MetricRedactingWebSocketProxy(RedactingWebSocketProxy):
            async def send_json(self, payload: dict[str, Any]) -> None:
                redacted_payload = apply_transcript_payload_policy(payload, policy=effective_stt_policy)
                if str(redacted_payload.get("type", "")).strip().lower() in {
                    "partial",
                    "transcription",
                    "full_transcript",
                }:
                    emit_stt_redaction_total(
                        endpoint="audio.stream.transcribe",
                        redaction_outcome=_stt_redaction_outcome_for_payload(
                            original_payload=payload,
                            redacted_payload=redacted_payload,
                            policy=effective_stt_policy,
                        ),
                    )
                await self._websocket.send_json(redacted_payload)

        policy_websocket = _MetricRedactingWebSocketProxy(websocket, policy=effective_stt_policy)

        acquired_stream = False

        def _ensure_stt_session_started() -> None:
            nonlocal stt_session_started
            if stt_session_started:
                return
            emit_stt_session_start_total(provider=stt_metrics_provider)
            stt_session_started = True

        ok_stream, msg_stream = await _can_start_stream(user_id_for_usage)
        if not ok_stream:
            if _outer_stream:
                await _outer_stream.send_json({"type": "error", "message": msg_stream})
            await websocket.close()
            return
        acquired_stream = True

        query_params = getattr(websocket, "query_params", {}) or {}
        persistence_enabled = _coerce_bool(
            query_params.get("persist_transcript", query_params.get("persist")),
            default=_coerce_bool(os.getenv("AUDIO_STREAM_TRANSCRIBE_PERSISTENCE", "0"), default=False),
        )
        persistence_partial_enabled = _coerce_bool(
            query_params.get("persist_partial_transcript"),
            default=_coerce_bool(os.getenv("AUDIO_STREAM_TRANSCRIBE_PARTIAL_PERSISTENCE", "1"), default=True),
        )
        persistence_partial_interval_s = _coerce_positive_float(
            os.getenv("AUDIO_STREAM_TRANSCRIBE_PARTIAL_INTERVAL_S", "2.0"),
            default=2.0,
            min_value=0.25,
        )
        persistence_media_id = _coerce_positive_int(query_params.get("media_id"))
        persistence_model = str(
            query_params.get("transcription_model", "").strip()
            if hasattr(query_params, "get")
            else ""
        ) or _stream_model_label(config)
        persistence_db = None
        persistence_warning_sent = False
        last_partial_persist_ts = 0.0
        persistence_idempotency_key = f"audio-ws:{request_id}"

        async def _send_persistence_warning(message: str, details: Optional[str] = None) -> None:
            nonlocal persistence_warning_sent
            if persistence_warning_sent:
                return
            persistence_warning_sent = True
            warning_payload: dict[str, Any] = {
                "type": "warning",
                "warning_type": "transcript_persistence_unavailable",
                "message": message,
            }
            if request_id:
                warning_payload["request_id"] = request_id
            if details:
                warning_payload["details"] = details
            try:
                if _outer_stream:
                    await _outer_stream.send_json(warning_payload)
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as send_exc:
                logger.debug(f"Audio WS transcript persistence warning send failed: {send_exc}")

        async def _ensure_persistence_context() -> Optional[tuple[Any, int]]:
            nonlocal persistence_db
            nonlocal persistence_enabled
            if not persistence_enabled:
                return None
            if persistence_media_id is None:
                persistence_enabled = False
                await _send_persistence_warning("Transcript persistence requested but media_id is missing")
                return None
            if persistence_db is not None:
                return persistence_db, persistence_media_id
            try:
                persistence_user = User(
                    id=int(user_id_for_usage),
                    username=f"audio_stream_{user_id_for_usage}",
                    role="user",
                    is_active=True,
                    is_verified=True,
                )
                persistence_db = _resolve_media_db_for_user(persistence_user)
                return persistence_db, persistence_media_id
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(
                    "Audio WS transcript persistence context unavailable; continuing without persistence. "
                    f"user_id={user_id_for_usage}, error={exc}"
                )
                persistence_enabled = False
                await _send_persistence_warning(
                    "Transcript persistence unavailable; continuing without persistence",
                    str(exc),
                )
                return None

        async def _persist_transcript_snapshot(text: str, *, is_final: bool) -> None:
            nonlocal last_partial_persist_ts
            nonlocal persistence_enabled
            snapshot = str(text or "").strip()
            if not snapshot or not persistence_enabled:
                return
            original_snapshot = snapshot
            snapshot = apply_transcript_text_policy(
                snapshot,
                policy=effective_stt_policy,
                is_partial=not is_final,
            )
            emit_stt_redaction_total(
                endpoint="audio.stream.transcribe",
                redaction_outcome=_stt_redaction_outcome_for_text(
                    original_text=original_snapshot,
                    redacted_text=snapshot,
                    policy=effective_stt_policy,
                ),
            )
            now = time.time()
            if not is_final:
                if not persistence_partial_enabled:
                    return
                if (now - last_partial_persist_ts) < persistence_partial_interval_s:
                    return
            context = await _ensure_persistence_context()
            if context is None:
                return
            db_instance, media_id = context
            try:
                write_payload = upsert_transcript(
                    db_instance,
                    media_id=media_id,
                    transcription=snapshot,
                    whisper_model=persistence_model,
                    idempotency_key=persistence_idempotency_key,
                )
                emit_stt_run_write_total(
                    provider=persistence_model,
                    write_result=(write_payload or {}).get("write_result", "created"),
                )
                if not is_final:
                    last_partial_persist_ts = now
            except Exception as exc:  # noqa: BLE001 - persistence is fail-open
                emit_stt_run_write_total(
                    provider=persistence_model,
                    write_result="failed",
                )
                logger.warning(
                    "Audio WS transcript persistence write failed; continuing without persistence. "
                    f"media_id={media_id}, model={persistence_model}, error={exc}"
                )
                persistence_enabled = False
                await _send_persistence_warning(
                    "Failed to persist transcript; continuing stream",
                    str(exc),
                )

        async def _on_stream_config_resolved(
            config_payload: dict[str, Any],
            resolved_config: "_UnifiedStreamingConfig",
        ) -> None:
            nonlocal persistence_enabled
            nonlocal persistence_partial_enabled
            nonlocal persistence_media_id
            nonlocal persistence_model
            nonlocal stt_metrics_provider
            nonlocal stt_metrics_model
            metadata = config_payload.get("metadata") if isinstance(config_payload, dict) else None
            if not isinstance(metadata, dict):
                metadata = {}
            persist_hint = config_payload.get("persist_transcript") if isinstance(config_payload, dict) else None
            if persist_hint is None and metadata:
                persist_hint = metadata.get("persist_transcript", metadata.get("persist"))
            if persist_hint is not None:
                persistence_enabled = _coerce_bool(persist_hint, default=persistence_enabled)

            partial_hint = (
                config_payload.get("persist_partial_transcript") if isinstance(config_payload, dict) else None
            )
            if partial_hint is None and metadata:
                partial_hint = metadata.get("persist_partial_transcript")
            if partial_hint is not None:
                persistence_partial_enabled = _coerce_bool(partial_hint, default=persistence_partial_enabled)

            media_id_hint = config_payload.get("media_id") if isinstance(config_payload, dict) else None
            if media_id_hint is None and metadata:
                media_id_hint = metadata.get("media_id")
            media_id_from_cfg = _coerce_positive_int(media_id_hint)
            if media_id_from_cfg is not None:
                persistence_media_id = media_id_from_cfg

            model_hint = config_payload.get("transcription_model") if isinstance(config_payload, dict) else None
            if model_hint is None and metadata:
                model_hint = metadata.get("transcription_model")
            if isinstance(model_hint, str) and model_hint.strip():
                persistence_model = model_hint.strip()
            else:
                persistence_model = _stream_model_label(resolved_config)
            stt_metrics_provider = getattr(resolved_config, "model", stt_metrics_provider)
            stt_metrics_model = _stream_model_label(resolved_config)
            _ensure_stt_session_started()

        async def _on_transcript_result(result: dict[str, Any], full_transcript: str) -> None:
            if not persistence_enabled:
                return
            result_type = str(result.get("type", "")).strip().lower()
            if result_type not in {"partial", "transcription"}:
                return
            text = result.get("text")
            if not isinstance(text, str) or not text.strip():
                return
            is_final = bool(result.get("is_final"))
            snapshot = _compose_transcript_snapshot(full_transcript, text)
            await _persist_transcript_snapshot(snapshot, is_final=is_final)

        async def _on_full_transcript(text: str, _auto_commit: bool) -> None:
            await _persist_transcript_snapshot(text, is_final=True)

        # Resource Governor: acquire a 'streams' concurrency lease (policy resolved via route_map)
        # Track and enforce minutes chunk-by-chunk
        used_minutes = 0.0
        _billing_minutes_accumulator = 0.0  # fractional minutes pending billing flush
        # Bounded fail-open budget in minutes if DB is unavailable while streaming
        FAIL_OPEN_CAP_MINUTES = _get_failopen_cap_minutes()
        failopen_remaining = FAIL_OPEN_CAP_MINUTES

        # Local snapshot of remaining minutes for this connection; when None,
        # the next chunk will trigger a DB refresh.
        remaining_minutes_snapshot: Optional[float] = None

        # Use shared exception class so inner handler can bubble it up
        _QuotaExceeded = QuotaExceeded

        async def _on_audio_quota(seconds: float, _sr: int) -> None:
            """
            Handle a chunk of audio for daily-minute quota accounting and enforcement.

            Parameters:
                seconds (float): Duration of the audio chunk in seconds.
                sr (int): Sample rate of the audio chunk in Hz (unused by this function but provided for callback compatibility).

            Raises:
                _QuotaExceeded: If adding this chunk would exceed the user's daily minutes quota.

            Notes:
                - Checks whether the user's remaining daily minutes allow this chunk; if allowed, increments the nonlocal
                  `used_minutes` counter and records the minutes via `_add_daily_minutes`.
            """
            nonlocal used_minutes, failopen_remaining, remaining_minutes_snapshot
            minutes_chunk = float(seconds) / 60.0
            deducted = False
            allow = False
            # Fast-path local check: if we have a remaining snapshot and this
            # chunk would exceed it, raise immediately without a DB round-trip.
            if remaining_minutes_snapshot is not None and minutes_chunk > remaining_minutes_snapshot:
                raise _QuotaExceeded("daily_minutes")

            # Refresh remaining snapshot periodically by asking the DB for this
            # chunk; on success, we treat the returned "remaining_after" value
            # as the new snapshot.
            try:
                allow, remaining_after = await _check_daily_minutes_allow(user_id_for_usage, minutes_chunk)
                if allow and remaining_after is not None:
                    remaining_minutes_snapshot = float(remaining_after)
            except EXPECTED_DB_EXC as e:
                # Backing store failed; allow temporarily but deduct from bounded fail-open budget
                logger.warning(
                    f"_check_daily_minutes_allow failed during streaming; temporarily allowing (bounded fail-open). user_id={user_id_for_usage}, error={e}"
                )
                allow = True
                failopen_remaining -= minutes_chunk
                try:
                    increment_counter(
                        "audio_failopen_minutes_total", value=float(minutes_chunk), labels={"reason": "db_check"}
                    )
                    increment_counter("audio_failopen_events_total", labels={"reason": "db_check"})
                except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as m_err:
                    logger.debug(f"metrics increment failed (audio_failopen_db_check): error={m_err}")
                deducted = True
                if failopen_remaining <= 0:
                    try:
                        increment_counter("audio_failopen_cap_exhausted_total", labels={"reason": "db_check"})
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as m_err:
                        logger.debug(f"metrics increment failed (audio_failopen_cap_db_check): error={m_err}")
                    raise _QuotaExceeded("daily_minutes") from None
            if not allow:
                # Raise structured signal to outer scope
                raise _QuotaExceeded("daily_minutes")
            used_minutes += minutes_chunk
            # Reduce the local snapshot so subsequent chunks can be checked
            # without hitting the DB until it is exhausted or refreshed on
            # the next successful check.
            if remaining_minutes_snapshot is not None:
                remaining_minutes_snapshot = max(0.0, remaining_minutes_snapshot - minutes_chunk)
            try:
                await _add_daily_minutes(user_id_for_usage, minutes_chunk)
            except EXPECTED_DB_EXC as e:
                # Could not record; continue streaming under bounded fail-open
                logger.warning(
                    f"Failed to record streaming minutes (bounded fail-open). user_id={user_id_for_usage}, error={e}"
                )
                if not deducted:
                    failopen_remaining -= minutes_chunk
                    try:
                        increment_counter(
                            "audio_failopen_minutes_total",
                            value=float(minutes_chunk),
                            labels={"reason": "db_record"},
                        )
                        increment_counter("audio_failopen_events_total", labels={"reason": "db_record"})
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as m_err:
                        logger.debug(f"metrics increment failed (audio_failopen_db_record): error={m_err}")
                    if failopen_remaining <= 0:
                        try:
                            increment_counter("audio_failopen_cap_exhausted_total", labels={"reason": "db_record"})
                        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as m_err:
                            logger.debug(f"metrics increment failed (audio_failopen_cap_db_record): error={m_err}")
                        raise _QuotaExceeded("daily_minutes") from None
            # Billing: accumulate fractional minutes; flush to cache when >= 1
            nonlocal _billing_minutes_accumulator
            if _ws_billing_org_id is not None:
                _billing_minutes_accumulator += minutes_chunk
                if _billing_minutes_accumulator >= 1.0:
                    _flush = int(_billing_minutes_accumulator)
                    _billing_minutes_accumulator -= _flush
                    try:
                        get_billing_enforcer().apply_usage_delta(
                            _ws_billing_org_id,
                            LimitCategory.TRANSCRIPTION_MINUTES_MONTH,
                            _flush,
                        )
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
                        pass  # fail-open
                    # Durable cost-units ledger write
                    try:
                        await cost_units.record_cost_units_for_entity(
                            entity_scope="org",
                            entity_value=str(_ws_billing_org_id),
                            minutes=float(_flush),
                        )
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
                        pass  # fail-open

        async def _on_heartbeat() -> None:
            """
            Send a heartbeat to update streaming quota/timestamp for the current user.

            Invokes the module-level `_heartbeat_stream` callback with
            `user_id_for_usage` to record activity; any Redis-related
            exceptions are logged and suppressed.
            """
            try:
                await _heartbeat_stream(user_id_for_usage)
            except EXPECTED_REDIS_EXC as _hb_e:
                logger.debug(f"Heartbeat failed for user_id={user_id_for_usage}: {_hb_e}")

        try:
            await handle_unified_websocket(
                policy_websocket,
                config,
                on_audio_seconds=_on_audio_quota,
                on_heartbeat=_on_heartbeat,
                on_stream_config_resolved=_on_stream_config_resolved,
                on_transcript_result=_on_transcript_result,
                on_full_transcript=_on_full_transcript,
            )
            stt_session_close_reason = "client_stop"
        except _QuotaExceeded as qe:
            stt_request_status = "quota_exceeded"
            stt_session_close_reason = "error"
            emit_stt_error_total(
                endpoint="audio.stream.transcribe",
                provider=stt_metrics_provider,
                reason="quota",
            )
            try:
                if _outer_stream:
                    await _outer_stream.send_json(
                        _audio_ws_quota_error_payload(
                            quota=qe.quota,
                            message="Streaming transcription quota exceeded (daily minutes)",
                            request_id=request_id,
                        )
                    )
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as send_exc:
                logger.debug(f"WebSocket send_json quota error failed: error={send_exc}")
            try:
                await websocket.close(code=_policy_close_code(), reason="quota_exceeded")
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as close_exc:
                logger.debug(f"WebSocket close (quota case) failed: error={close_exc}")
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
            stt_request_status = "internal_error"
            stt_session_close_reason = "error"
            emit_stt_error_total(
                endpoint="audio.stream.transcribe",
                provider=stt_metrics_provider,
                reason="internal",
            )
            raise
        finally:
            if acquired_stream and not stt_session_started:
                _ensure_stt_session_started()
            if stt_session_started:
                emit_stt_session_end_total(
                    provider=stt_metrics_provider,
                    session_close_reason=stt_session_close_reason,
                )
                emit_stt_request_total(
                    endpoint="audio.stream.transcribe",
                    provider=stt_metrics_provider,
                    model=stt_metrics_model,
                    status=stt_request_status,
                )
            if acquired_stream:
                try:
                    await _finish_stream(user_id_for_usage)
                except EXPECTED_DB_EXC as e:
                    logger.debug(
                        f"Failed to release streaming quota slot (stream/transcribe): "
                        f"user_id={user_id_for_usage}, error={e}"
                    )
            if persistence_db is not None:
                with contextlib.suppress(_AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS):
                    if hasattr(persistence_db, "release_context_connection"):
                        persistence_db.release_context_connection()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as e:
        if stt_session_started and stt_request_status == "ok":
            stt_request_status = "internal_error"
            stt_session_close_reason = "error"
            emit_stt_error_total(
                endpoint="audio.stream.transcribe",
                provider=stt_metrics_provider,
                reason="internal",
            )
        logger.error(f"WebSocket error: {e}")
        # Best-effort: map quota exception variants to structured error
        try:
            quota_name = getattr(e, "quota", None)
            txt = str(e)
            if not quota_name and ("daily_minutes" in txt or "concurrent_streams" in txt):
                quota_name = "daily_minutes" if "daily_minutes" in txt else "concurrent_streams"
            if quota_name:
                try:
                    if _outer_stream:
                        await _outer_stream.send_json(
                            _audio_ws_quota_error_payload(
                                quota=quota_name,
                                message="Streaming transcription quota exceeded",
                                request_id=request_id,
                            )
                        )
                finally:
                    try:
                        await websocket.close(code=_policy_close_code(), reason="quota_exceeded")
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as e:
                        logger.warning(f"WebSocket close after quota exceeded failed: error={e}")
                        try:
                            increment_counter(
                                "app_warning_events_total",
                                labels={"component": "audio", "event": "ws_close_quota_failed"},
                            )
                        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as m_err:
                            logger.debug(f"metrics increment failed (audio ws_close_quota_failed): error={m_err}")
            else:
                # Let inner handler's error payload (if any) be the authoritative one.
                # Avoid sending a duplicate generic error frame that could race the client.
                pass
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"Streaming transcription outer handler swallowed error: {e}")
            try:
                increment_counter(
                    "app_warning_events_total", labels={"component": "audio", "event": "stream_outer_handler_error"}
                )
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as m_err:
                logger.debug(f"metrics increment failed (audio stream_outer_handler_error): error={m_err}")
    finally:
        # Billing: flush any remaining fractional minutes before closing
        if _ws_billing_org_id is not None and _billing_minutes_accumulator >= 0.5:
            _final_flush = max(1, int(_billing_minutes_accumulator + 0.5))
            try:
                get_billing_enforcer().apply_usage_delta(
                    _ws_billing_org_id,
                    LimitCategory.TRANSCRIPTION_MINUTES_MONTH,
                    _final_flush,
                )
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
                pass
            # Durable cost-units ledger write
            try:
                await cost_units.record_cost_units_for_entity(
                    entity_scope="org",
                    entity_value=str(_ws_billing_org_id),
                    minutes=float(_final_flush),
                )
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
                pass
        try:
            await websocket.close()
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"WebSocket close failed: error={e}")
            try:
                increment_counter("app_warning_events_total", labels={"component": "audio", "event": "ws_close_failed"})
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as m_err:
                logger.debug(f"metrics increment failed (audio ws_close_failed): error={m_err}")


@ws_router.websocket("/chat/stream")
async def websocket_audio_chat_stream(
    websocket: WebSocket,
    token: Optional[str] = Query(None),  # noqa: ARG001 - kept for parity with other WS endpoints
):
    """
    WebSocket v2 speech chat:
      - Partial STT with VAD auto-commit
      - Streaming LLM deltas
      - Streaming TTS audio back (binary frames)

    Authentication:
      - Multi-user: `X-API-KEY` header, `Authorization: Bearer <JWT>`, `token` query parameter (API key or JWT),
        or an initial auth message frame (JWT).
      - Single-user: API key via header, `token` query parameter, or an initial auth message; optional IP allowlist.
    """
    await websocket.accept()

    try:
        _raw_idle = os.getenv("AUDIO_WS_IDLE_TIMEOUT_S") or os.getenv("STREAM_IDLE_TIMEOUT_S")
        _idle_timeout = float(_raw_idle) if _raw_idle else None
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        _idle_timeout = None

    aio = _shim_asyncio()
    wait_for = getattr(aio, "wait_for", asyncio.wait_for)
    iscoroutine = getattr(aio, "iscoroutine", asyncio.iscoroutine)
    create_task = getattr(aio, "create_task", asyncio.create_task)

    # Wrap websocket for consistent metrics/heartbeats; keep connection open across turns
    _outer_stream = None
    try:
        from tldw_Server_API.app.core.Streaming.streams import WebSocketStream as _WSStream

        _outer_stream = _WSStream(
            websocket,
            heartbeat_interval_s=None,
            compat_error_type=_audio_ws_compat_error_type_enabled(),
            close_on_done=False,
            idle_timeout_s=_idle_timeout,
            labels={"component": "audio", "endpoint": "audio_chat_ws"},
        )
        await _outer_stream.start()
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        _outer_stream = None

    try:
        _hdrs = websocket.headers or {}
        request_id = (
            _hdrs.get("x-request-id")
            or _hdrs.get("X-Request-Id")
            or (websocket.query_params.get("request_id") if hasattr(websocket, "query_params") else None)
            or str(uuid4())
        )
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        request_id = str(uuid4())
    try:
        logger.info(f"Audio chat WS connected: request_id={request_id}")
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"Audio chat WS connection logging failed: {exc}")

    reg = _shim_get_metrics_registry()
    stt_metrics_provider = "other"
    stt_metrics_model = "other"
    stt_request_status = "ok"
    stt_session_close_reason = "client_disconnect"
    stt_session_started = False

    def _policy_close_code() -> int:
        flag = str(os.getenv("AUDIO_WS_QUOTA_CLOSE_1008", "0")).strip().lower()
        return 1008 if is_truthy(flag) else 4003

    # Authenticate (parity with STT/WS TTS)
    auth_ok, jwt_user_id = await _shim_audio_ws_authenticate(
        websocket,
        _outer_stream,
        endpoint_id="audio.chat.stream",
        ws_path="/api/v1/audio/chat/stream",
    )
    if not auth_ok:
        return
    if not await _guard_audio_ws_work_start(
        websocket,
        kind="audio.chat.stream",
        outer_stream=_outer_stream,
        request_id=request_id,
    ):
        return

    # Determine quota user id
    if is_multi_user_mode() and jwt_user_id is not None:
        user_id_for_usage = int(jwt_user_id)
    else:
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings as _get_settings

        _s = _get_settings()
        user_id_for_usage = getattr(_s, "SINGLE_USER_FIXED_ID", 1)

    effective_stt_policy = await resolve_effective_stt_policy(
        principal=get_websocket_auth_principal(websocket),
        user_id=int(user_id_for_usage) if user_id_for_usage is not None else None,
        db=None,
    )

    acquired_stream = False

    try:
        # Concurrency guard
        try:
            ok_stream, msg_stream = await _can_start_stream(user_id_for_usage)
            if not ok_stream:
                if _outer_stream:
                    await _outer_stream.send_json(
                        _audio_ws_error_payload(
                            code="rate_limited",
                            message=msg_stream or "Concurrent audio streams limit reached",
                            request_id=request_id,
                        )
                    )
                await websocket.close(code=_policy_close_code())
                return
            acquired_stream = True
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
            if _outer_stream:
                await _outer_stream.send_json(
                    _audio_ws_error_payload(
                        code="quota",
                        message="Unable to evaluate audio stream quota or concurrency",
                        request_id=request_id,
                    )
                )
            await websocket.close(code=_policy_close_code())
            return

        # Daily minute tracking (bounded fail-open like STT WS)
        used_minutes = 0.0
        FAIL_OPEN_CAP_MINUTES = _get_failopen_cap_minutes()
        failopen_remaining = FAIL_OPEN_CAP_MINUTES
        remaining_minutes_snapshot: Optional[float] = None

        async def _on_audio_quota(seconds: float, _sr: int) -> None:
            nonlocal used_minutes, failopen_remaining, remaining_minutes_snapshot
            minutes_chunk = float(seconds) / 60.0
            deducted = False
            allow = False

            if remaining_minutes_snapshot is not None and minutes_chunk > remaining_minutes_snapshot:
                raise QuotaExceeded("daily_minutes")

            try:
                allow, remaining_after = await _check_daily_minutes_allow(user_id_for_usage, minutes_chunk)
                if allow and remaining_after is not None:
                    remaining_minutes_snapshot = float(remaining_after)
            except EXPECTED_DB_EXC as e:
                logger.warning(
                    f"_check_daily_minutes_allow failed during streaming; temporarily allowing "
                    f"(bounded fail-open). user_id={user_id_for_usage}, error={e}"
                )
                allow = True
                failopen_remaining -= minutes_chunk
                try:
                    increment_counter(
                        "audio_failopen_minutes_total", value=float(minutes_chunk), labels={"reason": "db_check"}
                    )
                    increment_counter("audio_failopen_events_total", labels={"reason": "db_check"})
                except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as m_err:  # noqa: BLE001
                    logger.debug(f"metrics increment failed (audio_chat_failopen_db_check): error={m_err}")
                deducted = True
                if failopen_remaining <= 0:
                    try:
                        increment_counter("audio_failopen_cap_exhausted_total", labels={"reason": "db_check"})
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as m_err:  # noqa: BLE001
                        logger.debug(
                            f"metrics increment failed (audio_chat_failopen_cap_db_check): error={m_err}"
                        )
                    raise QuotaExceeded("daily_minutes") from None

            if not allow:
                raise QuotaExceeded("daily_minutes") from None

            used_minutes += minutes_chunk
            if remaining_minutes_snapshot is not None:
                remaining_minutes_snapshot = max(0.0, remaining_minutes_snapshot - minutes_chunk)
            try:
                await _add_daily_minutes(user_id_for_usage, minutes_chunk)
            except EXPECTED_DB_EXC as e:
                logger.warning(
                    f"Failed to record streaming minutes (bounded fail-open). user_id={user_id_for_usage}, error={e}"
                )
                if not deducted:
                    failopen_remaining -= minutes_chunk
                    try:
                        increment_counter(
                            "audio_failopen_minutes_total",
                            value=float(minutes_chunk),
                            labels={"reason": "db_record"},
                        )
                        increment_counter("audio_failopen_events_total", labels={"reason": "db_record"})
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as m_err:
                        logger.debug(f"metrics increment failed (audio_chat_failopen_db_record): error={m_err}")
                    if failopen_remaining <= 0:
                        try:
                            increment_counter("audio_failopen_cap_exhausted_total", labels={"reason": "db_record"})
                        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as m_err:
                            logger.debug(
                                f"metrics increment failed (audio_chat_failopen_cap_db_record): error={m_err}"
                            )
                        raise QuotaExceeded("daily_minutes") from None

        async def _on_heartbeat() -> None:
            try:
                await _heartbeat_stream(user_id_for_usage)  # type: ignore[arg-type]
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as _hb_e:
                logger.debug(f"Heartbeat failed for user_id={user_id_for_usage}: {_hb_e}")

        control_session = _new_ws_control_session()
        paused_audio_chunks: deque[tuple[bytes, float]] = deque()

        # Parse initial config
        try:
            raw_cfg = await wait_for(websocket.receive_text(), timeout=15.0)
            cfg_data = json.loads(raw_cfg)
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
            stt_request_status = "bad_request"
            stt_session_close_reason = "error"
            emit_stt_error_total(
                endpoint="audio.chat.stream",
                provider=stt_metrics_provider,
                reason="validation_error",
            )
            if _outer_stream:
                await _outer_stream.send_json(
                    _audio_ws_error_payload(
                        code="bad_request",
                        message="config frame required",
                        request_id=request_id,
                        exc=exc,
                    )
                )
            await websocket.close(code=4400)
            return

        if cfg_data.get("type") != "config":
            stt_request_status = "bad_request"
            stt_session_close_reason = "error"
            emit_stt_error_total(
                endpoint="audio.chat.stream",
                provider=stt_metrics_provider,
                reason="validation_error",
            )
            if _outer_stream:
                await _outer_stream.send_json(
                    {"type": "error", "message": "First frame must be type=config"}
                )
            await websocket.close(code=4400)
            return

        stt_cfg = cfg_data.get("stt") or cfg_data
        llm_cfg = cfg_data.get("llm") or {}
        tts_cfg = cfg_data.get("tts") or {}
        raw_session_id = cfg_data.get("session_id")
        session_id = str(raw_session_id).strip() if raw_session_id else None
        metadata = cfg_data.get("metadata") if isinstance(cfg_data.get("metadata"), dict) else None
        protocol_decision = control_session.apply_config(cfg_data)

        def _coerce_bool(value: Any, default: bool = False) -> bool:
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                return is_truthy(value)
            return default

        persist_hint = cfg_data.get("persist_history")
        if persist_hint is None and metadata:
            persist_hint = (
                metadata.get("persist_history")
                if "persist_history" in metadata
                else metadata.get("persist_session")
            )
        persist_default = _coerce_bool(os.getenv("AUDIO_CHAT_WS_PERSISTENCE", "0"), default=False)
        persistence_enabled = _coerce_bool(persist_hint, default=persist_default)
        persistence_db: Optional[CharactersRAGDB] = None
        persistence_session_id: Optional[str] = session_id
        persistence_ready = False
        persistence_warning_sent = False
        persistence_announced = False

        config = _new_unified_streaming_config()
        try:
            variant_override = stt_cfg.get("variant") or stt_cfg.get("model_variant")
            (
                config.model,
                config.model_variant,
                config.whisper_model_size,
            ) = _resolve_audio_chat_streaming_model(
                raw_model=stt_cfg.get("model"),
                variant_override=variant_override,
                current_model=config.model,
                current_variant=config.model_variant,
                current_whisper_model_size=config.whisper_model_size,
                explicit_whisper_model_size=stt_cfg.get("whisper_model_size"),
            )
            config.sample_rate = stt_cfg.get("sample_rate", config.sample_rate)
            config.enable_partial = stt_cfg.get("enable_partial", config.enable_partial)
            config.enable_vad = bool(stt_cfg.get("enable_vad", config.enable_vad))
            config.vad_threshold = float(stt_cfg.get("vad_threshold", config.vad_threshold))
            config.vad_min_silence_ms = int(stt_cfg.get("min_silence_ms", config.vad_min_silence_ms))
            config.vad_turn_stop_secs = float(stt_cfg.get("turn_stop_secs", config.vad_turn_stop_secs))
            if "min_utterance_secs" in stt_cfg:
                config.vad_min_utterance_secs = float(stt_cfg.get("min_utterance_secs"))
            if "min_partial_duration" in stt_cfg:
                try:
                    config.min_partial_duration = max(0.0, float(stt_cfg.get("min_partial_duration")))
                except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
                    logger.debug(f"Invalid min_partial_duration value in audio chat config: {exc}")
            if "language" in stt_cfg:
                config.language = stt_cfg.get("language")
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as cfg_exc:
            logger.debug(f"Failed to parse streaming STT config: {cfg_exc}")
        stt_metrics_provider = getattr(config, "model", stt_metrics_provider)
        stt_metrics_model = _stream_model_label(config)
        emit_stt_session_start_total(provider=stt_metrics_provider)
        stt_session_started = True

        llm_provider = (llm_cfg.get("provider") or llm_cfg.get("api_provider") or DEFAULT_LLM_PROVIDER).lower()
        llm_model = llm_cfg.get("model") or os.getenv("AUDIO_CHAT_DEFAULT_LLM_MODEL") or "gpt-3.5-turbo"
        llm_temperature = llm_cfg.get("temperature")
        llm_max_tokens = llm_cfg.get("max_tokens")
        llm_system_prompt = llm_cfg.get("system_prompt") or llm_cfg.get("system")
        llm_extra_params = llm_cfg.get("extra_params") if isinstance(llm_cfg.get("extra_params"), dict) else None

        try:
            tts_speed_raw = tts_cfg.get("speed", 1.0)
            tts_speed = float(tts_speed_raw)
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
            tts_speed = 1.0
        response_format = tts_cfg.get("format") or tts_cfg.get("response_format") or "pcm"
        resolved_tts = resolve_tts_request_defaults(
            provider=tts_cfg.get("provider"),
            model=tts_cfg.get("model"),
            voice=tts_cfg.get("voice"),
        )
        tts_model = resolved_tts.model
        tts_voice = resolved_tts.voice
        tts_provider = resolved_tts.provider
        tts_extra_params = tts_cfg.get("extra_params") if isinstance(tts_cfg.get("extra_params"), dict) else None

        # Initialize STT transcriber + VAD gate
        try:
            TranscriberCls = _shim_transcriber_cls()
            transcriber = TranscriberCls(config)
            transcriber.initialize()
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
            stt_request_status = "model_unavailable"
            stt_session_close_reason = "error"
            emit_stt_error_total(
                endpoint="audio.chat.stream",
                provider=stt_metrics_provider,
                reason="model_unavailable",
            )
            logger.error(f"Streaming transcriber init failed: {exc}", exc_info=True)
            if _outer_stream:
                data_payload = {
                    "model": config.model,
                    "variant": getattr(config, "model_variant", None),
                    "request_id": request_id,
                }
                details = _maybe_debug_details(exc)
                if details:
                    data_payload["details"] = details
                await _outer_stream.error(
                    "model_unavailable",
                    "Failed to initialize streaming transcriber",
                    data=data_payload,
                )
            return

        turn_detector: Optional[Any] = None
        vad_warning_sent = False
        vad_status = "disabled"

        async def _send_vad_warning(message: str, details: Optional[str]) -> None:
            payload: dict[str, Any] = {
                "type": "warning",
                "warning_type": "vad_unavailable",
                "message": message,
                "details": details,
            }
            try:
                if _outer_stream:
                    await _outer_stream.send_json(payload)
                else:
                    await websocket.send_json(payload)
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as send_exc:
                logger.debug(f"audio.chat.stream VAD warning send failed: {send_exc}")

        if config.enable_vad:
            try:
                vad_status = "fail_open"
                SileroCls = _shim_silero_cls()
                turn_detector = SileroCls(
                    sample_rate=config.sample_rate,
                    enabled=True,
                    vad_threshold=config.vad_threshold,
                    min_silence_ms=config.vad_min_silence_ms,
                    turn_stop_secs=config.vad_turn_stop_secs,
                    min_utterance_secs=config.vad_min_utterance_secs,
                )
                if not turn_detector.available:
                    vad_warning_sent = True
                    await _send_vad_warning(
                        "Silero VAD unavailable; auto-commit disabled",
                        turn_detector.unavailable_reason,
                    )
                    turn_detector = None
                else:
                    vad_status = "enabled"
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as vad_exc:
                logger.debug(f"VAD init failed: {vad_exc}")
                turn_detector = None

        chat_history: list[dict[str, Any]] = []
        if session_id:
            try:
                chat_history.append({"role": "system", "content": f"session:{session_id}"})
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
                chat_history = []

        def _action_hint() -> Optional[str]:
            try:
                if metadata and isinstance(metadata, dict) and metadata.get("action"):
                    return str(metadata.get("action"))
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
                logger.debug(f"Failed to read action hint from metadata: {exc}")
            try:
                if llm_extra_params and isinstance(llm_extra_params, dict) and llm_extra_params.get("action"):
                    return str(llm_extra_params.get("action"))
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
                logger.debug(f"Failed to read action hint from llm_extra_params: {exc}")
            return None

        async def _send_persistence_warning(message: str, details: Optional[str] = None) -> None:
            nonlocal persistence_warning_sent
            if persistence_warning_sent:
                return
            persistence_warning_sent = True
            payload: dict[str, Any] = {
                "type": "warning",
                "warning_type": "persistence_unavailable",
                "message": message,
            }
            if details:
                payload["details"] = details
            try:
                if _outer_stream:
                    await _outer_stream.send_json(payload)
                else:
                    await websocket.send_json(payload)
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as send_exc:
                logger.debug(f"audio.chat.stream persistence warning send failed: {send_exc}")

        async def _ensure_persistence_context() -> Optional[tuple[CharactersRAGDB, str]]:
            nonlocal persistence_db
            nonlocal persistence_session_id
            nonlocal persistence_ready
            nonlocal persistence_enabled
            nonlocal session_id
            nonlocal persistence_announced
            if not persistence_enabled:
                return None
            if persistence_ready and persistence_db is not None and persistence_session_id:
                return persistence_db, persistence_session_id
            try:
                if persistence_db is None:
                    persistence_db = await _shim_get_chacha_db_for_user_id(
                        int(user_id_for_usage),
                        client_id=str(user_id_for_usage),
                    )
                loop = asyncio.get_running_loop()
                character_card, character_db_id = await _shim_get_or_create_character_context(
                    persistence_db,
                    None,
                    loop,
                )
                if not character_db_id:
                    raise ValueError("Unable to resolve character context for WS persistence")
                character_name = "Helpful AI Assistant"
                if isinstance(character_card, dict) and character_card.get("name"):
                    character_name = str(character_card.get("name"))

                persistence_session_id, _ = await _shim_get_or_create_conversation(
                    persistence_db,
                    persistence_session_id,
                    int(character_db_id),
                    character_name,
                    str(user_id_for_usage),
                    loop,
                )
                session_id = persistence_session_id

                if session_id and not any(
                    msg.get("role") == "system" and msg.get("content") == f"session:{session_id}"
                    for msg in chat_history
                ):
                    chat_history.insert(0, {"role": "system", "content": f"session:{session_id}"})

                action_hint = _action_hint()
                settings_payload = {
                    "audio_chat_ws": {
                        "session_id": persistence_session_id,
                        "action_hint": action_hint,
                        "metadata": metadata or {},
                    }
                }
                try:
                    await loop.run_in_executor(
                        None,
                        persistence_db.upsert_conversation_settings,
                        persistence_session_id,
                        settings_payload,
                    )
                except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as settings_exc:
                    logger.debug(f"audio.chat.stream conversation_settings upsert failed: {settings_exc}")

                if _outer_stream and not persistence_announced:
                    await _outer_stream.send_json({"type": "session", "session_id": persistence_session_id})
                    persistence_announced = True

                persistence_ready = True
                return persistence_db, persistence_session_id
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(f"audio.chat.stream persistence unavailable: {exc}")
                persistence_enabled = False
                await _send_persistence_warning(
                    "Session persistence unavailable; continuing without persistence",
                    _maybe_debug_details(exc),
                )
                return None

        async def _persist_turn(
            transcript_text: str,
            assistant_text: str,
            action_result: Optional[dict[str, Any]],
        ) -> None:
            context = await _ensure_persistence_context()
            if not context:
                return
            chat_db, conversation_id = context

            def _persist_sync() -> None:
                chat_db.add_message(
                    {
                        "conversation_id": conversation_id,
                        "sender": "user",
                        "content": transcript_text,
                        "client_id": str(user_id_for_usage),
                    }
                )
                chat_db.add_message(
                    {
                        "conversation_id": conversation_id,
                        "sender": "assistant",
                        "content": assistant_text,
                        "client_id": str(user_id_for_usage),
                    }
                )
                if action_result is not None:
                    try:
                        tool_content = json.dumps(action_result)
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
                        logger.warning(f"Failed to serialize action_result for WS persistence: {exc}")
                    else:
                        chat_db.add_message(
                            {
                                "conversation_id": conversation_id,
                                "sender": "tool",
                                "content": tool_content,
                                "client_id": str(user_id_for_usage),
                            }
                        )

            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, _persist_sync)
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(f"audio.chat.stream turn persistence failed: {exc}")
                await _send_persistence_warning(
                    "Failed to persist chat turn; continuing stream",
                    _maybe_debug_details(exc),
                )

        async def _maybe_run_action(transcript_text: str) -> Optional[dict[str, Any]]:
            action_name = _action_hint()
            if not action_name:
                return None
            try:
                enabled = getattr(speech_chat_service, "_actions_enabled", lambda: False)()
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
                enabled = False
            if not enabled:
                return None

            user_obj = SimpleNamespace(id=user_id_for_usage)
            try:
                return await speech_chat_service._execute_action(action_name, transcript_text, user_obj)
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(f"Streaming action execution failed: action={action_name}, error={exc}")
                payload = {
                    "action": action_name,
                    "status": "error",
                    "message": "Action execution failed",
                    "user_id": getattr(user_obj, "id", None),
                }
                details = _maybe_debug_details(exc)
                if details:
                    payload["details"] = details
                return payload

        processing_turn = False
        active_turn_id: Optional[str] = None
        active_turn_cancelled = False
        active_turn_task: Optional[asyncio.Task] = None
        active_tts_sender_task: Optional[asyncio.Task] = None
        turn_sequence = 0

        async def _send_stream_payload(payload: dict[str, Any]) -> None:
            redacted_payload = apply_transcript_payload_policy(payload, policy=effective_stt_policy)
            if str(redacted_payload.get("type", "")).strip().lower() in {"partial", "transcription", "full_transcript"}:
                emit_stt_redaction_total(
                    endpoint="audio.chat.stream",
                    redaction_outcome=_stt_redaction_outcome_for_payload(
                        original_payload=payload,
                        redacted_payload=redacted_payload,
                        policy=effective_stt_policy,
                    ),
                )
            if _outer_stream:
                await _outer_stream.send_json(redacted_payload)
            else:
                await websocket.send_json(redacted_payload)

        async def _iter_stream_lines(stream_obj):
            if hasattr(stream_obj, "__aiter__"):
                async for line in stream_obj:
                    yield line
            else:
                for line in stream_obj:
                    yield line

        async def _stream_llm(
            transcript_text: str,
            on_delta: Optional[Any] = None,
            turn_id: Optional[str] = None,
        ) -> tuple[str, Optional[str], Optional[dict[str, Any]]]:
            nonlocal chat_history
            nonlocal active_turn_cancelled
            nonlocal active_turn_id
            def _fallback_resolver(name: str) -> Optional[str]:
                try:
                    return _shim_get_api_keys().get(name)
                except (KeyError, FileNotFoundError, OSError, ValueError, configparser.Error) as exc:
                    logger.debug(f"LLM fallback resolver failed for provider '{name}': {exc}")
                    return None

            user_id_int = int(user_id_for_usage) if user_id_for_usage else None
            byok_resolution = await resolve_byok_credentials(
                llm_provider,
                user_id=user_id_int,
                request=websocket,
                fallback_resolver=_fallback_resolver,
            )
            app_config = ensure_app_config(byok_resolution.app_config)
            provider_api_key = byok_resolution.api_key or resolve_provider_api_key_from_config(
                llm_provider,
                app_config,
            )
            if not provider_api_key:
                if _outer_stream:
                    await _outer_stream.send_json(
                        _audio_ws_error_payload(
                            code="missing_provider_credentials",
                            message="No API key available for provider",
                            request_id=request_id,
                            extra={"provider": llm_provider},
                        )
                    )
                return "", "missing_provider_credentials", None
            messages_payload = list(chat_history)
            messages_payload.append({"role": "user", "content": transcript_text})
            try:
                adapter = get_registry().get_adapter(normalize_provider(llm_provider))
                if adapter is None:
                    llm_stream = await _shim_chat_api_call_async(
                        api_endpoint=llm_provider,
                        messages_payload=messages_payload,
                        api_key=provider_api_key,
                        temp=llm_temperature,
                        model=llm_model,
                        max_tokens=llm_max_tokens,
                        streaming=True,
                        system_message=llm_system_prompt,
                        user_identifier=str(user_id_for_usage),
                        extra_body=llm_extra_params,
                        app_config=app_config,
                    )
                else:
                    request_payload = {
                        "messages": messages_payload,
                        "system_message": llm_system_prompt,
                        "model": llm_model,
                        "api_key": provider_api_key,
                        "temperature": llm_temperature,
                        "max_tokens": llm_max_tokens,
                        "stream": True,
                        "user": str(user_id_for_usage),
                        "app_config": app_config,
                    }
                    if llm_extra_params:
                        for key, value in llm_extra_params.items():
                            if request_payload.get(key) is None:
                                request_payload[key] = value
                    stream_candidate = adapter.astream(request_payload)
                    if iscoroutine(stream_candidate):
                        try:
                            llm_stream = await stream_candidate
                        except NotImplementedError:
                            llm_stream = await _shim_chat_api_call_async(
                                api_endpoint=llm_provider,
                                messages_payload=messages_payload,
                                api_key=provider_api_key,
                                temp=llm_temperature,
                                model=llm_model,
                                max_tokens=llm_max_tokens,
                                streaming=True,
                                system_message=llm_system_prompt,
                                user_identifier=str(user_id_for_usage),
                                extra_body=llm_extra_params,
                                app_config=app_config,
                            )
                    else:
                        llm_stream = stream_candidate
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
                logger.error(f"LLM stream failed: {exc}", exc_info=True)
                if _outer_stream:
                    await _outer_stream.send_json(
                        _audio_ws_error_payload(
                            code="llm_error",
                            message="LLM call failed",
                            request_id=request_id,
                            exc=exc,
                        )
                    )
                return "", None, None

            deltas: list[str] = []
            finish_reason: Optional[str] = None
            usage_payload: Optional[dict[str, Any]] = None

            async for raw_line in _iter_stream_lines(llm_stream):
                if turn_id is not None and (active_turn_cancelled or active_turn_id != turn_id):
                    break
                try:
                    line_str = (
                        raw_line.decode("utf-8", errors="ignore")
                        if isinstance(raw_line, (bytes, bytearray))
                        else str(raw_line)
                    )
                except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
                    continue
                if not line_str:
                    continue
                stripped = line_str.strip()
                if stripped.lower() == "data: [done]" or stripped.lower() == "[done]":
                    break
                if stripped.lower().endswith("[done]"):
                    break
                payload_str = stripped
                if payload_str.startswith("data:"):
                    payload_str = payload_str[len("data:") :].strip()
                try:
                    payload = json.loads(payload_str)
                except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
                    continue
                if "error" in payload:
                    if _outer_stream:
                        await _outer_stream.send_json(
                            _audio_ws_error_payload(
                                code="llm_error",
                                message=str(payload.get("error") or "LLM streaming error"),
                                request_id=request_id,
                            )
                        )
                    continue
                choices = payload.get("choices") or []
                for choice in choices:
                    if turn_id is not None and (active_turn_cancelled or active_turn_id != turn_id):
                        break
                    delta = choice.get("delta") or choice.get("message") or {}
                    content = delta.get("content") or delta.get("text") if isinstance(delta, dict) else None
                    if content:
                        deltas.append(content)
                        if _outer_stream:
                            await _outer_stream.send_json({"type": "llm_delta", "delta": content})
                        if on_delta is not None:
                            try:
                                maybe_result = on_delta(content)
                                if iscoroutine(maybe_result):
                                    await maybe_result
                            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as cb_exc:
                                logger.debug(f"audio.chat.stream on_delta callback failed: {cb_exc}")
                    if choice.get("finish_reason"):
                        finish_reason = choice.get("finish_reason")
                if payload.get("usage"):
                    usage_payload = payload.get("usage")

            assistant_text = "".join(deltas).strip()
            chat_history.append({"role": "user", "content": transcript_text})
            if assistant_text:
                chat_history.append({"role": "assistant", "content": assistant_text})
            if len(chat_history) > CHAT_HISTORY_MAX_MESSAGES:
                chat_history = chat_history[-CHAT_HISTORY_MAX_MESSAGES:]
            try:
                await byok_resolution.touch_last_used()
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Failed to update BYOK last_used timestamp for LLM: {exc}")
            return assistant_text, finish_reason, usage_payload

        async def _stream_tts(text: str, voice_to_voice_start: float) -> None:
            if not text:
                if _outer_stream:
                    await _outer_stream.send_json(
                        _audio_ws_error_payload(
                            code="empty_assistant",
                            message="Assistant reply empty",
                            request_id=request_id,
                        )
                    )
                return
            allowed_formats = {"mp3", "opus", "aac", "flac", "wav", "pcm"}
            if response_format not in allowed_formats:
                if _outer_stream:
                    await _outer_stream.send_json(
                        _audio_ws_error_payload(
                            code="bad_request",
                            message=f"Unsupported format '{response_format}'",
                            request_id=request_id,
                        )
                    )
                return

            speech_req = OpenAISpeechRequest(
                model=tts_model,
                input=text,
                voice=tts_voice,
                response_format=response_format,
                speed=tts_speed,
                stream=True,
                extra_params=tts_extra_params,
            )

            tts_service = await _shim_get_tts_service()

            if _outer_stream:
                try:
                    await _outer_stream.send_json(
                        {
                            "type": "tts_start",
                            "format": response_format,
                            "provider": tts_provider or "auto",
                            "voice": tts_voice,
                        }
                    )
                except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as send_exc:
                    logger.debug(f"audio.chat.stream tts_start send failed: error={send_exc}")

            try:
                async def _error_handler(exc: Exception) -> None:
                    if _outer_stream:
                        try:
                            logger.error(f"audio.chat.stream TTS generation failed: {exc}", exc_info=True)
                            await _outer_stream.send_json(
                                _audio_ws_error_payload(
                                    code="tts_error",
                                    message="TTS generation failed",
                                    request_id=request_id,
                                    exc=exc,
                                )
                            )
                        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as send_exc:
                            logger.debug(
                                f"audio.chat.stream producer error frame send failed: error={send_exc}"
                            )

                await _stream_tts_to_websocket(
                    websocket=websocket,
                    speech_req=speech_req,
                    tts_service=tts_service,
                    provider=tts_provider,
                    outer_stream=_outer_stream,
                    reg=reg,
                    route="audio.chat.stream",
                    component_label="audio_chat_ws",
                    voice_to_voice_start=voice_to_voice_start,
                    error_handler=_error_handler,
                    asyncio_module=aio,
                )
            finally:
                if _outer_stream:
                    try:
                        await _outer_stream.send_json({"type": "tts_done"})
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as send_exc:
                        logger.debug(
                            f"audio.chat.stream tts_done frame send failed: error={send_exc}"
                        )

        async def _finalize_turn(
            commit_at: Optional[float],
            *,
            auto: bool = False,
            turn_id: Optional[str] = None,
        ) -> None:
            nonlocal processing_turn
            nonlocal active_turn_cancelled
            nonlocal active_tts_sender_task
            if processing_turn:
                return
            processing_turn = True
            try:
                raw_transcript_text = transcriber.get_full_transcript()
                transcript_text = apply_transcript_text_policy(
                    raw_transcript_text,
                    policy=effective_stt_policy,
                    is_partial=False,
                )
                final_emit_at = time.time()
                eos_detected_at = final_emit_at
                try:
                    commit_ts = float(commit_at) if commit_at is not None else None
                    if commit_ts and commit_ts > 0:
                        eos_detected_at = commit_ts
                except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
                    eos_detected_at = final_emit_at
                payload = {
                    "type": "full_transcript",
                    "text": transcript_text,
                    "timestamp": final_emit_at,
                    # EOS/turn-end anchor used for downstream voice-to-voice metric timing.
                    "voice_to_voice_start": eos_detected_at,
                }
                payload.update(
                    _build_transcript_diagnostics_payload(
                        auto_commit=auto,
                        vad_status=vad_status,
                        diarization_status="disabled",
                    )
                )
                await _send_stream_payload(payload)

                # Metric for commit->final emit latency
                try:
                    observe_stt_final_latency_seconds(
                        endpoint="audio.chat.stream",
                        model=getattr(config, "model", "parakeet"),
                        variant=getattr(config, "model_variant", "standard"),
                        value=max(0.0, final_emit_at - eos_detected_at),
                    )
                except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
                    logger.debug(
                        'metrics observe failed (stt_final_latency_seconds, endpoint=audio.chat.stream): {}',
                        exc,
                    )

                overlap_session = None
                overlap_sender_task = None
                overlap_chunker: Optional[PhraseChunker] = None
                overlap_provider = tts_provider or "auto"

                try:
                    tts_service = await _shim_get_tts_service()
                except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as tts_exc:
                    logger.debug(f"audio.chat.stream overlap session skipped (tts service unavailable): {tts_exc}")
                    tts_service = None

                if tts_service is not None and callable(getattr(tts_service, "open_realtime_session", None)):
                    try:
                        rt_config = RealtimeSessionConfig(
                            model=str(tts_model),
                            voice=str(tts_voice),
                            response_format=str(response_format),
                            speed=float(tts_speed),
                            lang_code=None,
                            extra_params=tts_extra_params,
                            provider=str(tts_provider) if tts_provider else None,
                        )
                        rt_handle = await tts_service.open_realtime_session(
                            config=rt_config,
                            provider_hint=str(tts_provider) if tts_provider else None,
                            route="audio.chat.stream",
                            user_id=int(user_id_for_usage) if user_id_for_usage else None,
                        )
                        overlap_session = getattr(rt_handle, "session", None)
                        overlap_provider = (
                            getattr(rt_handle, "provider", None)
                            or tts_provider
                            or "auto"
                        )
                        overlap_warning = getattr(rt_handle, "warning", None)
                        overlap_chunker = PhraseChunker()
                        if _outer_stream:
                            await _outer_stream.send_json(
                                {
                                    "type": "tts_start",
                                    "format": response_format,
                                    "provider": overlap_provider,
                                    "voice": tts_voice,
                                }
                            )
                            if overlap_warning:
                                await _outer_stream.send_json(
                                    {"type": "warning", "message": str(overlap_warning)}
                                )

                        async def _overlap_audio_sender() -> None:
                            try:
                                async for chunk in overlap_session.audio_stream():
                                    if not chunk:
                                        continue
                                    if turn_id is not None and (active_turn_cancelled or active_turn_id != turn_id):
                                        continue
                                    await websocket.send_bytes(chunk)
                                    if _outer_stream:
                                        _outer_stream.mark_activity()
                            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as send_exc:
                                logger.debug(f"audio.chat.stream overlap audio sender failed: {send_exc}")

                        overlap_sender_task = create_task(_overlap_audio_sender())
                        active_tts_sender_task = overlap_sender_task
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as overlap_exc:
                        logger.debug(f"audio.chat.stream overlap init failed: {overlap_exc}")
                        overlap_session = None
                        overlap_chunker = None
                        overlap_sender_task = None

                async def _on_overlap_delta(delta: str) -> None:
                    if overlap_session is None or overlap_chunker is None:
                        return
                    if turn_id is not None and (active_turn_cancelled or active_turn_id != turn_id):
                        return
                    phrases = overlap_chunker.push(delta)
                    for phrase in phrases:
                        if turn_id is not None and (active_turn_cancelled or active_turn_id != turn_id):
                            return
                        await overlap_session.push_text(phrase)
                        await overlap_session.commit()
                        await asyncio.sleep(0)

                assistant_text, finish_reason, usage_payload = await _stream_llm(
                    transcript_text,
                    on_delta=_on_overlap_delta if overlap_session is not None else None,
                    turn_id=turn_id,
                )

                if overlap_session is not None and overlap_chunker is not None:
                    try:
                        tail = overlap_chunker.flush().strip()
                        if tail and not (turn_id is not None and (active_turn_cancelled or active_turn_id != turn_id)):
                            await overlap_session.push_text(tail)
                            await overlap_session.commit()
                        await overlap_session.finish()
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as overlap_finish_exc:
                        logger.debug(f"audio.chat.stream overlap finish failed: {overlap_finish_exc}")
                    finally:
                        if overlap_sender_task is not None:
                            try:
                                await overlap_sender_task
                            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as overlap_wait_exc:
                                logger.debug(f"audio.chat.stream overlap sender wait failed: {overlap_wait_exc}")
                        if _outer_stream:
                            try:
                                await _outer_stream.send_json({"type": "tts_done"})
                            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as send_exc:
                                logger.debug(f"audio.chat.stream tts_done send failed: {send_exc}")
                        if active_tts_sender_task is overlap_sender_task:
                            active_tts_sender_task = None

                if _outer_stream and not (turn_id is not None and (active_turn_cancelled or active_turn_id != turn_id)):
                    await _outer_stream.send_json(
                        {
                            "type": "llm_message",
                            "text": assistant_text,
                            "finish_reason": finish_reason,
                            "usage": usage_payload,
                        }
                    )

                if turn_id is not None and (active_turn_cancelled or active_turn_id != turn_id):
                    return

                action_result = await _maybe_run_action(transcript_text)
                if _outer_stream:
                    await _outer_stream.send_json(
                        {
                            "type": "assistant_summary",
                            "finish_reason": finish_reason,
                            "usage": usage_payload,
                            "action": action_result,
                        }
                    )
                if action_result:
                    try:
                        chat_history.append({"role": "tool", "content": json.dumps(action_result)})
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
                        logger.debug(f"Failed to append action_result to chat_history: {exc}")
                    if _outer_stream:
                        await _outer_stream.send_json({"type": "action_result", **action_result})
                await _persist_turn(transcript_text, assistant_text, action_result)
                if overlap_session is None:
                    await _stream_tts(assistant_text, eos_detected_at)
            finally:
                try:
                    transcriber.reset()
                except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
                    logger.debug(f"audio.chat.stream transcriber.reset() failed in finalize_turn: {exc}")
                processing_turn = False

        async def _start_turn(commit_at: Optional[float], *, auto: bool) -> Optional[str]:
            nonlocal active_turn_task
            nonlocal active_turn_id
            nonlocal active_turn_cancelled
            nonlocal turn_sequence

            if active_turn_task is not None and not active_turn_task.done():
                return active_turn_id

            turn_sequence += 1
            turn_id = f"turn-{turn_sequence}"
            active_turn_id = turn_id
            active_turn_cancelled = False

            async def _runner() -> None:
                nonlocal active_turn_task
                nonlocal active_turn_id
                try:
                    await _finalize_turn(commit_at, auto=auto, turn_id=turn_id)
                except asyncio.CancelledError:
                    logger.debug(f"audio.chat.stream turn cancelled: turn_id={turn_id}")
                finally:
                    if active_turn_id == turn_id:
                        active_turn_id = None
                    if active_turn_task is task_ref:
                        active_turn_task = None

            task_ref = create_task(_runner())
            active_turn_task = task_ref
            return turn_id

        async def _cancel_active_turn(*, reason: str, emit_interrupted: bool) -> None:
            nonlocal active_turn_cancelled
            nonlocal active_tts_sender_task
            interrupted_turn_id = active_turn_id
            had_active_turn = bool(interrupted_turn_id) or (
                active_turn_task is not None and not active_turn_task.done()
            )
            active_turn_cancelled = True
            if active_turn_task is not None and not active_turn_task.done():
                active_turn_task.cancel()
            if active_tts_sender_task is not None and not active_tts_sender_task.done():
                active_tts_sender_task.cancel()
                with contextlib.suppress(_AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS):
                    await active_tts_sender_task
                active_tts_sender_task = None
            if had_active_turn:
                with contextlib.suppress(_AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS):
                    transcriber.reset()
            if emit_interrupted:
                await _send_stream_payload(
                    {
                        "type": "interrupted",
                        "turn_id": interrupted_turn_id,
                        "phase": "both",
                        "reason": reason,
                    }
                )

        async def _process_chat_audio(audio_bytes: bytes, *, allow_auto_commit: bool) -> None:
            nonlocal turn_detector
            nonlocal vad_warning_sent
            nonlocal vad_status
            auto_commit_triggered = False
            commit_at = time.time()
            if allow_auto_commit and turn_detector:
                auto_commit_triggered = turn_detector.observe(audio_bytes)
                if not turn_detector.available and not vad_warning_sent:
                    vad_warning_sent = True
                    vad_status = "fail_open"
                    await _send_vad_warning(
                        "Silero VAD disabled; continuing without auto-commit",
                        turn_detector.unavailable_reason,
                    )
                    turn_detector = None

            result = await transcriber.process_audio_chunk(audio_bytes)
            if result:
                result.pop("_audio_chunk", None)
                await _send_stream_payload(result)

            if allow_auto_commit and auto_commit_triggered:
                await _start_turn(
                    commit_at=getattr(turn_detector, "last_trigger_at", None)
                    if turn_detector
                    else commit_at,
                    auto=True,
                )

        async def _drain_paused_audio_queue() -> None:
            queued_audio = list(paused_audio_chunks)
            paused_audio_chunks.clear()
            control_session.release_paused_audio()
            for buffered_audio, _buffered_seconds in queued_audio:
                await _process_chat_audio(buffered_audio, allow_auto_commit=False)

        try:
            for event in protocol_decision.events:
                await _send_stream_payload(event)
            while True:
                raw_msg = await websocket.receive_text()
                try:
                    if _outer_stream:
                        _outer_stream.mark_activity()
                except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
                    logger.debug(f"audio.chat.stream outer_stream.mark_activity failed: {exc}")
                try:
                    data = json.loads(raw_msg)
                except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
                    emit_stt_error_total(
                        endpoint="audio.chat.stream",
                        provider=stt_metrics_provider,
                        reason="validation_error",
                    )
                    if _outer_stream:
                        await _outer_stream.send_json(
                            _audio_ws_error_payload(
                                code="bad_request",
                                message="Invalid JSON",
                                request_id=request_id,
                            )
                        )
                    continue

                msg_type = data.get("type")
                if msg_type == "audio":
                    audio_base64 = data.get("data", "")
                    try:
                        audio_bytes = base64.b64decode(audio_base64)
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
                        emit_stt_error_total(
                            endpoint="audio.chat.stream",
                            provider=stt_metrics_provider,
                            reason="validation_error",
                        )
                        if _outer_stream:
                            await _outer_stream.send_json(
                                _audio_ws_error_payload(
                                    code="bad_request",
                                    message="Invalid base64 audio frame",
                                    request_id=request_id,
                                )
                            )
                        continue

                    try:
                        seconds = _estimate_stream_audio_seconds(audio_bytes, int(config.sample_rate or 16000))
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
                        seconds = float(len(audio_bytes)) / float(
                            4 * max(1, int(config.sample_rate or 16000))
                        )

                    try:
                        await _on_heartbeat()
                    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as hb_exc:
                        logger.debug(f"audio.chat.stream heartbeat failed: error={hb_exc}")

                    if control_session.state == "paused":
                        buffered_seconds = seconds
                        paused_audio_chunks.append((audio_bytes, buffered_seconds))
                        paused_audio_decision = control_session.buffer_paused_audio(
                            buffered_seconds,
                            now=time.time(),
                        )
                        if paused_audio_decision.dropped_seconds > 0:
                            _drop_oldest_stream_audio(
                                paused_audio_chunks,
                                paused_audio_decision.dropped_seconds,
                            )
                        for event in paused_audio_decision.events:
                            await _send_stream_payload(event)
                        continue

                    # Bill usage only for audio that will actually be processed (not paused/dropped)
                    try:
                        await _on_audio_quota(seconds, int(config.sample_rate or 16000))
                    except QuotaExceeded as qe:
                        stt_request_status = "quota_exceeded"
                        stt_session_close_reason = "error"
                        emit_stt_error_total(
                            endpoint="audio.chat.stream",
                            provider=stt_metrics_provider,
                            reason="quota",
                        )
                        if _outer_stream:
                            try:
                                await _outer_stream.send_json(
                                    _audio_ws_quota_error_payload(
                                        quota=getattr(qe, "quota", "daily_minutes"),
                                        message="Streaming quota exceeded",
                                        request_id=request_id,
                                    )
                                )
                            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as send_exc:
                                logger.debug(
                                    f"WebSocket send_json quota error failed (audio.chat.stream): error={send_exc}"
                                )
                        try:
                            await websocket.close(code=_policy_close_code(), reason="quota_exceeded")
                        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as close_exc:
                            logger.debug(
                                f"WebSocket close (quota case) failed (audio.chat.stream): error={close_exc}"
                            )
                        break

                    await _process_chat_audio(audio_bytes, allow_auto_commit=True)

                elif msg_type == "interrupt":
                    reason = str(data.get("reason") or "client_cancel")
                    await _cancel_active_turn(reason=reason, emit_interrupted=True)
                elif msg_type in {"control", "commit", "reset", "stop"}:
                    decision = control_session.handle_frame(data)
                    if decision.error:
                        emit_stt_error_total(
                            endpoint="audio.chat.stream",
                            provider=stt_metrics_provider,
                            reason=decision.error.get("code") or "invalid_control",
                        )
                        await _send_stream_payload(decision.error)
                        continue

                    if decision.intent == "pause":
                        await _cancel_active_turn(reason="client_pause", emit_interrupted=False)

                    for event in decision.events:
                        await _send_stream_payload(event)

                    if decision.intent == "resume":
                        await _drain_paused_audio_queue()
                    elif decision.should_reset:
                        paused_audio_chunks.clear()
                        control_session.release_paused_audio()
                        try:
                            transcriber.reset()
                        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
                            logger.debug(f"audio.chat.stream transcriber.reset() failed on reset message: {exc}")
                    elif decision.intent == "commit":
                        await _start_turn(time.time(), auto=False)
                    elif decision.intent == "stop":
                        stt_session_close_reason = "client_stop"
                        paused_audio_chunks.clear()
                        control_session.release_paused_audio()
                        if active_turn_task is not None and not active_turn_task.done():
                            with contextlib.suppress(_AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS):
                                await active_turn_task
                        if _outer_stream:
                            await _outer_stream.done()
                        break
                else:
                    if _outer_stream:
                        await _outer_stream.send_json(
                            {"type": "warning", "message": f"Unknown message type {msg_type}"}
                        )

        except WebSocketDisconnect:
            stt_session_close_reason = "client_disconnect"
            logger.info("Audio chat WS disconnected")
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
            stt_request_status = "internal_error"
            stt_session_close_reason = "error"
            emit_stt_error_total(
                endpoint="audio.chat.stream",
                provider=stt_metrics_provider,
                reason="internal",
            )
            logger.error(f"Audio chat WS error: {exc}", exc_info=True)
            try:
                if _outer_stream:
                    await _outer_stream.send_json(
                        _audio_ws_error_payload(
                            code="internal_error",
                            message="Internal error",
                            request_id=request_id,
                            exc=exc,
                        )
                    )
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as send_exc:  # noqa: BLE001
                logger.debug(f"audio.chat.stream failed to send internal_error frame: {send_exc}")
    finally:
        if stt_session_started:
            emit_stt_session_end_total(
                provider=stt_metrics_provider,
                session_close_reason=stt_session_close_reason,
            )
            emit_stt_request_total(
                endpoint="audio.chat.stream",
                provider=stt_metrics_provider,
                model=stt_metrics_model,
                status=stt_request_status,
            )
        if acquired_stream:
            try:
                await _finish_stream(user_id_for_usage)
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
                logger.debug(
                    f"Failed to release streaming quota slot (audio.chat.stream): "
                    f"user_id={user_id_for_usage}, error={exc}"
                )
        try:
            if _outer_stream:
                await _outer_stream.stop()
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
            logger.debug(f"audio.chat.stream outer_stream.stop failed: {exc}")
        try:
            await websocket.close()
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
            logger.debug(f"audio.chat.stream websocket close failed in cleanup: {exc}")


@ws_router.websocket("/stream/tts")
async def websocket_tts(
    websocket: WebSocket,
    token: Optional[str] = Query(None),  # noqa: ARG001 - kept for parity with transcribe endpoint
):
    """
    WebSocket TTS streaming endpoint: accepts a prompt frame and streams audio bytes.

    Authentication mirrors `_audio_ws_authenticate`:
    - Multi-user: `X-API-KEY` header, `Authorization: Bearer <JWT>`, or `token` query parameter (API key or JWT).
    - Single-user: fixed API key via header, `token` query parameter, or initial auth message; optional IP allowlist.
    """
    await websocket.accept()

    try:
        _raw_idle = os.getenv("AUDIO_WS_IDLE_TIMEOUT_S") or os.getenv("STREAM_IDLE_TIMEOUT_S")
        _idle_timeout = float(_raw_idle) if _raw_idle else None
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        _idle_timeout = None

    aio = _shim_asyncio()
    wait_for = getattr(aio, "wait_for", asyncio.wait_for)

    # Wrap websocket for consistent metrics/heartbeats
    _outer_stream = None
    try:
        from tldw_Server_API.app.core.Streaming.streams import WebSocketStream as _WSStream

        _outer_stream = _WSStream(
            websocket,
            heartbeat_interval_s=None,
            compat_error_type=_audio_ws_compat_error_type_enabled(),
            close_on_done=True,
            idle_timeout_s=_idle_timeout,
            labels={"component": "audio", "endpoint": "audio_tts_ws"},
        )
        await _outer_stream.start()
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        _outer_stream = None

    # Correlate via request id
    try:
        _hdrs = websocket.headers or {}
        request_id = (
            _hdrs.get("x-request-id")
            or _hdrs.get("X-Request-Id")
            or (websocket.query_params.get("request_id") if hasattr(websocket, "query_params") else None)
            or str(uuid4())
        )
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        request_id = str(uuid4())
    try:
        logger.info(f"TTS WS connected: request_id={request_id}")
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"TTS WS connection logging failed: {exc}")

    def _policy_close_code() -> int:
        flag = str(os.getenv("AUDIO_WS_QUOTA_CLOSE_1008", "0")).strip().lower()
        return 1008 if is_truthy(flag) else 4003

    # Authenticate (parity with STT WS)
    auth_ok, jwt_user_id = await _shim_audio_ws_authenticate(
        websocket,
        _outer_stream,
        endpoint_id="audio.stream.tts",
        ws_path="/api/v1/audio/stream/tts",
    )
    if not auth_ok:
        return
    if not await _guard_audio_ws_work_start(
        websocket,
        kind="audio.stream.tts",
        outer_stream=_outer_stream,
        request_id=request_id,
    ):
        return

    # Determine quota user id
    if is_multi_user_mode() and jwt_user_id is not None:
        user_id_for_usage = int(jwt_user_id)
    else:
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings as _get_settings

        _s = _get_settings()
        user_id_for_usage = getattr(_s, "SINGLE_USER_FIXED_ID", 1)

    acquired_stream = False

    try:
        # Concurrency guard
        try:
            ok_stream, msg_stream = await _can_start_stream(user_id_for_usage)
            if not ok_stream:
                if _outer_stream:
                    await _outer_stream.send_json(
                        {"type": "error", "message": msg_stream or "Concurrent audio streams limit reached"}
                    )
                await websocket.close(code=_policy_close_code())
                return
            acquired_stream = True
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
            if _outer_stream:
                await _outer_stream.send_json(
                    {"type": "error", "message": "Unable to evaluate audio stream quota or concurrency"}
                )
            await websocket.close(code=_policy_close_code())
            return

        # Parse prompt frame
        try:
            prompt_message = await wait_for(websocket.receive_text(), timeout=10.0)
            prompt_data = json.loads(prompt_message)
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
            if _outer_stream:
                try:
                    data_payload = {"request_id": request_id}
                    details = _maybe_debug_details(exc)
                    if details:
                        data_payload["details"] = details
                    await _outer_stream.error(
                        "bad_request",
                        "Prompt frame required",
                        data=data_payload if data_payload else None,
                    )
                except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as send_exc:
                    logger.debug(f"TTS WS error frame send failed: {send_exc}")
            await websocket.close(code=4400)
            return

        if (prompt_data.get("type") or "prompt") not in {"prompt", "config"}:
            if _outer_stream:
                await _outer_stream.error("bad_request", "First frame must be type=prompt")
            await websocket.close(code=4400)
            return

        text = prompt_data.get("text") or prompt_data.get("input")
        if not text:
            if _outer_stream:
                await _outer_stream.error("bad_request", "Prompt text is required")
            await websocket.close(code=4400)
            return

        # Build TTS request
        response_format = prompt_data.get("format") or prompt_data.get("response_format") or "pcm"
        allowed_formats = {"mp3", "opus", "aac", "flac", "wav", "pcm"}
        if response_format not in allowed_formats:
            if _outer_stream:
                await _outer_stream.error("bad_request", f"Unsupported format '{response_format}'")
            await websocket.close(code=4400)
            return

        try:
            speed_val = float(prompt_data.get("speed", 1.0))
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
            speed_val = 1.0

        extra_params = prompt_data.get("extra_params")
        if extra_params is not None and not isinstance(extra_params, dict):
            extra_params = None
        resolved_tts = resolve_tts_request_defaults(
            provider=prompt_data.get("provider"),
            model=prompt_data.get("model"),
            voice=prompt_data.get("voice"),
        )

        speech_req = OpenAISpeechRequest(
            model=resolved_tts.model,
            input=text,
            voice=resolved_tts.voice,
            response_format=response_format,
            speed=speed_val,
            stream=True,
            lang_code=prompt_data.get("lang") or prompt_data.get("lang_code"),
            extra_params=extra_params,
        )

        provider_hint = resolved_tts.provider
        reg = _shim_get_metrics_registry()
        tts_service = await _shim_get_tts_service()

        async def _ws_tts_error_handler(exc: Exception) -> None:
            if not _outer_stream:
                return
            logger.error(f"audio.stream.tts TTS generation failed: {exc}", exc_info=True)
            data_payload = {"request_id": request_id}
            details = _maybe_debug_details(exc)
            if details:
                data_payload["details"] = details
            await _outer_stream.error(
                "internal_error",
                "TTS generation failed",
                data=data_payload if data_payload else None,
            )

        # Delegate to the shared streaming helper; it manages its own
        # producer/consumer tasks and error handling.
        await _stream_tts_to_websocket(
            websocket=websocket,
            speech_req=speech_req,
            tts_service=tts_service,
            provider=provider_hint,
            outer_stream=_outer_stream,
            reg=reg,
            route="audio.stream.tts",
            component_label="audio_tts_ws",
            error_handler=_ws_tts_error_handler if _outer_stream else None,
            asyncio_module=aio,
        )
    finally:
        if acquired_stream:
            try:
                await _finish_stream(user_id_for_usage)
            except EXPECTED_DB_EXC as e:
                logger.debug(
                    f"Failed to release streaming quota slot (audio.stream.tts): "
                    f"user_id={user_id_for_usage}, error={e}"
                )
        try:
            if _outer_stream:
                await _outer_stream.done()
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as outer_exc:
            try:
                await websocket.close()
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as close_exc:
                logger.debug(
                    f"audio.stream.tts websocket close failed after _outer_stream.done error: "
                    f"outer_error={outer_exc}, close_error={close_exc}"
                )


@ws_router.websocket("/stream/tts/realtime")
async def websocket_tts_realtime(
    websocket: WebSocket,
    token: Optional[str] = Query(None),  # noqa: ARG001 - kept for parity
):
    """
    WebSocket realtime TTS endpoint: accepts streaming text frames and streams audio bytes.

    Client frames:
      - type=config: set provider/model/voice/format/speed/lang/extra_params/auto_flush_ms/auto_flush_tokens
      - type=text:   text delta (fields: delta | text | input)
      - type=commit: flush buffered text to synthesis
      - type=final:  flush and close session

    Server frames:
      - JSON status/warning/error frames
      - Binary audio frames for synthesized audio
    """
    await websocket.accept()

    try:
        _raw_idle = os.getenv("AUDIO_WS_IDLE_TIMEOUT_S") or os.getenv("STREAM_IDLE_TIMEOUT_S")
        _idle_timeout = float(_raw_idle) if _raw_idle else None
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        _idle_timeout = None

    aio = _shim_asyncio()
    wait_for = getattr(aio, "wait_for", asyncio.wait_for)
    TimeoutError = getattr(aio, "TimeoutError", asyncio.TimeoutError)
    create_task = getattr(aio, "create_task", asyncio.create_task)

    _outer_stream = None
    try:
        from tldw_Server_API.app.core.Streaming.streams import WebSocketStream as _WSStream

        _outer_stream = _WSStream(
            websocket,
            heartbeat_interval_s=None,
            compat_error_type=_audio_ws_compat_error_type_enabled(),
            close_on_done=True,
            idle_timeout_s=_idle_timeout,
            labels={"component": "audio", "endpoint": "audio_tts_realtime_ws"},
        )
        await _outer_stream.start()
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        _outer_stream = None

    try:
        _hdrs = websocket.headers or {}
        request_id = (
            _hdrs.get("x-request-id")
            or _hdrs.get("X-Request-Id")
            or (websocket.query_params.get("request_id") if hasattr(websocket, "query_params") else None)
            or str(uuid4())
        )
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        request_id = str(uuid4())

    def _policy_close_code() -> int:
        flag = str(os.getenv("AUDIO_WS_QUOTA_CLOSE_1008", "0")).strip().lower()
        return 1008 if is_truthy(flag) else 4003

    async def _send_json(payload: dict[str, Any]) -> None:
        if _outer_stream:
            await _outer_stream.send_json(payload)
        else:
            await websocket.send_json(payload)

    done_sent = False
    error_sent = False

    def _allowed_formats_for(provider_name: Optional[str]) -> set[str]:
        try:
            from tldw_Server_API.app.core.TTS.tts_validation import ProviderLimits

            limits = ProviderLimits.get_limits(str(provider_name).lower()) if provider_name else ProviderLimits.get_limits("default")
            return set(limits.get("valid_formats", {"pcm", "wav", "mp3"}))
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
            return {"pcm", "wav", "mp3", "opus", "flac"}

    async def _send_error(code: str, message: str, *, close: bool = False, close_code: Optional[int] = None) -> None:
        nonlocal error_sent
        payload = {
            "type": "error",
            "code": code,
            "message": message,
            "request_id": request_id,
            "data": {"request_id": request_id},
        }
        if _audio_ws_compat_error_type_enabled():
            payload["error_type"] = code
        if _outer_stream:
            await _outer_stream.send_json(payload)
        else:
            await websocket.send_json(payload)
        error_sent = True
        if not close:
            return
        if close_code is None:
            if code == "quota_exceeded":
                close_code = _policy_close_code()
            else:
                try:
                    close_code = _outer_stream._map_close_code(code) if _outer_stream else 1011
                except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
                    close_code = 1011
        with contextlib.suppress(_AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS):
            await websocket.close(code=close_code)

    auth_ok, jwt_user_id = await _shim_audio_ws_authenticate(
        websocket,
        _outer_stream,
        endpoint_id="audio.stream.tts.realtime",
        ws_path="/api/v1/audio/stream/tts/realtime",
    )
    if not auth_ok:
        return
    if not await _guard_audio_ws_work_start(
        websocket,
        kind="audio.stream.tts.realtime",
        outer_stream=_outer_stream,
        request_id=request_id,
    ):
        return

    if is_multi_user_mode() and jwt_user_id is not None:
        user_id_for_usage = int(jwt_user_id)
    else:
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings as _get_settings

        _s = _get_settings()
        user_id_for_usage = getattr(_s, "SINGLE_USER_FIXED_ID", 1)

    acquired_stream = False
    session = None
    sender_task: Optional[asyncio.Task] = None

    def _coerce_float(val: Any, default: float) -> float:
        try:
            return float(val)
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
            return default

    def _coerce_int(val: Any, default: int) -> int:
        try:
            return int(val)
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
            return default

    try:
        try:
            ok_stream, msg_stream = await _can_start_stream(user_id_for_usage)
            if not ok_stream:
                await _send_error("quota_exceeded", msg_stream or "Concurrent audio streams limit reached", close=False)
                await websocket.close(code=_policy_close_code())
                return
            acquired_stream = True
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
            await _send_error("quota_error", "Unable to evaluate audio stream quota or concurrency", close=False)
            await websocket.close(code=_policy_close_code())
            return

        # Read initial config or text frame
        try:
            raw_msg = await wait_for(websocket.receive_text(), timeout=10.0)
            first = json.loads(raw_msg)
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"TTS realtime WS initial frame parse failed: {exc}")
            await _send_error("bad_request", "Initial config or text frame required", close=False)
            await websocket.close(code=4400)
            return

        msg_type = (first.get("type") or "").lower()
        if msg_type not in {"config", "prompt", "text"}:
            await _send_error("bad_request", "First frame must be type=config, prompt, or text", close=False)
            await websocket.close(code=4400)
            return

        # Defaults
        default_provider = "vibevoice_realtime"
        default_model = "vibevoice-realtime-0.5b"
        default_voice = "default"
        default_format = "pcm"
        default_speed = 1.0
        default_auto_flush_ms = _coerce_int(os.getenv("TTS_REALTIME_AUTO_FLUSH_MS"), 600)
        default_auto_flush_tokens = _coerce_int(os.getenv("TTS_REALTIME_AUTO_FLUSH_TOKENS"), 60)

        provider_hint = first.get("provider") or default_provider
        model = first.get("model") or default_model
        voice = first.get("voice") or default_voice
        response_format = first.get("format") or first.get("response_format") or default_format
        response_format = str(response_format).lower()
        speed = _coerce_float(first.get("speed", default_speed), default_speed)
        lang_code = first.get("lang") or first.get("lang_code")
        extra_params = first.get("extra_params")
        if extra_params is not None and not isinstance(extra_params, dict):
            extra_params = None

        auto_flush_ms = _coerce_int(first.get("auto_flush_ms", default_auto_flush_ms), default_auto_flush_ms)
        auto_flush_tokens = _coerce_int(first.get("auto_flush_tokens", default_auto_flush_tokens), default_auto_flush_tokens)
        if auto_flush_ms < 0:
            auto_flush_ms = 0
        if auto_flush_tokens < 0:
            auto_flush_tokens = 0

        allowed_formats = _allowed_formats_for(str(provider_hint) if provider_hint else None)
        if response_format not in allowed_formats:
            await _send_error("bad_request", f"Unsupported format '{response_format}'")
            await websocket.close(code=4400)
            return

        tts_service = await _shim_get_tts_service()
        config = RealtimeSessionConfig(
            model=str(model),
            voice=str(voice),
            response_format=str(response_format),
            speed=float(speed),
            lang_code=lang_code,
            extra_params=extra_params,
            provider=str(provider_hint) if provider_hint else None,
        )
        handle = await tts_service.open_realtime_session(
            config=config,
            provider_hint=str(provider_hint) if provider_hint else None,
            route="audio.stream.tts.realtime",
            user_id=user_id_for_usage,
        )
        session = handle.session
        if handle.provider:
            provider_allowed = _allowed_formats_for(handle.provider)
            if response_format not in provider_allowed:
                await _send_error(
                    "bad_request",
                    f"Unsupported format '{response_format}' for provider '{handle.provider}'",
                    close=True,
                    close_code=4400,
                )
                with contextlib.suppress(_AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS):
                    await session.finish()
                return

        await _send_json(
            {
                "type": "ready",
                "provider": handle.provider or provider_hint,
                "format": response_format,
                "sample_rate": 24000,
                "request_id": request_id,
            }
        )
        if handle.warning:
            await _send_json({"type": "warning", "message": handle.warning, "request_id": request_id})

        async def _audio_sender(session_obj: Any) -> None:
            try:
                async for chunk in session_obj.audio_stream():
                    if not chunk:
                        continue
                    await websocket.send_bytes(chunk)
                    if _outer_stream:
                        _outer_stream.mark_activity()
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"TTS realtime audio sender failed: {exc}")

        sender_task = create_task(_audio_sender(session))

        # Handle the initial frame as text if provided
        initial_text = first.get("delta") or first.get("text") or first.get("input")
        buffered_tokens = 0
        buffered_chars = 0
        last_input_ts: Optional[float] = None

        if isinstance(initial_text, str) and initial_text:
            await session.push_text(initial_text)
            buffered_chars += len(initial_text)
            buffered_tokens += len(initial_text.split())
            last_input_ts = time.monotonic()

        while True:
            timeout = None
            if auto_flush_ms and buffered_chars > 0:
                now = time.monotonic()
                elapsed = now - (last_input_ts or now)
                remaining = (auto_flush_ms / 1000.0) - elapsed
                timeout = max(0.0, remaining)

            try:
                raw_msg = await wait_for(websocket.receive_text(), timeout=timeout)
            except TimeoutError:
                if buffered_chars > 0:
                    await session.commit()
                    buffered_chars = 0
                    buffered_tokens = 0
                    last_input_ts = None
                continue

            try:
                data = json.loads(raw_msg)
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
                await _send_error("bad_request", "Invalid JSON frame", close=False)
                await websocket.close(code=4400)
                return

            msg_type = (data.get("type") or "").lower()
            if msg_type in {"text", "input"}:
                delta = data.get("delta") or data.get("text") or data.get("input")
                if not isinstance(delta, str) or not delta:
                    continue
                await session.push_text(delta)
                buffered_chars += len(delta)
                buffered_tokens += len(delta.split())
                last_input_ts = time.monotonic()
                if auto_flush_tokens and buffered_tokens >= auto_flush_tokens:
                    await session.commit()
                    buffered_chars = 0
                    buffered_tokens = 0
                    last_input_ts = None
            elif msg_type == "commit":
                await session.commit()
                buffered_chars = 0
                buffered_tokens = 0
                last_input_ts = None
            elif msg_type == "interrupt":
                reason = str(data.get("reason") or "client_cancel")
                with contextlib.suppress(_AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS):
                    await session.finish()
                if sender_task and not sender_task.done():
                    sender_task.cancel()
                    with contextlib.suppress(_AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS):
                        await sender_task

                try:
                    handle = await tts_service.open_realtime_session(
                        config=config,
                        provider_hint=str(provider_hint) if provider_hint else None,
                        route="audio.stream.tts.realtime",
                        user_id=user_id_for_usage,
                    )
                    session = handle.session
                    if handle.provider:
                        provider_allowed = _allowed_formats_for(handle.provider)
                        if response_format not in provider_allowed:
                            await _send_error(
                                "bad_request",
                                f"Unsupported format '{response_format}' for provider '{handle.provider}'",
                                close=True,
                                close_code=4400,
                            )
                            with contextlib.suppress(_AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS):
                                await session.finish()
                            return
                    sender_task = create_task(_audio_sender(session))
                    buffered_chars = 0
                    buffered_tokens = 0
                    last_input_ts = None
                    await _send_json(
                        {
                            "type": "interrupted",
                            "phase": "tts",
                            "reason": reason,
                            "request_id": request_id,
                        }
                    )
                except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as interrupt_exc:
                    logger.error(f"TTS realtime interrupt recovery failed: {interrupt_exc}", exc_info=True)
                    await _send_error("internal_error", "Failed to recover realtime TTS session", close=True)
                    return
            elif msg_type == "final":
                await session.finish()
                break
            elif msg_type == "config":
                await _send_json({"type": "warning", "message": "Config updates are ignored after session start."})
            elif msg_type == "ping":
                await _send_json({"type": "pong"})
            else:
                await _send_json({"type": "warning", "message": f"Unknown message type '{msg_type}'"})

        if sender_task:
            await sender_task
        if getattr(session, "error", None):
            await _send_error("internal_error", "Realtime TTS session failed", close=True)
        if not error_sent:
            if _outer_stream:
                await _outer_stream.done()
            else:
                await _send_json({"type": "done"})
            done_sent = True
    except WebSocketDisconnect:
        logger.info("TTS realtime WS disconnected")
    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(f"TTS realtime WS error: {exc}", exc_info=True)
        with contextlib.suppress(_AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS):
            await _send_error("internal_error", "Internal error", close=True)
    finally:
        if session is not None:
            with contextlib.suppress(_AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS):
                await session.finish()
        if sender_task and not sender_task.done():
            sender_task.cancel()
            with contextlib.suppress(_AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS):
                await sender_task
        if acquired_stream:
            try:
                await _finish_stream(user_id_for_usage)
            except EXPECTED_DB_EXC as e:
                logger.debug(
                    f"Failed to release streaming quota slot (audio.stream.tts.realtime): "
                    f"user_id={user_id_for_usage}, error={e}"
                )
        try:
            if _outer_stream and not done_sent and not error_sent:
                await _outer_stream.done()
        except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as outer_exc:
            try:
                await websocket.close()
            except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS as close_exc:
                logger.debug(
                    "audio.stream.tts.realtime websocket close failed after _outer_stream.done error: "
                    f"outer_error={outer_exc}, close_error={close_exc}"
                )


@router.get(
    "/stream/status",
    response_model=StreamingStatusResponse,
    summary="Check streaming transcription availability",
)
async def streaming_status():
    """
    Report availability and capabilities of the streaming transcription WebSocket endpoint.

    Returns:
        StreamingStatusResponse with the following keys:
          - `status` (str): "available" if at least one streaming model is present, "unavailable" otherwise, or "error" on failure.
          - `available_models` (list[str]): Names of detected streaming model variants (e.g., "parakeet-mlx", "parakeet-standard", "parakeet-onnx").
          - `websocket_endpoint` (str): URL path of the streaming transcription WebSocket.
          - `supported_features` (dict): Feature flags indicating supported streaming capabilities (boolean values).
    """
    try:
        # Check available models
        available_models = []

        import importlib.util as _importlib_util

        # Check for MLX variant
        if _importlib_util.find_spec(
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX"
        ):
            available_models.append("parakeet-mlx")

        # Check for standard variant (NeMo)
        if _importlib_util.find_spec(
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo"
        ):
            available_models.append("parakeet-standard")

        # Check for ONNX variant
        if _importlib_util.find_spec("onnxruntime") and _importlib_util.find_spec(
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX"
        ):
            available_models.append("parakeet-onnx")

        return {
            "status": "available" if available_models else "unavailable",
            "available_models": available_models,
            "websocket_endpoint": "/api/v1/audio/stream/transcribe",
            "supported_features": {
                "partial_results": True,
                "multiple_languages": True,
                "concurrent_streams": True,
                "segment_metadata": True,
                "live_insights": True,
                "meeting_notes": True,
                "speaker_diarization": True,
                "audio_persistence": True,
            },
        }

    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        logger.error("Error checking streaming status", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": "An internal error occurred. Please try again later."},
        )


@router.get(
    "/stream/limits",
    response_model=StreamingLimitsResponse,
    summary="Get user's streaming quota and usage",
)
async def streaming_limits(
    request: Request,
    current_user: User = Depends(get_request_user),
):
    """
    Return the current user's streaming quota and usage summary.

    Returns:
        StreamingLimitsResponse with the following keys:
            - user_id (str): The user's identifier.
            - tier (str): The user's tier name (e.g., "free").
            - limits (dict): The resolved limit values (e.g., daily_minutes, concurrent_streams, concurrent_jobs, max_file_size_mb).
            - used_today_minutes (float): Minutes already used today (0.0 if unavailable).
            - remaining_minutes (float|None): Minutes remaining today (0.0 if none left, `None` if unknown/unbounded).
            - active_streams (int): Number of currently active streams (0 if unavailable).
            - _can_start_stream (bool): Whether the user may start another stream given current active streams and concurrent_streams limit.
    """
    # Correlate logs with request_id if available
    rid = ensure_request_id(request) if request is not None else None
    try:
        limits = await _get_limits_for_user(current_user.id)
    except EXPECTED_DB_EXC as e:
        get_ps_logger(request_id=rid, ps_component="endpoint", ps_job_kind="audio").warning(
            "Failed to get limits for user %s, falling back to defaults: %s", current_user.id, e
        )
        # Fallback to default free limits
        limits = {
            "daily_minutes": 30.0,
            "concurrent_streams": 1,
            "concurrent_jobs": 1,
            "max_file_size_mb": 25,
        }
    try:
        used_minutes = await _get_daily_minutes_used(current_user.id)
    except EXPECTED_DB_EXC as e:
        get_ps_logger(request_id=rid, ps_component="endpoint", ps_job_kind="audio").warning(
            "Failed to get used minutes for user %s, falling back to 0: %s", current_user.id, e
        )
        used_minutes = 0.0
    limit_minutes = limits.get("daily_minutes")
    if limit_minutes is None:
        remaining_minutes = None
    else:
        try:
            remaining_minutes = max(0.0, float(limit_minutes) - float(used_minutes))
        except (ValueError, TypeError) as e:
            get_ps_logger(request_id=rid, ps_component="endpoint", ps_job_kind="audio").warning(
                "Could not calculate remaining minutes for user %s: %s", current_user.id, e
            )
            remaining_minutes = None
    try:
        active_streams = await _active_streams_count(current_user.id)
    except EXPECTED_REDIS_EXC as e:
        get_ps_logger(request_id=rid, ps_component="endpoint", ps_job_kind="audio").warning(
            "Failed to get active streams for user %s, falling back to 0: %s", current_user.id, e
        )
        active_streams = 0
    try:
        tier = await _get_user_tier(current_user.id)
    except EXPECTED_DB_EXC as e:
        get_ps_logger(request_id=rid, ps_component="endpoint", ps_job_kind="audio").warning(
            "Failed to get tier for user %s, falling back to 'free': %s", current_user.id, e
        )
        tier = "free"
    try:
        max_streams = int(limits.get("concurrent_streams") or 0)
    except (ValueError, TypeError) as e:
        get_ps_logger(request_id=rid, ps_component="endpoint", ps_job_kind="audio").warning(
            "Could not parse concurrent_streams limit for user %s: %s", current_user.id, e
        )
        max_streams = 0
    can_start = (max_streams == 0) or (active_streams < max_streams)
    return {
        "user_id": current_user.id,
        "tier": tier,
        "limits": limits,
        "used_today_minutes": used_minutes,
        "remaining_minutes": remaining_minutes,
        "active_streams": active_streams,
        "can_start_stream": can_start,
        "_can_start_stream": can_start,
    }


@router.post(
    "/stream/test",
    response_model=StreamingTestResponse,
    summary="Test streaming transcription setup",
)
async def test_streaming():
    """
    Run a lightweight end-to-end check of the streaming transcription pipeline using a short generated audio sample.

    Performs a minimal initialization of the Parakeet streaming transcriber, sends a short synthetic audio chunk, and returns the transcriber's immediate response or a buffering status.

    Returns:
        StreamingTestResponse: On success, a JSON object with keys:
            - "status": "success"
            - "test_passed": True
            - "message": Human-readable success message
            - "test_result": Transcriber response or the string "Buffer accumulating"
        On failure, a JSONResponse with status_code 500 and a JSON object containing:
            - "status": "error"
            - "test_passed": False
            - "message": Error message describing the failure
    """
    try:
        import base64

        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import (
            UnifiedStreamingConfig,
            UnifiedStreamingTranscriber,
        )

        # Try to initialize transcriber
        config = UnifiedStreamingConfig(model="parakeet", model_variant="mlx")
        transcriber = UnifiedStreamingTranscriber(config)
        transcriber.initialize()

        # Generate test audio
        sample_rate = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = (0.5 * np.sin(440 * 2 * np.pi * t)).astype(np.float32)
        encoded = base64.b64encode(audio.tobytes()).decode("utf-8")

        # Try processing
        result = await transcriber.process_audio_chunk(base64.b64decode(encoded))

        return {
            "status": "success",
            "test_passed": True,
            "message": "Streaming transcription is working",
            "test_result": result if result else "Buffer accumulating",
        }

    except _AUDIO_STREAMING_NONCRITICAL_EXCEPTIONS:
        logger.error("Streaming test failed", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "test_passed": False,
                "message": "An internal error occurred during the streaming test. Please contact support if the problem persists.",
            },
        )
