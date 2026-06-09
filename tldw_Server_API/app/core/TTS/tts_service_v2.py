# tts_service_v2.py
# Description: Enhanced TTS service using the adapter pattern
#
# Imports
import asyncio
import base64
import copy
import inspect
import os
import tempfile
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

#
# Third-party Imports
import numpy as np
from loguru import logger

#
# Local Imports
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.Logging.log_context import new_request_id
from tldw_Server_API.app.core.Metrics import get_metrics_registry
from tldw_Server_API.app.core.Metrics.metrics_manager import MetricDefinition, MetricType

from .adapter_registry import (
    TTSAdapterFactory,
    TTSAdapterRegistry,
    TTSProvider,
    close_tts_factory,
    get_tts_factory,
)
from .adapters.base import AudioFormat, TTSAdapter, TTSCapabilities, TTSRequest, TTSResponse
from .adapters.omnivoice_sidecar_supervisor import OmniVoiceSidecarSupervisor
from .adapters.pocket_tts_cpp_runtime import (
    cleanup_transient_voice_reference,
    get_runtime_dir,
    materialize_custom_voice_reference,
    materialize_direct_voice_reference,
    prune_materialized_voice_cache,
    PROVIDER_MANAGED_VOICE_TOKEN_KEY,
    register_provider_managed_voice_path,
    revoke_provider_managed_voice_token,
)
from .audio_converter import AudioConverter
from .audio_utils import (
    crossfade_audio,
    evaluate_audio_quality,
    split_text_into_chunks,
    trim_trailing_silence,
)
from .circuit_breaker import (
    CircuitBreakerManager,
    CircuitOpenError,
    build_qwen_runtime_breaker_key,
    get_circuit_manager,
)
from .realtime_session import (
    BufferedRealtimeSession,
    RealtimeSessionConfig,
    RealtimeSessionHandle,
    RealtimeTTSSession,
)
from .streaming_audio_writer import AudioNormalizer, StreamingAudioWriter
from .tts_exceptions import (
    TTSAudioQualityError,
    TTSError,
    TTSFallbackExhaustedError,
    TTSGenerationError,
    TTSInvalidVoiceReferenceError,
    TTSProviderError,
    TTSProviderNotConfiguredError,
    TTSResourceError,
    TTSValidationError,
    categorize_error,
    is_retryable_error,
)
from .tts_resource_manager import get_resource_manager
from .tts_validation import validate_text_input, validate_tts_request
from .utils import estimate_max_new_tokens, parse_bool

#
#######################################################################################################################
#
# Enhanced TTS Service with Adapter Pattern

_TTS_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    TTSError,
    TTSAudioQualityError,
    TTSFallbackExhaustedError,
    TTSGenerationError,
    TTSInvalidVoiceReferenceError,
    TTSProviderError,
    TTSProviderNotConfiguredError,
    TTSResourceError,
    TTSValidationError,
    CircuitOpenError,
)
_OMNIVOICE_ALIAS_VALUES = {"omnivoice", "omni-voice", "omni_voice"}
_OMNIVOICE_INSTRUCT_KEYS = ("instruct", "voice_design", "voice_description")
_OMNIVOICE_SEMANTIC_GENERATION_KEYS = {
    "num_step",
    "guidance_scale",
    "denoise",
    "t_shift",
    "position_temperature",
    "class_temperature",
    "layer_penalty_factor",
    "postprocess_output",
    "preprocess_prompt",
    "audio_chunk_duration",
    "audio_chunk_threshold",
}

class TTSServiceV2:
    """
    Enhanced TTS service that uses the adapter pattern for multiple providers.
    Provides intelligent provider selection and fallback capabilities.
    """

    def __init__(self, factory: Optional[TTSAdapterFactory] = None, circuit_manager: Optional[CircuitBreakerManager] = None):
        """
        Initialize the TTS service.

        Args:
            factory: TTS adapter factory instance
            circuit_manager: Optional circuit breaker manager
        """
        # New DI-friendly factory (may be None in tests); keep legacy alias _factory
        self.factory = factory
        # Backwards-compat: some unit tests expect an internal `_factory` attribute
        self._factory: Optional[TTSAdapterFactory] = factory
        # In unit tests, get_tts_factory is patched (AsyncMock) and the tests expect
        # `_factory` to equal its return_value immediately upon construction.
        # Detect that case and use the mock's return_value directly without awaiting.
        try:
            if hasattr(get_tts_factory, "return_value"):
                # Patched with mock/AsyncMock
                self._factory = get_tts_factory.return_value  # type: ignore[assignment]
            else:
                # Legacy behavior: only call if it's a regular (non-async) function
                if not asyncio.iscoroutinefunction(get_tts_factory):
                    maybe_factory = get_tts_factory()  # type: ignore[func-returns-value]
                    if not asyncio.iscoroutine(maybe_factory):
                        self._factory = maybe_factory  # type: ignore[assignment]
        except _TTS_NONCRITICAL_EXCEPTIONS:
            # Safe to ignore - tests may override `_factory` directly
            pass
        self.circuit_manager = circuit_manager
        # Limit concurrent generations; honor config if available
        max_concurrent = 4
        # Default to structured HTTP errors instead of embedding error bytes in audio
        stream_errors_as_audio = False
        env_stream_override = os.getenv("TTS_STREAM_ERRORS_AS_AUDIO")
        if env_stream_override is not None:
            normalized = env_stream_override.strip().lower()
            stream_errors_as_audio = normalized not in {"0", "false", "no", "off"}
        try:
            if self.factory and hasattr(self.factory, "registry") and hasattr(self.factory.registry, "config"):
                perf_cfg = self.factory.registry.config.get("performance", {})  # type: ignore[attr-defined]
                # Support Pydantic models or dictionaries
                if not isinstance(perf_cfg, dict):
                    if hasattr(perf_cfg, "model_dump"):  # Pydantic v2
                        perf_cfg = perf_cfg.model_dump()  # type: ignore[assignment]
                    elif hasattr(perf_cfg, "dict"):
                        perf_cfg = perf_cfg.dict()  # type: ignore[assignment]
                if isinstance(perf_cfg, dict):
                    mcg = perf_cfg.get("max_concurrent_generations", max_concurrent)
                    try:
                        max_concurrent = int(mcg)
                        if max_concurrent <= 0:
                            max_concurrent = 1
                    except _TTS_NONCRITICAL_EXCEPTIONS:
                        max_concurrent = 4
                    if env_stream_override is None and "stream_errors_as_audio" in perf_cfg:
                        try:
                            from .utils import parse_bool
                            # When config entry is missing or invalid, default to False
                            # so errors propagate as HTTP errors instead of audio bytes.
                            stream_errors_as_audio = parse_bool(
                                perf_cfg.get("stream_errors_as_audio"),
                                default=False,
                            )
                        except _TTS_NONCRITICAL_EXCEPTIONS:
                            stream_errors_as_audio = bool(perf_cfg.get("stream_errors_as_audio"))
        except _TTS_NONCRITICAL_EXCEPTIONS:
            # Fallback to default on any parsing/config errors
            max_concurrent = 4
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._stream_errors_as_audio = stream_errors_as_audio
        if self._stream_errors_as_audio:
            logger.warning(
                "TTSServiceV2 initialized with stream_errors_as_audio=True. "
                "Errors will be embedded as audio bytes; this mode is not "
                "recommended for production deployments."
            )
        self._active_request_counts: dict[str, int] = {}
        self._active_requests_lock = asyncio.Lock()
        self._provider_semaphores: dict[str, asyncio.Semaphore] = {}
        self._provider_limits: dict[str, int] = {}
        self._closing = False
        self._omnivoice_supervisor: Optional[OmniVoiceSidecarSupervisor] = None

        # Initialize metrics
        self.metrics = get_metrics_registry()
        self._register_tts_metrics()

    def _get_validation_config(self) -> Optional[dict[str, Any]]:
        """Return config dictionary for validation (best-effort)."""
        try:
            registry = None
            if self.factory:
                registry = getattr(self.factory, "registry", None)
            if registry is None and getattr(self, "_factory", None) is not None:
                registry = getattr(self._factory, "registry", None)
            cfg = getattr(registry, "config", None) if registry else None
            if isinstance(cfg, dict):
                return cfg
        except _TTS_NONCRITICAL_EXCEPTIONS:
            return None

    @staticmethod
    def _normalize_language_code(language: Optional[Any]) -> Optional[str]:
        """Normalize incoming language codes (e.g. pt-BR -> pt, en_US -> en)."""
        if language is None:
            return None
        try:
            normalized = str(language).strip()
        except _TTS_NONCRITICAL_EXCEPTIONS:
            return None
        if not normalized:
            return None
        normalized = normalized.replace("_", "-").lower()
        if "-" in normalized:
            base = normalized.split("-", 1)[0].strip()
            if base:
                return base
        return normalized

    @staticmethod
    def _extract_observability_value(extras: Any, keys: tuple[str, ...]) -> Optional[str]:
        if not isinstance(extras, dict):
            return None
        for key in keys:
            value = extras.get(key)
            if value is None:
                continue
            try:
                parsed = str(value).strip()
            except _TTS_NONCRITICAL_EXCEPTIONS:
                continue
            if parsed:
                return parsed
        return None

    def _resolve_observability_context(
        self,
        request: OpenAISpeechRequest,
        explicit_request_id: Optional[str] = None,
    ) -> tuple[str, Optional[str]]:
        extras = getattr(request, "extra_params", None)
        request_id = None
        if explicit_request_id is not None:
            with suppress(_TTS_NONCRITICAL_EXCEPTIONS):
                request_id = str(explicit_request_id).strip()
        if not request_id:
            request_id = self._extract_observability_value(
                extras,
                ("request_id", "x_request_id", "x-request-id"),
            )
        if not request_id:
            request_id = new_request_id()
        correlation_id = self._extract_observability_value(
            extras,
            ("correlation_id", "x_correlation_id", "x-correlation-id"),
        )
        return request_id, correlation_id

    def _get_provider_runtime_config(self, provider_key: str) -> dict[str, Any]:
        """Best-effort provider config lookup for runtime materialization settings."""
        config_source = None
        for factory in (self.factory, self._factory):
            registry = getattr(factory, "registry", None) if factory is not None else None
            if registry is None:
                continue
            config_source = getattr(registry, "config", None)
            if config_source is not None:
                break

        if config_source is None:
            return {}

        if hasattr(config_source, "model_dump"):
            config_source = config_source.model_dump()
        elif hasattr(config_source, "dict"):
            config_source = config_source.dict()

        providers_cfg = None
        if isinstance(config_source, dict):
            providers_cfg = config_source.get("providers")
        else:
            providers_cfg = getattr(config_source, "providers", None)

        if providers_cfg is None:
            return {}

        provider_cfg = providers_cfg.get(provider_key) if isinstance(providers_cfg, dict) else getattr(providers_cfg, provider_key, None)
        if provider_cfg is None:
            return {}
        if hasattr(provider_cfg, "model_dump"):
            provider_cfg = provider_cfg.model_dump()
        elif hasattr(provider_cfg, "dict"):
            provider_cfg = provider_cfg.dict()
        return provider_cfg if isinstance(provider_cfg, dict) else {}

    @staticmethod
    def _extract_reference_text_from_extras(extras: dict[str, Any]) -> Optional[str]:
        for key in (
            "pocket_tts_cpp_reference_text",
            "reference_text",
            "ref_text",
            "voice_reference_text",
        ):
            value = extras.get(key)
            if value is None:
                continue
            try:
                parsed = str(value).strip()
            except _TTS_NONCRITICAL_EXCEPTIONS:
                continue
            if parsed:
                return parsed
        return None

    async def _apply_pocket_tts_cpp_runtime_materialization(
        self,
        request: TTSRequest,
        *,
        user_id: int,
        voice_manager: Any,
        metadata: Optional[Any],
    ) -> None:
        extras = request.extra_params or {}
        if not isinstance(extras, dict):
            extras = {}

        provider_cfg = self._get_provider_runtime_config("pocket_tts_cpp")

        voice_path = None
        is_transient = False
        voice_name = request.voice or ""
        if isinstance(voice_name, str) and voice_name.startswith("custom:"):
            voice_id = voice_name.split("custom:", 1)[-1].strip()
            if voice_id:
                voice_path = await materialize_custom_voice_reference(
                    voice_manager=voice_manager,
                    user_id=user_id,
                    voice_id=voice_id,
                    cache_max_bytes=provider_cfg.get("cache_max_bytes_per_user"),
                )
        elif request.voice_reference:
            voice_path, is_transient = await materialize_direct_voice_reference(
                voice_manager=voice_manager,
                user_id=user_id,
                voice_reference=request.voice_reference,
                persist_direct_voice_references=bool(
                    provider_cfg.get("persist_direct_voice_references", False)
                ),
                cache_max_bytes=provider_cfg.get("cache_max_bytes_per_user"),
            )

        if voice_path is None:
            request.extra_params = extras
            return

        reference_text = self._extract_reference_text_from_extras(extras)
        if reference_text is None and metadata is not None:
            reference_text = getattr(metadata, "reference_text", None)

        trust_token = register_provider_managed_voice_path(voice_path)
        runtime_dir = get_runtime_dir(voice_manager=voice_manager, user_id=user_id)
        prune_materialized_voice_cache(
            runtime_dir,
            cache_ttl_hours=provider_cfg.get("cache_ttl_hours"),
            cache_max_bytes=provider_cfg.get("cache_max_bytes_per_user"),
            protected_paths={voice_path},
        )
        extras["pocket_tts_cpp_voice_path"] = str(voice_path)
        extras[PROVIDER_MANAGED_VOICE_TOKEN_KEY] = trust_token
        if reference_text:
            extras["pocket_tts_cpp_reference_text"] = reference_text
        if is_transient:
            extras["_pocket_tts_cpp_transient_voice_path"] = str(voice_path)
        else:
            extras.pop("_pocket_tts_cpp_transient_voice_path", None)
        request.extra_params = extras

    def _cleanup_transient_pocket_tts_cpp_voice_path(
        self,
        request: TTSRequest,
    ) -> None:
        extras = getattr(request, "extra_params", None)
        if not isinstance(extras, dict):
            return
        managed_voice_path_raw = extras.get("pocket_tts_cpp_voice_path")
        trust_token = extras.pop(PROVIDER_MANAGED_VOICE_TOKEN_KEY, None)
        revoke_provider_managed_voice_token(
            trust_token,
            Path(str(managed_voice_path_raw)) if managed_voice_path_raw else None,
        )
        raw_path = extras.pop("_pocket_tts_cpp_transient_voice_path", None)
        if not raw_path:
            raw_path = None
        if raw_path:
            cleanup_transient_voice_reference(Path(str(raw_path)), True)

    async def _close_response_audio_stream(self, response: Optional[TTSResponse]) -> None:
        if response is None:
            return
        audio_stream = getattr(response, "audio_stream", None)
        if audio_stream and hasattr(audio_stream, "aclose"):
            with suppress(_TTS_NONCRITICAL_EXCEPTIONS):
                await audio_stream.aclose()

    async def _prepare_generate_speech_request(
        self,
        *,
        request: OpenAISpeechRequest,
        tts_request: TTSRequest,
        provider: Optional[str],
        provider_hint: Optional[str],
        provider_overrides: Optional[dict[str, Any]],
        fallback: bool,
        user_id: Optional[int],
    ) -> tuple[TTSAdapter, str, TTSRequest]:
        """Resolve provider-managed request state before execution begins."""
        prepared = False
        try:
            self._apply_token_defaults(tts_request)
            # Run a generic validation pass first so provider-specific requirements
            # can be satisfied by stored custom voice enrichment before the final check.
            validate_tts_request(
                tts_request,
                provider=None,
                config=self._get_validation_config(),
            )
            await self._apply_custom_voice_reference(tts_request, user_id, provider_hint)

            adapter = await self._get_adapter(request.model, provider, overrides=provider_overrides)
            if not adapter and fallback:
                adapter = await self._get_fallback_adapter(tts_request)
            if not adapter:
                raise TTSProviderNotConfiguredError(
                    f"No TTS adapter available for model '{request.model}'",
                    provider=provider,
                )

            provider_key = self._resolve_provider_key(adapter)
            try:
                resource_mgr = await get_resource_manager()
                resource_mgr.touch_model(provider_key, getattr(tts_request, "model", None))
            except _TTS_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning(
                    "Non-critical touch_model failure for provider {} model {}: {}",
                    provider_key,
                    getattr(tts_request, "model", None),
                    exc,
                )

            request_for_provider = self._maybe_sanitize_request(tts_request, provider_key)
            validate_tts_request(
                request_for_provider,
                provider=provider_key,
                config=self._get_validation_config(),
            )
            prepared = True
            return adapter, provider_key, request_for_provider
        finally:
            if not prepared:
                self._cleanup_transient_pocket_tts_cpp_voice_path(tts_request)

    def _get_tts_request_observability(
        self,
        request: TTSRequest,
    ) -> tuple[Optional[str], Optional[str]]:
        extras = request.extra_params if isinstance(request.extra_params, dict) else {}
        request_id = self._extract_observability_value(extras, ("request_id",))
        correlation_id = self._extract_observability_value(extras, ("correlation_id",))
        return request_id, correlation_id

    def _attach_response_metadata(
        self,
        target: Any,
        response: TTSResponse,
        provider_key: str,
        request_for_provider: TTSRequest,
    ) -> None:
        metadata: dict[str, Any] = {}
        if isinstance(response.metadata, dict):
            metadata.update(response.metadata)
        request_id, correlation_id = self._get_tts_request_observability(request_for_provider)

        if metadata.get("provider") is None:
            metadata["provider"] = response.provider or provider_key
        if metadata.get("model") is None:
            metadata["model"] = response.model or request_for_provider.model
        if metadata.get("voice") is None:
            metadata["voice"] = response.voice_used or request_for_provider.voice
        if metadata.get("format") is None:
            fmt_val = None
            try:
                fmt_val = response.format.value  # type: ignore[attr-defined]
            except _TTS_NONCRITICAL_EXCEPTIONS:
                try:
                    fmt_val = str(response.format)
                except _TTS_NONCRITICAL_EXCEPTIONS:
                    fmt_val = None
            if not fmt_val:
                try:
                    fmt_val = request_for_provider.format.value
                except _TTS_NONCRITICAL_EXCEPTIONS:
                    fmt_val = None
            if fmt_val:
                metadata["format"] = fmt_val
        if metadata.get("duration_seconds") is None:
            duration = response.duration_seconds if response.duration_seconds is not None else response.duration
            if duration is not None:
                with suppress(_TTS_NONCRITICAL_EXCEPTIONS):
                    metadata["duration_seconds"] = float(duration)
        if metadata.get("sample_rate") is None and response.sample_rate:
            metadata["sample_rate"] = response.sample_rate
        if metadata.get("request_id") is None and request_id:
            metadata["request_id"] = request_id
        if metadata.get("correlation_id") is None and correlation_id:
            metadata["correlation_id"] = correlation_id

        with suppress(_TTS_NONCRITICAL_EXCEPTIONS):
            target._tts_metadata = metadata
        return None

    def _get_performance_config(self) -> dict[str, Any]:
        """Return performance config dict (best-effort)."""
        cfg = self._get_validation_config()
        if isinstance(cfg, dict):
            perf = cfg.get("performance")
            if isinstance(perf, dict):
                return perf
        return {}

    def _apply_token_defaults(self, request: TTSRequest) -> None:
        """Apply max/min token defaults when not explicitly provided."""
        extras = request.extra_params if isinstance(request.extra_params, dict) else {}
        perf = self._get_performance_config()

        enabled = perf.get("token_estimation_enabled", True)
        if parse_bool(extras.get("disable_token_estimation"), default=False):
            enabled = False
        if not enabled:
            request.extra_params = extras
            return

        def _coerce_int(value: Any) -> Optional[int]:
            try:
                if value is None:
                    return None
                return int(value)
            except _TTS_NONCRITICAL_EXCEPTIONS:
                return None

        max_new = _coerce_int(extras.get("max_new_tokens"))
        if max_new is None:
            max_new = estimate_max_new_tokens(
                request.text,
                tokens_per_char=perf.get("token_estimate_per_char", 2.5),
                safety=perf.get("token_estimate_safety", 1.3),
                min_tokens=perf.get("token_estimate_min_tokens", 256),
                max_cap=perf.get("max_new_tokens_cap", 4096),
            )
            extras["max_new_tokens"] = max_new

        min_new = _coerce_int(extras.get("min_new_tokens"))
        if min_new is None:
            min_default = _coerce_int(perf.get("min_new_tokens_default", 60))
            if min_default is not None and min_default > 0:
                extras["min_new_tokens"] = min_default
                min_new = min_default

        try:
            if min_new is not None and max_new is not None and min_new > max_new:
                extras["min_new_tokens"] = max_new
        except _TTS_NONCRITICAL_EXCEPTIONS:
            pass

        request.extra_params = extras

    def _resolve_chunking_params(self, extras: dict[str, Any]) -> tuple[bool, int, int, int, int]:
        """Resolve chunking parameters from extras with conservative defaults."""
        if not isinstance(extras, dict):
            return False, 0, 0, 0, 0

        def _pick_int(keys: tuple[str, ...], default: int) -> int:
            for key in keys:
                if key in extras and extras.get(key) is not None:
                    try:
                        return int(extras.get(key))
                    except _TTS_NONCRITICAL_EXCEPTIONS:
                        continue
            return default

        enabled: Optional[bool] = None
        if "chunking_service" in extras:
            enabled = parse_bool(extras.get("chunking_service"), default=False)
        elif "chunking" in extras:
            enabled = parse_bool(extras.get("chunking"), default=False)
        elif "split_text" in extras:
            enabled = parse_bool(extras.get("split_text"), default=False)
        else:
            for key in ("chunk_size", "chunk_target_chars", "chunk_max_chars", "chunk_min_chars", "chunk_crossfade_ms"):
                if key in extras:
                    enabled = True
                    break

        if not enabled:
            return False, 0, 0, 0, 0

        target = _pick_int(("chunk_target_chars", "chunk_target", "chunk_chars_target", "chunk_size"), 120)
        max_chars = _pick_int(("chunk_max_chars", "chunk_max", "chunk_chars_max", "chunk_size"), 150)
        min_chars = _pick_int(("chunk_min_chars", "chunk_min", "chunk_chars_min"), 50)
        crossfade_ms = _pick_int(("chunk_crossfade_ms", "crossfade_ms"), 50)
        if max_chars <= 0:
            return False, 0, 0, 0, 0
        if target > max_chars:
            target = max_chars
        if min_chars > max_chars:
            min_chars = max_chars
        return True, target, max_chars, min_chars, crossfade_ms

    def _resolve_audio_check_params(
        self,
        extras: dict[str, Any],
        *,
        default_enabled: bool = True,
        default_per_chunk: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(extras, dict):
            extras = {}

        def _pick_float(keys: tuple[str, ...], default: float) -> float:
            for key in keys:
                if key in extras and extras.get(key) is not None:
                    try:
                        return float(extras.get(key))
                    except _TTS_NONCRITICAL_EXCEPTIONS:
                        continue
            return default

        def _pick_int(keys: tuple[str, ...], default: int) -> int:
            for key in keys:
                if key in extras and extras.get(key) is not None:
                    try:
                        return int(extras.get(key))
                    except _TTS_NONCRITICAL_EXCEPTIONS:
                        continue
            return default

        def _pick_bool(keys: tuple[str, ...], default: bool) -> bool:
            for key in keys:
                if key in extras:
                    return parse_bool(extras.get(key), default=default)
            return default

        return {
            "enabled": _pick_bool(("audio_checks", "audio_quality_checks"), default_enabled),
            "strict": _pick_bool(("audio_checks_strict", "audio_quality_strict"), False),
            "per_chunk": _pick_bool(("audio_checks_per_chunk",), default_per_chunk),
            "trim_trailing_silence": _pick_bool(
                ("audio_trim_trailing_silence", "trim_trailing_silence"),
                False,
            ),
            "min_rms": _pick_float(("audio_min_rms", "min_rms"), 0.001),
            "min_peak": _pick_float(("audio_min_peak", "min_peak"), 0.02),
            "silence_threshold": _pick_float(("audio_silence_threshold", "silence_threshold"), 0.01),
            "trailing_silence_ms": _pick_int(
                ("audio_trailing_silence_ms", "trailing_silence_ms", "silence_tail_ms"),
                800,
            ),
            "expected_chars_per_sec": _pick_float(
                ("audio_expected_chars_per_sec", "expected_chars_per_sec", "chars_per_sec"),
                15.0,
            ),
            "min_duration_ratio": _pick_float(
                ("audio_min_duration_ratio", "min_duration_ratio"),
                0.5,
            ),
            "min_duration_seconds": _pick_float(
                ("audio_min_duration_seconds", "min_duration_seconds"),
                0.4,
            ),
            "min_text_length": _pick_int(
                ("audio_min_text_length", "min_text_length"),
                40,
            ),
        }

    def _build_silence_for_text(
        self,
        text: str,
        sample_rate: int,
        expected_chars_per_sec: float,
        min_duration_seconds: float,
    ) -> np.ndarray:
        try:
            sample_rate = int(sample_rate)
        except _TTS_NONCRITICAL_EXCEPTIONS:
            sample_rate = 24000
        if sample_rate <= 0:
            sample_rate = 24000
        duration = max(min_duration_seconds, 0.0)
        if expected_chars_per_sec > 0:
            duration = max(duration, len(text or "") / float(expected_chars_per_sec))
        samples = int(sample_rate * duration)
        if samples <= 0:
            return np.zeros(0, dtype=np.int16)
        return np.zeros(samples, dtype=np.int16)

    def _convert_pcm_to_format(
        self,
        audio: np.ndarray,
        *,
        target_format: AudioFormat,
        sample_rate: int,
    ) -> bytes:
        normalizer = AudioNormalizer()
        writer = StreamingAudioWriter(
            format=target_format.value,
            sample_rate=sample_rate,
            channels=1,
        )
        try:
            if audio.dtype != np.int16:
                audio = normalizer.normalize(audio, target_dtype=np.int16)
            chunk_bytes = writer.write_chunk(audio) or b""
            final_bytes = writer.write_chunk(finalize=True) or b""
            if target_format == AudioFormat.PCM:
                return chunk_bytes
            return chunk_bytes + final_bytes
        finally:
            writer.close()

    def _resolve_segment_retry_params(self, extras: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(extras, dict):
            extras = {}

        def _pick_int(keys: tuple[str, ...], default: int) -> int:
            for key in keys:
                if key in extras and extras.get(key) is not None:
                    try:
                        return int(extras.get(key))
                    except _TTS_NONCRITICAL_EXCEPTIONS:
                        continue
            return default

        def _pick_float(keys: tuple[str, ...], default: float) -> float:
            for key in keys:
                if key in extras and extras.get(key) is not None:
                    try:
                        return float(extras.get(key))
                    except _TTS_NONCRITICAL_EXCEPTIONS:
                        continue
            return default

        def _pick_bool(keys: tuple[str, ...], default: bool) -> bool:
            for key in keys:
                if key in extras:
                    return parse_bool(extras.get(key), default=default)
            return default

        return {
            "max_retries": _pick_int(("segment_retry_max", "segment_retries"), 2),
            "backoff_ms": _pick_int(("segment_retry_backoff_ms", "segment_backoff_ms"), 250),
            "backoff_factor": _pick_float(("segment_retry_backoff_factor", "segment_backoff_factor"), 2.0),
            "max_backoff_ms": _pick_int(("segment_retry_max_backoff_ms", "segment_backoff_max_ms"), 4000),
            "allow_partial": _pick_bool(("segment_allow_partial", "chunk_allow_partial"), True),
            "silence_on_fail": _pick_bool(("segment_silence_on_fail", "chunk_silence_on_fail"), True),
        }

    def _is_retryable_segment_error(self, error: Exception) -> bool:
        if isinstance(error, TTSError):
            return is_retryable_error(error)
        return False

    def _apply_audio_quality_checks(
        self,
        *,
        audio: np.ndarray,
        text: str,
        sample_rate: int,
        params: dict[str, Any],
        provider_key: str,
        context: str,
    ) -> tuple[np.ndarray, dict[str, float], list[str]]:
        if not params.get("enabled", True):
            return audio, {}, []

        metrics, warnings = evaluate_audio_quality(
            audio,
            sample_rate,
            text_length=len(text or ""),
            min_text_length=int(params["min_text_length"]),
            min_rms=float(params["min_rms"]),
            min_peak=float(params["min_peak"]),
            silence_threshold=float(params["silence_threshold"]),
            trailing_silence_ms=int(params["trailing_silence_ms"]),
            expected_chars_per_sec=float(params["expected_chars_per_sec"]),
            min_duration_ratio=float(params["min_duration_ratio"]),
            min_duration_seconds=float(params["min_duration_seconds"]),
        )

        if params.get("trim_trailing_silence") and params["trailing_silence_ms"] > 0:
            if metrics.get("trailing_silence_ms", 0.0) >= params["trailing_silence_ms"]:
                trimmed = trim_trailing_silence(
                    audio,
                    sample_rate,
                    threshold=float(params["silence_threshold"]),
                    min_silence_ms=int(params["trailing_silence_ms"]),
                )
                if trimmed.shape[0] < np.asarray(audio).reshape(-1).shape[0]:
                    audio = trimmed
                    metrics, warnings = evaluate_audio_quality(
                        audio,
                        sample_rate,
                        text_length=len(text or ""),
                        min_text_length=int(params["min_text_length"]),
                        min_rms=float(params["min_rms"]),
                        min_peak=float(params["min_peak"]),
                        silence_threshold=float(params["silence_threshold"]),
                        trailing_silence_ms=int(params["trailing_silence_ms"]),
                        expected_chars_per_sec=float(params["expected_chars_per_sec"]),
                        min_duration_ratio=float(params["min_duration_ratio"]),
                        min_duration_seconds=float(params["min_duration_seconds"]),
                    )

        if warnings:
            details = {
                "context": context,
                "metrics": metrics,
                "warnings": warnings,
            }
            if params.get("strict"):
                raise TTSAudioQualityError(
                    "TTS audio failed quality checks",
                    provider=provider_key,
                    details=details,
                )
            logger.warning(
                f"{provider_key}: audio checks flagged ({context}): {', '.join(warnings)}"
            )
        return audio, metrics, warnings

    def _should_service_chunk(self, request: TTSRequest, adapter: TTSAdapter) -> tuple[bool, int, int, int, int]:
        if request.stream:
            return False, 0, 0, 0, 0
        if getattr(adapter, "handles_text_chunking", False):
            return False, 0, 0, 0, 0
        extras = request.extra_params if isinstance(request.extra_params, dict) else {}
        enabled, target, max_chars, min_chars, crossfade_ms = self._resolve_chunking_params(extras)
        if not enabled:
            return False, 0, 0, 0, 0
        if not request.text:
            return False, 0, 0, 0, 0
        if len(request.text) <= max_chars:
            return False, 0, 0, 0, 0
        return True, target, max_chars, min_chars, crossfade_ms

    async def _generate_chunked_response(
        self,
        adapter: TTSAdapter,
        request: TTSRequest,
        provider_key: str,
        target_chars: int,
        max_chars: int,
        min_chars: int,
        crossfade_ms: int,
    ) -> Optional[TTSResponse]:
        extras = request.extra_params if isinstance(request.extra_params, dict) else {}
        check_params = self._resolve_audio_check_params(
            extras,
            default_enabled=True,
            default_per_chunk=True,
        )
        retry_params = self._resolve_segment_retry_params(extras)
        caps = None
        try:
            caps = getattr(adapter, "_capabilities", None)
            if caps is None or not isinstance(caps, TTSCapabilities):
                caps = await adapter.get_capabilities()
        except _TTS_NONCRITICAL_EXCEPTIONS:
            caps = None
        if not isinstance(caps, TTSCapabilities) or AudioFormat.PCM not in caps.supported_formats:
            logger.debug(
                f"{provider_key}: chunking requested but PCM not supported; skipping service-level chunking"
            )
            return None
        try:
            max_len = getattr(caps, "max_text_length", None)
            if isinstance(max_len, int) and max_len > 0:
                max_chars = min(max_chars, max_len)
        except _TTS_NONCRITICAL_EXCEPTIONS:
            pass
        if max_chars <= 0:
            return None
        if target_chars > max_chars:
            target_chars = max_chars
        if min_chars > max_chars:
            min_chars = max_chars

        chunks = split_text_into_chunks(
            request.text,
            target_chars=target_chars,
            max_chars=max_chars,
            min_chars=min_chars,
        )
        if len(chunks) <= 1:
            return None

        chunk_extras = dict(extras)
        for key in (
            "chunking_service",
            "chunking",
            "chunk_target_chars",
            "chunk_target",
            "chunk_chars_target",
            "chunk_max_chars",
            "chunk_max",
            "chunk_chars_max",
            "chunk_min_chars",
            "chunk_min",
            "chunk_chars_min",
            "chunk_crossfade_ms",
            "crossfade_ms",
        ):
            chunk_extras.pop(key, None)

        base_request = replace(request, stream=False, format=AudioFormat.PCM)
        base_request.extra_params = chunk_extras

        audio_parts: list[np.ndarray] = []
        sample_rate: Optional[int] = None
        quality_events: list[dict[str, Any]] = []
        segment_events: list[dict[str, Any]] = []
        last_error: Optional[Exception] = None
        fallback_sample_rate = getattr(adapter, "sample_rate", 24000)
        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_request = replace(base_request, text=chunk_text)
            attempts = 0
            last_error = None
            max_attempts = max(1, int(retry_params["max_retries"]))
            backoff_ms = max(0, int(retry_params["backoff_ms"]))
            backoff_factor = float(retry_params["backoff_factor"]) if retry_params["backoff_factor"] else 1.0
            max_backoff_ms = max(backoff_ms, int(retry_params["max_backoff_ms"]))

            while True:
                attempts += 1
                try:
                    response = await adapter.generate(chunk_request)
                    pcm_bytes = response.audio_data or response.audio_content
                    if pcm_bytes is None and response.audio_stream is not None:
                        collected = bytearray()
                        async for data in response.audio_stream:
                            if data:
                                collected.extend(data)
                        pcm_bytes = bytes(collected)
                    if not pcm_bytes:
                        raise TTSGenerationError(
                            f"{provider_key} returned empty audio for chunked request",
                            provider=provider_key,
                        )
                    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
                    if sample_rate is None:
                        sample_rate = response.sample_rate or getattr(adapter, "sample_rate", None)
                    if check_params.get("enabled") and check_params.get("per_chunk"):
                        pcm, metrics, warnings = self._apply_audio_quality_checks(
                            audio=pcm,
                            text=chunk_text,
                            sample_rate=sample_rate or fallback_sample_rate,
                            params=check_params,
                            provider_key=provider_key,
                            context="chunk",
                        )
                        if warnings:
                            quality_events.append(
                                {
                                    "context": "chunk",
                                    "chunk_index": len(audio_parts),
                                    "warnings": warnings,
                                    "metrics": metrics,
                                }
                            )
                    audio_parts.append(pcm)
                    segment_events.append(
                        {
                            "index": chunk_idx,
                            "status": "success",
                            "attempts": attempts,
                        }
                    )
                    break
                except _TTS_NONCRITICAL_EXCEPTIONS as exc:
                    last_error = exc
                    retryable = self._is_retryable_segment_error(exc)
                    if attempts >= max_attempts or not retryable:
                        segment_events.append(
                            {
                                "index": chunk_idx,
                                "status": "failed",
                                "attempts": attempts,
                                "error": str(exc),
                                "error_type": exc.__class__.__name__,
                                "retryable": retryable,
                                "details": getattr(exc, "details", None),
                            }
                        )
                        break
                    delay_ms = int(backoff_ms * (backoff_factor ** (attempts - 1)))
                    delay_ms = min(delay_ms, max_backoff_ms)
                    if delay_ms > 0:
                        await asyncio.sleep(delay_ms / 1000.0)

            if last_error is not None and segment_events[-1]["status"] == "failed":
                if not retry_params.get("allow_partial", True):
                    raise last_error
                if retry_params.get("silence_on_fail", True):
                    silence_rate = sample_rate or fallback_sample_rate
                    silence = self._build_silence_for_text(
                        chunk_text,
                        silence_rate,
                        expected_chars_per_sec=float(check_params["expected_chars_per_sec"]),
                        min_duration_seconds=float(check_params["min_duration_seconds"]),
                    )
                    if silence.size:
                        audio_parts.append(silence)
                # Continue to next chunk; partial success is allowed if at least one segment succeeded.
                continue

        if not audio_parts:
            if last_error:
                raise last_error
            return None
        if sample_rate is None:
            sample_rate = fallback_sample_rate

        merged = audio_parts[0]
        for part in audio_parts[1:]:
            merged = crossfade_audio(
                merged,
                part,
                sample_rate=sample_rate,
                crossfade_ms=crossfade_ms,
            )

        if check_params.get("enabled"):
            merged, metrics, warnings = self._apply_audio_quality_checks(
                audio=merged,
                text=request.text,
                sample_rate=sample_rate,
                params=check_params,
                provider_key=provider_key,
                context="merged",
            )
            if warnings:
                quality_events.append(
                    {
                        "context": "merged",
                        "warnings": warnings,
                        "metrics": metrics,
                    }
                )

        audio_bytes: Optional[bytes] = None
        try:
            if hasattr(adapter, "convert_audio_format"):
                maybe = adapter.convert_audio_format(  # type: ignore[call-arg]
                    merged,
                    source_format=AudioFormat.PCM,
                    target_format=request.format,
                    sample_rate=sample_rate,
                )
                if asyncio.iscoroutine(maybe):
                    audio_bytes = await maybe
                elif isinstance(maybe, (bytes, bytearray)):
                    audio_bytes = bytes(maybe)
        except _TTS_NONCRITICAL_EXCEPTIONS:
            audio_bytes = None
        if audio_bytes is None:
            audio_bytes = self._convert_pcm_to_format(
                merged,
                target_format=request.format,
                sample_rate=sample_rate,
            )

        return TTSResponse(
            audio_data=audio_bytes,
            format=request.format,
            sample_rate=sample_rate,
            provider=provider_key,
            model=request.model,
            metadata={
                "chunked": True,
                "chunk_count": len(chunks),
                "chunk_crossfade_ms": crossfade_ms,
                "audio_quality_warnings": quality_events,
                "segments": segment_events,
                "partial": any(seg.get("status") == "failed" for seg in segment_events),
            },
        )

    def _get_provider_concurrency_limit(self, provider_key: str) -> Optional[int]:
        """Resolve provider-specific concurrency limit from config (if set)."""
        cfg = self._get_validation_config()
        if not isinstance(cfg, dict):
            return None
        providers_cfg = cfg.get("providers")
        if not isinstance(providers_cfg, dict):
            return None
        provider_cfg = providers_cfg.get(provider_key)
        if not isinstance(provider_cfg, dict):
            return None
        raw_limit = provider_cfg.get("max_concurrent_generations")
        if raw_limit is None:
            return None
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            return None
        if limit <= 0:
            return None
        return limit

    @asynccontextmanager
    async def _provider_concurrency_guard(self, provider_key: str):
        """Optional provider-specific semaphore guard."""
        limit = self._get_provider_concurrency_limit(provider_key)
        if limit is None:
            yield
            return
        current_limit = self._provider_limits.get(provider_key)
        semaphore = self._provider_semaphores.get(provider_key)
        if semaphore is None or current_limit != limit:
            semaphore = asyncio.Semaphore(limit)
            self._provider_semaphores[provider_key] = semaphore
            self._provider_limits[provider_key] = limit
        async with semaphore:
            yield

    # ---------------------------------------------------------------------------------
    # Backwards-compatibility methods expected by older unit tests
    # ---------------------------------------------------------------------------------
    async def _ensure_factory(self) -> TTSAdapterFactory:
        """
        Ensure a usable adapter factory is available.
        Favors injected or legacy `_factory` references before creating the singleton.
        """
        if self.factory is not None:
            return self.factory

        if self._factory is not None:
            self.factory = self._factory  # type: ignore[assignment]
            return self.factory  # type: ignore[return-value]

        factory = await get_tts_factory()
        self.factory = factory
        self._factory = factory
        return factory

    async def shutdown(self) -> None:
        """Gracefully close any underlying factory/adapters (best-effort)."""
        self._closing = True
        supervisor = self._omnivoice_supervisor
        if supervisor is not None:
            with suppress(_TTS_NONCRITICAL_EXCEPTIONS):
                supervisor.mark_closing()
            shutdown_supervisor = getattr(supervisor, "shutdown", None)
            if callable(shutdown_supervisor):
                with suppress(_TTS_NONCRITICAL_EXCEPTIONS):
                    maybe_stop = shutdown_supervisor()
                    if asyncio.iscoroutine(maybe_stop):
                        await maybe_stop
            else:
                stop_process = getattr(supervisor, "_stop_process", None)
                if callable(stop_process):
                    with suppress(_TTS_NONCRITICAL_EXCEPTIONS):
                        maybe_stop = stop_process()
                        if asyncio.iscoroutine(maybe_stop):
                            await maybe_stop

        if self.factory and hasattr(self.factory, "close"):
            with suppress(_TTS_NONCRITICAL_EXCEPTIONS):
                maybe = self.factory.close()  # type: ignore[attr-defined]
                if asyncio.iscoroutine(maybe):
                    await maybe  # type: ignore[func-returns-value]

        # Some tests set/patch `_factory` only
        if self._factory and self._factory is not self.factory and hasattr(self._factory, "close"):
            with suppress(_TTS_NONCRITICAL_EXCEPTIONS):
                maybe2 = self._factory.close()  # type: ignore[attr-defined]
                if asyncio.iscoroutine(maybe2):
                    await maybe2  # type: ignore[func-returns-value]

        self._omnivoice_supervisor = None
        self.factory = None
        self._factory = None

    async def generate(self, request: TTSRequest) -> TTSResponse:
        """Legacy synchronous-style generation wrapper expected by unit tests."""
        provider = getattr(request, "provider", None) or getattr(self, "_default_provider", "openai")
        adapter = None
        if hasattr(self, "_factory") and self._factory is not None:
            try:
                # Many tests patch `_factory.get_adapter(provider)`
                adapter = self._factory.get_adapter(provider)  # type: ignore[attr-defined]
            except _TTS_NONCRITICAL_EXCEPTIONS:
                adapter = None
        if adapter is None and self.factory is not None:
            # Try to resolve via new factory/registry by provider enum name
            try:
                prov_enum = TTSAdapterRegistry.resolve_provider(provider)
                if prov_enum is not None:
                    adapter = await self.factory.registry.get_adapter(prov_enum)  # type: ignore[union-attr]
            except _TTS_NONCRITICAL_EXCEPTIONS:
                adapter = None
        if adapter is None:
            raise TTSProviderNotConfiguredError(f"Provider not found: {provider}")

        # Optional resource check hook expected by tests
        try:
            resource_mgr = await get_resource_manager()
            with suppress(_TTS_NONCRITICAL_EXCEPTIONS):
                resource_mgr.touch_model(provider, getattr(request, "model", None))
            try:
                ok = await resource_mgr.check_resources()
            except TypeError:
                # Some mocks are non-async
                ok = resource_mgr.check_resources()
            if not ok:
                raise TTSResourceError("Insufficient resources")
        except _TTS_NONCRITICAL_EXCEPTIONS:
            # Ignore resource check errors in legacy path
            pass

        try:
            return await adapter.generate(request)  # type: ignore[union-attr]
        finally:
            self._cleanup_transient_pocket_tts_cpp_voice_path(request)

    async def generate_stream(self, request: TTSRequest) -> AsyncGenerator[bytes, None]:
        """Legacy streaming wrapper expected by unit tests."""
        provider = getattr(request, "provider", None) or getattr(self, "_default_provider", "openai")
        adapter = None
        if hasattr(self, "_factory") and self._factory is not None:
            try:
                adapter = self._factory.get_adapter(provider)  # type: ignore[attr-defined]
            except _TTS_NONCRITICAL_EXCEPTIONS:
                adapter = None
        if adapter is None and self.factory is not None:
            try:
                prov_enum = TTSAdapterRegistry.resolve_provider(provider)
                if prov_enum is not None:
                    adapter = await self.factory.registry.get_adapter(prov_enum)  # type: ignore[union-attr]
            except _TTS_NONCRITICAL_EXCEPTIONS:
                adapter = None
        if adapter is None:
            raise TTSProviderNotConfiguredError(f"Provider not found: {provider}")

        try:
            resource_mgr = await get_resource_manager()
            resource_mgr.touch_model(provider, getattr(request, "model", None))
        except _TTS_NONCRITICAL_EXCEPTIONS:
            pass

        try:
            stream = await adapter.generate_stream(request)  # type: ignore[attr-defined]
            async for chunk in stream:
                yield chunk
        finally:
            self._cleanup_transient_pocket_tts_cpp_voice_path(request)

    async def list_providers(self) -> list[str]:
        """Legacy provider listing wrapper."""
        if hasattr(self, "_factory") and self._factory is not None and hasattr(self._factory, "list_available_providers"):
            return self._factory.list_available_providers()  # type: ignore[attr-defined,return-value]
        # Fallback: derive from registry
        try:
            from .adapter_registry import TTSProvider
            return [p.value for p in TTSProvider]
        except _TTS_NONCRITICAL_EXCEPTIONS:
            return []

    async def get_capabilities(self) -> dict[str, Any]:
        """
        Return capabilities for all available TTS providers.

        The structure is JSON-serializable and suitable for the
        `/api/v1/audio/providers` endpoint.
        """
        capabilities: dict[str, Any] = {}

        try:
            factory = await self._ensure_factory()
        except _TTS_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"get_capabilities: unable to acquire TTS factory: {e}")
            return capabilities

        registry = getattr(factory, "registry", None)
        if registry is None:
            return capabilities

        # Some tests inject a helper on the registry; prefer it when present
        helper = getattr(registry, "get_all_capabilities", None)
        if helper is not None:
            try:
                maybe = helper()
                raw_caps = await maybe if asyncio.iscoroutine(maybe) else maybe
                if isinstance(raw_caps, dict):
                    for key, value in raw_caps.items():
                        provider_key = getattr(key, "value", str(key))
                        capabilities[provider_key] = self._serialize_capabilities(value)
                    return capabilities
            except _TTS_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"get_capabilities: get_all_capabilities helper failed: {e}")

        # Fallback: iterate known providers and lazily materialize adapters
        try:
            from .adapter_registry import TTSProvider as _TTSProviderEnum
            providers = list(_TTSProviderEnum)
        except _TTS_NONCRITICAL_EXCEPTIONS:
            providers = []

        for prov in providers:
            try:
                adapter = await registry.get_adapter(prov)  # type: ignore[union-attr]
            except _TTS_NONCRITICAL_EXCEPTIONS:
                adapter = None
            if not adapter or not getattr(adapter, "capabilities", None):
                continue
            capabilities[prov.value] = self._serialize_capabilities(adapter.capabilities)

        return capabilities

    async def list_voices(self) -> dict[str, list[dict[str, Any]]]:
        """
        Return a mapping of provider -> list of voice descriptors.

        Used by `/api/v1/audio/voices/catalog` and WebUI audio configuration.
        """
        voices_by_provider: dict[str, list[dict[str, Any]]] = {}
        caps = await self.get_capabilities()
        for provider, provider_caps in caps.items():
            voices: Optional[list[dict[str, Any]]] = None
            if isinstance(provider_caps, dict):
                maybe_voices = provider_caps.get("voices")
                if isinstance(maybe_voices, list):
                    voices = maybe_voices
            if voices:
                voices_by_provider[provider] = voices
        return voices_by_provider

    async def open_realtime_session(
        self,
        *,
        config: RealtimeSessionConfig,
        provider_hint: Optional[str] = None,
        route: str = "audio.stream.tts.realtime",
        user_id: Optional[int] = None,
    ) -> RealtimeSessionHandle:
        """
        Open a realtime TTS session if supported; otherwise return a buffered fallback session.

        Returns a handle containing the session, provider (if known), and optional warning.
        """
        factory = await self._ensure_factory()
        adapter: Optional[TTSAdapter] = None
        warning: Optional[str] = None
        provider_used: Optional[str] = None
        hint = (provider_hint or config.provider or "").strip().lower() or None

        # Try provider hint first
        if hint:
            try:
                adapter = await self._get_adapter(config.model, hint)
            except _TTS_NONCRITICAL_EXCEPTIONS:
                adapter = None

        # Fall back to model-based resolution
        if adapter is None and config.model:
            try:
                adapter = await factory.get_adapter_by_model(config.model)
            except _TTS_NONCRITICAL_EXCEPTIONS:
                adapter = None

        if adapter is not None:
            provider_used = self._resolve_provider_key(adapter)
            create_session = getattr(adapter, "create_realtime_session", None)
            if callable(create_session):
                try:
                    maybe_session = create_session(config)
                    session = await maybe_session if asyncio.iscoroutine(maybe_session) else maybe_session
                    if isinstance(session, RealtimeTTSSession):
                        return RealtimeSessionHandle(session=session, provider=provider_used)
                    # Duck-typed sessions are allowed; skip strict type checks.
                    return RealtimeSessionHandle(session=session, provider=provider_used)
                except _TTS_NONCRITICAL_EXCEPTIONS as exc:
                    logger.warning(f"Realtime session init failed for {provider_used}: {exc}")
                    warning = (
                        f"Realtime provider '{provider_used}' failed to initialize; "
                        "falling back to buffered synthesis."
                    )
            else:
                warning = (
                    f"Provider '{provider_used}' does not support realtime sessions; "
                    "falling back to buffered synthesis."
                )

        if warning is None and hint:
            warning = (
                f"Realtime provider '{hint}' unavailable; "
                "falling back to buffered synthesis."
            )

        session = BufferedRealtimeSession(
            tts_service=self,
            config=config,
            provider_hint=provider_used or hint,
            route=route,
            user_id=user_id,
        )
        return RealtimeSessionHandle(session=session, provider=provider_used or hint, warning=warning)

    def _serialize_capabilities(self, caps_obj: Any) -> dict[str, Any]:
        """
        Convert a TTSCapabilities instance (or compatible mapping)
        into a JSON-serializable dictionary.
        """
        # If already a mapping, normalize formats and return
        if isinstance(caps_obj, dict):
            out = dict(caps_obj)
            fmts = out.get("formats")
            if isinstance(fmts, (set, list, tuple)):
                out["formats"] = [getattr(f, "value", str(f)) for f in fmts]
            out["metadata"] = dict(out.get("metadata") or {})
            return out

        # Dataclass / object case
        try:
            from dataclasses import asdict
            data = asdict(caps_obj)
        except _TTS_NONCRITICAL_EXCEPTIONS:
            try:
                data = dict(caps_obj)
            except _TTS_NONCRITICAL_EXCEPTIONS:
                return {}

        languages = data.get("supported_languages") or []
        formats = data.get("supported_formats") or []
        voices = data.get("supported_voices") or []

        # Normalize language set and formats
        try:
            data["languages"] = sorted(languages)
        except _TTS_NONCRITICAL_EXCEPTIONS:
            data["languages"] = list(languages)
        data["formats"] = [getattr(f, "value", str(f)) for f in formats]
        data["metadata"] = dict(data.get("metadata") or {})

        # Normalize voices (VoiceInfo dataclasses) into plain dicts
        norm_voices: list[dict[str, Any]] = []
        for v in voices:
            v_dict: Optional[dict[str, Any]] = None
            try:
                from dataclasses import asdict as _asdict
                v_dict = _asdict(v)
            except _TTS_NONCRITICAL_EXCEPTIONS:
                try:
                    v_dict = dict(v)
                except _TTS_NONCRITICAL_EXCEPTIONS:
                    v_dict = None
            if v_dict is not None:
                norm_voices.append(v_dict)
        data["voices"] = norm_voices

        # Drop internal fields not needed by API callers
        for key in ("supported_languages", "supported_formats", "supported_voices"):
            data.pop(key, None)

        # Default format as string when present
        df = data.get("default_format")
        if df is not None:
            data["default_format"] = getattr(df, "value", str(df))

        return data

    def _resolve_circuit_breaker_key(self, provider_key: str, adapter: Optional[Any] = None) -> str:
        """Return the circuit-breaker key, namespacing Qwen runtimes when available."""
        if provider_key != "qwen3_tts" or adapter is None:
            return provider_key

        runtime_name = None
        runtime_getter = getattr(adapter, "_get_runtime", None)
        if callable(runtime_getter):
            try:
                runtime = runtime_getter()
                runtime_name = getattr(runtime, "runtime_name", None)
            except _TTS_NONCRITICAL_EXCEPTIONS:
                runtime_name = None

        if not runtime_name:
            runtime_resolver = getattr(adapter, "_resolve_runtime_name", None)
            if callable(runtime_resolver):
                try:
                    runtime_name = runtime_resolver()
                except _TTS_NONCRITICAL_EXCEPTIONS:
                    runtime_name = None

        if not runtime_name:
            return provider_key
        return build_qwen_runtime_breaker_key(provider_key, str(runtime_name))

    async def get_provider_info(self, provider: str) -> dict[str, Any]:
        """Legacy provider information wrapper used by tests."""
        adapter = None
        if hasattr(self, "_factory") and self._factory is not None:
            try:
                adapter = self._factory.get_adapter(provider)  # type: ignore[attr-defined]
            except _TTS_NONCRITICAL_EXCEPTIONS:
                adapter = None
        if adapter and hasattr(adapter, "get_info"):
            return adapter.get_info()  # type: ignore[attr-defined,return-value]
        # Minimal fallback info
        return {"name": provider}

    async def set_default_provider(self, provider: str) -> None:
        """Set default provider (legacy behavior for tests)."""
        self._default_provider = provider

    async def generate_with_fallback(self, request: TTSRequest, fallback_providers: Optional[list[str]] = None) -> TTSResponse:
        """Legacy helper to try primary provider then fall back to others."""
        getattr(request, "provider", None) or getattr(self, "_default_provider", "openai")
        # Try primary
        try:
            return await self.generate(request)
        except TTSGenerationError as first_err:
            # Try fallbacks in order
            if not fallback_providers:
                raise
            last_exc: Optional[Exception] = first_err
            for prov in fallback_providers:
                try:
                    req2 = request
                    req2.provider = prov
                    return await self.generate(req2)
                except _TTS_NONCRITICAL_EXCEPTIONS as e:  # keep trying
                    last_exc = e
                    continue
            # If all failed, raise the last error
            if last_exc:
                raise last_exc from first_err
            raise

    def _register_tts_metrics(self):
        """Register TTS-specific metrics"""
        # TTS request metrics
        self.metrics.register_metric(
            MetricDefinition(
                name="tts_requests_total",
                type=MetricType.COUNTER,
                description="Total number of TTS requests",
                labels=["provider", "model", "voice", "format", "status"]
            )
        )

        self.metrics.register_metric(
            MetricDefinition(
                name="tts_request_duration_seconds",
                type=MetricType.HISTOGRAM,
                description="TTS request duration in seconds",
                unit="s",
                labels=["provider", "model", "voice"],
                buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60]
            )
        )

        self.metrics.register_metric(
            MetricDefinition(
                name="tts_text_length_characters",
                type=MetricType.HISTOGRAM,
                description="Length of text processed",
                unit="characters",
                labels=["provider"],
                buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000]
            )
        )

        self.metrics.register_metric(
            MetricDefinition(
                name="tts_audio_size_bytes",
                type=MetricType.HISTOGRAM,
                description="Size of generated audio",
                unit="bytes",
                labels=["provider", "format"],
                buckets=[1024, 10240, 102400, 1048576, 10485760]  # 1KB, 10KB, 100KB, 1MB, 10MB
            )
        )

        self.metrics.register_metric(
            MetricDefinition(
                name="tts_active_requests",
                type=MetricType.GAUGE,
                description="Number of active TTS requests",
                labels=["provider"]
            )
        )

        self.metrics.register_metric(
            MetricDefinition(
                name="tts_fallback_attempts",
                type=MetricType.COUNTER,
                description="Number of fallback attempts",
                labels=["from_provider", "to_provider", "success"]
            )
        )

        self.metrics.register_metric(
            MetricDefinition(
                name="tts_fallback_outcomes_total",
                type=MetricType.COUNTER,
                description="Categorized fallback outcomes",
                labels=["from_provider", "to_provider", "outcome", "category"],
            )
        )

        self.metrics.register_metric(
            MetricDefinition(
                name="tts_ttfb_seconds",
                type=MetricType.HISTOGRAM,
                description="Time to first byte for TTS responses",
                unit="s",
                labels=["provider", "voice", "format"],
                buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10],
            )
        )

        self.metrics.register_metric(
            MetricDefinition(
                name="voice_to_voice_seconds",
                type=MetricType.HISTOGRAM,
                description="Voice-to-voice latency from end-of-speech to first synthesized audio",
                unit="s",
                labels=["provider", "route"],
                buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10, 20],
            )
        )

    async def generate_speech(
        self,
        request: OpenAISpeechRequest,
        provider: Optional[str] = None,
        fallback: bool = True,
        provider_overrides: Optional[dict[str, Any]] = None,
        voice_to_voice_start: Optional[float] = None,
        voice_to_voice_route: str = "audio.speech",
        user_id: Optional[int] = None,
        metadata_only: bool = False,
        request_id: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate speech from text using the best available provider.

        Args:
            request: OpenAI-compatible speech request
            provider: Optional specific provider to use
            fallback: Whether to fallback to other providers on failure
            metadata_only: When true, populate request metadata without streaming audio

        Yields:
            Audio chunks in the requested format
        """
        # Convert OpenAI request to unified TTSRequest
        tts_request = self._convert_request(request)
        request_id_ctx, correlation_id_ctx = self._resolve_observability_context(
            request,
            explicit_request_id=request_id,
        )
        if not isinstance(tts_request.extra_params, dict):
            tts_request.extra_params = {}
        tts_request.extra_params.setdefault("request_id", request_id_ctx)
        if correlation_id_ctx:
            tts_request.extra_params.setdefault("correlation_id", correlation_id_ctx)
        logger.info(
            "tts_request_start request_id={} correlation_id={} model={} provider_hint={} stream={} metadata_only={}",
            request_id_ctx,
            correlation_id_ctx or "none",
            request.model,
            provider or "auto",
            bool(getattr(request, "stream", False)),
            metadata_only,
        )
        factory = await self._ensure_factory()

        provider_hint: Optional[str] = None
        if provider:
            provider_hint = provider.lower()
        else:
            try:
                provider_source = None
                if factory:
                    provider_source = factory
                elif self._factory and hasattr(self._factory, "get_provider_for_model"):
                    provider_source = self._factory  # type: ignore[assignment]
                if provider_source and hasattr(provider_source, "get_provider_for_model"):
                    provider_enum = provider_source.get_provider_for_model(request.model)  # type: ignore[call-arg]
                    if inspect.isawaitable(provider_enum):
                        provider_enum = await provider_enum  # type: ignore[assignment]
                    if provider_enum:
                        provider_hint = getattr(provider_enum, "value", str(provider_enum)).lower()
            except _TTS_NONCRITICAL_EXCEPTIONS:
                provider_hint = None
        fallback = fallback and not self._is_explicit_omnivoice_request(
            request,
            provider=provider,
            provider_hint=provider_hint,
        )

        try:
            adapter, provider_key, request_for_provider = await self._prepare_generate_speech_request(
                request=request,
                tts_request=tts_request,
                provider=provider,
                provider_hint=provider_hint,
                provider_overrides=provider_overrides,
                fallback=fallback,
                user_id=user_id,
            )
        except TTSValidationError as e:
            logger.error(f"TTS request validation failed: {e}")
            if self._stream_errors_as_audio:
                yield b"ERROR: Unable to generate audio."
                return
            else:
                raise
        except TTSProviderNotConfiguredError as error:
            logger.error(str(error))
            if self._stream_errors_as_audio:
                yield f"ERROR: {str(error)}".encode()
                return
            raise error

        # Track metrics
        start_time = time.time()
        audio_size = 0
        chunks_count = 0
        released_active_slot = False
        fallback_plan: Optional[tuple[list[str], str]] = None
        voice_to_voice_recorded = False
        voice_to_voice_route_label = voice_to_voice_route or "audio.speech"
        voice_to_voice_start_ts: Optional[float] = None
        response: Optional[TTSResponse] = None
        try:
            if voice_to_voice_start is not None:
                start_val = float(voice_to_voice_start)
                if start_val > 0:
                    voice_to_voice_start_ts = start_val
        except _TTS_NONCRITICAL_EXCEPTIONS:
            voice_to_voice_start_ts = None

        if voice_to_voice_start_ts is not None:
            try:
                tts_request.voice_to_voice_start = voice_to_voice_start_ts
                tts_request.voice_to_voice_route = voice_to_voice_route_label
            except _TTS_NONCRITICAL_EXCEPTIONS:
                pass

        def _record_voice_to_voice(provider_name: str) -> None:
            nonlocal voice_to_voice_recorded
            if voice_to_voice_recorded or voice_to_voice_start_ts is None:
                return
            try:
                self.metrics.observe(
                    "voice_to_voice_seconds",
                    max(0.0, time.time() - voice_to_voice_start_ts),
                    labels={"provider": provider_name, "route": voice_to_voice_route_label},
                )
                voice_to_voice_recorded = True
            except _TTS_NONCRITICAL_EXCEPTIONS:
                pass

        await self._increment_active_requests(provider_key)

        # Generate speech with circuit breaker and comprehensive error handling
        try:
            async with self._semaphore:
                async with self._provider_concurrency_guard(provider_key):
                    logger.info(f"Generating speech with {provider_key}")

                    # Get circuit breaker if available
                    circuit_breaker = None
                    breaker_provider_key = provider_key
                    manual_stream_breaker = False
                    manual_stream_breaker_recorded = False
                    if self.circuit_manager:
                        breaker_provider_key = self._resolve_circuit_breaker_key(provider_key, adapter)
                        circuit_breaker = await self.circuit_manager.get_breaker(breaker_provider_key)

                    async def _generate_with_adapter() -> TTSResponse:
                        should_chunk, target_chars, max_chars, min_chars, crossfade_ms = self._should_service_chunk(
                            request_for_provider,
                            adapter,
                        )
                        if should_chunk:
                            chunked = await self._generate_chunked_response(
                                adapter=adapter,
                                request=request_for_provider,
                                provider_key=provider_key,
                                target_chars=target_chars,
                                max_chars=max_chars,
                                min_chars=min_chars,
                                crossfade_ms=crossfade_ms,
                            )
                            if chunked is not None:
                                return chunked
                        return await adapter.generate(request_for_provider)

                    if circuit_breaker:
                        if request_for_provider.stream and not metadata_only:
                            try:
                                await circuit_breaker.guard()
                                response = await _generate_with_adapter()
                                manual_stream_breaker = True
                            except CircuitOpenError as e:
                                logger.warning(f"Circuit open for {provider_key}: {e}")
                                if fallback:
                                    self._record_fallback_event(
                                        from_provider=provider_key,
                                        to_provider="any",
                                        success="pending",
                                        outcome="initiated",
                                        error=e,
                                        request_id=request_id_ctx,
                                    )
                                    await self._decrement_active_requests(provider_key)
                                    released_active_slot = True
                                    fallback_plan = (self._build_exclude_tokens(adapter), provider_key)
                                else:
                                    raise TTSProviderError(
                                        f"Circuit open for {provider_key}",
                                        provider=provider_key,
                                        details={"circuit_state": "open"}
                                    ) from e
                            except Exception as e:
                                await circuit_breaker.record_manual_failure(e)
                                manual_stream_breaker_recorded = True
                                raise
                        else:
                            try:
                                response = await circuit_breaker.call(_generate_with_adapter)
                            except CircuitOpenError as e:
                                logger.warning(f"Circuit open for {provider_key}: {e}")
                                if fallback:
                                    self._record_fallback_event(
                                        from_provider=provider_key,
                                        to_provider="any",
                                        success="pending",
                                        outcome="initiated",
                                        error=e,
                                        request_id=request_id_ctx,
                                    )
                                    await self._decrement_active_requests(provider_key)
                                    released_active_slot = True
                                    fallback_plan = (self._build_exclude_tokens(adapter), provider_key)
                                else:
                                    raise TTSProviderError(
                                        f"Circuit open for {provider_key}",
                                        provider=provider_key,
                                        details={"circuit_state": "open"}
                                    ) from e
                    else:
                        response = await _generate_with_adapter()

                    if fallback_plan is None and response is not None:
                        if not metadata_only:
                            response = await self._convert_response_if_needed(
                                response,
                                request_for_provider,
                                provider_key=provider_key,
                            )
                        self._attach_response_metadata(
                            request,
                            response,
                            provider_key,
                            request_for_provider,
                        )
                        if metadata_only:
                            await self._close_response_audio_stream(response)
                            with suppress(_TTS_NONCRITICAL_EXCEPTIONS):
                                self._record_tts_metrics(
                                    provider=provider_key,
                                    model=request_for_provider.model or "default",
                                    voice=request_for_provider.voice or "default",
                                    format=request_for_provider.format.value,
                                    text_length=len(request_for_provider.text),
                                    audio_size=0,
                                    duration=time.time() - start_time,
                                    success=True,
                                )
                            return
                        if response.audio_stream:
                            try:
                                async for chunk in response.audio_stream:
                                    # Record TTFB on first emitted chunk
                                    if chunks_count == 0:
                                        try:
                                            self.metrics.observe(
                                                "tts_ttfb_seconds",
                                                max(0.0, time.time() - start_time),
                                                labels={
                                                    "provider": provider_key,
                                                    "voice": request_for_provider.voice or "default",
                                                    "format": request_for_provider.format.value,
                                                },
                                            )
                                            _record_voice_to_voice(provider_key)
                                        except _TTS_NONCRITICAL_EXCEPTIONS:
                                            pass
                                    chunks_count += 1
                                    audio_size += len(chunk)
                                    yield chunk
                            except Exception as stream_error:
                                if manual_stream_breaker and circuit_breaker and not manual_stream_breaker_recorded:
                                    await circuit_breaker.record_manual_failure(stream_error)
                                    manual_stream_breaker_recorded = True
                                raise
                            else:
                                if manual_stream_breaker and circuit_breaker and not manual_stream_breaker_recorded:
                                    await circuit_breaker.record_manual_success()
                                    manual_stream_breaker_recorded = True
                            with suppress(_TTS_NONCRITICAL_EXCEPTIONS):
                                await self._maybe_store_qwen3_voice_prompt(
                                    request_for_provider, user_id, provider_key
                                )
                        elif response.audio_data:
                            chunks_count = 1
                            # Record TTFB when first audio bytes are available
                            try:
                                self.metrics.observe(
                                    "tts_ttfb_seconds",
                                    max(0.0, time.time() - start_time),
                                    labels={
                                        "provider": provider_key,
                                        "voice": request_for_provider.voice or "default",
                                        "format": request_for_provider.format.value,
                                    },
                                )
                                _record_voice_to_voice(provider_key)
                            except _TTS_NONCRITICAL_EXCEPTIONS:
                                pass
                            audio_size = len(response.audio_data)
                            yield response.audio_data
                            if manual_stream_breaker and circuit_breaker and not manual_stream_breaker_recorded:
                                await circuit_breaker.record_manual_success()
                                manual_stream_breaker_recorded = True
                            with suppress(_TTS_NONCRITICAL_EXCEPTIONS):
                                await self._maybe_store_qwen3_voice_prompt(
                                    request_for_provider, user_id, provider_key
                                )
                        else:
                            error_msg = f"No audio data returned by {provider_key}"
                            logger.error(error_msg)
                            if fallback:
                                # Record a soft failure for observability before falling back.
                                with suppress(_TTS_NONCRITICAL_EXCEPTIONS):
                                    self._record_tts_metrics(
                                        provider=provider_key,
                                        model=request_for_provider.model or "default",
                                        voice=request_for_provider.voice or "default",
                                        format=request_for_provider.format.value,
                                        text_length=len(request_for_provider.text),
                                        audio_size=audio_size,
                                        duration=max(0.0, time.time() - start_time),
                                        success=False,
                                        error=error_msg,
                                    )
                                await self._handle_provider_fallback(
                                    request_for_provider,
                                    breaker_provider_key,
                                    error_msg,
                                )
                                await self._decrement_active_requests(provider_key)
                                released_active_slot = True
                                fallback_plan = (self._build_exclude_tokens(adapter), provider_key)
                            else:
                                if self._stream_errors_as_audio:
                                    yield f"ERROR: {error_msg}".encode()
                                else:
                                    raise TTSGenerationError(error_msg, provider=provider_key)

                if fallback_plan is None:
                    self._record_tts_metrics(
                        provider=provider_key,
                        model=request_for_provider.model or "default",
                        voice=request_for_provider.voice or "default",
                        format=request_for_provider.format.value,
                        text_length=len(request_for_provider.text),
                        audio_size=audio_size,
                        duration=time.time() - start_time,
                        success=True
                    )

        except TTSError as e:
            # Handle TTS-specific errors with proper categorization
            error_msg = f"Error generating speech with {provider_key}: {str(e)}"
            logger.error(error_msg)

            if manual_stream_breaker and circuit_breaker and not manual_stream_breaker_recorded:
                await circuit_breaker.record_manual_failure(e)
                manual_stream_breaker_recorded = True

            # Record failure metrics
            self._record_tts_metrics(
                provider=provider_key,
                model=request_for_provider.model or "default",
                voice=request_for_provider.voice or "default",
                format=request_for_provider.format.value,
                text_length=len(request_for_provider.text),
                audio_size=audio_size,
                duration=time.time() - start_time,
                success=False,
                error=str(e)
            )

            # Check if error is retryable and fallback is enabled
            if fallback and is_retryable_error(e):
                logger.info(f"Attempting fallback due to retryable error: {type(e).__name__}")
                self._record_fallback_event(
                    from_provider=provider_key,
                    to_provider="any",
                    success="pending",
                    outcome="initiated",
                    error=e,
                    request_id=request_id_ctx,
                )
                await self._decrement_active_requests(provider_key)
                released_active_slot = True
                fallback_plan = (self._build_exclude_tokens(adapter), provider_key)
            else:
                # For non-recoverable errors or when fallback is disabled
                if self._stream_errors_as_audio:
                    yield f"ERROR: {error_msg}".encode()
                else:
                    raise
        except _TTS_NONCRITICAL_EXCEPTIONS as e:
            # Handle unexpected errors
            error_msg = f"Unexpected error generating speech with {provider_key}: {str(e)}"
            logger.error(error_msg, exc_info=True)

            if manual_stream_breaker and circuit_breaker and not manual_stream_breaker_recorded:
                await circuit_breaker.record_manual_failure(e)
                manual_stream_breaker_recorded = True

            # Record failure metrics
            self._record_tts_metrics(
                provider=provider_key,
                model=request_for_provider.model or "default",
                voice=request_for_provider.voice or "default",
                format=request_for_provider.format.value,
                text_length=len(request_for_provider.text),
                audio_size=audio_size,
                duration=time.time() - start_time,
                success=False,
                error=str(e)
            )

            # Wrap in TTS error for consistency
            tts_error = TTSGenerationError(
                f"Unexpected error in {provider_key}",
                provider=provider_key,
                details={"error": str(e), "error_type": type(e).__name__}
            )

            if fallback:
                logger.info("Attempting fallback due to unexpected error")
                self._record_fallback_event(
                    from_provider=provider_key,
                    to_provider="any",
                    success="pending",
                    outcome="initiated",
                    error=tts_error,
                    request_id=request_id_ctx,
                )
                await self._decrement_active_requests(provider_key)
                released_active_slot = True
                fallback_plan = (self._build_exclude_tokens(adapter), provider_key)
            else:
                if self._stream_errors_as_audio:
                    yield f"ERROR: {error_msg}".encode()
                else:
                    raise tts_error from e
        finally:
            await self._close_response_audio_stream(response)
            self._cleanup_transient_pocket_tts_cpp_voice_path(request_for_provider)
            try:
                if not released_active_slot:
                    await self._decrement_active_requests(provider_key)
            except _TTS_NONCRITICAL_EXCEPTIONS:
                pass

        if fallback_plan:
            if metadata_only:
                async for _ in self._try_fallback_providers(
                    tts_request,
                    fallback_plan[0],
                    fallback_plan[1],
                    metadata_only=True,
                    metadata_target=request,
                    user_id=user_id,
                ):
                    pass
                return
            async for chunk in self._try_fallback_providers(
                tts_request,
                fallback_plan[0],
                fallback_plan[1],
                metadata_only=False,
                metadata_target=request,
                user_id=user_id,
            ):
                yield chunk
            return

    async def _generate_with_adapter(
        self,
        adapter: TTSAdapter,
        request: TTSRequest,
        metadata_only: bool = False,
        metadata_target: Optional[Any] = None,
        user_id: Optional[int] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Generate audio with a specific adapter"""
        provider_key = self._resolve_provider_key(adapter)
        request_for_provider = request
        active_requests_incremented = False
        start_time = time.time()
        audio_size = 0
        success = False
        error_message: Optional[str] = None
        voice_metric_recorded = False
        v2v_start: Optional[float] = None
        v2v_route = getattr(request_for_provider, "voice_to_voice_route", "audio.speech") or "audio.speech"
        response: Optional[TTSResponse] = None
        # Ensure the request is valid for the concrete adapter/provider.
        try:
            if provider_key == "pocket_tts_cpp" and user_id is not None:
                extras = request.extra_params if isinstance(request.extra_params, dict) else {}
                if not (
                    isinstance(extras, dict)
                    and extras.get("pocket_tts_cpp_voice_path")
                    and extras.get(PROVIDER_MANAGED_VOICE_TOKEN_KEY)
                ):
                    await self._apply_custom_voice_reference(request, user_id, provider_key)
            request_for_provider = self._maybe_sanitize_request(request, provider_key)
            validate_tts_request(request_for_provider, provider=provider_key, config=self._get_validation_config())
        except _TTS_NONCRITICAL_EXCEPTIONS as e:
            if provider_key == "pocket_tts_cpp":
                self._cleanup_transient_pocket_tts_cpp_voice_path(request)
            logger.error(f"TTS request validation failed for provider {provider_key}: {e}")
            if self._stream_errors_as_audio:
                yield b"ERROR: Unable to generate audio."
                return
            else:
                raise

        await self._increment_active_requests(provider_key)
        active_requests_incremented = True
        start_time = time.time()
        try:
            raw = getattr(request_for_provider, "voice_to_voice_start", None)
            if raw is not None:
                parsed = float(raw)
                if parsed > 0:
                    v2v_start = parsed
        except _TTS_NONCRITICAL_EXCEPTIONS:
            v2v_start = None

        def _record_voice_to_voice() -> None:
            nonlocal voice_metric_recorded
            if voice_metric_recorded or v2v_start is None:
                return
            try:
                self.metrics.observe(
                    "voice_to_voice_seconds",
                    max(0.0, time.time() - v2v_start),
                    labels={"provider": provider_key, "route": v2v_route},
                )
                voice_metric_recorded = True
            except _TTS_NONCRITICAL_EXCEPTIONS:
                pass

        try:
            async with self._semaphore:
                response = await adapter.generate(request_for_provider)

                target = metadata_target if metadata_target is not None else request_for_provider
                self._attach_response_metadata(
                    target,
                    response,
                    provider_key,
                    request_for_provider,
                )

                if metadata_only:
                    success = True
                    if response.audio_stream and hasattr(response.audio_stream, "aclose"):
                        with suppress(_TTS_NONCRITICAL_EXCEPTIONS):
                            await response.audio_stream.aclose()
                    return

                if response.audio_stream:
                    first_emitted = False
                    async for chunk in response.audio_stream:
                        if not first_emitted:
                            first_emitted = True
                            try:
                                self.metrics.observe(
                                    "tts_ttfb_seconds",
                                    max(0.0, time.time() - start_time),
                                    labels={
                                        "provider": provider_key,
                                        "voice": request_for_provider.voice or "default",
                                        "format": request_for_provider.format.value,
                                    },
                                )
                                _record_voice_to_voice()
                            except _TTS_NONCRITICAL_EXCEPTIONS:
                                pass
                        audio_size += len(chunk)
                        yield chunk
                elif response.audio_data:
                    try:
                        self.metrics.observe(
                            "tts_ttfb_seconds",
                            max(0.0, time.time() - start_time),
                            labels={
                                "provider": provider_key,
                                "voice": request_for_provider.voice or "default",
                                "format": request_for_provider.format.value,
                            },
                        )
                        _record_voice_to_voice()
                    except _TTS_NONCRITICAL_EXCEPTIONS:
                        pass
                    audio_size = len(response.audio_data)
                    yield response.audio_data
                else:
                    error_message = f"No audio data returned by {provider_key}"
                    logger.error(error_message)
                    if self._stream_errors_as_audio:
                        yield f"ERROR: {error_message}".encode()
                    raise TTSGenerationError(error_message, provider=provider_key)
                success = True
        except _TTS_NONCRITICAL_EXCEPTIONS as e:
            logger.error(
                "Fallback generation failed for provider {}: {}",
                provider_key,
                type(e).__name__,
            )
            error_message = "All providers failed"
            if self._stream_errors_as_audio:
                yield f"ERROR: All providers failed - {str(e)}".encode()
            raise TTSGenerationError(
                error_message,
                provider=provider_key,
                details={"error_type": type(e).__name__},
            ) from e
        finally:
            await self._close_response_audio_stream(response)
            self._cleanup_transient_pocket_tts_cpp_voice_path(request_for_provider)
            if active_requests_incremented:
                with suppress(_TTS_NONCRITICAL_EXCEPTIONS):
                    await self._decrement_active_requests(provider_key)
                try:
                    duration = time.time() - start_time
                    self._record_tts_metrics(
                        provider=provider_key,
                        model=getattr(request_for_provider, "model", None) or provider_key,
                        voice=request_for_provider.voice or "default",
                        format=request_for_provider.format.value,
                        text_length=len(request_for_provider.text),
                        audio_size=audio_size,
                        duration=duration if duration >= 0 else 0.0,
                        success=success,
                        error=error_message if not success else None
                    )
                except _TTS_NONCRITICAL_EXCEPTIONS:
                    pass

    async def convert_chatterbox_voice(
        self,
        *,
        source_audio_path: str,
        target_voice_path: Optional[str],
        response_format: str | AudioFormat = "wav",
        stream: bool = False,
        provider_overrides: Optional[dict[str, Any]] = None,
    ) -> TTSResponse:
        """Convert source speech to a target voice using the Chatterbox VC adapter."""
        if isinstance(response_format, AudioFormat):
            audio_format = response_format
        else:
            try:
                audio_format = AudioFormat(str(response_format).strip().lower())
            except _TTS_NONCRITICAL_EXCEPTIONS as exc:
                raise TTSValidationError(
                    f"Unsupported response_format for Chatterbox voice conversion: {response_format}",
                    provider="chatterbox",
                ) from exc

        adapter = await self._get_adapter("chatterbox", "chatterbox", overrides=provider_overrides)
        if adapter is None:
            raise TTSProviderNotConfiguredError(
                "Chatterbox adapter is not configured",
                provider="chatterbox",
            )

        convert_voice = getattr(adapter, "convert_voice", None)
        if convert_voice is None:
            raise TTSProviderNotConfiguredError(
                "Chatterbox adapter does not support voice conversion",
                provider="chatterbox",
            )

        return await convert_voice(
            source_audio_path=source_audio_path,
            target_voice_path=target_voice_path,
            format=audio_format,
            stream=stream,
        )

    def _convert_request(self, request: OpenAISpeechRequest) -> TTSRequest:
        """Convert OpenAI request to unified TTS request"""
        # Map format
        format_mapping = {
            "mp3": AudioFormat.MP3,
            "opus": AudioFormat.OPUS,
            "aac": AudioFormat.AAC,
            "flac": AudioFormat.FLAC,
            "wav": AudioFormat.WAV,
            "pcm": AudioFormat.PCM,
            "ogg": AudioFormat.OGG,
            "webm": AudioFormat.WEBM,
            "ulaw": AudioFormat.ULAW
        }

        model_id = str(getattr(request, "model", "") or "").strip().lower()
        response_format = request.response_format
        output_format = getattr(request, "output_format", None)
        if output_format:
            try:
                explicit_fields = getattr(request, "model_fields_set", None)
                if explicit_fields is None:
                    explicit_fields = getattr(request, "__fields_set__", set())
                if model_id.startswith("chatterbox") and "response_format" not in explicit_fields:
                    response_format = output_format
            except _TTS_NONCRITICAL_EXCEPTIONS:
                pass

        audio_format = format_mapping.get(
            response_format.lower(),
            AudioFormat.MP3
        )
        # Optional language code mapping (lang_code primary; Chatterbox language alias next; extra_params.language override)
        language = self._normalize_language_code(getattr(request, 'lang_code', None))
        if language is None and model_id.startswith("chatterbox"):
            language = self._normalize_language_code(getattr(request, "language", None))
        # Optional voice reference decoding (base64)
        voice_ref_bytes = None
        if getattr(request, 'voice_reference', None):
            try:
                voice_ref_bytes = base64.b64decode(request.voice_reference)
            except _TTS_NONCRITICAL_EXCEPTIONS as exc:
                raise TTSInvalidVoiceReferenceError(
                    "Voice reference data is not valid base64",
                    details={"error": str(exc)}
                ) from exc
        # Provider-specific extras passthrough
        extras = getattr(request, 'extra_params', None) or {}
        target_sample_rate: Optional[int] = None
        seed: Optional[int] = None
        try:
            requested_rate = getattr(request, "target_sample_rate", None)
            if requested_rate is not None:
                parsed_rate = int(requested_rate)
                if parsed_rate > 0:
                    target_sample_rate = parsed_rate
        except _TTS_NONCRITICAL_EXCEPTIONS:
            target_sample_rate = None

        if isinstance(extras, dict):
            if target_sample_rate is None:
                for rate_key in ("target_sample_rate", "sample_rate"):
                    try:
                        maybe_rate = extras.get(rate_key)
                        if maybe_rate is None:
                            continue
                        parsed_rate = int(maybe_rate)
                        if parsed_rate > 0:
                            target_sample_rate = parsed_rate
                            break
                    except _TTS_NONCRITICAL_EXCEPTIONS:
                        continue

            extra_language = extras.get("language")
            if isinstance(extra_language, str):
                normalized_extra_language = self._normalize_language_code(extra_language)
                if normalized_extra_language:
                    language = normalized_extra_language
                    extras["language"] = normalized_extra_language
            elif extra_language is not None:
                try:
                    coerced_language = str(extra_language)
                except _TTS_NONCRITICAL_EXCEPTIONS:
                    coerced_language = None
                if coerced_language:
                    normalized_extra_language = self._normalize_language_code(coerced_language)
                    if normalized_extra_language:
                        language = normalized_extra_language
                        extras["language"] = normalized_extra_language
            if getattr(request, "reference_duration_min", None) is not None:
                extras["reference_duration_min"] = request.reference_duration_min
            if target_sample_rate is not None:
                extras["target_sample_rate"] = target_sample_rate
                # Alias for providers that currently look up `sample_rate` in extra params.
                extras["sample_rate"] = target_sample_rate
            for seed_key in ("seed", "generation_seed"):
                maybe_seed = extras.get(seed_key)
                if maybe_seed is None:
                    continue
                try:
                    seed = int(maybe_seed)
                    extras["seed"] = seed
                    break
                except _TTS_NONCRITICAL_EXCEPTIONS:
                    seed = None

        tts_request = TTSRequest(
            text=request.input,
            voice=request.voice,
            format=audio_format,
            target_sample_rate=target_sample_rate,
            speed=request.speed,
            stream=request.stream if hasattr(request, 'stream') else True,
            language=language,
            voice_reference=voice_ref_bytes,
            seed=seed,
            # Additional parameters can be added via extra_params
            extra_params=extras
        )

        # Preserve originating model for metrics/diagnostics when available
        tts_request.model = getattr(request, "model", None)

        return tts_request

    def _extract_voice_clone_prompt_payload(self, payload: Any) -> tuple[Optional[str], Optional[str]]:
        """Normalize voice_clone_prompt payload for storage (base64 + optional format)."""
        if payload is None:
            return None, None
        if isinstance(payload, (bytes, bytearray)):
            return base64.b64encode(bytes(payload)).decode("ascii"), None
        if isinstance(payload, str):
            return payload, None
        if isinstance(payload, dict):
            data_b64 = payload.get("data_b64") or payload.get("data")
            if isinstance(data_b64, str) and data_b64.strip():
                fmt = payload.get("format")
                if fmt is not None:
                    fmt = str(fmt)
                return data_b64, fmt
        return None, None

    @staticmethod
    def _coerce_nonempty_string(value: Any) -> str:
        """Return a stripped string value or an empty string for invalid inputs."""
        if value is None:
            return ""
        try:
            return str(value).strip()
        except _TTS_NONCRITICAL_EXCEPTIONS:
            return ""

    def _resolve_custom_voice_reference_id(
        self,
        request: TTSRequest,
        provider_hint: Optional[str],
    ) -> str:
        """Resolve stored custom voice ids from canonical and safe provider aliases."""
        voice_id = self._coerce_nonempty_string(request.voice)
        if voice_id.startswith("custom:"):
            return voice_id.split("custom:", 1)[-1].strip()

        if (provider_hint or "").lower() != "chatterbox":
            return ""

        extras = request.extra_params if isinstance(request.extra_params, dict) else {}
        voice_mode = self._coerce_nonempty_string(extras.get("voice_mode")).lower()
        if voice_mode != "predefined":
            return ""
        return self._coerce_nonempty_string(extras.get("predefined_voice_id"))

    async def _apply_custom_voice_reference(
        self,
        request: TTSRequest,
        user_id: Optional[int],
        provider_hint: Optional[str],
    ) -> None:
        """Populate voice_reference and provider artifacts for custom: voices."""
        if not user_id:
            return
        raw_id = self._resolve_custom_voice_reference_id(request, provider_hint)
        provider_key = (provider_hint or "").lower()
        try:
            from tldw_Server_API.app.core.TTS.voice_manager import VoiceProcessingError, get_voice_manager

            voice_manager = get_voice_manager()
            metadata = None
            if raw_id and request.voice_reference is None:
                request.voice_reference = await voice_manager.load_voice_reference_audio(user_id, raw_id)

            extras = request.extra_params or {}
            if not isinstance(extras, dict):
                extras = {}

            if raw_id:
                metadata = await voice_manager.load_reference_metadata(user_id, raw_id)
                ref_text_keys = ("reference_text", "ref_text", "voice_reference_text")
                has_ref_text = any(extras.get(key) for key in ref_text_keys)

                if metadata:
                    artifacts = metadata.provider_artifacts.get(provider_key) if provider_key else None
                    if artifacts:
                        if "ref_codes" not in extras and artifacts.get("ref_codes") is not None:
                            extras["ref_codes"] = artifacts.get("ref_codes")
                        if not has_ref_text:
                            extras["reference_text"] = (
                                artifacts.get("reference_text") or metadata.reference_text
                            )
                    elif not has_ref_text and metadata.reference_text:
                        extras["reference_text"] = metadata.reference_text

                    if provider_key == "qwen3_tts" and "voice_clone_prompt" not in extras:
                        if metadata.voice_clone_prompt_b64:
                            if metadata.voice_clone_prompt_format:
                                extras["voice_clone_prompt"] = {
                                    "format": metadata.voice_clone_prompt_format,
                                    "data_b64": metadata.voice_clone_prompt_b64,
                                }
                            else:
                                extras["voice_clone_prompt"] = metadata.voice_clone_prompt_b64

            if (provider_hint or "").lower() == "pocket_tts_cpp":
                await self._apply_pocket_tts_cpp_runtime_materialization(
                    request,
                    user_id=user_id,
                    voice_manager=voice_manager,
                    metadata=metadata,
                )
                extras = request.extra_params if isinstance(request.extra_params, dict) else extras
            request.extra_params = extras
        except VoiceProcessingError as e:
            if provider_key == "pocket_tts_cpp":
                raise TTSValidationError(
                    "PocketTTS.cpp voice reference preparation failed",
                    provider="pocket_tts_cpp",
                    error_code="pocket_tts_cpp_voice_materialization_failed",
                    details={
                        "voice_id": raw_id or None,
                        "reason": str(e),
                    },
                ) from e
            request_id, _ = self._get_tts_request_observability(request)
            logger.warning(
                "Custom voice resolution failed for {} (request_id={}): {}",
                raw_id or "direct-reference",
                request_id or "unknown",
                e,
            )
        except _TTS_NONCRITICAL_EXCEPTIONS as e:
            if provider_key == "pocket_tts_cpp":
                raise TTSGenerationError(
                    "PocketTTS.cpp voice reference preparation failed",
                    provider="pocket_tts_cpp",
                    error_code="pocket_tts_cpp_voice_materialization_failed",
                    details={
                        "voice_id": raw_id or None,
                        "reason": str(e),
                    },
                ) from e
            request_id, _ = self._get_tts_request_observability(request)
            logger.warning(
                "Custom voice resolution error for {} (request_id={}): {}",
                raw_id or "direct-reference",
                request_id or "unknown",
                e,
            )

    async def _maybe_store_qwen3_voice_prompt(
        self,
        request: TTSRequest,
        user_id: Optional[int],
        provider_key: str,
    ) -> None:
        """Persist Qwen3 voice_clone_prompt metadata for custom voices."""
        if not user_id or provider_key != "qwen3_tts":
            return
        voice_id = request.voice or ""
        if not isinstance(voice_id, str) or not voice_id.startswith("custom:"):
            return
        raw_id = voice_id.split("custom:", 1)[-1].strip()
        if not raw_id:
            return
        extras = request.extra_params or {}
        if not isinstance(extras, dict):
            return
        payload = extras.get("voice_clone_prompt")
        data_b64, fmt = self._extract_voice_clone_prompt_payload(payload)
        if not data_b64:
            return
        try:
            from tldw_Server_API.app.core.TTS.voice_manager import VoiceReferenceMetadata, get_voice_manager

            voice_manager = get_voice_manager()
            metadata = await voice_manager.load_reference_metadata(user_id, raw_id)
            if metadata is None:
                metadata = VoiceReferenceMetadata(voice_id=raw_id)
            if (
                metadata.voice_clone_prompt_b64 == data_b64
                and metadata.voice_clone_prompt_format == fmt
            ):
                return
            metadata.voice_clone_prompt_b64 = data_b64
            metadata.voice_clone_prompt_format = fmt
            await voice_manager.save_reference_metadata(user_id, metadata)
        except _TTS_NONCRITICAL_EXCEPTIONS as exc:
            request_id, _ = self._get_tts_request_observability(request)
            logger.debug(
                "Failed to persist Qwen3 voice_clone_prompt for {} (request_id={}): {}",
                raw_id,
                request_id or "unknown",
                exc,
            )

    async def _get_adapter(
        self,
        model: str,
        provider: Optional[str] = None,
        overrides: Optional[dict[str, Any]] = None,
    ) -> Optional[TTSAdapter]:
        """Get appropriate adapter for the request"""
        factory = await self._ensure_factory()
        if provider:
            # Specific provider requested
            provider_enum = TTSAdapterRegistry.resolve_provider(provider)
            if provider_enum is None:
                logger.warning(f"Unknown provider: {provider}")
            else:
                if provider_enum == TTSProvider.OMNIVOICE:
                    return await factory.registry.create_adapter_with_overrides(
                        provider_enum,
                        self._build_omnivoice_adapter_overrides(overrides),
                    )
                if overrides:
                    return await factory.registry.create_adapter_with_overrides(provider_enum, overrides)
                return await factory.registry.get_adapter(provider_enum)

        # Get adapter by model name. Some tests and integrations inject a
        # minimal factory that only implements get_adapter_by_model.
        model_provider = None
        if hasattr(factory, "get_provider_for_model"):
            model_provider = factory.get_provider_for_model(model)
        if model_provider == TTSProvider.OMNIVOICE:
            return await factory.registry.create_adapter_with_overrides(
                model_provider,
                self._build_omnivoice_adapter_overrides(overrides),
            )
        return await factory.get_adapter_by_model(model)

    def _build_omnivoice_adapter_overrides(
        self,
        overrides: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        merged_overrides = dict(overrides or {})
        merged_overrides["_supervisor"] = self._get_or_create_omnivoice_supervisor()
        return merged_overrides

    @staticmethod
    def _is_explicit_omnivoice_request(
        request: OpenAISpeechRequest,
        provider: Optional[str] = None,
        provider_hint: Optional[str] = None,
    ) -> bool:
        provider_value = (provider or "").strip().lower()
        if provider_value in _OMNIVOICE_ALIAS_VALUES:
            return True
        if provider_value:
            return False
        model_value = (getattr(request, "model", None) or "").strip().lower()
        if (
            model_value.startswith("omnivoice")
            or model_value.startswith("omni-voice")
            or model_value.startswith("omni_voice")
        ):
            return True
        provider_hint_value = (provider_hint or "").strip().lower()
        if provider_hint_value and provider_hint_value not in _OMNIVOICE_ALIAS_VALUES:
            return False
        voice = (getattr(request, "voice", None) or "").strip().lower()
        has_omnivoice_semantics = False
        if voice.startswith("custom:"):
            has_omnivoice_semantics = True
        if getattr(request, "voice_reference", None):
            has_omnivoice_semantics = True
        extras = getattr(request, "extra_params", None)
        if not isinstance(extras, dict):
            return has_omnivoice_semantics
        if any(extras.get(key) is not None for key in _OMNIVOICE_INSTRUCT_KEYS):
            has_omnivoice_semantics = True
        if any(key in extras for key in _OMNIVOICE_SEMANTIC_GENERATION_KEYS):
            has_omnivoice_semantics = True
        mode = extras.get("omnivoice_mode", extras.get("mode"))
        if isinstance(mode, str) and mode.strip().lower() in {"design", "clone"}:
            has_omnivoice_semantics = True
        return has_omnivoice_semantics

    def _get_or_create_omnivoice_supervisor(self) -> OmniVoiceSidecarSupervisor:
        if self._closing:
            raise RuntimeError("TTS service is closing")
        if self._omnivoice_supervisor is None:
            provider_cfg = self._get_provider_runtime_config("omnivoice")
            repo_root = Path(__file__).resolve().parents[4]
            self._omnivoice_supervisor = OmniVoiceSidecarSupervisor(
                provider_cfg,
                repo_root=repo_root,
            )
        return self._omnivoice_supervisor

    @staticmethod
    def _write_temp_audio_file_sync(audio_data: bytes, *, suffix: str) -> Path:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(audio_data)
            return Path(handle.name)

    @staticmethod
    def _reserve_temp_audio_path_sync(*, suffix: str) -> Path:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            return Path(handle.name)

    @staticmethod
    def _remove_temp_path_sync(path: Path) -> None:
        with suppress(_TTS_NONCRITICAL_EXCEPTIONS):
            path.unlink()

    async def _convert_response_if_needed(
        self,
        response: TTSResponse,
        request: TTSRequest,
        *,
        provider_key: str,
    ) -> TTSResponse:
        if response.audio_data is None:
            return response
        if response.format == request.format:
            return response
        if response.format != AudioFormat.WAV:
            return response

        input_path = None
        output_path = None
        try:
            input_path = await asyncio.to_thread(
                self._write_temp_audio_file_sync,
                response.audio_data,
                suffix=".wav",
            )
            output_path = await asyncio.to_thread(
                self._reserve_temp_audio_path_sync,
                suffix=f".{request.format.value}",
            )

            converted = await AudioConverter.convert_format(
                input_path,
                output_path,
                request.format.value,
                sample_rate=response.sample_rate,
                channels=response.channels,
            )
            if not converted or output_path is None or not output_path.exists():
                raise TTSGenerationError(
                    "TTS output conversion failed",
                    provider=provider_key,
                    details={"target_format": request.format.value},
                )

            converted_bytes = await asyncio.to_thread(output_path.read_bytes)
            response.audio_data = converted_bytes
            response.audio_content = converted_bytes
            response.format = request.format
            return response
        finally:
            if input_path is not None:
                await asyncio.to_thread(self._remove_temp_path_sync, input_path)
            if output_path is not None:
                await asyncio.to_thread(self._remove_temp_path_sync, output_path)

    def _resolve_provider_key(self, adapter: TTSAdapter) -> str:
        provider_key = getattr(adapter, "provider_key", None)
        if isinstance(provider_key, str) and provider_key:
            return provider_key.lower()
        provider_name = getattr(adapter, "provider_name", None)
        if isinstance(provider_name, str) and provider_name:
            return provider_name.lower()
        return "unknown"

    def _get_tts_config(self) -> Optional[dict[str, Any]]:
        for factory in (self.factory, self._factory):
            registry = getattr(factory, "registry", None) if factory else None
            cfg = getattr(registry, "config", None) if registry else None
            if isinstance(cfg, dict):
                return cfg
        return None

    def _get_provider_config(self, provider_key: str) -> Optional[Any]:
        config = self._get_tts_config()
        if not isinstance(config, dict):
            return None
        providers = config.get("providers")
        if isinstance(providers, dict):
            return providers.get(provider_key)
        return None

    def _get_strict_validation(self) -> bool:
        config = self._get_tts_config()
        if isinstance(config, dict) and "strict_validation" in config:
            value = config.get("strict_validation")
            if isinstance(value, bool):
                return value
            try:
                from .utils import parse_bool
                return parse_bool(value, default=True)
            except _TTS_NONCRITICAL_EXCEPTIONS:
                return bool(value)
        return True

    def _is_text_sanitization_enabled(self, provider_key: str) -> bool:
        provider_cfg = self._get_provider_config(provider_key)
        if provider_cfg is None:
            return False
        if isinstance(provider_cfg, dict):
            value = provider_cfg.get("sanitize_text", False)
        else:
            value = getattr(provider_cfg, "sanitize_text", False)
        if isinstance(value, bool):
            return value
        try:
            from .utils import parse_bool
            return parse_bool(value, default=False)
        except _TTS_NONCRITICAL_EXCEPTIONS:
            return bool(value)

    def _maybe_sanitize_request(self, request: TTSRequest, provider_key: str) -> TTSRequest:
        if not self._is_text_sanitization_enabled(provider_key):
            return request
        validator_cfg = {"strict_validation": self._get_strict_validation()}
        sanitized_text = validate_text_input(request.text, provider=provider_key, config=validator_cfg)
        if sanitized_text == request.text:
            return request
        sanitized_request = copy.copy(request)
        sanitized_request.text = sanitized_text
        return sanitized_request

    def _provider_aliases(self, adapter: TTSAdapter) -> set[str]:
        """Return a normalized alias set for a provider/adapter."""
        aliases = set()
        provider_name = getattr(adapter, "provider_name", None)
        if isinstance(provider_name, str) and provider_name:
            aliases.add(provider_name.lower())
        provider_key = self._resolve_provider_key(adapter)
        if provider_key:
            aliases.add(provider_key)
        aliases.add(adapter.__class__.__name__.lower())
        provider_key_attr = getattr(adapter, "PROVIDER_KEY", None)
        if provider_key_attr:
            aliases.add(str(provider_key_attr).lower())
        for provider in TTSProvider:
            if provider.value.lower() in aliases or provider.name.lower() in aliases:
                aliases.add(provider.value.lower())
                aliases.add(provider.name.lower())
        normalized = set()
        for alias in aliases:
            normalized.add(alias)
            normalized.add(alias.replace("adapter", ""))
            normalized.add(alias.replace("adapter", "").replace("_", "").replace("-", ""))
        return {alias for alias in normalized if alias}

    def _build_exclude_tokens(self, adapter: TTSAdapter) -> list[str]:
        """Build normalized exclude tokens for a provider."""
        return list(self._provider_aliases(adapter))

    async def _increment_active_requests(self, provider: str) -> None:
        """Increment per-provider active request count and update gauge."""
        async with self._active_requests_lock:
            current = self._active_request_counts.get(provider, 0) + 1
            self._active_request_counts[provider] = current
        with suppress(_TTS_NONCRITICAL_EXCEPTIONS):
            self.metrics.set_gauge(
                "tts_active_requests",
                current,
                labels={"provider": provider}
            )

    async def _decrement_active_requests(self, provider: str) -> None:
        """Decrement per-provider active request count and update gauge."""
        async with self._active_requests_lock:
            current = self._active_request_counts.get(provider, 0)
            if current > 0:
                current -= 1
            if current == 0:
                self._active_request_counts.pop(provider, None)
            else:
                self._active_request_counts[provider] = current
        with suppress(_TTS_NONCRITICAL_EXCEPTIONS):
            self.metrics.set_gauge(
                "tts_active_requests",
                current,
                labels={"provider": provider}
            )

    async def _get_fallback_adapter(
        self,
        request: TTSRequest,
        exclude: Optional[list[str]] = None
    ) -> Optional[TTSAdapter]:
        """Get a fallback adapter that can handle the request"""
        factory = await self._ensure_factory()
        registry = getattr(factory, "registry", None)
        exclude_tokens = {token.lower() for token in (exclude or [])}
        # Normalize tokens to cover enum values (e.g., "OpenAIAdapter" -> "openai")
        normalized_tokens = set(exclude_tokens)
        for token in list(exclude_tokens):
            cleaned = token.replace("adapter", "").replace("_", "").replace("-", "")
            if cleaned and cleaned != token:
                normalized_tokens.add(cleaned)
        for provider in TTSProvider:
            if provider.value.lower() in exclude_tokens or provider.name.lower() in exclude_tokens:
                normalized_tokens.add(provider.value.lower())
                normalized_tokens.add(provider.name.lower())
        exclude_tokens = normalized_tokens

        # Find adapter that supports the requirements
        adapter = await factory.get_best_adapter(
            language=request.language,
            format=request.format,
            supports_streaming=request.stream
        )

        # Check if adapter is not in exclude list
        if adapter:
            provider_aliases = self._provider_aliases(adapter)
            if not provider_aliases & exclude_tokens:
                return adapter
            exclude_tokens.update(provider_aliases)

        # Try any available adapter
        for provider in TTSProvider:
            if provider.value.lower() in exclude_tokens or provider.name.lower() in exclude_tokens:
                continue
            # Skip providers without registered adapters (e.g., TODO placeholders)
            if registry and hasattr(registry, "_adapter_specs"):
                specs = registry._adapter_specs
                try:
                    if provider not in specs:
                        continue
                except TypeError:
                    # If specs is not dict-like, fall back to attempting fetch
                    pass

            try:
                adapter = await factory.registry.get_adapter(provider)
            except TTSProviderNotConfiguredError:
                logger.debug(f"Skipping provider {provider.value} - no adapter configured")
                continue
            except _TTS_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"Skipping provider {provider.value} due to error: {exc}")
                continue

            if adapter:
                # Validate if it can handle the request
                validation_result = await adapter.validate_request(request)
                if isinstance(validation_result, tuple):
                    is_valid, _ = validation_result
                elif validation_result is None:
                    is_valid = True
                else:
                    is_valid = bool(validation_result)
                if is_valid:
                    return adapter
                exclude_tokens.update(self._provider_aliases(adapter))

        return None

    def _record_tts_metrics(
        self,
        provider: str,
        model: str,
        voice: str,
        format: str,
        text_length: int,
        audio_size: int,
        duration: float,
        success: bool,
        error: Optional[str] = None
    ):
        """Record TTS request metrics"""
        # Record request counter
        self.metrics.increment(
            "tts_requests_total",
            labels={
                "provider": provider,
                "model": model,
                "voice": voice,
                "format": format,
                "status": "success" if success else "failure"
            }
        )

        # Record duration histogram
        self.metrics.observe(
            "tts_request_duration_seconds",
            duration,
            labels={"provider": provider, "model": model, "voice": voice}
        )

        # Record text length histogram
        self.metrics.observe(
            "tts_text_length_characters",
            text_length,
            labels={"provider": provider}
        )

        # Record audio size if successful
        if success and audio_size > 0:
            self.metrics.observe(
                "tts_audio_size_bytes",
                audio_size,
                labels={"provider": provider, "format": format}
            )

        # Log performance metrics
        if success:
            chars_per_second = text_length / duration if duration > 0 else 0
            logger.info(
                f"TTS metrics: provider={provider}, duration={duration:.2f}s, "
                f"text_length={text_length}, audio_size={audio_size}, "
                f"chars/sec={chars_per_second:.1f}"
            )
        else:
            logger.warning(
                f"TTS failed: provider={provider}, duration={duration:.2f}s, "
                f"error={error}"
            )

    def _record_fallback_event(
        self,
        *,
        from_provider: str,
        to_provider: str,
        success: str,
        outcome: str,
        error: Optional[Exception] = None,
        request_id: Optional[str] = None,
    ) -> None:
        category = "none"
        if error is not None:
            with suppress(_TTS_NONCRITICAL_EXCEPTIONS):
                category = self._categorize_error(error)
        labels_attempt = {
            "from_provider": from_provider or "unknown",
            "to_provider": to_provider or "unknown",
            "success": success or "pending",
        }
        labels_outcome = {
            "from_provider": from_provider or "unknown",
            "to_provider": to_provider or "unknown",
            "outcome": outcome or "unknown",
            "category": category,
        }
        with suppress(_TTS_NONCRITICAL_EXCEPTIONS):
            self.metrics.increment("tts_fallback_attempts", labels=labels_attempt)
        with suppress(_TTS_NONCRITICAL_EXCEPTIONS):
            self.metrics.increment("tts_fallback_outcomes_total", labels=labels_outcome)
        logger.info(
            "tts_fallback_event request_id={} from_provider={} to_provider={} success={} outcome={} category={}",
            request_id or "unknown",
            labels_attempt["from_provider"],
            labels_attempt["to_provider"],
            labels_attempt["success"],
            labels_outcome["outcome"],
            labels_outcome["category"],
        )

    def _categorize_error(self, error: Exception) -> str:
        """
        Categorize error types for better error handling decisions.
        Uses the new exception system's categorization.

        Args:
            error: Exception that occurred

        Returns:
            Error category string
        """
        # Use the new exception system's categorization
        return categorize_error(error)

    async def _handle_provider_fallback(self, request: TTSRequest, failed_provider: str, error_msg: str):
        """
        Handle provider fallback logging and circuit breaker updates.

        Args:
            request: Original TTS request
            failed_provider: Name of the provider that failed
            error_msg: Error message from the failed provider
        """
        logger.warning(f"Provider {failed_provider} failed: {error_msg}")
        logger.info(f"Attempting fallback for request: text_length={len(request.text)}, voice={request.voice}")

        # Update circuit breaker state if available
        if self.circuit_manager:
            breaker = await self.circuit_manager.get_breaker(failed_provider)
            if breaker:
                # The circuit breaker will track the failure internally
                logger.debug(f"Circuit breaker updated for {failed_provider}")

    async def _try_fallback_providers(
        self,
        request: TTSRequest,
        exclude_providers: list[str],
        failed_provider: Optional[str],
        *,
        metadata_only: bool = False,
        metadata_target: Optional[Any] = None,
        user_id: Optional[int] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        Try fallback providers in priority order.

        Args:
            request: TTS request to fulfill
            exclude_providers: List of provider names to exclude
            failed_provider: Canonical provider name that just failed

        Yields:
            Audio chunks from successful provider
        """
        origin_provider = failed_provider or "unknown"
        request_id, _correlation_id = self._get_tts_request_observability(request)
        fallback_adapter = await self._get_fallback_adapter(request, exclude_providers)

        if fallback_adapter:
            try:
                fallback_provider_key = self._resolve_provider_key(fallback_adapter)
                original_model = getattr(request, "model", None)
                fallback_model = (
                    getattr(fallback_adapter, "default_model", None)
                    or fallback_provider_key
                )
                request.model = fallback_model
                try:
                    async for chunk in self._generate_with_adapter(
                        fallback_adapter,
                        request,
                        metadata_only=metadata_only,
                        metadata_target=metadata_target,
                        user_id=user_id,
                    ):
                        yield chunk
                finally:
                    request.model = original_model
                logger.info(f"Successfully fell back to {fallback_provider_key}")
                self._record_fallback_event(
                    from_provider=origin_provider,
                    to_provider=fallback_provider_key,
                    success="true",
                    outcome="success",
                    request_id=request_id,
                )
            except TTSError as e:
                logger.error(f"Fallback provider {fallback_provider_key} also failed: {e}")
                self._record_fallback_event(
                    from_provider=origin_provider,
                    to_provider=fallback_provider_key,
                    success="false",
                    outcome="failed",
                    error=e,
                    request_id=request_id,
                )

                # Try one more fallback if available and error is retryable
                if is_retryable_error(e):
                    exclude_providers.extend(
                        token for token in self._build_exclude_tokens(fallback_adapter)
                        if token not in exclude_providers
                    )
                    next_failed_provider = fallback_provider_key
                    final_fallback = await self._get_fallback_adapter(request, exclude_providers)

                    if final_fallback:
                        try:
                            final_provider_key = self._resolve_provider_key(final_fallback)
                            secondary_original_model = getattr(request, "model", None)
                            secondary_model = (
                                getattr(final_fallback, "default_model", None)
                                or final_provider_key
                            )
                            request.model = secondary_model
                            try:
                                async for chunk in self._generate_with_adapter(
                                    final_fallback,
                                    request,
                                    metadata_only=metadata_only,
                                    metadata_target=metadata_target,
                                    user_id=user_id,
                                ):
                                    yield chunk
                            finally:
                                request.model = secondary_original_model
                            logger.info(f"Final fallback to {final_provider_key} succeeded")
                            self._record_fallback_event(
                                from_provider=next_failed_provider,
                                to_provider=final_provider_key,
                                success="true",
                                outcome="success",
                                request_id=request_id,
                            )
                        except _TTS_NONCRITICAL_EXCEPTIONS as final_e:
                            # Wrap non-TTS errors
                            if not isinstance(final_e, TTSError):
                                final_e = TTSGenerationError(
                                    "Final fallback failed",
                                    provider=final_provider_key,
                                    details={"error": str(final_e)}
                                )
                            self._record_fallback_event(
                                from_provider=next_failed_provider,
                                to_provider=final_provider_key,
                                success="false",
                                outcome="failed",
                                error=final_e,
                                request_id=request_id,
                            )
                            error_msg = f"All providers failed. Last error: {str(final_e)}"
                            logger.error(error_msg)
                            if self._stream_errors_as_audio:
                                yield f"ERROR: {error_msg}".encode()
                            else:
                                raise
                    else:
                        origin_provider = next_failed_provider
                        self._record_fallback_event(
                            from_provider=origin_provider,
                            to_provider="none",
                            success="false",
                            outcome="exhausted",
                            error=e,
                            request_id=request_id,
                        )
                        if self._stream_errors_as_audio:
                            yield b"ERROR: All fallback providers exhausted"
                        else:
                            raise TTSFallbackExhaustedError("All fallback providers exhausted") from e
                else:
                    # Non-retryable error, don't attempt more fallbacks
                    self._record_fallback_event(
                        from_provider=origin_provider,
                        to_provider=fallback_provider_key,
                        success="false",
                        outcome="failed_non_retryable",
                        error=e,
                        request_id=request_id,
                    )
                    if self._stream_errors_as_audio:
                        yield f"ERROR: {str(e)} (non-retryable)".encode()
                    else:
                        raise
            except _TTS_NONCRITICAL_EXCEPTIONS as e:
                # Handle unexpected errors
                logger.error(f"Unexpected error in fallback: {e}", exc_info=True)
                self._record_fallback_event(
                    from_provider=origin_provider,
                    to_provider="unknown",
                    success="false",
                    outcome="error",
                    error=e if isinstance(e, Exception) else None,
                    request_id=request_id,
                )
                if self._stream_errors_as_audio:
                    yield f"ERROR: Unexpected error during fallback: {str(e)}".encode()
                else:
                    raise TTSGenerationError(f"Unexpected error during fallback: {str(e)}") from e
        else:
            self._record_fallback_event(
                from_provider=origin_provider,
                to_provider="none",
                success="false",
                outcome="unavailable",
                error=TTSFallbackExhaustedError("No fallback providers available"),
                request_id=request_id,
            )
            if self._stream_errors_as_audio:
                yield b"ERROR: No fallback providers available"
            else:
                raise TTSFallbackExhaustedError("No fallback providers available")

    def get_status(self) -> dict[str, Any]:
        """Get service status"""
        factory = self.factory or self._factory
        if not factory or not hasattr(factory, "get_status"):
            return {"providers": {}, "initialized": False}
        status = factory.get_status()

        # Add circuit breaker status if available
        if self.circuit_manager:
            status["circuit_breakers"] = self.circuit_manager.get_all_status()

        return status

    async def unload_provider(self, provider: str) -> dict[str, Any]:
        """Release one cached TTS provider runtime without shutting down the whole service."""
        factory = self.factory or self._factory
        if factory is None:
            factory = await get_tts_factory()
            self.factory = factory
            self._factory = factory

        unload = getattr(factory, "unload_provider", None)
        if unload is None:
            raise TTSProviderNotConfiguredError(
                "TTS factory does not support provider unload",
                provider=str(provider),
            )

        result = unload(provider)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, dict):
            return result
        return {"provider": str(provider), "unloaded": bool(result)}


# Singleton management
_service_instance: Optional[TTSServiceV2] = None
_service_lock = asyncio.Lock()


async def get_tts_service_v2(config: Optional[dict[str, Any]] = None) -> TTSServiceV2:
    """
    Get or create the enhanced TTS service singleton.

    Args:
        config: Configuration for the service

    Returns:
        TTSServiceV2 instance
    """
    global _service_instance

    if _service_instance is None:
        async with _service_lock:
            if _service_instance is None:
                # Load configuration if not provided
                if config is None:
                    from tldw_Server_API.app.core.config import load_comprehensive_config_with_tts
                    config_obj = load_comprehensive_config_with_tts()
                    config = config_obj.get_tts_config()

                # Get factory
                factory = await get_tts_factory(config)

                # Get circuit breaker manager
                circuit_manager = await get_circuit_manager(config)

                # Create service
                _service_instance = TTSServiceV2(factory, circuit_manager)
                logger.info("Enhanced TTS Service (V2) initialized")

    return _service_instance


async def close_tts_service_v2():
    """Close the enhanced TTS service"""
    global _service_instance

    if _service_instance:
        await _service_instance.shutdown()
        await close_tts_factory()
        _service_instance = None
        logger.info("Enhanced TTS Service (V2) closed")


# Backwards compatibility wrapper
class TTSServiceAdapter:
    """
    Adapter to make V2 service compatible with existing code.
    Maps old interface to new service.
    """

    def __init__(self):
        self.service_v2: Optional[TTSServiceV2] = None

    async def generate_audio_stream(
        self,
        request: OpenAISpeechRequest,
        internal_model_id: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate audio stream (backwards compatible interface).

        Args:
            request: OpenAI speech request
            internal_model_id: Internal model identifier

        Yields:
            Audio chunks
        """
        # Get V2 service
        if not self.service_v2:
            self.service_v2 = await get_tts_service_v2()

        # Map internal model ID to provider/model
        # This maintains compatibility with existing code
        model_mapping = {
            "openai_official_tts-1": "tts-1",
            "openai_official_tts-1-hd": "tts-1-hd",
            "local_kokoro_default_onnx": "kokoro",
            "elevenlabs_english_v1": "elevenlabs",  # TODO: Implement
            "alltalk_api_backend": "alltalk"  # TODO: Implement
        }

        # Update request model if needed
        request.model = model_mapping.get(internal_model_id, internal_model_id)

        # Generate with V2 service
        async for chunk in self.service_v2.generate_speech(request, fallback=True):
            yield chunk


# Export the adapter for backwards compatibility
TTSService = TTSServiceAdapter

#
# End of tts_service_v2.py
#######################################################################################################################
