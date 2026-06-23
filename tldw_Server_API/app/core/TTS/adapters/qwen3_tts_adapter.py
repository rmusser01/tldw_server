# qwen3_tts_adapter.py
# Description: Adapter for Qwen3-TTS local models
#
# Imports
from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib
import inspect
import tempfile
from collections.abc import AsyncGenerator, Iterable
from pathlib import Path
from typing import Any, Callable

import numpy as np
from loguru import logger

from ..audio_utils import (
    analyze_audio_signal,
    crossfade_audio,
    split_text_into_chunks,
    trim_trailing_silence,
)
from ..streaming_audio_writer import AudioNormalizer, StreamingAudioWriter
from ..tts_exceptions import (
    TTSAudioQualityError,
    TTSGenerationError,
    TTSInvalidVoiceReferenceError,
    TTSProviderInitializationError,
    TTSStreamingError,
    TTSValidationError,
)
from ..utils import parse_bool, resolve_qwen3_runtime_name
from .base import (
    AudioFormat,
    TTSAdapter,
    TTSCapabilities,
    TTSRequest,
    TTSResponse,
    VoiceInfo,
)
from .qwen3_runtime_base import Qwen3Runtime

_QWEN3_COERCE_EXCEPTIONS = (
    TypeError,
    ValueError,
    OverflowError,
)

_QWEN3_SIGNATURE_EXCEPTIONS = (
    TypeError,
    ValueError,
)

_QWEN3_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ImportError,
)

_QWEN3_WORKER_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ImportError,
    TTSAudioQualityError,
    TTSGenerationError,
    TTSInvalidVoiceReferenceError,
    TTSProviderInitializationError,
    TTSStreamingError,
    TTSValidationError,
)


class Qwen3TTSAdapter(TTSAdapter):
    """Adapter for Qwen3-TTS local models (CustomVoice/VoiceDesign/Base)."""

    PROVIDER_KEY = "qwen3_tts"
    handles_text_chunking = True

    MODEL_CUSTOMVOICE_17B = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    MODEL_CUSTOMVOICE_06B = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    MODEL_VOICEDESIGN_17B = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    MODEL_BASE_17B = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    MODEL_BASE_06B = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

    SUPPORTED_LANGUAGES = {
        "auto",
        "zh",
        "en",
        "ja",
        "ko",
        "de",
        "fr",
        "ru",
        "pt",
        "es",
        "it",
    }

    CUSTOMVOICE_SPEAKERS = [
        "Vivian",
        "Serena",
        "Uncle_Fu",
        "Dylan",
        "Eric",
        "Ryan",
        "Aiden",
        "Ono_Anna",
        "Sohee",
    ]

    _CUSTOMVOICE_METHODS = (
        "generate_custom_voice",
        "custom_voice",
        "generate_custom",
        "generate",
        "__call__",
    )
    _VOICEDESIGN_METHODS = (
        "generate_voice_design",
        "voice_design",
        "generate_design",
        "generate",
        "__call__",
    )
    _VOICECLONE_METHODS = (
        "generate_voice_clone",
        "voice_clone",
        "generate_clone",
        "generate",
        "__call__",
    )
    _CUSTOMVOICE_STREAM_METHODS = (
        "stream_custom_voice",
        "generate_custom_voice_stream",
        "stream",
    )
    _VOICEDESIGN_STREAM_METHODS = (
        "stream_voice_design",
        "generate_voice_design_stream",
        "stream",
    )
    _VOICECLONE_STREAM_METHODS = (
        "stream_voice_clone",
        "generate_voice_clone_stream",
        "stream",
    )

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config=config)
        cfg = config or {}
        self.runtime = (cfg.get("runtime") or "auto").strip().lower()
        self.model = (cfg.get("model") or "auto").strip()
        self.model_path = cfg.get("model_path")
        self.tokenizer_model = cfg.get("tokenizer_model") or "Qwen/Qwen3-TTS-Tokenizer-12Hz"
        self.device = (cfg.get("device") or "cpu").strip().lower()
        self.dtype = (cfg.get("dtype") or "float16").strip().lower()
        self.attn_implementation = (cfg.get("attn_implementation") or "default").strip().lower()
        self.auto_download = parse_bool(cfg.get("auto_download"), default=False)
        self.auto_min_vram_gb = self._coerce_int(cfg.get("auto_min_vram_gb")) or 12
        self.stream_chunk_size_ms = self._coerce_int(cfg.get("stream_chunk_size_ms")) or 200
        self.sample_rate = self._coerce_int(cfg.get("sample_rate")) or 24000
        self._backend = None
        self._backend_module = None
        self._runtime_impl: Qwen3Runtime | None = None
        self._pipeline_builders: list[tuple[str, Callable[[str], Any]]] = []
        self._pipelines: dict[str, Any] = {}
        self._pipeline_lock = asyncio.Lock()
        self._audio_normalizer = AudioNormalizer()
        self._model_aliases = {
            self.MODEL_CUSTOMVOICE_17B.lower(): self.MODEL_CUSTOMVOICE_17B,
            self.MODEL_CUSTOMVOICE_06B.lower(): self.MODEL_CUSTOMVOICE_06B,
            self.MODEL_VOICEDESIGN_17B.lower(): self.MODEL_VOICEDESIGN_17B,
            self.MODEL_BASE_17B.lower(): self.MODEL_BASE_17B,
            self.MODEL_BASE_06B.lower(): self.MODEL_BASE_06B,
        }

    def _resolve_runtime_name(self) -> str:
        return resolve_qwen3_runtime_name(self.runtime)

    def _build_runtime(self) -> Qwen3Runtime:
        runtime_name = self._resolve_runtime_name()
        if runtime_name == "mlx":
            from .qwen3_runtime_mlx import Qwen3MlxRuntime

            return Qwen3MlxRuntime(self)
        if runtime_name == "remote":
            from .qwen3_runtime_remote import RemoteQwenRuntime

            return RemoteQwenRuntime(self)
        if runtime_name != "upstream":
            logger.warning(
                f"{self.provider_name}: runtime '{runtime_name}' is not implemented yet; "
                "falling back to upstream runtime"
            )
        from .qwen3_runtime_upstream import Qwen3UpstreamRuntime

        return Qwen3UpstreamRuntime(self)

    def _get_runtime(self) -> Qwen3Runtime:
        if self._runtime_impl is None:
            self._runtime_impl = self._build_runtime()
        return self._runtime_impl

    def _coerce_int(self, value: Any) -> int | None:
        try:
            if value is None:
                return None
            return int(value)
        except _QWEN3_COERCE_EXCEPTIONS:
            return None

    def _coerce_float(self, value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except _QWEN3_COERCE_EXCEPTIONS:
            return None

    def _resolve_chunking_params(self, extras: dict[str, Any]) -> tuple[bool, int, int, int, int]:
        cfg = self.config or {}

        def _pick_int(keys: tuple[str, ...], default: int) -> int:
            for key in keys:
                if key in extras and extras.get(key) is not None:
                    val = self._coerce_int(extras.get(key))
                    if val is not None:
                        return val
                if key in cfg and cfg.get(key) is not None:
                    val = self._coerce_int(cfg.get(key))
                    if val is not None:
                        return val
            return default

        enabled = parse_bool(extras.get("chunking"), default=True)
        target = _pick_int(("chunk_target_chars", "chunk_target", "chunk_chars_target"), 120)
        max_chars = _pick_int(("chunk_max_chars", "chunk_max", "chunk_chars_max"), 150)
        min_chars = _pick_int(("chunk_min_chars", "chunk_min", "chunk_chars_min"), 50)
        crossfade_ms = _pick_int(("chunk_crossfade_ms", "crossfade_ms"), 50)
        return enabled, target, max_chars, min_chars, crossfade_ms

    def _resolve_audio_check_params(self, extras: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config or {}

        def _pick_float(keys: tuple[str, ...], default: float) -> float:
            for key in keys:
                if key in extras and extras.get(key) is not None:
                    val = self._coerce_float(extras.get(key))
                    if val is not None:
                        return val
                if key in cfg and cfg.get(key) is not None:
                    val = self._coerce_float(cfg.get(key))
                    if val is not None:
                        return val
            return default

        def _pick_int(keys: tuple[str, ...], default: int) -> int:
            for key in keys:
                if key in extras and extras.get(key) is not None:
                    val = self._coerce_int(extras.get(key))
                    if val is not None:
                        return val
                if key in cfg and cfg.get(key) is not None:
                    val = self._coerce_int(cfg.get(key))
                    if val is not None:
                        return val
            return default

        def _pick_bool(keys: tuple[str, ...], default: bool) -> bool:
            for key in keys:
                if key in extras:
                    return parse_bool(extras.get(key), default=default)
                if key in cfg:
                    return parse_bool(cfg.get(key), default=default)
            return default

        return {
            "enabled": _pick_bool(("audio_checks", "audio_quality_checks"), True),
            "strict": _pick_bool(("audio_checks_strict", "audio_quality_strict"), False),
            "per_chunk": _pick_bool(("audio_checks_per_chunk",), False),
            "trim_trailing_silence": _pick_bool(("audio_trim_trailing_silence", "trim_trailing_silence"), False),
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

    def _apply_audio_checks(
        self,
        audio: np.ndarray,
        text: str,
        extras: dict[str, Any],
        context: str,
        params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        check_params = params or self._resolve_audio_check_params(extras)
        if not check_params.get("enabled", True):
            return audio

        metrics = analyze_audio_signal(
            audio,
            self.sample_rate,
            silence_threshold=check_params["silence_threshold"],
        )

        if check_params.get("trim_trailing_silence") and check_params["trailing_silence_ms"] > 0:
            if metrics["trailing_silence_ms"] >= check_params["trailing_silence_ms"]:
                trimmed = trim_trailing_silence(
                    audio,
                    self.sample_rate,
                    threshold=check_params["silence_threshold"],
                    min_silence_ms=check_params["trailing_silence_ms"],
                )
                if trimmed.shape[0] < np.asarray(audio).reshape(-1).shape[0]:
                    audio = trimmed
                    metrics = analyze_audio_signal(
                        audio,
                        self.sample_rate,
                        silence_threshold=check_params["silence_threshold"],
                    )

        warnings: list[str] = []
        if metrics["rms"] < check_params["min_rms"] or metrics["peak"] < check_params["min_peak"]:
            warnings.append(
                f"low_levels(rms={metrics['rms']:.4f}, peak={metrics['peak']:.4f})"
            )
        if check_params["trailing_silence_ms"] > 0 and metrics["trailing_silence_ms"] >= check_params["trailing_silence_ms"]:
            warnings.append(f"trailing_silence_ms={metrics['trailing_silence_ms']:.0f}")

        if text and check_params["expected_chars_per_sec"] > 0 and len(text) >= check_params["min_text_length"]:
            expected = len(text) / float(check_params["expected_chars_per_sec"])
            min_expected = max(check_params["min_duration_seconds"], expected * check_params["min_duration_ratio"])
            if metrics["duration_sec"] < min_expected:
                warnings.append(
                    f"duration_short(actual={metrics['duration_sec']:.2f}s, expected>={min_expected:.2f}s)"
                )

        if warnings:
            details = {
                "context": context,
                "metrics": metrics,
                "warnings": warnings,
            }
            if check_params.get("strict"):
                raise TTSAudioQualityError(
                    "Qwen3-TTS audio checks failed",
                    provider=self.PROVIDER_KEY,
                    details=details,
                )
            logger.warning(
                f"{self.provider_name}: audio checks flagged ({context}): {', '.join(warnings)}"
            )
        return audio

    def _should_chunk_text(self, text: str, extras: dict[str, Any]) -> tuple[bool, int, int, int, int]:
        enabled, target, max_chars, min_chars, crossfade_ms = self._resolve_chunking_params(extras)
        if not enabled:
            return False, target, max_chars, min_chars, crossfade_ms
        if not text:
            return False, target, max_chars, min_chars, crossfade_ms
        if len(text) <= max_chars:
            return False, target, max_chars, min_chars, crossfade_ms
        return True, target, max_chars, min_chars, crossfade_ms

    def _decode_base64_payload(self, payload: str) -> bytes:
        if "," in payload:
            payload = payload.split(",", 1)[1]
        try:
            return base64.b64decode(payload, validate=True)
        except Exception as exc:
            raise TTSValidationError(
                "Invalid base64 payload provided",
                provider=self.PROVIDER_KEY,
            ) from exc

    def _decode_voice_clone_prompt(self, payload: Any) -> bytes | None:
        if payload is None:
            return None
        if isinstance(payload, (bytes, bytearray)):
            return bytes(payload)
        if isinstance(payload, str):
            return self._decode_base64_payload(payload)
        if isinstance(payload, dict):
            data_b64 = payload.get("data_b64") or payload.get("data")
            if not isinstance(data_b64, str) or not data_b64.strip():
                raise TTSValidationError(
                    "voice_clone_prompt payload missing data_b64",
                    provider=self.PROVIDER_KEY,
                )
            return self._decode_base64_payload(data_b64)
        raise TTSValidationError(
            "voice_clone_prompt payload must be base64 string or {format,data_b64}",
            provider=self.PROVIDER_KEY,
        )

    def _extract_voice_reference_bytes(self, voice_reference: Any) -> bytes | None:
        if voice_reference is None:
            return None
        if isinstance(voice_reference, (bytes, bytearray)):
            return bytes(voice_reference)
        if isinstance(voice_reference, str):
            try:
                return self._decode_base64_payload(voice_reference)
            except Exception as exc:
                raise TTSInvalidVoiceReferenceError(
                    "Voice reference is not valid base64",
                    provider=self.PROVIDER_KEY,
                    details={"error": str(exc)},
                ) from exc
        raise TTSInvalidVoiceReferenceError(
            "Voice reference must be raw bytes or base64 string",
            provider=self.PROVIDER_KEY,
            details={"type": type(voice_reference).__name__},
        )

    def _resolve_language(self, request: TTSRequest) -> str:
        language = request.language
        extras = request.extra_params or {}
        if not language and isinstance(extras, dict):
            extra_language = extras.get("language")
            if extra_language is not None:
                try:
                    extra_language = str(extra_language)
                except _QWEN3_NONCRITICAL_EXCEPTIONS:
                    extra_language = None
            if extra_language:
                language = extra_language
        if not language or not str(language).strip():
            return "auto"
        return str(language).strip().lower()

    def _normalize_speaker(self, speaker: str) -> str:
        normalized = speaker.strip().lower()
        normalized = normalized.replace(" ", "_").replace("-", "_")
        while "__" in normalized:
            normalized = normalized.replace("__", "_")
        return normalized

    def _resolve_speaker(self, voice: str | None) -> str | None:
        if not voice or not isinstance(voice, str):
            return None
        if voice.startswith("custom:"):
            return None
        normalized = self._normalize_speaker(voice)
        for speaker in self.CUSTOMVOICE_SPEAKERS:
            if normalized == self._normalize_speaker(speaker):
                return speaker
        return voice.strip()

    def _resolve_torch_dtype(self) -> Any | None:
        dtype = (self.dtype or "").strip().lower()
        if not dtype:
            return None
        try:
            import torch
        except ImportError:
            return self.dtype
        mapping = {
            "float16": getattr(torch, "float16", None),
            "fp16": getattr(torch, "float16", None),
            "bfloat16": getattr(torch, "bfloat16", None),
            "bf16": getattr(torch, "bfloat16", None),
            "float32": getattr(torch, "float32", None),
            "fp32": getattr(torch, "float32", None),
        }
        mapping = {key: value for key, value in mapping.items() if value is not None}
        return mapping.get(dtype, self.dtype)

    def _build_model_kwargs(self, model_id: str) -> dict[str, Any]:
        dtype = self._resolve_torch_dtype()
        kwargs: dict[str, Any] = {
            "device": self.device,
            "dtype": dtype,
            "torch_dtype": dtype,
            "attn_implementation": None if self.attn_implementation == "default" else self.attn_implementation,
            "tokenizer_model": self.tokenizer_model,
            "tokenizer": self.tokenizer_model,
            "local_files_only": not self.auto_download,
            "auto_download": self.auto_download,
            "download": self.auto_download,
            "trust_remote_code": True,
        }
        if self.model_path:
            kwargs["model_path"] = self.model_path
        return kwargs

    def _should_skip_builder(self, exc: TypeError) -> bool:
        msg = str(exc)
        return any(
            token in msg
            for token in (
                "unexpected keyword",
                "positional argument",
                "missing 1 required positional",
                "missing required positional",
            )
        )

    def _invoke_builder(self, builder: Callable[..., Any], model_id: str) -> Any:
        kwargs = {k: v for k, v in self._build_model_kwargs(model_id).items() if v is not None}
        try:
            sig = inspect.signature(builder)
        except _QWEN3_SIGNATURE_EXCEPTIONS:
            sig = None
        params = []
        if sig:
            params = list(sig.parameters.values())
            if params and params[0].name in ("self", "cls"):
                params = params[1:]
        param_names = {p.name for p in params}
        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in params)
        accepts_varargs = any(p.kind == p.VAR_POSITIONAL for p in params)

        model_param = None
        for candidate in ("model", "model_id", "model_name", "model_path", "path"):
            if candidate in param_names:
                model_param = candidate
                break

        filtered_kwargs = kwargs if accepts_kwargs else {k: v for k, v in kwargs.items() if k in param_names}
        if model_param:
            filtered_kwargs[model_param] = model_id
            return builder(**filtered_kwargs)
        if params or accepts_varargs:
            return builder(model_id, **filtered_kwargs)
        return builder(**filtered_kwargs)

    def _discover_pipeline_builders(self, module: Any) -> list[tuple[str, Callable[[str], Any]]]:
        builders: list[tuple[str, Callable[[str], Any]]] = []
        seen: set[str] = set()

        def add_builder(name: str, builder: Callable[..., Any]) -> None:
            if name in seen:
                return
            seen.add(name)

            def _factory(model_id: str, _builder: Callable[..., Any] = builder) -> Any:
                return self._invoke_builder(_builder, model_id)

            builders.append((name, _factory))

        class_candidates = (
            "Qwen3TTSModel",
            "QwenTTSModel",
            "Qwen3TTS",
            "QwenTTS",
            "Qwen3TTSPipeline",
            "QwenTTSPipeline",
            "TTSPipeline",
            "TTS",
        )
        function_candidates = (
            "from_pretrained",
            "load_model",
            "load_pipeline",
            "create_pipeline",
            "get_pipeline",
        )

        def scan_module(mod: Any) -> None:
            for name in class_candidates:
                cls = getattr(mod, name, None)
                if cls is None:
                    continue
                for method_name in ("from_pretrained", "load", "load_model", "create"):
                    method = getattr(cls, method_name, None)
                    if callable(method):
                        add_builder(f"{name}.{method_name}", method)
                        break
                else:
                    if callable(cls):
                        add_builder(name, cls)

            for name in function_candidates:
                fn = getattr(mod, name, None)
                if callable(fn):
                    add_builder(name, fn)

        scan_module(module)
        for sub in ("pipeline", "tts"):
            try:
                submodule = importlib.import_module(f"qwen_tts.{sub}")
            except ImportError:
                continue
            scan_module(submodule)

        return builders

    def _has_module_generation(self, module: Any) -> bool:
        for name in (
            *self._CUSTOMVOICE_METHODS,
            *self._VOICEDESIGN_METHODS,
            *self._VOICECLONE_METHODS,
            *self._CUSTOMVOICE_STREAM_METHODS,
            *self._VOICEDESIGN_STREAM_METHODS,
            *self._VOICECLONE_STREAM_METHODS,
        ):
            if callable(getattr(module, name, None)):
                return True
        return False

    async def _get_pipeline(self, model_id: str) -> Any | None:
        if not self._pipeline_builders:
            return None
        load_id = self.model_path or model_id
        if load_id in self._pipelines:
            return self._pipelines[load_id]
        async with self._pipeline_lock:
            if load_id in self._pipelines:
                return self._pipelines[load_id]
            pipeline = await asyncio.to_thread(self._build_pipeline, load_id)
            self._pipelines[load_id] = pipeline
            # Register pipeline with resource manager for cache tracking (best-effort)
            try:
                from ..tts_resource_manager import get_resource_manager
                resource_manager = await get_resource_manager()
                register_result = resource_manager.register_model(
                    provider=self.PROVIDER_KEY,
                    model_instance=pipeline,
                    cleanup_callback=lambda: self._drop_pipeline(load_id),
                    model_key=str(load_id),
                )
                if asyncio.iscoroutine(register_result):
                    await register_result
            except _QWEN3_NONCRITICAL_EXCEPTIONS:
                pass
            return pipeline

    def _drop_pipeline(self, load_id: str) -> None:
        """Remove cached pipeline and run any best-effort cleanup."""
        pipeline = self._pipelines.pop(load_id, None)
        if pipeline is None:
            return
        for attr in ("close", "shutdown", "cleanup"):
            handler = getattr(pipeline, attr, None)
            if callable(handler):
                with contextlib.suppress(_QWEN3_NONCRITICAL_EXCEPTIONS):
                    handler()

    def _build_pipeline(self, model_id: str) -> Any:
        last_error: Exception | None = None
        for name, builder in self._pipeline_builders:
            try:
                pipeline = builder(model_id)
                if pipeline is not None:
                    logger.info(f"{self.provider_name}: loaded Qwen3-TTS pipeline using {name}")
                    return pipeline
            except TypeError as exc:
                last_error = exc
                if self._should_skip_builder(exc):
                    continue
                raise
            except Exception as exc:
                last_error = exc
                raise
        if last_error:
            raise last_error
        raise TTSProviderInitializationError(
            "No suitable Qwen3-TTS pipeline builder found",
            provider=self.PROVIDER_KEY,
        )

    def _select_backend_callable(self, target: Any, candidates: Iterable[str]) -> Callable[..., Any] | None:
        for name in candidates:
            fn = getattr(target, name, None)
            if callable(fn):
                return fn
        return None

    def _build_call_kwargs(
        self,
        fn: Callable[..., Any],
        payload: dict[str, Any],
        alias_map: dict[str, tuple[str, ...]],
        prefer_ref_path: bool = False,
    ) -> dict[str, Any]:
        try:
            sig = inspect.signature(fn)
        except _QWEN3_SIGNATURE_EXCEPTIONS:
            sig = None
        params = []
        if sig:
            params = list(sig.parameters.values())
            if params and params[0].name in ("self", "cls"):
                params = params[1:]
        param_names = {p.name for p in params}
        accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in params)

        kwargs: dict[str, Any] = {}
        for canonical, aliases in alias_map.items():
            value = payload.get(canonical)
            if value is None:
                continue
            if canonical == "ref_audio" and prefer_ref_path and payload.get("ref_audio_path"):
                continue
            target = None
            for alias in aliases:
                if alias in param_names:
                    target = alias
                    break
            if target:
                kwargs[target] = value
            elif accepts_kwargs:
                kwargs[canonical] = value
        return kwargs

    async def _invoke_backend(self, fn: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
        if inspect.iscoroutinefunction(fn):
            return await fn(**kwargs)
        result = await asyncio.to_thread(fn, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def _coerce_audio_output(self, output: Any) -> np.ndarray:
        if output is None:
            return np.zeros(0, dtype=np.int16)
        if isinstance(output, dict):
            for key in ("audio", "wav", "pcm", "samples"):
                if key in output:
                    output = output[key]
                    break
        if isinstance(output, (list, tuple)) and output:
            output = output[0]
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        if isinstance(output, (bytes, bytearray)):
            return np.frombuffer(output, dtype=np.int16)
        audio = np.asarray(output)
        if audio.ndim > 1:
            audio = audio.reshape(-1)
        if audio.dtype != np.int16:
            audio = self._audio_normalizer.normalize(audio, target_dtype=np.int16)
        return audio

    def _chunk_pcm_audio(self, pcm_audio: np.ndarray) -> Iterable[np.ndarray]:
        if pcm_audio.size == 0:
            return []
        chunk_samples = int(self.sample_rate * (self.stream_chunk_size_ms / 1000.0))
        if chunk_samples <= 0:
            chunk_samples = len(pcm_audio)
        return (
            pcm_audio[start:start + chunk_samples]
            for start in range(0, len(pcm_audio), chunk_samples)
        )

    def _wants_ref_audio_path(self, fn: Callable[..., Any]) -> bool:
        try:
            sig = inspect.signature(fn)
        except _QWEN3_SIGNATURE_EXCEPTIONS:
            return False
        param_names = set(sig.parameters.keys())
        if any(name in param_names for name in ("ref_audio_path", "reference_audio_path", "audio_prompt_path")):
            return True
        return any("path" in name and ("audio" in name or "ref" in name) for name in param_names)

    def _write_temp_audio(self, audio_bytes: bytes, suffix: str = ".wav") -> str:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, prefix="qwen3_voice_") as tmp_file:
            tmp_file.write(audio_bytes)
            return tmp_file.name

    def _detect_audio_suffix(self, audio_bytes: bytes) -> str:
        if audio_bytes.startswith(b"RIFF"):
            return ".wav"
        if audio_bytes.startswith(b"ID3") or audio_bytes[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
            return ".mp3"
        if audio_bytes.startswith(b"fLaC"):
            return ".flac"
        if audio_bytes.startswith(b"OggS"):
            return ".ogg"
        if len(audio_bytes) >= 8 and audio_bytes[4:8] == b"ftyp":
            return ".m4a"
        return ".wav"

    def _encode_ref_audio_payload(self, audio_bytes: bytes) -> str:
        return base64.b64encode(audio_bytes).decode("ascii")

    def _resolve_mode(self, model_id: str) -> str:
        model_key = model_id.lower()
        if "voicedesign" in model_key:
            return "voice_design"
        if model_key.endswith("base") or model_key.endswith("-base") or "base" in model_key:
            return "voice_clone"
        return "custom_voice"

    def _can_stream_transcode(self, target_format: AudioFormat) -> tuple[bool, str | None]:
        try:
            writer = StreamingAudioWriter(
                format=target_format.value,
                sample_rate=self.sample_rate,
                channels=1,
            )
            writer.close()
            return True, None
        except _QWEN3_NONCRITICAL_EXCEPTIONS as exc:
            return False, str(exc)

    def _chunk_bytes(self, payload: bytes, chunk_size: int = 64 * 1024) -> AsyncGenerator[bytes, None]:
        async def _iterator():
            if not payload:
                return
            size = max(1024, int(chunk_size))
            for idx in range(0, len(payload), size):
                yield payload[idx:idx + size]
        return _iterator()

    async def _initialize_upstream_runtime(self) -> bool:
        """Initialize the upstream qwen_tts runtime."""
        try:
            module = importlib.import_module("qwen_tts")
        except Exception as exc:
            raise TTSProviderInitializationError(
                "qwen-tts package is required for Qwen3-TTS adapter",
                provider=self.PROVIDER_KEY,
            ) from exc

        self._backend_module = module
        self._pipeline_builders = self._discover_pipeline_builders(module)
        if not self._pipeline_builders and not self._has_module_generation(module):
            raise TTSProviderInitializationError(
                "Qwen3-TTS backend is missing required pipeline or generation methods",
                provider=self.PROVIDER_KEY,
            )
        self._backend = module
        return True

    async def initialize(self) -> bool:
        """Initialize the selected Qwen3 runtime."""
        runtime = self._get_runtime()
        return await runtime.initialize()

    async def _get_upstream_capabilities(self) -> TTSCapabilities:
        max_text_length = self._coerce_int(self.config.get("max_text_length")) or 5000
        voices = [VoiceInfo(id=speaker, name=speaker) for speaker in self.CUSTOMVOICE_SPEAKERS]
        return TTSCapabilities(
            provider_name=self.provider_name,
            supported_languages=set(self.SUPPORTED_LANGUAGES),
            supported_voices=voices,
            supported_formats={
                AudioFormat.MP3,
                AudioFormat.OPUS,
                AudioFormat.AAC,
                AudioFormat.WAV,
                AudioFormat.PCM,
            },
            max_text_length=max_text_length,
            supports_streaming=True,
            supports_voice_cloning=True,
            supports_emotion_control=True,
            sample_rate=self.sample_rate,
            default_format=AudioFormat.PCM,
            metadata={
                "runtime": "upstream",
                "supported_modes": [
                    "custom_voice_preset",
                    "uploaded_custom_voice",
                    "voice_design",
                ],
                "supports_uploaded_custom_voices": True,
            },
        )

    async def get_capabilities(self) -> TTSCapabilities:
        runtime = self._get_runtime()
        return await runtime.get_capabilities()

    def _is_voice_design_request(self, request: TTSRequest) -> bool:
        voice = request.voice
        if voice is None:
            return True
        return bool(isinstance(voice, str) and not voice.strip())

    def _is_voice_clone_request(self, request: TTSRequest) -> bool:
        if request.voice_reference:
            return True
        extras = request.extra_params or {}
        if extras.get("reference_text") or extras.get("x_vector_only_mode"):
            return True
        return bool(extras.get("voice_clone_prompt"))

    def _resolve_auto_model(self) -> str:
        if self.device.startswith("cuda"):
            resolved = self.MODEL_CUSTOMVOICE_06B
            total_gb = None
            try:
                import torch
                if torch.cuda.is_available():
                    device_idx = 0
                    if ":" in self.device:
                        try:
                            device_idx = int(self.device.split(":", 1)[1])
                        except (ValueError, TypeError, IndexError):
                            device_idx = 0
                    props = torch.cuda.get_device_properties(device_idx)
                    total_gb = props.total_memory / (1024 ** 3)
                    if total_gb >= float(self.auto_min_vram_gb):
                        resolved = self.MODEL_CUSTOMVOICE_17B
            except _QWEN3_NONCRITICAL_EXCEPTIONS:
                logger.debug("Qwen3-TTS auto model selection could not read CUDA VRAM; falling back")
            logger.info(
                f"{self.provider_name}: auto model resolved to {resolved} "
                f"(device={self.device}, vram_gb={total_gb}, threshold={self.auto_min_vram_gb})"
            )
            return resolved
        if self.device == "mps":
            logger.info(
                f"{self.provider_name}: auto model resolved to {self.MODEL_CUSTOMVOICE_06B} "
                f"(device={self.device})"
            )
            return self.MODEL_CUSTOMVOICE_06B
        logger.info(
            f"{self.provider_name}: auto model resolved to {self.MODEL_CUSTOMVOICE_06B} "
            f"(device={self.device})"
        )
        return self.MODEL_CUSTOMVOICE_06B

    def _resolve_model(self, request: TTSRequest) -> str:
        requested = (getattr(request, "model", None) or self.model or "auto")
        requested = requested.strip() if isinstance(requested, str) else str(requested)
        requested_key = requested.lower()
        provider_aliases = {
            self.PROVIDER_KEY.lower(),
            self.PROVIDER_KEY.lower().replace("_", "-"),
        }
        if requested_key in provider_aliases:
            configured = (self.model or "auto").strip()
            configured_key = configured.lower()
            if configured_key in provider_aliases:
                configured = "auto"
                configured_key = "auto"
            requested = configured
            requested_key = configured_key
        if requested_key == "auto":
            if self._is_voice_design_request(request) or self._is_voice_clone_request(request):
                raise TTSValidationError(
                    "model='auto' is only valid for CustomVoice requests; specify a VoiceDesign/Base model",
                    provider=self.PROVIDER_KEY,
                )
            return self._resolve_auto_model()
        return self._model_aliases.get(requested_key, requested)

    async def _stream_transcoded_pcm(
        self,
        pcm_stream: AsyncGenerator[np.ndarray, None],
        request_format: AudioFormat,
    ) -> AsyncGenerator[bytes, None]:
        audio_normalizer = AudioNormalizer()
        writer = StreamingAudioWriter(
            format=request_format.value,
            sample_rate=self.sample_rate,
            channels=1,
        )
        try:
            async for chunk in pcm_stream:
                if chunk is None:
                    continue
                if isinstance(chunk, (bytes, bytearray)):
                    pcm = np.frombuffer(chunk, dtype=np.int16)
                else:
                    pcm = np.asarray(chunk)
                if pcm.dtype != np.int16:
                    pcm = audio_normalizer.normalize(pcm, target_dtype=np.int16)
                data = writer.write_chunk(pcm)
                if data:
                    yield data
            tail = writer.write_chunk(finalize=True)
            if tail:
                yield tail
        finally:
            writer.close()

    def _generate_pcm_stream(
        self,
        request: TTSRequest,
        model_id: str,
    ) -> AsyncGenerator[np.ndarray, None]:
        async def _iterate_sync_generator(gen: Iterable[Any]) -> AsyncGenerator[Any, None]:
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()

            def _worker() -> None:
                try:
                    for item in gen:
                        loop.call_soon_threadsafe(queue.put_nowait, item)
                except _QWEN3_WORKER_EXCEPTIONS as exc:
                    loop.call_soon_threadsafe(queue.put_nowait, exc)
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            loop.run_in_executor(None, _worker)
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item

        async def _stream():
            extras = request.extra_params or {}
            if not isinstance(extras, dict):
                extras = {}
            should_chunk, target_chars, max_chars, min_chars, crossfade_ms = self._should_chunk_text(
                request.text,
                extras,
            )
            if should_chunk:
                merged_pcm = await self._generate_pcm(request, model_id)
                for chunk in self._chunk_pcm_audio(merged_pcm):
                    yield chunk
                return

            pipeline = await self._get_pipeline(model_id)
            module = self._backend_module
            if pipeline is None and module is None:
                raise TTSProviderInitializationError(
                    "Qwen3-TTS backend is not configured in this build",
                    provider=self.PROVIDER_KEY,
                )

            language = self._resolve_language(request)
            mode = self._resolve_mode(model_id)

            payload: dict[str, Any] = {
                "text": request.text,
                "language": language,
                "model": model_id,
                "device": self.device,
                "dtype": self._resolve_torch_dtype() or self.dtype,
                "attn_implementation": self.attn_implementation,
                "tokenizer_model": self.tokenizer_model,
                "stream": True,
            }
            max_new_tokens = self._coerce_int(extras.get("max_new_tokens"))
            if max_new_tokens is not None:
                payload["max_new_tokens"] = max_new_tokens
            min_new_tokens = self._coerce_int(extras.get("min_new_tokens"))
            if min_new_tokens is not None:
                payload["min_new_tokens"] = min_new_tokens

            call_target = pipeline or module
            cleanup_path: str | None = None

            if mode == "custom_voice":
                speaker = self._resolve_speaker(request.voice)
                if not speaker:
                    raise TTSValidationError(
                        "CustomVoice requests require a valid speaker",
                        provider=self.PROVIDER_KEY,
                    )
                payload["speaker"] = speaker
                instruct = extras.get("instruct") or extras.get("instruction") or extras.get("style")
                if isinstance(instruct, str) and instruct.strip():
                    payload["instruct"] = instruct.strip()
                fn = self._select_backend_callable(call_target, self._CUSTOMVOICE_STREAM_METHODS)
                if fn is None and module is not None:
                    fn = self._select_backend_callable(module, self._CUSTOMVOICE_STREAM_METHODS)
                if fn is None:
                    fn = self._select_backend_callable(call_target, self._CUSTOMVOICE_METHODS)
                if fn is None and module is not None:
                    fn = self._select_backend_callable(module, self._CUSTOMVOICE_METHODS)
                alias_map = {
                    "text": ("text", "input", "utterance", "sentence"),
                    "language": ("language", "lang", "lang_code", "language_code"),
                    "speaker": ("speaker", "voice", "speaker_id", "speaker_name"),
                    "instruct": ("instruct", "instruction", "style", "prompt", "style_prompt", "voice_prompt"),
                    "max_new_tokens": ("max_new_tokens", "max_tokens", "max_new_token"),
                    "min_new_tokens": ("min_new_tokens", "min_tokens", "min_new_token"),
                    "model": ("model", "model_id", "model_name"),
                    "device": ("device", "device_map"),
                    "dtype": ("dtype", "torch_dtype", "precision"),
                    "attn_implementation": ("attn_implementation", "attn_impl", "attention_impl"),
                    "tokenizer_model": ("tokenizer_model", "tokenizer", "tokenizer_name"),
                    "stream": ("stream", "streaming"),
                }
                prefer_ref_path = False
            elif mode == "voice_design":
                instruct = extras.get("instruct") or extras.get("instruction") or extras.get("description")
                if not (isinstance(instruct, str) and instruct.strip()):
                    raise TTSValidationError(
                        "VoiceDesign requests require extra_params.instruct",
                        provider=self.PROVIDER_KEY,
                    )
                payload["instruct"] = instruct.strip()
                fn = self._select_backend_callable(call_target, self._VOICEDESIGN_STREAM_METHODS)
                if fn is None and module is not None:
                    fn = self._select_backend_callable(module, self._VOICEDESIGN_STREAM_METHODS)
                if fn is None:
                    fn = self._select_backend_callable(call_target, self._VOICEDESIGN_METHODS)
                if fn is None and module is not None:
                    fn = self._select_backend_callable(module, self._VOICEDESIGN_METHODS)
                alias_map = {
                    "text": ("text", "input", "utterance", "sentence"),
                    "language": ("language", "lang", "lang_code", "language_code"),
                    "instruct": (
                        "instruct",
                        "instruction",
                        "description",
                        "prompt",
                        "voice_description",
                        "voice_prompt",
                    ),
                    "max_new_tokens": ("max_new_tokens", "max_tokens", "max_new_token"),
                    "min_new_tokens": ("min_new_tokens", "min_tokens", "min_new_token"),
                    "model": ("model", "model_id", "model_name"),
                    "device": ("device", "device_map"),
                    "dtype": ("dtype", "torch_dtype", "precision"),
                    "attn_implementation": ("attn_implementation", "attn_impl", "attention_impl"),
                    "tokenizer_model": ("tokenizer_model", "tokenizer", "tokenizer_name"),
                    "stream": ("stream", "streaming"),
                }
                prefer_ref_path = False
            else:
                voice_bytes = self._extract_voice_reference_bytes(request.voice_reference)
                x_vector_only = parse_bool(extras.get("x_vector_only_mode"), default=False)
                if voice_bytes is None:
                    raise TTSInvalidVoiceReferenceError(
                        "Voice reference is required for Qwen3 Base model requests",
                        provider=self.PROVIDER_KEY,
                    )
                ref_text = (
                    extras.get("reference_text")
                    or extras.get("ref_text")
                    or extras.get("voice_reference_text")
                )
                if not (isinstance(ref_text, str) and ref_text.strip()) and not x_vector_only:
                    raise TTSValidationError(
                        "Qwen3 Base models require reference_text unless x_vector_only_mode is true",
                        provider=self.PROVIDER_KEY,
                    )
                if isinstance(ref_text, str) and ref_text.strip():
                    payload["ref_text"] = ref_text.strip()
                payload["x_vector_only_mode"] = x_vector_only
                voice_clone_prompt = extras.get("voice_clone_prompt")
                decoded_prompt = self._decode_voice_clone_prompt(voice_clone_prompt)
                if decoded_prompt is not None:
                    payload["voice_clone_prompt"] = decoded_prompt

                fn = self._select_backend_callable(call_target, self._VOICECLONE_STREAM_METHODS)
                if fn is None and module is not None:
                    fn = self._select_backend_callable(module, self._VOICECLONE_STREAM_METHODS)
                if fn is None:
                    fn = self._select_backend_callable(call_target, self._VOICECLONE_METHODS)
                if fn is None and module is not None:
                    fn = self._select_backend_callable(module, self._VOICECLONE_METHODS)
                if voice_bytes:
                    if fn is not None and self._wants_ref_audio_path(fn):
                        cleanup_path = self._write_temp_audio(
                            voice_bytes, suffix=self._detect_audio_suffix(voice_bytes)
                        )
                        payload["ref_audio_path"] = cleanup_path
                    else:
                        payload["ref_audio"] = self._encode_ref_audio_payload(voice_bytes)
                alias_map = {
                    "text": ("text", "input", "utterance", "sentence"),
                    "language": ("language", "lang", "lang_code", "language_code"),
                    "ref_audio_path": ("ref_audio_path", "reference_audio_path", "audio_prompt_path"),
                    "ref_audio": ("ref_audio", "reference_audio", "audio_prompt", "prompt_audio", "audio"),
                    "ref_text": ("ref_text", "reference_text", "prompt_text", "transcript"),
                    "x_vector_only_mode": ("x_vector_only_mode", "xvector_only_mode", "xvector_only", "use_xvector_only"),
                    "voice_clone_prompt": ("voice_clone_prompt", "clone_prompt", "prompt_emb", "speaker_emb"),
                    "max_new_tokens": ("max_new_tokens", "max_tokens", "max_new_token"),
                    "min_new_tokens": ("min_new_tokens", "min_tokens", "min_new_token"),
                    "model": ("model", "model_id", "model_name"),
                    "device": ("device", "device_map"),
                    "dtype": ("dtype", "torch_dtype", "precision"),
                    "attn_implementation": ("attn_implementation", "attn_impl", "attention_impl"),
                    "tokenizer_model": ("tokenizer_model", "tokenizer", "tokenizer_name"),
                    "stream": ("stream", "streaming"),
                }
                prefer_ref_path = payload.get("ref_audio_path") is not None

            if fn is None:
                raise TTSProviderInitializationError(
                    "Qwen3-TTS backend does not expose a streaming generation method",
                    provider=self.PROVIDER_KEY,
                )

            kwargs = self._build_call_kwargs(fn, payload, alias_map, prefer_ref_path=prefer_ref_path)
            try:
                if inspect.isasyncgenfunction(fn):
                    stream_obj = fn(**kwargs)
                elif inspect.iscoroutinefunction(fn):
                    stream_obj = await fn(**kwargs)
                else:
                    stream_obj = fn(**kwargs)
                    if asyncio.iscoroutine(stream_obj):
                        stream_obj = await stream_obj

                if hasattr(stream_obj, "__aiter__"):
                    async for chunk in stream_obj:  # type: ignore[func-returns-value]
                        yield self._coerce_audio_output(chunk)
                elif isinstance(stream_obj, Iterable) and not isinstance(
                    stream_obj, (bytes, bytearray, np.ndarray)
                ):
                    async for chunk in _iterate_sync_generator(stream_obj):  # type: ignore[arg-type]
                        yield self._coerce_audio_output(chunk)
                else:
                    audio = self._coerce_audio_output(stream_obj)
                    for chunk in self._chunk_pcm_audio(audio):
                        yield chunk
            finally:
                if cleanup_path:
                    Path(cleanup_path).unlink(missing_ok=True)

        return _stream()

    async def _generate_pcm(
        self,
        request: TTSRequest,
        model_id: str,
    ) -> np.ndarray:
        extras = request.extra_params or {}
        if not isinstance(extras, dict):
            extras = {}
        should_chunk, target_chars, max_chars, min_chars, crossfade_ms = self._should_chunk_text(
            request.text,
            extras,
        )
        check_params = self._resolve_audio_check_params(extras)
        if should_chunk:
            chunks = split_text_into_chunks(
                request.text,
                target_chars=target_chars,
                max_chars=max_chars,
                min_chars=min_chars,
            )
            if len(chunks) > 1:
                audio_parts: list[np.ndarray] = []
                for chunk in chunks:
                    part = await self._generate_pcm_for_text(request, model_id, chunk)
                    if check_params.get("enabled") and check_params.get("per_chunk"):
                        part = self._apply_audio_checks(
                            part,
                            chunk,
                            extras,
                            context="chunk",
                            params=check_params,
                        )
                    audio_parts.append(part)
                merged = audio_parts[0]
                for part in audio_parts[1:]:
                    merged = crossfade_audio(
                        merged, part, sample_rate=self.sample_rate, crossfade_ms=crossfade_ms
                    )
                return self._apply_audio_checks(
                    merged,
                    request.text,
                    extras,
                    context="merged",
                    params=check_params,
                )
        pcm_audio = await self._generate_pcm_for_text(request, model_id, request.text)
        return self._apply_audio_checks(
            pcm_audio,
            request.text,
            extras,
            context="full",
            params=check_params,
        )

    async def _generate_pcm_for_text(
        self,
        request: TTSRequest,
        model_id: str,
        text: str,
    ) -> np.ndarray:
        pipeline = await self._get_pipeline(model_id)
        module = self._backend_module
        if pipeline is None and module is None:
            raise TTSProviderInitializationError(
                "Qwen3-TTS backend is not configured in this build",
                provider=self.PROVIDER_KEY,
            )

        extras = request.extra_params or {}
        if not isinstance(extras, dict):
            extras = {}
        language = self._resolve_language(request)
        mode = self._resolve_mode(model_id)

        payload: dict[str, Any] = {
            "text": text,
            "language": language,
            "model": model_id,
            "device": self.device,
            "dtype": self._resolve_torch_dtype() or self.dtype,
            "attn_implementation": self.attn_implementation,
            "tokenizer_model": self.tokenizer_model,
        }
        max_new_tokens = self._coerce_int(extras.get("max_new_tokens"))
        if max_new_tokens is not None:
            payload["max_new_tokens"] = max_new_tokens
        min_new_tokens = self._coerce_int(extras.get("min_new_tokens"))
        if min_new_tokens is not None:
            payload["min_new_tokens"] = min_new_tokens

        call_target = pipeline or module
        cleanup_path: str | None = None

        if mode == "custom_voice":
            speaker = self._resolve_speaker(request.voice)
            if not speaker:
                raise TTSValidationError(
                    "CustomVoice requests require a valid speaker",
                    provider=self.PROVIDER_KEY,
                )
            payload["speaker"] = speaker
            instruct = extras.get("instruct") or extras.get("instruction") or extras.get("style")
            if isinstance(instruct, str) and instruct.strip():
                payload["instruct"] = instruct.strip()
            fn = self._select_backend_callable(call_target, self._CUSTOMVOICE_METHODS)
            if fn is None and module is not None:
                fn = self._select_backend_callable(module, self._CUSTOMVOICE_METHODS)
            alias_map = {
                "text": ("text", "input", "utterance", "sentence"),
                "language": ("language", "lang", "lang_code", "language_code"),
                "speaker": ("speaker", "voice", "speaker_id", "speaker_name"),
                "instruct": ("instruct", "instruction", "style", "prompt", "style_prompt", "voice_prompt"),
                "max_new_tokens": ("max_new_tokens", "max_tokens", "max_new_token"),
                "min_new_tokens": ("min_new_tokens", "min_tokens", "min_new_token"),
                "model": ("model", "model_id", "model_name"),
                "device": ("device", "device_map"),
                "dtype": ("dtype", "torch_dtype", "precision"),
                "attn_implementation": ("attn_implementation", "attn_impl", "attention_impl"),
                "tokenizer_model": ("tokenizer_model", "tokenizer", "tokenizer_name"),
            }
            prefer_ref_path = False
        elif mode == "voice_design":
            instruct = extras.get("instruct") or extras.get("instruction") or extras.get("description")
            if not (isinstance(instruct, str) and instruct.strip()):
                raise TTSValidationError(
                    "VoiceDesign requests require extra_params.instruct",
                    provider=self.PROVIDER_KEY,
                )
            payload["instruct"] = instruct.strip()
            fn = self._select_backend_callable(call_target, self._VOICEDESIGN_METHODS)
            if fn is None and module is not None:
                fn = self._select_backend_callable(module, self._VOICEDESIGN_METHODS)
            alias_map = {
                "text": ("text", "input", "utterance", "sentence"),
                "language": ("language", "lang", "lang_code", "language_code"),
                "instruct": (
                    "instruct",
                    "instruction",
                    "description",
                    "prompt",
                    "voice_description",
                    "voice_prompt",
                ),
                "max_new_tokens": ("max_new_tokens", "max_tokens", "max_new_token"),
                "min_new_tokens": ("min_new_tokens", "min_tokens", "min_new_token"),
                "model": ("model", "model_id", "model_name"),
                "device": ("device", "device_map"),
                "dtype": ("dtype", "torch_dtype", "precision"),
                "attn_implementation": ("attn_implementation", "attn_impl", "attention_impl"),
                "tokenizer_model": ("tokenizer_model", "tokenizer", "tokenizer_name"),
            }
            prefer_ref_path = False
        else:
            voice_bytes = self._extract_voice_reference_bytes(request.voice_reference)
            x_vector_only = parse_bool(extras.get("x_vector_only_mode"), default=False)
            if voice_bytes is None:
                raise TTSInvalidVoiceReferenceError(
                    "Voice reference is required for Qwen3 Base model requests",
                    provider=self.PROVIDER_KEY,
                )
            ref_text = (
                extras.get("reference_text")
                or extras.get("ref_text")
                or extras.get("voice_reference_text")
            )
            if not (isinstance(ref_text, str) and ref_text.strip()) and not x_vector_only:
                raise TTSValidationError(
                    "Qwen3 Base models require reference_text unless x_vector_only_mode is true",
                    provider=self.PROVIDER_KEY,
                )
            if isinstance(ref_text, str) and ref_text.strip():
                payload["ref_text"] = ref_text.strip()
            payload["x_vector_only_mode"] = x_vector_only
            voice_clone_prompt = extras.get("voice_clone_prompt")
            decoded_prompt = self._decode_voice_clone_prompt(voice_clone_prompt)
            if decoded_prompt is not None:
                payload["voice_clone_prompt"] = decoded_prompt

            fn = self._select_backend_callable(call_target, self._VOICECLONE_METHODS)
            if fn is None and module is not None:
                fn = self._select_backend_callable(module, self._VOICECLONE_METHODS)
            if voice_bytes:
                if fn is not None and self._wants_ref_audio_path(fn):
                    cleanup_path = self._write_temp_audio(
                        voice_bytes, suffix=self._detect_audio_suffix(voice_bytes)
                    )
                    payload["ref_audio_path"] = cleanup_path
                else:
                    payload["ref_audio"] = self._encode_ref_audio_payload(voice_bytes)
            alias_map = {
                "text": ("text", "input", "utterance", "sentence"),
                "language": ("language", "lang", "lang_code", "language_code"),
                "ref_audio_path": ("ref_audio_path", "reference_audio_path", "audio_prompt_path"),
                "ref_audio": ("ref_audio", "reference_audio", "audio_prompt", "prompt_audio", "audio"),
                "ref_text": ("ref_text", "reference_text", "prompt_text", "transcript"),
                "x_vector_only_mode": ("x_vector_only_mode", "xvector_only_mode", "xvector_only", "use_xvector_only"),
                "voice_clone_prompt": ("voice_clone_prompt", "clone_prompt", "prompt_emb", "speaker_emb"),
                "max_new_tokens": ("max_new_tokens", "max_tokens", "max_new_token"),
                "min_new_tokens": ("min_new_tokens", "min_tokens", "min_new_token"),
                "model": ("model", "model_id", "model_name"),
                "device": ("device", "device_map"),
                "dtype": ("dtype", "torch_dtype", "precision"),
                "attn_implementation": ("attn_implementation", "attn_impl", "attention_impl"),
                "tokenizer_model": ("tokenizer_model", "tokenizer", "tokenizer_name"),
            }
            prefer_ref_path = payload.get("ref_audio_path") is not None

        if fn is None:
            raise TTSProviderInitializationError(
                "Qwen3-TTS backend does not expose a generation method",
                provider=self.PROVIDER_KEY,
            )

        try:
            kwargs = self._build_call_kwargs(fn, payload, alias_map, prefer_ref_path=prefer_ref_path)
            result = await self._invoke_backend(fn, kwargs)
            return self._coerce_audio_output(result)
        except Exception as exc:
            raise TTSGenerationError(
                "Qwen3-TTS generation failed",
                provider=self.PROVIDER_KEY,
                details={"error": str(exc)},
            ) from exc
        finally:
            if cleanup_path:
                Path(cleanup_path).unlink(missing_ok=True)

    async def _generate_with_upstream_runtime(
        self,
        *,
        request: TTSRequest,
        resolved_model: str,
        mode: str,
    ) -> TTSResponse:
        logger.info(
            f"{self.provider_name}: request model={resolved_model}, mode={mode}, device={self.device}, "
            f"format={request.format.value}, stream={bool(request.stream)}"
        )
        if self._backend is None:
            raise TTSProviderInitializationError(
                "Qwen3-TTS backend is not configured in this build",
                provider=self.PROVIDER_KEY,
            )
        if request.stream:
            if request.format == AudioFormat.PCM:
                pcm_stream = self._generate_pcm_stream(request, resolved_model)

                async def _pcm_bytes() -> AsyncGenerator[bytes, None]:
                    async for chunk in pcm_stream:
                        if chunk is None:
                            continue
                        if isinstance(chunk, (bytes, bytearray)):
                            yield bytes(chunk)
                        else:
                            yield np.asarray(chunk, dtype=np.int16).tobytes()

                audio_stream = _pcm_bytes()
                return TTSResponse(
                    audio_stream=audio_stream,
                    format=request.format,
                    sample_rate=self.sample_rate,
                    provider=self.PROVIDER_KEY,
                    model=resolved_model,
                )

            can_transcode, transcode_err = self._can_stream_transcode(request.format)
            if can_transcode:
                pcm_stream = self._generate_pcm_stream(request, resolved_model)
                audio_stream = self._stream_transcoded_pcm(pcm_stream, request.format)
                return TTSResponse(
                    audio_stream=audio_stream,
                    format=request.format,
                    sample_rate=self.sample_rate,
                    provider=self.PROVIDER_KEY,
                    model=resolved_model,
                )

            logger.warning(
                f"{self.provider_name}: streaming transcode unavailable for {request.format.value}; "
                f"falling back to buffered streaming. error={transcode_err}"
            )
            try:
                pcm_audio = await self._generate_pcm(request, resolved_model)
                audio_bytes = await self.convert_audio_format(
                    pcm_audio,
                    source_format=AudioFormat.PCM,
                    target_format=request.format,
                    sample_rate=self.sample_rate,
                )
            except Exception as exc:
                raise TTSStreamingError(
                    "Qwen3-TTS streaming transcode failed",
                    provider=self.PROVIDER_KEY,
                    details={"format": request.format.value, "error": str(exc)},
                ) from exc

            audio_stream = self._chunk_bytes(audio_bytes)
            return TTSResponse(
                audio_stream=audio_stream,
                format=request.format,
                sample_rate=self.sample_rate,
                provider=self.PROVIDER_KEY,
                model=resolved_model,
                metadata={
                    "streaming_fallback": "buffered",
                    "streaming_fallback_reason": transcode_err or "transcoder_unavailable",
                },
            )

        pcm_audio = await self._generate_pcm(request, resolved_model)
        audio_bytes = await self.convert_audio_format(
            pcm_audio,
            source_format=AudioFormat.PCM,
            target_format=request.format,
            sample_rate=self.sample_rate,
        )
        return TTSResponse(
            audio_content=audio_bytes,
            format=request.format,
            sample_rate=self.sample_rate,
            provider=self.PROVIDER_KEY,
            model=resolved_model,
        )

    async def generate(self, request: TTSRequest) -> TTSResponse:
        """Generate speech from text using the selected runtime."""
        if not await self.ensure_initialized():
            raise TTSProviderInitializationError(
                "Qwen3-TTS adapter not initialized",
                provider=self.PROVIDER_KEY,
            )
        resolved_model = self._resolve_model(request)
        mode = self._resolve_mode(resolved_model)
        runtime = self._get_runtime()
        return await runtime.generate(request, resolved_model, mode)
