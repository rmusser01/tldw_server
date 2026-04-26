"""
STT Provider adapter and registry.

This module introduces a lightweight adapter/registry for STT providers as
described in `Docs/Product/STT_Module_PRD.md`. It focuses on capability
discovery and config-driven provider selection without pulling in heavy ML
dependencies. Transcription methods will be layered on gradually.
"""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from tldw_Server_API.app.core.Infrastructure.provider_registry import ProviderRegistryBase
from tldw_Server_API.app.core.config import get_stt_config
from tldw_Server_API.app.core.exceptions import BadRequestError, CancelCheckError, TranscriptionCancelled
from tldw_Server_API.app.core.Ingestion_Media_Processing.path_utils import resolve_safe_local_path

def _fallback_parse_transcription_model(
    model_name: str,
) -> tuple[str, str, str | None]:
    model_name = (model_name or "").strip()
    lowered = model_name.lower() or "whisper-1"

    if "parakeet" in lowered:
        if lowered.endswith("-mlx"):
            return "parakeet", "parakeet", "mlx"
        if lowered.endswith("-onnx"):
            return "parakeet", "parakeet", "onnx"
        if lowered.endswith("-cuda"):
            return "parakeet", "parakeet", "cuda"
        if lowered.endswith("-standard") or "nemo-parakeet" in lowered:
            return "parakeet", "parakeet", "standard"
        try:
            stt_cfg = get_stt_config() or {}
        except Exception:
            stt_cfg = {}
        variant = _normalize_parakeet_variant(stt_cfg.get("nemo_model_variant"))
        return "parakeet", "parakeet", variant

    if "canary" in lowered:
        return "canary", "canary", "standard"

    if lowered in {"qwen", "qwen2audio", "qwen2-audio"} or "qwen2audio" in lowered or "qwen2-audio" in lowered:
        return "qwen2audio", ("qwen2audio" if lowered == "qwen" else model_name or "qwen2audio"), None

    if "vibevoice" in lowered:
        if lowered in {"vibevoice", "vibevoice-asr", "vibevoice_asr"}:
            try:
                stt_cfg = get_stt_config() or {}
            except Exception:
                stt_cfg = {}
            model_id = str(stt_cfg.get("vibevoice_model_id", "microsoft/VibeVoice-ASR")).strip()
            return "vibevoice", model_id or "microsoft/VibeVoice-ASR", None
        return "vibevoice", model_name, None

    if "qwen3" in lowered and "asr" in lowered:
        try:
            stt_cfg = get_stt_config() or {}
        except Exception:
            stt_cfg = {}
        base_path = str(stt_cfg.get("qwen3_asr_model_path", "./models/qwen3_asr/1.7B")).strip()
        if not base_path:
            base_path = "./models/qwen3_asr/1.7B"
        if "0.6b" in lowered:
            if "1.7B" in base_path or "1.7b" in base_path:
                model_path = base_path.replace("1.7B", "0.6B").replace("1.7b", "0.6b")
            else:
                model_path = str(Path(base_path).parent / "0.6B")
        else:
            model_path = base_path
        return "qwen3-asr", model_path, None

    if lowered.startswith("external:"):
        return "external", model_name or "external:default", None

    # Default to Whisper-family identifiers when the real parser is unavailable.
    return "whisper", model_name or "whisper-1", None


def _parse_transcription_model(model_name: str) -> tuple[str, str, str | None]:
    """
    Resolve model names without importing heavy STT modules at import time.

    Importing Audio_Transcription_Lib can pull in torch/ctranslate stacks in
    some environments; keep that dependency lazy so API module import remains
    stable for lightweight endpoints and tests.
    """
    try:
        # Reuse the central model-name parser so HTTP/OpenAI-style model
        # identifiers resolve consistently across REST, ingestion, and jobs.
        from .Audio_Transcription_Lib import parse_transcription_model as _real_parser
    except Exception:
        return _fallback_parse_transcription_model(model_name)
    try:
        return _real_parser(model_name)
    except Exception:
        return _fallback_parse_transcription_model(model_name)


def parse_transcription_model(model_name: str) -> tuple[str, str, str | None]:
    """Public parser hook kept for test monkeypatch/backwards compatibility."""
    return _parse_transcription_model(model_name)


_SUPPORTED_PARAKEET_VARIANTS = {"standard", "onnx", "mlx", "cuda"}
_CANONICAL_PARAKEET_ONNX_MODEL = "parakeet-tdt-0.6b-v3-onnx"
_STT_PROVIDER_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _segment_text_value(segment: dict[str, Any]) -> str:
    """Return segment text across legacy and normalized keys."""
    return str(segment.get("text") or segment.get("Text") or "").strip()


def _raise_if_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is None:
        return
    try:
        result = cancel_check()
        if inspect.isawaitable(result):
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = None
            if loop is not None and loop.is_running():
                raise CancelCheckError(
                    "cancel_check must be synchronous; received awaitable while event loop is running"
                )
            should_cancel = asyncio.run(result)
        else:
            should_cancel = bool(result)
    except CancelCheckError:
        raise
    except Exception as exc:
        raise CancelCheckError(f"cancel_check failed: {exc}") from exc
    if should_cancel:
        raise TranscriptionCancelled("Cancelled by user")


def _normalize_parakeet_variant(raw: str | None) -> str:
    variant = (raw or "").strip().lower()
    if not variant or variant not in _SUPPORTED_PARAKEET_VARIANTS:
        return "standard"
    return variant


def _parakeet_model_name_for_variant(variant: str) -> str:
    normalized = _normalize_parakeet_variant(variant)
    if normalized == "onnx":
        return _CANONICAL_PARAKEET_ONNX_MODEL
    return f"parakeet-{normalized}"


def _resolve_default_model_for_provider(
    provider: str,
    stt_cfg: dict[str, Any],
) -> tuple[str, str | None]:
    normalized = (provider or "").strip().lower()
    if normalized == SttProviderName.PARAKEET.value:
        variant = _normalize_parakeet_variant(stt_cfg.get("nemo_model_variant"))
        return _parakeet_model_name_for_variant(variant), variant
    if normalized == SttProviderName.CANARY.value:
        return "nemo-canary-1b", "standard"
    if normalized == SttProviderName.QWEN2AUDIO.value:
        return "qwen2audio", None
    if normalized == SttProviderName.QWEN3_ASR.value:
        # Default to 1.7B model (production quality)
        model_path = str(stt_cfg.get("qwen3_asr_model_path", "./models/qwen3_asr/1.7B")).strip()
        return model_path or "qwen3-asr-1.7b", None
    if normalized == SttProviderName.VIBEVOICE.value:
        model_id = str(stt_cfg.get("vibevoice_model_id", "microsoft/VibeVoice-ASR")).strip()
        return model_id or "microsoft/VibeVoice-ASR", None
    if normalized == SttProviderName.EXTERNAL.value:
        return "external:default", None
    return "", None


class SttProviderName(str, Enum):
    """Canonical provider identifiers used across the STT module."""

    FASTER_WHISPER = "faster-whisper"
    PARAKEET = "parakeet"
    CANARY = "canary"
    QWEN2AUDIO = "qwen2audio"
    QWEN3_ASR = "qwen3-asr"
    VIBEVOICE = "vibevoice"
    EXTERNAL = "external"


@dataclass(frozen=True)
class SttProviderCapabilities:
    """
    Capability metadata for an STT provider.

    This is intentionally small and focused on the questions higher-level code
    needs to answer when routing work: can this provider handle batch
    transcriptions, streaming, and diarization?
    """

    name: SttProviderName
    supports_batch: bool = True
    supports_streaming: bool = False
    supports_diarization: bool = False
    notes: str | None = None


class SttProviderAdapter(ABC):
    """
    Abstract base class for STT provider adapters.

    Concrete adapters will gradually add batch and streaming entrypoints
    (e.g. `transcribe_batch`, `create_streaming_transcriber`). For the first
    iteration we only require `get_capabilities` so that provider selection
    and capability discovery can be unified and tested.
    """

    def __init__(self, name: SttProviderName) -> None:
        self._name = name

    @property
    def name(self) -> SttProviderName:
        return self._name

    @abstractmethod
    def get_capabilities(self) -> SttProviderCapabilities:
        """Return capability metadata for this provider."""

    @abstractmethod
    def transcribe_batch(
        self,
        audio_path: str,
        *,
        model: str | None = None,
        language: str | None = None,
        task: str = "transcribe",
        word_timestamps: bool = False,
        prompt: str | None = None,
        hotwords: Sequence[str] | None = None,
        base_dir: Path | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """
        Perform a batch transcription and return a normalized artifact.

        Normalized artifact shape (aligned with STT Module PRD):
        {
          "text": str,
          "language": Optional[str],
          "segments": list,
          "diarization": {"enabled": bool, "speakers": Optional[int]},
          "usage": {"duration_ms": Optional[int], "tokens": Optional[int]},
          "metadata": {...},
        }
        """


class FasterWhisperAdapter(SttProviderAdapter):
    """Adapter metadata for faster-whisper based transcription."""

    def __init__(self) -> None:
        super().__init__(SttProviderName.FASTER_WHISPER)

    def get_capabilities(self) -> SttProviderCapabilities:
        # Batch + streaming are supported; diarization is available via the
        # separate diarization library integration.
        return SttProviderCapabilities(
            name=self.name,
            supports_batch=True,
            supports_streaming=True,
            supports_diarization=True,
        )

    def transcribe_batch(
        self,
        audio_path: str,
        *,
        model: str | None = None,
        language: str | None = None,
        task: str = "transcribe",
        word_timestamps: bool = False,
        prompt: str | None = None,
        hotwords: Sequence[str] | None = None,
        base_dir: Path | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        # We reuse the core speech_to_text helper so behavior stays aligned
        # with existing REST/media ingestion flows.
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (  # type: ignore
            speech_to_text as fw_speech_to_text,
        )
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
            strip_whisper_metadata_header,
        )

        # Map task to STT language handling:
        #  - transcribe: honor explicit language when provided
        #  - translate: let backend auto-detect source language
        selected_lang = None if task == "translate" else language or None

        model_name = model or "distil-large-v3"
        _raise_if_cancelled(cancel_check)
        result = fw_speech_to_text(
            audio_path,
            whisper_model=model_name,
            selected_source_lang=selected_lang,
            vad_filter=False,
            diarize=False,
            word_timestamps=word_timestamps,
            return_language=True,
            initial_prompt=prompt,
            task=task,
            base_dir=base_dir,
            cancel_check=cancel_check,
        )

        segments_list, detected_lang = result
        # Strip Whisper metadata header so callers see only user content
        segments_for_response = strip_whisper_metadata_header(segments_list)
        text = " ".join(
            _segment_text_value(seg)
            for seg in segments_for_response
            if isinstance(seg, dict)
        )

        return {
            "text": text,
            "language": language or detected_lang,
            "segments": segments_for_response,
            "diarization": {"enabled": False, "speakers": None},
            "usage": {"duration_ms": None, "tokens": None},
            "metadata": {
                "provider": self.name.value,
                "model": model_name,
            },
        }


class ParakeetAdapter(SttProviderAdapter):
    """Adapter metadata for NVIDIA Parakeet models."""

    def __init__(self) -> None:
        super().__init__(SttProviderName.PARAKEET)

    def get_capabilities(self) -> SttProviderCapabilities:
        # Parakeet supports batch and streaming; diarization is not a primary
        # focus in current usage.
        return SttProviderCapabilities(
            name=self.name,
            supports_batch=True,
            supports_streaming=True,
            supports_diarization=False,
        )

    def transcribe_batch(
        self,
        audio_path: str,
        *,
        model: str | None = None,
        language: str | None = None,
        task: str = "transcribe",
        word_timestamps: bool = False,
        prompt: str | None = None,
        hotwords: Sequence[str] | None = None,
        base_dir: Path | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        # Parakeet batch flows are routed through speech_to_text's Parakeet
        # branch by encoding the model name (e.g. "parakeet-standard").
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (  # type: ignore
            speech_to_text,
        )

        if model:
            model_name = model
        else:
            try:
                stt_cfg = get_stt_config() or {}
            except _STT_PROVIDER_NONCRITICAL_EXCEPTIONS:
                stt_cfg = {}
            model_name, _ = _resolve_default_model_for_provider(self.name.value, stt_cfg)
            if not model_name:
                model_name = "parakeet-standard"
        _raise_if_cancelled(cancel_check)
        segments_list, lang = speech_to_text(
            audio_path,
            whisper_model=model_name,
            selected_source_lang=language,
            vad_filter=False,
            diarize=False,
            return_language=True,
            base_dir=base_dir,
            cancel_check=cancel_check,
        )
        text = " ".join(
            _segment_text_value(seg)
            for seg in segments_list
            if isinstance(seg, dict)
        )
        return {
            "text": text,
            "language": language or lang,
            "segments": segments_list,
            "diarization": {"enabled": False, "speakers": None},
            "usage": {"duration_ms": None, "tokens": None},
            "metadata": {
                "provider": self.name.value,
                "model": model_name,
            },
        }


class CanaryAdapter(SttProviderAdapter):
    """Adapter metadata for NVIDIA Canary models."""

    def __init__(self) -> None:
        super().__init__(SttProviderName.CANARY)

    def get_capabilities(self) -> SttProviderCapabilities:
        # Canary is used for batch multilingual transcription today; streaming
        # support may be added later.
        return SttProviderCapabilities(
            name=self.name,
            supports_batch=True,
            supports_streaming=False,
            supports_diarization=False,
        )

    def transcribe_batch(
        self,
        audio_path: str,
        *,
        model: str | None = None,
        language: str | None = None,
        task: str = "transcribe",
        word_timestamps: bool = False,
        prompt: str | None = None,
        hotwords: Sequence[str] | None = None,
        base_dir: Path | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore

        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (  # type: ignore
            transcribe_with_canary,
        )

        path_obj = Path(audio_path)
        if base_dir is not None:
            # Enforce that local audio paths stay within base_dir for path safety.
            safe_path = resolve_safe_local_path(path_obj, base_dir)
            if safe_path is None:
                raise BadRequestError(f"Audio path rejected outside base_dir: {audio_path}")
            path_obj = safe_path

        _raise_if_cancelled(cancel_check)
        try:
            audio_np, sample_rate = sf.read(str(path_obj))
        except Exception as e:
            raise BadRequestError(f"Failed to read audio file {path_obj}: {e}") from e
        if not isinstance(audio_np, np.ndarray):
            audio_np = np.array(audio_np, dtype="float32")

        # For Canary we mirror the create_transcription behavior: language
        # controls ASR language, task="translate" can be interpreted by the
        # underlying helper (if supported).
        _raise_if_cancelled(cancel_check)
        text = transcribe_with_canary(
            audio_np,
            sample_rate,
            language,
            task=task,
            target_language="en" if task == "translate" else None,
        )
        segments = [
            {
                "start_seconds": 0.0,
                "end_seconds": 0.0,
                "Text": text,
            }
        ]
        return {
            "text": text,
            "language": language or None,
            "segments": segments,
            "diarization": {"enabled": False, "speakers": None},
            "usage": {"duration_ms": None, "tokens": None},
            "metadata": {
                "provider": self.name.value,
                "model": model or "",
            },
        }


class Qwen2AudioAdapter(SttProviderAdapter):
    """Adapter metadata for Qwen2Audio models."""

    def __init__(self) -> None:
        super().__init__(SttProviderName.QWEN2AUDIO)

    def get_capabilities(self) -> SttProviderCapabilities:
        # Qwen2Audio currently exposes batch-style transcription only.
        return SttProviderCapabilities(
            name=self.name,
            supports_batch=True,
            supports_streaming=False,
            supports_diarization=False,
        )

    def transcribe_batch(
        self,
        audio_path: str,
        *,
        model: str | None = None,
        language: str | None = None,
        task: str = "transcribe",
        word_timestamps: bool = False,
        prompt: str | None = None,
        hotwords: Sequence[str] | None = None,
        base_dir: Path | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (  # type: ignore
            speech_to_text,
        )

        model_name = model or "qwen2audio"
        _raise_if_cancelled(cancel_check)
        segments_list, lang = speech_to_text(
            audio_path,
            whisper_model=model_name,
            selected_source_lang=language,
            vad_filter=False,
            diarize=False,
            return_language=True,
            base_dir=base_dir,
            cancel_check=cancel_check,
        )
        text = " ".join(
            _segment_text_value(seg)
            for seg in segments_list
            if isinstance(seg, dict)
        )
        return {
            "text": text,
            "language": language or lang,
            "segments": segments_list,
            "diarization": {"enabled": False, "speakers": None},
            "usage": {"duration_ms": None, "tokens": None},
            "metadata": {
                "provider": self.name.value,
                "model": model_name,
            },
        }


class Qwen3ASRAdapter(SttProviderAdapter):
    """Adapter for Qwen3-ASR models (1.7B and 0.6B variants).

    Features:
    - 30 languages + 22 Chinese dialects (auto-detected)
    - State-of-the-art accuracy (1.63 WER on LibriSpeech clean)
    - Optional word-level timestamps via Qwen3-ForcedAligner
    - Default model: 1.7B (production quality)
    """

    def __init__(self) -> None:
        super().__init__(SttProviderName.QWEN3_ASR)

    def get_capabilities(self) -> SttProviderCapabilities:
        # Qwen3-ASR supports batch transcription; streaming is available via vLLM HTTP
        # Word timestamps via forced aligner are supported when configured
        # Check if vLLM streaming is configured
        try:
            stt_cfg = get_stt_config() or {}
            vllm_url = str(stt_cfg.get("qwen3_asr_vllm_base_url", "")).strip()
            backend = str(stt_cfg.get("qwen3_asr_backend", "")).lower()
            streaming_available = bool(vllm_url and backend == "vllm")
        except _STT_PROVIDER_NONCRITICAL_EXCEPTIONS:
            streaming_available = False

        notes = "Qwen3-ASR: 30 languages, word timestamps via ForcedAligner"
        if streaming_available:
            notes += "; streaming via vLLM HTTP"

        return SttProviderCapabilities(
            name=self.name,
            supports_batch=True,
            supports_streaming=streaming_available,
            supports_diarization=False,
            notes=notes,
        )

    def transcribe_batch(
        self,
        audio_path: str,
        *,
        model: str | None = None,
        language: str | None = None,
        task: str = "transcribe",
        word_timestamps: bool = False,
        prompt: str | None = None,
        hotwords: Sequence[str] | None = None,
        base_dir: Path | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Qwen3ASR import (
            transcribe_with_qwen3_asr,
        )

        # Resolve model path from config if not provided
        if model:
            model_path = model
        else:
            try:
                stt_cfg = get_stt_config() or {}
            except _STT_PROVIDER_NONCRITICAL_EXCEPTIONS:
                stt_cfg = {}
            model_path, _ = _resolve_default_model_for_provider(self.name.value, stt_cfg)
            if not model_path:
                model_path = "./models/qwen3_asr/1.7B"

        _raise_if_cancelled(cancel_check)
        artifact = transcribe_with_qwen3_asr(
            audio_path,
            model_path=model_path,
            language=language,
            word_timestamps=word_timestamps,
            base_dir=base_dir,
            cancel_check=cancel_check,
        )
        if not isinstance(artifact, dict):
            raise BadRequestError("Qwen3-ASR transcription did not return a valid artifact")
        return artifact


class VibeVoiceAdapter(SttProviderAdapter):
    """Adapter metadata for VibeVoice-ASR models."""

    def __init__(self) -> None:
        super().__init__(SttProviderName.VIBEVOICE)

    def get_capabilities(self) -> SttProviderCapabilities:
        # VibeVoice-ASR is batch-first and includes diarization metadata.
        return SttProviderCapabilities(
            name=self.name,
            supports_batch=True,
            supports_streaming=False,
            supports_diarization=True,
            notes="VibeVoice-ASR supports batch transcription with diarization metadata; streaming is not supported.",
        )

    def transcribe_batch(
        self,
        audio_path: str,
        *,
        model: str | None = None,
        language: str | None = None,
        task: str = "transcribe",
        word_timestamps: bool = False,
        prompt: str | None = None,
        hotwords: Sequence[str] | None = None,
        base_dir: Path | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_VibeVoice import (  # type: ignore
            transcribe_with_vibevoice,
        )

        if model:
            model_name = model
        else:
            try:
                stt_cfg = get_stt_config() or {}
            except _STT_PROVIDER_NONCRITICAL_EXCEPTIONS:
                stt_cfg = {}
            model_name, _ = _resolve_default_model_for_provider(self.name.value, stt_cfg)
            if not model_name:
                model_name = "microsoft/VibeVoice-ASR"

        _raise_if_cancelled(cancel_check)
        artifact = transcribe_with_vibevoice(
            audio_path,
            model_id=model_name,
            language=language,
            hotwords=list(hotwords) if hotwords else None,
            base_dir=base_dir,
            cancel_check=cancel_check,
        )
        if not isinstance(artifact, dict):
            raise BadRequestError("VibeVoice-ASR transcription did not return a valid artifact")
        return artifact


class ExternalAdapter(SttProviderAdapter):
    """Adapter metadata for external/custom STT providers."""

    def __init__(self) -> None:
        super().__init__(SttProviderName.EXTERNAL)

    def get_capabilities(self) -> SttProviderCapabilities:
        # External providers are assumed to handle batch requests; streaming
        # and diarization support depend on the concrete integration.
        return SttProviderCapabilities(
            name=self.name,
            supports_batch=True,
            supports_streaming=False,
            supports_diarization=False,
        )

    def transcribe_batch(
        self,
        audio_path: str,
        *,
        model: str | None = None,
        language: str | None = None,
        task: str = "transcribe",
        word_timestamps: bool = False,
        prompt: str | None = None,
        hotwords: Sequence[str] | None = None,
        base_dir: Path | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (  # type: ignore
            transcribe_with_external_provider,
        )

        provider_name = "default"
        model_id = model or "whisper-1"
        if model_id.startswith("external:"):
            provider_name = model_id.split(":", 1)[1] or "default"

        # Pass base_dir so external providers validate local paths consistently.
        _raise_if_cancelled(cancel_check)
        text = transcribe_with_external_provider(
            audio_path,
            provider_name=provider_name,
            base_dir=base_dir,
        )
        segments = [
            {
                "start_seconds": 0.0,
                "end_seconds": 0.0,
                "Text": text,
            }
        ]
        return {
            "text": text,
            "language": language or None,
            "segments": segments,
            "diarization": {"enabled": False, "speakers": None},
            "usage": {"duration_ms": None, "tokens": None},
            "metadata": {
                "provider": self.name.value,
                "model": model_id,
                "external_provider_name": provider_name,
            },
        }

_STT_PROVIDER_ALIASES: dict[str, str] = {
    # Whisper/faster-whisper aliases
    "whisper": SttProviderName.FASTER_WHISPER.value,
    "fasterwhisper": SttProviderName.FASTER_WHISPER.value,
    "fw": SttProviderName.FASTER_WHISPER.value,
    # Nemo family aliases
    "nemo-parakeet": SttProviderName.PARAKEET.value,
    "nemo-canary": SttProviderName.CANARY.value,
    # VibeVoice aliases
    "vibevoice-asr": SttProviderName.VIBEVOICE.value,
    # Qwen3-ASR aliases
    "qwen3asr": SttProviderName.QWEN3_ASR.value,
    "qwen-3-asr": SttProviderName.QWEN3_ASR.value,
    # External aliases
    "external-provider": SttProviderName.EXTERNAL.value,
}


class SttProviderRegistry:
    """
    Registry for STT providers and their adapters.

    This registry is intentionally lightweight: it does not instantiate heavy
    ML models and only exposes capability metadata and config-driven selection.
    """

    DEFAULT_ADAPTERS: dict[str, type[SttProviderAdapter]] = {
        SttProviderName.FASTER_WHISPER.value: FasterWhisperAdapter,
        SttProviderName.PARAKEET.value: ParakeetAdapter,
        SttProviderName.CANARY.value: CanaryAdapter,
        SttProviderName.QWEN2AUDIO.value: Qwen2AudioAdapter,
        SttProviderName.QWEN3_ASR.value: Qwen3ASRAdapter,
        SttProviderName.VIBEVOICE.value: VibeVoiceAdapter,
        SttProviderName.EXTERNAL.value: ExternalAdapter,
    }

    def __init__(self) -> None:
        self._base: ProviderRegistryBase[SttProviderAdapter] = ProviderRegistryBase(
            aliases=_STT_PROVIDER_ALIASES,
            adapter_validator=lambda adapter: isinstance(adapter, SttProviderAdapter),
            provider_enabled_callback=self._is_provider_enabled_by_config,
        )
        for provider_name, adapter_spec in self.DEFAULT_ADAPTERS.items():
            self._base.register_adapter(provider_name, adapter_spec)

    def normalize_provider_name(self, provider_name: str | None) -> str:
        """
        Normalize provider identifiers using the shared base registry.
        """
        return self._base.resolve_provider_name(provider_name)

    def _is_provider_enabled_by_config(self, provider_name: str) -> bool | None:
        """
        STT keeps provider enablement decisions outside registry lookup today.

        This callback intentionally returns no opinion so existing STT
        precedence remains unchanged while still wiring the shared callback
        interface required for cross-domain parity.
        """
        _ = provider_name
        return None

    def register_adapter(
        self,
        provider_name: str,
        adapter: Any,
        *,
        aliases: list[str] | tuple[str, ...] | set[str] | None = None,
        enabled: bool = True,
    ) -> None:
        normalized = self.normalize_provider_name(provider_name)
        if not normalized:
            raise ValueError("Provider name must be non-empty")
        self._base.register_adapter(normalized, adapter, aliases=aliases, enabled=enabled)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def get_default_provider_name(self) -> str:
        """
        Return the default provider name based on `[STT-Settings]`.

        This mirrors the behavior of the config loader:
        - Prefer `default_transcriber` when present.
        - Fall back to `default_stt_provider`.
        - Final fallback is 'faster-whisper'.
        """
        cfg: dict[str, Any]
        try:
            cfg = get_stt_config() or {}
        except _STT_PROVIDER_NONCRITICAL_EXCEPTIONS:
            cfg = {}

        raw_default = cfg.get("default_transcriber") or cfg.get("default_stt_provider") or "faster-whisper"
        normalized = self.normalize_provider_name(raw_default)
        return normalized or "faster-whisper"

    def get_adapter(self, provider_name: str | None = None) -> SttProviderAdapter:
        """
        Return the adapter for the given provider name.

        When `provider_name` is None or unknown, the default provider is
        resolved via config and used. As a final safety net, the
        'faster-whisper' adapter is returned.
        """
        key = self.normalize_provider_name(provider_name) if provider_name else self.get_default_provider_name()

        adapter = self._base.get_adapter(key)
        if adapter is not None:
            return adapter

        # Defensive fallback to faster-whisper
        fallback = self._base.get_adapter(SttProviderName.FASTER_WHISPER.value)
        if fallback is not None:
            return fallback
        raise RuntimeError("faster-whisper adapter is not available")

    def get_capabilities(self, provider_name: str | None = None) -> SttProviderCapabilities:
        """
        Convenience helper to fetch capability metadata for a provider.
        """
        return self.get_adapter(provider_name).get_capabilities()

    def get_status(self, provider_name: str | None) -> str:
        """
        Return canonical availability status for a provider.
        """
        return self._base.get_status(provider_name).value

    def list_capabilities(self, *, include_disabled: bool = True) -> list[dict[str, Any]]:
        """
        Return capability envelopes for all registered STT providers.
        """
        return self._base.list_capabilities(
            capability_getter=lambda adapter: adapter.get_capabilities(),
            include_disabled=include_disabled,
        )

    def resolve_provider_for_model(self, model_name: str | None) -> tuple[str, str, str | None]:
        """
        Resolve an HTTP/OpenAI-style model name to (provider, model, variant).

        This wraps `parse_transcription_model` so that all call sites rely on
        a single mapping from model identifiers to providers. The provider
        name returned is normalized (e.g. 'faster-whisper').
        """
        if not model_name or not str(model_name).strip():
            # When no model is specified, return the default provider and a
            # config-aware default model for non-Whisper backends. Whisper
            # defaults are handled by higher-level callers so they can apply
            # endpoint-specific alias mapping.
            provider = self.get_default_provider_name()
            try:
                stt_cfg = get_stt_config() or {}
            except _STT_PROVIDER_NONCRITICAL_EXCEPTIONS:
                stt_cfg = {}
            model, variant = _resolve_default_model_for_provider(provider, stt_cfg)
            return provider, model, variant

        try:
            normalized_name = (model_name or "").strip()
            lowered = normalized_name.lower()
            # Preserve legacy alias: bare "qwen" maps to Qwen2Audio.
            if lowered == "qwen":
                provider = SttProviderName.QWEN2AUDIO.value
                return provider, "qwen2audio", None
            # Handle qwen3-asr model names
            if lowered.startswith("qwen3-asr") or lowered.startswith("qwen3_asr"):
                provider = SttProviderName.QWEN3_ASR.value
                # Map model name to HuggingFace path
                if "0.6b" in lowered:
                    return provider, "Qwen/Qwen3-ASR-0.6B", None
                elif "1.7b" in lowered:
                    return provider, "Qwen/Qwen3-ASR-1.7B", None
                else:
                    # Default to 1.7B
                    return provider, "Qwen/Qwen3-ASR-1.7B", None
            if lowered.startswith("external:"):
                provider = SttProviderName.EXTERNAL.value
                return provider, normalized_name, None

            raw_provider, model, variant = parse_transcription_model(normalized_name)
        except _STT_PROVIDER_NONCRITICAL_EXCEPTIONS:
            # Defensive: treat unknown models as Whisper-family
            raw_provider, model, variant = "whisper", (model_name or "").strip(), None

        provider = self.normalize_provider_name(raw_provider)
        if provider == "whisper":
            # Internally, Whisper-family models are handled via faster-whisper.
            provider = SttProviderName.FASTER_WHISPER.value
        return provider, model, variant


_REGISTRY: SttProviderRegistry | None = None


def get_stt_provider_registry() -> SttProviderRegistry:
    """
    Return the process-wide STT provider registry.

    This is a simple singleton to keep lookup overhead low while still
    allowing tests to reset/monkeypatch behavior if needed.
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = SttProviderRegistry()
    return _REGISTRY


def resolve_default_transcription_model(fallback_whisper_model: str) -> str:
    """
    Resolve a config-aware default transcription model string.

    For non-Whisper providers, this returns a provider-specific default model
    (e.g., "parakeet-mlx" when configured). For Whisper defaults, callers
    supply the endpoint-specific fallback (e.g., "whisper-1" or a faster-whisper
    model size).
    """
    registry = get_stt_provider_registry()
    try:
        stt_cfg = get_stt_config() or {}
    except _STT_PROVIDER_NONCRITICAL_EXCEPTIONS:
        stt_cfg = {}

    configured_batch_model = str(stt_cfg.get("default_batch_transcription_model", "")).strip()
    if configured_batch_model:
        return configured_batch_model

    provider = registry.get_default_provider_name()
    model, _ = _resolve_default_model_for_provider(provider, stt_cfg)
    if provider == SttProviderName.FASTER_WHISPER.value:
        return fallback_whisper_model
    return model or fallback_whisper_model


def reset_stt_provider_registry() -> None:
    """
    Reset the global registry (used by tests).
    """
    global _REGISTRY
    _REGISTRY = None
