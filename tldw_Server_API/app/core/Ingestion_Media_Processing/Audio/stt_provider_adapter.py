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
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from tldw_Server_API.app.core.config import (
    get_stt_config,
    resolve_default_transcription_model_setting,
)
from tldw_Server_API.app.core.exceptions import (
    BadRequestError,
    CancelCheckError,
    STTExecutionPlanError,
    STTExecutionUnsupportedError,
    TranscriptionCancelled,
)
from tldw_Server_API.app.core.exceptions import (
    STTTranscriptionError as STTTranscriptionError,
)
from tldw_Server_API.app.core.Infrastructure.provider_registry import ProviderRegistryBase
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
    SttActualExecution as SttActualExecution,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
    SttAudioEgress as SttAudioEgress,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
    SttBatchExecutionPlan,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
    SttExecutionDescriptor as SttExecutionDescriptor,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
    SttExecutionRoute as SttExecutionRoute,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
    SttLoadedRuntime as SttLoadedRuntime,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
    SttPlanScalar as SttPlanScalar,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
    SttTranscriptionOutcome as SttTranscriptionOutcome,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
    _classify_audio_egress as _classify_audio_egress,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
    _normalize_audio_endpoint as _normalize_audio_endpoint,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
    _resolve_audio_transcription_endpoint as _resolve_audio_transcription_endpoint,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
    actual_execution_from_route as actual_execution_from_route,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
    finalize_stt_artifact as finalize_stt_artifact,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
    require_local_execution_route as require_local_execution_route,
)
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
_IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

_LOCAL_RUNTIME_MODEL_PATH = "model_path"
_LOCAL_RUNTIME_REVISION = "revision"
_LOCAL_RUNTIME_DEVICE = "device"
_LOCAL_RUNTIME_DEVICE_MAP = "device_map"
_LOCAL_RUNTIME_COMPUTE_TYPE = "compute_type"
_LOCAL_RUNTIME_DTYPE = "dtype"
_LOCAL_RUNTIME_VARIANT = "variant"
_EXTERNAL_RUNTIME_API_KEY = "external_api_key"
_EXTERNAL_RUNTIME_BASE_URL = "external_base_url"
_EXTERNAL_RUNTIME_HEADER_NAMES = "external_header_names"
_EXTERNAL_RUNTIME_HEADER_VALUES = "external_header_values"
_EXTERNAL_RUNTIME_LANGUAGE = "external_language"
_EXTERNAL_RUNTIME_MAX_RETRIES = "external_max_retries"
_EXTERNAL_RUNTIME_MODEL = "external_model"
_EXTERNAL_RUNTIME_PROMPT = "external_prompt"
_EXTERNAL_RUNTIME_PROVIDER = "external_provider"
_EXTERNAL_RUNTIME_RESPONSE_FORMAT = "external_response_format"
_EXTERNAL_RUNTIME_TEMPERATURE = "external_temperature"
_EXTERNAL_RUNTIME_TIMEOUT = "external_timeout"
_EXTERNAL_RUNTIME_TRANSPORT = "external_transport"
_EXTERNAL_RUNTIME_VERIFY_SSL = "external_verify_ssl"


def _segment_text_value(segment: dict[str, Any]) -> str:
    """Return segment text across legacy and normalized keys."""
    return str(segment.get("text") or segment.get("Text") or "").strip()


def _safe_requested_model_label(model: str) -> str:
    normalized = model.strip()
    if (
        not normalized
        or normalized.startswith((".", "/", "~"))
        or "\\" in normalized
        or "://" in normalized
        or normalized.count("/") > 1
    ):
        return "local-model"
    return normalized


def _validate_execution_plan_request(
    adapter: SttProviderAdapter,
    plan: SttBatchExecutionPlan,
    *,
    model: str | None,
    language: str | None,
    task: str,
    word_timestamps: bool,
    prompt: str | None,
    hotwords: Sequence[str] | None,
) -> None:
    """Fail before provider entry when a supplied plan does not match the call."""
    descriptor = plan.descriptor
    route = descriptor.primary_route
    if (
        descriptor.resolved_provider != adapter.name.value
        or route.provider != adapter.name.value
    ):
        raise STTExecutionPlanError(
            f"Execution plan provider does not match {adapter.name.value}"
        )
    if route.model_label != descriptor.resolved_model_label:
        raise STTExecutionPlanError(
            "Execution plan route model does not match resolved model"
        )
    if model is not None:
        model_label = _safe_requested_model_label(model)
        if model_label != descriptor.requested_model_label:
            raise STTExecutionPlanError("Execution plan model does not match request")
    if (
        plan.task != task
        or plan.language != language
        or plan.word_timestamps != word_timestamps
        or plan.prompt != prompt
        or plan.hotwords != tuple(hotwords or ())
    ):
        raise STTExecutionPlanError(
            "Execution plan semantic settings do not match request"
        )


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


def _resolve_adapter_audio_path(audio_path: str, base_dir: Path | None) -> Path:
    """Resolve an adapter input path and enforce base_dir containment when provided."""
    path_obj = Path(audio_path)
    if base_dir is None:
        return path_obj

    safe_path = resolve_safe_local_path(path_obj, base_dir)
    if safe_path is None:
        raise BadRequestError(f"Audio path rejected outside base_dir: {audio_path}")
    return safe_path


def _canonicalize_wav_for_soundfile_adapter(audio_path: str, base_dir: Path | None) -> Path:
    """Convert compressed adapter input to a fresh WAV path accepted by soundfile loaders."""
    path_obj = _resolve_adapter_audio_path(audio_path, base_dir)
    if path_obj.suffix.lower() == ".wav":
        return path_obj

    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (  # type: ignore
            ConversionError,
            convert_to_wav,
        )
    except ImportError as exc:
        raise BadRequestError("Audio WAV conversion is not available for this STT provider") from exc

    try:
        converted_path = convert_to_wav(
            str(path_obj),
            offset=0,
            overwrite=True,
            base_dir=base_dir,
        )
    except (ConversionError, OSError, RuntimeError, ValueError) as exc:
        raise BadRequestError(f"Failed to convert audio file to WAV: {exc}") from exc

    if not converted_path:
        raise BadRequestError("Audio conversion did not produce a usable WAV file")

    converted_obj = Path(converted_path)
    if base_dir is not None:
        safe_converted = resolve_safe_local_path(converted_obj, base_dir)
        if safe_converted is None:
            raise BadRequestError(f"Converted audio path rejected outside base_dir: {converted_path}")
        converted_obj = safe_converted

    if converted_obj.suffix.lower() != ".wav" or not converted_obj.exists():
        raise BadRequestError("Audio conversion did not produce a usable WAV file")

    return converted_obj


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


def _require_benchmark_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized not in {"neutral-v1", "production-v1"}:
        raise STTExecutionUnsupportedError(
            f"Unsupported STT benchmark mode: {mode}"
        )
    if normalized == "production-v1":
        raise STTExecutionUnsupportedError(
            "production-v1 is unsupported for local STT providers"
        )
    return normalized


def _primary_language(language: str | None) -> str | None:
    if language is None:
        return None
    return str(language).replace("_", "-").split("-", 1)[0].strip().lower()


def _require_existing_path(
    value: object,
    *,
    provider: str,
    suffix: str | None = None,
    directory: bool = False,
) -> Path:
    raw = str(value or "").strip()
    path = Path(raw)
    valid = path.is_dir() if directory else path.is_file()
    if suffix is not None:
        valid = valid and path.suffix.lower() == suffix
    if not raw or not valid:
        description = "directory" if directory else f"{suffix or ''} artifact"
        raise STTExecutionUnsupportedError(
            f"{provider} requires an explicit existing local {description}"
        )
    return path.resolve()


def _require_planned_device(
    device: str,
    *,
    provider: str,
    allowed: set[str],
) -> str:
    if device not in allowed:
        raise STTExecutionUnsupportedError(
            f"{provider} planned device is unsupported"
        )
    return device


def _require_planned_precision(
    value: str,
    *,
    provider: str,
    label: str,
    allowed: set[str],
) -> str:
    if value not in allowed:
        raise STTExecutionUnsupportedError(
            f"{provider} planned {label} is unsupported"
        )
    return value


def _plan_semantics(
    *,
    task: str,
    language: str | None,
    word_timestamps: bool,
    prompt: str | None,
    hotwords: Sequence[str] | None,
    diarization: bool,
    fixed_english: bool,
) -> tuple[
    str,
    str | None,
    bool,
    str | None,
    tuple[str, ...],
    bool,
    tuple[tuple[str, SttPlanScalar], ...],
]:
    if fixed_english and _primary_language(language) != "en":
        raise STTExecutionUnsupportedError(
            "This provider supports neutral-v1 only for English language tags"
        )
    decoding = (
        (("language_contract", "fixed:en"),)
        if fixed_english
        else ()
    )
    return (
        "transcribe",
        language,
        False,
        None,
        (),
        False,
        decoding,
    )


def _build_local_plan(
    *,
    provider: str,
    requested_model: str,
    resolved_model: str,
    backend: str,
    model_path: Path | str,
    device: str | None,
    compute_type: str | None,
    dtype: str | None,
    revision: str | None,
    task: str,
    language: str | None,
    word_timestamps: bool,
    prompt: str | None,
    hotwords: Sequence[str] | None,
    diarization: bool,
    fixed_english: bool,
    runtime_settings: dict[str, SttPlanScalar],
    source_modules: Sequence[str],
    dependency_distributions: Sequence[str],
) -> SttBatchExecutionPlan:
    (
        task,
        language,
        word_timestamps,
        prompt,
        planned_hotwords,
        diarization,
        decoding_settings,
    ) = _plan_semantics(
        task=task,
        language=language,
        word_timestamps=word_timestamps,
        prompt=prompt,
        hotwords=hotwords,
        diarization=diarization,
        fixed_english=fixed_english,
    )
    artifact_id = (
        revision
        if revision is not None and _IMMUTABLE_REVISION_RE.fullmatch(revision)
        else None
    )
    identity_resolved = artifact_id is not None
    decoding_ids = tuple(key for key, _value in decoding_settings)
    route = SttExecutionRoute(
        route_id="local-1",
        provider=provider,
        model_label=resolved_model,
        artifact_id=artifact_id,
        identity_resolved=identity_resolved,
        backend=backend,
        source="local",
        audio_egress=SttAudioEgress.NONE,
        endpoint_id=None,
        device=device,
        compute_type=compute_type,
        dtype=dtype,
        decoding_ids=decoding_ids,
        local_model_available=True,
        would_download=False,
    )
    descriptor = SttExecutionDescriptor(
        requested_provider=provider,
        requested_model_label=requested_model,
        resolved_provider=provider,
        resolved_model_label=resolved_model,
        routes=(route,),
        honors_task=task == "transcribe",
        honors_language=language is not None,
        honors_prompt_absence=prompt is None,
        honors_hotword_absence=not planned_hotwords,
        honors_diarization=not diarization,
        honors_word_timestamps=not word_timestamps,
        decoding_settings=decoding_settings,
        source_modules=tuple(sorted(source_modules)),
        dependency_distributions=tuple(sorted(dependency_distributions)),
    )
    runtime_values = {
        _LOCAL_RUNTIME_MODEL_PATH: str(model_path),
        **runtime_settings,
    }
    return SttBatchExecutionPlan(
        descriptor=descriptor,
        task=task,
        language=language,
        prompt=prompt,
        hotwords=planned_hotwords,
        diarization=diarization,
        word_timestamps=word_timestamps,
        runtime_settings=tuple(sorted(runtime_values.items())),
    )


def _run_local_planned_helper(
    audio_path: str,
    *,
    execution_plan: SttBatchExecutionPlan,
    base_dir: Path | None,
    cancel_check: Callable[[], bool] | None,
) -> SttTranscriptionOutcome:
    from .Audio_Transcription_Lib import speech_to_text

    runtime = execution_plan.runtime_values()
    outcome = speech_to_text(
        audio_path,
        whisper_model=str(runtime[_LOCAL_RUNTIME_MODEL_PATH]),
        selected_source_lang=execution_plan.language,
        vad_filter=False,
        diarize=execution_plan.diarization,
        word_timestamps=execution_plan.word_timestamps,
        initial_prompt=execution_plan.prompt,
        hotwords=execution_plan.hotwords,
        task=execution_plan.task,
        base_dir=base_dir,
        cancel_check=cancel_check,
        include_metadata_header=False,
        execution_plan=execution_plan,
    )
    if not isinstance(outcome, SttTranscriptionOutcome):
        raise STTExecutionPlanError(
            "Planned local STT helper did not report typed actual execution"
        )
    return outcome


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
        execution_plan: SttBatchExecutionPlan | None = None,
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

    def plan_batch_execution(
        self,
        *,
        model: str | None,
        language: str | None,
        task: str,
        word_timestamps: bool,
        prompt: str | None,
        hotwords: Sequence[str] | None,
        diarization: bool,
        mode: str,
    ) -> SttBatchExecutionPlan:
        """Fail closed until a concrete adapter exposes enforceable planning."""
        raise STTExecutionUnsupportedError(
            f"Provider {self.name.value} does not expose enforceable benchmark planning"
        )

    def _transcribe_planned_batch(
        self,
        audio_path: str,
        *,
        execution_plan: SttBatchExecutionPlan,
        base_dir: Path | None,
        cancel_check: Callable[[], bool] | None,
    ) -> SttTranscriptionOutcome:
        """Run the four native local providers through their shared plan helper."""
        if self.name not in {
            SttProviderName.FASTER_WHISPER,
            SttProviderName.PARAKEET,
            SttProviderName.CANARY,
            SttProviderName.QWEN2AUDIO,
        }:
            raise STTExecutionUnsupportedError(
                f"Provider {self.name.value} cannot yet honor planned execution"
            )
        return _run_local_planned_helper(
            audio_path,
            execution_plan=execution_plan,
            base_dir=base_dir,
            cancel_check=cancel_check,
        )

    def _run_planned_batch(
        self,
        audio_path: str,
        *,
        execution_plan: SttBatchExecutionPlan,
        model: str | None,
        language: str | None,
        task: str,
        word_timestamps: bool,
        prompt: str | None,
        hotwords: Sequence[str] | None,
        base_dir: Path | None,
        cancel_check: Callable[[], bool] | None,
    ) -> dict[str, Any]:
        """Validate, execute, and safely finalize one planned batch request."""
        _validate_execution_plan_request(
            self,
            execution_plan,
            model=model,
            language=language,
            task=task,
            word_timestamps=word_timestamps,
            prompt=prompt,
            hotwords=hotwords,
        )
        outcome = self._transcribe_planned_batch(
            audio_path,
            execution_plan=execution_plan,
            base_dir=base_dir,
            cancel_check=cancel_check,
        )
        if not isinstance(outcome, SttTranscriptionOutcome):
            raise STTExecutionPlanError(
                "Planned STT provider did not report typed actual execution"
            )
        return finalize_stt_artifact(
            outcome.artifact,
            plan=execution_plan,
            actual=outcome.actual_execution,
            runtime_mismatches=outcome.runtime_mismatches,
        )


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

    def plan_batch_execution(
        self,
        *,
        model: str | None,
        language: str | None,
        task: str,
        word_timestamps: bool,
        prompt: str | None,
        hotwords: Sequence[str] | None,
        diarization: bool,
        mode: str,
    ) -> SttBatchExecutionPlan:
        _require_benchmark_mode(mode)
        from . import Audio_Transcription_Lib as atlib

        requested = model or "distil-large-v3"
        requested_label = _safe_requested_model_label(requested)
        if requested_label == "local-model":
            try:
                model_path: Path | str = atlib.validate_whisper_model_identifier(
                    requested
                )
            except ValueError:
                raise STTExecutionUnsupportedError(
                    "faster-whisper local model artifact is invalid"
                ) from None
        else:
            if not atlib.check_model_exists(requested):
                raise STTExecutionUnsupportedError(
                    f"Whisper model {requested} is not available locally"
                )
            model_path = requested
        stt_cfg = get_stt_config() or {}
        device = str(
            stt_cfg.get("whisper_device") or atlib.processing_choice or "cpu"
        ).strip().lower()
        device = _require_planned_device(
            device,
            provider=self.name.value,
            allowed={"cpu", "cuda"},
        )
        compute_type = str(
            stt_cfg.get("whisper_compute_type") or ""
        ).strip().lower()
        if not compute_type or compute_type == "auto":
            compute_type = "float16" if "cuda" in device else "int8"
        compute_type = _require_planned_precision(
            compute_type,
            provider=self.name.value,
            label="compute type",
            allowed=(
                {"float16", "int8", "int8_float16", "float32", "bfloat16"}
                if device == "cuda"
                else {"int8", "int8_float32", "float32"}
            ),
        )
        if device == "cuda" and not atlib.faster_whisper_cuda_available():
            raise STTExecutionUnsupportedError(
                "faster-whisper planned CUDA backend is unavailable"
            )
        return _build_local_plan(
            provider=self.name.value,
            requested_model=requested_label,
            resolved_model=requested_label,
            backend="ctranslate2",
            model_path=model_path,
            device=device,
            compute_type=compute_type,
            dtype=None,
            revision=None,
            task=task,
            language=language,
            word_timestamps=word_timestamps,
            prompt=prompt,
            hotwords=hotwords,
            diarization=diarization,
            fixed_english=False,
            runtime_settings={
                _LOCAL_RUNTIME_COMPUTE_TYPE: compute_type,
                _LOCAL_RUNTIME_DEVICE: device,
            },
            source_modules=(
                "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib",
                "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter",
            ),
            dependency_distributions=("faster-whisper",),
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
        execution_plan: SttBatchExecutionPlan | None = None,
    ) -> dict[str, Any]:
        if execution_plan is not None:
            return self._run_planned_batch(
                audio_path,
                execution_plan=execution_plan,
                model=model,
                language=language,
                task=task,
                word_timestamps=word_timestamps,
                prompt=prompt,
                hotwords=hotwords,
                base_dir=base_dir,
                cancel_check=cancel_check,
            )
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
        call_kwargs: dict[str, Any] = {
            "whisper_model": model_name,
            "selected_source_lang": selected_lang,
            "vad_filter": False,
            "diarize": False,
            "word_timestamps": word_timestamps,
            "return_language": True,
            "initial_prompt": prompt,
            "task": task,
            "base_dir": base_dir,
            "cancel_check": cancel_check,
        }
        result = fw_speech_to_text(audio_path, **call_kwargs)

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

    def plan_batch_execution(
        self,
        *,
        model: str | None,
        language: str | None,
        task: str,
        word_timestamps: bool,
        prompt: str | None,
        hotwords: Sequence[str] | None,
        diarization: bool,
        mode: str,
    ) -> SttBatchExecutionPlan:
        _require_benchmark_mode(mode)
        requested = model or "parakeet-standard"
        requested_label = _safe_requested_model_label(requested)
        lowered = requested.lower()
        if (
            requested_label == "local-model"
            and Path(requested).suffix.lower() == ".nemo"
        ) or lowered in {"parakeet-standard", "nemo-parakeet"}:
            variant = "standard"
        elif lowered in {
            "parakeet-onnx",
            _CANONICAL_PARAKEET_ONNX_MODEL,
        }:
            variant = "onnx"
        elif lowered == "parakeet-mlx":
            raise STTExecutionUnsupportedError(
                "Parakeet MLX planning cannot prove local device and dtype"
            )
        elif lowered == "parakeet-cuda":
            variant = "cuda"
        else:
            raise STTExecutionUnsupportedError(
                "Unsupported Parakeet variant"
            )
        stt_cfg = get_stt_config() or {}
        if variant in {"standard", "cuda"}:
            from . import Audio_Transcription_Nemo as nemo

            path_value = (
                requested
                if requested_label == "local-model"
                else stt_cfg.get("parakeet_model_path")
            )
            model_path = nemo._require_local_nemo_path(path_value)
            backend = "nemo"
            device = (
                "cuda"
                if variant == "cuda"
                else str(stt_cfg.get("nemo_device") or "cpu").strip().lower()
            )
            dtype = str(
                stt_cfg.get("nemo_compute_type") or "float32"
            ).strip().lower()
            dtype = _require_planned_precision(
                dtype,
                provider=self.name.value,
                label="dtype",
                allowed=(
                    {"float16", "bfloat16", "float32"}
                    if device == "cuda"
                    else {"float32"}
                ),
            )
            if device == "cuda" and not nemo._torch_cuda_available(
                allow_import=False
            ):
                raise STTExecutionUnsupportedError(
                    "Parakeet planned CUDA backend is unavailable"
                )
            dependencies = ("nemo-toolkit",)
            modules = (
                "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib",
                "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo",
                "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter",
            )
        elif variant == "onnx":
            from . import Audio_Transcription_Parakeet_ONNX as parakeet_onnx

            model_path = parakeet_onnx.validate_local_onnx_artifact(
                str(stt_cfg.get("parakeet_onnx_model_id") or "")
            )
            backend = "onnxruntime"
            device = str(
                stt_cfg.get("parakeet_onnx_device") or "cpu"
            ).strip().lower()
            dtype = None
            if device == "cuda":
                get_available = getattr(
                    parakeet_onnx.ort,
                    "get_available_providers",
                    None,
                )
                available = (
                    set(map(str, get_available()))
                    if callable(get_available)
                    else set()
                )
                if "CUDAExecutionProvider" not in available:
                    raise STTExecutionUnsupportedError(
                        "Parakeet planned CUDA backend is unavailable"
                    )
            dependencies = ("onnxruntime",)
            modules = (
                "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib",
                "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX",
                "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter",
            )
        else:
            model_path = _require_existing_path(
                stt_cfg.get("mlx_model_id"),
                provider=self.name.value,
                directory=True,
            )
            backend = "mlx"
            device = "mps"
            dtype = "bfloat16"
            dependencies = ("parakeet-mlx",)
            modules = (
                "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib",
                "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX",
                "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter",
            )
        device = _require_planned_device(
            device,
            provider=self.name.value,
            allowed={"cpu", "cuda"},
        )
        return _build_local_plan(
            provider=self.name.value,
            requested_model=requested_label,
            resolved_model=requested_label,
            backend=backend,
            model_path=model_path,
            device=device,
            compute_type=None,
            dtype=dtype,
            revision=None,
            task=task,
            language=language,
            word_timestamps=word_timestamps,
            prompt=prompt,
            hotwords=hotwords,
            diarization=diarization,
            fixed_english=True,
            runtime_settings={
                _LOCAL_RUNTIME_DEVICE: device,
                _LOCAL_RUNTIME_DTYPE: dtype,
                _LOCAL_RUNTIME_VARIANT: variant,
            },
            source_modules=modules,
            dependency_distributions=dependencies,
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
        execution_plan: SttBatchExecutionPlan | None = None,
    ) -> dict[str, Any]:
        if execution_plan is not None:
            return self._run_planned_batch(
                audio_path,
                execution_plan=execution_plan,
                model=model,
                language=language,
                task=task,
                word_timestamps=word_timestamps,
                prompt=prompt,
                hotwords=hotwords,
                base_dir=base_dir,
                cancel_check=cancel_check,
            )
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
        call_kwargs = {
            "whisper_model": model_name,
            "selected_source_lang": language,
            "vad_filter": False,
            "diarize": False,
            "return_language": True,
            "base_dir": base_dir,
            "cancel_check": cancel_check,
        }
        result = speech_to_text(audio_path, **call_kwargs)
        segments_list, lang = result
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

    def plan_batch_execution(
        self,
        *,
        model: str | None,
        language: str | None,
        task: str,
        word_timestamps: bool,
        prompt: str | None,
        hotwords: Sequence[str] | None,
        diarization: bool,
        mode: str,
    ) -> SttBatchExecutionPlan:
        _require_benchmark_mode(mode)
        requested = model or "nemo-canary-1b"
        requested_label = _safe_requested_model_label(requested)
        primary_language = _primary_language(language)
        from .Audio_Transcription_Nemo import CANARY_SUPPORTED_LANG_CODES

        if primary_language not in CANARY_SUPPORTED_LANG_CODES:
            raise STTExecutionUnsupportedError(
                "Canary does not support the requested language"
            )
        stt_cfg = get_stt_config() or {}
        from . import Audio_Transcription_Nemo as nemo

        path_value = (
            requested
            if requested_label == "local-model"
            else stt_cfg.get("canary_model_path")
        )
        model_path = nemo._require_local_nemo_path(path_value)
        device = str(stt_cfg.get("nemo_device") or "cpu").strip().lower()
        device = _require_planned_device(
            device,
            provider=self.name.value,
            allowed={"cpu", "cuda"},
        )
        dtype = str(
            stt_cfg.get("nemo_compute_type") or "float32"
        ).strip().lower()
        dtype = _require_planned_precision(
            dtype,
            provider=self.name.value,
            label="dtype",
            allowed=(
                {"float16", "bfloat16", "float32"}
                if device == "cuda"
                else {"float32"}
            ),
        )
        if device == "cuda" and not nemo._torch_cuda_available(
            allow_import=False
        ):
            raise STTExecutionUnsupportedError(
                "Canary planned CUDA backend is unavailable"
            )
        return _build_local_plan(
            provider=self.name.value,
            requested_model=requested_label,
            resolved_model=requested_label,
            backend="nemo",
            model_path=model_path,
            device=device,
            compute_type=None,
            dtype=dtype,
            revision=None,
            task=task,
            language=language,
            word_timestamps=word_timestamps,
            prompt=prompt,
            hotwords=hotwords,
            diarization=diarization,
            fixed_english=False,
            runtime_settings={
                _LOCAL_RUNTIME_DEVICE: device,
                _LOCAL_RUNTIME_DTYPE: dtype,
                _LOCAL_RUNTIME_VARIANT: "standard",
            },
            source_modules=(
                "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib",
                "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo",
                "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter",
            ),
            dependency_distributions=("nemo-toolkit",),
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
        execution_plan: SttBatchExecutionPlan | None = None,
    ) -> dict[str, Any]:
        if execution_plan is not None:
            return self._run_planned_batch(
                audio_path,
                execution_plan=execution_plan,
                model=model,
                language=language,
                task=task,
                word_timestamps=word_timestamps,
                prompt=prompt,
                hotwords=hotwords,
                base_dir=base_dir,
                cancel_check=cancel_check,
            )
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore

        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (  # type: ignore
            transcribe_with_canary,
        )

        path_obj = _canonicalize_wav_for_soundfile_adapter(audio_path, base_dir)

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
        call_kwargs = {
            "task": task,
            "target_language": "en" if task == "translate" else None,
        }
        result = transcribe_with_canary(
            audio_np,
            sample_rate,
            language,
            **call_kwargs,
        )
        text = result
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

    def plan_batch_execution(
        self,
        *,
        model: str | None,
        language: str | None,
        task: str,
        word_timestamps: bool,
        prompt: str | None,
        hotwords: Sequence[str] | None,
        diarization: bool,
        mode: str,
    ) -> SttBatchExecutionPlan:
        _require_benchmark_mode(mode)
        if _primary_language(language) != "en":
            raise STTExecutionUnsupportedError(
                "Qwen2Audio neutral-v1 supports only English language tags"
            )
        requested = model or "qwen2audio"
        requested_label = _safe_requested_model_label(requested)
        from . import Audio_Transcription_Lib as atlib

        stt_cfg = get_stt_config() or {}
        path_value = (
            requested
            if requested_label == "local-model" and Path(requested).is_dir()
            else stt_cfg.get("qwen2audio_model_id")
        )
        try:
            model_path = atlib.validate_qwen2audio_model_identifier(
                str(path_value or "")
            )
        except ValueError:
            raise STTExecutionUnsupportedError(
                "Qwen2Audio local model artifact is invalid"
            ) from None
        revision_value = stt_cfg.get("qwen2audio_revision")
        revision = str(revision_value).strip() if revision_value else None
        device_map = str(
            stt_cfg.get("qwen2audio_device_map") or "auto"
        ).strip().lower()
        device_map = _require_planned_device(
            device_map,
            provider=self.name.value,
            allowed={"cpu", "cuda"},
        )
        dtype = str(
            stt_cfg.get("qwen2audio_dtype") or "float16"
        ).strip().lower()
        dtype = _require_planned_precision(
            dtype,
            provider=self.name.value,
            label="dtype",
            allowed=(
                {"float16", "bfloat16", "float32"}
                if device_map == "cuda"
                else {"float32"}
            ),
        )
        if device_map == "cuda" and not atlib._torch_cuda_available(
            allow_import=False
        ):
            raise STTExecutionUnsupportedError(
                "Qwen2Audio planned CUDA backend is unavailable"
            )
        runtime_settings: dict[str, SttPlanScalar] = {
            _LOCAL_RUNTIME_DEVICE_MAP: device_map,
            _LOCAL_RUNTIME_DTYPE: dtype,
            _LOCAL_RUNTIME_REVISION: revision,
        }
        return _build_local_plan(
            provider=self.name.value,
            requested_model=requested_label,
            resolved_model=requested_label,
            backend="transformers",
            model_path=model_path,
            device=None if device_map == "auto" else device_map,
            compute_type=None,
            dtype=dtype,
            revision=revision,
            task=task,
            language=language,
            word_timestamps=word_timestamps,
            prompt=prompt,
            hotwords=hotwords,
            diarization=diarization,
            fixed_english=True,
            runtime_settings=runtime_settings,
            source_modules=(
                "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib",
                "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter",
            ),
            dependency_distributions=("transformers",),
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
        execution_plan: SttBatchExecutionPlan | None = None,
    ) -> dict[str, Any]:
        if execution_plan is not None:
            return self._run_planned_batch(
                audio_path,
                execution_plan=execution_plan,
                model=model,
                language=language,
                task=task,
                word_timestamps=word_timestamps,
                prompt=prompt,
                hotwords=hotwords,
                base_dir=base_dir,
                cancel_check=cancel_check,
            )
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (  # type: ignore
            speech_to_text,
        )

        model_name = model or "qwen2audio"
        _raise_if_cancelled(cancel_check)
        call_kwargs = {
            "whisper_model": model_name,
            "selected_source_lang": language,
            "vad_filter": False,
            "diarize": False,
            "return_language": True,
            "base_dir": base_dir,
            "cancel_check": cancel_check,
        }
        result = speech_to_text(audio_path, **call_kwargs)
        segments_list, lang = result
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

    def plan_batch_execution(
        self,
        *,
        model: str | None,
        language: str | None,
        task: str,
        word_timestamps: bool,
        prompt: str | None,
        hotwords: Sequence[str] | None,
        diarization: bool,
        mode: str,
    ) -> SttBatchExecutionPlan:
        normalized_mode = str(mode or "").strip().lower()
        if normalized_mode not in {"neutral-v1", "production-v1"}:
            raise STTExecutionUnsupportedError(
                f"Unsupported STT benchmark mode: {mode}"
            )
        if (
            task != "transcribe"
            or word_timestamps
            or prompt is not None
            or hotwords
            or diarization
        ):
            raise STTExecutionUnsupportedError(
                "Qwen3-ASR cannot honor the requested benchmark semantics"
            )
        from . import Audio_Transcription_Qwen3ASR as qwen3

        settings = qwen3._resolve_settings()
        if not settings.get("enabled"):
            raise STTExecutionUnsupportedError("Qwen3-ASR is disabled")
        requested = (
            model
            or str(settings.get("model_path") or "qwen3-asr")
        ).strip()
        model_label = _safe_requested_model_label(requested)
        if (
            str(settings.get("backend") or "").strip().lower()
            == "vllm"
            and model_label == "local-model"
        ):
            raise STTExecutionUnsupportedError(
                "Qwen3-ASR vLLM requires a safe request model identifier"
            )
        backend = str(settings.get("backend") or "").strip().lower()
        runtime: dict[str, SttPlanScalar]
        if backend == "vllm":
            base_url = str(
                settings.get("vllm_base_url") or ""
            ).strip()
            endpoint, egress, endpoint_id = _normalize_audio_endpoint(
                _resolve_audio_transcription_endpoint(base_url)
            )
            route = SttExecutionRoute(
                route_id="vllm-http-1",
                provider=self.name.value,
                model_label=model_label,
                artifact_id=None,
                identity_resolved=False,
                backend="vllm_http",
                source="vllm_http",
                audio_egress=egress,
                endpoint_id=endpoint_id,
                device=None,
                compute_type=None,
                dtype=None,
                decoding_ids=(),
                local_model_available=False,
                would_download=False,
            )
            runtime = {
                "backend": "vllm",
                "endpoint": endpoint,
                "endpoint_id": endpoint_id,
                "request_model": requested,
                "sample_rate": int(
                    settings.get("sample_rate") or 16000
                ),
            }
            dependencies = ("httpx",)
        else:
            if language is not None:
                raise STTExecutionUnsupportedError(
                    "Qwen3-ASR local execution cannot honor a language hint"
                )
            model_path = Path(
                str(model or settings.get("model_path") or "")
            )
            if (
                not model_path.is_dir()
                or bool(settings.get("allow_download"))
            ):
                raise STTExecutionUnsupportedError(
                    "Qwen3-ASR requires an existing no-download local model"
                )
            device = str(
                settings.get("device") or "cpu"
            ).strip().lower()
            if qwen3._resolve_device(device) != device:
                raise STTExecutionUnsupportedError(
                    "Qwen3-ASR planned device is unavailable"
                )
            dtype = _require_planned_precision(
                str(
                    settings.get("dtype") or "float32"
                ).strip().lower(),
                provider=self.name.value,
                label="dtype",
                allowed={"float16", "bfloat16", "float32"},
            )
            revision_value = str(
                settings.get("model_revision") or ""
            ).strip()
            revision = revision_value or None
            artifact_id = (
                revision
                if revision is not None
                and _IMMUTABLE_REVISION_RE.fullmatch(revision)
                else None
            )
            route = SttExecutionRoute(
                route_id="local-1",
                provider=self.name.value,
                model_label=model_label,
                artifact_id=artifact_id,
                identity_resolved=artifact_id is not None,
                backend="transformers",
                source="local",
                audio_egress=SttAudioEgress.NONE,
                endpoint_id=None,
                device=device,
                compute_type=None,
                dtype=dtype,
                decoding_ids=(),
                local_model_available=True,
                would_download=False,
            )
            runtime = {
                "allow_download": False,
                "backend": "transformers",
                "device": device,
                "dtype": dtype,
                "max_new_tokens": int(
                    settings.get("max_new_tokens") or 4096
                ),
                "model_path": str(model_path.resolve()),
                "model_revision": revision,
                "sample_rate": int(
                    settings.get("sample_rate") or 16000
                ),
            }
            dependencies = ("transformers",)
        descriptor = SttExecutionDescriptor(
            requested_provider=self.name.value,
            requested_model_label=model_label,
            resolved_provider=self.name.value,
            resolved_model_label=model_label,
            routes=(route,),
            honors_task=True,
            honors_language=True,
            honors_prompt_absence=True,
            honors_hotword_absence=True,
            honors_diarization=True,
            honors_word_timestamps=True,
            decoding_settings=(),
            source_modules=tuple(
                sorted(
                    (
                        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Qwen3ASR",
                        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract",
                        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter",
                        *(
                            (
                                "tldw_Server_API.app.core.http_client",
                                "tldw_Server_API.app.core.stt_observability_context",
                            )
                            if backend == "vllm"
                            else ()
                        ),
                    )
                )
            ),
            dependency_distributions=dependencies,
        )
        return SttBatchExecutionPlan(
            descriptor=descriptor,
            task=task,
            language=language,
            runtime_settings=tuple(sorted(runtime.items())),
        )

    def _transcribe_planned_batch(
        self,
        audio_path: str,
        *,
        execution_plan: SttBatchExecutionPlan,
        base_dir: Path | None,
        cancel_check: Callable[[], bool] | None,
    ) -> SttTranscriptionOutcome:
        from .Audio_Transcription_Qwen3ASR import (
            transcribe_with_qwen3_asr,
        )

        outcome = transcribe_with_qwen3_asr(
            audio_path,
            language=execution_plan.language,
            word_timestamps=execution_plan.word_timestamps,
            base_dir=base_dir,
            cancel_check=cancel_check,
            execution_plan=execution_plan,
        )
        if not isinstance(outcome, SttTranscriptionOutcome):
            raise STTExecutionPlanError(
                "Planned Qwen3-ASR execution did not report its route"
            )
        return outcome

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
        execution_plan: SttBatchExecutionPlan | None = None,
    ) -> dict[str, Any]:
        if execution_plan is not None:
            return self._run_planned_batch(
                audio_path,
                execution_plan=execution_plan,
                model=model,
                language=language,
                task=task,
                word_timestamps=word_timestamps,
                prompt=prompt,
                hotwords=hotwords,
                base_dir=base_dir,
                cancel_check=cancel_check,
            )
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
        audio_path_for_provider = str(_canonicalize_wav_for_soundfile_adapter(audio_path, base_dir))
        call_kwargs = {
            "model_path": model_path,
            "language": language,
            "word_timestamps": word_timestamps,
            "base_dir": base_dir,
            "cancel_check": cancel_check,
        }
        artifact = transcribe_with_qwen3_asr(audio_path_for_provider, **call_kwargs)
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

    def plan_batch_execution(
        self,
        *,
        model: str | None,
        language: str | None,
        task: str,
        word_timestamps: bool,
        prompt: str | None,
        hotwords: Sequence[str] | None,
        diarization: bool,
        mode: str,
    ) -> SttBatchExecutionPlan:
        normalized_mode = str(mode or "").strip().lower()
        if normalized_mode not in {"neutral-v1", "production-v1"}:
            raise STTExecutionUnsupportedError(
                f"Unsupported STT benchmark mode: {mode}"
            )
        if (
            task != "transcribe"
            or word_timestamps
            or prompt is not None
            or diarization
            or (normalized_mode == "neutral-v1" and hotwords)
        ):
            raise STTExecutionUnsupportedError(
                "VibeVoice cannot honor the requested benchmark semantics"
            )
        from . import Audio_Transcription_VibeVoice as vibe

        settings = vibe._resolve_settings()
        original_model = str(
            settings.get("model_id") or "microsoft/VibeVoice-ASR"
        )
        if model and model.strip():
            settings["model_id"] = model.strip()
            if str(
                settings.get("vllm_model_id") or ""
            ).strip() in {"", original_model}:
                settings["vllm_model_id"] = model.strip()
        requested = model or str(settings["model_id"])
        model_label = _safe_requested_model_label(requested)
        vllm_model_id = str(
            settings.get("vllm_model_id")
            or settings["model_id"]
        ).strip()
        vllm_model_label = _safe_requested_model_label(
            vllm_model_id
        )
        local_model_label = _safe_requested_model_label(
            str(settings["model_id"])
        )
        if (
            settings.get("vllm_enabled")
            and vllm_model_label == "local-model"
        ):
            raise STTExecutionUnsupportedError(
                "VibeVoice vLLM requires a safe request model identifier"
            )
        planned_hotwords = tuple(hotwords or ())
        decoding_settings = (
            (
                ("hotword_count", len(planned_hotwords)),
                ("prompt_present", False),
            )
            if normalized_mode == "production-v1"
            else ()
        )
        decoding_ids = tuple(
            key for key, _value in decoding_settings
        )
        routes: list[SttExecutionRoute] = []
        runtime: dict[str, SttPlanScalar] = {
            "allow_download": False,
            "cache_dir": str(settings.get("cache_dir") or ""),
            "device": str(settings.get("device") or "cpu").lower(),
            "dtype": str(settings.get("dtype") or "float32").lower(),
            "max_new_tokens": int(
                settings.get("max_new_tokens") or 4096
            ),
            "model_id": str(settings["model_id"]),
            "model_revision": str(
                settings.get("model_revision") or ""
            )
            or None,
            "sample_rate": int(
                settings.get("sample_rate") or 16000
            ),
            "strict_semantics": False,
            "local_model_label": local_model_label,
            "vllm_api_key": settings.get("vllm_api_key"),
            "vllm_model_id": vllm_model_id,
            "vllm_timeout_seconds": int(
                settings.get("vllm_timeout_seconds") or 600
            ),
        }
        if settings.get("vllm_enabled"):
            endpoint, egress, endpoint_id = _normalize_audio_endpoint(
                _resolve_audio_transcription_endpoint(
                    str(settings.get("vllm_base_url") or "")
                )
            )
            runtime["endpoint"] = endpoint
            runtime["endpoint_id"] = endpoint_id
            routes.append(
                SttExecutionRoute(
                    route_id="vllm-http-1",
                    provider=self.name.value,
                    model_label=vllm_model_label,
                    artifact_id=None,
                    identity_resolved=False,
                    backend="vllm_http",
                    source="vllm_http",
                    audio_egress=egress,
                    endpoint_id=endpoint_id,
                    device=None,
                    compute_type=None,
                    dtype=None,
                    decoding_ids=decoding_ids,
                    local_model_available=False,
                    would_download=False,
                )
            )
        include_local = (
            not routes or normalized_mode == "production-v1"
        ) and bool(settings.get("enabled"))
        if include_local:
            model_path = Path(str(settings["model_id"]))
            if (
                not model_path.is_dir()
                or bool(settings.get("allow_download"))
            ):
                raise STTExecutionUnsupportedError(
                    "VibeVoice production fallback is not an explicit no-download local model"
                )
            device = str(
                settings.get("device") or "cpu"
            ).strip().lower()
            if vibe._resolve_device(device) != device:
                raise STTExecutionUnsupportedError(
                    "VibeVoice planned device is unavailable"
                )
            dtype = _require_planned_precision(
                str(
                    settings.get("dtype") or "float32"
                ).strip().lower(),
                provider=self.name.value,
                label="dtype",
                allowed={"float16", "bfloat16", "float32"},
            )
            revision = str(
                settings.get("model_revision") or ""
            ).strip() or None
            artifact_id = (
                revision
                if revision is not None
                and _IMMUTABLE_REVISION_RE.fullmatch(revision)
                else None
            )
            runtime.update(
                device=device,
                dtype=dtype,
                model_id=str(model_path.resolve()),
                model_revision=revision,
            )
            local_model_label = _safe_requested_model_label(
                str(model_path.resolve())
            )
            runtime["local_model_label"] = local_model_label
            routes.append(
                SttExecutionRoute(
                    route_id=(
                        "local-2" if len(routes) == 1 else "local-1"
                    ),
                    provider=self.name.value,
                    model_label=local_model_label,
                    artifact_id=artifact_id,
                    identity_resolved=artifact_id is not None,
                    backend="transformers",
                    source="local",
                    audio_egress=SttAudioEgress.NONE,
                    endpoint_id=None,
                    device=device,
                    compute_type=None,
                    dtype=dtype,
                    decoding_ids=decoding_ids,
                    local_model_available=True,
                    would_download=False,
                )
            )
        if not routes:
            raise STTExecutionUnsupportedError(
                "VibeVoice has no enforceable configured execution route"
            )
        descriptor = SttExecutionDescriptor(
            requested_provider=self.name.value,
            requested_model_label=model_label,
            resolved_provider=self.name.value,
            resolved_model_label=routes[0].model_label,
            routes=tuple(routes),
            honors_task=True,
            honors_language=True,
            honors_prompt_absence=True,
            honors_hotword_absence=True,
            honors_diarization=True,
            honors_word_timestamps=True,
            decoding_settings=decoding_settings,
            source_modules=tuple(
                sorted(
                    (
                        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_VibeVoice",
                        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract",
                        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter",
                        *(
                            (
                                "tldw_Server_API.app.core.Security.egress",
                                "tldw_Server_API.app.core.http_client",
                                "tldw_Server_API.app.core.stt_observability_context",
                            )
                            if any(
                                route.backend == "vllm_http"
                                for route in routes
                            )
                            else ()
                        ),
                    )
                )
            ),
            dependency_distributions=(
                ("httpx", "transformers")
                if len(routes) == 2
                else (
                    ("httpx",)
                    if routes[0].backend == "vllm_http"
                    else ("transformers",)
                )
            ),
        )
        return SttBatchExecutionPlan(
            descriptor=descriptor,
            task=task,
            language=language,
            hotwords=planned_hotwords,
            runtime_settings=tuple(sorted(runtime.items())),
        )

    def _transcribe_planned_batch(
        self,
        audio_path: str,
        *,
        execution_plan: SttBatchExecutionPlan,
        base_dir: Path | None,
        cancel_check: Callable[[], bool] | None,
    ) -> SttTranscriptionOutcome:
        from .Audio_Transcription_VibeVoice import (
            transcribe_with_vibevoice,
        )

        outcome = transcribe_with_vibevoice(
            audio_path,
            language=execution_plan.language,
            hotwords=execution_plan.hotwords,
            base_dir=base_dir,
            cancel_check=cancel_check,
            execution_plan=execution_plan,
        )
        if not isinstance(outcome, SttTranscriptionOutcome):
            raise STTExecutionPlanError(
                "Planned VibeVoice execution did not report its route"
            )
        return outcome

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
        execution_plan: SttBatchExecutionPlan | None = None,
    ) -> dict[str, Any]:
        if execution_plan is not None:
            return self._run_planned_batch(
                audio_path,
                execution_plan=execution_plan,
                model=model,
                language=language,
                task=task,
                word_timestamps=word_timestamps,
                prompt=prompt,
                hotwords=hotwords,
                base_dir=base_dir,
                cancel_check=cancel_check,
            )
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
        audio_path_for_provider = str(_canonicalize_wav_for_soundfile_adapter(audio_path, base_dir))
        call_kwargs = {
            "model_id": model_name,
            "language": language,
            "hotwords": list(hotwords) if hotwords else None,
            "base_dir": base_dir,
            "cancel_check": cancel_check,
        }
        artifact = transcribe_with_vibevoice(audio_path_for_provider, **call_kwargs)
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

    def plan_batch_execution(
        self,
        *,
        model: str | None,
        language: str | None,
        task: str,
        word_timestamps: bool,
        prompt: str | None,
        hotwords: Sequence[str] | None,
        diarization: bool,
        mode: str,
    ) -> SttBatchExecutionPlan:
        normalized_mode = str(mode or "").strip().lower()
        if normalized_mode not in {"neutral-v1", "production-v1"}:
            raise STTExecutionUnsupportedError(
                f"Unsupported STT benchmark mode: {mode}"
            )
        if (
            task != "transcribe"
            or word_timestamps
            or hotwords
            or diarization
            or (normalized_mode == "neutral-v1" and prompt is not None)
        ):
            raise STTExecutionUnsupportedError(
                "External STT cannot honor the requested benchmark semantics"
            )
        from . import Audio_Transcription_External_Provider as external

        requested = model or "external:default"
        model_label = _safe_requested_model_label(requested)
        provider_name = (
            requested.split(":", 1)[1] or "default"
            if requested.startswith("external:")
            else "default"
        )
        config = external.load_external_provider_config(provider_name)
        if config is None:
            raise STTExecutionUnsupportedError(
                "External STT provider is not configured"
            )
        actual_model_id = str(config.model).strip()
        actual_model_label = _safe_requested_model_label(
            actual_model_id
        )
        if actual_model_label == "local-model":
            raise STTExecutionUnsupportedError(
                "External STT requires a safe request model identifier"
            )
        if normalized_mode == "neutral-v1" and config.prompt is not None:
            raise STTExecutionUnsupportedError(
                "External STT configured prompt is incompatible with neutral-v1"
            )
        if (
            normalized_mode == "production-v1"
            and config.prompt != prompt
        ):
            raise STTExecutionUnsupportedError(
                "External STT production prompt does not match the request"
            )
        if (
            language is not None
            and config.language is not None
            and config.language != language
        ):
            raise STTExecutionUnsupportedError(
                "External STT configured language does not match the request"
            )
        planned_language = language or config.language
        planned_prompt = (
            config.prompt
            if normalized_mode == "production-v1"
            else None
        )
        endpoint, egress, endpoint_id = _normalize_audio_endpoint(
            _resolve_audio_transcription_endpoint(config.base_url)
        )
        from tldw_Server_API.app.core.http_client import (
            resolve_afetch_transport,
        )

        try:
            transport = resolve_afetch_transport()
        except (RuntimeError, ValueError):
            raise STTExecutionUnsupportedError(
                "External STT has no available async HTTP transport"
            ) from None
        decoding_settings = (
            (
                ("hotword_count", 0),
                ("prompt_present", planned_prompt is not None),
            )
            if normalized_mode == "production-v1"
            else ()
        )
        decoding_ids = tuple(
            key for key, _value in decoding_settings
        )
        route = SttExecutionRoute(
            route_id="external-http-1",
            provider=self.name.value,
            model_label=actual_model_label,
            artifact_id=None,
            identity_resolved=False,
            backend="openai_compatible",
            source="external_http",
            audio_egress=egress,
            endpoint_id=endpoint_id,
            device=None,
            compute_type=None,
            dtype=None,
            decoding_ids=decoding_ids,
            local_model_available=False,
            would_download=False,
            transport=transport,
        )
        header_items = tuple(
            sorted((config.custom_headers or {}).items())
        )
        runtime: dict[str, SttPlanScalar] = {
            _EXTERNAL_RUNTIME_API_KEY: config.api_key,
            _EXTERNAL_RUNTIME_BASE_URL: endpoint,
            _EXTERNAL_RUNTIME_HEADER_NAMES: tuple(
                name for name, _value in header_items
            ),
            _EXTERNAL_RUNTIME_HEADER_VALUES: tuple(
                value for _name, value in header_items
            ),
            _EXTERNAL_RUNTIME_LANGUAGE: planned_language,
            _EXTERNAL_RUNTIME_MAX_RETRIES: int(config.max_retries),
            _EXTERNAL_RUNTIME_MODEL: actual_model_id,
            _EXTERNAL_RUNTIME_PROMPT: planned_prompt,
            _EXTERNAL_RUNTIME_PROVIDER: provider_name,
            _EXTERNAL_RUNTIME_RESPONSE_FORMAT: (
                config.response_format
            ),
            _EXTERNAL_RUNTIME_TEMPERATURE: float(
                config.temperature
            ),
            _EXTERNAL_RUNTIME_TIMEOUT: float(config.timeout),
            _EXTERNAL_RUNTIME_TRANSPORT: transport,
            _EXTERNAL_RUNTIME_VERIFY_SSL: bool(config.verify_ssl),
        }
        descriptor = SttExecutionDescriptor(
            requested_provider=self.name.value,
            requested_model_label=model_label,
            resolved_provider=self.name.value,
            resolved_model_label=actual_model_label,
            routes=(route,),
            honors_task=True,
            honors_language=True,
            honors_prompt_absence=planned_prompt is None,
            honors_hotword_absence=True,
            honors_diarization=True,
            honors_word_timestamps=True,
            decoding_settings=decoding_settings,
            source_modules=tuple(
                sorted(
                    (
                        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider",
                        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract",
                        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter",
                        "tldw_Server_API.app.core.Security.egress",
                        "tldw_Server_API.app.core.http_client",
                        "tldw_Server_API.app.core.stt_observability_context",
                    )
                )
            ),
            dependency_distributions=(transport,),
        )
        return SttBatchExecutionPlan(
            descriptor=descriptor,
            task=task,
            language=planned_language,
            prompt=planned_prompt,
            runtime_settings=tuple(sorted(runtime.items())),
        )

    def _transcribe_planned_batch(
        self,
        audio_path: str,
        *,
        execution_plan: SttBatchExecutionPlan,
        base_dir: Path | None,
        cancel_check: Callable[[], bool] | None,
    ) -> SttTranscriptionOutcome:
        from .Audio_Transcription_External_Provider import (
            ExternalProviderConfig,
            transcribe_with_external_provider,
        )

        runtime = execution_plan.runtime_values()
        header_names = runtime[_EXTERNAL_RUNTIME_HEADER_NAMES]
        header_values = runtime[_EXTERNAL_RUNTIME_HEADER_VALUES]
        if (
            not isinstance(header_names, tuple)
            or not isinstance(header_values, tuple)
            or len(header_names) != len(header_values)
        ):
            raise STTExecutionPlanError(
                "External STT header snapshot is invalid"
            )
        config = ExternalProviderConfig(
            base_url=str(runtime[_EXTERNAL_RUNTIME_BASE_URL]),
            api_key=(
                str(runtime[_EXTERNAL_RUNTIME_API_KEY])
                if runtime[_EXTERNAL_RUNTIME_API_KEY] is not None
                else None
            ),
            model=str(runtime[_EXTERNAL_RUNTIME_MODEL]),
            timeout=float(runtime[_EXTERNAL_RUNTIME_TIMEOUT]),
            max_retries=int(
                runtime[_EXTERNAL_RUNTIME_MAX_RETRIES]
            ),
            verify_ssl=bool(
                runtime[_EXTERNAL_RUNTIME_VERIFY_SSL]
            ),
            custom_headers=dict(zip(header_names, header_values)),
            response_format=str(
                runtime[_EXTERNAL_RUNTIME_RESPONSE_FORMAT]
            ),
            temperature=float(
                runtime[_EXTERNAL_RUNTIME_TEMPERATURE]
            ),
            language=(
                str(runtime[_EXTERNAL_RUNTIME_LANGUAGE])
                if runtime[_EXTERNAL_RUNTIME_LANGUAGE] is not None
                else None
            ),
            prompt=(
                str(runtime[_EXTERNAL_RUNTIME_PROMPT])
                if runtime[_EXTERNAL_RUNTIME_PROMPT] is not None
                else None
            ),
        )
        outcome = transcribe_with_external_provider(
            audio_path,
            provider_name=str(
                runtime[_EXTERNAL_RUNTIME_PROVIDER]
            ),
            config=config,
            base_dir=base_dir,
            execution_plan=execution_plan,
            transport=str(runtime[_EXTERNAL_RUNTIME_TRANSPORT]),
        )
        if not isinstance(outcome, SttTranscriptionOutcome):
            raise STTExecutionPlanError(
                "Planned external STT execution did not report its route"
            )
        return outcome

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
        execution_plan: SttBatchExecutionPlan | None = None,
    ) -> dict[str, Any]:
        if execution_plan is not None:
            return self._run_planned_batch(
                audio_path,
                execution_plan=execution_plan,
                model=model,
                language=language,
                task=task,
                word_timestamps=word_timestamps,
                prompt=prompt,
                hotwords=hotwords,
                base_dir=base_dir,
                cancel_check=cancel_check,
            )
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_External_Provider import (  # type: ignore
            transcribe_with_external_provider,
        )

        provider_name = "default"
        model_id = model or "whisper-1"
        if model_id.startswith("external:"):
            provider_name = model_id.split(":", 1)[1] or "default"

        # Pass base_dir so external providers validate local paths consistently.
        _raise_if_cancelled(cancel_check)
        call_kwargs = {
            "provider_name": provider_name,
            "base_dir": base_dir,
        }
        result = transcribe_with_external_provider(audio_path, **call_kwargs)
        text = result
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

    def get_adapter_strict(self, provider_name: str) -> SttProviderAdapter:
        """Return only a directly registered adapter, failing closed if absent."""
        key = self.normalize_provider_name(provider_name)
        adapter = self._base.get_adapter(key)
        if adapter is None:
            raise STTExecutionPlanError(
                f"No STT adapter is registered for provider {provider_name!r}"
            )
        return adapter

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
    if configured_batch_model.lower() == "auto":
        return resolve_default_transcription_model_setting(configured_batch_model)
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
