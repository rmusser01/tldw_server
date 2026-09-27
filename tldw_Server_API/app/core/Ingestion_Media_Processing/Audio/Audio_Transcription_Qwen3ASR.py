"""
Qwen3-ASR transcription helpers.

This module provides transcription via Qwen3-ASR models:
- Qwen3-ASR-1.7B (production quality, default)
- Qwen3-ASR-0.6B (resource-constrained / high-throughput)
- Optional Qwen3-ForcedAligner-0.6B for word-level timestamps

Key features:
- 30 languages + 22 Chinese dialects (auto-detected)
- State-of-the-art accuracy (1.63 WER on LibriSpeech clean)
- Optional word-level timestamps via forced alignment
- Lazy model loading with thread-safe caching

Models must be manually downloaded before use:
    hf download Qwen/Qwen3-ASR-1.7B --local-dir ./models/qwen3_asr/1.7B
    hf download Qwen/Qwen3-ASR-0.6B --local-dir ./models/qwen3_asr/0.6B
    hf download Qwen/Qwen3-ForcedAligner-0.6B --local-dir ./models/qwen3_asr/aligner
"""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf  # type: ignore
from loguru import logger

from tldw_Server_API.app.core.config import get_stt_config
from tldw_Server_API.app.core.exceptions import (
    BadRequestError,
    CancelCheckError,
    STTExecutionPlanError,
    TranscriptionCancelled,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
    SttAudioEgress,
    SttBatchExecutionPlan,
    SttTranscriptionOutcome,
    _normalize_audio_endpoint,
    actual_execution_from_route,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.path_utils import resolve_safe_local_path
from tldw_Server_API.app.core.testing import is_truthy

# Global cache for loaded models
_MODEL_CACHE: dict[str, tuple[Any, Any, str]] = {}
_MODEL_LOCK = threading.Lock()

# Global cache for forced aligner
_ALIGNER_CACHE: dict[str, tuple[Any, Any]] = {}
_ALIGNER_LOCK = threading.Lock()
_QWEN3_ASR_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _as_bool(value: Any, default: bool = False) -> bool:
    """Convert config value to boolean."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    s = str(value).strip().lower()
    if is_truthy(s):
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _as_int(value: Any, default: int) -> int:
    """Convert config value to integer."""
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, default: str) -> str:
    """Convert config value to string."""
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default


def _as_float(value: Any, default: float) -> float:
    """Convert config value to float."""
    if value is None:
        return default
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _check_cancel(cancel_check: Callable[[], bool] | None, *, label: str) -> None:
    """Check if cancellation was requested."""
    if cancel_check is None:
        return
    try:
        should_cancel = bool(cancel_check())
    except Exception as exc:
        raise CancelCheckError(f"cancel_check failed during {label}: {exc}") from exc
    if should_cancel:
        raise TranscriptionCancelled(f"Cancelled during {label}")


def _resolve_settings() -> dict[str, Any]:
    """Resolve Qwen3-ASR settings from config."""
    try:
        stt_cfg = get_stt_config() or {}
    except _QWEN3_ASR_NONCRITICAL_EXCEPTIONS:
        stt_cfg = {}

    # Default model path points to 1.7B (production quality)
    default_model_path = "./models/qwen3_asr/1.7B"

    settings = {
        "enabled": _as_bool(stt_cfg.get("qwen3_asr_enabled"), False),
        "model_path": _as_str(stt_cfg.get("qwen3_asr_model_path"), default_model_path),
        "model_revision": _as_str(stt_cfg.get("qwen3_asr_model_revision"), ""),
        "device": _as_str(stt_cfg.get("qwen3_asr_device"), "cuda"),
        "dtype": _as_str(stt_cfg.get("qwen3_asr_dtype"), "bfloat16"),
        "max_batch_size": _as_int(stt_cfg.get("qwen3_asr_max_batch_size"), 32),
        "max_new_tokens": _as_int(stt_cfg.get("qwen3_asr_max_new_tokens"), 4096),
        "allow_download": _as_bool(stt_cfg.get("qwen3_asr_allow_download"), False),
        "sample_rate": _as_int(stt_cfg.get("qwen3_asr_sample_rate"), 16000),
        # Forced aligner settings
        "aligner_enabled": _as_bool(stt_cfg.get("qwen3_asr_aligner_enabled"), False),
        "aligner_path": _as_str(stt_cfg.get("qwen3_asr_aligner_path"), "./models/qwen3_asr/aligner"),
        "aligner_revision": _as_str(stt_cfg.get("qwen3_asr_aligner_revision"), ""),
        # Backend selection
        "backend": _as_str(stt_cfg.get("qwen3_asr_backend"), "transformers"),
        "vllm_gpu_memory_utilization": _as_float(stt_cfg.get("qwen3_asr_vllm_gpu_memory_utilization"), 0.7),
        "vllm_base_url": _as_str(stt_cfg.get("qwen3_asr_vllm_base_url"), ""),
    }
    return settings


def _resolve_audio_path(audio_path: str, base_dir: Path | None) -> Path:
    """Resolve and validate audio path."""
    path_obj = Path(audio_path)
    base = Path(base_dir) if base_dir is not None else path_obj.parent
    safe_path = resolve_safe_local_path(path_obj, base)
    if safe_path is None:
        raise BadRequestError(f"Audio path rejected outside base_dir: {audio_path}")
    return safe_path


def _load_audio(
    audio_path: Path,
    *,
    target_sample_rate: int,
) -> tuple[np.ndarray, int, float]:
    """Load audio file and resample if needed."""
    try:
        audio_np, sample_rate = sf.read(str(audio_path))
    except Exception as exc:
        raise BadRequestError(f"Failed to read audio file {audio_path}: {exc}") from exc

    if not isinstance(audio_np, np.ndarray):
        audio_np = np.array(audio_np, dtype="float32")

    # Fold stereo/multi-channel to mono
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1)

    audio_np = np.asarray(audio_np, dtype="float32")
    duration_seconds = float(len(audio_np)) / float(sample_rate or 1)

    if sample_rate != target_sample_rate and target_sample_rate > 0:
        resampled, sr_out = _maybe_resample(audio_np, sample_rate, target_sample_rate)
        audio_np, sample_rate = resampled, sr_out
        duration_seconds = float(len(audio_np)) / float(sample_rate or 1)

    return audio_np, int(sample_rate), duration_seconds


def _maybe_resample(audio_np: np.ndarray, sample_rate: int, target_sample_rate: int) -> tuple[np.ndarray, int]:
    """Resample audio to target sample rate."""
    if sample_rate == target_sample_rate:
        return audio_np, sample_rate
    try:
        import torch
        import torchaudio.functional as F  # type: ignore  # noqa: N812

        wav = torch.from_numpy(np.asarray(audio_np, dtype="float32")).unsqueeze(0)
        resampled = F.resample(wav, sample_rate, target_sample_rate)
        return resampled.squeeze(0).cpu().numpy().astype("float32"), target_sample_rate
    except _QWEN3_ASR_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(
            'Qwen3-ASR: resampling from {} Hz to {} Hz failed; proceeding at original rate. Error: {}',
            sample_rate,
            target_sample_rate,
            exc,
        )
        return audio_np, sample_rate


def _get_torch_dtype(dtype_name: str) -> Any:
    """Convert dtype name to torch dtype."""
    import torch

    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(str(dtype_name).strip().lower(), torch.bfloat16)


def _resolve_device(requested_device: str) -> str:
    """Resolve device, falling back to CPU if CUDA unavailable."""
    try:
        import torch
    except ImportError:
        return "cpu"
    dev = (requested_device or "cpu").strip().lower()
    if dev.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("Qwen3-ASR: CUDA requested but not available; falling back to CPU")
        return "cpu"
    return dev or "cpu"


def _validate_model_path(model_path: str, allow_download: bool) -> Path:
    """Validate that model path exists or downloading is allowed."""
    path = Path(model_path)
    if path.exists():
        return path
    if not allow_download:
        raise BadRequestError(
            f"Qwen3-ASR model path does not exist: {model_path}. "
            "Download the model first with: "
            f"hf download Qwen/Qwen3-ASR-1.7B --local-dir {model_path}"
        )
    return path


def _load_qwen3_asr_model(settings: dict[str, Any]) -> tuple[Any, Any, str]:
    """Load Qwen3-ASR model with caching."""
    model_path = str(settings["model_path"])
    device = _resolve_device(str(settings["device"]))
    dtype_name = str(settings["dtype"])
    allow_download = bool(settings["allow_download"])
    model_revision = str(settings.get("model_revision") or "").strip() or None
    planned_device = settings.get("planned_device")
    if planned_device is not None and device != str(planned_device):
        raise STTExecutionPlanError(
            "Qwen3-ASR loaded device does not match the execution plan"
        )

    cache_key = (
        f"qwen3_asr|{model_path}|{model_revision}|{device}|"
        f"{dtype_name}|{allow_download}"
    )
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        # Validate model path
        validated_path = _validate_model_path(model_path, allow_download)

        try:
            from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
        except Exception as exc:
            raise BadRequestError(
                "transformers is required for Qwen3-ASR. Install with: pip install transformers"
            ) from exc


        torch_dtype = _get_torch_dtype(dtype_name)
        local_only = not allow_download

        logger.info("Qwen3-ASR: loading local model on device '{}'", device)

        processor = AutoProcessor.from_pretrained(
            str(validated_path),
            revision=model_revision,
            trust_remote_code=True,
            local_files_only=local_only,
        )  # nosec B615

        device_map = "auto" if device != "cpu" else None
        model = AutoModelForCausalLM.from_pretrained(
            str(validated_path),
            revision=model_revision,
            trust_remote_code=True,
            local_files_only=local_only,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )  # nosec B615

        if device_map is None:
            model = model.to(device)
        model.eval()

        _MODEL_CACHE[cache_key] = (processor, model, device)
        logger.info("Qwen3-ASR: successfully loaded model on device '{}'", device)
        return processor, model, device


def _load_forced_aligner(settings: dict[str, Any]) -> tuple[Any, Any]:
    """Load Qwen3-ForcedAligner model with caching."""
    aligner_path = str(settings["aligner_path"])
    allow_download = bool(settings["allow_download"])
    aligner_revision = str(settings.get("aligner_revision") or "").strip() or None

    cache_key = f"qwen3_aligner|{aligner_path}"
    with _ALIGNER_LOCK:
        cached = _ALIGNER_CACHE.get(cache_key)
        if cached is not None:
            return cached

        # Validate aligner path
        path = Path(aligner_path)
        if not path.exists() and not allow_download:
            raise BadRequestError(
                f"Qwen3-ForcedAligner path does not exist: {aligner_path}. "
                "Download the aligner first with: "
                f"hf download Qwen/Qwen3-ForcedAligner-0.6B --local-dir {aligner_path}"
            )

        try:
            from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
        except Exception as exc:
            raise BadRequestError(
                "transformers is required for Qwen3-ForcedAligner"
            ) from exc

        import torch

        logger.info("Qwen3-ASR: loading forced aligner from '{}'", aligner_path)

        processor = AutoProcessor.from_pretrained(
            str(path),
            revision=aligner_revision,
            trust_remote_code=True,
            local_files_only=not allow_download,
        )  # nosec B615

        model = AutoModelForCausalLM.from_pretrained(
            str(path),
            revision=aligner_revision,
            trust_remote_code=True,
            local_files_only=not allow_download,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )  # nosec B615
        model.eval()

        _ALIGNER_CACHE[cache_key] = (processor, model)
        logger.info("Qwen3-ASR: successfully loaded forced aligner")
        return processor, model


def _run_forced_alignment(
    text: str,
    audio_np: np.ndarray,
    sample_rate: int,
    settings: dict[str, Any],
    language: str | None,
) -> list[dict[str, Any]]:
    """Run forced alignment to get word-level timestamps."""
    try:
        aligner_processor, aligner_model = _load_forced_aligner(settings)
    except (BadRequestError, *_QWEN3_ASR_NONCRITICAL_EXCEPTIONS) as exc:
        logger.warning("Qwen3-ASR: forced aligner not available: {}", exc)
        return []

    import torch

    # Build aligner inputs
    inputs = aligner_processor(
        audio=audio_np,
        sampling_rate=sample_rate,
        text=text,
        return_tensors="pt",
    )

    # Move inputs to device
    device = aligner_model.device if hasattr(aligner_model, "device") else next(aligner_model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        try:
            outputs = aligner_model.align(**inputs)
        except AttributeError:
            # Fall back to generate if align method not available
            outputs = aligner_model.generate(**inputs, max_new_tokens=4096)
            # Parse output for timestamps - format depends on model version
            if hasattr(aligner_processor, "decode_alignment"):
                return aligner_processor.decode_alignment(outputs)
            return []

    # Parse alignment outputs into word timestamps
    words: list[dict[str, Any]] = []
    if isinstance(outputs, dict) and "words" in outputs:
        for w in outputs["words"]:
            words.append({
                "word": str(w.get("word", "")),
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
            })
    elif isinstance(outputs, list):
        for w in outputs:
            if isinstance(w, dict):
                words.append({
                    "word": str(w.get("word", w.get("text", ""))),
                    "start": float(w.get("start", w.get("start_time", 0.0))),
                    "end": float(w.get("end", w.get("end_time", 0.0))),
                })

    return words


def _transcribe_local(
    *,
    audio_np: np.ndarray,
    sample_rate: int,
    duration_seconds: float,
    settings: dict[str, Any],
    language: str | None,
    word_timestamps: bool,
    cancel_check: Callable[[], bool] | None,
) -> dict[str, Any]:
    """Perform local transcription using Qwen3-ASR model."""
    _check_cancel(cancel_check, label="model loading")
    processor, model, device = _load_qwen3_asr_model(settings)
    _check_cancel(cancel_check, label="preprocessing")

    import torch

    # Build processor inputs
    inputs = processor(
        audio=audio_np,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    # Move inputs to device
    if device != "cpu":
        dev = torch.device(device)
        inputs = {k: v.to(dev) if hasattr(v, "to") else v for k, v in inputs.items()}

    gen_kwargs = {
        "do_sample": False,
        "max_new_tokens": int(settings.get("max_new_tokens") or 4096),
    }

    _check_cancel(cancel_check, label="transcription")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    # Decode output
    if hasattr(processor, "batch_decode"):
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
        text = str(decoded[0]).strip() if decoded else ""
    elif hasattr(processor, "decode"):
        text = str(processor.decode(generated_ids[0], skip_special_tokens=True)).strip()
    else:
        text = str(generated_ids).strip()

    # Extract language if detected
    detected_language = None
    if hasattr(model, "detected_language"):
        detected_language = model.detected_language
    elif isinstance(generated_ids, torch.Tensor) and hasattr(processor, "decode_with_metadata"):
        try:
            metadata = processor.decode_with_metadata(generated_ids)
            detected_language = metadata.get("language")
        except _QWEN3_ASR_NONCRITICAL_EXCEPTIONS:
            pass

    # Get word timestamps if requested
    words: list[dict[str, Any]] = []
    if word_timestamps and settings.get("aligner_enabled"):
        _check_cancel(cancel_check, label="forced alignment")
        words = _run_forced_alignment(text, audio_np, sample_rate, settings, language)

    return _normalize_artifact(
        text=text,
        duration_seconds=duration_seconds,
        language_hint=language,
        detected_language=detected_language,
        model_path=str(settings["model_path"]),
        words=words,
    )


def _normalize_artifact(
    *,
    text: str,
    duration_seconds: float,
    language_hint: str | None,
    detected_language: str | None,
    model_path: str,
    words: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Normalize transcription output to standard artifact format."""
    # Build segments from text
    segments = [
        {
            "start_seconds": 0.0,
            "end_seconds": float(duration_seconds) if duration_seconds > 0 else 0.0,
            "Text": text,
        }
    ]

    artifact: dict[str, Any] = {
        "text": text,
        "language": detected_language or language_hint,
        "segments": segments,
        "diarization": {"enabled": False, "speakers": None},
        "usage": {
            "duration_ms": int(duration_seconds * 1000.0) if duration_seconds > 0 else None,
            "tokens": None,
        },
        "metadata": {
            "provider": "qwen3-asr",
            "model": model_path,
            "source": "local",
        },
    }

    if words:
        artifact["words"] = words

    return artifact


def _transcribe_vllm_http(
    audio_path: Path,
    settings: dict[str, Any],
    language: str | None,
    cancel_check: Callable[[], bool] | None,
) -> dict[str, Any]:
    """
    Transcribe audio via external vLLM server using OpenAI-compatible API.

    Args:
        audio_path: Resolved path to audio file
        settings: Qwen3-ASR settings dict with vllm_base_url
        language: Language hint (optional)
        cancel_check: Cancellation callback

    Returns:
        Normalized transcription artifact
    """
    try:
        import httpx
    except ImportError as exc:
        raise BadRequestError(
            "httpx is required for vLLM HTTP transcription. Install with: pip install httpx"
        ) from exc

    base_url = str(settings.get("vllm_base_url") or "").rstrip("/")
    endpoint = str(settings.get("endpoint") or "").strip()
    if not endpoint and not base_url:
        raise BadRequestError(
            "vLLM base URL not configured. Set [STT-Settings].qwen3_asr_vllm_base_url in config."
        )

    url = endpoint or f"{base_url}/v1/audio/transcriptions"
    planned = bool(endpoint)

    _check_cancel(cancel_check, label="vllm http request")

    # Get audio duration for metadata
    try:
        audio_np, sample_rate, duration_seconds = _load_audio(
            audio_path,
            target_sample_rate=int(settings.get("sample_rate") or 16000),
        )
    except (BadRequestError, *_QWEN3_ASR_NONCRITICAL_EXCEPTIONS) as exc:
        logger.warning(f"Could not read audio for duration metadata: {exc}")
        duration_seconds = 0.0

    # Build multipart form data and send request
    planned_failure: str | None = None
    try:
        with open(audio_path, "rb") as f:
            # Determine content type based on file extension
            ext = audio_path.suffix.lower()
            content_types = {
                ".wav": "audio/wav",
                ".mp3": "audio/mpeg",
                ".flac": "audio/flac",
                ".ogg": "audio/ogg",
                ".m4a": "audio/mp4",
            }
            content_type = content_types.get(ext, "audio/wav")

            files = {"file": (audio_path.name, f, content_type)}
            data: dict[str, Any] = {
                "model": str(
                    settings.get("request_model")
                    or "qwen3-asr"
                )
            }
            if language:
                data["language"] = language

            # Use sync httpx client with generous timeout for large files
            if planned:
                from tldw_Server_API.app.core.http_client import (
                    RetryPolicy,
                    create_client,
                    fetch,
                    opaque_stt_http_observability,
                )
                from tldw_Server_API.app.core.Security.egress import (
                    ConfiguredEndpointScope,
                )

                endpoint_id = str(
                    settings.get("endpoint_id") or ""
                )
                with opaque_stt_http_observability(endpoint_id):
                    with create_client(
                        timeout=300.0,
                        trust_env=False,
                    ) as client:
                        response = fetch(
                            method="POST",
                            url=url,
                            client=client,
                            timeout=300.0,
                            allow_redirects=False,
                            files=files,
                            data=data,
                            retry=RetryPolicy(attempts=1),
                            configured_endpoint=ConfiguredEndpointScope.from_url(
                                url
                            ),
                        )
                    response.raise_for_status()
                    result = response.json()
            else:
                with httpx.Client(
                    timeout=300.0,
                    follow_redirects=False,
                ) as client:
                    response = client.post(
                        url,
                        files=files,
                        data=data,
                    )
                    response.raise_for_status()
                    result = response.json()

    except httpx.HTTPStatusError as exc:
        if planned:
            planned_failure = (
                "Planned Qwen3-ASR server returned an HTTP error"
            )
        else:
            raise BadRequestError(
                f"vLLM server returned error: {exc.response.status_code} - {exc.response.text}"
            ) from exc
    except httpx.RequestError as exc:
        if planned:
            planned_failure = (
                "Planned Qwen3-ASR server request failed"
            )
        else:
            raise BadRequestError(
                f"Failed to connect to vLLM server at {base_url}: {exc}"
            ) from exc
    except Exception as exc:
        if planned:
            planned_failure = (
                "Planned Qwen3-ASR HTTP transcription failed"
            )
        else:
            raise BadRequestError(
                f"vLLM HTTP transcription failed: {exc}"
            ) from exc
    if planned_failure is not None:
        raise BadRequestError(planned_failure) from None

    # Extract result fields with sensible defaults
    text = str(result.get("text", "")).strip()
    detected_language = result.get("language")
    result_duration = result.get("duration")
    if result_duration is not None:
        with contextlib.suppress(ValueError, TypeError):
            duration_seconds = float(result_duration)

    return _normalize_artifact(
        text=text,
        duration_seconds=duration_seconds,
        language_hint=language,
        detected_language=detected_language,
        model_path=f"vllm:{base_url}",
    )


def is_qwen3_asr_available() -> bool:
    """Check if Qwen3-ASR is available for use."""
    settings = _resolve_settings()
    if not settings["enabled"]:
        return False

    # Check if model path exists
    model_path = Path(settings["model_path"])
    if not model_path.exists():
        logger.debug("Qwen3-ASR: model path does not exist: {}", model_path)
        return False

    # Check for required dependencies
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False

    return True


def is_qwen3_asr_aligner_available() -> bool:
    """Check if Qwen3-ForcedAligner is available for word timestamps."""
    settings = _resolve_settings()
    if not settings.get("aligner_enabled"):
        return False

    aligner_path = Path(settings["aligner_path"])
    return aligner_path.exists()


def get_qwen3_asr_capabilities() -> dict[str, Any]:
    """Return capability information for Qwen3-ASR."""
    settings = _resolve_settings()
    available = is_qwen3_asr_available()
    aligner_available = is_qwen3_asr_aligner_available()

    # Streaming is available when vLLM backend is configured with a base URL
    backend = str(settings.get("backend", "")).lower()
    vllm_url = str(settings.get("vllm_base_url", "")).strip()
    streaming_available = bool(backend == "vllm" and vllm_url)

    return {
        "available": available,
        "enabled": settings["enabled"],
        "model_path": settings["model_path"],
        "device": settings["device"],
        "word_timestamps": aligner_available,
        "streaming": streaming_available,
        "backend": settings["backend"],
        "vllm_base_url": vllm_url if streaming_available else None,
    }


def transcribe_with_qwen3_asr(
    audio_path: str,
    *,
    model_path: str | None = None,
    language: str | None = None,
    word_timestamps: bool = False,
    base_dir: Path | None = None,
    cancel_check: Callable[[], bool] | None = None,
    execution_plan: SttBatchExecutionPlan | None = None,
) -> dict[str, Any] | SttTranscriptionOutcome:
    """
    Transcribe audio using Qwen3-ASR.

    Args:
        audio_path: Path to audio file
        model_path: Override model path (optional)
        language: Language hint (optional, auto-detected by default)
        word_timestamps: Enable word-level timestamps via forced alignment
        base_dir: Base directory for path validation
        cancel_check: Cancellation callback

    Returns:
        Normalized transcription artifact with text, segments, and metadata
    """
    if execution_plan is not None:
        route = execution_plan.descriptor.primary_route
        runtime = execution_plan.runtime_values()
        if (
            len(execution_plan.descriptor.routes) != 1
            or route.provider != "qwen3-asr"
            or route.model_label
            != execution_plan.descriptor.resolved_model_label
        ):
            raise STTExecutionPlanError(
                "Invalid Qwen3-ASR execution route"
            )
        settings = dict(runtime)
        if route.backend == "vllm_http":
            endpoint, egress, endpoint_id = _normalize_audio_endpoint(
                str(runtime.get("endpoint") or "")
            )
            if (
                route.source != "vllm_http"
                or route.audio_egress is not egress
                or route.endpoint_id != endpoint_id
                or runtime.get("request_model")
                != route.model_label
            ):
                raise STTExecutionPlanError(
                    "Qwen3-ASR endpoint does not match its plan"
                )
            settings["endpoint"] = endpoint
            settings["endpoint_id"] = endpoint_id
            resolved_path = _resolve_audio_path(audio_path, base_dir)
            artifact = _transcribe_vllm_http(
                resolved_path,
                settings,
                execution_plan.language,
                cancel_check,
            )
            actual = actual_execution_from_route(
                route,
                device=None,
            )
        elif route.backend == "transformers":
            if (
                route.source != "local"
                or route.audio_egress is not SttAudioEgress.NONE
                or not route.local_model_available
                or route.would_download
                or execution_plan.language is not None
                or runtime.get("device") != route.device
                or runtime.get("dtype") != route.dtype
                or runtime.get("allow_download") is not False
            ):
                raise STTExecutionPlanError(
                    "Qwen3-ASR local route does not match its plan"
                )
            settings["planned_device"] = route.device
            resolved_path = _resolve_audio_path(audio_path, base_dir)
            _check_cancel(cancel_check, label="audio loading")
            audio_np, sample_rate, duration_seconds = _load_audio(
                resolved_path,
                target_sample_rate=int(
                    settings.get("sample_rate") or 16000
                ),
            )
            artifact = _transcribe_local(
                audio_np=audio_np,
                sample_rate=sample_rate,
                duration_seconds=duration_seconds,
                settings=settings,
                language=None,
                word_timestamps=False,
                cancel_check=cancel_check,
            )
            actual = actual_execution_from_route(
                route,
                device=route.device,
                dtype=route.dtype,
            )
        else:
            raise STTExecutionPlanError(
                "Qwen3-ASR plan selected an unsupported route"
            )
        return SttTranscriptionOutcome(
            artifact=artifact,
            actual_execution=actual,
        )

    settings = _resolve_settings()

    if not settings["enabled"]:
        raise BadRequestError(
            "Qwen3-ASR is disabled. Set [STT-Settings].qwen3_asr_enabled=true in config."
        )

    # Override model path if provided
    if model_path and str(model_path).strip():
        settings["model_path"] = str(model_path).strip()

    # Validate word timestamps request
    if word_timestamps and not settings.get("aligner_enabled"):
        logger.warning(
            "Qwen3-ASR: word timestamps requested but aligner not enabled. "
            "Set qwen3_asr_aligner_enabled=true and provide qwen3_asr_aligner_path."
        )

    resolved_path = _resolve_audio_path(audio_path, base_dir)

    # Route to vLLM HTTP backend when configured
    backend = str(settings.get("backend", "")).lower()
    vllm_base_url = str(settings.get("vllm_base_url", "")).strip()
    if backend == "vllm" and vllm_base_url:
        logger.info(f"Qwen3-ASR: using vLLM HTTP backend at {vllm_base_url}")
        return _transcribe_vllm_http(
            resolved_path,
            settings,
            language,
            cancel_check,
        )

    # Use local transformers backend
    _check_cancel(cancel_check, label="audio loading")
    audio_np, sample_rate, duration_seconds = _load_audio(
        resolved_path,
        target_sample_rate=int(settings.get("sample_rate") or 16000),
    )

    return _transcribe_local(
        audio_np=audio_np,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        settings=settings,
        language=language,
        word_timestamps=word_timestamps,
        cancel_check=cancel_check,
    )


__all__ = [
    "is_qwen3_asr_available",
    "is_qwen3_asr_aligner_available",
    "get_qwen3_asr_capabilities",
    "transcribe_with_qwen3_asr",
]
