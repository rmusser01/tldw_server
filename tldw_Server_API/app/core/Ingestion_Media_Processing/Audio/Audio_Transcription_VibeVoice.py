"""
VibeVoice-ASR transcription helpers.

This module provides two staged inference paths for VibeVoice-ASR:

1) Local inference via Hugging Face + trust_remote_code
2) Optional HTTP-only vLLM path gated by STT config

The goal here is to keep routing/configuration stable even when the exact
upstream inference API evolves. The parsing logic is intentionally defensive
and normalizes a variety of plausible response shapes into the project's
standard artifact contract.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

import numpy as np
import soundfile as sf  # type: ignore
from loguru import logger

from tldw_Server_API.app.core.config import get_stt_config
from tldw_Server_API.app.core.exceptions import (
    BadRequestError,
    CancelCheckError,
    STTExecutionPlanError,
    STTTranscriptionError,
    TranscriptionCancelled,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
    SttAudioEgress,
    SttBatchExecutionPlan,
    SttTranscriptionOutcome,
    _normalize_audio_endpoint,
    actual_execution_from_route,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.path_utils import (
    open_safe_local_path,
    resolve_safe_local_path,
)
from tldw_Server_API.app.core.testing import is_truthy

# Global cache for local model components
_MODEL_CACHE: dict[str, tuple[Any, Any, str]] = {}
_MODEL_LOCK = threading.Lock()

_VIBEVOICE_PARSE_EXCEPTIONS = (TypeError, ValueError, json.JSONDecodeError)
_VIBEVOICE_RUNTIME_EXCEPTIONS = (
    AttributeError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_VIBEVOICE_INFERENCE_EXCEPTIONS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _as_bool(value: Any, default: bool = False) -> bool:
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
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, default: str) -> str:
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default


def _check_cancel(cancel_check: Callable[[], bool] | None, *, label: str) -> None:
    if cancel_check is None:
        return
    try:
        should_cancel = bool(cancel_check())
    except Exception as exc:
        raise CancelCheckError(f"cancel_check failed during {label}: {exc}") from exc
    if should_cancel:
        raise TranscriptionCancelled(f"Cancelled during {label}")


def _resolve_settings() -> dict[str, Any]:
    try:
        stt_cfg = get_stt_config() or {}
    except _VIBEVOICE_RUNTIME_EXCEPTIONS:
        stt_cfg = {}

    model_id = _as_str(stt_cfg.get("vibevoice_model_id"), "microsoft/VibeVoice-ASR")
    settings = {
        "enabled": _as_bool(stt_cfg.get("vibevoice_enabled"), False),
        "model_id": model_id,
        "model_revision": _as_str(stt_cfg.get("vibevoice_model_revision"), ""),
        "device": _as_str(stt_cfg.get("vibevoice_device"), "cuda"),
        "dtype": _as_str(stt_cfg.get("vibevoice_dtype"), "bfloat16"),
        "cache_dir": _as_str(stt_cfg.get("vibevoice_cache_dir"), "./models/vibevoice"),
        "allow_download": _as_bool(stt_cfg.get("vibevoice_allow_download"), True),
        "sample_rate": _as_int(stt_cfg.get("vibevoice_sample_rate"), 16000),
        "max_new_tokens": _as_int(stt_cfg.get("vibevoice_max_new_tokens"), 4096),
        # Optional vLLM HTTP path
        "vllm_enabled": _as_bool(stt_cfg.get("vibevoice_vllm_enabled"), False),
        "vllm_base_url": str(stt_cfg.get("vibevoice_vllm_base_url") or "").strip(),
        "vllm_model_id": _as_str(stt_cfg.get("vibevoice_vllm_model_id"), model_id),
        "vllm_api_key": str(stt_cfg.get("vibevoice_vllm_api_key") or "").strip() or None,
        "vllm_timeout_seconds": _as_int(stt_cfg.get("vibevoice_vllm_timeout_seconds"), 600),
    }
    return settings


def _resolve_audio_path(audio_path: str, base_dir: Path | None) -> Path:
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
    if sample_rate == target_sample_rate:
        return audio_np, sample_rate
    try:
        import torch
        import torchaudio.functional as F  # type: ignore  # noqa: N812

        wav = torch.from_numpy(np.asarray(audio_np, dtype="float32")).unsqueeze(0)
        resampled = F.resample(wav, sample_rate, target_sample_rate)
        return resampled.squeeze(0).cpu().numpy().astype("float32"), target_sample_rate
    except _VIBEVOICE_INFERENCE_EXCEPTIONS as exc:
        logger.warning(
            'VibeVoice: resampling from {} Hz to {} Hz failed; proceeding at original rate. Error: {}',
            sample_rate,
            target_sample_rate,
            exc,
        )
        return audio_np, sample_rate


def _coerce_hotwords(hotwords: Sequence[str] | str | None) -> list[str]:
    if hotwords is None:
        return []
    if isinstance(hotwords, str):
        raw = hotwords.strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except _VIBEVOICE_PARSE_EXCEPTIONS:
                # Fall through to CSV handling
                pass
        return [part.strip() for part in raw.split(",") if part.strip()]
    out: list[str] = []
    for item in hotwords:
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def _speaker_label_from_id(speaker_id: Any) -> str:
    try:
        sid = int(speaker_id)
        return f"SPEAKER_{sid}"
    except (TypeError, ValueError):
        s = str(speaker_id).strip()
        return s or "SPEAKER_0"


def _normalize_segments(segments: Iterable[Any], *, duration_seconds: float) -> list[dict[str, Any]]:
    segments_list = [seg for seg in segments if isinstance(seg, dict)]
    total_segments = len(segments_list)
    slice_len = duration_seconds / max(1, total_segments) if duration_seconds > 0 else 0.0

    normalized: list[dict[str, Any]] = []
    for seg in segments_list:

        start_raw = (
            seg.get("start_seconds")
            or seg.get("start")
            or seg.get("start_time")
            or seg.get("begin")
            or 0.0
        )
        end_raw = (
            seg.get("end_seconds")
            or seg.get("end")
            or seg.get("end_time")
            or seg.get("finish")
            or start_raw
        )
        try:
            start_s = float(start_raw)
        except (TypeError, ValueError):
            start_s = 0.0
        try:
            end_s = float(end_raw)
        except (TypeError, ValueError):
            end_s = start_s

        if end_s < start_s:
            end_s = start_s
        if end_s == start_s and duration_seconds > 0:
            # Best-effort spread when timestamps are missing
            end_s = min(duration_seconds, start_s + slice_len)

        text = (
            seg.get("Text")
            or seg.get("text")
            or seg.get("content")
            or seg.get("utterance")
            or ""
        )
        text = str(text).strip()
        if not text:
            continue

        speaker = (
            seg.get("speaker")
            or seg.get("speaker_label")
            or seg.get("speakerLabel")
            or seg.get("speaker_id")
            or seg.get("speakerId")
        )

        entry: dict[str, Any] = {
            "start_seconds": float(max(0.0, start_s)),
            "end_seconds": float(max(start_s, end_s)),
            "Text": text,
        }
        if speaker is not None and str(speaker).strip():
            entry["speaker"] = str(speaker).strip()
        if "speaker_id" in seg:
            entry["speaker_id"] = seg.get("speaker_id")
            entry["speaker"] = _speaker_label_from_id(seg.get("speaker_id"))
        if "speaker_label" in seg and seg.get("speaker_label"):
            entry["speaker_label"] = seg.get("speaker_label")
            entry["speaker"] = str(seg.get("speaker_label")).strip()
        normalized.append(entry)

    # If we were unable to parse any segments, return an empty list and let
    # the caller fall back to a single-segment transcript.
    return normalized


def _extract_text_from_response(resp: Any) -> str:
    if isinstance(resp, str):
        return resp.strip()
    if isinstance(resp, dict):
        for key in ("text", "transcript", "content", "output_text"):
            if key in resp and str(resp.get(key, "")).strip():
                return str(resp.get(key)).strip()
        # OpenAI-style chat fallback
        try:
            return (
                resp.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            ).strip()
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return ""
    return ""


def _count_speakers(segments: Sequence[dict[str, Any]]) -> int | None:
    speakers = {str(seg.get("speaker")).strip() for seg in segments if seg.get("speaker")}
    return len(speakers) if speakers else None


def _normalize_artifact(
    raw_resp: Any,
    *,
    duration_seconds: float,
    language_hint: str | None,
    model_id: str,
    source: str,
    hotwords: Sequence[str],
) -> dict[str, Any]:
    language_out: str | None = None
    segments_out: list[dict[str, Any]] = []

    if isinstance(raw_resp, dict):
        language_out = (
            raw_resp.get("language")
            or raw_resp.get("detected_language")
            or raw_resp.get("lang")
            or None
        )
        segments_raw = (
            raw_resp.get("segments")
            or raw_resp.get("utterances")
            or raw_resp.get("results")
            or None
        )
        if isinstance(segments_raw, list):
            segments_out = _normalize_segments(segments_raw, duration_seconds=duration_seconds)

    text_out = _extract_text_from_response(raw_resp)
    if not segments_out and text_out:
        segments_out = [
            {
                "start_seconds": 0.0,
                "end_seconds": float(duration_seconds if duration_seconds > 0 else 0.0),
                "Text": text_out,
            }
        ]
    elif segments_out and not text_out:
        text_out = " ".join(str(seg.get("Text", "")).strip() for seg in segments_out if seg.get("Text"))

    diarization_enabled = any(seg.get("speaker") for seg in segments_out)
    diarization_speakers = _count_speakers(segments_out) if diarization_enabled else None

    return {
        "text": text_out,
        "language": language_out or language_hint,
        "segments": segments_out,
        "diarization": {"enabled": diarization_enabled, "speakers": diarization_speakers},
        "usage": {"duration_ms": int(duration_seconds * 1000.0) if duration_seconds > 0 else None, "tokens": None},
        "metadata": {
            "provider": "vibevoice",
            "model": model_id,
            "source": source,
            "hotwords": list(hotwords),
        },
    }


def _get_torch_dtype(dtype_name: str) -> Any:
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
    return mapping.get(str(dtype_name).strip().lower(), torch.float32)


def _resolve_device(requested_device: str) -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    dev = (requested_device or "cpu").strip().lower()
    if dev.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("VibeVoice: CUDA requested but not available; falling back to CPU")
        return "cpu"
    return dev or "cpu"


def _load_local_components(settings: dict[str, Any]) -> tuple[Any, Any, str]:
    model_id = str(settings["model_id"])
    device = _resolve_device(str(settings["device"]))
    dtype_name = str(settings["dtype"])
    allow_download = bool(settings["allow_download"])
    model_revision = str(settings.get("model_revision") or "").strip() or None
    cache_dir = Path(str(settings["cache_dir"]))
    cache_dir.mkdir(parents=True, exist_ok=True)
    planned_device = settings.get("planned_device")
    if planned_device is not None and device != str(planned_device):
        raise STTExecutionPlanError(
            "VibeVoice loaded device does not match the execution plan"
        )

    cache_key = (
        f"{model_id}|{model_revision}|{device}|{dtype_name}|"
        f"{allow_download}"
    )
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
        except Exception as exc:
            raise BadRequestError(
                "transformers is required for local VibeVoice-ASR inference. "
                "Install with: pip install transformers"
            ) from exc


        torch_dtype = _get_torch_dtype(dtype_name)
        local_only = not allow_download

        processor = AutoProcessor.from_pretrained(
            model_id,
            revision=model_revision,
            trust_remote_code=True,
            cache_dir=str(cache_dir),
            local_files_only=local_only,
        )  # nosec B615

        device_map = "auto" if device != "cpu" else None
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=model_revision,
            trust_remote_code=True,
            cache_dir=str(cache_dir),
            local_files_only=local_only,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )  # nosec B615

        if device_map is None:
            model = model.to(device)
        model.eval()

        _MODEL_CACHE[cache_key] = (processor, model, device)
        logger.info("VibeVoice: loaded local model '{}' on device '{}'", model_id, device)
        return processor, model, device


def _build_processor_inputs(
    processor: Any,
    *,
    audio_np: np.ndarray,
    sample_rate: int,
    language: str | None,
    hotwords: Sequence[str],
    strict_semantics: bool = False,
) -> Any:
    base_kwargs: dict[str, Any] = {
        "audio": audio_np,
        "sampling_rate": sample_rate,
        "return_tensors": "pt",
    }
    if language:
        base_kwargs["language"] = language
    if hotwords:
        base_kwargs["hotwords"] = list(hotwords)

    try:
        return processor(**base_kwargs)
    except TypeError:
        if strict_semantics and (language or hotwords):
            raise STTExecutionPlanError(
                "VibeVoice local processor dropped planned semantics"
            ) from None
        # Retry without optional fields when the processor does not accept them.
        base_kwargs.pop("hotwords", None)
        try:
            return processor(**base_kwargs)
        except TypeError:
            base_kwargs.pop("language", None)
            return processor(**base_kwargs)


def _move_inputs_to_device(inputs: Any, device: str) -> Any:
    try:
        import torch
    except ImportError:
        return inputs
    if device == "cpu":
        return inputs
    dev = torch.device(device)
    if hasattr(inputs, "to"):
        try:
            return inputs.to(dev)
        except _VIBEVOICE_INFERENCE_EXCEPTIONS:
            return inputs
    if isinstance(inputs, dict):
        out: dict[str, Any] = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                try:
                    out[k] = v.to(dev)
                    continue
                except _VIBEVOICE_INFERENCE_EXCEPTIONS:
                    pass
            out[k] = v
        return out
    return inputs


def _decode_generated(processor: Any, generated_ids: Any) -> str:
    try:
        if hasattr(processor, "batch_decode"):
            decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
            if isinstance(decoded, list) and decoded:
                return str(decoded[0]).strip()
        if hasattr(processor, "decode"):
            # Some processors expose decode for a single sequence.
            if isinstance(generated_ids, (list, tuple)) and generated_ids:
                return str(processor.decode(generated_ids[0])).strip()
            return str(processor.decode(generated_ids)).strip()
    except _VIBEVOICE_RUNTIME_EXCEPTIONS:
        pass
    return str(generated_ids).strip()


def _transcribe_local(
    *,
    audio_np: np.ndarray,
    sample_rate: int,
    duration_seconds: float,
    settings: dict[str, Any],
    language: str | None,
    hotwords: Sequence[str],
    cancel_check: Callable[[], bool] | None,
) -> dict[str, Any]:
    strict_semantics = bool(settings.get("strict_semantics"))
    _check_cancel(cancel_check, label="local model load")
    processor, model, device = _load_local_components(settings)
    _check_cancel(cancel_check, label="local preprocessing")

    # Prefer a dedicated transcribe method when the upstream implementation
    # exposes one; otherwise, fall back to processor+generate.
    if hasattr(model, "transcribe"):
        try:
            raw_resp = model.transcribe(
                audio_np,
                sample_rate=sample_rate,
                language=language,
                hotwords=list(hotwords) if hotwords else None,
            )
            return _normalize_artifact(
                raw_resp,
                duration_seconds=duration_seconds,
                language_hint=language,
                model_id=str(settings["model_id"]),
                source="local",
                hotwords=hotwords,
            )
        except TypeError:
            if strict_semantics and (language or hotwords):
                raise STTExecutionPlanError(
                    "VibeVoice local model dropped planned semantics"
                ) from None
            raw_resp = model.transcribe(audio_np, sample_rate=sample_rate)
            return _normalize_artifact(
                raw_resp,
                duration_seconds=duration_seconds,
                language_hint=language,
                model_id=str(settings["model_id"]),
                source="local",
                hotwords=hotwords,
            )

    try:
        import torch
    except Exception as exc:
        raise BadRequestError("torch is required for local VibeVoice-ASR inference") from exc

    inputs = _build_processor_inputs(
        processor,
        audio_np=audio_np,
        sample_rate=sample_rate,
        language=language,
        hotwords=hotwords,
        strict_semantics=strict_semantics,
    )
    inputs = _move_inputs_to_device(inputs, device)

    gen_kwargs: dict[str, Any] = {
        "do_sample": False,
        "max_new_tokens": int(settings.get("max_new_tokens") or 4096),
    }
    if language:
        gen_kwargs["language"] = language
    if hotwords:
        gen_kwargs["hotwords"] = list(hotwords)

    _check_cancel(cancel_check, label="local generation")
    with torch.no_grad():
        try:
            generated_ids = model.generate(**inputs, **gen_kwargs)
        except TypeError:
            if strict_semantics and hotwords:
                raise STTExecutionPlanError(
                    "VibeVoice generation dropped planned hotwords"
                ) from None
            gen_kwargs.pop("hotwords", None)
            generated_ids = model.generate(**inputs, **gen_kwargs)

    text = _decode_generated(processor, generated_ids)
    raw_resp = {"text": text}
    return _normalize_artifact(
        raw_resp,
        duration_seconds=duration_seconds,
        language_hint=language,
        model_id=str(settings["model_id"]),
        source="local",
        hotwords=hotwords,
    )


def _resolve_vllm_endpoint(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.path.rstrip("/").endswith("/v1/audio/transcriptions"):
        return base_url
    return urljoin(base_url.rstrip("/") + "/", "v1/audio/transcriptions")


def _audio_duration_seconds(audio_path: Path) -> float:
    try:
        info = sf.info(str(audio_path))
        if info.samplerate and info.frames:
            return float(info.frames) / float(info.samplerate)
    except (OSError, RuntimeError, TypeError, ValueError):
        pass
    return 0.0


def _transcribe_via_vllm_http(
    *,
    audio_path: Path,
    base_dir: Path,
    settings: dict[str, Any],
    language: str | None,
    hotwords: Sequence[str],
    cancel_check: Callable[[], bool] | None,
) -> dict[str, Any]:
    base_url = str(settings.get("vllm_base_url") or "").strip()
    endpoint = str(settings.get("endpoint") or "").strip()
    if not endpoint and not base_url:
        raise BadRequestError("vibevoice_vllm_enabled is true but vibevoice_vllm_base_url is not set")

    endpoint = endpoint or _resolve_vllm_endpoint(base_url)
    duration_seconds = _audio_duration_seconds(audio_path)
    _check_cancel(cancel_check, label="vLLM request preparation")

    file_handle = open_safe_local_path(audio_path, base_dir, mode="rb")
    if file_handle is None:
        raise BadRequestError("Audio file path rejected outside allowed base directory")

    headers: dict[str, str] = {}
    api_key = settings.get("vllm_api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    model_id = str(settings.get("vllm_model_id") or settings.get("model_id") or "microsoft/VibeVoice-ASR")
    data: dict[str, Any] = {
        "model": model_id,
        "response_format": "json",
    }
    if language:
        data["language"] = language
    if hotwords:
        data["hotwords"] = json.dumps(list(hotwords))

    from tldw_Server_API.app.core.http_client import fetch_json

    timeout = int(settings.get("vllm_timeout_seconds") or 600)
    with file_handle:
        files = {"file": (audio_path.name, file_handle, "application/octet-stream")}
        raw_resp = fetch_json(
            method="POST",
            url=endpoint,
            headers=headers,
            data=data,
            files=files,
            timeout=timeout,
            allow_redirects=False,
        )

    _check_cancel(cancel_check, label="vLLM response parsing")
    return _normalize_artifact(
        raw_resp,
        duration_seconds=duration_seconds,
        language_hint=language,
        model_id=model_id,
        source="vllm_http",
        hotwords=hotwords,
    )


def is_vibevoice_available() -> bool:
    """Lightweight availability check for local VibeVoice-ASR support."""
    settings = _resolve_settings()
    if not settings["enabled"]:
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def transcribe_with_vibevoice(
    audio_path: str,
    *,
    model_id: str | None = None,
    language: str | None = None,
    hotwords: Sequence[str] | str | None = None,
    base_dir: Path | None = None,
    cancel_check: Callable[[], bool] | None = None,
    execution_plan: SttBatchExecutionPlan | None = None,
) -> dict[str, Any] | SttTranscriptionOutcome:
    """
    Transcribe audio via VibeVoice-ASR.

    Routing rules:
    - If vibevoice_vllm_enabled=true and a base URL is configured, prefer the
      vLLM HTTP path.
    - Otherwise use local inference when vibevoice_enabled=true.
    """
    if execution_plan is not None:
        routes = execution_plan.descriptor.routes
        runtime = execution_plan.runtime_values()
        if (
            not routes
            or any(
                route.provider != "vibevoice"
                or route.model_label
                != execution_plan.descriptor.resolved_model_label
                for route in routes
            )
        ):
            raise STTExecutionPlanError(
                "Invalid VibeVoice execution routes"
            )
        for route in routes:
            if route.backend == "vllm_http":
                _endpoint, egress, endpoint_id = (
                    _normalize_audio_endpoint(
                        str(runtime.get("endpoint") or "")
                    )
                )
                if (
                    route.source != "vllm_http"
                    or route.audio_egress is not egress
                    or route.endpoint_id != endpoint_id
                ):
                    raise STTExecutionPlanError(
                        "VibeVoice endpoint does not match its plan"
                    )
            elif route.backend == "transformers":
                if (
                    route.source != "local"
                    or route.audio_egress is not SttAudioEgress.NONE
                    or not route.local_model_available
                    or route.would_download
                    or runtime.get("device") != route.device
                    or runtime.get("dtype") != route.dtype
                    or runtime.get("allow_download") is not False
                    or runtime.get("strict_semantics") is not True
                ):
                    raise STTExecutionPlanError(
                        "VibeVoice local route does not match its plan"
                    )
            else:
                raise STTExecutionPlanError(
                    "VibeVoice plan contains an unsupported route"
                )
        resolved_path = _resolve_audio_path(audio_path, base_dir)
        resolved_base = (
            Path(base_dir)
            if base_dir is not None
            else resolved_path.parent
        )
        hotwords_list = list(execution_plan.hotwords)
        final_error: BaseException | None = None
        for route in routes:
            settings = dict(runtime)
            try:
                if route.backend == "vllm_http":
                    artifact = _transcribe_via_vllm_http(
                        audio_path=resolved_path,
                        base_dir=resolved_base,
                        settings=settings,
                        language=execution_plan.language,
                        hotwords=hotwords_list,
                        cancel_check=cancel_check,
                    )
                    actual = actual_execution_from_route(
                        route,
                        device=None,
                    )
                else:
                    settings["planned_device"] = route.device
                    _check_cancel(
                        cancel_check,
                        label="audio loading",
                    )
                    audio_np, sample_rate, duration_seconds = (
                        _load_audio(
                            resolved_path,
                            target_sample_rate=int(
                                settings.get("sample_rate")
                                or 16000
                            ),
                        )
                    )
                    artifact = _transcribe_local(
                        audio_np=audio_np,
                        sample_rate=sample_rate,
                        duration_seconds=duration_seconds,
                        settings=settings,
                        language=execution_plan.language,
                        hotwords=hotwords_list,
                        cancel_check=cancel_check,
                    )
                    actual = actual_execution_from_route(
                        route,
                        device=route.device,
                        dtype=route.dtype,
                    )
                return SttTranscriptionOutcome(
                    artifact=artifact,
                    actual_execution=actual,
                )
            except (
                STTExecutionPlanError,
                TranscriptionCancelled,
            ):
                raise
            except Exception as exc:  # noqa: BLE001 - provider routes may raise optional-runtime errors
                final_error = exc
        if final_error is not None:
            raise STTTranscriptionError(
                "Planned VibeVoice transcription failed"
            ) from None
        raise STTExecutionPlanError(
            "VibeVoice plan did not select a route"
        )

    settings = _resolve_settings()
    original_model_id = str(settings.get("model_id") or "microsoft/VibeVoice-ASR")
    if model_id and str(model_id).strip():
        override = str(model_id).strip()
        settings["model_id"] = override
        if not settings.get("vllm_model_id") or str(settings.get("vllm_model_id")).strip() in {"", original_model_id}:
            settings["vllm_model_id"] = override
    hotwords_list = _coerce_hotwords(hotwords)

    resolved_path = _resolve_audio_path(audio_path, base_dir)
    resolved_base = Path(base_dir) if base_dir is not None else resolved_path.parent

    if settings.get("vllm_enabled"):
        try:
            return _transcribe_via_vllm_http(
                audio_path=resolved_path,
                base_dir=resolved_base,
                settings=settings,
                language=language,
                hotwords=hotwords_list,
                cancel_check=cancel_check,
            )
        except Exception as exc:
            logger.warning("VibeVoice vLLM HTTP path failed; falling back to local inference: {}", exc)
            if not settings.get("enabled"):
                raise

    if not settings.get("enabled"):
        raise BadRequestError(
            "VibeVoice-ASR local inference is disabled. Set [STT-Settings].vibevoice_enabled=true "
            "or configure the vLLM HTTP path."
        )

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
        hotwords=hotwords_list,
        cancel_check=cancel_check,
    )


__all__ = [
    "is_vibevoice_available",
    "transcribe_with_vibevoice",
]
