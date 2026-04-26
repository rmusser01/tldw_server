import base64
import importlib
import io
from typing import Any, Optional

import numpy as np
import soundfile as sf
from fastapi import HTTPException
from starlette import status

from tldw_Server_API.app.core.Audio.error_payloads import _http_error_detail
from tldw_Server_API.app.core.TTS.tts_config import get_tts_config_manager


def _coerce_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_qwen3_tokenizer_settings() -> dict[str, Any]:
    cfg_mgr = get_tts_config_manager()
    provider_cfg = cfg_mgr.get_provider_config("qwen3_tts")
    defaults = {
        "tokenizer_model": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "tokenizer_max_audio_seconds": 300,
        "tokenizer_max_tokens": 20000,
        "tokenizer_max_payload_mb": 20,
        "auto_download": False,
    }
    if provider_cfg is None:
        return defaults
    return {
        "tokenizer_model": provider_cfg.tokenizer_model or defaults["tokenizer_model"],
        "tokenizer_max_audio_seconds": _coerce_int(
            provider_cfg.tokenizer_max_audio_seconds, defaults["tokenizer_max_audio_seconds"]
        ),
        "tokenizer_max_tokens": _coerce_int(provider_cfg.tokenizer_max_tokens, defaults["tokenizer_max_tokens"]),
        "tokenizer_max_payload_mb": _coerce_int(
            provider_cfg.tokenizer_max_payload_mb, defaults["tokenizer_max_payload_mb"]
        ),
        "auto_download": bool(provider_cfg.auto_download),
    }


def _decode_base64_payload(raw: str) -> bytes:
    payload = raw
    if "," in payload:
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload, validate=True)


def _enforce_payload_limit(payload_bytes: bytes, max_payload_mb: int, request_id: Optional[str]) -> None:
    if max_payload_mb <= 0:
        return
    max_bytes = max_payload_mb * 1024 * 1024
    if len(payload_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=_http_error_detail(
                f"Payload too large ({len(payload_bytes)} bytes, max {max_bytes})",
                request_id,
            ),
        )


def _enforce_payload_size(size_bytes: int, max_payload_mb: int, request_id: Optional[str]) -> None:
    if max_payload_mb <= 0:
        return
    max_bytes = max_payload_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=_http_error_detail(
                f"Payload too large ({size_bytes} bytes, max {max_bytes})",
                request_id,
            ),
        )


def _read_audio_from_bytes(
    audio_bytes: bytes,
    sample_rate_hint: Optional[int],
    request_id: Optional[str],
) -> tuple[np.ndarray, int, float]:
    try:
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception:
        if sample_rate_hint is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=_http_error_detail(
                    "Unable to decode audio; provide sample_rate for raw PCM",
                    request_id,
                ),
            ) from None
        pcm = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_data = pcm.astype(np.float32) / 32768.0
        sample_rate = int(sample_rate_hint)

    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    duration_seconds = float(len(audio_data)) / float(sample_rate or 24000)
    return audio_data, int(sample_rate), duration_seconds


def _resolve_tokenizer_sample_rate(tokenizer: Any, fallback: int) -> int:
    for attr in ("sample_rate", "sampling_rate", "sr"):
        value = getattr(tokenizer, attr, None)
        try:
            if value:
                return int(value)
        except Exception:  # nosec B112
            continue
    return fallback


def _resolve_tokenizer_frame_rate(tokenizer: Any) -> Optional[float]:
    for attr in ("frame_rate", "tps", "token_rate"):
        value = getattr(tokenizer, attr, None)
        try:
            if value:
                return float(value)
        except Exception:  # nosec B112
            continue
    return None


def _instantiate_tokenizer(tokenizer_cls: Any, model_id: str, allow_download: bool) -> Any:
    if hasattr(tokenizer_cls, "from_pretrained"):
        try:
            return tokenizer_cls.from_pretrained(model_id, local_files_only=not allow_download)
        except TypeError:
            return tokenizer_cls.from_pretrained(model_id)
    try:
        return tokenizer_cls(model_id)
    except TypeError:
        return tokenizer_cls()


def _load_qwen3_tokenizer(model_id: str, allow_download: bool) -> Any:
    try:
        module = importlib.import_module("qwen_tts")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="qwen-tts package not available",
        ) from exc

    for name in ("Qwen3TTSTokenizer", "QwenTTSTokenizer", "TTSTokenizer"):
        tokenizer_cls = getattr(module, name, None)
        if tokenizer_cls is not None:
            return _instantiate_tokenizer(tokenizer_cls, model_id, allow_download)

    for fn_name in ("load_tokenizer", "get_tokenizer", "create_tokenizer"):
        fn = getattr(module, fn_name, None)
        if callable(fn):
            try:
                return fn(model_id)
            except Exception:
                return fn()

    try:
        tokenizer_mod = importlib.import_module("qwen_tts.tokenizer")
        tokenizer_cls = getattr(tokenizer_mod, "Qwen3TTSTokenizer", None)
        if tokenizer_cls is not None:
            return _instantiate_tokenizer(tokenizer_cls, model_id, allow_download)
    except Exception as tokenizer_error:
        _ = tokenizer_error  # allow fallback error path below

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Qwen3-TTS tokenizer backend is not available in this build",
    )


def _normalize_tokens(tokens: Any) -> tuple[list[int], Optional[float]]:
    frame_rate = None
    if isinstance(tokens, dict):
        frame_rate = tokens.get("frame_rate") or tokens.get("tps")
        tokens = tokens.get("tokens") or tokens.get("codes") or tokens.get("ids")
    if isinstance(tokens, tuple) and len(tokens) == 2:
        tokens, frame_rate = tokens
    if isinstance(tokens, np.ndarray):
        tokens = tokens.tolist()
    if not isinstance(tokens, list):
        tokens = list(tokens)
    return [int(tok) for tok in tokens], frame_rate


def _serialize_tokens(tokens: list[int], token_format: str) -> Any:
    if token_format == "base64":  # nosec B105
        token_bytes = np.asarray(tokens, dtype=np.int32).tobytes()
        return base64.b64encode(token_bytes).decode("ascii")
    return tokens


def _coerce_tokens_payload(payload: Any) -> list[int]:
    if isinstance(payload, list):
        return [int(tok) for tok in payload]
    if isinstance(payload, str):
        data = _decode_base64_payload(payload)
        tokens = np.frombuffer(data, dtype=np.int32).tolist()
        return [int(tok) for tok in tokens]
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="tokens must be a list of ints or base64-encoded bytes",
    )


def _serialize_audio_output(audio: Any, sample_rate: int, response_format: str) -> bytes:
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)

    audio_np = np.asarray(audio)
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1)
    if audio_np.dtype != np.int16:
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_np = (audio_np * 32767.0).astype(np.int16)
    if response_format == "pcm":
        return audio_np.tobytes()
    buffer = io.BytesIO()
    sf.write(buffer, audio_np, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()
