"""Strict configuration and local contract validation for audio.cpp."""

from __future__ import annotations

import ipaddress
import json
import math
import os
import re
import stat
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, NoReturn, cast
from urllib.parse import urlparse

from tldw_Server_API.app.core.exceptions import STTExecutionUnsupportedError
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
    _normalize_audio_endpoint,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.path_utils import (
    open_safe_local_path,
)

AUDIO_CPP_ENABLED_ENV = "STT_AUDIO_CPP_ENABLED"
AUDIO_CPP_BASE_URL_ENV = "STT_AUDIO_CPP_BASE_URL"
AUDIO_CPP_DEFAULT_MODEL_ENV = "STT_AUDIO_CPP_DEFAULT_MODEL"
AUDIO_CPP_TIMEOUT_SECONDS_ENV = "STT_AUDIO_CPP_TIMEOUT_SECONDS"

MAX_AUDIO_CPP_RESPONSE_BYTES = 1_048_576
MAX_AUDIO_CPP_CATALOG_ENTRIES = 256
MAX_AUDIO_CPP_HEALTH_MODELS = MAX_AUDIO_CPP_CATALOG_ENTRIES
MAX_AUDIO_CPP_JSON_INTEGER_DIGITS = 128
MAX_AUDIO_CPP_TRANSCRIPT_CHARS = 262_144

_TRUE_TOKENS = frozenset({"1", "true", "yes", "y", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "n", "off"})
_SELECTORS = ("audio-cpp", "audiocpp", "audio_cpp")
_SUPPORTED_ASR_MODES = frozenset({"offline", "streaming"})
_MODEL_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._+-]*"
    r"(?:/[A-Za-z0-9][A-Za-z0-9._+-]*)?$"
)
_CONTRACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_MAX_MODEL_ID_LENGTH = 256
_MAX_CONTRACT_ID_LENGTH = 64
_WAV_READ_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True)
class AudioCppConfig:
    """Validated settings for one external audio.cpp server."""

    enabled: bool
    origin: str
    default_model: str | None
    timeout_seconds: float


@dataclass(frozen=True)
class AudioCppDiscovery:
    """Validated audio.cpp model metadata selected from discovery."""

    backend: str
    model_id: str
    family: str
    mode: str


class _InvalidAudioCppJSON(ValueError):
    """Internal marker for invalid JSON values."""


def _reject_json_constant(_value: str) -> NoReturn:
    raise _InvalidAudioCppJSON


def _parse_bounded_json_int(value: str) -> int:
    digits = value[1:] if value.startswith("-") else value
    if len(digits) > MAX_AUDIO_CPP_JSON_INTEGER_DIGITS:
        raise _InvalidAudioCppJSON
    return int(value)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidAudioCppJSON
        result[key] = value
    return result


def _parse_json_object(body: bytes, *, response_name: str) -> dict[str, object]:
    message = f"audio.cpp {response_name} response is invalid"
    if not isinstance(body, bytes) or len(body) > MAX_AUDIO_CPP_RESPONSE_BYTES:
        raise STTExecutionUnsupportedError(message)
    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_int=_parse_bounded_json_int,
        )
    except (RecursionError, TypeError, ValueError):
        raise STTExecutionUnsupportedError(message) from None
    if not isinstance(value, dict):
        raise STTExecutionUnsupportedError(message)
    return value


def _is_safe_contract_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) <= _MAX_CONTRACT_ID_LENGTH
        and _CONTRACT_ID_RE.fullmatch(value) is not None
    )


def _is_safe_model_id(value: object) -> bool:
    return isinstance(value, str) and len(value) <= _MAX_MODEL_ID_LENGTH and _MODEL_ID_RE.fullmatch(value) is not None


def parse_audio_cpp_health(body: bytes) -> str:
    """Validate one bounded ``/health`` response and return its backend."""
    payload = _parse_json_object(body, response_name="health")
    backend = payload.get("backend")
    model_count = payload.get("models")
    if (
        payload.get("status") != "ok"
        or not _is_safe_contract_id(backend)
        or isinstance(model_count, bool)
        or not isinstance(model_count, int)
        or model_count < 0
        or model_count > MAX_AUDIO_CPP_HEALTH_MODELS
    ):
        raise STTExecutionUnsupportedError("audio.cpp health response is invalid")
    return backend


def parse_audio_cpp_catalog(
    body: bytes,
    *,
    backend: str,
    model_id: str,
) -> AudioCppDiscovery:
    """Validate a bounded model catalog and select an exact ASR model."""
    requested_model = _safe_model_id(model_id)
    if not _is_safe_contract_id(backend):
        raise STTExecutionUnsupportedError("audio.cpp catalog response is invalid")
    payload = _parse_json_object(body, response_name="catalog")
    entries = payload.get("data")
    if payload.get("object") != "list" or not isinstance(entries, list) or len(entries) > MAX_AUDIO_CPP_CATALOG_ENTRIES:
        raise STTExecutionUnsupportedError("audio.cpp catalog response is invalid")

    selected: AudioCppDiscovery | None = None
    seen_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise STTExecutionUnsupportedError("audio.cpp catalog response is invalid")
        entry_id = entry.get("id")
        family = entry.get("family")
        task = entry.get("task")
        mode = entry.get("mode")
        if (
            not _is_safe_model_id(entry_id)
            or entry_id in seen_ids
            or not _is_safe_contract_id(family)
            or not _is_safe_contract_id(task)
            or not _is_safe_contract_id(mode)
        ):
            raise STTExecutionUnsupportedError("audio.cpp catalog response is invalid")
        seen_ids.add(entry_id)
        if entry_id == requested_model:
            if task != "asr" or mode not in _SUPPORTED_ASR_MODES:
                raise STTExecutionUnsupportedError("audio.cpp catalog response is invalid")
            selected = AudioCppDiscovery(
                backend=backend,
                model_id=entry_id,
                family=family,
                mode=mode,
            )

    if selected is None:
        raise STTExecutionUnsupportedError("audio.cpp requested model is unavailable")
    return selected


def parse_audio_cpp_transcription(body: bytes) -> str:
    """Validate a bounded transcription response and return its exact text."""
    payload = _parse_json_object(body, response_name="transcription")
    text = payload.get("text")
    if not isinstance(text, str) or len(text) > MAX_AUDIO_CPP_TRANSCRIPT_CHARS:
        raise STTExecutionUnsupportedError("audio.cpp transcription response is invalid")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise STTExecutionUnsupportedError("audio.cpp transcription response is invalid") from None
    return text


def _raw_setting(
    settings: Mapping[str, object],
    key: str,
    *,
    environment: Mapping[str, str],
    environment_key: str,
    default: object,
) -> object:
    if environment_key in environment:
        return environment[environment_key]
    return settings.get(key, default)


def _parse_enabled(raw: object) -> bool:
    if not isinstance(raw, str):
        raise STTExecutionUnsupportedError("audio.cpp enabled setting is invalid")
    token = raw.strip().casefold()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    raise STTExecutionUnsupportedError("audio.cpp enabled setting is invalid")


def _parse_timeout(raw: object) -> float:
    if isinstance(raw, bool) or type(raw) not in {str, int, float}:
        raise STTExecutionUnsupportedError("audio.cpp timeout setting is invalid")
    try:
        timeout = float(raw)
    except (OverflowError, TypeError, ValueError):
        raise STTExecutionUnsupportedError("audio.cpp timeout setting is invalid") from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise STTExecutionUnsupportedError("audio.cpp timeout setting is invalid")
    return timeout


def _canonical_origin(raw: object) -> str:
    if not isinstance(raw, str) or raw != raw.strip():
        raise STTExecutionUnsupportedError("audio.cpp origin is invalid")
    try:
        parsed = urlparse(raw)
        if parsed.path not in {"", "/"}:
            raise ValueError
        hostname = parsed.hostname
        if hostname is None:
            raise ValueError
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None
        if ("[" in parsed.netloc or "]" in parsed.netloc) and not isinstance(address, ipaddress.IPv6Address):
            raise ValueError
        if address is None and len(hostname) > 253:
            raise ValueError
        normalized, _egress, _endpoint_id = _normalize_audio_endpoint(raw)
        normalized_parsed = urlparse(normalized)
    except (STTExecutionUnsupportedError, TypeError, ValueError):
        raise STTExecutionUnsupportedError("audio.cpp origin is invalid") from None
    return f"{normalized_parsed.scheme}://{normalized_parsed.netloc}"


def _safe_model_id(raw: object) -> str:
    if not isinstance(raw, str):
        raise STTExecutionUnsupportedError("audio.cpp model is invalid")
    stripped = raw.strip()
    if not stripped:
        raise STTExecutionUnsupportedError("audio.cpp model is required")
    if raw != stripped:
        raise STTExecutionUnsupportedError("audio.cpp model is invalid")
    model_id = raw
    if len(model_id) > _MAX_MODEL_ID_LENGTH or _MODEL_ID_RE.fullmatch(model_id) is None:
        raise STTExecutionUnsupportedError("audio.cpp model is invalid")
    return model_id


def load_audio_cpp_config(
    stt_settings: Mapping[str, object],
    *,
    env: Mapping[str, str] | None = None,
) -> AudioCppConfig:
    """Load and strictly validate raw audio.cpp settings with env precedence."""
    environment = os.environ if env is None else env
    enabled = _parse_enabled(
        _raw_setting(
            stt_settings,
            "audio_cpp_enabled",
            environment=environment,
            environment_key=AUDIO_CPP_ENABLED_ENV,
            default="false",
        )
    )
    origin = _canonical_origin(
        _raw_setting(
            stt_settings,
            "audio_cpp_base_url",
            environment=environment,
            environment_key=AUDIO_CPP_BASE_URL_ENV,
            default="http://127.0.0.1:8080",
        )
    )
    raw_default_model = _raw_setting(
        stt_settings,
        "audio_cpp_default_model",
        environment=environment,
        environment_key=AUDIO_CPP_DEFAULT_MODEL_ENV,
        default="",
    )
    if not isinstance(raw_default_model, str):
        raise STTExecutionUnsupportedError("audio.cpp model is invalid")
    default_model = _safe_model_id(raw_default_model) if raw_default_model.strip() else None
    timeout_seconds = _parse_timeout(
        _raw_setting(
            stt_settings,
            "audio_cpp_timeout_seconds",
            environment=environment,
            environment_key=AUDIO_CPP_TIMEOUT_SECONDS_ENV,
            default="600",
        )
    )
    return AudioCppConfig(
        enabled=enabled,
        origin=origin,
        default_model=default_model,
        timeout_seconds=timeout_seconds,
    )


def normalize_audio_cpp_model(
    model: str | None,
    *,
    default_model: str | None,
) -> str:
    """Return the exact safe server model selected for audio.cpp."""
    if model is None:
        selected: object = default_model
    elif not isinstance(model, str):
        selected = model
    else:
        selected = model
        if selected in _SELECTORS:
            selected = default_model
        else:
            for selector in _SELECTORS:
                prefix = f"{selector}:"
                if selected.startswith(prefix):
                    selected = selected.removeprefix(prefix)
                    break
    if selected is None:
        raise STTExecutionUnsupportedError("audio.cpp model is required")
    return _safe_model_id(selected)


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise ValueError
    return value


def _validate_audio_cpp_pcm_wav(handle: BinaryIO, *, file_size: int) -> None:
    header = _read_exact(handle, 12)
    riff_size = int.from_bytes(header[4:8], byteorder="little")
    if header[:4] != b"RIFF" or header[8:] != b"WAVE" or riff_size < 4 or riff_size + 8 != file_size:
        raise ValueError
    handle.seek(0)

    reader = wave.open(handle, "rb")
    try:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        frame_rate = reader.getframerate()
        frame_count = reader.getnframes()
        frame_size = channels * sample_width
        if (
            reader.getcomptype() != "NONE"
            or channels < 1
            or sample_width < 1
            or frame_rate < 1
            or frame_count < 1
            or frame_size > _WAV_READ_CHUNK_BYTES
        ):
            raise ValueError

        expected_total = frame_count * frame_size
        if expected_total > file_size:
            raise ValueError
        frames_per_chunk = max(1, _WAV_READ_CHUNK_BYTES // frame_size)
        remaining_frames = frame_count
        total_bytes = 0
        while remaining_frames:
            requested_frames = min(remaining_frames, frames_per_chunk)
            chunk = reader.readframes(requested_frames)
            expected_chunk_bytes = requested_frames * frame_size
            if len(chunk) != expected_chunk_bytes:
                raise ValueError
            total_bytes += len(chunk)
            remaining_frames -= requested_frames
        if total_bytes != expected_total or reader.readframes(1):
            raise ValueError
    finally:
        reader.close()

    if handle.closed:
        raise ValueError
    handle.seek(0)


def open_audio_cpp_wav(
    path: str | os.PathLike[str],
    *,
    base_dir: str | os.PathLike[str],
) -> BinaryIO:
    """Open and fully validate one regular uncompressed PCM RIFF/WAVE file."""
    message = "audio.cpp WAV input is invalid"
    handle: BinaryIO | None = None
    try:
        path_object = Path(path)
        base_path = Path(base_dir).resolve(strict=False)
        candidate = path_object if path_object.is_absolute() else base_path / path_object
        if candidate.suffix.casefold() != ".wav":
            raise ValueError
        before = os.lstat(candidate)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError

        opened = open_safe_local_path(candidate, base_path, mode="rb")
        if opened is None:
            raise ValueError
        handle = cast(BinaryIO, opened)
        after = os.fstat(handle.fileno())
        if not stat.S_ISREG(after.st_mode) or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError

        _validate_audio_cpp_pcm_wav(handle, file_size=after.st_size)
        handle.seek(0)
        return handle
    except (
        OSError,
        OverflowError,
        TypeError,
        UnicodeError,
        ValueError,
        EOFError,
        wave.Error,
    ):
        if handle is not None:
            handle.close()
        raise STTExecutionUnsupportedError(message) from None
