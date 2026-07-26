"""Strict configuration and model selection for the external audio.cpp server."""

from __future__ import annotations

import ipaddress
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlparse

from tldw_Server_API.app.core.exceptions import STTExecutionUnsupportedError
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_execution_contract import (
    _normalize_audio_endpoint,
)

AUDIO_CPP_ENABLED_ENV = "STT_AUDIO_CPP_ENABLED"
AUDIO_CPP_BASE_URL_ENV = "STT_AUDIO_CPP_BASE_URL"
AUDIO_CPP_DEFAULT_MODEL_ENV = "STT_AUDIO_CPP_DEFAULT_MODEL"
AUDIO_CPP_TIMEOUT_SECONDS_ENV = "STT_AUDIO_CPP_TIMEOUT_SECONDS"

_TRUE_TOKENS = frozenset({"1", "true", "yes", "y", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "n", "off"})
_SELECTORS = ("audio-cpp", "audiocpp", "audio_cpp")
_MODEL_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._+-]*"
    r"(?:/[A-Za-z0-9][A-Za-z0-9._+-]*)?$"
)
_MAX_MODEL_ID_LENGTH = 256


@dataclass(frozen=True)
class AudioCppConfig:
    """Validated settings for one external audio.cpp server."""

    enabled: bool
    origin: str
    default_model: str | None
    timeout_seconds: float


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
