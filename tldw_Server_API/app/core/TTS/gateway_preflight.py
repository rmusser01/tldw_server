"""Pure, config-only validation for persisted explicit TTS gateway routes."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters.base import AudioFormat
from .gateway_config import (
    GatewaySpec,
    canonicalize_gateway_id,
    validate_gateway_extra_params,
)

_KEY_NORMALIZER = re.compile(r"[^a-z0-9]")
_CAMEL_BOUNDARY_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")
_FORBIDDEN_PERSISTED_KEYS = frozenset(
    {
        "access_token",
        "auth",
        "api_key",
        "authorization",
        "base_url",
        "credential",
        "credentials",
        "credential_revision",
        "credential_scope",
        "credential_scope_token",
        "credential_source",
        "endpoint",
        "header",
        "headers",
        "host",
        "hostname",
        "key",
        "keys",
        "models_path",
        "origin",
        "path",
        "password",
        "private_key",
        "provider_override",
        "provider_overrides",
        "refresh_token",
        "speech_path",
        "token",
        "url",
    }
)
_FORBIDDEN_NORMALIZED_KEYS = frozenset(
    _KEY_NORMALIZER.sub("", key.casefold()) for key in _FORBIDDEN_PERSISTED_KEYS
)
_SUPPORTED_FIELDS = frozenset(
    {
        "allow_fallback",
        "backend",
        "extra_params",
        "input",
        "lang_code",
        "language",
        "model",
        "response_format",
        "speed",
        "stream",
        "target_sample_rate",
        "voice",
    }
)


def _persisted_key_tokens(key: str) -> tuple[str, ...]:
    split = _CAMEL_BOUNDARY_1.sub(r"\1 \2", key)
    split = _CAMEL_BOUNDARY_2.sub(r"\1 \2", split)
    return tuple(part.casefold() for part in re.findall(r"[A-Za-z0-9]+", split))


def _is_forbidden_persisted_key(key: str) -> bool:
    """Match common aliases without treating ordinary fields such as author as auth."""
    normalized = _KEY_NORMALIZER.sub("", key.strip().casefold())
    if normalized in _FORBIDDEN_NORMALIZED_KEYS:
        return True
    if "credential" in normalized or "provideroverride" in normalized:
        return True
    tokens = _persisted_key_tokens(key)
    token_set = frozenset(tokens)
    if token_set & {"password", "secret"}:
        return True
    if tokens and tokens[-1] == "bearer":
        return True
    if tokens and tokens[-1] in {"endpoint", "host", "hostname", "origin"}:
        return True
    if "key" in token_set and token_set & {
        "access",
        "api",
        "auth",
        "client",
        "oauth",
        "private",
        "secret",
    }:
        return True
    if normalized.startswith(("authentication", "authorization", "oauth")):
        return True
    return normalized.endswith(
        (
            "apikey",
            "bearer",
            "header",
            "headers",
            "password",
            "path",
            "privatekey",
            "secret",
            "token",
            "uri",
            "url",
        )
    )


@dataclass(frozen=True)
class GatewayPreflightResult:
    """Canonical route identity safe to persist in JSON jobs and presets."""

    backend: str
    model: str
    voice: str
    response_format: str
    allow_fallback: bool
    conversion_required: bool


def gateway_route_provenance(
    *,
    requested_backend: str | None,
    requested_model: str | None,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the fixed, non-secret route metadata allowed in persistence."""
    if not requested_backend:
        return {}
    safe_metadata = metadata if isinstance(metadata, Mapping) else {}
    actual_backend = (
        safe_metadata.get("actual_backend")
        or safe_metadata.get("actual_provider")
        or safe_metadata.get("provider")
        or requested_backend
    )
    actual_model = safe_metadata.get("model") or requested_model
    return {
        "requested_backend": requested_backend,
        "actual_backend": str(actual_backend),
        "requested_model": requested_model,
        "actual_model": str(actual_model) if actual_model is not None else None,
        "fallback_used": bool(safe_metadata.get("fallback_used", False)),
        "conversion_used": bool(safe_metadata.get("conversion_used", False)),
    }


def reject_gateway_persistence_authority(value: Any) -> None:
    """Reject nested credential or endpoint authority fields before persistence."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _is_forbidden_persisted_key(str(key)):
                raise ValueError("gateway persistence payload contains credential or route authority")
            reject_gateway_persistence_authority(child)
    elif isinstance(value, (list, tuple)):
        for item in value:
            reject_gateway_persistence_authority(item)


def preflight_gateway_speech(
    *,
    backend: str,
    model: str,
    voice: str | None,
    voice_supplied: bool,
    response_format: str,
    allow_fallback: bool,
    supplied_fields: frozenset[str] | set[str] | tuple[str, ...],
    gateway_specs: Mapping[str, GatewaySpec] | None = None,
    speed: float | None = None,
    lang_code: str | None = None,
    language: str | None = None,
    target_sample_rate: int | None = None,
    extra_params: Mapping[str, Any] | None = None,
    text_length: int | None = None,
) -> GatewayPreflightResult:
    """Resolve a deterministic gateway route without credentials, discovery, or synthesis."""
    try:
        backend_id = canonicalize_gateway_id(backend)
    except ValueError as exc:
        raise ValueError("invalid TTS gateway backend") from exc
    if gateway_specs is None:
        from .tts_config import get_tts_config_manager

        gateway_specs = get_tts_config_manager().get_gateway_specs()
    spec = gateway_specs.get(backend_id)
    if spec is None:
        raise ValueError("TTS gateway backend is not configured")
    if not spec.enabled:
        raise ValueError("TTS gateway backend is disabled")

    if not isinstance(model, str) or not model.strip():
        raise ValueError("TTS gateway model is required")
    if not spec.allows_model(model):
        raise ValueError("TTS gateway model must be statically configured with exact casing")

    fields = frozenset(str(field) for field in supplied_fields)
    unsupported = fields - _SUPPORTED_FIELDS
    if unsupported:
        raise ValueError("TTS gateway request contains unsupported fields")

    capabilities = spec.capabilities_for_model(model)
    resolved_voice = voice if voice_supplied else spec.default_voice_for_model(model)
    if not isinstance(resolved_voice, str) or not resolved_voice.strip():
        raise ValueError("TTS gateway voice is required")
    overlay = spec.model_overrides.get(model)
    if overlay is not None and overlay.voices and resolved_voice not in overlay.voices:
        raise ValueError("TTS gateway voice is not authorized for this model")

    if "speed" in fields and not capabilities.supports_speed:
        raise ValueError("TTS gateway does not support speed")
    if fields & {"lang_code", "language"} and not capabilities.supports_language:
        raise ValueError("TTS gateway does not support language")
    if "target_sample_rate" in fields and not capabilities.supports_target_sample_rate:
        raise ValueError("TTS gateway does not support target_sample_rate")
    if "lang_code" in fields and "language" in fields and lang_code != language:
        raise ValueError("TTS gateway language fields conflict")
    del speed, target_sample_rate

    reject_gateway_persistence_authority(extra_params or {})
    try:
        validate_gateway_extra_params(extra_params or {}, spec.allowed_request_options)
    except ValueError as exc:
        raise ValueError("TTS gateway extra_params validation failed") from exc
    if text_length is not None and text_length > capabilities.max_input_characters:
        raise ValueError("TTS gateway input exceeds the configured limit")

    try:
        requested_format = AudioFormat(str(response_format).strip().lower())
    except ValueError as exc:
        raise ValueError("TTS gateway response format is unsupported") from exc
    conversion_required = requested_format.value not in capabilities.formats
    if conversion_required:
        conversion = spec.conversion
        executable = spec.ffmpeg_path
        if (
            not conversion.enabled
            or requested_format.value not in conversion.target_formats
            or conversion.source_format not in capabilities.formats
            or not executable
            or not Path(executable).is_file()
            or not os.access(executable, os.X_OK)
        ):
            raise ValueError("TTS gateway response format is unavailable")

    effective_fallback = bool(
        allow_fallback
        and spec.fallback.targets
        and spec.fallback.max_attempts > 1
    )
    return GatewayPreflightResult(
        backend=backend_id,
        model=model,
        voice=resolved_voice,
        response_format=requested_format.value,
        allow_fallback=effective_fallback,
        conversion_required=conversion_required,
    )


__all__ = [
    "GatewayPreflightResult",
    "gateway_route_provenance",
    "preflight_gateway_speech",
    "reject_gateway_persistence_authority",
]
