"""Validated, immutable configuration for OpenAI-compatible TTS gateways."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import os
import re
import shutil
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import unquote, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

_SLUG_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,62}\Z")
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_HEADER_NAME_RE = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+\Z")
_UNSET = object()
_MAX_PATH_DECODE_PASSES = 8
_FALLBACK_CATEGORIES = frozenset(
    {
        "timeout",
        "network_error",
        "upstream_5xx",
        "circuit_open",
        "rate_limited",
        "quota_exceeded",
        "authentication_failed",
        "model_not_found",
        "invalid_audio",
    }
)
_RESERVED_BACKENDS = frozenset(
    {
        "alltalk",
        "chatterbox",
        "dia",
        "echo_tts",
        "elevenlabs",
        "fish_s2",
        "higgs",
        "index_tts",
        "kitten_tts",
        "kokoro",
        "lux_tts",
        "mock",
        "neutts",
        "omnivoice",
        "openai",
        "openrouter",
        "pocket_tts",
        "pocket_tts_cpp",
        "qwen3_tts",
        "supertonic",
        "supertonic2",
        "vibevoice",
        "vibevoice_realtime",
    }
)
_RESERVED_OPTION_TOKENS = frozenset(
    {
        "api-key",
        "api_key",
        "access_token",
        "auth",
        "authorization",
        "base-url",
        "base_url",
        "bearer",
        "credential",
        "credentials",
        "header",
        "headers",
        "input",
        "lang_code",
        "language",
        "model",
        "models_path",
        "password",
        "path",
        "response_format",
        "secret",
        "speech_path",
        "speed",
        "target_sample_rate",
        "token",
        "url",
        "voice",
    }
)

_MAX_EXTRA_PARAM_DEPTH = 8
_MAX_EXTRA_PARAM_SCALAR_LEAVES = 64
_MAX_EXTRA_PARAM_STRING_LENGTH = 4096
_MAX_EXTRA_PARAM_SERIALIZED_BYTES = 65536


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        hide_input_in_errors=True,
    )


class GatewayPCMCapabilities(_FrozenModel):
    """Raw PCM framing advertised by a gateway."""

    sample_rate: int = Field(default=24000, gt=0)
    channels: int = Field(default=1, gt=0, le=8)
    sample_width_bits: int = Field(default=16, gt=0)


class GatewayCapabilities(_FrozenModel):
    """Server-controlled request and output capabilities."""

    formats: tuple[str, ...] = ("mp3",)
    supports_speed: bool = False
    supports_language: bool = False
    supports_target_sample_rate: bool = False
    allow_octet_stream: bool = False
    max_input_characters: int = Field(default=12000, gt=0)
    max_response_bytes: int = Field(default=26214400, gt=0)
    pcm: GatewayPCMCapabilities = Field(default_factory=GatewayPCMCapabilities)

    @field_validator("formats")
    @classmethod
    def validate_formats(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(str(item).strip().lower() for item in value)
        if not normalized or any(not item for item in normalized):
            raise ValueError("capability formats cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("capability formats cannot contain duplicates")
        return normalized


class ModelOverlay(_FrozenModel):
    """Per-model fields applied after gateway capability defaults."""

    default_voice: str | None = None
    voices: tuple[str, ...] = ()
    formats: tuple[str, ...] | None = None
    supports_speed: bool | None = None
    supports_language: bool | None = None
    supports_target_sample_rate: bool | None = None
    allow_octet_stream: bool | None = None
    max_input_characters: int | None = Field(default=None, gt=0)
    max_response_bytes: int | None = Field(default=None, gt=0)
    pcm: GatewayPCMCapabilities | None = None

    def apply(self, defaults: GatewayCapabilities) -> GatewayCapabilities:
        changes = {
            field_name: value
            for field_name in GatewayCapabilities.model_fields
            if (value := getattr(self, field_name)) is not None
        }
        return defaults.model_copy(update=changes)


class GatewayDiscoveryPolicy(_FrozenModel):
    """Local discovery settings; normalization never performs discovery."""

    enabled: bool = False
    models_path: str | None = None
    query: tuple[tuple[str, str | int | float | bool], ...] = ()
    ttl_seconds: int = Field(default=600, ge=0)
    stale_ttl_seconds: int = Field(default=3600, ge=0)
    timeout_seconds: float = Field(default=5.0, gt=0)

    @field_validator("query", mode="before")
    @classmethod
    def freeze_query(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return tuple(value.items())
        return value


class GatewayConversionPolicy(_FrozenModel):
    """Bounds for later full-buffer audio conversion."""

    enabled: bool = False
    source_format: str = "mp3"
    target_formats: tuple[str, ...] = ()
    max_input_bytes: int = Field(default=26214400, gt=0)
    max_output_bytes: int = Field(default=52428800, gt=0)
    timeout_seconds: float = Field(default=30.0, gt=0)

    @field_validator("source_format")
    @classmethod
    def normalize_source_format(cls, value: str) -> str:
        value = value.strip().lower()
        if not value:
            raise ValueError("conversion source_format cannot be empty")
        return value

    @field_validator("target_formats")
    @classmethod
    def normalize_target_formats(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(str(item).strip().lower() for item in value)
        if any(not item for item in normalized):
            raise ValueError("conversion target formats cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("conversion target formats cannot contain duplicates")
        return normalized


class GatewayFallbackTarget(_FrozenModel):
    """One server-configured fallback route."""

    backend: str
    model: str
    voice: str | None = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("fallback target model cannot be blank")
        return value

    @field_validator("voice")
    @classmethod
    def validate_voice(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("fallback target voice cannot be blank")
        return value


class GatewayFallbackPolicy(_FrozenModel):
    """Bounded fallback policy for a gateway."""

    on: tuple[str, ...] = ()
    max_attempts: int = Field(default=1, ge=1, le=4)
    targets: tuple[GatewayFallbackTarget, ...] = Field(default=(), max_length=3)

    @field_validator("on")
    @classmethod
    def validate_categories(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not category.strip() for category in value):
            raise ValueError("fallback categories cannot be blank")
        unknown = set(value) - _FALLBACK_CATEGORIES
        if unknown:
            raise ValueError("fallback contains an unknown category")
        if len(set(value)) != len(value):
            raise ValueError("fallback categories cannot contain duplicates")
        return value


class GatewayConfig(_FrozenModel):
    """Administrator input for a built-in or named speech gateway."""

    enabled: bool = False
    display_name: str | None = None
    base_url: str | None = None
    allow_insecure_http: bool = False
    speech_path: str | None = None
    models_path: str | None = None
    headers: tuple[tuple[str, str], ...] = ()
    api_key: str | None = Field(default=None, repr=False)
    allow_user_api_key: bool = False
    speech_timeout_seconds: float = Field(default=30.0, gt=0)
    default_model: str | None = None
    default_voice: str | None = None
    allowed_models: tuple[str, ...] | None = None
    allow_discovered_models: bool = False
    model_overrides: dict[str, ModelOverlay] = Field(default_factory=dict)
    capability_defaults: GatewayCapabilities = Field(default_factory=GatewayCapabilities)
    allowed_request_options: tuple[str, ...] = ()
    fallback: GatewayFallbackPolicy = Field(default_factory=GatewayFallbackPolicy)
    discovery: GatewayDiscoveryPolicy = Field(default_factory=GatewayDiscoveryPolicy)
    conversion: GatewayConversionPolicy = Field(default_factory=GatewayConversionPolicy)

    @field_validator("headers", mode="before")
    @classmethod
    def freeze_headers(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return tuple(value.items())
        return value


@dataclass(frozen=True)
class GatewaySpec:
    """Effective immutable server-controlled configuration for one backend."""

    backend_id: str
    display_name: str
    enabled: bool
    base_url: str
    speech_path: str
    models_path: str | None
    discovery_query: tuple[tuple[str, str], ...]
    headers: tuple[tuple[str, str], ...]
    api_key: str | None = dataclass_field(repr=False)
    allow_user_api_key: bool
    speech_timeout_seconds: float
    default_model: str | None
    default_voice: str | None
    allowed_models: frozenset[str]
    allowed_models_configured: bool
    allow_discovered_models: bool
    model_overrides: Mapping[str, ModelOverlay]
    capability_defaults: GatewayCapabilities
    allowed_request_options: frozenset[str]
    fallback: GatewayFallbackPolicy
    discovery: GatewayDiscoveryPolicy
    conversion: GatewayConversionPolicy
    ffmpeg_path: str | None
    config_generation: str

    def allows_model(
        self,
        model: str,
        discovered_models: set[str] | frozenset[str] = frozenset(),
    ) -> bool:
        """Return whether an exact-cased model ID is authorized."""
        if self.allowed_models_configured:
            return model in self.allowed_models
        configured = {self.default_model, *self.model_overrides.keys()} - {None}
        if model in configured:
            return True
        return self.allow_discovered_models and model in discovered_models

    def capabilities_for_model(self, model: str | None) -> GatewayCapabilities:
        """Apply the configured per-model overlay after gateway defaults."""
        overlay = self.model_overrides.get(model) if model is not None else None
        return overlay.apply(self.capability_defaults) if overlay else self.capability_defaults

    def default_voice_for_model(self, model: str | None) -> str | None:
        """Resolve a model-specific voice without leaking the gateway default."""
        overlay = self.model_overrides.get(model) if model is not None else None
        if overlay and overlay.default_voice:
            return overlay.default_voice
        return self.default_voice if model == self.default_model else None


def canonicalize_gateway_id(value: str, *, builtin: bool = False) -> str:
    """Return the canonical backend ID for a built-in ID or custom slug."""
    if not isinstance(value, str):
        raise ValueError("gateway slug must be a string")
    raw = value
    if raw == "openrouter":
        return "openrouter"
    if raw.startswith("gateway:"):
        slug = raw.removeprefix("gateway:")
    else:
        slug = raw
    if not _SLUG_RE.fullmatch(slug):
        raise ValueError("gateway slug must match [a-z0-9][a-z0-9-]{0,62}")
    if builtin:
        raise ValueError(f"unknown built-in gateway slug: {slug}")
    return f"gateway:{slug}"


def validate_relative_gateway_path(value: str, *, field_name: str) -> str:
    """Validate an administrator path without permitting authority replacement."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty relative path")
    candidate = value
    for _ in range(_MAX_PATH_DECODE_PASSES):
        if candidate.startswith("/") or "\\" in candidate:
            raise ValueError(f"{field_name} must be a strict relative path")
        parsed = httpx.URL(candidate)
        if parsed.scheme or parsed.host or parsed.query or parsed.fragment:
            raise ValueError(
                f"{field_name} must not contain scheme, authority, query, or fragment"
            )
        if any(segment in {".", "..", ""} for segment in candidate.split("/")):
            raise ValueError(f"{field_name} must not contain empty or dot segments")
        decoded = unquote(candidate)
        if decoded == candidate:
            return value
        candidate = decoded
    raise ValueError(f"{field_name} contains too many encoding layers")


def build_gateway_url(base_url: str, relative_path: str) -> httpx.URL:
    """Join a previously validated relative path without replacing authority."""
    path = validate_relative_gateway_path(relative_path, field_name="gateway path")
    base = httpx.URL(base_url)
    if not base.is_absolute_url:
        raise ValueError("base_url must be absolute")
    base_path = base.path.rstrip("/")
    return base.copy_with(path=f"{base_path}/{path}")


def decode_json_pointer(pointer: str) -> tuple[str, ...]:
    """Decode a strict RFC 6901 JSON Pointer into path tokens."""
    if not isinstance(pointer, str):
        raise ValueError("JSON Pointer must be a string")
    if not pointer:
        return ()
    if not pointer.startswith("/"):
        raise ValueError("JSON Pointer must start with '/'")
    raw_tokens = pointer[1:].split("/")
    decoded: list[str] = []
    for token in raw_tokens:
        index = 0
        chars: list[str] = []
        while index < len(token):
            char = token[index]
            if char != "~":
                chars.append(char)
                index += 1
                continue
            if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
                raise ValueError("JSON Pointer contains an invalid escape")
            chars.append("~" if token[index + 1] == "0" else "/")
            index += 2
        decoded.append("".join(chars))
    return tuple(decoded)


def _pointer_value(document: Mapping[str, Any], tokens: tuple[str, ...]) -> Any:
    current: Any = document
    for token in tokens:
        if not isinstance(current, Mapping) or token not in current:
            return _UNSET
        current = current[token]
    return current


def validate_gateway_extra_params(
    extra_params: Mapping[str, Any],
    allowed_pointers: frozenset[str] | set[str] | tuple[str, ...],
) -> None:
    """Validate bounded JSON options and require exact RFC 6901 leaf matches."""
    if not isinstance(extra_params, Mapping):
        raise ValueError("gateway extra_params must be a JSON object")

    document = dict(extra_params)
    authorization_leaves: set[tuple[str, ...]] = set()
    scalar_leaves = 0

    def walk(value: Any, path: tuple[str, ...], depth: int, *, authorize: bool) -> None:
        nonlocal scalar_leaves
        if depth > _MAX_EXTRA_PARAM_DEPTH:
            raise ValueError("gateway extra_params exceeds maximum depth 8")
        if isinstance(value, Mapping):
            if not value and path and authorize:
                raise ValueError("gateway extra_params contains an empty container leaf")
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError("gateway extra_params object keys must be strings")
                if len(key) > _MAX_EXTRA_PARAM_STRING_LENGTH:
                    raise ValueError("gateway extra_params key exceeds 4096 characters")
                if key.casefold() in _RESERVED_OPTION_TOKENS:
                    raise ValueError("gateway extra_params contains a reserved field")
                walk(child, (*path, key), depth + 1, authorize=authorize)
            return
        if isinstance(value, list):
            if authorize:
                authorization_leaves.add(path)
            for index, child in enumerate(value):
                walk(child, (*path, str(index)), depth + 1, authorize=False)
            return
        if isinstance(value, str):
            if len(value) > _MAX_EXTRA_PARAM_STRING_LENGTH:
                raise ValueError("gateway extra_params string exceeds 4096 characters")
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("gateway extra_params numbers must be finite JSON values")
        elif not isinstance(value, (int, bool)) and value is not None:
            raise ValueError("gateway extra_params contains a non-JSON value")
        scalar_leaves += 1
        if scalar_leaves > _MAX_EXTRA_PARAM_SCALAR_LEAVES:
            raise ValueError("gateway extra_params exceeds 64 scalar leaves")
        if authorize:
            authorization_leaves.add(path)

    walk(document, (), 0, authorize=True)
    try:
        serialized = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("gateway extra_params must contain JSON values") from exc
    if len(serialized) > _MAX_EXTRA_PARAM_SERIALIZED_BYTES:
        raise ValueError("gateway extra_params exceeds 65536 serialized bytes")

    decoded_pointers: set[tuple[str, ...]] = set()
    for pointer in allowed_pointers:
        tokens = decode_json_pointer(pointer)
        if not tokens:
            raise ValueError("gateway extra_params cannot allow the whole document")
        if any(token.casefold() in _RESERVED_OPTION_TOKENS for token in tokens):
            raise ValueError("gateway extra_params allowlist contains a reserved field")
        configured_value = _pointer_value(document, tokens)
        if isinstance(configured_value, Mapping):
            raise ValueError("gateway extra_params pointer identifies only a container")
        decoded_pointers.add(tokens)

    unknown = authorization_leaves - decoded_pointers
    if unknown:
        raise ValueError("gateway extra_params contains a leaf not in the allowlist")


def copy_gateway_extra_params(
    extra_params: Mapping[str, Any],
    allowed_pointers: frozenset[str] | set[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Return a detached copy after validating every supplied option leaf."""
    validate_gateway_extra_params(extra_params, allowed_pointers)
    return deepcopy(dict(extra_params))


def _validate_request_options(pointers: tuple[str, ...]) -> frozenset[str]:
    if len(set(pointers)) != len(pointers):
        raise ValueError("allowed_request_options cannot contain duplicates")
    for pointer in pointers:
        tokens = decode_json_pointer(pointer)
        if not tokens:
            raise ValueError(
                "allowed_request_options cannot authorize the whole document"
            )
        if any(token.casefold() in _RESERVED_OPTION_TOKENS for token in tokens):
            raise ValueError("allowed_request_options points to a reserved field")
    return frozenset(pointers)


def _validate_headers(headers: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    seen: set[str] = set()
    normalized: list[tuple[str, str]] = []
    for name, value in headers:
        folded = name.casefold()
        if not _HEADER_NAME_RE.fullmatch(name) or folded == "authorization":
            raise ValueError("gateway headers contain a forbidden header name")
        if folded in seen:
            raise ValueError("gateway headers contain duplicate names")
        if "\r" in value or "\n" in value:
            raise ValueError("gateway headers contain an invalid value")
        seen.add(folded)
        normalized.append((name, value))
    return tuple(sorted(normalized))


def _validate_base_url(value: str, *, allow_insecure_http: bool) -> str:
    try:
        url = httpx.URL(value)
    except Exception as exc:
        raise ValueError("base_url must be a valid absolute URL") from exc
    raw_components = urlsplit(value)
    if not url.is_absolute_url or not url.host:
        raise ValueError("base_url must be an absolute URL")
    if (
        "@" in raw_components.netloc
        or "?" in value
        or "#" in value
        or url.userinfo
        or url.query
        or url.fragment
    ):
        raise ValueError("base_url cannot contain credentials, query, or fragment")
    if url.scheme not in {"https", "http"}:
        raise ValueError("base_url must use HTTPS")
    if url.scheme == "http":
        if not allow_insecure_http:
            raise ValueError("HTTP base_url requires allow_insecure_http=true")
        host = url.host.rstrip(".").casefold()
        if host != "localhost":
            try:
                address = ipaddress.ip_address(host)
            except ValueError as exc:
                raise ValueError("HTTP base_url host must be localhost or a private IP literal") from exc
            if not (
                address.is_loopback
                or address.is_private
                or address.is_link_local
            ):
                raise ValueError("public HTTP base_url is forbidden")
    normalized_path = url.path.rstrip("/") + "/"
    return str(url.copy_with(path=normalized_path))


def _resolve_environment(
    value: Any,
    *,
    path: str,
    missing_paths: set[str],
) -> Any:
    if isinstance(value, dict):
        resolved_mapping: dict[str, Any] = {}
        for key, item in value.items():
            resolved_item = _resolve_environment(
                item,
                path=f"{path}.{key}",
                missing_paths=missing_paths,
            )
            if resolved_item is not _UNSET:
                resolved_mapping[key] = resolved_item
        return resolved_mapping
    if isinstance(value, (list, tuple)):
        resolved_items: list[Any] = []
        for index, item in enumerate(value):
            resolved_item = _resolve_environment(
                item,
                path=f"{path}.{index}",
                missing_paths=missing_paths,
            )
            if resolved_item is not _UNSET:
                resolved_items.append(resolved_item)
        return resolved_items
    if not isinstance(value, str) or "${" not in value:
        return value

    missing = False

    def replace(match: re.Match[str]) -> str:
        nonlocal missing
        resolved = os.getenv(match.group(1))
        if resolved is None:
            missing = True
            return ""
        return resolved

    resolved = _ENV_RE.sub(replace, value)
    if "${" in resolved:
        missing = True
    if missing:
        missing_paths.add(path)
        return _UNSET
    return resolved


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        dumped = value.model_dump()
        if isinstance(value, GatewayConfig):
            dumped["headers"] = dict(dumped.get("headers") or ())
            discovery = dumped.get("discovery")
            if isinstance(discovery, dict):
                discovery["query"] = dict(discovery.get("query") or ())
        return dumped
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError("gateway definition must be a mapping")


def _openrouter_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    result = dict(raw)
    if result.get("base_url") is None:
        result["base_url"] = "https://openrouter.ai/api/v1/"
    if result.get("speech_path") is None:
        result["speech_path"] = "audio/speech"
    discovery = dict(result.get("discovery") or {})
    discovery.setdefault("enabled", True)
    discovery.setdefault("models_path", "models")
    discovery.setdefault("query", {"output_modalities": "speech"})
    result["discovery"] = discovery
    headers = dict(result.get("headers") or {})
    site_url = os.getenv("OPENROUTER_SITE_URL")
    site_name = os.getenv("OPENROUTER_SITE_NAME")
    if site_url:
        headers.setdefault("HTTP-Referer", site_url)
    if site_name:
        headers.setdefault("X-Title", site_name)
    result["headers"] = headers
    if result.get("display_name") is None:
        result["display_name"] = "OpenRouter"
    return result


def materialize_gateway_config(
    value: Any,
    *,
    path: str,
    openrouter: bool = False,
) -> GatewayConfig:
    """Resolve gateway environment values before nested Pydantic validation."""
    raw = _as_mapping(value)
    if openrouter:
        raw = _openrouter_defaults(raw)
    missing_paths: set[str] = set()
    resolved = _resolve_environment(raw, path=path, missing_paths=missing_paths)
    config = GatewayConfig.model_validate(resolved)

    if config.allowed_models is not None and config.allow_discovered_models:
        raise ValueError(
            f"{path}: allowed_models and allow_discovered_models cannot both be set"
        )
    if not config.enabled:
        return config

    default_overlay = config.model_overrides.get(config.default_model or "")
    effective_overlay_voice = (
        default_overlay.default_voice if default_overlay else None
    )
    blank_fields = [
        name
        for name, value in {
            "default_model": config.default_model,
            "default_voice": config.default_voice,
            "api_key": config.api_key,
            "model_overrides default_voice": effective_overlay_voice,
        }.items()
        if value is not None and not value.strip()
    ]
    if blank_fields:
        raise ValueError(f"{path}: enabled gateway has blank {', '.join(blank_fields)}")
    effective_default_voice = config.default_voice or effective_overlay_voice
    required_fields = {
        "base_url": config.base_url,
        "speech_path": config.speech_path,
        "default_model": config.default_model,
        "default_voice": effective_default_voice,
    }
    missing = [name for name, field_value in required_fields.items() if not field_value]
    if missing:
        placeholder_path = next(
            (
                f"{path}.{name}"
                for name in missing
                if f"{path}.{name}" in missing_paths
            ),
            None,
        )
        if placeholder_path:
            raise ValueError(
                f"unresolved environment placeholder at {placeholder_path}"
            )
        raise ValueError(f"{path}: enabled gateway requires {', '.join(missing)}")
    if not config.api_key and not config.allow_user_api_key:
        if f"{path}.api_key" in missing_paths:
            raise ValueError(f"unresolved environment placeholder at {path}.api_key")
        raise ValueError(f"{path}: enabled gateway requires a credential source")
    return config


def _config_generation(spec_data: dict[str, Any]) -> str:
    canonical = json.dumps(
        spec_data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _available_executable_path(value: str | None) -> str | None:
    """Return one absolute executable identity, or None when unavailable."""
    if not value:
        return None
    try:
        path = Path(value).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            return None
    except OSError:
        return None
    return str(path)


def _normalize_one(
    backend_id: str,
    raw: dict[str, Any],
    *,
    path: str,
    ffmpeg_path: str | None,
) -> GatewaySpec:
    config = materialize_gateway_config(raw, path=path)

    base_url = _validate_base_url(
        config.base_url or "https://disabled.invalid/",
        allow_insecure_http=config.allow_insecure_http,
    )
    speech_path = validate_relative_gateway_path(
        config.speech_path or "audio/speech",
        field_name="speech_path",
    )
    top_models_path = config.models_path
    discovery_models_path = config.discovery.models_path
    if top_models_path is not None:
        top_models_path = validate_relative_gateway_path(
            top_models_path,
            field_name="models_path",
        )
    if discovery_models_path is not None:
        discovery_models_path = validate_relative_gateway_path(
            discovery_models_path,
            field_name="discovery.models_path",
        )
    if (
        top_models_path is not None
        and discovery_models_path is not None
        and top_models_path != discovery_models_path
    ):
        raise ValueError(f"{path}: conflicting models_path definitions")
    models_path = top_models_path or discovery_models_path

    allowed_models_configured = config.allowed_models is not None
    allowed_models = frozenset(config.allowed_models or ())
    if config.allowed_models is not None:
        if len(allowed_models) != len(config.allowed_models):
            raise ValueError(f"{path}.allowed_models contains duplicates")
        configured_models = {config.default_model, *config.model_overrides.keys()} - {None}
        disallowed = configured_models - allowed_models
        if disallowed:
            raise ValueError(f"{path}.allowed_models omits configured models")

    immutable_overrides = MappingProxyType(dict(config.model_overrides))
    request_options = _validate_request_options(config.allowed_request_options)
    conversion = config.conversion
    if conversion.enabled and ffmpeg_path is None:
        conversion = conversion.model_copy(update={"target_formats": ()})

    discovery_query = tuple(
        sorted((str(key), str(value)) for key, value in config.discovery.query)
    )
    discovery = config.discovery.model_copy(
        update={"models_path": models_path, "query": discovery_query}
    )
    headers = _validate_headers(config.headers)
    fallback = config.fallback.model_copy(
        update={
            "targets": tuple(
                target.model_copy(
                    update={"backend": canonicalize_gateway_id(target.backend)}
                )
                for target in config.fallback.targets
            )
        }
    )
    fallback_output = fallback.model_dump(mode="json")
    fallback_output["on"] = sorted(fallback.on)
    output_fields = {
        "backend_id": backend_id,
        "display_name": config.display_name or backend_id,
        "enabled": config.enabled,
        "base_url": base_url,
        "speech_path": speech_path,
        "models_path": models_path,
        "discovery_query": discovery_query,
        "headers": headers,
        "allow_user_api_key": config.allow_user_api_key,
        "speech_timeout_seconds": config.speech_timeout_seconds,
        "default_model": config.default_model,
        "default_voice": config.default_voice,
        "allowed_models": sorted(allowed_models),
        "allowed_models_configured": allowed_models_configured,
        "allow_discovered_models": config.allow_discovered_models,
        "model_overrides": {
            key: value.model_dump(mode="json")
            for key, value in sorted(config.model_overrides.items())
        },
        "capability_defaults": config.capability_defaults.model_dump(mode="json"),
        "allowed_request_options": sorted(request_options),
        "fallback": fallback_output,
        "discovery": discovery.model_dump(mode="json"),
        "conversion": conversion.model_dump(mode="json"),
        "ffmpeg_path": ffmpeg_path,
    }
    return GatewaySpec(
        backend_id=backend_id,
        display_name=config.display_name or backend_id,
        enabled=config.enabled,
        base_url=base_url,
        speech_path=speech_path,
        models_path=models_path,
        discovery_query=discovery_query,
        headers=headers,
        api_key=config.api_key,
        allow_user_api_key=config.allow_user_api_key,
        speech_timeout_seconds=config.speech_timeout_seconds,
        default_model=config.default_model,
        default_voice=config.default_voice,
        allowed_models=allowed_models,
        allowed_models_configured=allowed_models_configured,
        allow_discovered_models=config.allow_discovered_models,
        model_overrides=immutable_overrides,
        capability_defaults=config.capability_defaults,
        allowed_request_options=request_options,
        fallback=fallback,
        discovery=discovery,
        conversion=conversion,
        ffmpeg_path=ffmpeg_path,
        config_generation=_config_generation(output_fields),
    )


def _validate_fallback_graph(specs: Mapping[str, GatewaySpec]) -> None:
    graph: dict[str, tuple[str, ...]] = {}
    for backend_id, spec in specs.items():
        targets: list[str] = []
        for target in spec.fallback.targets:
            canonical = canonicalize_gateway_id(target.backend)
            if canonical == backend_id:
                raise ValueError(f"fallback for {backend_id} cannot target itself")
            if canonical not in specs:
                raise ValueError(f"fallback for {backend_id} has unknown target")
            if canonical in targets:
                raise ValueError(f"fallback for {backend_id} has duplicate target")
            if target.voice is None and specs[canonical].default_voice_for_model(
                target.model
            ) is None:
                raise ValueError(
                    f"fallback for {backend_id} target {canonical} model "
                    f"{target.model!r} has no configured default voice"
                )
            targets.append(canonical)
        graph[backend_id] = tuple(targets)

    state: dict[str, int] = {}
    for backend_id in graph:
        if state.get(backend_id) == 2:
            continue
        state[backend_id] = 1
        stack = [(backend_id, iter(graph[backend_id]))]
        while stack:
            node, targets = stack[-1]
            try:
                target = next(targets)
            except StopIteration:
                state[node] = 2
                stack.pop()
                continue
            if state.get(target) == 1:
                raise ValueError("fallback graph contains a cycle")
            if state.get(target, 0) == 0:
                state[target] = 1
                stack.append((target, iter(graph[target])))


def normalize_gateway_specs(
    providers: Mapping[str, Any],
    gateways: Mapping[str, Any],
    *,
    ffmpeg_available: bool | None = None,
    ffmpeg_path: str | None = None,
) -> Mapping[str, GatewaySpec]:
    """Validate all definitions locally and return immutable normalized specs."""
    ffmpeg_candidate = ffmpeg_path if ffmpeg_path is not None else shutil.which("ffmpeg")
    if ffmpeg_available is False:
        ffmpeg_candidate = None
    effective_ffmpeg_path = _available_executable_path(ffmpeg_candidate)
    raw_definitions: list[tuple[str, dict[str, Any], str]] = []
    openrouter = providers.get("openrouter")
    if openrouter is not None:
        raw_definitions.append(
            ("openrouter", _openrouter_defaults(_as_mapping(openrouter)), "providers.openrouter")
        )

    folded: dict[str, str] = {}
    for slug, definition in gateways.items():
        if not isinstance(slug, str):
            raise ValueError("gateway slug must be a string")
        folded_slug = slug.casefold()
        if folded_slug in folded:
            raise ValueError("gateway slug collision after case normalization")
        folded[folded_slug] = slug
        canonical = canonicalize_gateway_id(slug)
        if slug.casefold() in _RESERVED_BACKENDS or canonical.removeprefix("gateway:") in _RESERVED_BACKENDS:
            raise ValueError(f"gateway slug {slug!r} is reserved")
        raw_definitions.append((canonical, _as_mapping(definition), f"gateways.{slug}"))

    specs: dict[str, GatewaySpec] = {}
    for backend_id, definition, path in raw_definitions:
        if backend_id in specs:
            raise ValueError(f"duplicate gateway ID: {backend_id}")
        specs[backend_id] = _normalize_one(
            backend_id,
            definition,
            path=path,
            ffmpeg_path=effective_ffmpeg_path,
        )
    _validate_fallback_graph(specs)
    return MappingProxyType(specs)


__all__ = [
    "GatewayCapabilities",
    "GatewayConfig",
    "GatewayConversionPolicy",
    "GatewayDiscoveryPolicy",
    "GatewayFallbackPolicy",
    "GatewayFallbackTarget",
    "GatewayPCMCapabilities",
    "GatewaySpec",
    "ModelOverlay",
    "build_gateway_url",
    "canonicalize_gateway_id",
    "decode_json_pointer",
    "materialize_gateway_config",
    "normalize_gateway_specs",
    "validate_relative_gateway_path",
]
