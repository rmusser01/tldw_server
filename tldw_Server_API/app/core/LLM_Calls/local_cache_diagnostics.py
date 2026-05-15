"""Diagnostic helpers for local inference prefix and prompt caches.

Local vLLM and llama.cpp cache signals are runtime and latency hints, not
provider billing evidence. This module reports sanitized request and runtime
metadata without exposing prompt text or local file paths.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


LOCAL_CACHE_DIAGNOSTIC_PROVIDERS: frozenset[str] = frozenset({"vllm", "llama.cpp"})
_MAX_FINGERPRINT_METADATA_CHARS = 96
_MAX_PROVIDER_HINT_KEYS = 16
_LLAMACPP_PROMPT_CACHE_KEYS = (
    "cache_prompt",
    "cache_reuse",
    "prompt_cache",
    "prompt_cache_all",
    "prompt_cache_ro",
)
_PATH_LIKE_LLAMA_KEYS = frozenset({"prompt_cache"})


@dataclass(frozen=True)
class InferencePrefixCacheIntent:
    """Local inference prefix-cache intent, separate from provider billing cache controls."""

    enabled: bool = False
    scope: tuple[str, ...] = ()
    static_segment_fingerprint: str | None = None
    provider_hint_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class LocalCacheDiagnostic:
    """Bounded metadata for local runtime/prefix-cache compatibility checks."""

    provider: str
    prefix_cache_intent_requested: bool
    scope: tuple[str, ...] = ()
    static_segment_fingerprint: str | None = None
    provider_hint_keys: tuple[str, ...] = ()
    request_shape_stable: bool | None = None
    warnings: tuple[str, ...] = ()
    runtime_cache_mode: str | None = None
    runtime_flags: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    request_extension_keys: tuple[str, ...] = ()
    billing_cache_authoritative: bool = False
    billing_cache_savings_reported: bool = False

    @property
    def has_signal(self) -> bool:
        """Return True when this diagnostic contains user/runtime cache information."""
        return bool(
            self.prefix_cache_intent_requested
            or self.runtime_flags
            or self.request_extension_keys
            or self.warnings
        )

    def to_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "provider": self.provider,
            "prefix_cache_intent_requested": self.prefix_cache_intent_requested,
            "billing_cache_authoritative": self.billing_cache_authoritative,
            "billing_cache_savings_reported": self.billing_cache_savings_reported,
        }
        if self.scope:
            metadata["scope"] = list(self.scope)
        if self.static_segment_fingerprint:
            metadata["static_segment_fingerprint"] = self.static_segment_fingerprint
        if self.provider_hint_keys:
            metadata["provider_hint_keys"] = list(self.provider_hint_keys)
        if self.request_shape_stable is not None:
            metadata["request_shape_stable"] = self.request_shape_stable
        if self.warnings:
            metadata["warnings"] = list(self.warnings)
        if self.runtime_cache_mode:
            metadata["runtime_cache_mode"] = self.runtime_cache_mode
        if self.runtime_flags:
            metadata["runtime_flags"] = {key: dict(value) for key, value in self.runtime_flags.items()}
        if self.request_extension_keys:
            metadata["request_extension_keys"] = list(self.request_extension_keys)
        return metadata


def _provider_key(provider: str | None) -> str:
    return str(provider or "").strip().lower()


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _bound_text(value: Any, limit: int = _MAX_FINGERPRINT_METADATA_CHARS) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:limit]


def _normalize_scope(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = value
    else:
        return ()

    normalized: list[str] = []
    for item in items:
        text = str(item or "").strip().lower()
        if text and text not in normalized:
            normalized.append(text[:64])
    return tuple(normalized)


def _provider_hint_keys(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Mapping):
        return ()
    keys: list[str] = []
    for raw_key in value.keys():
        if not isinstance(raw_key, str):
            continue
        key = raw_key.strip()
        if key and key not in keys:
            keys.append(key[:80])
        if len(keys) >= _MAX_PROVIDER_HINT_KEYS:
            break
    return tuple(sorted(keys))


def parse_inference_prefix_cache_intent(value: Any) -> InferencePrefixCacheIntent:
    """Parse local inference prefix-cache intent, defaulting to disabled."""
    if isinstance(value, InferencePrefixCacheIntent):
        return value
    if not isinstance(value, Mapping):
        return InferencePrefixCacheIntent()
    return InferencePrefixCacheIntent(
        enabled=_truthy(value.get("enabled")),
        scope=_normalize_scope(value.get("scope")),
        static_segment_fingerprint=_bound_text(value.get("static_segment_fingerprint")),
        provider_hint_keys=_provider_hint_keys(value.get("provider_hint")),
    )


def _request_intent(request: Mapping[str, Any]) -> InferencePrefixCacheIntent:
    return parse_inference_prefix_cache_intent(
        request.get("inference_prefix_cache_intent")
        if request.get("inference_prefix_cache_intent") is not None
        else request.get("local_prefix_cache_intent")
    )


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _vllm_shape_warnings(payload: Mapping[str, Any], intent: InferencePrefixCacheIntent) -> tuple[bool, tuple[str, ...]]:
    warnings: list[str] = []
    shape_unstable = False

    if intent.enabled and not intent.static_segment_fingerprint:
        warnings.append("missing_static_segment_fingerprint")
        shape_unstable = True

    choice_count = _int_or_none(payload.get("n"))
    if choice_count is not None and choice_count != 1:
        shape_unstable = True

    temperature = _float_or_none(payload.get("temperature"))
    if temperature is not None and temperature > 0 and payload.get("seed") is None:
        shape_unstable = True

    if shape_unstable:
        warnings.append("request_shape_unstable")

    return not shape_unstable, tuple(dict.fromkeys(warnings))


def _llama_settings(app_config: Mapping[str, Any] | None, runtime_context: Mapping[str, Any] | None) -> Mapping[str, Any]:
    config = _as_mapping(app_config)
    context = _as_mapping(runtime_context)
    merged: dict[str, Any] = {}
    for section in ("llama_api", "LlamaCpp", "llamacpp", "llama_cpp"):
        merged.update(_as_mapping(config.get(section)))
    merged.update(context)
    return merged


def resolve_llamacpp_prompt_cache_flags(
    *,
    app_config: Mapping[str, Any] | None = None,
    runtime_context: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return sanitized llama.cpp prompt-cache runtime flags without raw paths."""
    settings = _llama_settings(app_config, runtime_context)
    flags: dict[str, dict[str, Any]] = {}
    for key in _LLAMACPP_PROMPT_CACHE_KEYS:
        if key not in settings:
            continue
        value = settings.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        entry: dict[str, Any] = {"configured": True}
        if key in _PATH_LIKE_LLAMA_KEYS:
            pass
        elif key == "cache_reuse":
            tokens = _int_or_none(value)
            if tokens is not None:
                entry["tokens"] = max(0, tokens)
            else:
                entry["enabled"] = _truthy(value)
        else:
            entry["enabled"] = _truthy(value)
        flags[key] = entry
    return flags


def _llamacpp_runtime_mode(flags: Mapping[str, Mapping[str, Any]]) -> str:
    prompt_cache_ro = _as_mapping(flags.get("prompt_cache_ro"))
    prompt_cache_all = _as_mapping(flags.get("prompt_cache_all"))
    cache_prompt = _as_mapping(flags.get("cache_prompt"))
    cache_reuse = _as_mapping(flags.get("cache_reuse"))

    if prompt_cache_ro.get("enabled"):
        return "read_only"
    if flags.get("prompt_cache") or prompt_cache_all.get("enabled"):
        return "writable"
    cache_reuse_enabled = bool(cache_reuse.get("enabled"))
    cache_reuse_tokens = _int_or_none(cache_reuse.get("tokens"))
    if (
        cache_prompt.get("enabled")
        or cache_reuse_enabled
        or (cache_reuse_tokens is not None and cache_reuse_tokens > 0)
    ):
        return "request_reuse"
    return "disabled_or_unknown"


def _request_extension_keys(request: Mapping[str, Any]) -> tuple[str, ...]:
    extra_body = _as_mapping(request.get("extra_body"))
    keys = [key for key in _LLAMACPP_PROMPT_CACHE_KEYS if key in extra_body and extra_body.get(key) is not None]
    return tuple(keys)


def build_local_cache_diagnostic(
    *,
    provider: str,
    request: Mapping[str, Any],
    payload: Mapping[str, Any],
    app_config: Mapping[str, Any] | None = None,
    runtime_context: Mapping[str, Any] | None = None,
) -> LocalCacheDiagnostic:
    """Build cost-neutral local cache diagnostics for vLLM and llama.cpp requests."""
    provider_key = _provider_key(provider)
    intent = _request_intent(request)
    shape_stable: bool | None = None
    warnings: tuple[str, ...] = ()
    runtime_flags: Mapping[str, Mapping[str, Any]] = {}
    runtime_cache_mode: str | None = None
    request_extensions: tuple[str, ...] = ()

    if provider_key == "vllm":
        shape_stable, warnings = _vllm_shape_warnings(payload, intent)
    elif provider_key == "llama.cpp":
        runtime_flags = resolve_llamacpp_prompt_cache_flags(
            app_config=app_config,
            runtime_context=runtime_context,
        )
        runtime_cache_mode = _llamacpp_runtime_mode(runtime_flags) if runtime_flags else "disabled_or_unknown"
        request_extensions = _request_extension_keys(request)

    return LocalCacheDiagnostic(
        provider=provider_key,
        prefix_cache_intent_requested=bool(intent.enabled),
        scope=intent.scope,
        static_segment_fingerprint=intent.static_segment_fingerprint,
        provider_hint_keys=intent.provider_hint_keys,
        request_shape_stable=shape_stable,
        warnings=warnings,
        runtime_cache_mode=runtime_cache_mode,
        runtime_flags=runtime_flags,
        request_extension_keys=request_extensions,
        billing_cache_authoritative=False,
        billing_cache_savings_reported=False,
    )


__all__ = [
    "InferencePrefixCacheIntent",
    "LocalCacheDiagnostic",
    "build_local_cache_diagnostic",
    "parse_inference_prefix_cache_intent",
    "resolve_llamacpp_prompt_cache_flags",
]
