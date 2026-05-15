from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


SUPPORTED_BILLING_CACHE_INTENT_PROVIDERS: frozenset[str] = frozenset(
    {"openai", "anthropic", "google", "openrouter"}
)
_MAX_FINGERPRINT_METADATA_CHARS = 96
_MAX_PROMPT_CACHE_KEY_CHARS = 64
_OPENROUTER_PROVIDER_KEYS = frozenset(
    {
        "allow_fallbacks",
        "order",
        "require_parameters",
        "sort",
    }
)


@dataclass(frozen=True)
class BillingPromptCacheIntent:
    """Provider-neutral request to apply documented billing prompt-cache controls."""

    enabled: bool = False
    scope: tuple[str, ...] = ()
    ttl_seconds: int | None = None
    static_segment_fingerprint: str | None = None
    provider_hint: Mapping[str, Any] = field(default_factory=dict)
    fail_open: bool = True


@dataclass(frozen=True)
class BillingPromptCacheIntentDiagnostic:
    """Bounded metadata describing request intent, not provider-proven cache usage."""

    provider: str
    cache_intent_requested: bool
    cache_intent_applied: bool
    reason: str
    scope: tuple[str, ...] = ()
    ttl_seconds: int | None = None
    static_segment_fingerprint: str | None = None
    provider_hint_keys: tuple[str, ...] = ()
    applied_fields: tuple[str, ...] = ()
    provider_usage_authoritative: bool = False

    def to_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "provider": self.provider,
            "cache_intent_requested": self.cache_intent_requested,
            "cache_intent_applied": self.cache_intent_applied,
            "reason": self.reason,
            "provider_usage_authoritative": self.provider_usage_authoritative,
        }
        if self.scope:
            metadata["scope"] = list(self.scope)
        if self.ttl_seconds is not None:
            metadata["ttl_seconds"] = self.ttl_seconds
        if self.static_segment_fingerprint:
            metadata["static_segment_fingerprint"] = _bound_text(
                self.static_segment_fingerprint,
                _MAX_FINGERPRINT_METADATA_CHARS,
            )
        if self.provider_hint_keys:
            metadata["provider_hint_keys"] = list(self.provider_hint_keys)
        if self.applied_fields:
            metadata["applied_fields"] = list(self.applied_fields)
        return metadata


def _provider_key(provider: str | None) -> str:
    return str(provider or "").strip().lower()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return False


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


def _ttl_seconds(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return min(parsed, 86_400)


def _bound_text(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[:limit]


def _provider_hint(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    hints: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            continue
        key = raw_key.strip()
        if not key:
            continue
        hints[key[:80]] = raw_value
        if len(hints) >= 16:
            break
    return hints


def parse_billing_prompt_cache_intent(value: Any) -> BillingPromptCacheIntent:
    """Parse a caller-supplied cache intent mapping, defaulting to disabled."""

    if isinstance(value, BillingPromptCacheIntent):
        return value
    if not isinstance(value, Mapping):
        return BillingPromptCacheIntent()
    return BillingPromptCacheIntent(
        enabled=_truthy(value.get("enabled")),
        scope=_normalize_scope(value.get("scope")),
        ttl_seconds=_ttl_seconds(value.get("ttl_seconds")),
        static_segment_fingerprint=_bound_text(
            value.get("static_segment_fingerprint"),
            _MAX_FINGERPRINT_METADATA_CHARS,
        ),
        provider_hint=_provider_hint(value.get("provider_hint")),
        fail_open=value.get("fail_open") is not False,
    )


def _request_intent(request: Mapping[str, Any]) -> BillingPromptCacheIntent:
    return parse_billing_prompt_cache_intent(
        request.get("billing_prompt_cache_intent")
        if request.get("billing_prompt_cache_intent") is not None
        else request.get("prompt_cache_intent")
    )


def _base_diagnostic(
    provider: str,
    intent: BillingPromptCacheIntent,
    *,
    applied: bool,
    reason: str,
    applied_fields: Sequence[str] = (),
) -> BillingPromptCacheIntentDiagnostic:
    return BillingPromptCacheIntentDiagnostic(
        provider=provider,
        cache_intent_requested=bool(intent.enabled),
        cache_intent_applied=bool(applied),
        reason=reason,
        scope=intent.scope,
        ttl_seconds=intent.ttl_seconds,
        static_segment_fingerprint=intent.static_segment_fingerprint,
        provider_hint_keys=tuple(sorted(str(key) for key in intent.provider_hint.keys())),
        applied_fields=tuple(applied_fields),
        provider_usage_authoritative=False,
    )


def _sanitize_prompt_cache_key(value: Any, *, from_fingerprint: bool = False) -> str | None:
    text = _bound_text(value, _MAX_PROMPT_CACHE_KEY_CHARS)
    if not text:
        return None
    text = re.sub(r"[^A-Za-z0-9._:-]+", "-", text)
    if not text:
        return None
    if from_fingerprint:
        text = f"tldw:{text}"
    return text[:_MAX_PROMPT_CACHE_KEY_CHARS]


def _cache_control(intent: BillingPromptCacheIntent, hint: Mapping[str, Any] | None = None) -> dict[str, str]:
    control = {"type": "ephemeral"}
    hint = hint or {}
    requested_ttl = hint.get("ttl")
    if requested_ttl in {"1h", "5m"}:
        if requested_ttl == "1h":
            control["ttl"] = "1h"
        return control
    if intent.ttl_seconds is not None and intent.ttl_seconds >= 3600:
        control["ttl"] = "1h"
    return control


def _apply_openai(
    payload: dict[str, Any],
    intent: BillingPromptCacheIntent,
    provider: str,
) -> tuple[dict[str, Any], BillingPromptCacheIntentDiagnostic]:
    applied_fields: list[str] = []
    hint_key = intent.provider_hint.get("prompt_cache_key")
    cache_key = _sanitize_prompt_cache_key(hint_key)
    if cache_key is None:
        cache_key = _sanitize_prompt_cache_key(intent.static_segment_fingerprint, from_fingerprint=True)
    if cache_key:
        payload["prompt_cache_key"] = cache_key
        applied_fields.append("prompt_cache_key")

    retention = intent.provider_hint.get("prompt_cache_retention")
    if retention not in {"in_memory", "24h"} and intent.ttl_seconds is not None and intent.ttl_seconds >= 86_400:
        retention = "24h"
    if retention in {"in_memory", "24h"}:
        payload["prompt_cache_retention"] = retention
        applied_fields.append("prompt_cache_retention")

    return payload, _base_diagnostic(
        provider,
        intent,
        applied=bool(applied_fields),
        reason="applied" if applied_fields else "openai_prompt_cache_automatic_no_request_hints",
        applied_fields=applied_fields,
    )


def _mark_content_block(content: Any, cache_control: Mapping[str, str]) -> tuple[Any, bool]:
    if isinstance(content, str):
        if not content:
            return content, False
        return [{"type": "text", "text": content, "cache_control": dict(cache_control)}], True
    if not isinstance(content, list):
        return content, False

    copied = copy.deepcopy(content)
    for item in reversed(copied):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "text" or not isinstance(item.get("text"), str) or not item.get("text"):
            continue
        item["cache_control"] = dict(cache_control)
        return copied, True
    return copied, False


def _apply_anthropic(
    payload: dict[str, Any],
    intent: BillingPromptCacheIntent,
    provider: str,
) -> tuple[dict[str, Any], BillingPromptCacheIntentDiagnostic]:
    applied_fields: list[str] = []
    control = _cache_control(intent, intent.provider_hint)
    if payload.get("system") is not None and (
        not intent.scope or any(scope in intent.scope for scope in ("system", "static", "world_book"))
    ):
        system_content, applied = _mark_content_block(payload.get("system"), control)
        payload["system"] = system_content
        if applied:
            applied_fields.append("system.cache_control")

    return payload, _base_diagnostic(
        provider,
        intent,
        applied=bool(applied_fields),
        reason="applied" if applied_fields else "anthropic_cacheable_system_block_required",
        applied_fields=applied_fields,
    )


def _nested_hint(intent: BillingPromptCacheIntent, key: str) -> Mapping[str, Any]:
    value = intent.provider_hint.get(key)
    return value if isinstance(value, Mapping) else {}


def _gemini_cached_content_name(intent: BillingPromptCacheIntent) -> str | None:
    google_hint = _nested_hint(intent, "google")
    for source in (intent.provider_hint, google_hint):
        for key in ("cachedContent", "cached_content"):
            value = source.get(key)
            if isinstance(value, str):
                candidate = value.strip()
                if candidate.startswith("cachedContents/"):
                    return candidate
    return None


def _apply_google(
    payload: dict[str, Any],
    intent: BillingPromptCacheIntent,
    provider: str,
) -> tuple[dict[str, Any], BillingPromptCacheIntentDiagnostic]:
    cached_content = _gemini_cached_content_name(intent)
    if cached_content:
        payload["cachedContent"] = cached_content
        return payload, _base_diagnostic(
            provider,
            intent,
            applied=True,
            reason="applied",
            applied_fields=("cachedContent",),
        )
    return payload, _base_diagnostic(
        provider,
        intent,
        applied=False,
        reason="gemini_cached_content_reference_required",
    )


def _sanitize_openrouter_provider(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    sanitized: dict[str, Any] = {}
    for key, raw_value in value.items():
        if key not in _OPENROUTER_PROVIDER_KEYS:
            continue
        if key == "order" and isinstance(raw_value, list):
            order = [str(item) for item in raw_value if isinstance(item, str) and item.strip()]
            if order:
                sanitized[key] = order[:16]
        elif key in {"allow_fallbacks", "require_parameters"} and isinstance(raw_value, bool):
            sanitized[key] = raw_value
        elif key == "sort" and isinstance(raw_value, str) and raw_value.strip():
            sanitized[key] = raw_value.strip()[:64]
    return sanitized


def _openrouter_cache_control_hint(intent: BillingPromptCacheIntent) -> Mapping[str, Any]:
    openrouter_hint = _nested_hint(intent, "openrouter")
    raw_control = openrouter_hint.get("cache_control", intent.provider_hint.get("cache_control"))
    if isinstance(raw_control, Mapping):
        control_type = raw_control.get("type")
        if control_type == "ephemeral":
            return _cache_control(intent, raw_control)
    if raw_control in {"automatic", "anthropic_ephemeral", True}:
        return _cache_control(intent, openrouter_hint)
    return {}


def _apply_openrouter(
    payload: dict[str, Any],
    intent: BillingPromptCacheIntent,
    provider: str,
) -> tuple[dict[str, Any], BillingPromptCacheIntentDiagnostic]:
    applied_fields: list[str] = []
    openrouter_hint = _nested_hint(intent, "openrouter")
    provider_hint = openrouter_hint.get("provider", intent.provider_hint.get("provider"))
    provider_payload = _sanitize_openrouter_provider(provider_hint)
    if provider_payload:
        payload["provider"] = provider_payload
        applied_fields.append("provider")

    control = _openrouter_cache_control_hint(intent)
    if control:
        payload["cache_control"] = dict(control)
        applied_fields.insert(0, "cache_control")

    return payload, _base_diagnostic(
        provider,
        intent,
        applied=bool(applied_fields),
        reason="applied" if applied_fields else "openrouter_provider_cache_hint_required",
        applied_fields=applied_fields,
    )


def apply_billing_prompt_cache_intent(
    provider: str,
    payload: Mapping[str, Any],
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], BillingPromptCacheIntentDiagnostic]:
    """Apply an explicit billing prompt-cache intent to a provider payload.

    The returned diagnostic describes request intent only. Authoritative cache
    hits, writes, and billing effects must still come from provider usage data.
    """

    provider_name = _provider_key(provider)
    intent = _request_intent(request)
    payload_copy = copy.deepcopy(dict(payload or {}))
    if not intent.enabled:
        return payload_copy, _base_diagnostic(
            provider_name,
            intent,
            applied=False,
            reason="intent_disabled",
        )
    if provider_name not in SUPPORTED_BILLING_CACHE_INTENT_PROVIDERS:
        return payload_copy, _base_diagnostic(
            provider_name,
            intent,
            applied=False,
            reason="provider_not_supported",
        )
    if provider_name == "openai":
        return _apply_openai(payload_copy, intent, provider_name)
    if provider_name == "anthropic":
        return _apply_anthropic(payload_copy, intent, provider_name)
    if provider_name == "google":
        return _apply_google(payload_copy, intent, provider_name)
    if provider_name == "openrouter":
        return _apply_openrouter(payload_copy, intent, provider_name)
    return payload_copy, _base_diagnostic(
        provider_name,
        intent,
        applied=False,
        reason="provider_not_supported",
    )


def attach_cache_intent_metadata(
    response: Any,
    diagnostic: BillingPromptCacheIntentDiagnostic,
) -> Any:
    """Attach bounded intent metadata to a response without modifying usage fields."""

    if (
        isinstance(response, dict)
        and diagnostic.cache_intent_requested
        and "tldw_cache_intent" not in response
    ):
        response["tldw_cache_intent"] = diagnostic.to_metadata()
    return response


__all__ = [
    "BillingPromptCacheIntent",
    "BillingPromptCacheIntentDiagnostic",
    "SUPPORTED_BILLING_CACHE_INTENT_PROVIDERS",
    "apply_billing_prompt_cache_intent",
    "attach_cache_intent_metadata",
    "parse_billing_prompt_cache_intent",
]
