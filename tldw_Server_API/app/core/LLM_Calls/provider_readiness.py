"""Provider readiness helpers used by LLM catalog endpoints and clients."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Set as AbstractSet
from typing import Any
from urllib.parse import urlparse

from tldw_Server_API.app.core.Security.egress import evaluate_url_policy

_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}

_CHAT_PROVIDER_ALIASES = {
    "custom_openai_api": "custom-openai-api",
    "customopenaiapi": "custom-openai-api",
    "custom_openai_api_2": "custom-openai-api-2",
    "custom_openai_api2": "custom-openai-api-2",
    "customopenaiapi2": "custom-openai-api-2",
    "llama": "llama.cpp",
    "llamacpp": "llama.cpp",
    "llama_cpp": "llama.cpp",
    "llama-cpp": "llama.cpp",
    "tabby": "tabbyapi",
}

_UNAVAILABLE_PROVIDER_HEALTH_STATES = {
    "circuit_open",
    "disabled",
    "failed",
    "open",
    "unavailable",
    "unhealthy",
}

_UNAVAILABLE_PROVIDER_AVAILABILITY_STATES = {
    "disabled",
    "failed",
    "not-configured",
    "unavailable",
}


def _truthy(value: Any) -> bool:
    """Return whether a config or environment value represents an enabled flag."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_VALUES
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def normalize_catalog_provider_for_chat(provider_name: str) -> str:
    """Map provider catalog identifiers onto chat-completions provider names."""
    raw = (provider_name or "").strip().lower()
    compact = raw.replace(".", "_").replace("-", "_")
    alias = _CHAT_PROVIDER_ALIASES.get(raw) or _CHAT_PROVIDER_ALIASES.get(compact)
    if alias:
        return alias

    match = re.fullmatch(r"custom_openai(?:_api)?_?(\d{1,2})", compact)
    if match:
        return f"custom-openai-api-{match.group(1)}"
    return raw


def custom_openai_endpoint_requires_credentials(
    chat_provider: str,
    endpoint_url: str | None,
    api_key_value: str | None,
) -> bool:
    """Return whether a custom OpenAI-compatible endpoint needs credentials."""
    if api_key_value:
        return False
    if not chat_provider.startswith("custom-openai-api"):
        return False
    try:
        host = (urlparse((endpoint_url or "").strip()).hostname or "").lower()
    except ValueError:
        return False
    return host == "api.openai.com" or host.endswith(".api.openai.com")


def configured_endpoint_probe_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether readiness should probe local provider endpoints in-band."""
    values = env if env is not None else os.environ
    return _truthy(values.get("LLM_PROVIDER_READINESS_PROBE_ENDPOINTS", "0"))


def provider_readiness(
    *,
    provider_name: str,
    provider_info: dict[str, Any],
    is_configured: bool,
    endpoint_url: str | None,
    api_key_value: str | None,
    model_discovery: str | None,
    current_availability: Any,
    health_entry: Any,
    supported_chat_providers: AbstractSet[str],
    discover_models_from_endpoint: Callable[[str, str, str, str | None], list[str]] | None = None,
    endpoint_probe_enabled: bool = False,
) -> dict[str, Any]:
    """Compute user-facing readiness metadata for one configured provider."""
    chat_provider = normalize_catalog_provider_for_chat(provider_name)
    provider_enabled = bool(is_configured)
    availability = (
        str(current_availability).strip().lower()
        if isinstance(current_availability, str) and current_availability.strip()
        else ("enabled" if is_configured else "not-configured")
    )
    reason_code: str | None = None
    message: str | None = None

    if not is_configured:
        provider_enabled = False
        availability = "not-configured"
        reason_code = "provider_not_configured"
        message = "Provider is not configured."
    elif chat_provider not in supported_chat_providers:
        provider_enabled = False
        availability = "unavailable"
        reason_code = "unsupported_chat_provider"
        message = (
            f"Provider '{provider_name}' is not supported by chat completions. "
            "Choose a provider that can be used with /api/v1/chat/completions."
        )
    elif custom_openai_endpoint_requires_credentials(
        chat_provider,
        endpoint_url,
        api_key_value,
    ):
        provider_enabled = False
        availability = "not-configured"
        reason_code = "missing_credentials"
        message = (
            f"{provider_info.get('display_name') or provider_name} requires credentials "
            "for this endpoint before chat generation can run."
        )
    elif endpoint_url:
        try:
            policy = evaluate_url_policy(endpoint_url)
        except Exception as exc:  # noqa: BLE001 - readiness metadata should fail closed.
            provider_enabled = False
            availability = "unavailable"
            reason_code = "egress_policy_unavailable"
            message = f"Provider endpoint egress policy could not be evaluated: {exc}"
        else:
            if not policy.allowed:
                provider_enabled = False
                availability = "unavailable"
                reason_code = "egress_blocked"
                message = (
                    "Provider endpoint is blocked by the server egress policy"
                    + (f": {policy.reason}" if policy.reason else ".")
                )

    if provider_enabled and isinstance(health_entry, dict):
        health_status = str(health_entry.get("status") or "").strip().lower()
        if health_status in _UNAVAILABLE_PROVIDER_HEALTH_STATES:
            provider_enabled = False
            availability = "unavailable"
            reason_code = "provider_health_unavailable"
            message = (
                f"Provider health is {health_status}. Check provider settings "
                "before generating."
            )

    if (
        provider_enabled
        and provider_info.get("type") == "local"
        and endpoint_url
        and model_discovery
        and endpoint_probe_enabled
        and discover_models_from_endpoint is not None
    ):
        try:
            discovered_models = discover_models_from_endpoint(
                provider_name,
                endpoint_url,
                model_discovery,
                api_key_value,
            )
        except Exception:  # noqa: BLE001 - best-effort endpoint probes must not break listings.
            discovered_models = []
        if not discovered_models:
            provider_enabled = False
            availability = "unavailable"
            reason_code = "endpoint_unreachable"
            message = (
                f"{provider_info.get('display_name') or provider_name} endpoint "
                "could not be reached or did not return models."
            )

    if provider_enabled and availability in _UNAVAILABLE_PROVIDER_AVAILABILITY_STATES:
        provider_enabled = False
        if not reason_code:
            reason_code = "provider_unavailable"
            message = "Provider is unavailable."

    return {
        "availability": availability,
        "provider_enabled": provider_enabled,
        "readiness_reason_code": reason_code,
        "readiness_message": message,
        "chat_provider": chat_provider,
    }
