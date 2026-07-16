from __future__ import annotations

import asyncio
import os
from typing import Any, Literal

from tldw_Server_API.app.core.AuthNZ.byok_config import build_app_config_overrides
from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import normalize_provider_name
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAPIError,
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatProviderError,
)
from tldw_Server_API.app.core.Chat.chat_orchestrator import chat_api_call
from tldw_Server_API.app.core.config import load_comprehensive_config
from tldw_Server_API.app.core.http_client import RetryPolicy, afetch_json
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import normalize_provider
from tldw_Server_API.app.core.LLM_Calls.provider_metadata import list_registered_providers
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.TTS.gateway_catalog import (
    MAX_DISCOVERY_BYTES,
    _parse_discovered_models,
)
from tldw_Server_API.app.core.TTS.gateway_config import build_gateway_url

_INVALID_TEST_KEY_PREFIXES = ("invalid-", "test-invalid-", "bad-key-", "dummy-invalid-")
GatewayVerificationStatus = Literal["verified", "stored-unverified", "rejected"]


class _GatewayCredentialRejected(Exception):
    pass


class _GatewayProbeUnavailable(Exception):
    pass


def _is_test_mode() -> bool:
    return is_test_mode() or os.getenv("PYTEST_CURRENT_TEST") is not None


def is_obviously_invalid_key(api_key: str) -> bool:
    key = (api_key or "").strip()
    if not key:
        return False
    lowered = key.lower()
    return any(lowered.startswith(prefix) for prefix in _INVALID_TEST_KEY_PREFIXES)


def resolve_default_model_for_provider(provider: str) -> str | None:
    try:
        from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
            get_llm_provider_override,
            get_override_default_model,
        )

        override_default = get_override_default_model(provider)
        if override_default:
            return override_default
        override = get_llm_provider_override(provider)
        if override and override.allowed_models:
            return override.allowed_models[0]
    except Exception as override_error:
        _ = override_error  # continue with default model fallback

    normalized = (provider or "").replace(".", "_").replace("-", "_")
    env_key = f"DEFAULT_MODEL_{normalized.upper()}"
    env_val = os.getenv(env_key)
    if isinstance(env_val, str) and env_val.strip():
        return env_val.strip()

    cfg = None
    try:
        cfg = load_comprehensive_config()
    except Exception:
        cfg = None

    if cfg is not None and getattr(cfg, "has_section", None) and cfg.has_section("Chat-Module"):
        config_key = f"default_model_{normalized.lower()}"
        try:
            cfg_val = cfg.get("Chat-Module", config_key, fallback=None)
        except Exception:
            cfg_val = None
        if isinstance(cfg_val, str) and cfg_val.strip():
            return cfg_val.strip()

    return None


def build_app_config_for_provider(provider: str, credential_fields: dict[str, Any] | None) -> dict[str, Any]:
    return build_app_config_overrides(provider, credential_fields)


async def probe_gateway_credentials(
    *,
    spec: Any,
    api_key: str,
) -> GatewayVerificationStatus:
    """Probe configured model discovery without issuing a synthesis request."""
    if (
        not bool(getattr(spec, "enabled", False))
        or not bool(getattr(getattr(spec, "discovery", None), "enabled", False))
        or not getattr(spec, "models_path", None)
    ):
        return "stored-unverified"

    async def _classify_status(status: int, _headers: Any) -> None:
        if status in {401, 403}:
            raise _GatewayCredentialRejected
        if status < 200 or status >= 300:
            raise _GatewayProbeUnavailable

    headers = dict(getattr(spec, "headers", ()) or ())
    headers["Authorization"] = f"Bearer {api_key}"
    try:
        payload = await afetch_json(
            method="GET",
            url=str(build_gateway_url(spec.base_url, spec.models_path)),
            params=dict(getattr(spec, "discovery_query", ()) or ()),
            headers=headers,
            timeout=spec.discovery.timeout_seconds,
            retry=RetryPolicy(attempts=1),
            require_json_ct=True,
            max_bytes=MAX_DISCOVERY_BYTES,
            allow_redirects=False,
            on_response=_classify_status,
        )
        _parse_discovered_models(payload)
    except _GatewayCredentialRejected:
        return "rejected"
    except Exception:
        return "stored-unverified"
    return "verified"


async def test_provider_credentials(
    *,
    provider: str,
    api_key: str,
    credential_fields: dict[str, Any] | None = None,
    model: str | None = None,
) -> str:
    provider_norm = normalize_provider_name(provider)
    provider_registry_name = normalize_provider(provider_norm)
    adapter = get_registry().get_adapter(provider_registry_name)
    if adapter is None and provider_registry_name not in list_registered_providers():
        raise ValueError(f"Provider '{provider_norm}' does not support key tests yet")

    model_to_use = model or resolve_default_model_for_provider(provider_norm)
    if not model_to_use:
        raise ValueError(
            f"Model is required for provider '{provider_norm}'. "
            f"Configure DEFAULT_MODEL_{provider_norm.replace('.', '_').replace('-', '_').upper()} or pass model."
        )

    if _is_test_mode():
        if is_obviously_invalid_key(api_key):
            raise ChatAuthenticationError(
                message=f"Provider '{provider_norm}' rejected the supplied credentials.",
                provider=provider_norm,
            )
        return model_to_use

    app_config = build_app_config_for_provider(provider_norm, credential_fields)
    messages_payload = [{"role": "user", "content": "ping"}]
    try:
        if adapter is not None:
            request = {
                "messages": messages_payload,
                "system_message": None,
                "model": model_to_use,
                "api_key": api_key,
                "temperature": 0.0,
                "max_tokens": 1,
                "app_config": app_config,
            }
            await asyncio.to_thread(adapter.chat, request)
        else:
            await asyncio.to_thread(
                chat_api_call,
                api_endpoint=provider_norm,
                messages_payload=messages_payload,
                api_key=api_key,
                model=model_to_use,
                temp=0.0,
                max_tokens=1,
                streaming=False,
                app_config=app_config,
            )
    except ChatAPIError:
        raise
    except ValueError as exc:
        raise ChatBadRequestError(message=str(exc), provider=provider_norm) from exc
    except Exception as exc:
        raise ChatProviderError(
            provider=provider_norm,
            message="Provider test call failed",
            status_code=502,
            details=str(exc),
        ) from exc

    return model_to_use
