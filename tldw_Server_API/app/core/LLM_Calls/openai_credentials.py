"""Shared OpenAI credential metadata helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.core.exceptions import ChatConfigurationError

_MAX_CREDENTIAL_HEADER_VALUE_LENGTH = 512
OPENAI_EMBEDDING_RUNTIME_BOUNDARY_FLAG = "_require_provider_call_credentials"


def _credential_header_value(value: Any, *, provider: str) -> str | None:
    """Return one safe credential-derived header value or fail closed."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ChatConfigurationError(
            provider=provider,
            message="Invalid OpenAI credential header configuration.",
        ) from None
    cleaned = value.strip()
    if (
        not cleaned
        or len(cleaned) > _MAX_CREDENTIAL_HEADER_VALUE_LENGTH
        or not cleaned.isascii()
        or any(ord(character) < 32 or ord(character) == 127 for character in cleaned)
    ):
        raise ChatConfigurationError(
            provider=provider,
            message="Invalid OpenAI credential header configuration.",
        ) from None
    return cleaned


def openai_credential_base_url(app_config: Mapping[str, Any] | None) -> str | None:
    """Return the endpoint captured in one validated OpenAI config snapshot."""

    section = app_config.get("openai_api") if isinstance(app_config, Mapping) else None
    if not isinstance(section, Mapping):
        return None
    for field in ("api_base_url", "api_base", "base_url", "api_url", "endpoint"):
        value = section.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def openai_credential_headers(
    api_key: str | None,
    app_config: Mapping[str, Any] | None,
    *,
    provider: str = "openai",
) -> dict[str, str]:
    """Build OpenAI auth headers from one validated credential snapshot."""

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    section = app_config.get("openai_api") if isinstance(app_config, Mapping) else None
    provider_config = section if isinstance(section, Mapping) else {}
    organization = (
        provider_config.get("org_id")
        or provider_config.get("organization_id")
        or provider_config.get("organization")
    )
    project = provider_config.get("project_id") or provider_config.get("project")
    organization = _credential_header_value(organization, provider=provider)
    project = _credential_header_value(project, provider=provider)
    if organization is not None:
        headers["OpenAI-Organization"] = organization
    if project is not None:
        headers["OpenAI-Project"] = project
    return headers


def bind_openai_embedding_credentials(
    *,
    provider_credentials: object | None,
    credentials_resolved: bool,
    api_key_override: str | None,
    base_url_override: str | None,
) -> tuple[str | None, str | None, dict[str, Any]]:
    """Consume an authentic OpenAI capability into one atomic transport tuple."""

    from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    )
    from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
        bind_provider_call_credentials,
    )

    request: dict[str, Any] = {
        "api_key": api_key_override,
        "base_url": base_url_override,
        "credentials_resolved": credentials_resolved,
    }
    if provider_credentials is not None:
        request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] = provider_credentials
    bound, credentials = bind_provider_call_credentials(
        "openai",
        request,
        consume=True,
    )
    if credentials is None:
        raise ChatConfigurationError(
            provider="openai",
            message="Provider credentials require an active runtime capability.",
        )
    app_config = bound.get("app_config")
    config = app_config if isinstance(app_config, dict) else {}
    return (
        bound.get("api_key"),
        openai_credential_base_url(config),
        config,
    )
