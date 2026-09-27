"""Authentic provider credential capabilities for adapter boundary tests."""

from __future__ import annotations

import asyncio
from typing import Any

from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCallCredentials,
    ProviderCredentialRuntime,
)


async def issue_provider_call_credentials_async(
    provider: str,
    *,
    api_key: str | None,
    app_config: dict[str, Any] | None,
    model: str | None = None,
    auth_source: str | None = None,
    credential_fields: dict[str, Any] | None = None,
) -> ProviderCallCredentials:
    """Issue one genuine execution capability without consulting live config."""

    async def resolver(
        normalized_provider: str,
        **_kwargs: Any,
    ) -> ResolvedByokCredentials:
        return ResolvedByokCredentials(
            provider=normalized_provider,
            api_key=api_key,
            app_config=app_config,
            credential_fields=dict(credential_fields or {}),
            source="user",
            allowlisted=True,
            status=ByokResolutionStatus.RESOLVED,
            auth_source=auth_source or ("api_key" if api_key else None),
        )

    runtime = ProviderCredentialRuntime(
        user_id=7,
        team_ids=(),
        org_ids=(),
        trusted_base_url_override=True,
        server_config_snapshot={},
        resolver=resolver,
    )
    try:
        return await runtime.resolve(provider, model=model)
    finally:
        await runtime.close()


def issue_provider_call_credentials(
    provider: str,
    *,
    api_key: str | None,
    app_config: dict[str, Any] | None,
    model: str | None = None,
    auth_source: str | None = None,
    credential_fields: dict[str, Any] | None = None,
) -> ProviderCallCredentials:
    """Synchronously issue one genuine execution capability for unit tests."""

    return asyncio.run(
        issue_provider_call_credentials_async(
            provider,
            api_key=api_key,
            app_config=app_config,
            model=model,
            auth_source=auth_source,
            credential_fields=credential_fields,
        )
    )


def resolved_request_fields(
    provider: str,
    *,
    api_key: str | None,
    app_config: dict[str, Any] | None,
    model: str | None = None,
    auth_source: str | None = None,
    credential_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the marker and authentic capability expected by adapters."""

    handle = issue_provider_call_credentials(
        provider,
        api_key=api_key,
        app_config=app_config,
        model=model,
        auth_source=auth_source,
        credential_fields=credential_fields,
    )
    return {
        "api_key": handle.api_key,
        "app_config": handle.app_config,
        "credentials_resolved": True,
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: handle,
    }


async def resolved_request_fields_async(
    provider: str,
    *,
    api_key: str | None,
    app_config: dict[str, Any] | None,
    model: str | None = None,
    auth_source: str | None = None,
    credential_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return authentic resolved request fields from an active event loop."""

    handle = await issue_provider_call_credentials_async(
        provider,
        api_key=api_key,
        app_config=app_config,
        model=model,
        auth_source=auth_source,
        credential_fields=credential_fields,
    )
    return {
        "api_key": handle.api_key,
        "app_config": handle.app_config,
        "credentials_resolved": True,
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: handle,
    }
