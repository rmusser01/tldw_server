"""A runtime-issued credential handle for tests that run the real LLM gateway against a stub adapter."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.AuthNZ import provider_credential_runtime as runtime_module


def issued_credentials(provider: str, *, api_key: str | None, app_config: dict[str, Any]) -> Any:
    """Build the handle the credential runtime would issue; the gateway rejects any other object."""
    return runtime_module.ProviderCallCredentials(
        provider=provider,
        api_key=api_key,
        app_config=app_config,
        auth_source="api_key",
        runtime_generation=1,
        runtime_identity=object(),
        credential_identity=object(),
        _issued_token=runtime_module._RUNTIME_ISSUED_CREDENTIAL_TOKEN,
    )
