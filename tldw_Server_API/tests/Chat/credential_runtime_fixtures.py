"""Provider credential runtime fixtures shared by Chat endpoint tests."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCallCredentials,
    ProviderCredentialRuntime,
)


@pytest.fixture
def execution_scoped_provider_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Install a real, request-owned credential runtime at the endpoint boundary."""

    async def resolve(
        provider: str,
        **_kwargs: Any,
    ) -> ResolvedByokCredentials:
        provider = provider.strip().lower()
        return ResolvedByokCredentials(
            provider=provider,
            api_key=f"test-{provider}-key",
            app_config={},
            credential_fields={},
            source="user",
            allowlisted=True,
            status=ByokResolutionStatus.RESOLVED,
            auth_source="api_key",
        )

    class Runtime(ProviderCredentialRuntime):
        instances: list[Runtime] = []

        def __init__(self, **kwargs: Any) -> None:
            kwargs.pop("override_snapshot_resolver", None)
            super().__init__(
                resolver=resolve,
                server_config_snapshot={},
                **kwargs,
            )
            self.handles: list[ProviderCallCredentials] = []
            self.close_calls = 0
            self.__class__.instances.append(self)

        async def resolve(
            self,
            provider: str,
            *,
            model: str | None = None,
            force_refresh: bool = False,
        ) -> ProviderCallCredentials:
            handle = await super().resolve(
                provider,
                model=model,
                force_refresh=force_refresh,
            )
            self.handles.append(handle)
            return handle

        async def close(self) -> None:
            self.close_calls += 1
            await super().close()

    monkeypatch.setattr(chat_endpoint, "ProviderCredentialRuntime", Runtime)
    yield

    assert len(Runtime.instances) == 1
    assert Runtime.instances[0].handles
    assert all(
        isinstance(handle, ProviderCallCredentials)
        for handle in Runtime.instances[0].handles
    )
    assert Runtime.instances[0].close_calls == 1
