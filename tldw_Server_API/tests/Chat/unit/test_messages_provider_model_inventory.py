"""Native Messages generation and token counting must use live model IDs."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import messages
from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.LLM_Calls import provider_model_inventory

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["messages", "count_tokens"])
@pytest.mark.parametrize("inventory_status, model, expected_status", [
    ("ready", "retired-model", 400),
    ("unavailable", "current-model", 503),
    ("ready", "current-model", None),
])
async def test_native_messages_validate_resolved_credential_inventory(
    monkeypatch, operation, inventory_status, model, expected_status,
):
    credentials = SimpleNamespace(
        api_key="synthetic-scope-key", credentials_resolved=True,
        app_config={"anthropic_api": {"api_base_url": "https://scope.example.invalid/v1"}},
    )

    async def resolve(provider, *, model):
        assert provider == "anthropic"
        return credentials

    calls = []

    def discover(provider, key, **kwargs):
        calls.append((provider, key, kwargs["base_url"]))
        return SimpleNamespace(status=inventory_status, models=("current-model",))

    monkeypatch.setattr(messages, "provider_auth_is_resolved", lambda *args, **kwargs: True)
    monkeypatch.setattr(chat_service, "discover_provider_models", discover)
    monkeypatch.setattr(
        provider_model_inventory, "_http_fetch",
        lambda **_kwargs: SimpleNamespace(status_code=404, json=lambda: {}, close=lambda: None),
    )
    runtime = SimpleNamespace(resolve=resolve)
    if expected_status:
        with pytest.raises(HTTPException) as error:
            await messages._resolve_messages_credentials(runtime, "anthropic", model, operation=operation)
        assert error.value.status_code == expected_status
        assert "synthetic-scope-key" not in str(error.value.detail)
    else:
        assert await messages._resolve_messages_credentials(
            runtime, "anthropic", model, operation=operation,
        ) is credentials
    assert calls == [("anthropic", "synthetic-scope-key", "https://scope.example.invalid/v1")]


@pytest.mark.asyncio
async def test_native_local_messages_do_not_require_cloud_inventory(monkeypatch):
    credentials = SimpleNamespace(api_key=None, app_config={}, credentials_resolved=True)

    async def resolve(*args, **kwargs):
        return credentials

    monkeypatch.setattr(chat_service, "discover_provider_models", lambda *args, **kwargs: pytest.fail("Local discovery"))
    assert await messages._resolve_messages_credentials(
        SimpleNamespace(resolve=resolve), "llama.cpp", "local-model", operation="messages",
    ) is credentials
