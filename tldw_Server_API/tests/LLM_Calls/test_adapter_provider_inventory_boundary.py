"""Direct adapter calls must enforce the same inventory as ordinary Chat."""

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError
from tldw_Server_API.app.core.LLM_Calls.capability_registry import ProviderCallPolicy
from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter


@pytest.mark.unit
@pytest.mark.parametrize("method", ["chat", "stream"])
def test_direct_adapter_retired_model_never_reaches_generation(monkeypatch, method):
    monkeypatch.setattr(
        chat_service, "discover_provider_models",
        lambda *_args, **_kwargs: SimpleNamespace(status="ready", models=("current-model",)),
    )
    monkeypatch.setattr(
        chat_service, "resolve_provider_models_base_url",
        lambda *_args, **_kwargs: "https://provider.example.invalid/v1",
    )
    def unexpected_generation(*_args, **_kwargs):
        pytest.fail("A retired direct-adapter selection reached generation")

    adapter = OpenAIAdapter()
    monkeypatch.setattr(adapter, "_build_openai_payload", unexpected_generation)
    request = {"model": "retired-model", "api_key": "synthetic-key", "messages": [], "app_config": {}}
    with pytest.raises(ChatBadRequestError):
        result = getattr(adapter, method)(request)
        if method == "stream":
            list(result)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["achat", "astream"])
async def test_direct_async_adapter_retired_model_never_reaches_generation(monkeypatch, method):
    monkeypatch.setattr(
        chat_service, "discover_provider_models",
        lambda *_args, **_kwargs: SimpleNamespace(status="ready", models=("current-model",)),
    )
    monkeypatch.setattr(
        chat_service, "resolve_provider_models_base_url",
        lambda *_args, **_kwargs: "https://provider.example.invalid/v1",
    )
    adapter = OpenAIAdapter()
    request = {"model": "retired-model", "api_key": "synthetic-key", "messages": [], "app_config": {}}
    with pytest.raises(ChatBadRequestError):
        if method == "achat":
            await adapter.achat(request)
        else:
            async for _frame in adapter.astream(request):
                pytest.fail("Retired model yielded a generation frame")


@pytest.mark.unit
def test_direct_adapter_discovery_consumes_call_policy_deadline(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers import base

    monkeypatch.setattr(chat_service, "_validate_provider_model_selection", lambda *_args: None)
    clock = iter([10.0, 12.0])
    monkeypatch.setattr(base.time, "monotonic", lambda: next(clock, 12.0))
    request = {"model": "current-model", "api_key": "synthetic-key", "app_config": {},
               "call_policy": ProviderCallPolicy(maximum_timeout_seconds=10)}
    result = OpenAIAdapter()._bind_request_credentials(request)
    assert result["call_policy"].maximum_timeout_seconds == 8
    assert request["call_policy"].maximum_timeout_seconds == 10


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["cohere", "moonshot"])
@pytest.mark.parametrize("app_config", [None, {}])
def test_legacy_functional_adapter_freezes_actual_config_and_key(monkeypatch, provider, app_config):
    from importlib import import_module

    module = import_module(f"tldw_Server_API.app.core.LLM_Calls.providers.{provider}_adapter")
    config = {f"{provider}_api": {
        "api_key": "synthetic-config-key", "model": "current-model",
        "api_base_url": "https://configured.example.invalid/v1",
    }}
    monkeypatch.setattr(module, "load_and_log_configs", lambda: config)
    calls = []

    def discover(name, key, **kwargs):
        calls.append((name, key, kwargs["base_url"]))
        return SimpleNamespace(status="ready", models=("current-model",))

    monkeypatch.setattr(chat_service, "discover_provider_models", discover)
    adapter = getattr(module, "CohereAdapter" if provider == "cohere" else "MoonshotAdapter")()
    bound = adapter._bind_request_credentials({"app_config": app_config})
    assert calls == [(provider, "synthetic-config-key", "https://configured.example.invalid/v1")]
    assert bound["api_key"] == "synthetic-config-key"
    assert bound["app_config"] == config
    assert bound["base_url"] == "https://configured.example.invalid/v1"
    assert bound["model"] == "current-model"


@pytest.mark.unit
def test_legacy_cohere_nested_api_uses_same_inventory_endpoint(monkeypatch):
    from tldw_Server_API.app.core.LLM_Calls.providers.cohere_adapter import CohereAdapter

    calls = []

    def discover(provider, key, **kwargs):
        calls.append((key, kwargs["base_url"]))
        return SimpleNamespace(status="ready", models=("current-model",))

    monkeypatch.setattr(chat_service, "discover_provider_models", discover)
    config = {"API": {"cohere": {"api_key": "nested-key", "model": "current-model",
                                "api_base_url": "https://nested.example.invalid/v1"}}}
    bound = CohereAdapter()._bind_request_credentials({"app_config": config})
    assert calls == [("nested-key", "https://nested.example.invalid/v1")]
    assert bound["base_url"] == "https://nested.example.invalid/v1"
    assert "cohere_api" not in config
