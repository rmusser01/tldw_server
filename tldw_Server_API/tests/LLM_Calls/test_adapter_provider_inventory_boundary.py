"""Direct adapter calls must enforce the same inventory as ordinary Chat."""

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError
from tldw_Server_API.app.core.LLM_Calls.capability_registry import ProviderCallPolicy
from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope


@pytest.mark.unit
@pytest.mark.parametrize("method", ["chat", "stream"])
def test_direct_adapter_retired_model_never_reaches_generation(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    monkeypatch.setattr(
        chat_service, "discover_provider_models",
        lambda *_args, **_kwargs: SimpleNamespace(status="ready", models=("current-model",)),
    )
    monkeypatch.setattr(
        chat_service, "resolve_provider_models_base_url",
        lambda *_args, **_kwargs: "https://provider.example.invalid/v1",
    )
    def unexpected_generation(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("A retired direct-adapter selection reached generation")

    from tldw_Server_API.app.core.LLM_Calls.providers import openai_adapter

    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_OPENAI", "1")
    adapter = OpenAIAdapter()
    monkeypatch.setattr(openai_adapter, "http_client_factory", unexpected_generation)
    request = {"model": "retired-model", "api_key": "synthetic-key", "messages": [], "app_config": {}}
    with pytest.raises(ChatBadRequestError):
        result = getattr(adapter, method)(request)
        if method == "stream":
            list(result)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["achat", "astream"])
async def test_direct_async_adapter_retired_model_never_reaches_generation(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
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
def test_direct_adapter_discovery_consumes_call_policy_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.LLM_Calls.providers import base

    monkeypatch.setattr(chat_service, "_validate_provider_model_selection", lambda *_args: None)
    clock = iter([10.0, 12.0])
    monkeypatch.setattr(base.time, "monotonic", lambda: next(clock, 12.0))
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_OPENAI", "1")
    calls = []

    def fetch(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"choices": []})

    adapter = OpenAIAdapter()
    monkeypatch.setattr(adapter, "http_fetcher", fetch)
    request = {"model": "current-model", "api_key": "synthetic-key", "app_config": {},
               "base_url": "https://provider.example.invalid/v1",
               "messages": [{"role": "user", "content": "synthetic"}],
               "call_policy": ProviderCallPolicy(maximum_timeout_seconds=10,
                   required_endpoint_scope=ConfiguredEndpointScope.from_url("https://provider.example.invalid/v1"))}
    adapter.chat(request)
    assert calls[0]["timeout"] == 8
    assert request["call_policy"].maximum_timeout_seconds == 10


def _record_generation_requests(monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]]) -> None:
    """Capture the public adapters' HTTP collaborator without external traffic."""
    from tldw_Server_API.app.core.LLM_Calls import chat_calls

    def post(url: str, **kwargs: Any) -> SimpleNamespace:
        calls.append({"url": url, **kwargs})
        return SimpleNamespace(
            status_code=200, raise_for_status=lambda: None, close=lambda: None,
            json=lambda: {"text": "ok", "choices": [{"message": {"content": "ok"}}]},
        )

    monkeypatch.setattr(chat_calls, "create_session_with_retries", lambda **kwargs: SimpleNamespace(post=post, close=lambda: None))


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["cohere", "moonshot"])
@pytest.mark.parametrize("app_config", [None, {}])
def test_legacy_functional_adapter_freezes_actual_config_and_key(
    monkeypatch: pytest.MonkeyPatch, provider: str, app_config: dict[str, Any] | None,
) -> None:
    from importlib import import_module

    module = import_module(f"tldw_Server_API.app.core.LLM_Calls.providers.{provider}_adapter")
    config = {f"{provider}_api": {
        "api_key": "synthetic-config-key", "model": "current-model",
        "api_base_url": "https://configured.example.invalid",
    }}
    monkeypatch.setattr(module, "load_and_log_configs", lambda: config)
    calls = []

    def discover(name: str, key: str | None, **kwargs: Any) -> SimpleNamespace:
        calls.append((name, key, kwargs["base_url"]))
        return SimpleNamespace(status="ready", models=("current-model",))

    monkeypatch.setattr(chat_service, "discover_provider_models", discover)
    generation = []
    _record_generation_requests(monkeypatch, generation)
    adapter = getattr(module, "CohereAdapter" if provider == "cohere" else "MoonshotAdapter")()
    adapter.chat({"app_config": app_config, "messages": [{"role": "user", "content": "synthetic"}]})
    assert calls == [(provider, "synthetic-config-key", "https://configured.example.invalid")]
    assert generation[0]["headers"]["Authorization"] == "Bearer synthetic-config-key"
    assert generation[0]["json"]["model"] == "current-model"
    assert generation[0]["url"] == "https://configured.example.invalid" + ("/v1/chat" if provider == "cohere" else "/chat/completions")


@pytest.mark.unit
def test_legacy_cohere_nested_api_uses_same_inventory_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.LLM_Calls.providers.cohere_adapter import CohereAdapter

    calls = []

    def discover(provider: str, key: str | None, **kwargs: Any) -> SimpleNamespace:
        calls.append((key, kwargs["base_url"]))
        return SimpleNamespace(status="ready", models=("current-model",))

    monkeypatch.setattr(chat_service, "discover_provider_models", discover)
    generation = []
    _record_generation_requests(monkeypatch, generation)
    config = {"API": {"cohere": {"api_key": "nested-key", "model": "current-model",
                                "api_base_url": "https://nested.example.invalid"}}}
    CohereAdapter().chat({"app_config": config, "messages": [{"role": "user", "content": "synthetic"}]})
    assert calls == [("nested-key", "https://nested.example.invalid")]
    assert generation[0]["url"] == "https://nested.example.invalid/v1/chat"
    assert generation[0]["headers"]["Authorization"] == "Bearer nested-key"
    assert "cohere_api" not in config
