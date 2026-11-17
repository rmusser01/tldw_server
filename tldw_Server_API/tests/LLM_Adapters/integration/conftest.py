"""Offline model inventories for tests that mock adapter generation traffic."""

import os
from collections.abc import Iterator
from typing import Any

import pytest

from tldw_Server_API.app.core.LLM_Calls.provider_readiness import ModelDiscoveryResult

_MODELS = {
    "openai": ("gpt-4o-mini", "test-model"),
    "anthropic": ("claude-sonnet", "test-model"),
    "groq": ("llama3-8b", "llama3-groq-8b", "llama-3.1-8b-instant", "test-model"),
    "openrouter": ("meta-llama/llama-3-8b", "openrouter/auto", "dummy", "test-model"),
    "google": ("gemini-2.5-pro", "test-model"),
    "mistral": ("mistral-large-latest", "dummy", "test-model"),
    "qwen": ("qwen-2", "qwen2.5:7b", "test-model"),
    "deepseek": ("deepseek-chat", "test-model"),
    "huggingface": ("meta-llama/Meta-Llama-3-8B-Instruct", "test-model"),
}


@pytest.fixture(autouse=True)
def offline_adapter_inventory(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> Iterator[list[str]]:
    """Keep unit inventory offline; non-unit tests may explicitly opt into live providers."""
    calls: list[str] = []
    network_enabled = os.getenv("ENABLE_NETWORK_TESTS", "").lower() in {"1", "true", "yes", "y", "on"}
    if network_enabled and request.node.get_closest_marker("unit") is None:
        yield calls
        return

    request.getfixturevalue("no_live_network")
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.LLM_Calls import gateway, provider_model_inventory

    def discover(provider: str, _api_key: str | None, **_kwargs: Any) -> ModelDiscoveryResult:
        """Return a provider-specific catalog, never echo the requested model."""
        calls.append(provider)
        return ModelDiscoveryResult("ready", _MODELS[provider])

    for module in (chat_service, gateway, provider_model_inventory):
        monkeypatch.setattr(module, "discover_provider_models", discover)
    yield calls
