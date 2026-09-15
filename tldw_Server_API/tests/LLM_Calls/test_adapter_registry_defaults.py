from __future__ import annotations

from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry
from tldw_Server_API.app.core.LLM_Calls.providers.base import ChatProvider


def test_catalog_llama_alias_uses_the_same_adapter_and_credential_identity():
    from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name

    registry = get_registry()
    assert canonical_provider_name("llama") == "llama.cpp"
    assert registry.resolve_provider_name("llama") == "llama.cpp"


def test_registry_default_adapters_initialize():


    registry = get_registry()
    expected = {
        "openai",
        "anthropic",
        "groq",
        "openrouter",
        "google",
        "mistral",
        "qwen",
        "deepseek",
        "huggingface",
        "bedrock",
        "custom-openai-api",
        "custom-openai-api-2",
        "custom-openai-api-37",
        "custom-openai-api-99",
        "novita",
        "poe",
        "together",
        "mlx",
        "cohere",
    }

    for name in expected:
        adapter = registry.get_adapter(name)
        assert adapter is not None, f"Adapter missing for {name}"
        assert isinstance(adapter, ChatProvider)
        caps = adapter.capabilities()
        assert isinstance(caps, dict)
        assert "supports_streaming" in caps
