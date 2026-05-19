import pytest

def test_provider_requires_key_map_basic():

    from tldw_Server_API.app.core.LLM_Calls.provider_metadata import provider_requires_api_key

    # Commercial providers should require keys
    for prov in [
        "openai",
        "anthropic",
        "google",
        "mistral",
        "cohere",
        "groq",
        "openrouter",
        "novita",
        "poe",
        "together",
        "custom-openai-api-99",
    ]:
        assert provider_requires_api_key(prov) is True

    # Local providers typically do not require keys
    for prov in ["llama.cpp", "kobold", "ooba", "tabbyapi", "vllm", "local-llm", "ollama", "aphrodite"]:
        assert provider_requires_api_key(prov) is False

    # Unknown providers default to requiring keys
    assert provider_requires_api_key("unknown-provider") is True
