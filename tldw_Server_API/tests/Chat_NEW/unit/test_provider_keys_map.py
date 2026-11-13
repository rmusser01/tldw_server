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
    ]:
        assert provider_requires_api_key(prov) is True

    # Local and user-configured OpenAI-compatible providers typically do not require keys.
    for prov in [
        "llama.cpp",
        "kobold",
        "ooba",
        "tabbyapi",
        "vllm",
        "local-llm",
        "ollama",
        "aphrodite",
        "custom-openai-api",
        "custom-openai-api-2",
        "custom-openai-api-99",
    ]:
        assert provider_requires_api_key(prov) is False

    # Unknown providers default to requiring keys
    assert provider_requires_api_key("unknown-provider") is True


@pytest.mark.parametrize(
    ("provider", "requires_key"),
    [
        ("llamacpp", False),
        (" LLAMA_CPP ", False),
        ("koboldcpp", False),
        ("oobabooga", False),
        ("tabby_api", False),
        ("local_llm", False),
        ("openai-compatible-99", False),
        ("oai", True),
        ("aws-bedrock", True),
        ("unknown-provider", True),
        ("", True),
    ],
)
def test_registered_alias_uses_provider_credential_policy(provider, requires_key):
    from tldw_Server_API.app.core.LLM_Calls.provider_metadata import provider_requires_api_key

    assert provider_requires_api_key(provider) is requires_key  # nosec B101 -- credential-policy regression.
