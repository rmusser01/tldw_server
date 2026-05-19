import pytest

from tldw_Server_API.app.core.Evaluations.ms_g_eval import validate_inputs


@pytest.mark.unit
@pytest.mark.parametrize(
    ("api_name", "api_key"),
    [
        ("custom-openai-api", "key"),
        ("custom-openai-api-2", "key"),
        ("custom-openai-api-99", "key"),
        ("google", "key"),
        ("qwen", "key"),
        ("aphrodite", "key"),
        ("llama.cpp", None),
    ],
)
def test_validate_inputs_accepts_supported_providers(api_name, api_key):
    # Should not raise when called with supported providers
    validate_inputs("document", "summary", api_name, api_key)


@pytest.mark.unit
@pytest.mark.parametrize(
    "api_name",
    ["google", "qwen", "custom-openai-api", "custom-openai-api-2", "custom-openai-api-99", "aphrodite"],
)
def test_validate_inputs_enforces_keys_for_commercial_apis(api_name):
    with pytest.raises(ValueError, match="API key is required"):
        validate_inputs("document", "summary", api_name, api_key=None)


@pytest.mark.unit
@pytest.mark.parametrize("api_name", [None, "", "   ", 123])
def test_validate_inputs_rejects_empty_or_invalid_api_name(api_name):
    with pytest.raises(ValueError, match="Unsupported API"):
        validate_inputs("document", "summary", api_name, api_key="key")
