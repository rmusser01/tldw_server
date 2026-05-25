import pytest

from tldw_Server_API.app.core.Evaluations import ms_g_eval
from tldw_Server_API.app.core.Evaluations.ms_g_eval import validate_inputs


@pytest.mark.unit
@pytest.mark.parametrize(
    ("api_name", "api_key"),
    [
        ("custom-openai-api", None),
        ("custom-openai-api-2", None),
        ("custom-openai-api-99", None),
        ("aphrodite", None),
        ("google", "key"),
        ("qwen", "key"),
        ("llama.cpp", None),
    ],
)
def test_validate_inputs_accepts_supported_providers(api_name, api_key):
    # Should not raise when called with supported providers
    validate_inputs("document", "summary", api_name, api_key)


@pytest.mark.unit
@pytest.mark.parametrize(
    "api_name",
    ["google", "qwen", "openai"],
)
def test_validate_inputs_enforces_keys_for_commercial_apis(api_name):
    with pytest.raises(ValueError, match="API key is required"):
        validate_inputs("document", "summary", api_name, api_key=None)


@pytest.mark.unit
def test_run_geval_accepts_custom_openai_provider_without_key(monkeypatch):
    monkeypatch.setattr(ms_g_eval, "geval_summarization", lambda *args, **kwargs: 4)

    result = ms_g_eval.run_geval(
        "document",
        "summary",
        api_key=None,
        api_name="custom-openai-api",
        save=False,
    )

    assert not result["assessment"].startswith("Validation error")
    assert result["metrics"]["coherence"] == 4
    assert result["average_score"] == 4.0


@pytest.mark.unit
@pytest.mark.parametrize("api_name", [None, "", "   ", 123])
def test_validate_inputs_rejects_empty_or_invalid_api_name(api_name):
    with pytest.raises(ValueError, match="Unsupported API"):
        validate_inputs("document", "summary", api_name, api_key="key")
