import pytest

from tldw_Server_API.app.core.Claims_Extraction.runtime_config import (
    resolve_claims_alignment_config,
    resolve_claims_context_window_chars,
    resolve_claims_extraction_passes,
    resolve_claims_json_parse_mode,
    resolve_claims_llm_config,
    resolve_claims_prompt_validation_config,
)


@pytest.mark.unit
def test_resolve_claims_llm_config_prefers_claims_values():
    settings_obj = {
        "CLAIMS_LLM_PROVIDER": "groq",
        "CLAIMS_LLM_MODEL": "claims-model",
        "CLAIMS_LLM_TEMPERATURE": "0.42",
        "RAG": {"default_llm_provider": "openai", "default_llm_model": "rag-model"},
        "default_api": "anthropic",
    }
    provider, model, temperature = resolve_claims_llm_config(settings_obj)
    assert provider == "groq"
    assert model == "claims-model"
    assert temperature == pytest.approx(0.42)


@pytest.mark.unit
def test_resolve_claims_llm_config_falls_back_to_rag_then_default_api():
    provider, model, temperature = resolve_claims_llm_config(
        {
            "RAG": {"default_llm_provider": "google", "default_llm_model": "rag-model"},
            "default_api": "deepseek",
        }
    )
    assert provider == "google"
    assert model == "rag-model"
    assert temperature == pytest.approx(0.1)

    provider2, model2, temperature2 = resolve_claims_llm_config({"default_api": "deepseek"})
    assert provider2 == "deepseek"
    assert model2 is None
    assert temperature2 == pytest.approx(0.1)


@pytest.mark.unit
def test_resolve_claims_json_parse_mode_defaults_to_lenient_on_invalid():
    assert resolve_claims_json_parse_mode({"CLAIMS_JSON_PARSE_MODE": "strict"}) == "strict"
    assert resolve_claims_json_parse_mode({"CLAIMS_JSON_PARSE_MODE": "invalid"}) == "lenient"


@pytest.mark.unit
def test_resolve_claims_alignment_config_validates_and_clamps_threshold():
    mode, threshold = resolve_claims_alignment_config(
        {"CLAIMS_ALIGNMENT_MODE": "fuzzy", "CLAIMS_ALIGNMENT_THRESHOLD": 1.7}
    )
    assert mode == "fuzzy"
    assert threshold == pytest.approx(1.0)

    mode2, threshold2 = resolve_claims_alignment_config(
        {"CLAIMS_ALIGNMENT_MODE": "not-valid", "CLAIMS_ALIGNMENT_THRESHOLD": -3}
    )
    assert mode2 == "fuzzy"
    assert threshold2 == pytest.approx(0.0)


@pytest.mark.unit
def test_resolve_claims_prompt_validation_config_defaults_and_normalizes():
    mode, strict = resolve_claims_prompt_validation_config(
        {
            "CLAIMS_PROMPT_VALIDATION_MODE": "ERROR",
            "CLAIMS_PROMPT_VALIDATION_STRICT": "true",
        }
    )
    assert mode == "error"
    assert strict is True

    mode2, strict2 = resolve_claims_prompt_validation_config(
        {
            "CLAIMS_PROMPT_VALIDATION_MODE": "invalid",
            "CLAIMS_PROMPT_VALIDATION_STRICT": "not-a-bool",
        }
    )
    assert mode2 == "warning"
    assert strict2 is False


@pytest.mark.unit
def test_resolve_claims_context_window_chars_and_passes_are_bounded():
    assert resolve_claims_context_window_chars({"CLAIMS_CONTEXT_WINDOW_CHARS": -100}) == 0
    assert resolve_claims_context_window_chars({"CLAIMS_CONTEXT_WINDOW_CHARS": "2048"}) == 2048
    assert resolve_claims_context_window_chars({"CLAIMS_CONTEXT_WINDOW_CHARS": "999999"}) == 20000

    assert resolve_claims_extraction_passes({"CLAIMS_EXTRACTION_PASSES": -1}) == 1
    assert resolve_claims_extraction_passes({"CLAIMS_EXTRACTION_PASSES": "3"}) == 3
    assert resolve_claims_extraction_passes({"CLAIMS_EXTRACTION_PASSES": "999"}) == 10
