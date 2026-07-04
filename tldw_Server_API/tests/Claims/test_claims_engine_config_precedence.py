import pytest

from tldw_Server_API.app.core.Claims_Extraction.claims_engine import _resolve_claims_llm_config
import tldw_Server_API.app.core.config as config_mod


def _snapshot(keys: list[str]) -> dict[str, object]:
    settings = config_mod.settings
    return {key: settings.get(key) for key in keys}


def _restore(snapshot: dict[str, object]) -> None:
    settings = config_mod.settings
    for key, value in snapshot.items():
        if value is None:
            settings.pop(key, None)
        else:
            settings[key] = value


@pytest.mark.unit
def test_claims_llm_config_prefers_claims_specific_settings():
    keys = ["CLAIMS_LLM_PROVIDER", "CLAIMS_LLM_MODEL", "CLAIMS_LLM_TEMPERATURE", "RAG", "default_api"]
    snap = _snapshot(keys)
    settings = config_mod.settings
    try:
        settings["CLAIMS_LLM_PROVIDER"] = "groq"
        settings["CLAIMS_LLM_MODEL"] = "claims-model"
        settings["CLAIMS_LLM_TEMPERATURE"] = 0.42
        settings["RAG"] = {"default_llm_provider": "openai", "default_llm_model": "rag-model"}
        settings["default_api"] = "anthropic"
        provider, model, temperature = _resolve_claims_llm_config()
        assert provider == "groq"
        assert model == "claims-model"
        assert temperature == pytest.approx(0.42)
    finally:
        _restore(snap)


@pytest.mark.unit
def test_claims_llm_config_falls_back_to_rag_defaults():
    keys = ["CLAIMS_LLM_PROVIDER", "CLAIMS_LLM_MODEL", "CLAIMS_LLM_TEMPERATURE", "RAG", "default_api"]
    snap = _snapshot(keys)
    settings = config_mod.settings
    try:
        settings.pop("CLAIMS_LLM_PROVIDER", None)
        settings.pop("CLAIMS_LLM_MODEL", None)
        settings.pop("CLAIMS_LLM_TEMPERATURE", None)
        settings["RAG"] = {"default_llm_provider": "google", "default_llm_model": "rag-default"}
        settings["default_api"] = "openai"
        provider, model, temperature = _resolve_claims_llm_config()
        assert provider == "google"
        assert model == "rag-default"
        assert temperature == pytest.approx(0.1)
    finally:
        _restore(snap)


@pytest.mark.unit
def test_claims_llm_config_falls_back_to_default_api_when_no_claims_or_rag():
    keys = ["CLAIMS_LLM_PROVIDER", "CLAIMS_LLM_MODEL", "CLAIMS_LLM_TEMPERATURE", "RAG", "default_api"]
    snap = _snapshot(keys)
    settings = config_mod.settings
    try:
        settings.pop("CLAIMS_LLM_PROVIDER", None)
        settings.pop("CLAIMS_LLM_MODEL", None)
        settings.pop("CLAIMS_LLM_TEMPERATURE", None)
        settings.pop("RAG", None)
        settings["default_api"] = "deepseek"
        provider, model, temperature = _resolve_claims_llm_config()
        assert provider == "deepseek"
        assert model is None
        assert temperature == pytest.approx(0.1)
    finally:
        _restore(snap)
