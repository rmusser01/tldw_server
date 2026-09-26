"""Fresh QA defaults follow configured Chat; explicit provider choices remain authoritative."""

import configparser

import pytest

from tldw_Server_API.app.core import config as config_module
from tldw_Server_API.app.core.RAG.rag_service.generation import AnswerGenerator
from tldw_Server_API.app.core.RAG.rag_service.streaming_executor import _generation_config


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "rag_provider,rag_model,request_provider,request_model,expected",
    [
        ("", "", None, None, ("llama.cpp", "local-study-model")),
        ("anthropic", "claude-custom", None, None, ("anthropic", "claude-custom")),
        ("anthropic", "claude-custom", "llama.cpp", None, ("llama.cpp", "local-study-model")),
        ("anthropic", "claude-custom", "llama", "explicit-local", ("llama.cpp", "explicit-local")),
        ("", "", "openai", None, ("openai", "configured-openai-model")),
    ],
)
def test_generation_defaults_preserve_provider_model_pair(
    monkeypatch, streaming, rag_provider, rag_model, request_provider, request_model, expected
):
    monkeypatch.delenv("RAG_DEFAULT_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("RAG_DEFAULT_LLM_MODEL", raising=False)
    monkeypatch.setattr(config_module, "load_and_log_configs", lambda: {
        "llm_api_settings": {"default_api": "llama"},
        "llama_api": {"model": "local-study-model"},
        "openai_api": {"model": "configured-openai-model"},
        "RAG_DEFAULT_LLM_PROVIDER": rag_provider,
        "RAG_DEFAULT_LLM_MODEL": rag_model,
    })
    if streaming:
        value = _generation_config(payload={
            "generation_provider": request_provider, "generation_model": request_model
        }, request_defaults={})
        actual = (value["provider"], value["model"])
    else:
        value = AnswerGenerator(provider=request_provider, model=request_model)
        actual = (value.provider, value.model)
    assert actual == expected


def test_explicit_rag_environment_defaults_override_chat(monkeypatch):
    monkeypatch.setenv("RAG_DEFAULT_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("RAG_DEFAULT_LLM_MODEL", "environment-model")
    monkeypatch.setattr(config_module, "load_and_log_configs", lambda: {
        "llm_api_settings": {"default_api": "llama"}, "llama_api": {"model": "local-model"}
    })
    value = _generation_config(payload={}, request_defaults={})
    assert (value["provider"], value["model"]) == ("anthropic", "environment-model")


@pytest.mark.parametrize("streaming", [False, True])
def test_provider_only_environment_override_does_not_reuse_previous_provider_model(monkeypatch, streaming):
    monkeypatch.setenv("RAG_DEFAULT_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("RAG_DEFAULT_LLM_MODEL", raising=False)
    monkeypatch.setattr(config_module, "load_and_log_configs", lambda: {
        "RAG_DEFAULT_LLM_PROVIDER": "openai", "RAG_DEFAULT_LLM_MODEL": "gpt-4o-mini",
        "anthropic_api": {"model": "claude-configured"},
    })
    if streaming:
        value = _generation_config(payload={}, request_defaults={})
        pair = (value["provider"], value["model"])
    else:
        value = AnswerGenerator()
        pair = (value.provider, value.model)
    assert pair == ("anthropic", "claude-configured")


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "configured_provider,configured_model,env_provider,env_model,expected",
    [
        ("openai", "gpt-4o-mini", "anthropic", None, ("anthropic", "claude-configured")),
        ("anthropic", "claude-custom", "anthropic", None, ("anthropic", "claude-custom")),
        ("llama", "custom-local", "llama.cpp", None, ("llama.cpp", "custom-local")),
        ("openai", "gpt-4o-mini", "anthropic", "env-model", ("anthropic", "env-model")),
        ("", "", None, None, ("llama.cpp", "local-study-model")),
    ],
)
def test_real_config_loader_preserves_generation_pair(
    monkeypatch, streaming, configured_provider, configured_model, env_provider, env_model, expected
):
    config = configparser.ConfigParser(interpolation=None)
    config.read_dict({
        "API": {"default_api": "llama", "anthropic_model": "claude-configured"},
        "Local-API": {"llama_model": "local-study-model"},
        "RAG": {"default_llm_provider": configured_provider, "default_llm_model": configured_model},
    })
    monkeypatch.setattr(config_module, "load_comprehensive_config", lambda: config)
    for key, value in (("RAG_DEFAULT_LLM_PROVIDER", env_provider), ("RAG_DEFAULT_LLM_MODEL", env_model)):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    if streaming:
        value = _generation_config(payload={}, request_defaults={})
        pair = (value["provider"], value["model"])
    else:
        value = AnswerGenerator()
        pair = (value.provider, value.model)
    assert pair == expected
