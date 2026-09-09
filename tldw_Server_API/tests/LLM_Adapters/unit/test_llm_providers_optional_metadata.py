"""Keep optional catalog defaults distinct from invalid provider configuration."""

import configparser

import pytest

from tldw_Server_API.app.api.v1.endpoints import llm_providers

pytestmark = pytest.mark.unit


def _catalog_config(max_tokens: str) -> configparser.ConfigParser:
    """Build the minimal configured OpenAI catalog input.

    Args:
        max_tokens: Raw optional token limit, including blank or invalid values.

    Returns:
        An isolated parser with one OpenAI model and an empty local section.
    """
    config = configparser.ConfigParser()
    config.add_section("API")
    config.set("API", "openai_api_key", "sk-test")
    config.set("API", "openai_model", "gpt-4o-mini")
    config.set("API", "openai_max_tokens", max_tokens)
    config.add_section("Local-API")
    return config


def _patch_catalog_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    config: configparser.ConfigParser,
) -> None:
    """Isolate catalog assembly from providers, inventory and tokenizer probes.

    Args:
        monkeypatch: Scoped replacements for catalog dependencies.
        config: In-memory provider settings returned by the configuration loader.
    """
    monkeypatch.setattr(llm_providers, "load_comprehensive_config", lambda: config)
    monkeypatch.setattr(llm_providers, "get_api_keys", lambda: {})
    monkeypatch.setattr(llm_providers, "get_provider_manager", lambda: None)
    monkeypatch.setattr(llm_providers, "list_provider_models", lambda _provider: [])
    monkeypatch.setattr(llm_providers, "_llm_registry_capability_envelopes", lambda: {})
    monkeypatch.setattr(llm_providers, "_configured_endpoint_probe_enabled", lambda: False)
    monkeypatch.setattr(
        llm_providers,
        "_resolve_model_tokenizer_support",
        lambda *_args, **_kwargs: {
            "available": False,
            "tokenizer": None,
            "kind": None,
            "source": None,
            "detokenize": False,
            "count_accuracy": "unavailable",
            "strict_mode_effective": False,
        },
    )


def test_blank_optional_max_tokens_keeps_provider_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a configured provider visible when its optional token limit is blank.

    Args:
        monkeypatch: Scoped catalog dependency replacements.
    """
    _patch_catalog_dependencies(monkeypatch, _catalog_config(""))

    result = llm_providers.get_configured_providers()

    openai = next(provider for provider in result["providers"] if provider["name"] == "openai")
    assert "max_tokens" not in openai


def test_valid_optional_max_tokens_is_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain valid token, temperature and streaming metadata through the route.

    Args:
        monkeypatch: Scoped catalog dependency replacements.
    """
    config = _catalog_config("8192")
    config.set("API", "openai_temperature", "0.35")
    config.set("API", "openai_streaming", "True")
    _patch_catalog_dependencies(monkeypatch, config)

    result = llm_providers.get_configured_providers()

    openai = next(provider for provider in result["providers"] if provider["name"] == "openai")
    assert openai["max_tokens"] == 8192
    assert openai["default_temperature"] == 0.35
    assert openai["supports_streaming"] is True


def test_invalid_nonblank_max_tokens_keeps_config_error_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the existing sanitized catalog error for a malformed token limit.

    Args:
        monkeypatch: Scoped catalog dependency replacements.
    """
    _patch_catalog_dependencies(monkeypatch, _catalog_config("not-an-integer"))

    result = llm_providers.get_configured_providers()

    assert result["providers"] == []
    assert result["error"]
