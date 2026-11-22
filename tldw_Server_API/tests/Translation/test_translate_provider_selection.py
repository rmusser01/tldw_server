from configparser import ConfigParser

import pytest

from tldw_Server_API.app.api.v1.endpoints.translate import _get_default_provider
from tldw_Server_API.app.core import config
from tldw_Server_API.app.core.Provider_Credentials import llm_provider_overrides

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("source", ["config", "server_override", "blank_preferred_provider"])
def test_default_translation_uses_provider_with_configured_key(
    monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    key = "fixture-key"
    settings = ConfigParser()
    settings["API"] = {}
    if source != "server_override":
        settings["API"]["deepseek_api_key"] = key
    if source == "blank_preferred_provider":
        settings["API"]["openai_api_key"] = "  "
    monkeypatch.setattr(config, "load_comprehensive_config", lambda: settings)
    for provider in ("OPENAI", "ANTHROPIC", "OPENROUTER", "GROQ", "DEEPSEEK"):
        monkeypatch.delenv(provider + "_API_KEY", raising=False)

    def get_override(provider: str) -> llm_provider_overrides.LLMProviderOverride | None:
        if source == "server_override" and provider == "deepseek":
            return llm_provider_overrides.LLMProviderOverride(provider=provider, api_key=key)
        return None

    monkeypatch.setattr(llm_provider_overrides, "get_llm_provider_override", get_override)

    assert _get_default_provider() == "deepseek"
