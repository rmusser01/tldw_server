import pytest

from tldw_Server_API.app.api.v1.endpoints import messages


@pytest.mark.unit
def test_messages_default_provider_uses_test_mode_single_letter_y(monkeypatch):
    monkeypatch.setattr("tldw_Server_API.app.core.config.loaded_config_data", {}, raising=False)
    monkeypatch.delenv("DEFAULT_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("TEST_MODE", "y")
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)

    assert messages._get_default_provider() == "local-llm"


@pytest.mark.unit
def test_messages_default_provider_uses_tldw_test_mode_single_letter_y(monkeypatch):
    monkeypatch.setattr("tldw_Server_API.app.core.config.loaded_config_data", {}, raising=False)
    monkeypatch.delenv("DEFAULT_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "y")

    assert messages._get_default_provider() == "local-llm"
