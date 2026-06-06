"""Unit tests for chat persona exemplar default budget resolution."""

import pytest

from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint_module


@pytest.mark.unit
def test_persona_default_budget_uses_fallback_when_missing(monkeypatch):
    monkeypatch.delenv("PERSONA_EXEMPLAR_DEFAULT_BUDGET_TOKENS", raising=False)

    resolved = chat_endpoint_module._resolve_persona_default_budget_tokens({})

    assert resolved == 600


@pytest.mark.unit
def test_persona_default_budget_prefers_env_over_config(monkeypatch):
    monkeypatch.setenv("PERSONA_EXEMPLAR_DEFAULT_BUDGET_TOKENS", "777")

    resolved = chat_endpoint_module._resolve_persona_default_budget_tokens(
        {"persona_exemplar_default_budget_tokens": "444"}
    )

    assert resolved == 777


@pytest.mark.unit
def test_persona_default_budget_uses_config_when_env_not_set(monkeypatch):
    monkeypatch.delenv("PERSONA_EXEMPLAR_DEFAULT_BUDGET_TOKENS", raising=False)

    resolved = chat_endpoint_module._resolve_persona_default_budget_tokens(
        {"persona_exemplar_default_budget_tokens": "512"}
    )

    assert resolved == 512


@pytest.mark.unit
def test_persona_default_budget_invalid_values_fallback_or_clamp(monkeypatch):
    monkeypatch.delenv("PERSONA_EXEMPLAR_DEFAULT_BUDGET_TOKENS", raising=False)

    assert chat_endpoint_module._resolve_persona_default_budget_tokens(
        {"persona_exemplar_default_budget_tokens": "not-a-number"}
    ) == 600
    assert chat_endpoint_module._resolve_persona_default_budget_tokens(
        {"persona_exemplar_default_budget_tokens": "0"}
    ) == 1
    assert chat_endpoint_module._resolve_persona_default_budget_tokens(
        {"persona_exemplar_default_budget_tokens": "999999"}
    ) == 20_000


@pytest.mark.unit
def test_persona_effective_budget_auto_adjusts_on_sustained_alert_window(monkeypatch):
    monkeypatch.setattr(chat_endpoint_module, "_PERSONA_IOO_BUDGET_AUTO_ADJUST_ENABLED", True)
    monkeypatch.setattr(chat_endpoint_module, "_PERSONA_IOO_BUDGET_AUTO_REDUCTION_FACTOR", 0.5)
    monkeypatch.setattr(chat_endpoint_module, "_PERSONA_IOO_BUDGET_AUTO_MIN_TOKENS", 120)

    with chat_endpoint_module._persona_alert_guard:
        chat_endpoint_module._persona_ioo_windows.clear()
        window = chat_endpoint_module._get_persona_ioo_window_locked("u1:42")
        for _ in range(int(chat_endpoint_module._PERSONA_IOO_SUSTAIN_WINDOW)):
            window.append(1)

    budget, adjusted, reason = chat_endpoint_module._resolve_effective_persona_budget_tokens(
        budget_override=None,
        user_id="u1",
        character_id=42,
    )

    assert adjusted is True
    assert reason == "ioo_sustained_alert_window"
    assert budget < int(chat_endpoint_module._PERSONA_EXEMPLAR_DEFAULT_BUDGET)


@pytest.mark.unit
def test_persona_effective_budget_override_bypasses_auto_adjust(monkeypatch):
    monkeypatch.setattr(chat_endpoint_module, "_PERSONA_IOO_BUDGET_AUTO_ADJUST_ENABLED", True)
    monkeypatch.setattr(chat_endpoint_module, "_PERSONA_IOO_BUDGET_AUTO_REDUCTION_FACTOR", 0.5)
    monkeypatch.setattr(chat_endpoint_module, "_PERSONA_IOO_BUDGET_AUTO_MIN_TOKENS", 120)

    with chat_endpoint_module._persona_alert_guard:
        chat_endpoint_module._persona_ioo_windows.clear()
        window = chat_endpoint_module._get_persona_ioo_window_locked("u2:7")
        for _ in range(int(chat_endpoint_module._PERSONA_IOO_SUSTAIN_WINDOW)):
            window.append(1)

    budget, adjusted, reason = chat_endpoint_module._resolve_effective_persona_budget_tokens(
        budget_override=333,
        user_id="u2",
        character_id=7,
    )

    assert budget == 333
    assert adjusted is False
    assert reason == "request_override"
