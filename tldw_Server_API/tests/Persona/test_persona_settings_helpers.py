from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep
from tldw_Server_API.app.core import config as config_module

pytestmark = pytest.mark.unit


def test_persona_int_setting_invalid_value_logs_fallback(monkeypatch):
    debug_calls = []

    def _fake_debug(message, *args, **kwargs):
        debug_calls.append((message, args, kwargs))

    monkeypatch.setattr(config_module, "settings", {"PERSONA_MAX_TOOL_STEPS": "not-a-number"}, raising=False)
    monkeypatch.setattr(persona_ep.logger, "debug", _fake_debug)

    assert persona_ep._get_persona_max_tool_steps() == 3
    assert debug_calls
    assert any("PERSONA_MAX_TOOL_STEPS" in str(arg) for _, args, _ in debug_calls for arg in args)


def test_persona_ws_auth_interval_invalid_value_logs_and_falls_back(monkeypatch):
    debug_calls = []

    def _fake_debug(message, *args, **kwargs):
        debug_calls.append((message, args, kwargs))

    monkeypatch.setattr(
        config_module,
        "settings",
        {"PERSONA_WS_AUTH_REVALIDATE_INTERVAL_S": "not-a-number"},
        raising=False,
    )
    monkeypatch.setattr(persona_ep.logger, "debug", _fake_debug)

    assert persona_ep._get_persona_ws_auth_revalidate_interval_s() == 15.0
    assert debug_calls
    assert any(
        "PERSONA_WS_AUTH_REVALIDATE_INTERVAL_S" in str(arg)
        for _, args, _ in debug_calls
        for arg in args
    )


def test_persona_ws_auth_interval_keeps_disable_zero_and_clamp_behavior(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "settings",
        {"PERSONA_WS_AUTH_REVALIDATE_INTERVAL_S": "0"},
        raising=False,
    )
    assert persona_ep._get_persona_ws_auth_revalidate_interval_s() == 0.0

    monkeypatch.setattr(
        config_module,
        "settings",
        {"PERSONA_WS_AUTH_REVALIDATE_INTERVAL_S": "0.1"},
        raising=False,
    )
    assert persona_ep._get_persona_ws_auth_revalidate_interval_s() == 0.5

    monkeypatch.setattr(
        config_module,
        "settings",
        {"PERSONA_WS_AUTH_REVALIDATE_INTERVAL_S": "999"},
        raising=False,
    )
    assert persona_ep._get_persona_ws_auth_revalidate_interval_s() == 300.0
