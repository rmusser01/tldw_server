"""Unit tests for Persona Chat telemetry hook label and alert-window behavior."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint_module


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_persona_ioo_windows():
    with chat_endpoint_module._persona_alert_guard:
        chat_endpoint_module._persona_ioo_windows.clear()
    yield
    with chat_endpoint_module._persona_alert_guard:
        chat_endpoint_module._persona_ioo_windows.clear()


def test_character_telemetry_keeps_legacy_label_shape(monkeypatch):
    recorded: list[dict[str, str]] = []

    def _record_histogram(metric_name: str, value: float, labels: dict[str, str] | None = None):  # noqa: ARG001
        recorded.append(dict(labels or {}))

    monkeypatch.setattr(chat_endpoint_module, "log_histogram", _record_histogram)
    monkeypatch.setattr(chat_endpoint_module, "log_counter", lambda *args, **kwargs: None)

    chat_endpoint_module._record_persona_telemetry_hooks(
        telemetry={"ioo": 0.1, "ior": 0.2, "lcs": 0.05, "safety_flags": []},
        provider="openai",
        model="gpt-4o-mini",
        user_id="1",
        character_id=42,
        assistant_kind="character",
        assistant_id="42",
        debug_id=None,
    )

    assert recorded
    assert recorded[0] == {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "user_id": "1",
        "character_id": "42",
    }


def test_character_sustained_ioo_reader_matches_writer_key(monkeypatch):
    monkeypatch.setattr(chat_endpoint_module, "log_histogram", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_endpoint_module, "log_counter", lambda *args, **kwargs: None)

    for _ in range(chat_endpoint_module._PERSONA_IOO_SUSTAIN_WINDOW):
        chat_endpoint_module._record_persona_telemetry_hooks(
            telemetry={"ioo": 1.0, "ior": 0.2, "lcs": 0.05, "safety_flags": []},
            provider="openai",
            model="gpt-4o-mini",
            user_id="1",
            character_id=42,
            assistant_kind="character",
            assistant_id="42",
            debug_id=None,
        )

    assert chat_endpoint_module._has_sustained_persona_ioo_alerts("1", 42) is True


def test_persona_telemetry_sanitizes_assistant_id_and_caps_alert_windows(monkeypatch):
    recorded: list[dict[str, str]] = []

    def _record_histogram(metric_name: str, value: float, labels: dict[str, str] | None = None):  # noqa: ARG001
        recorded.append(dict(labels or {}))

    monkeypatch.setattr(chat_endpoint_module, "log_histogram", _record_histogram)
    monkeypatch.setattr(chat_endpoint_module, "log_counter", lambda *args, **kwargs: None)
    monkeypatch.setattr(chat_endpoint_module, "_PERSONA_IOO_WINDOW_MAX_KEYS", 2)

    for assistant_id in (
        "safe-persona",
        "unsafe/persona@example.com",
        "third-persona",
    ):
        chat_endpoint_module._record_persona_telemetry_hooks(
            telemetry={"ioo": 0.1, "ior": 0.2, "lcs": 0.05, "safety_flags": []},
            provider="openai",
            model="gpt-4o-mini",
            user_id="1",
            character_id=None,
            assistant_kind="persona",
            assistant_id=assistant_id,
            debug_id=None,
        )

    unsafe_label = recorded[3]["assistant_id"]
    assert unsafe_label.startswith("hash:")
    assert "unsafe/persona@example.com" not in unsafe_label
    with chat_endpoint_module._persona_alert_guard:
        assert len(chat_endpoint_module._persona_ioo_windows) == 2
