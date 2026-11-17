"""Unit tests for the sandbox WS teardown shutdown-deadline resolution."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.endpoints.sandbox import _ws_shutdown_deadline_s

pytestmark = pytest.mark.unit


def test_env_override_in_milliseconds(monkeypatch):
    monkeypatch.setenv("WS_SHUTDOWN_TIMEOUT_MS", "250")
    assert _ws_shutdown_deadline_s() == pytest.approx(0.25)


def test_invalid_env_falls_back_to_production_default(monkeypatch):
    monkeypatch.setenv("WS_SHUTDOWN_TIMEOUT_MS", "not-a-number")
    monkeypatch.delenv("TEST_MODE", raising=False)
    assert _ws_shutdown_deadline_s() == 5.0


def test_test_mode_gets_tight_default(monkeypatch):
    monkeypatch.delenv("WS_SHUTDOWN_TIMEOUT_MS", raising=False)
    monkeypatch.setenv("TEST_MODE", "true")
    assert _ws_shutdown_deadline_s() == 0.5


def test_production_default(monkeypatch):
    monkeypatch.delenv("WS_SHUTDOWN_TIMEOUT_MS", raising=False)
    monkeypatch.delenv("TEST_MODE", raising=False)
    assert _ws_shutdown_deadline_s() == 5.0


def test_tiny_override_is_floored(monkeypatch):
    monkeypatch.setenv("WS_SHUTDOWN_TIMEOUT_MS", "0")
    assert _ws_shutdown_deadline_s() == 0.001
