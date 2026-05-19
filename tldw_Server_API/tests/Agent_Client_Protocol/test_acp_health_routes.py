"""Tests for route-gating status in ACP health endpoint response."""
from __future__ import annotations

import importlib.machinery
import os
import sys
import types
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


# Stub heavyweight audio deps before app import in shared fixtures.
if "torch" not in sys.modules:
    _fake_torch = types.ModuleType("torch")
    _fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    _fake_torch.Tensor = object
    _fake_torch.nn = types.SimpleNamespace(Module=object)
    sys.modules["torch"] = _fake_torch

if "faster_whisper" not in sys.modules:
    _fake_fw = types.ModuleType("faster_whisper")
    _fake_fw.__spec__ = importlib.machinery.ModuleSpec("faster_whisper", loader=None)

    class _StubWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

    _fake_fw.WhisperModel = _StubWhisperModel
    _fake_fw.BatchedInferencePipeline = _StubWhisperModel
    sys.modules["faster_whisper"] = _fake_fw

if "transformers" not in sys.modules:
    _fake_tf = types.ModuleType("transformers")
    _fake_tf.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)

    class _StubProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _StubModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    _fake_tf.AutoProcessor = _StubProcessor
    _fake_tf.Qwen2AudioForConditionalGeneration = _StubModel
    sys.modules["transformers"] = _fake_tf


class StubRunnerClient:
    def __init__(self) -> None:
        self.agent_capabilities = {"promptCapabilities": {"image": False}}
        self.is_running = True

    async def create_session(self, cwd, mcp_servers=None, agent_type=None, user_id=None, **kwargs):
        return "session-123"

    async def prompt(self, session_id, prompt):
        return {"stopReason": "end"}

    async def cancel(self, session_id):
        pass

    async def close_session(self, session_id):
        pass

    def pop_updates(self, session_id, limit=100):
        return []


@pytest.fixture()
def stub_runner_client(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

    stub = StubRunnerClient()

    async def _get_runner_client():
        return stub

    monkeypatch.setattr(acp_endpoints, "get_runner_client", _get_runner_client)
    return stub


def test_health_includes_routes_section(client_user_only, stub_runner_client):
    """Health endpoint response includes a 'routes' section."""
    resp = client_user_only.get("/api/v1/acp/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "routes" in data
    routes = data["routes"]
    assert isinstance(routes, dict)
    assert "stable_only" in routes
    assert "acp_enabled" in routes
    assert isinstance(routes["stable_only"], bool)
    assert isinstance(routes["acp_enabled"], bool)


def test_health_routes_note_when_hidden(client_user_only, stub_runner_client, monkeypatch):
    """When stable_only hides ACP, the routes section includes a note."""
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_mod

    # Mock the route functions to simulate stable_only hiding ACP
    def _mock_route_enabled(key, *, default_stable=True):
        if key == "acp":
            return False
        return True

    def _mock_policy():
        return {"stable_only": True, "enable": set(), "disable": set(), "experimental": {"acp"}}

    with patch(
        "tldw_Server_API.app.core.config.route_enabled",
        side_effect=_mock_route_enabled,
    ), patch(
        "tldw_Server_API.app.core.config._route_toggle_policy",
        side_effect=_mock_policy,
    ):
        resp = client_user_only.get("/api/v1/acp/health")

    assert resp.status_code == 200
    data = resp.json()
    routes = data["routes"]
    assert routes["stable_only"] is True
    assert routes["acp_enabled"] is False
    assert routes["note"] is not None
    assert "stable_only" in routes["note"]


def test_health_routes_no_note_when_enabled(client_user_only, stub_runner_client, monkeypatch):
    """When ACP is enabled, no warning note is present."""
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_mod

    def _mock_route_enabled(key, *, default_stable=True):
        return True

    def _mock_policy():
        return {"stable_only": False, "enable": set(), "disable": set(), "experimental": set()}

    with patch(
        "tldw_Server_API.app.core.config.route_enabled",
        side_effect=_mock_route_enabled,
    ), patch(
        "tldw_Server_API.app.core.config._route_toggle_policy",
        side_effect=_mock_policy,
    ):
        resp = client_user_only.get("/api/v1/acp/health")

    assert resp.status_code == 200
    data = resp.json()
    routes = data["routes"]
    assert routes["stable_only"] is False
    assert routes["acp_enabled"] is True
    assert routes["note"] is None
