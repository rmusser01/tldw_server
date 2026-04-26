"""
Tests for ACP WebSocket endpoint.

These tests verify the WebSocket-based real-time ACP session streaming,
including connection, message handling, and permission flows.
"""

import asyncio
import importlib.machinery
import json
import sys
import time
import types
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import (
    ACPRunnerClient,
    PendingPermission,
    SessionWebSocketRegistry,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.event_bus import SessionEventBus
from tldw_Server_API.app.core.Agent_Client_Protocol.stdio_client import ACPMessage
from tldw_Server_API.app.services.acp_runtime_policy_service import (
    ACPRuntimePolicySnapshot,
)

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


def _wait_for_permission_resolution(
    runner_client: "MockRunnerClient",
    session_id: str,
    request_id: str,
    timeout_s: float = 1.0,
) -> bool:
    """Poll for async websocket permission handling to update pending state."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if request_id not in runner_client._pending_permissions.get(session_id, {}):
            return True
        time.sleep(0.01)
    return request_id not in runner_client._pending_permissions.get(session_id, {})


class MockRunnerClient:
    """Mock ACP runner client for testing WebSocket interactions."""

    def __init__(self) -> None:
        self.agent_capabilities = {"promptCapabilities": {"image": False}}
        self._ws_registry: Dict[str, SessionWebSocketRegistry] = {}
        self._ws_callbacks: Dict[str, List[Any]] = {}
        self.prompt_calls: List[tuple] = []
        self.cancel_calls: List[str] = []
        self._pending_permissions: Dict[str, Dict[str, PendingPermission]] = {}
        self.expected_user_id: Optional[int] = None
        self.access_checks: List[tuple[str, int]] = []

    async def register_websocket(self, session_id: str, send_callback) -> None:
        if session_id not in self._ws_callbacks:
            self._ws_callbacks[session_id] = []
        self._ws_callbacks[session_id].append(send_callback)

    async def unregister_websocket(self, session_id: str, send_callback) -> None:
        if session_id in self._ws_callbacks:
            try:
                self._ws_callbacks[session_id].remove(send_callback)
            except ValueError:
                pass

    def has_websocket_connections(self, session_id: str) -> bool:
        return session_id in self._ws_callbacks and len(self._ws_callbacks[session_id]) > 0

    async def verify_session_access(self, session_id: str, user_id: int) -> bool:
        self.access_checks.append((session_id, user_id))
        if self.expected_user_id is None:
            return True
        return int(user_id) == int(self.expected_user_id)

    async def respond_to_permission(
        self,
        session_id: str,
        request_id: str,
        approved: bool,
        batch_approve_tier: Optional[str] = None,
    ) -> bool:
        if session_id in self._pending_permissions:
            if request_id in self._pending_permissions[session_id]:
                del self._pending_permissions[session_id][request_id]
                return True
        return False

    async def prompt(self, session_id: str, prompt: List[Dict]) -> Dict[str, Any]:
        self.prompt_calls.append((session_id, prompt))
        return {"stopReason": "end", "detail": "ok"}

    async def cancel(self, session_id: str) -> None:
        self.cancel_calls.append(session_id)

    async def broadcast_update(self, session_id: str, message: Dict[str, Any]) -> None:
        """Simulate broadcasting an update to connected WebSockets."""
        if session_id in self._ws_callbacks:
            for callback in self._ws_callbacks[session_id]:
                await callback(message)


class _FakeMultiplexWebSocket:
    """Minimal async WebSocket stub for direct multiplex endpoint tests."""

    def __init__(self, incoming: list[str] | None = None, headers: dict[str, str] | None = None) -> None:
        self._incoming = list(incoming or [])
        self.headers = headers or {}
        self.accepted = False
        self.closed_codes: list[int] = []
        self.sent_text: list[str] = []

    async def accept(self) -> None:
        self.accepted = True

    async def close(self, code: int = 1000) -> None:
        self.closed_codes.append(code)

    async def receive_text(self) -> str:
        if self._incoming:
            return self._incoming.pop(0)
        raise WebSocketDisconnect(code=1000)

    async def send_text(self, raw: str) -> None:
        self.sent_text.append(raw)


@pytest.fixture
def mock_runner_client():
    """Create a mock runner client."""
    return MockRunnerClient()


@pytest.fixture
def mock_get_runner_client(mock_runner_client, monkeypatch):
    """Patch get_runner_client to return our mock."""
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

    async def _get_runner_client():
        return mock_runner_client

    monkeypatch.setattr(acp_endpoints, "get_runner_client", _get_runner_client)
    return mock_runner_client


@pytest.fixture
def mock_jwt_manager(monkeypatch):
    """Mock JWT manager for WebSocket authentication."""
    mock_manager = MagicMock()
    mock_token_data = MagicMock()
    mock_token_data.user_id = 1
    mock_manager.verify_token.return_value = mock_token_data

    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

    monkeypatch.setattr(acp_endpoints, "get_jwt_manager", lambda: mock_manager)
    return mock_manager


@pytest.mark.asyncio
async def test_acp_session_stream_start_failure_skips_unset_callback_cleanup(monkeypatch):
    """Startup failures before callback assignment must still execute deterministic cleanup."""
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

    class _StubClient:
        def __init__(self) -> None:
            self.agent_capabilities = {"promptCapabilities": {"image": False}}
            self.unregister_calls: list[tuple[str, Any]] = []

        async def verify_session_access(self, session_id: str, user_id: int) -> bool:
            return True

        async def unregister_websocket(self, session_id: str, send_callback: Any) -> None:
            self.unregister_calls.append((session_id, send_callback))

    lifecycle = {"stop_calls": 0}
    stub_client = _StubClient()

    class _FailingStream:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def start(self) -> None:
            raise RuntimeError("stream_start_failure")

        async def stop(self) -> None:
            lifecycle["stop_calls"] += 1

        async def send_json(self, payload: dict[str, Any]) -> None:
            return None

        async def receive_json(self) -> dict[str, Any]:
            return {}

    class _FakeWebSocket:
        async def close(self, code: int = 1000) -> None:
            return None

    async def _fake_authenticate_ws(websocket, token=None, api_key=None, required_scope="read"):
        return 1

    async def _fake_get_runner_client():
        return stub_client

    monkeypatch.setattr(acp_endpoints, "_authenticate_ws", _fake_authenticate_ws)
    monkeypatch.setattr(acp_endpoints, "get_runner_client", _fake_get_runner_client)
    monkeypatch.setattr(acp_endpoints, "WebSocketStream", _FailingStream)

    await acp_endpoints.acp_session_stream(_FakeWebSocket(), "session-start-failure", token="valid-token")

    if stub_client.unregister_calls:
        pytest.fail("Expected no unregister call when stream start fails before callback assignment")
    if lifecycle["stop_calls"] != 1:
        pytest.fail(f"Expected exactly one stream.stop() call, got {lifecycle['stop_calls']}")


class TestACPWebSocketConnection:
    """Tests for WebSocket connection handling."""

    def test_websocket_connect_with_valid_token(
        self, client_user_only, mock_get_runner_client, mock_jwt_manager
    ):
        """Test successful WebSocket connection with valid JWT token."""
        with client_user_only.websocket_connect(
            "/api/v1/acp/sessions/test-session/stream?token=valid-token"
        ) as websocket:
            # Should receive connected message
            data = websocket.receive_json()
            assert data["type"] == "connected"
            assert data["session_id"] == "test-session"
            assert "agent_capabilities" in data

    def test_websocket_connect_with_subprotocol_bearer(
        self, client_user_only, mock_get_runner_client, mock_jwt_manager
    ):
        """Test WebSocket connection using Sec-WebSocket-Protocol bearer token."""
        with client_user_only.websocket_connect(
            "/api/v1/acp/sessions/test-session/stream",
            subprotocols=["bearer", "valid-token"],
        ) as websocket:
            data = websocket.receive_json()
            assert data["type"] == "connected"
            assert data["session_id"] == "test-session"

    def test_websocket_connect_with_async_jwt_manager(
        self, client_user_only, mock_get_runner_client, monkeypatch
    ):
        """Async verify_token implementations should be supported for JWT auth."""
        import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

        class AsyncJWTManager:
            async def verify_token(self, token: str):
                return types.SimpleNamespace(user_id=1)

        monkeypatch.setattr(acp_endpoints, "get_jwt_manager", lambda: AsyncJWTManager())

        with client_user_only.websocket_connect(
            "/api/v1/acp/sessions/test-session/stream?token=valid-token"
        ) as websocket:
            data = websocket.receive_json()
            assert data["type"] == "connected"

    def test_websocket_connect_with_revoked_token_fails(
        self, client_user_only, mock_get_runner_client, monkeypatch
    ):
        """Revoked/blacklisted JWT tokens should be rejected."""
        import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

        class RevokedJWTManager:
            async def verify_token(self, token: str):
                return None

        monkeypatch.setattr(acp_endpoints, "get_jwt_manager", lambda: RevokedJWTManager())

        with pytest.raises(Exception):
            with client_user_only.websocket_connect(
                "/api/v1/acp/sessions/test-session/stream?token=revoked-token"
            ):
                pass

    def test_websocket_connect_with_api_key(
        self, client_user_only, mock_get_runner_client, monkeypatch
    ):
        """Test WebSocket connection with API key in single-user mode."""
        import os

        monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key")

        with client_user_only.websocket_connect(
            "/api/v1/acp/sessions/test-session/stream?api_key=test-api-key"
        ) as websocket:
            data = websocket.receive_json()
            assert data["type"] == "connected"

    def test_websocket_stream_rejects_read_only_api_key_in_multi_user_mode(
        self, client_user_only, mock_get_runner_client, monkeypatch
    ):
        """ACP control stream must require write scope for multi-user API keys."""
        import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

        seen_required_scopes: list[str | None] = []

        class _Settings:
            AUTH_MODE = "multi_user"

        class _APIKeyManager:
            async def validate_api_key(self, api_key: str, required_scope: str | None = None, ip_address: str | None = None):
                seen_required_scopes.append(required_scope)
                if api_key == "write-key" and required_scope == "write":
                    return {"user_id": 1}
                raise RuntimeError("scope_denied")

        monkeypatch.setattr(acp_endpoints, "get_auth_settings", lambda: _Settings())
        monkeypatch.setattr(acp_endpoints, "resolve_client_ip", lambda websocket, settings=None: "127.0.0.1")

        async def _get_api_key_manager():
            return _APIKeyManager()

        monkeypatch.setattr(acp_endpoints, "get_api_key_manager", _get_api_key_manager)

        with pytest.raises(Exception):
            with client_user_only.websocket_connect(
                "/api/v1/acp/sessions/test-session/stream?api_key=read-key"
            ):
                pass

        assert seen_required_scopes == ["write"]

    def test_websocket_ssh_rejects_read_only_api_key_in_multi_user_mode(
        self, client_user_only, monkeypatch
    ):
        """ACP SSH socket must require write scope for multi-user API keys."""
        import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

        seen_required_scopes: list[str | None] = []

        class _Settings:
            AUTH_MODE = "multi_user"

        class _APIKeyManager:
            async def validate_api_key(self, api_key: str, required_scope: str | None = None, ip_address: str | None = None):
                seen_required_scopes.append(required_scope)
                raise RuntimeError("scope_denied")

        monkeypatch.setattr(acp_endpoints, "get_auth_settings", lambda: _Settings())
        monkeypatch.setattr(acp_endpoints, "resolve_client_ip", lambda websocket, settings=None: "127.0.0.1")

        async def _get_api_key_manager():
            return _APIKeyManager()

        monkeypatch.setattr(acp_endpoints, "get_api_key_manager", _get_api_key_manager)

        with pytest.raises(Exception):
            with client_user_only.websocket_connect(
                "/api/v1/acp/sessions/test-session/ssh?api_key=read-key"
            ):
                pass

        assert seen_required_scopes == ["write"]

    def test_websocket_connect_with_api_key_uses_configured_fixed_id(
        self, client_user_only, mock_get_runner_client, monkeypatch
    ):
        """Ensure API-key auth uses SINGLE_USER_FIXED_ID rather than hardcoded ID."""
        import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

        class _Settings:
            AUTH_MODE = "single_user"
            SINGLE_USER_API_KEY = "test-api-key"
            SINGLE_USER_FIXED_ID = 42

        mock_get_runner_client.expected_user_id = 42
        monkeypatch.setattr(acp_endpoints, "get_auth_settings", lambda: _Settings())
        monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key")

        with client_user_only.websocket_connect(
            "/api/v1/acp/sessions/test-session/stream?api_key=test-api-key"
        ) as websocket:
            data = websocket.receive_json()
            assert data["type"] == "connected"
        assert ("test-session", 42) in mock_get_runner_client.access_checks

    def test_websocket_rejects_single_user_test_key_outside_pytest_runtime(
        self, client_user_only, mock_get_runner_client, monkeypatch
    ):
        """SINGLE_USER_TEST_API_KEY must not be accepted outside explicit pytest runtime."""
        import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

        class _Settings:
            AUTH_MODE = "single_user"
            SINGLE_USER_API_KEY = "primary-api-key"
            SINGLE_USER_FIXED_ID = 1

        monkeypatch.setattr(acp_endpoints, "get_auth_settings", lambda: _Settings())
        monkeypatch.setattr(acp_endpoints, "is_explicit_pytest_runtime", lambda: False)
        monkeypatch.setattr(acp_endpoints, "resolve_client_ip", lambda websocket, settings=None: "127.0.0.1")
        monkeypatch.setattr(acp_endpoints, "is_single_user_ip_allowed", lambda ip, settings=None: True)
        monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", "test-only-key")
        monkeypatch.delenv("SINGLE_USER_API_KEY", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)

        with pytest.raises(Exception):
            with client_user_only.websocket_connect(
                "/api/v1/acp/sessions/test-session/stream?api_key=test-only-key"
            ):
                pass

    def test_websocket_rejects_single_user_api_key_when_ip_disallowed(
        self, client_user_only, mock_get_runner_client, monkeypatch
    ):
        """Single-user API-key auth should enforce IP allowlist parity."""
        import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

        class _Settings:
            AUTH_MODE = "single_user"
            SINGLE_USER_API_KEY = "test-api-key"
            SINGLE_USER_FIXED_ID = 7

        monkeypatch.setattr(acp_endpoints, "get_auth_settings", lambda: _Settings())
        monkeypatch.setattr(acp_endpoints, "resolve_client_ip", lambda websocket, settings=None: "10.0.0.25")
        monkeypatch.setattr(acp_endpoints, "is_single_user_ip_allowed", lambda ip, settings=None: False)
        monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key")

        with pytest.raises(Exception):
            with client_user_only.websocket_connect(
                "/api/v1/acp/sessions/test-session/stream?api_key=test-api-key"
            ):
                pass

    def test_websocket_connect_without_session_access_fails(
        self, client_user_only, mock_get_runner_client, mock_jwt_manager
    ):
        """Connection should be rejected when the user does not own the session."""
        mock_get_runner_client.expected_user_id = 999
        with pytest.raises(Exception):
            with client_user_only.websocket_connect(
                "/api/v1/acp/sessions/test-session/stream?token=valid-token"
            ):
                pass

    def test_websocket_connect_without_auth_fails(
        self, client_user_only, mock_get_runner_client, monkeypatch
    ):
        """Test that WebSocket connection without authentication closes with 4401."""
        # Clear any auth environment
        monkeypatch.delenv("SINGLE_USER_API_KEY", raising=False)

        # Mock JWT manager to fail
        import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

        mock_manager = MagicMock()
        mock_manager.verify_token.return_value = None
        monkeypatch.setattr(acp_endpoints, "get_jwt_manager", lambda: mock_manager)

        with pytest.raises(Exception):
            # Connection should be rejected
            with client_user_only.websocket_connect(
                "/api/v1/acp/sessions/test-session/stream"
            ) as websocket:
                pass

    def test_websocket_connection_quota_per_user_enforced(
        self, client_user_only, mock_get_runner_client, mock_jwt_manager, monkeypatch
    ):
        """Second concurrent stream from same user should be rejected when per-user cap is reached."""
        monkeypatch.setenv("ACP_WS_MAX_CONNECTIONS_PER_USER", "1")

        with client_user_only.websocket_connect(
            "/api/v1/acp/sessions/test-session/stream?token=valid-token"
        ) as ws1:
            data = ws1.receive_json()
            assert data["type"] == "connected"

            with pytest.raises(Exception):
                with client_user_only.websocket_connect(
                    "/api/v1/acp/sessions/test-session/stream?token=valid-token"
                ):
                    pass

        # Ensure quota slot is released on disconnect.
        with client_user_only.websocket_connect(
            "/api/v1/acp/sessions/test-session/stream?token=valid-token"
        ) as ws2:
            data = ws2.receive_json()
            assert data["type"] == "connected"


class TestACPWebSocketMessages:
    """Tests for WebSocket message handling."""

    def test_send_prompt_via_websocket(
        self, client_user_only, mock_get_runner_client, mock_jwt_manager
    ):
        """Test sending a prompt message via WebSocket."""
        with client_user_only.websocket_connect(
            "/api/v1/acp/sessions/test-session/stream?token=valid-token"
        ) as websocket:
            # Receive connected message
            connected = websocket.receive_json()
            assert connected["type"] == "connected"

            # Send prompt
            websocket.send_json({
                "type": "prompt",
                "session_id": "test-session",
                "prompt": [{"role": "user", "content": "Hello"}],
            })

            # Should receive prompt_complete message
            response = websocket.receive_json()
            assert response["type"] == "prompt_complete"
            assert response["session_id"] == "test-session"
            assert response["stop_reason"] == "end"

    def test_websocket_prompt_shadow_deny_is_not_blocked(
        self, client_user_only, mock_jwt_manager, monkeypatch
    ):
        """Shadow rollout deny decisions should not be blocked by the WS endpoint."""
        import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

        class _ShadowRunner(MockRunnerClient):
            async def check_prompt_governance(
                self,
                session_id: str,
                prompt: List[Dict],
                *,
                user_id: Optional[int] = None,
                metadata: Optional[Dict[str, Any]] = None,
            ) -> Dict[str, Any]:
                return {
                    "action": "deny",
                    "status": "deny",
                    "category": "acp",
                    "rollout_mode": "shadow",
                }

        runner = _ShadowRunner()

        async def _get_runner_client():
            return runner

        monkeypatch.setattr(acp_endpoints, "get_runner_client", _get_runner_client)

        with client_user_only.websocket_connect(
            "/api/v1/acp/sessions/test-session/stream?token=valid-token"
        ) as websocket:
            connected = websocket.receive_json()
            assert connected["type"] == "connected"

            websocket.send_json({
                "type": "prompt",
                "session_id": "test-session",
                "prompt": [{"role": "user", "content": "Hello"}],
            })

            response = websocket.receive_json()
            assert response["type"] == "prompt_complete"
            assert response["raw_result"]["detail"] == "ok"

    def test_websocket_prompt_records_prompt_in_session_store(
        self, client_user_only, mock_get_runner_client, mock_jwt_manager, monkeypatch
    ):
        """WS prompt path must persist prompt history through the ACP session store."""
        import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

        class _Store:
            def __init__(self) -> None:
                self.calls: list[tuple[str, List[Dict], Dict[str, Any]]] = []

            async def record_prompt(self, session_id: str, prompt: List[Dict], result: Dict[str, Any]):
                self.calls.append((session_id, prompt, result))
                return None

        store = _Store()

        async def _get_store():
            return store

        monkeypatch.setattr(acp_endpoints, "get_acp_session_store", _get_store)

        with client_user_only.websocket_connect(
            "/api/v1/acp/sessions/test-session/stream?token=valid-token"
        ) as websocket:
            connected = websocket.receive_json()
            assert connected["type"] == "connected"

            websocket.send_json({
                "type": "prompt",
                "session_id": "test-session",
                "prompt": [{"role": "user", "content": "Persist this"}],
            })

            response = websocket.receive_json()
            assert response["type"] == "prompt_complete"

        assert store.calls == [
            (
                "test-session",
                [{"role": "user", "content": "Persist this"}],
                {"stopReason": "end", "detail": "ok"},
            )
        ]

    def test_cancel_operation_via_websocket(
        self, client_user_only, mock_get_runner_client, mock_jwt_manager
    ):
        """Test cancelling an operation via WebSocket."""
        with client_user_only.websocket_connect(
            "/api/v1/acp/sessions/test-session/stream?token=valid-token"
        ) as websocket:
            # Receive connected message
            connected = websocket.receive_json()
            assert connected["type"] == "connected"

            # Send cancel
            websocket.send_json({
                "type": "cancel",
                "session_id": "test-session",
            })

            # Should receive update about cancellation
            response = websocket.receive_json()
            assert response["type"] == "update"
            assert response["update_type"] == "cancelled"

            # Verify cancel was called on client
            assert "test-session" in mock_get_runner_client.cancel_calls

    def test_unknown_message_type_returns_error(
        self, client_user_only, mock_get_runner_client, mock_jwt_manager
    ):
        """Test that unknown message types return an error."""
        with client_user_only.websocket_connect(
            "/api/v1/acp/sessions/test-session/stream?token=valid-token"
        ) as websocket:
            # Receive connected message
            connected = websocket.receive_json()
            assert connected["type"] == "connected"

            # Send unknown message type
            websocket.send_json({
                "type": "unknown_type",
                "data": "test",
            })

            # Should receive error
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert response["code"] == "unknown_message_type"

    def test_invalid_json_returns_error(
        self, client_user_only, mock_get_runner_client, mock_jwt_manager
    ):
        """Test that invalid JSON returns an error."""
        with client_user_only.websocket_connect(
            "/api/v1/acp/sessions/test-session/stream?token=valid-token"
        ) as websocket:
            # Receive connected message
            connected = websocket.receive_json()
            assert connected["type"] == "connected"

            # Send invalid JSON (as text)
            websocket.send_text("not valid json")

            # Should receive error
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert response["code"] == "invalid_json"


class TestACPPermissionFlow:
    """Tests for permission request/response flow."""

    def test_permission_response_approved(
        self, client_user_only, mock_get_runner_client, mock_jwt_manager
    ):
        """Test approving a permission request."""
        # Set up a pending permission
        mock_get_runner_client._pending_permissions["test-session"] = {
            "perm-123": MagicMock()
        }

        with client_user_only.websocket_connect(
            "/api/v1/acp/sessions/test-session/stream?token=valid-token"
        ) as websocket:
            # Receive connected message
            connected = websocket.receive_json()
            assert connected["type"] == "connected"

            # Send permission response (approve)
            websocket.send_json({
                "type": "permission_response",
                "request_id": "perm-123",
                "approved": True,
            })

            # Permission should be removed from pending
            assert _wait_for_permission_resolution(
                mock_get_runner_client,
                "test-session",
                "perm-123",
            )

    def test_permission_response_denied(
        self, client_user_only, mock_get_runner_client, mock_jwt_manager
    ):
        """Test denying a permission request."""
        mock_get_runner_client._pending_permissions["test-session"] = {
            "perm-456": MagicMock()
        }

        with client_user_only.websocket_connect(
            "/api/v1/acp/sessions/test-session/stream?token=valid-token"
        ) as websocket:
            connected = websocket.receive_json()
            assert connected["type"] == "connected"

            # Send permission response (deny)
            websocket.send_json({
                "type": "permission_response",
                "request_id": "perm-456",
                "approved": False,
            })

            # Permission should be removed from pending
            assert _wait_for_permission_resolution(
                mock_get_runner_client,
                "test-session",
                "perm-456",
            )

    def test_permission_response_not_found(
        self, client_user_only, mock_get_runner_client, mock_jwt_manager
    ):
        """Test permission response for non-existent request."""
        with client_user_only.websocket_connect(
            "/api/v1/acp/sessions/test-session/stream?token=valid-token"
        ) as websocket:
            connected = websocket.receive_json()
            assert connected["type"] == "connected"

            # Send permission response for non-existent request
            websocket.send_json({
                "type": "permission_response",
                "request_id": "non-existent",
                "approved": True,
            })

            # Should receive error
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert response["code"] == "permission_not_found"

    def test_permission_response_missing_request_id(
        self, client_user_only, mock_get_runner_client, mock_jwt_manager
    ):
        """Test permission response without request_id."""
        with client_user_only.websocket_connect(
            "/api/v1/acp/sessions/test-session/stream?token=valid-token"
        ) as websocket:
            connected = websocket.receive_json()
            assert connected["type"] == "connected"

            # Send permission response without request_id
            websocket.send_json({
                "type": "permission_response",
                "approved": True,
            })

            # Should receive error
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert response["code"] == "missing_request_id"


class TestACPRunnerClientPermissions:
    """Tests for the runner client permission handling."""

    @pytest.mark.asyncio
    async def test_determine_permission_tier_auto(self):
        """Test that read operations get auto tier."""
        from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import ACPRunnerClient

        # Create client with mock config
        mock_config = MagicMock()
        mock_config.command = "echo"
        mock_config.args = []
        mock_config.env = {}
        mock_config.cwd = None
        mock_config.startup_timeout_sec = 10

        client = ACPRunnerClient(mock_config)

        # Read operations should be auto
        assert client._determine_permission_tier("fs.read") == "auto"
        assert client._determine_permission_tier("git.status") == "auto"
        assert client._determine_permission_tier("search.grep") == "auto"
        assert client._determine_permission_tier("list_files") == "auto"

    @pytest.mark.asyncio
    async def test_determine_permission_tier_individual(self):
        """Test that destructive operations get individual tier."""
        from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import ACPRunnerClient

        mock_config = MagicMock()
        mock_config.command = "echo"
        mock_config.args = []
        mock_config.env = {}
        mock_config.cwd = None
        mock_config.startup_timeout_sec = 10

        client = ACPRunnerClient(mock_config)

        # Destructive operations should be individual
        assert client._determine_permission_tier("fs.delete") == "individual"
        assert client._determine_permission_tier("exec.run") == "individual"
        assert client._determine_permission_tier("git.push") == "individual"
        assert client._determine_permission_tier("terminal.execute") == "individual"

    @pytest.mark.asyncio
    async def test_determine_permission_tier_batch(self):
        """Test that write operations get batch tier."""
        from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import ACPRunnerClient

        mock_config = MagicMock()
        mock_config.command = "echo"
        mock_config.args = []
        mock_config.env = {}
        mock_config.cwd = None
        mock_config.startup_timeout_sec = 10

        client = ACPRunnerClient(mock_config)

        # Write operations should be batch
        assert client._determine_permission_tier("fs.write") == "batch"
        assert client._determine_permission_tier("git.commit") == "batch"
        assert client._determine_permission_tier("modify_file") == "batch"

    @pytest.mark.asyncio
    async def test_permission_request_message_includes_runtime_policy_metadata(self):
        mock_config = MagicMock()
        mock_config.command = "echo"
        mock_config.args = []
        mock_config.env = {}
        mock_config.cwd = None
        mock_config.startup_timeout_sec = 10

        client = ACPRunnerClient(mock_config)
        session_id = "session-policy-message"

        async def _send(_payload: dict[str, Any]) -> None:
            return None

        registry = SessionWebSocketRegistry(session_id=session_id)
        registry.websockets.add(_send)
        client._ws_registry[session_id] = registry

        async def _fake_snapshot(
            sid: str,
            *,
            force_refresh: bool = False,
        ) -> ACPRuntimePolicySnapshot | None:
            del sid, force_refresh
            return ACPRuntimePolicySnapshot(
                session_id=session_id,
                user_id=7,
                policy_snapshot_version="resolved-v1",
                policy_snapshot_fingerprint="snapshot-message",
                policy_snapshot_refreshed_at="2026-03-14T12:00:00+00:00",
                policy_summary={"approval_mode": "require_approval"},
                policy_provenance_summary={"source_kinds": ["profile"]},
                resolved_policy_document={
                    "allowed_tools": ["web.search"],
                    "approval_mode": "require_approval",
                },
                approval_summary={"mode": "require_approval"},
                context_summary={},
                execution_config={},
            )

        client._get_runtime_policy_snapshot = _fake_snapshot  # type: ignore[attr-defined]

        prompts: list[dict[str, Any]] = []

        async def _fake_broadcast(sid: str, message: dict[str, Any]) -> None:
            if message.get("type") == "permission_request":
                prompts.append(message)
                await client.respond_to_permission(sid, str(message["request_id"]), True)

        client._broadcast_to_session = _fake_broadcast  # type: ignore[method-assign]

        response = await client._handle_request(
            ACPMessage(
                jsonrpc="2.0",
                id="perm-message-1",
                method="session/request_permission",
                params={
                    "sessionId": session_id,
                    "tool": {"name": "web.search", "input": {"query": "opa"}},
                },
            )
        )

        assert response.result == {"outcome": {"outcome": "approved"}}
        assert len(prompts) == 1
        assert prompts[0]["approval_requirement"] == "approval_required"
        assert prompts[0]["provenance_summary"] == {"source_kinds": ["profile"]}
        assert prompts[0]["policy_snapshot_fingerprint"] == "snapshot-message"


class TestACPMultiplexWebSocket:
    @pytest.mark.asyncio
    async def test_multiplex_rejects_unauthenticated_client(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import tldw_Server_API.app.api.v1.endpoints.acp_multiplex as multiplex_endpoints

        async def _fake_authenticate_ws(*args: Any, **kwargs: Any) -> None:
            return None

        monkeypatch.setattr(multiplex_endpoints, "_authenticate_ws", _fake_authenticate_ws)
        ws = _FakeMultiplexWebSocket()

        await multiplex_endpoints.acp_multiplex_ws(ws, token="bad-token")

        assert ws.accepted is False
        assert ws.closed_codes == [4401]

    @pytest.mark.asyncio
    async def test_multiplex_stream_open_checks_access_and_releases_quota(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import tldw_Server_API.app.api.v1.endpoints.acp_multiplex as multiplex_endpoints

        quota_acquire_calls: list[dict[str, Any]] = []
        quota_release_calls: list[dict[str, str | None]] = []
        bus = SessionEventBus("sess-1")

        class _StubClient:
            def __init__(self) -> None:
                self.access_checks: list[tuple[str, int]] = []

            async def verify_session_access(self, session_id: str, user_id: int) -> bool:
                self.access_checks.append((session_id, user_id))
                return True

            async def get_session_metadata(
                self,
                session_id: str,
                user_id: int | None = None,
            ) -> dict[str, Any]:
                return {"persona_id": "persona-1"}

        stub_client = _StubClient()

        async def _fake_authenticate_ws(*args: Any, **kwargs: Any) -> int:
            return 7

        async def _fake_get_runner_client() -> _StubClient:
            return stub_client

        def _fake_get_session_event_bus(session_id: str) -> SessionEventBus | None:
            return bus if session_id == "sess-1" else None

        def _fake_try_acquire_quota(*, user_id: int, session_id: str, persona_id: str | None):
            quota_acquire_calls.append(
                {"user_id": user_id, "session_id": session_id, "persona_id": persona_id}
            )
            return {
                "user_key": f"user:{user_id}",
                "persona_key": f"persona:{persona_id}",
                "session_key": f"session:{session_id}",
            }, None

        def _fake_release_quota(token: dict[str, str | None] | None) -> None:
            if token is not None:
                quota_release_calls.append(token)

        monkeypatch.setattr(multiplex_endpoints, "_authenticate_ws", _fake_authenticate_ws)
        monkeypatch.setattr(multiplex_endpoints, "get_runner_client", _fake_get_runner_client)
        monkeypatch.setattr(multiplex_endpoints, "get_session_event_bus", _fake_get_session_event_bus)
        monkeypatch.setattr(multiplex_endpoints, "_acp_ws_try_acquire_quota", _fake_try_acquire_quota)
        monkeypatch.setattr(multiplex_endpoints, "_acp_ws_release_quota", _fake_release_quota)

        ws = _FakeMultiplexWebSocket(
            incoming=[multiplex_endpoints.MultiplexMessage.stream_open("sess-1").to_json()],
        )

        await multiplex_endpoints.acp_multiplex_ws(ws, token="valid-token")

        assert ws.accepted is True
        assert stub_client.access_checks == [("sess-1", 7)]
        assert quota_acquire_calls == [
            {"user_id": 7, "session_id": "sess-1", "persona_id": "persona-1"}
        ]
        assert quota_release_calls == [
            {
                "user_key": "user:7",
                "persona_key": "persona:persona-1",
                "session_key": "session:sess-1",
            }
        ]

    @pytest.mark.asyncio
    async def test_multiplex_sends_error_frame_for_handler_exception(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import tldw_Server_API.app.api.v1.endpoints.acp_multiplex as multiplex_endpoints

        lifecycle = {"started": 0, "stopped": 0}

        class _StubManager:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def start(self) -> None:
                lifecycle["started"] += 1

            async def handle_message(self, raw: str) -> None:
                raise RuntimeError("boom")

            async def stop(self) -> None:
                lifecycle["stopped"] += 1

        async def _fake_authenticate_ws(*args: Any, **kwargs: Any) -> int:
            return 1

        async def _fake_get_runner_client() -> MockRunnerClient:
            return MockRunnerClient()

        monkeypatch.setattr(multiplex_endpoints, "_authenticate_ws", _fake_authenticate_ws)
        monkeypatch.setattr(multiplex_endpoints, "get_runner_client", _fake_get_runner_client)
        monkeypatch.setattr(multiplex_endpoints, "MultiplexManager", _StubManager)

        ws = _FakeMultiplexWebSocket(
            incoming=[multiplex_endpoints.MultiplexMessage.ping().to_json()],
        )

        await multiplex_endpoints.acp_multiplex_ws(ws, token="valid-token")

        assert lifecycle == {"started": 1, "stopped": 1}
        assert len(ws.sent_text) == 1
        error_message = multiplex_endpoints.MultiplexMessage.from_json(ws.sent_text[0])
        assert error_message.type.value == "error"
