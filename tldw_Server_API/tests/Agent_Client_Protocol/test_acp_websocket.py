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
from fastapi import HTTPException, WebSocketDisconnect, WebSocketException
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import (
    ACPRunnerClient,
    PendingPermission,
    SessionWebSocketRegistry,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.event_bus import SessionEventBus
from tldw_Server_API.app.core.Agent_Client_Protocol.events import AgentEvent, AgentEventKind
from tldw_Server_API.app.core.Agent_Client_Protocol.stdio_client import ACPMessage
from tldw_Server_API.app.core.AuthNZ import websocket_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services.acp_runtime_policy_service import (
    ACPRuntimePolicySnapshot,
)

pytestmark = pytest.mark.unit

STREAM = "/api/v1/acp/sessions/test-session/stream"
AUTH = {"Authorization": "Bearer valid-token"}  # what fake_verifier admits as user 1


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

    async def prompt(self, session_id: str, prompt: List[Dict], *, on_native_text=None) -> Dict[str, Any]:
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

    async def accept(self, subprotocol: str | None = None) -> None:
        self.accepted = True

    async def close(self, code: int = 1000) -> None:
        self.closed_codes.append(code)

    async def receive_text(self) -> str:
        if self._incoming:
            return self._incoming.pop(0)
        raise WebSocketDisconnect(code=1000)

    async def send_text(self, raw: str) -> None:
        self.sent_text.append(raw)


class _LoggerStub:
    def __init__(self) -> None:
        self.warning_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.exception_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.info_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warning_records.append((message, args, kwargs))

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.exception_records.append((message, args, kwargs))

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.info_records.append((message, args, kwargs))


def _render_log_records(records: list[tuple[str, tuple[Any, ...], dict[str, Any]]]) -> str:
    rendered: list[str] = []
    for message, args, kwargs in records:
        try:
            rendered.append(message.format(*args, **kwargs))
        except Exception:
            rendered.append(message)
        rendered.extend(repr(arg) for arg in args)
        rendered.extend(f"{key}={value!r}" for key, value in kwargs.items())
    return "\n".join(rendered)


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
def fake_verifier(monkeypatch):
    """Stand in for HTTP's get_auth_principal behind resolve_ws_principal.

    "Bearer valid-token" and the API key "test-api-key" are user 1; anything else is a 401.
    Returns the (Authorization, X-API-KEY) pairs it was asked to verify, in order.
    """
    seen: list[tuple[str | None, str | None]] = []

    async def _get_auth_principal(request):
        credential = (request.headers.get("authorization"), request.headers.get("x-api-key"))
        seen.append(credential)
        if credential in {("Bearer valid-token", None), (None, "test-api-key")}:
            return AuthPrincipal(kind="user", user_id=1)
        raise HTTPException(status_code=401)

    monkeypatch.setattr(websocket_principal, "get_auth_principal", _get_auth_principal)
    return seen


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
        headers: dict[str, str] = {}

        async def accept(self, subprotocol: str | None = None) -> None:
            return None

        async def close(self, code: int = 1000) -> None:
            return None

    async def _fake_authenticate_ws(websocket, *, required_scope):
        return 1

    async def _fake_get_runner_client():
        return stub_client

    monkeypatch.setattr(acp_endpoints, "_authenticate_ws", _fake_authenticate_ws)
    monkeypatch.setattr(acp_endpoints, "get_runner_client", _fake_get_runner_client)
    monkeypatch.setattr(acp_endpoints, "WebSocketStream", _FailingStream)

    await acp_endpoints.acp_session_stream(
        _FakeWebSocket(),
        "session-start-failure",
    )

    if stub_client.unregister_calls:
        pytest.fail("Expected no unregister call when stream start fails before callback assignment")
    if lifecycle["stop_calls"] != 1:
        pytest.fail(f"Expected exactly one stream.stop() call, got {lifecycle['stop_calls']}")


async def test_acp_session_stream_reconnect_cleans_up_replay_broadcaster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reconnect replay broadcasters must not leave event-bus subscribers behind."""
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

    session_id = "session-reconnect-cleanup"
    bus = SessionEventBus(session_id=session_id)
    await bus.publish(
        AgentEvent(
            session_id=session_id,
            kind=AgentEventKind.COMPLETION,
            payload={"detail": "replay-me"},
        )
    )
    stub_client = MockRunnerClient()
    release_calls: list[str] = []
    streams: list[Any] = []

    class _DisconnectingStream:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.sent_payloads: list[dict[str, Any]] = []
            self.stop_calls = 0
            streams.append(self)

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            self.stop_calls += 1

        async def send_json(self, payload: dict[str, Any]) -> None:
            self.sent_payloads.append(payload)

        async def receive_json(self) -> dict[str, Any]:
            raise WebSocketDisconnect(code=1000)

    class _FakeWebSocket:
        headers: dict[str, str] = {}

        async def accept(self, subprotocol: str | None = None) -> None:
            return None

        async def close(self, code: int = 1000) -> None:
            return None

    async def _fake_authenticate_ws(websocket: Any, *, required_scope: str) -> int:
        return 1

    async def _fake_get_runner_client() -> MockRunnerClient:
        return stub_client

    async def _fake_resolve_persona_id(client: Any, *, session_id: str, user_id: int) -> None:
        return None

    def _fake_acquire_quota(
        *, user_id: int, session_id: str, persona_id: str | None
    ) -> tuple[str, None]:
        return "quota-token", None

    def _fake_release_quota(token: str) -> None:
        release_calls.append(token)

    monkeypatch.setattr(acp_endpoints, "_authenticate_ws", _fake_authenticate_ws)
    monkeypatch.setattr(acp_endpoints, "get_runner_client", _fake_get_runner_client)
    monkeypatch.setattr(acp_endpoints, "get_session_event_bus", lambda _: bus)
    monkeypatch.setattr(acp_endpoints, "_resolve_acp_session_persona_id", _fake_resolve_persona_id)
    monkeypatch.setattr(acp_endpoints, "_acp_ws_try_acquire_quota", _fake_acquire_quota)
    monkeypatch.setattr(acp_endpoints, "_acp_ws_release_quota", _fake_release_quota)
    monkeypatch.setattr(acp_endpoints, "WebSocketStream", _DisconnectingStream)

    await acp_endpoints.acp_session_stream(
        _FakeWebSocket(),
        session_id,
        last_sequence=1,
    )

    assert bus._subscribers == {}
    assert not stub_client.has_websocket_connections(session_id)
    assert release_calls == ["quota-token"]
    assert streams and streams[0].stop_calls == 1
    replayed_payloads = [
        payload
        for payload in streams[0].sent_payloads
        if payload.get("kind") == AgentEventKind.COMPLETION.value
    ]
    assert len(replayed_payloads) == 1
    assert replayed_payloads[0]["session_id"] == session_id
    assert replayed_payloads[0]["sequence"] == 1
    assert replayed_payloads[0]["payload"] == {"detail": "replay-me"}


class TestACPWebSocketConnection:
    """Tests for WebSocket connection handling."""

    def test_websocket_connect_with_bearer_header(
        self, client_user_only, mock_get_runner_client, fake_verifier
    ):
        """Test successful WebSocket connection with a valid bearer header."""
        with client_user_only.websocket_connect(STREAM, headers=AUTH) as websocket:
            # Should receive connected message
            data = websocket.receive_json()
            assert data["type"] == "connected"
            assert data["session_id"] == "test-session"
            assert "agent_capabilities" in data
        assert fake_verifier == [("Bearer valid-token", None)]

    def test_websocket_connect_with_subprotocol_bearer(
        self, client_user_only, mock_get_runner_client, fake_verifier
    ):
        """The bearer subprotocol is verified and only its marker is echoed (browsers need the echo)."""
        with client_user_only.websocket_connect(STREAM, subprotocols=["bearer", "valid-token"]) as websocket:
            assert websocket.accepted_subprotocol == "bearer"
            data = websocket.receive_json()
            assert data["type"] == "connected"
            assert data["session_id"] == "test-session"
        assert fake_verifier == [("Bearer valid-token", None)]

    @pytest.mark.parametrize(
        ("query", "forwarded"),
        [("token=valid-token", ("Bearer valid-token", None)), ("api_key=test-api-key", (None, "test-api-key"))],
    )
    def test_deprecated_query_credential_goes_to_the_same_verifier(
        self, client_user_only, mock_get_runner_client, fake_verifier, query, forwarded
    ):
        """Shipped clients still send ?token= / ?api_key=; the shim forwards them, unparsed (TASK-13950.7)."""
        with client_user_only.websocket_connect(f"{STREAM}?{query}") as websocket:
            assert websocket.accepted_subprotocol is None
            assert websocket.receive_json()["type"] == "connected"
        assert fake_verifier == [forwarded]

    def test_deprecated_x_api_key_subprotocol_goes_to_the_same_verifier(
        self, client_user_only, mock_get_runner_client, fake_verifier
    ):
        """The ACP SSH panel offers ``x-api-key, <key>``; echo the marker, never the key."""
        with client_user_only.websocket_connect(STREAM, subprotocols=["x-api-key", "test-api-key"]) as websocket:
            assert websocket.accepted_subprotocol == "x-api-key"
            assert websocket.receive_json()["type"] == "connected"
        assert fake_verifier == [(None, "test-api-key")]

    def test_a_header_credential_bypasses_the_deprecated_shim(
        self, client_user_only, mock_get_runner_client, fake_verifier
    ):
        with client_user_only.websocket_connect(f"{STREAM}?token=revoked-token", headers=AUTH) as websocket:
            assert websocket.receive_json()["type"] == "connected"
        assert fake_verifier == [("Bearer valid-token", None)]

    def test_websocket_connect_with_revoked_token_fails(
        self, client_user_only, mock_get_runner_client, fake_verifier
    ):
        """Revoked/blacklisted JWT tokens should be rejected, on the deprecated query too."""
        with pytest.raises(WebSocketDisconnect) as rejection:
            with client_user_only.websocket_connect(f"{STREAM}?token=revoked-token"):
                pytest.fail("A revoked query token was accepted")
        assert rejection.value.code == 4401
        assert fake_verifier == [("Bearer revoked-token", None)]

    def test_rejected_bearer_header_is_verified_once_and_closes_4401(
        self, client_user_only, mock_get_runner_client, fake_verifier
    ):
        """A rejected header bearer was re-read and re-verified until RecursionError (TASK-13950.7)."""
        with pytest.raises(WebSocketDisconnect) as rejection:
            with client_user_only.websocket_connect(STREAM, headers={"Authorization": "Bearer revoked-token"}):
                pytest.fail("A revoked bearer was accepted")

        assert rejection.value.code == 4401
        assert fake_verifier == [("Bearer revoked-token", None)]

    @pytest.mark.parametrize(
        ("headers", "subprotocols"),
        [({"Authorization": "X-API-KEY test-api-key"}, None), ({}, ["api-key", "test-api-key"])],
        ids=["authorization-x-api-key", "api-key-subprotocol"],
    )
    def test_retired_credential_channels_are_refused(
        self, client_user_only, mock_get_runner_client, fake_verifier, headers, subprotocols
    ):
        """No client sent these; they are removed, not deprecated (TASK-13950.7)."""
        with pytest.raises(WebSocketDisconnect) as rejection:
            with client_user_only.websocket_connect(STREAM, headers=headers, subprotocols=subprotocols):
                pytest.fail("A retired credential channel was accepted")
        assert rejection.value.code == 4401

    @pytest.mark.parametrize("path", ["stream", "ssh"])
    def test_websocket_rejects_read_only_api_key_in_multi_user_mode(
        self, client_user_only, mock_get_runner_client, monkeypatch, path
    ):
        """ACP control and SSH sockets must require write scope for multi-user API keys."""

        async def _read_scope_key(request):
            request.state._api_key_scope = "read"
            return AuthPrincipal(kind="api_key", user_id=1, api_key_id=3)

        monkeypatch.setattr(websocket_principal, "get_auth_principal", _read_scope_key)

        with pytest.raises(WebSocketDisconnect) as rejection:
            with client_user_only.websocket_connect(
                f"/api/v1/acp/sessions/test-session/{path}", headers={"X-API-KEY": "read-key"}
            ):
                pytest.fail("A read-scope key opened an ACP socket")
        assert rejection.value.code == 4403

    def test_websocket_connect_uses_the_resolved_user_id(
        self, client_user_only, mock_get_runner_client, monkeypatch
    ):
        """Session access is checked for the principal's user, e.g. a SINGLE_USER_FIXED_ID of 42."""

        async def _user_42(request):
            return AuthPrincipal(kind="user", user_id=42)

        monkeypatch.setattr(websocket_principal, "get_auth_principal", _user_42)
        mock_get_runner_client.expected_user_id = 42

        with client_user_only.websocket_connect(STREAM, headers=AUTH) as websocket:
            data = websocket.receive_json()
            assert data["type"] == "connected"
        assert ("test-session", 42) in mock_get_runner_client.access_checks

    def test_websocket_ssh_echoes_the_x_api_key_subprotocol(
        self, client_user_only, mock_get_runner_client, fake_verifier, monkeypatch
    ):
        """The SSH panel offers ``x-api-key, <key>`` and a browser fails a handshake that echoes nothing."""

        async def _ssh_info(session_id, user_id=None):
            return ("127.0.0.1", 22, "agent", "not-a-key")

        def _no_ssh(*args, **kwargs):
            raise ConnectionError("no sandbox in this test")

        mock_get_runner_client.get_ssh_info = _ssh_info
        monkeypatch.setitem(sys.modules, "asyncssh", types.SimpleNamespace(import_private_key=str, connect=_no_ssh))

        with client_user_only.websocket_connect(
            "/api/v1/acp/sessions/test-session/ssh", subprotocols=["x-api-key", "test-api-key"]
        ) as websocket:
            assert websocket.accepted_subprotocol == "x-api-key"
            assert websocket.receive()["code"] == 1011  # the fake sandbox refused, after accept
        assert fake_verifier == [(None, "test-api-key")]

    def test_websocket_connect_without_session_access_fails(
        self, client_user_only, mock_get_runner_client, fake_verifier
    ):
        """Connection should be rejected when the user does not own the session."""
        mock_get_runner_client.expected_user_id = 999
        with pytest.raises(Exception):
            with client_user_only.websocket_connect(
                STREAM, headers=AUTH
            ):
                pass

    def test_websocket_connect_without_auth_fails(
        self, client_user_only, mock_get_runner_client, fake_verifier
    ):
        """Test that WebSocket connection without authentication closes with 4401."""
        with pytest.raises(WebSocketDisconnect) as rejection:
            with client_user_only.websocket_connect(STREAM):
                pytest.fail("An unauthenticated socket was accepted")
        assert rejection.value.code == 4401
        assert fake_verifier == []

    def test_websocket_connection_quota_per_user_enforced(
        self, client_user_only, mock_get_runner_client, fake_verifier, monkeypatch
    ):
        """Second concurrent stream from same user should be rejected when per-user cap is reached."""
        monkeypatch.setenv("ACP_WS_MAX_CONNECTIONS_PER_USER", "1")

        with client_user_only.websocket_connect(
            STREAM, headers=AUTH
        ) as ws1:
            data = ws1.receive_json()
            assert data["type"] == "connected"

            with pytest.raises(Exception):
                with client_user_only.websocket_connect(
                    STREAM, headers=AUTH
                ):
                    pass

        # Ensure quota slot is released on disconnect.
        with client_user_only.websocket_connect(
            STREAM, headers=AUTH
        ) as ws2:
            data = ws2.receive_json()
            assert data["type"] == "connected"


class TestACPWebSocketMessages:
    """Tests for WebSocket message handling."""

    def test_send_prompt_via_websocket(
        self, client_user_only, mock_get_runner_client, fake_verifier
    ):
        """Test sending a prompt message via WebSocket."""
        with client_user_only.websocket_connect(
            STREAM, headers=AUTH
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
        self, client_user_only, fake_verifier, monkeypatch
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
            STREAM, headers=AUTH
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
        self, client_user_only, mock_get_runner_client, fake_verifier, monkeypatch
    ):
        """WS prompt path must persist prompt history through the ACP session store."""
        import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

        class _Store:
            def __init__(self) -> None:
                self.calls: list[tuple[str, List[Dict], Dict[str, Any]]] = []

            async def record_prompt(self, session_id: str, prompt: List[Dict], result: Dict[str, Any], *, assistant_text=None):
                if assistant_text is not None:
                    raise AssertionError("This legacy fixture emits no native text")
                self.calls.append((session_id, prompt, result))
                return None

        store = _Store()

        async def _get_store():
            return store

        monkeypatch.setattr(acp_endpoints, "get_acp_session_store", _get_store)

        with client_user_only.websocket_connect(
            STREAM, headers=AUTH
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
        self, client_user_only, mock_get_runner_client, fake_verifier
    ):
        """Test cancelling an operation via WebSocket."""
        with client_user_only.websocket_connect(
            STREAM, headers=AUTH
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
        self, client_user_only, mock_get_runner_client, fake_verifier
    ):
        """Test that unknown message types return an error."""
        with client_user_only.websocket_connect(
            STREAM, headers=AUTH
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
        self, client_user_only, mock_get_runner_client, fake_verifier
    ):
        """Test that invalid JSON returns an error."""
        with client_user_only.websocket_connect(
            STREAM, headers=AUTH
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
        self, client_user_only, mock_get_runner_client, fake_verifier
    ):
        """Test approving a permission request."""
        # Set up a pending permission
        mock_get_runner_client._pending_permissions["test-session"] = {
            "perm-123": MagicMock()
        }

        with client_user_only.websocket_connect(
            STREAM, headers=AUTH
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
        self, client_user_only, mock_get_runner_client, fake_verifier
    ):
        """Test denying a permission request."""
        mock_get_runner_client._pending_permissions["test-session"] = {
            "perm-456": MagicMock()
        }

        with client_user_only.websocket_connect(
            STREAM, headers=AUTH
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
        self, client_user_only, mock_get_runner_client, fake_verifier
    ):
        """Test permission response for non-existent request."""
        with client_user_only.websocket_connect(
            STREAM, headers=AUTH
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
        self, client_user_only, mock_get_runner_client, fake_verifier
    ):
        """Test permission response without request_id."""
        with client_user_only.websocket_connect(
            STREAM, headers=AUTH
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
        assert client._determine_permission_tier("fs.write") == "individual"
        assert client._determine_permission_tier("exec.run") == "individual"
        assert client._determine_permission_tier("git.push") == "individual"
        assert client._determine_permission_tier("modify_file") == "individual"
        assert client._determine_permission_tier("terminal.execute") == "individual"

    @pytest.mark.asyncio
    async def test_determine_permission_tier_batch(self):
        """Batch tier is the fallback for tools that are neither destructive nor read-only.

        Write/modify operations were promoted to the stricter "individual"
        tier by the ACP runner security hardening (see permission_tiers.py);
        this test previously pinned the pre-hardening behavior.
        """
        from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import ACPRunnerClient

        mock_config = MagicMock()
        mock_config.command = "echo"
        mock_config.args = []
        mock_config.env = {}
        mock_config.cwd = None
        mock_config.startup_timeout_sec = 10

        client = ACPRunnerClient(mock_config)

        # Write/modify operations are individual tier post-hardening
        assert client._determine_permission_tier("fs.write") == "individual"
        assert client._determine_permission_tier("modify_file") == "individual"

        # Tools matching neither the individual nor auto token sets fall back to batch
        assert client._determine_permission_tier("git.commit") == "batch"
        assert client._determine_permission_tier("fs.chmod") == "batch"

    @pytest.mark.asyncio
    async def test_permission_request_message_includes_runtime_policy_metadata(self, admitted_permission_prompt):
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

        await admitted_permission_prompt(client, session_id)
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

        async def _fake_authenticate_ws(*args: Any, **kwargs: Any) -> int:
            raise WebSocketException(code=4401, reason="authentication_required")

        monkeypatch.setattr(multiplex_endpoints, "_authenticate_ws", _fake_authenticate_ws)
        ws = _FakeMultiplexWebSocket()

        await multiplex_endpoints.acp_multiplex_ws(ws)

        assert ws.accepted is False
        assert ws.closed_codes == [4401]

    @pytest.mark.asyncio
    async def test_multiplex_sanitizes_runner_client_unavailable_log(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import tldw_Server_API.app.api.v1.endpoints.acp_multiplex as multiplex_endpoints

        logger_stub = _LoggerStub()

        async def _fake_authenticate_ws(*args: Any, **kwargs: Any) -> int:
            return 7

        async def _fake_get_runner_client() -> MockRunnerClient:
            raise RuntimeError("runner leaked /private/acp.sock")

        monkeypatch.setattr(multiplex_endpoints, "logger", logger_stub)
        monkeypatch.setattr(multiplex_endpoints, "_authenticate_ws", _fake_authenticate_ws)
        monkeypatch.setattr(multiplex_endpoints, "get_runner_client", _fake_get_runner_client)

        ws = _FakeMultiplexWebSocket()

        await multiplex_endpoints.acp_multiplex_ws(ws)

        joined = _render_log_records(logger_stub.warning_records)

        assert ws.accepted is False
        assert ws.closed_codes == [4404]
        assert "Multiplex WS runner client unavailable" in joined
        assert "runner leaked" not in joined
        assert "/private/" not in joined

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

        await multiplex_endpoints.acp_multiplex_ws(ws)

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
    async def test_multiplex_sanitizes_access_check_failure_log(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import tldw_Server_API.app.api.v1.endpoints.acp_multiplex as multiplex_endpoints

        logger_stub = _LoggerStub()
        bus = SessionEventBus("secret-session")

        async def _fake_authenticate_ws(*args: Any, **kwargs: Any) -> int:
            return 7

        async def _fake_get_runner_client() -> MockRunnerClient:
            return MockRunnerClient()

        def _fake_get_session_event_bus(session_id: str) -> SessionEventBus | None:
            return bus if session_id == "secret-session" else None

        async def _fake_require_session_access(*args: Any, **kwargs: Any) -> None:
            raise TimeoutError("authorize leaked /private/acp-access.log")

        monkeypatch.setattr(multiplex_endpoints, "logger", logger_stub)
        monkeypatch.setattr(multiplex_endpoints, "_authenticate_ws", _fake_authenticate_ws)
        monkeypatch.setattr(multiplex_endpoints, "get_runner_client", _fake_get_runner_client)
        monkeypatch.setattr(multiplex_endpoints, "get_session_event_bus", _fake_get_session_event_bus)
        monkeypatch.setattr(multiplex_endpoints, "_require_session_access", _fake_require_session_access)

        ws = _FakeMultiplexWebSocket(
            incoming=[multiplex_endpoints.MultiplexMessage.stream_open("secret-session").to_json()],
        )

        await multiplex_endpoints.acp_multiplex_ws(ws)

        error_message = multiplex_endpoints.MultiplexMessage.from_json(ws.sent_text[0])
        joined = _render_log_records(logger_stub.warning_records)

        assert error_message.payload == {"error": "Unable to authorize stream"}
        assert "Multiplex WS access check failed" in joined
        assert "secret-session" not in joined
        assert "authorize leaked" not in joined
        assert "/private/" not in joined

    @pytest.mark.asyncio
    async def test_multiplex_sends_error_frame_for_handler_exception(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import tldw_Server_API.app.api.v1.endpoints.acp_multiplex as multiplex_endpoints

        logger_stub = _LoggerStub()
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

        monkeypatch.setattr(multiplex_endpoints, "logger", logger_stub)
        monkeypatch.setattr(multiplex_endpoints, "_authenticate_ws", _fake_authenticate_ws)
        monkeypatch.setattr(multiplex_endpoints, "get_runner_client", _fake_get_runner_client)
        monkeypatch.setattr(multiplex_endpoints, "MultiplexManager", _StubManager)

        ws = _FakeMultiplexWebSocket(
            incoming=[multiplex_endpoints.MultiplexMessage.ping().to_json()],
        )

        await multiplex_endpoints.acp_multiplex_ws(ws)

        assert lifecycle == {"started": 1, "stopped": 1}
        assert len(ws.sent_text) == 1
        error_message = multiplex_endpoints.MultiplexMessage.from_json(ws.sent_text[0])
        assert error_message.type.value == "error"
        assert logger_stub.exception_records == [("Multiplex WS handler failure", (), {})]
