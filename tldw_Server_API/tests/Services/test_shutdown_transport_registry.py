from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException

from tldw_Server_API.app.services.app_lifecycle import mark_lifecycle_shutdown


pytestmark = pytest.mark.unit


class _FakeTransportSocket:
    def __init__(self) -> None:
        self.closed: list[tuple[int, str]] = []

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed.append((code, reason))


class _BlockingPromptStudioSocket:
    def __init__(
        self,
        *,
        started: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
    ) -> None:
        self.started = started
        self.release = release
        self.sent_messages: list[str] = []
        self.closed: list[tuple[int, str]] = []

    async def send_text(self, message: str) -> None:
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()
        self.sent_messages.append(message)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed.append((code, reason))


class _BlockingClosePromptStudioSocket:
    def __init__(
        self,
        *,
        started: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
    ) -> None:
        self.started = started
        self.release = release
        self.closed: list[tuple[int, str]] = []

    async def close(self, code: int = 1000, reason: str = "") -> None:
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()
        self.closed.append((code, reason))


class _FakeOuterStream:
    def __init__(self, websocket, **_: object) -> None:
        self.websocket = websocket

    async def start(self) -> None:
        await self.websocket.accept()

    async def send_json(self, payload: dict) -> None:
        self.websocket.sent_json.append(payload)

    async def error(self, code: str, message: str, *, data: dict | None = None) -> None:
        payload = {"type": "error", "code": code, "message": message}
        if data is not None:
            payload["data"] = data
        await self.send_json(payload)

    async def done(self) -> None:
        return None

    def mark_activity(self) -> None:
        return None

    async def stop(self) -> None:
        return None


class _FakeAudioWebSocket:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.query_params: dict[str, str] = {}
        self.client = SimpleNamespace(host="127.0.0.1", port=8000)
        self.accepted = 0
        self.closed: list[tuple[int, str]] = []
        self.sent_json: list[dict] = []
        self.app = FastAPI()

    async def accept(self) -> None:
        self.accepted += 1

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed.append((code, reason))

    async def send_json(self, payload: dict) -> None:
        self.sent_json.append(payload)

    async def receive_text(self) -> str:
        raise AssertionError("receive_text should not be reached while draining")

    async def receive_json(self) -> dict:
        raise AssertionError("receive_json should not be reached while draining")


class _PausingPromptStudioWebSocket(_FakeAudioWebSocket):
    def __init__(self, accepted: asyncio.Event, release_accept: asyncio.Event) -> None:
        super().__init__()
        self._accepted = accepted
        self._release_accept = release_accept

    async def accept(self) -> None:
        self.accepted += 1
        self._accepted.set()
        await self._release_accept.wait()


def test_transport_registry_snapshot_reports_active_counts() -> None:
    from tldw_Server_API.app.services.shutdown_transport_registry import ShutdownTransportRegistry

    registry = ShutdownTransportRegistry()
    registry.register_family(
        "alpha",
        active_count=lambda: 2,
        drain=None,
    )
    registry.register_family(
        "beta",
        active_count=lambda: 5,
        drain=None,
    )

    snapshot = {item.name: item for item in registry.snapshot()}

    assert snapshot["alpha"].active_count == 2
    assert snapshot["beta"].active_count == 5
    assert registry.total_active_sessions() == 7


def test_transport_registry_duplicate_registration_logs_and_replaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.services import shutdown_transport_registry as registry_module

    registry = registry_module.ShutdownTransportRegistry()
    warning_messages: list[str] = []

    monkeypatch.setattr(
        registry_module.logger,
        "warning",
        lambda message, *args, **kwargs: warning_messages.append(str(message)),
    )

    first = registry.register_family(
        "alpha",
        active_count=lambda: 1,
        drain=None,
    )
    second = registry.register_family(
        "alpha",
        active_count=lambda: 2,
        drain=None,
    )

    assert first is not second
    assert registry.get_family("alpha") is second
    assert any("alpha" in message for message in warning_messages)


@pytest.mark.asyncio
async def test_prompt_studio_transport_registry_hook_tracks_and_drains_connections() -> None:
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_websocket
    from tldw_Server_API.app.services.shutdown_transport_registry import get_shutdown_transport_registry

    first = _FakeTransportSocket()
    second = _FakeTransportSocket()
    third = _FakeTransportSocket()
    original_active = prompt_studio_websocket.connection_manager.active_connections
    original_metadata = prompt_studio_websocket.connection_manager.connection_metadata
    prompt_studio_websocket.connection_manager.active_connections = {
        "client-a": {first, second},
        "client-b": {third},
    }
    prompt_studio_websocket.connection_manager.connection_metadata = {
        first: {"client_id": "client-a"},
        second: {"client_id": "client-a"},
        third: {"client_id": "client-b"},
    }

    try:
        family = get_shutdown_transport_registry().get_family("prompt_studio.websocket")

        assert family is not None
        assert family.current_active_count() == 3

        await family.drain(timeout_s=0.2)

        assert prompt_studio_websocket.connection_manager.get_connection_count() == 0
        assert first.closed == [(1001, "Server shutdown")]
        assert second.closed == [(1001, "Server shutdown")]
        assert third.closed == [(1001, "Server shutdown")]
    finally:
        prompt_studio_websocket.connection_manager.active_connections = original_active
        prompt_studio_websocket.connection_manager.connection_metadata = original_metadata


def test_transport_registry_rejects_sync_drain_hooks() -> None:
    from tldw_Server_API.app.services.shutdown_transport_registry import ShutdownTransportRegistry

    registry = ShutdownTransportRegistry()

    def _sync_drain(timeout_s: float | None = None) -> None:
        del timeout_s

    with pytest.raises(TypeError, match="must be declared with async def"):
        registry.register_family(
            "alpha",
            active_count=lambda: 0,
            drain=_sync_drain,
        )


@pytest.mark.asyncio
async def test_transport_components_forward_runtime_timeout_budget() -> None:
    from tldw_Server_API.app.services.shutdown_transport_registry import (
        ShutdownTransportRegistry,
        build_shutdown_components,
    )

    observed_timeout_s: list[float | None] = []

    async def _drain(timeout_s: float | None = None) -> None:
        observed_timeout_s.append(timeout_s)

    registry = ShutdownTransportRegistry()
    registry.register_family(
        "alpha",
        active_count=lambda: 0,
        drain=_drain,
    )

    component = build_shutdown_components(registry, default_timeout_ms=1000)[0]
    result = component.stop(125)
    if result is not None:
        await result

    assert observed_timeout_s == [0.125]


@pytest.mark.asyncio
async def test_prompt_studio_broadcast_and_close_all_can_interleave_without_iteration_errors() -> None:
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_websocket import (
        ConnectionManager,
    )

    manager = ConnectionManager()
    started = asyncio.Event()
    release = asyncio.Event()
    first = _BlockingPromptStudioSocket(started=started, release=release)
    second = _BlockingPromptStudioSocket(started=started, release=release)
    third = _BlockingPromptStudioSocket()
    manager.active_connections = {
        "client-a": {first, second},
        "client-b": {third},
    }
    manager.connection_metadata = {
        first: {"client_id": "client-a"},
        second: {"client_id": "client-a"},
        third: {"client_id": "client-b"},
    }

    broadcast_task = asyncio.create_task(manager.broadcast_to_all("shutdown-check"))
    await started.wait()
    await manager.close_all(timeout_s=0.2)
    release.set()
    await broadcast_task

    assert manager.active_connections == {}
    assert manager.connection_metadata == {}
    assert first.closed == [(1001, "Server shutdown")]
    assert second.closed == [(1001, "Server shutdown")]
    assert third.closed == [(1001, "Server shutdown")]


@pytest.mark.asyncio
async def test_mcp_server_registers_transport_family_with_shutdown_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer
    from tldw_Server_API.app.services.shutdown_transport_registry import get_shutdown_transport_registry

    server = MCPServer()
    server.connections = {"one": object(), "two": object()}
    called: list[float | None] = []

    async def _fake_close_all_connections() -> None:
        called.append(1.0)

    monkeypatch.setattr(server, "_close_all_connections", _fake_close_all_connections)
    family = get_shutdown_transport_registry().get_family("mcp.websocket")

    assert family is not None
    assert family.current_active_count() == 2

    await family.drain(timeout_s=0.2)

    assert called == [1.0]


@pytest.mark.asyncio
@pytest.mark.parametrize("mcp_session_id", [None, "stable-session-123"])
async def test_mcp_websocket_rejects_new_connections_while_draining(
    monkeypatch: pytest.MonkeyPatch,
    mcp_session_id: str | None,
) -> None:
    from tldw_Server_API.app.core.MCP_unified import server as mcp_server_module
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    websocket = _FakeAudioWebSocket()
    mark_lifecycle_shutdown(websocket.app)
    server = MCPServer()
    server.config.ws_auth_required = False
    server.config.ws_max_connections = 100
    server.config.ws_max_connections_per_ip = 0
    server.config.ws_allowed_origins = []

    class _AllowAllController:
        def resolve_client_ip(self, raw_remote_ip, forwarded_for, real_ip):
            return "127.0.0.1"

        def is_allowed(self, ip: str) -> bool:
            return True

    class _UnexpectedStream:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("MCP websocket stream should not start while draining")

    session_calls: list[tuple[str, str | None, str | None, str | None]] = []

    async def _unexpected_get_or_create_session(
        session_id: str,
        user_id: str | None = None,
        workspace_id: str | None = None,
        cwd: str | None = None,
    ) -> object:
        session_calls.append((session_id, user_id, workspace_id, cwd))
        return object()

    monkeypatch.setattr(mcp_server_module, "get_ip_access_controller", lambda: _AllowAllController())
    monkeypatch.setattr(mcp_server_module, "enforce_client_certificate_headers", lambda *args, **kwargs: None)
    monkeypatch.setattr(mcp_server_module, "WebSocketStream", _UnexpectedStream)
    monkeypatch.setattr(server, "_get_or_create_session", _unexpected_get_or_create_session)

    await server.handle_websocket(
        websocket,
        client_id="client-a",
        mcp_session_id=mcp_session_id,
    )

    assert websocket.closed == [(1013, "shutdown_draining")]
    assert server.connections == {}
    assert session_calls == []


@pytest.mark.asyncio
async def test_prompt_studio_connect_rechecks_drain_after_accept() -> None:
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_websocket import (
        ConnectionManager,
    )

    accepted = asyncio.Event()
    release_accept = asyncio.Event()
    websocket = _PausingPromptStudioWebSocket(accepted, release_accept)
    manager = ConnectionManager()

    connect_task = asyncio.create_task(manager.connect(websocket, "global"))
    await asyncio.wait_for(accepted.wait(), timeout=1.0)
    mark_lifecycle_shutdown(websocket.app)
    release_accept.set()

    connected = await asyncio.wait_for(connect_task, timeout=1.0)

    assert connected is False
    assert websocket.accepted == 1
    assert websocket.closed == [(1013, "shutdown_draining")]
    assert manager.get_connection_count() == 0


@pytest.mark.asyncio
async def test_prompt_studio_close_all_keeps_tracking_when_cancelled_mid_close() -> None:
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_websocket import (
        ConnectionManager,
    )

    started = asyncio.Event()
    release = asyncio.Event()
    first = _BlockingClosePromptStudioSocket(started=started, release=release)
    second = _BlockingClosePromptStudioSocket()
    manager = ConnectionManager()
    manager.active_connections = {
        "client-a": {first, second},
    }
    manager.connection_metadata = {
        first: {"client_id": "client-a"},
        second: {"client_id": "client-a"},
    }

    close_task = asyncio.create_task(manager.close_all(timeout_s=0.2))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    close_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_task

    assert manager.get_connection_count() == 2
    assert len(manager.connection_metadata) == 2
    release.set()


class _UnexpectedPromptStudioStream:
    def __init__(self, *args, **kwargs) -> None:
        return None

    async def start(self) -> None:
        raise AssertionError("Prompt Studio websocket stream should not start while draining")

    async def send_json(self, payload: dict) -> None:
        return None

    def mark_activity(self) -> None:
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint_name", "kwargs"),
    [
        (
            "websocket_endpoint_base",
            {"token": None, "api_key": None, "project_id": None},
        ),
        (
            "websocket_endpoint",
            {"project_id": 7, "token": None, "api_key": None},
        ),
    ],
)
async def test_prompt_studio_websockets_reject_new_connections_while_draining(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_name: str,
    kwargs: dict,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_websocket

    websocket = _FakeAudioWebSocket()
    mark_lifecycle_shutdown(websocket.app)
    connect_calls: list[str] = []

    async def _unexpected_connect(*args, **kwargs) -> None:
        connect_calls.append("called")
        raise AssertionError("Prompt Studio connect should not run while draining")

    async def _allow_auth(*args, **kwargs) -> str:
        return "1"

    async def _allow_project(*args, **kwargs) -> bool:
        return True

    monkeypatch.setattr(prompt_studio_websocket, "WebSocketStream", _UnexpectedPromptStudioStream)
    monkeypatch.setattr(prompt_studio_websocket.connection_manager, "connect", _unexpected_connect)
    monkeypatch.setattr(
        prompt_studio_websocket,
        "_allow_prompt_studio_cookie_websocket",
        _allow_auth,
    )
    monkeypatch.setattr(
        prompt_studio_websocket,
        "_authorize_prompt_studio_project",
        _allow_project,
    )

    endpoint = getattr(prompt_studio_websocket, endpoint_name)
    await endpoint(websocket, **kwargs)

    assert connect_calls == []
    assert websocket.accepted == 1
    assert websocket.closed == [(1013, "shutdown_draining")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint_name", "expected_kind"),
    [
        ("websocket_transcribe", "audio.stream.transcribe"),
        ("websocket_audio_chat_stream", "audio.chat.stream"),
        ("websocket_tts", "audio.stream.tts"),
        ("websocket_tts_realtime", "audio.stream.tts.realtime"),
    ],
)
async def test_audio_websocket_handlers_block_new_work_when_draining(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_name: str,
    expected_kind: str,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints.audio import audio_streaming
    from tldw_Server_API.app.core.Streaming import streams as streaming_streams

    websocket = _FakeAudioWebSocket()
    mark_lifecycle_shutdown(websocket.app)
    observed_kinds: list[str] = []

    async def _fake_authenticate(*args, **kwargs):
        return True, 7

    def _fake_guard(app: FastAPI, kind: str) -> None:
        observed_kinds.append(kind)
        raise HTTPException(status_code=503, detail={"kind": kind})

    async def _unexpected_handle(*args, **kwargs):
        raise AssertionError("expensive audio work should not start while draining")

    monkeypatch.setattr(audio_streaming, "_shim_audio_ws_authenticate", _fake_authenticate)
    monkeypatch.setattr(audio_streaming, "assert_may_start_work", _fake_guard)
    monkeypatch.setattr(audio_streaming, "handle_unified_websocket", _unexpected_handle)
    monkeypatch.setattr(audio_streaming, "_stream_tts_to_websocket", _unexpected_handle)
    monkeypatch.setattr(streaming_streams, "WebSocketStream", _FakeOuterStream)

    endpoint = getattr(audio_streaming, endpoint_name)
    await endpoint(websocket, token=None)

    assert observed_kinds == [expected_kind]
    assert websocket.accepted >= 1
    assert websocket.closed


@pytest.mark.asyncio
async def test_ingest_sse_rejects_new_stream_creation_while_draining() -> None:
    from tldw_Server_API.app.api.v1.endpoints.media.ingest_jobs import (
        stream_media_ingest_job_events,
    )

    app = FastAPI()
    mark_lifecycle_shutdown(app)
    request = SimpleNamespace(app=app)
    current_user = SimpleNamespace(id=42)
    principal = SimpleNamespace(roles=[], permissions=[])
    jm = SimpleNamespace()

    with pytest.raises(HTTPException) as exc_info:
        await stream_media_ingest_job_events(
            request=request,
            batch_id=None,
            after_id=0,
            current_user=current_user,
            principal=principal,
            jm=jm,
        )

    assert exc_info.value.status_code == 503
