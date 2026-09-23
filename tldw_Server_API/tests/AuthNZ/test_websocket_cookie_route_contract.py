from __future__ import annotations

import ast
import importlib
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import WebSocket

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]

FIRST_PARTY_WEBSOCKET_AUTH_FILES = (
    "tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py",
    "tldw_Server_API/app/api/v1/endpoints/acp_multiplex.py",
    "tldw_Server_API/app/api/v1/endpoints/persona.py",
    "tldw_Server_API/app/api/v1/endpoints/watchlists.py",
    "tldw_Server_API/app/api/v1/endpoints/workflows.py",
    "tldw_Server_API/app/api/v1/endpoints/meetings.py",
    "tldw_Server_API/app/api/v1/API_Deps/Meetings_DB_Deps.py",
    "tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_websocket.py",
    "tldw_Server_API/app/api/v1/endpoints/sandbox.py",
    "tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py",
    "tldw_Server_API/app/api/v1/endpoints/voice_assistant.py",
    "tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py",
    "tldw_Server_API/app/core/Audio/streaming_service.py",
)

EXPECTED_WEBSOCKET_HANDLERS = {
    "tldw_Server_API/app/api/v1/endpoints/audio/audio_realtime.py": {"websocket_realtime"},
    "tldw_Server_API/app/api/v1/endpoints/realtime_compat.py": {"websocket_realtime_compat"},
    "tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py": {
        "acp_session_stream",
        "acp_session_ssh",
    },
    "tldw_Server_API/app/api/v1/endpoints/acp_multiplex.py": {"acp_multiplex_ws"},
    "tldw_Server_API/app/api/v1/endpoints/persona.py": {"persona_stream"},
    "tldw_Server_API/app/api/v1/endpoints/watchlists.py": {"stream_run"},
    "tldw_Server_API/app/api/v1/endpoints/workflows.py": {"workflows_ws"},
    "tldw_Server_API/app/api/v1/endpoints/meetings.py": {"stream_session_ws"},
    "tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_websocket.py": {
        "websocket_endpoint_base",
        "websocket_endpoint",
    },
    "tldw_Server_API/app/api/v1/endpoints/sandbox.py": {"stream_run_logs"},
    "tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py": {"websocket_endpoint"},
    "tldw_Server_API/app/api/v1/endpoints/voice_assistant.py": {"websocket_voice_assistant"},
    "tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py": {
        "websocket_transcribe",
        "websocket_audio_chat_stream",
        "websocket_tts",
        "websocket_tts_realtime",
    },
}


def _websocket_handlers(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    handlers: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            call = decorator if isinstance(decorator, ast.Call) else None
            func = call.func if call is not None else decorator
            if isinstance(func, ast.Attribute) and func.attr == "websocket":
                handlers.add(node.name)
    return handlers


def test_first_party_websocket_route_inventory_is_complete():
    api_root = REPO_ROOT / "tldw_Server_API/app/api/v1"
    discovered = {
        path.relative_to(REPO_ROOT).as_posix(): handlers
        for path in api_root.rglob("*.py")
        if (handlers := _websocket_handlers(path))
    }
    assert discovered == EXPECTED_WEBSOCKET_HANDLERS


def test_first_party_websocket_auth_files_use_shared_cookie_resolver():
    for relative_path in FIRST_PARTY_WEBSOCKET_AUTH_FILES:
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        if relative_path.endswith("endpoints/meetings.py"):
            assert "get_meetings_db_for_websocket" in source, relative_path
        else:
            assert "resolve_single_user_cookie_websocket" in source, relative_path


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_name", "handler_name", "path"),
    [
        ("audio.audio_realtime", "websocket_realtime", "/api/v1/audio/realtime"),
        ("realtime_compat", "websocket_realtime_compat", "/v1/realtime"),
    ],
)
@pytest.mark.parametrize("untrusted_cookie", [False, True], ids=["missing-credentials", "untrusted-cookie-origin"])
async def test_realtime_routes_enforce_shared_websocket_auth(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    handler_name: str,
    path: str,
    untrusted_cookie: bool,
) -> None:
    from tldw_Server_API.app.core.Audio import streaming_service
    from tldw_Server_API.app.core.Audio.Realtime import handler
    from tldw_Server_API.app.core.AuthNZ import settings as auth_settings
    from tldw_Server_API.app.core.AuthNZ import websocket_session_auth

    module = importlib.import_module(f"tldw_Server_API.app.api.v1.endpoints.{module_name}")
    settings = auth_settings.get_settings()
    monkeypatch.setattr(settings, "AUTH_MODE", "single_user")
    monkeypatch.setattr(streaming_service, "is_multi_user_mode", lambda: False)
    monkeypatch.setattr(websocket_session_auth, "trusted_webui_origins", lambda: {"http://localhost:3000"})
    pipeline_factory = Mock(side_effect=AssertionError("Unauthenticated realtime pipeline started"))
    monkeypatch.setattr(module, "DEFAULT_REALTIME_PIPELINE_FACTORY", pipeline_factory)
    headers = []
    if untrusted_cookie:
        headers = [
            (b"cookie", f"{settings.SINGLE_USER_SESSION_COOKIE_NAME}=untrusted-session".encode()),
            (b"origin", b"https://untrusted.example"),
        ]
    send = AsyncMock()
    receive = AsyncMock(side_effect=AssertionError("Realtime auth consumed protocol data"))
    websocket = WebSocket(
        scope={
            "type": "websocket",
            "path": path,
            "headers": headers,
            "query_string": b"",
            "client": ("127.0.0.1", 12345),
        },
        receive=receive,
        send=send,
    )

    assert module.handle_realtime_websocket is handler.handle_realtime_websocket
    await getattr(module, handler_name)(websocket)

    send.assert_awaited_once_with(
        {"type": "websocket.close", "code": 4403 if untrusted_cookie else 4401, "reason": ""},
    )
    receive.assert_not_awaited()
    pipeline_factory.assert_not_called()
