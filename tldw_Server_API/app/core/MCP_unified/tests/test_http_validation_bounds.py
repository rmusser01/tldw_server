import asyncio
import os
import tempfile

import pytest
from fastapi.testclient import TestClient


def _run(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def _setup_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")
    monkeypatch.setenv("PROFILE", "single_user")
    monkeypatch.setenv("SINGLE_USER_ALLOWED_IPS", "")
    monkeypatch.setenv("MCP_JWT_SECRET", "x" * 64)
    monkeypatch.setenv("MCP_API_KEY_SALT", "s" * 64)
    monkeypatch.setenv("MCP_ENABLE_MEDIA_MODULE", "false")
    monkeypatch.setenv("MCP_MODULES_CONFIG", os.path.join(
        tempfile.gettempdir(),
        "mcp_modules_empty.yaml",
    ))
    monkeypatch.setenv("MCP_MODULES", (
        "media=tldw_Server_API.app.core.MCP_unified.modules.implementations.media_module:MediaModule,"
        "chats=tldw_Server_API.app.core.MCP_unified.modules.implementations.chats_module:ChatsModule,"
        "characters=tldw_Server_API.app.core.MCP_unified.modules.implementations.characters_module:CharactersModule"
    ))
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
        reset_settings()
    except Exception:
        _ = None


@pytest.fixture(scope="module")
def client():
    monkeypatch = pytest.MonkeyPatch()
    _setup_env(monkeypatch)
    from tldw_Server_API.app.core.MCP_unified import get_mcp_server
    from tldw_Server_API.app.core.MCP_unified.config import get_config
    from tldw_Server_API.app.core.MCP_unified.server import reset_mcp_server
    from tldw_Server_API.app.core.MCP_unified.tests.support import (
        build_mcp_admin_auth_override,
        build_mcp_test_client,
    )

    try:
        get_config.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None

    _run(reset_mcp_server())

    try:
        with build_mcp_test_client(auth_principal_override=build_mcp_admin_auth_override()) as c:
            class _AllowAll:
                async def check_permission(self, *args, **kwargs):
                    return True

            get_mcp_server().protocol.rbac_policy = _AllowAll()
            yield c
    finally:
        monkeypatch.undo()
        get_config.cache_clear()  # type: ignore[attr-defined]
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
        reset_settings()
        _run(reset_mcp_server())


def _auth_headers():
    return {"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}


def test_media_get_chars_per_token_zero_returns_400(client: TestClient):
    resp = client.post(
        "/api/v1/mcp/tools/execute",
        json={
            "tool_name": "media.get",
            "arguments": {
                "media_id": 1,
                "retrieval": {"chars_per_token": 0},
            },
        },
        headers=_auth_headers(),
    )
    assert resp.status_code == 400
    detail = resp.json().get("detail")
    assert "chars_per_token" in str(detail)


def test_chats_get_chars_per_token_zero_returns_400(client: TestClient):
    resp = client.post(
        "/api/v1/mcp/tools/execute",
        json={
            "tool_name": "chats.get",
            "arguments": {
                "conversation_id": "c1",
                "retrieval": {"chars_per_token": 0},
            },
        },
        headers=_auth_headers(),
    )
    assert resp.status_code == 400
    detail = resp.json().get("detail")
    assert "chars_per_token" in str(detail)


def test_media_search_enforces_bounds(client: TestClient):
    resp = client.post(
        "/api/v1/mcp/tools/execute",
        json={
            "tool_name": "media.search",
            "arguments": {"query": "x", "limit": 0},
        },
        headers=_auth_headers(),
    )
    assert resp.status_code == 400
    detail = resp.json().get("detail")
    assert "limit" in str(detail)


def test_characters_search_enforces_bounds(client: TestClient):
    resp = client.post(
        "/api/v1/mcp/tools/execute",
        json={
            "tool_name": "characters.search",
            "arguments": {"query": "x", "snippet_length": 10},
        },
        headers=_auth_headers(),
    )
    assert resp.status_code == 400
    detail = resp.json().get("detail")
    assert "snippet_length" in str(detail)
