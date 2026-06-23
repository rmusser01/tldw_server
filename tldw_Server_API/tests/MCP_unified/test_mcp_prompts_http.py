from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint


class _CaptureServer:
    initialized = True

    def __init__(self) -> None:
        self.request = None

    async def handle_http_request(self, request, user_id=None, metadata=None):  # noqa: ANN001, ARG002
        self.request = request
        return type("Response", (), {"error": None, "result": {"prompts": [], "nextCursor": "next"}})()


def test_get_mcp_prompts_maps_cursor_to_protocol_request(monkeypatch) -> None:
    server = _CaptureServer()
    monkeypatch.setattr(mcp_unified_endpoint, "get_mcp_server", lambda: server)
    app = FastAPI()
    app.include_router(mcp_unified_endpoint.router, prefix="/api/v1")
    app.dependency_overrides[mcp_unified_endpoint.enforce_http_security] = lambda: None
    app.dependency_overrides[mcp_unified_endpoint.get_mcp_auth_context] = (
        lambda: mcp_unified_endpoint.McpAuthContext(
            user=None,
            principal=None,
            api_key_info=None,
            raw_api_key=None,
        )
    )
    client = TestClient(app)

    response = client.get("/api/v1/mcp/prompts?cursor=abc")

    assert response.status_code == 200  # nosec B101
    assert response.json() == {"prompts": [], "nextCursor": "next"}  # nosec B101
    assert server.request.method == "prompts/list"  # nosec B101
    assert server.request.params == {"cursor": "abc"}  # nosec B101


def test_get_mcp_prompts_preserves_empty_cursor_query(monkeypatch) -> None:
    server = _CaptureServer()
    monkeypatch.setattr(mcp_unified_endpoint, "get_mcp_server", lambda: server)
    app = FastAPI()
    app.include_router(mcp_unified_endpoint.router, prefix="/api/v1")
    app.dependency_overrides[mcp_unified_endpoint.enforce_http_security] = lambda: None
    app.dependency_overrides[mcp_unified_endpoint.get_mcp_auth_context] = (
        lambda: mcp_unified_endpoint.McpAuthContext(
            user=None,
            principal=None,
            api_key_info=None,
            raw_api_key=None,
        )
    )
    client = TestClient(app)

    response = client.get("/api/v1/mcp/prompts?cursor=")

    assert response.status_code == 200  # nosec B101
    assert response.json() == {"prompts": [], "nextCursor": "next"}  # nosec B101
    assert server.request.method == "prompts/list"  # nosec B101
    assert server.request.params == {"cursor": ""}  # nosec B101
