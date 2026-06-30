import json

import pytest
from starlette.requests import ClientDisconnect, Request

from tldw_Server_API.app.api.v1.utils import exception_handlers
from tldw_Server_API.app.main import (
    _client_disconnect_exception_handler,
    _global_unhandled_exception_handler,
)


def _build_request(
    method: str = "GET",
    path: str = "/api/v1/mcp/health",
    *,
    headers: list[tuple[bytes, bytes]] | None = None,
    query_string: bytes = b"",
    request_id: str | None = None,
) -> Request:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": query_string,
        "headers": headers or [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }

    async def _receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, _receive)
    if request_id is not None:
        request.state.request_id = request_id
    return request


class _StubLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def opt(self, **kwargs):
        return self

    def error(self, message: str, **kwargs) -> None:
        self.calls.append(("error", message, kwargs))

    def debug(self, message: str, **kwargs) -> None:
        self.calls.append(("debug", message, kwargs))


@pytest.mark.asyncio
async def test_client_disconnect_exception_handler_returns_499():
    request = _build_request(request_id="req-499")

    response = await _client_disconnect_exception_handler(request, ClientDisconnect())

    assert response.status_code == 499
    assert json.loads(response.body.decode("utf-8")) == {
        "error": {
            "code": "client_disconnected",
            "message": "Client disconnected",
            "request_id": "req-499",
        }
    }


@pytest.mark.asyncio
async def test_global_unhandled_exception_handler_treats_client_disconnect_as_499():
    request = _build_request(method="POST", request_id="req-499-post")

    response = await _global_unhandled_exception_handler(request, ClientDisconnect())

    assert response.status_code == 499
    assert json.loads(response.body.decode("utf-8")) == {
        "error": {
            "code": "client_disconnected",
            "message": "Client disconnected",
            "request_id": "req-499-post",
        }
    }


@pytest.mark.asyncio
async def test_global_handler_uses_request_state_request_id_before_header(monkeypatch):
    stub_logger = _StubLogger()
    monkeypatch.setattr(exception_handlers, "logger", stub_logger, raising=True)
    request = _build_request(
        headers=[(b"x-request-id", b"header-id")],
        request_id="state-id",
    )

    response = await exception_handlers.global_unhandled_exception_handler(
        request,
        RuntimeError("boom"),
    )

    assert response.status_code == 500
    assert json.loads(response.body.decode("utf-8")) == {
        "error": {
            "code": "internal_server_error",
            "message": "Internal server error",
            "request_id": "state-id",
        }
    }
    assert stub_logger.calls
    level, _, kwargs = stub_logger.calls[-1]
    assert level == "error"
    assert kwargs["rid"] == "state-id"


@pytest.mark.asyncio
async def test_exception_handler_logs_sanitized_path_without_query_string(monkeypatch):
    stub_logger = _StubLogger()
    monkeypatch.setattr(exception_handlers, "logger", stub_logger, raising=True)
    request = _build_request(
        path="/api/v1/chat/completions",
        query_string=b"api_key=secret-token&mode=debug",
        request_id="req-sanitized",
    )

    await exception_handlers.global_unhandled_exception_handler(request, RuntimeError("boom"))
    await exception_handlers.client_disconnect_handler(request, ClientDisconnect())

    assert len(stub_logger.calls) == 2
    for _, _, kwargs in stub_logger.calls:
        assert kwargs["path"] == "/api/v1/chat/completions"
        assert "url" not in kwargs
        assert "secret-token" not in repr(kwargs)
