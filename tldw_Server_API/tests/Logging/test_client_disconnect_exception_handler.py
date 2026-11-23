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


@pytest.mark.asyncio
async def test_the_app_answers_an_admin_refusal_and_passes_any_other_configuration_error_on():
    """TASK-15581 (P1): the app registers a ChatConfigurationError handler. An admin provider-override refusal gets its
    own 403; any other configuration error is raised on to the catch-all handler, as before."""
    from tldw_Server_API.app.core.exceptions import ChatConfigurationError
    from tldw_Server_API.app.main import app

    handler = app.exception_handlers[ChatConfigurationError]
    refusal = ChatConfigurationError("refused", provider="openai", error_code="provider_disabled")
    other = ChatConfigurationError("misconfigured", provider="openai")

    response = await handler(_build_request(request_id="req-403"), refusal)
    with pytest.raises(ChatConfigurationError) as passed_on:
        await handler(_build_request(), other)
    assert (response.status_code, passed_on.value) == (403, other)


@pytest.mark.asyncio
async def test_an_admin_refusal_answers_with_its_codes_own_message():
    """TASK-15582 (minor 3 of #3537's final review): a refusal rebuilt with another message (the privacy-safe rebuild
    says "Provider configuration is unavailable.") still answers with its code's own message."""
    from tldw_Server_API.app.core.exceptions import ChatConfigurationError
    from tldw_Server_API.app.core.LLM_Calls.provider_results import PROVIDER_STREAM_ERROR_MESSAGES

    rebuilt = ChatConfigurationError(
        "Provider configuration is unavailable.", provider="openai", error_code="provider_disabled"
    )
    response = await exception_handlers.admin_refusal_exception_handler(_build_request(request_id="req-msg"), rebuilt)
    assert (response.status_code, json.loads(response.body)["error"]["message"]) == (
        403,
        PROVIDER_STREAM_ERROR_MESSAGES["provider_disabled"],
    )


@pytest.mark.asyncio
async def test_the_app_passes_a_refusal_on_a_standalone_slides_route_to_the_catch_all():
    """TASK-15582 (minor 5 of #3537's final review): a standalone Slides route keeps its own redacted error shape, so
    the app's refusal handler raises the refusal on to the catch-all handler there."""
    from tldw_Server_API.app.core.exceptions import ChatConfigurationError
    from tldw_Server_API.app.main import app

    handler = app.exception_handlers[ChatConfigurationError]
    refusal = ChatConfigurationError("refused", provider="openai", error_code="provider_disabled")
    with pytest.raises(ChatConfigurationError) as passed_on:
        await handler(_build_request(path="/api/v1/slides/presentations/p-1"), refusal)
    assert passed_on.value is refusal
