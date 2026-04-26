"""Tests for the global exception handler module."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from starlette.datastructures import URL

from tldw_Server_API.app.api.v1.utils.exception_handlers import (
    _get_request_id,
    client_disconnect_handler,
    global_unhandled_exception_handler,
)


def _make_request(*, request_id: str | None = None) -> MagicMock:
    req = MagicMock()
    req.method = "GET"
    req.url = URL("http://testserver/api/v1/test?token=secret")
    headers = {"X-Request-ID": request_id} if request_id else {}
    req.headers = headers
    req.state = SimpleNamespace()
    return req


class TestGetRequestId:
    def test_returns_header_value_when_present(self):
        req = _make_request(request_id="abc-123")
        assert _get_request_id(req) == "abc-123"

    def test_generates_uuid_when_header_missing(self):
        req = _make_request()
        rid = _get_request_id(req)
        assert len(rid) == 36  # UUID format
        assert "-" in rid
        assert req.state.request_id == rid

    def test_prefers_state_request_id_over_header(self):
        req = _make_request(request_id="raw-header")
        req.state.request_id = "sanitized-state"
        assert _get_request_id(req) == "sanitized-state"


class TestGlobalUnhandledException:
    @pytest.mark.asyncio
    async def test_returns_500_with_error_envelope(self):
        import json

        req = _make_request(request_id="test-rid")
        exc = RuntimeError("boom")
        resp = await global_unhandled_exception_handler(req, exc)
        assert resp.status_code == 500
        body = json.loads(resp.body)
        assert body["detail"] == "Internal server error"
        assert body["error"]["code"] == "internal_server_error"
        assert body["error"]["request_id"] == "test-rid"
        assert resp.headers["X-Request-ID"] == "test-rid"

    @pytest.mark.asyncio
    async def test_client_disconnect_returns_499(self):
        import json

        from starlette.requests import ClientDisconnect

        req = _make_request(request_id="dc-rid")
        exc = ClientDisconnect()
        resp = await global_unhandled_exception_handler(req, exc)
        assert resp.status_code == 499
        body = json.loads(resp.body)
        assert body["detail"] == "Client disconnected"
        assert body["error"]["code"] == "client_disconnected"


class TestClientDisconnectHandler:
    @pytest.mark.asyncio
    async def test_returns_499_with_error_envelope(self):
        import json

        from starlette.requests import ClientDisconnect

        req = _make_request(request_id="cd-rid")
        exc = ClientDisconnect()
        resp = await client_disconnect_handler(req, exc)
        assert resp.status_code == 499
        body = json.loads(resp.body)
        assert body["detail"] == "Client disconnected"
        assert body["error"]["code"] == "client_disconnected"
        assert body["error"]["request_id"] == "cd-rid"
        assert resp.headers["X-Request-ID"] == "cd-rid"
