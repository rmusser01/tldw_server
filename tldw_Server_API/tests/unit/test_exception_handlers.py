"""Tests for the global exception handler module."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from tldw_Server_API.app.api.v1.utils.exception_handlers import (
    _get_request_id,
    client_disconnect_handler,
    global_unhandled_exception_handler,
)


def _make_request(*, request_id: str | None = None) -> MagicMock:
    req = MagicMock()
    req.method = "GET"
    req.url = "http://testserver/api/v1/test"
    headers = {"X-Request-ID": request_id} if request_id else {}
    req.headers = headers
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


class TestGlobalUnhandledException:
    @pytest.mark.asyncio
    async def test_returns_500_with_error_envelope(self):
        import json

        req = _make_request(request_id="test-rid")
        exc = RuntimeError("boom")
        resp = await global_unhandled_exception_handler(req, exc)
        assert resp.status_code == 500
        body = json.loads(resp.body)
        assert body["error"]["code"] == "internal_server_error"
        assert body["error"]["request_id"] == "test-rid"

    @pytest.mark.asyncio
    async def test_client_disconnect_returns_499(self):
        import json
        from starlette.requests import ClientDisconnect

        req = _make_request(request_id="dc-rid")
        exc = ClientDisconnect()
        resp = await global_unhandled_exception_handler(req, exc)
        assert resp.status_code == 499
        body = json.loads(resp.body)
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
        assert body["error"]["code"] == "client_disconnected"
        assert body["error"]["request_id"] == "cd-rid"
