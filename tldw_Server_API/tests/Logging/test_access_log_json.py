import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.testclient import TestClient
from starlette.requests import Request
from starlette.responses import Response

from tldw_Server_API.app.core.Security.request_id_middleware import RequestIDMiddleware
from tldw_Server_API.app.core.Logging.access_log_middleware import AccessLogMiddleware
from loguru import logger


def _make_app() -> FastAPI:


    app = FastAPI()

    # Minimal route for access log emission
    @app.get("/ping")
    async def ping():  # pragma: no cover - simple passthrough
        # Emit a direct loguru line to validate sink capture
        logger.bind(test_marker=True).info("route ping")
        return JSONResponse({"ok": True}, status_code=200)

    # Middlewares needed for request_id propagation and access logging
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(RequestIDMiddleware)
    return app


def _capture_access_log(monkeypatch):


    from tldw_Server_API.app.core.Logging import access_log_middleware as alm_mod

    captured = []

    class _StubLogger:
        def __init__(self, extra=None):
            self._extra = extra or {}

        def bind(self, **kwargs):

            new_extra = dict(self._extra)
            new_extra.update(kwargs)
            return _StubLogger(new_extra)

        def log(self, level, message):

            captured.append({"level": level, "message": message, "extra": dict(self._extra)})

    monkeypatch.setattr(alm_mod, "logger", _StubLogger(), raising=True)
    return captured


@pytest.mark.asyncio
async def test_access_log_emits_json_with_core_fields(monkeypatch):
    # Ensure JSON logging style is conceptually enabled (for documentation parity)
    monkeypatch.setenv("LOG_JSON", "true")

    # Prepare AccessLogMiddleware instance (standalone)
    alm = AccessLogMiddleware(object())

    captured = _capture_access_log(monkeypatch)

    # Build a minimal Starlette Request with X-Request-ID
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "path": "/ping",
        "raw_path": b"/ping",
        "headers": [(b"x-request-id", b"test-request-id"), (b"host", b"testserver")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    req = Request(scope)

    async def _call_next(_req: Request) -> Response:
        return Response(content=b"{}", media_type="application/json", status_code=200)

    # Invoke middleware directly
    resp = await alm.dispatch(req, _call_next)
    assert resp.status_code == 200

    # Validate captured access-log record
    assert captured, "no access-log record captured"
    rec = captured[-1]
    extra = rec.get("extra") or {}
    assert extra.get("request_id")  # synthesized or header value
    assert extra.get("method") == "GET"
    status_val = extra.get("status")
    assert status_val == 200 or status_val == "200"
    assert extra.get("path") == "/ping"
    assert isinstance(extra.get("duration_ms"), int)


def test_access_log_uses_sanitized_request_id(monkeypatch):


    app = _make_app()
    captured = _capture_access_log(monkeypatch)
    client = TestClient(app)
    resp = client.get("/ping", headers={"X-Request-ID": "bad value"})
    assert resp.status_code == 200

    assert captured, "no access-log record captured"
    rec = captured[-1]
    extra = rec.get("extra") or {}
    logged_id = extra.get("request_id")
    response_id = resp.headers.get("X-Request-ID")
    assert logged_id
    assert response_id
    assert logged_id == response_id
    assert logged_id != "bad value"


def test_access_log_generates_request_id_when_missing(monkeypatch):


    app = _make_app()
    captured = _capture_access_log(monkeypatch)
    client = TestClient(app)
    resp = client.get("/ping")
    assert resp.status_code == 200

    assert captured, "no access-log record captured"
    rec = captured[-1]
    extra = rec.get("extra") or {}
    logged_id = extra.get("request_id")
    response_id = resp.headers.get("X-Request-ID")
    assert logged_id
    assert response_id
    assert logged_id == response_id


def test_access_log_uses_uppercase_level_names(monkeypatch):
    app = _make_app()
    captured = _capture_access_log(monkeypatch)
    client = TestClient(app)

    resp = client.get("/ping")
    assert resp.status_code == 200

    assert captured, "no access-log record captured"
    rec = captured[-1]
    assert rec.get("level") == "INFO"


@pytest.mark.asyncio
async def test_access_log_emit_failure_debug_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.Logging import access_log_middleware as alm_mod

    debug_records = []

    class _FailingLogger:
        def bind(self, **_kwargs):
            return self

        def log(self, _level, _message):
            raise RuntimeError("access log sink failed at /private/access.log")

        def debug(self, message, **kwargs):
            debug_records.append({"message": message, "kwargs": kwargs})

    monkeypatch.setattr(alm_mod, "logger", _FailingLogger(), raising=True)
    alm = AccessLogMiddleware(object())
    req = Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "path": "/ping",
            "raw_path": b"/ping",
            "headers": [(b"host", b"testserver")],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
            "scheme": "http",
        }
    )

    async def _call_next(_req: Request) -> Response:
        return Response(content=b"{}", media_type="application/json", status_code=200)

    resp = await alm.dispatch(req, _call_next)
    joined = "\n".join(f"{record['message']} {record['kwargs']}" for record in debug_records)

    assert resp.status_code == 200
    assert "Access log middleware failed to emit request log" in joined
    assert "access log sink failed" not in joined
    assert "/private/access.log" not in joined
