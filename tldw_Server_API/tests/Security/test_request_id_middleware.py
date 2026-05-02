import uuid

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Security import request_id_middleware as request_id_middleware_module
from tldw_Server_API.app.core.Security.request_id_middleware import RequestIDMiddleware, _clean_request_id


class _CapturingLogger:
    def __init__(self):
        self.records = []

    def error(self, message, *args, **kwargs):
        self.records.append(("error", message, args, dict(kwargs)))

    def exception(self, message, *args, **kwargs):
        self.records.append(("exception", message, args, dict(kwargs)))


def _joined_records(logger: _CapturingLogger) -> str:
    return "\n".join(f"{level} {message} {args!r} {kwargs!r}" for level, message, args, kwargs in logger.records)


@pytest.fixture(scope="module")
def app_with_request_id():
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/ping")
    async def ping(request: Request):
        return {"request_id": request.state.request_id}

    return app


def test_request_id_preserves_clean_value(app_with_request_id):


    client = TestClient(app_with_request_id)
    req_id = "abc-123.DEF"
    resp = client.get("/ping", headers={"X-Request-ID": req_id})
    assert resp.status_code == 200
    assert resp.json()["request_id"] == req_id
    assert resp.headers["X-Request-ID"] == req_id


def test_request_id_rejects_malicious_value(app_with_request_id):


    client = TestClient(app_with_request_id)
    raw = "aaa\nbbb"
    resp = client.get("/ping", headers={"X-Request-ID": raw})
    assert resp.status_code == 200
    generated = resp.headers["X-Request-ID"]
    assert "\n" not in generated
    assert generated != raw
    uuid.UUID(generated)  # Raises if not valid UUID


def test_request_id_rejects_excessive_length(app_with_request_id):


    client = TestClient(app_with_request_id)
    oversized = "a" * 1024
    resp = client.get("/ping", headers={"X-Request-ID": oversized})
    assert resp.status_code == 200
    generated = resp.headers["X-Request-ID"]
    assert len(generated) < len(oversized)
    uuid.UUID(generated)


def test_clean_request_id_generates_when_missing():


    assert uuid.UUID(_clean_request_id(None))


def test_tracing_baggage_failure_log_is_sanitized(app_with_request_id, monkeypatch):
    import tldw_Server_API.app.core.Metrics.traces as traces_module

    logger = _CapturingLogger()

    def _raise_get_tracing_manager():
        raise RuntimeError("trace baggage failed at /private/traces.sock")

    monkeypatch.setattr(request_id_middleware_module, "logger", logger)
    monkeypatch.setattr(traces_module, "get_tracing_manager", _raise_get_tracing_manager)

    client = TestClient(app_with_request_id)
    response = client.get(
        "/ping",
        headers={
            "X-Request-ID": "req-tracing-sanitized",
            "X-Session-ID": "sess-tracing-sanitized",
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-tracing-sanitized"
    assert response.headers["X-Session-ID"] == "sess-tracing-sanitized"
    joined = _joined_records(logger)
    assert "Failed to set tracing baggage" in joined
    assert "trace baggage failed" not in joined
    assert "/private/traces.sock" not in joined
