import pytest
import os

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Security import middleware as security_middleware
from tldw_Server_API.app.core.Security.middleware import SecurityHeadersMiddleware


def _capture_security_middleware_logs():
    messages: list[str] = []
    sink_id = security_middleware.logger.add(
        lambda message: messages.append(str(message)),
        level="DEBUG",
    )
    return messages, sink_id


@pytest.fixture(scope="module")
def app_with_security_headers():
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/ping")
    async def ping():
        return {"status": "ok"}

    return app


def test_security_headers_applied(app_with_security_headers):


    client = TestClient(app_with_security_headers)
    response = client.get("/ping")

    assert response.status_code == 200
    headers = response.headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert "default-src 'self'" in headers["Content-Security-Policy"]
    assert "geolocation=()" in headers["Permissions-Policy"]
    assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert headers["X-Permitted-Cross-Domain-Policies"] == "none"
    assert "Server" not in headers
    assert "Strict-Transport-Security" not in headers


def test_hsts_applied_when_https_forwarded(monkeypatch):


    monkeypatch.setenv("SECURITY_ENABLE_HSTS", "true")

    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/ping")
    async def ping():
        return {"status": "ok"}

    client = TestClient(app)
    response = client.get("/ping", headers={"X-Forwarded-Proto": "https"})
    assert response.headers["Strict-Transport-Security"].startswith("max-age=31536000")


def test_hsts_disabled_via_env(monkeypatch):


    monkeypatch.setenv("SECURITY_ENABLE_HSTS", "false")

    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/ping")
    async def ping():
        return {"status": "ok"}

    client = TestClient(app)
    response = client.get("/ping", headers={"X-Forwarded-Proto": "https"})
    assert "Strict-Transport-Security" not in response.headers


def test_metric_increment_failure_log_is_sanitized(monkeypatch):
    class FailingRegistry:
        def increment(self, _name, _amount):
            raise RuntimeError("metrics backend failed at /private/security-metrics.sock")

    monkeypatch.setattr(
        security_middleware,
        "get_metrics_registry",
        lambda: FailingRegistry(),
    )

    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/ping")
    async def ping():
        return {"status": "ok"}

    messages, sink_id = _capture_security_middleware_logs()
    try:
        response = TestClient(app).get("/ping")
    finally:
        security_middleware.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert response.status_code == 200
    assert "Security headers metric increment failed" in joined
    assert "metrics backend failed" not in joined
    assert "/private/security-metrics.sock" not in joined
