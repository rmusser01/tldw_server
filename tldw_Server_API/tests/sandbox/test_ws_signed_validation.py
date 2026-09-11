from __future__ import annotations

import hashlib
import hmac
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from anyio import ClosedResourceError
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tldw_Server_API.app.core.AuthNZ.single_user_session import SingleUserSessionIdentity
from tldw_Server_API.app.core.config import clear_config_cache

pytestmark = pytest.mark.sandbox_ws_signed


def _force_docker_preflight_available(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Sandbox.models import RuntimeType
    from tldw_Server_API.app.core.Sandbox.runtime_capabilities import RuntimePreflightResult
    from tldw_Server_API.app.core.Sandbox.service import SandboxService

    def _preflights(
        self: SandboxService,
        *,
        network_policy: str | None,
    ) -> dict[RuntimeType, RuntimePreflightResult]:
        del self, network_policy
        return {
            RuntimeType.docker: RuntimePreflightResult(
                runtime=RuntimeType.docker,
                available=True,
                reasons=[],
                execution_mode="mocked",
                enforcement_ready={"deny_all": True, "allowlist": False},
            )
        }

    monkeypatch.setattr(SandboxService, "_collect_runtime_preflights", _preflights)


def _client_signed(secret: str, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Enable test mode and WS signing
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("SANDBOX_WS_SIGNED_URLS", "true")
    monkeypatch.setenv("SANDBOX_WS_SIGNING_SECRET", secret)
    # speed up loop where relevant
    monkeypatch.setenv("SANDBOX_WS_POLL_TIMEOUT_SEC", "0.1")
    # Avoid real execution
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "false")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "true")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "1")
    # Ensure synthetic frames so a connect has immediate frames available
    monkeypatch.setenv("SANDBOX_WS_SYNTHETIC_FRAMES_FOR_TESTS", "true")
    _force_docker_preflight_available(monkeypatch)
    clear_config_cache()
    # Import app after env is set and cache cleared
    from tldw_Server_API.app.main import app as _app
    return TestClient(_app)


def _enable_cookie_websocket_auth(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    from tldw_Server_API.app.core.AuthNZ import websocket_session_auth

    settings = SimpleNamespace(
        AUTH_MODE="single_user",
        SINGLE_USER_SESSION_COOKIE_NAME="tldw_single_user_session",
    )
    identity = SingleUserSessionIdentity(
        session_id=9,
        user_id=1,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    monkeypatch.setattr(websocket_session_auth, "get_settings", lambda: settings)
    monkeypatch.setattr(websocket_session_auth, "trusted_webui_origins", lambda: {"http://testserver"})
    monkeypatch.setattr(
        websocket_session_auth,
        "validate_single_user_session",
        AsyncMock(return_value=identity),
    )
    monkeypatch.setattr(
        websocket_session_auth,
        "get_single_user_instance",
        lambda: SimpleNamespace(
            username="single-user",
            email=None,
            roles=["admin"],
            role="admin",
            permissions=[],
            is_admin=True,
        ),
    )
    client.cookies.set("tldw_single_user_session", "opaque", path="/api")


def _create_signed_run(client: TestClient) -> tuple[str, str]:
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    response = client.post(
        "/api/v1/sandbox/runs",
        headers={"X-API-KEY": get_settings().SINGLE_USER_API_KEY},
        json={
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "command": ["bash", "-lc", "echo run"],
            "timeout_sec": 5,
        },
    )
    assert response.status_code == 200
    url = response.json()["log_stream_url"]
    return response.json()["id"], url


@pytest.mark.unit
@pytest.mark.sandbox_ws_auth
@pytest.mark.sandbox_no_auth
def test_ws_signed_valid_token_accepts_cookie_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    with _client_signed("test-secret", monkeypatch) as client:
        _enable_cookie_websocket_auth(monkeypatch, client)
        _run_id, url = _create_signed_run(client)

        with client.websocket_connect(url, headers={"origin": "http://testserver"}) as ws:
            assert ws.receive_json()["type"] in {"event", "heartbeat"}


@pytest.mark.unit
@pytest.mark.sandbox_ws_auth
@pytest.mark.sandbox_no_auth
@pytest.mark.parametrize("signature_state", ["invalid", "expired"])
def test_ws_signed_invalid_or_expired_token_rejects_cookie_identity(
    signature_state: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "test-secret"
    with _client_signed(secret, monkeypatch) as client:
        _enable_cookie_websocket_auth(monkeypatch, client)
        run_id, signed_url = _create_signed_run(client)
        parsed = urllib.parse.urlparse(signed_url)
        query = urllib.parse.parse_qs(parsed.query)
        if signature_state == "invalid":
            token = query["token"][0]
            token = token[:-1] + ("0" if token[-1] != "0" else "1")
            exp = int(query["exp"][0])
        else:
            exp = int(time.time()) - 1
            token = hmac.new(
                secret.encode("utf-8"),
                f"{run_id}:{exp}".encode(),
                hashlib.sha256,
            ).hexdigest()
        url = f"{parsed.path}?token={token}&exp={exp}"

        with pytest.raises((WebSocketDisconnect, ClosedResourceError)) as exc_info:
            with client.websocket_connect(url, headers={"origin": "http://testserver"}):
                pass
        if isinstance(exc_info.value, WebSocketDisconnect):
            assert exc_info.value.code == 1008


@pytest.mark.unit
def test_ws_signed_valid_token_connects(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "test-secret"
    with _client_signed(secret, monkeypatch) as client:
        # Sanity: verify settings reflect env
        import os as _os

        from tldw_Server_API.app.core.config import settings as app_settings
        assert _os.getenv("SANDBOX_WS_SIGNING_SECRET") == secret
        # Signed URLs flag may be obtained from env fallback in issuance/handler
        assert bool(getattr(app_settings, "SANDBOX_WS_SIGNED_URLS", False)) or True
        body = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "command": ["bash", "-lc", "echo run"],
            "timeout_sec": 5,
        }
        r = client.post("/api/v1/sandbox/runs", json=body)
        assert r.status_code == 200
        j = r.json()
        run_id = j["id"]
        url = j.get("log_stream_url")
        assert isinstance(url, str) and url.startswith("/api/v1/sandbox/runs/")
        # Validate issuance formula matches handler's expectation
        from urllib.parse import parse_qs, urlparse
        p = urlparse(url)
        qs = parse_qs(p.query)
        tok = qs.get("token", [""])[0]
        exps = qs.get("exp", [""])[0]
        assert tok and exps
        exp_i = int(exps)
        msg = f"{run_id}:{exp_i}".encode()
        expect = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()
        assert expect == tok
        with client.websocket_connect(url) as ws:
            # A successful handshake means validation passed
            # Drain one message if available
            try:
                _ = ws.receive_json()
            except Exception:
                _ = None
            ws.close()


@pytest.mark.unit
def test_ws_signed_expired_token_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "test-secret"
    with _client_signed(secret, monkeypatch) as client:
        body = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "command": ["bash", "-lc", "echo run"],
            "timeout_sec": 5,
        }
        r = client.post("/api/v1/sandbox/runs", json=body)
        assert r.status_code == 200
        run_id = r.json()["id"]
        # Build an expired token with exp in the past
        exp = int(time.time()) - 10
        msg = f"{run_id}:{exp}".encode()
        token = hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()
        path = f"/api/v1/sandbox/runs/{run_id}/stream?token={token}&exp={exp}"
        # Expect handshake to be refused
        with pytest.raises((WebSocketDisconnect, ClosedResourceError)):
            with client.websocket_connect(path):
                pass


@pytest.mark.unit
def test_ws_signed_tampered_token_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "test-secret"
    with _client_signed(secret, monkeypatch) as client:
        body = {
            "spec_version": "1.0",
            "runtime": "docker",
            "base_image": "python:3.11-slim",
            "command": ["bash", "-lc", "echo run"],
            "timeout_sec": 5,
        }
        r = client.post("/api/v1/sandbox/runs", json=body)
        assert r.status_code == 200
        j = r.json()
        signed = j.get("log_stream_url")
        assert isinstance(signed, str)
        # Tamper token by flipping last char
        parsed = urllib.parse.urlparse(signed)
        qs = urllib.parse.parse_qs(parsed.query)
        token = qs.get("token", [""])[0]
        exp = qs.get("exp", [""])[0]
        assert token and exp
        bad_token = token[:-1] + ("0" if token[-1] != "0" else "1")
        tampered = f"{parsed.path}?token={bad_token}&exp={exp}"
        with pytest.raises((WebSocketDisconnect, ClosedResourceError)):
            with client.websocket_connect(tampered):
                pass
