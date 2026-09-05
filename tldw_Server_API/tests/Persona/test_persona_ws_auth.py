import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep


pytestmark = pytest.mark.unit


fastapi_app = FastAPI()
fastapi_app.include_router(persona_ep.router, prefix="/api/v1/persona")


class _FakeWebSocket:
    def __init__(self, headers: dict[str, str]):
        self.headers = headers
        self.client = SimpleNamespace(host="127.0.0.1", port=54321)


@pytest.mark.asyncio
async def test_resolve_authenticated_user_id_single_user_bearer_api_key_path(monkeypatch):
    ws = _FakeWebSocket({"authorization": "Bearer test_api_key"})

    monkeypatch.setattr(persona_ep, "get_settings", lambda: SimpleNamespace(AUTH_MODE="single_user"))
    monkeypatch.setattr(persona_ep, "resolve_client_ip", lambda *_args, **_kwargs: "127.0.0.1")

    async def _verify_jwt_should_not_run(*_args, **_kwargs):
        raise AssertionError("JWT verification should not run for single-user bearer API key path")

    class _ApiKeyManager:
        async def validate_api_key(
            self,
            api_key: str,
            ip_address: str | None = None,
            required_scope: str | None = None,
            **_kwargs,
        ):
            assert api_key == "test_api_key"
            assert ip_address == "127.0.0.1"
            assert required_scope in (None, "", "read")
            return {"user_id": 42}

    async def _fake_get_api_key_manager():
        return _ApiKeyManager()

    monkeypatch.setattr(persona_ep, "verify_jwt_and_fetch_user", _verify_jwt_should_not_run)
    monkeypatch.setattr(persona_ep, "get_api_key_manager", _fake_get_api_key_manager)

    user_id, credentials_supplied, auth_ok = await persona_ep._resolve_authenticated_user_id(
        ws,
        token=None,
        api_key=None,
    )

    assert user_id == "42"
    assert credentials_supplied is True
    assert auth_ok is True


@pytest.mark.asyncio
async def test_resolve_authenticated_user_id_multi_user_non_jwt_bearer_api_key_path(monkeypatch):
    ws = _FakeWebSocket({"authorization": "Bearer not-a-jwt-token"})

    monkeypatch.setattr(persona_ep, "get_settings", lambda: SimpleNamespace(AUTH_MODE="multi_user"))
    monkeypatch.setattr(persona_ep, "resolve_client_ip", lambda *_args, **_kwargs: "127.0.0.1")

    async def _verify_jwt_should_not_run(*_args, **_kwargs):
        raise AssertionError("JWT verification should not run for non-JWT bearer token path")

    class _ApiKeyManager:
        async def validate_api_key(
            self,
            api_key: str,
            ip_address: str | None = None,
            required_scope: str | None = None,
            **_kwargs,
        ):
            assert api_key == "not-a-jwt-token"
            assert ip_address == "127.0.0.1"
            assert required_scope in (None, "", "read")
            return {"user_id": "api-user"}

    async def _fake_get_api_key_manager():
        return _ApiKeyManager()

    monkeypatch.setattr(persona_ep, "verify_jwt_and_fetch_user", _verify_jwt_should_not_run)
    monkeypatch.setattr(persona_ep, "get_api_key_manager", _fake_get_api_key_manager)

    user_id, credentials_supplied, auth_ok = await persona_ep._resolve_authenticated_user_id(
        ws,
        token=None,
        api_key=None,
    )

    assert user_id == "api-user"
    assert credentials_supplied is True
    assert auth_ok is True


@pytest.mark.asyncio
async def test_resolve_authenticated_user_id_multi_user_jwt_prefers_jwt_verification(monkeypatch):
    ws = _FakeWebSocket({"authorization": "Bearer header.payload.signature"})

    monkeypatch.setattr(persona_ep, "get_settings", lambda: SimpleNamespace(AUTH_MODE="multi_user"))

    async def _verify_jwt(_request, token: str):
        assert token == "header.payload.signature"
        return SimpleNamespace(id="jwt-user")

    class _ApiKeyManager:
        async def validate_api_key(self, *_args, **_kwargs):
            raise AssertionError("API key manager should not be used when JWT verification succeeds")

    async def _fake_get_api_key_manager():
        return _ApiKeyManager()

    monkeypatch.setattr(persona_ep, "verify_jwt_and_fetch_user", _verify_jwt)
    monkeypatch.setattr(persona_ep, "get_api_key_manager", _fake_get_api_key_manager)

    user_id, credentials_supplied, auth_ok = await persona_ep._resolve_authenticated_user_id(
        ws,
        token=None,
        api_key=None,
    )

    assert user_id == "jwt-user"
    assert credentials_supplied is True
    assert auth_ok is True


@pytest.mark.parametrize("credentials_supplied", [False, True])
def test_persona_stream_auth_failure_rejects_before_stream_start(monkeypatch, credentials_supplied: bool):
    monkeypatch.setattr(persona_ep, "is_persona_enabled", lambda: True)

    async def _reject_auth(*_args, **_kwargs):
        return None, credentials_supplied, False

    started = False

    async def _record_start(_self):
        nonlocal started
        started = True

    monkeypatch.setattr(persona_ep, "_resolve_authenticated_user_id", _reject_auth)
    monkeypatch.setattr(persona_ep.WebSocketStream, "start", _record_start, raising=True)

    with TestClient(fastapi_app) as client:
        try:
            with client.websocket_connect("/api/v1/persona/stream") as ws:
                with pytest.raises(WebSocketDisconnect):
                    ws.receive_text()
        except WebSocketDisconnect:
            pass

    assert started is False


@pytest.mark.parametrize("marker", [None, "bearer", "Bearer", "unsupported"])
@pytest.mark.parametrize("valid_credential", [True, False])
def test_persona_stream_negotiates_only_safe_auth_marker(monkeypatch, marker, valid_credential):
    monkeypatch.setattr(persona_ep, "is_persona_enabled", lambda: True)
    monkeypatch.setattr(persona_ep, "get_settings", lambda: SimpleNamespace(AUTH_MODE="single_user"))
    monkeypatch.setattr(persona_ep, "resolve_client_ip", lambda *_args, **_kwargs: "127.0.0.1")

    async def _verify_jwt_should_not_run(*_args, **_kwargs):
        raise AssertionError("JWT verification should not run for single-user bearer API key path")

    class _ApiKeyManager:
        async def validate_api_key(
            self,
            api_key: str,
            ip_address: str | None = None,
            required_scope: str | None = None,
            **_kwargs,
        ):
            if api_key == "test_api_key":
                assert required_scope in (None, "", "read")
                return {"user_id": 99}
            return None

    async def _fake_get_api_key_manager():
        return _ApiKeyManager()

    monkeypatch.setattr(persona_ep, "verify_jwt_and_fetch_user", _verify_jwt_should_not_run)
    monkeypatch.setattr(persona_ep, "get_api_key_manager", _fake_get_api_key_manager)

    credential = "test_api_key" if valid_credential else "invalid_api_key"
    headers = {}
    protocols = []
    selected_protocol = None
    if marker in ("bearer", "Bearer"):
        protocols = [marker, credential]
        selected_protocol = marker
    else:
        headers = {"Authorization": f"Bearer {credential}"}
        if marker:
            protocols = [marker, credential]

    with TestClient(fastapi_app) as client:
        if not valid_credential:
            with pytest.raises(WebSocketDisconnect) as rejected:
                with client.websocket_connect(
                    "/api/v1/persona/stream", headers=headers, subprotocols=protocols
                ):
                    pytest.fail("Invalid credentials must not complete a handshake")
            assert rejected.value.code == 4401
            return
        with client.websocket_connect(
            "/api/v1/persona/stream",
            headers=headers,
            subprotocols=protocols,
        ) as ws:
            assert ws.accepted_subprotocol == selected_protocol
            first_event = json.loads(ws.receive_text())
            assert first_event.get("event") == "notice"
            assert "connected" in str(first_event.get("message", "")).lower()
