import os
from typing import Any, Dict

import pytest
from fastapi import HTTPException
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import prompt_studio_deps as deps
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User


def _make_app() -> FastAPI:


    app = FastAPI()

    @app.get("/ps/me")
    async def whoami(user_ctx: Dict[str, Any] = Depends(deps.get_prompt_studio_user)):
        return user_ctx

    return app


@pytest.fixture(autouse=True)
def _ensure_not_test_mode(monkeypatch):
    # Ensure get_prompt_studio_user exercises header-forwarding path
    monkeypatch.setenv("TEST_MODE", "false")
    yield


@pytest.mark.parametrize(
    "headers,expected_api_key,expected_token,expected_user",
    [
        ( {"X-API-KEY": "abc123"}, "abc123", None, "u-key" ),
        ( {"Authorization": "Bearer tok-xyz"}, None, "tok-xyz", "u-bearer" ),
        ( {"X-API-KEY": "abc123", "Authorization": "Bearer tok-xyz"}, "abc123", "tok-xyz", "u-bearer" ),
    ],
)
def test_get_prompt_studio_user_header_forwarding(monkeypatch, headers, expected_api_key, expected_token, expected_user):
    calls: Dict[str, Any] = {}

    async def fake_get_request_user(request, api_key=None, token=None, legacy_token_header=None):
        calls["api_key"] = api_key
        calls["token"] = token
        # Simulate auth precedence: prefer bearer token when present
        uid = "u-bearer" if token else "u-key"
        return User(id=uid, username=uid, is_active=True)

    monkeypatch.setattr(deps, "get_request_user", fake_get_request_user, raising=True)

    app = _make_app()
    client = TestClient(app)
    r = client.get("/ps/me", headers=headers)
    assert r.status_code == 200, r.text
    data = r.json()

    # Header extraction forwarded correctly
    assert calls.get("api_key") == expected_api_key
    assert calls.get("token") == expected_token
    # Resulting user context reflects fake identity (Bearer preferred when present)
    assert data.get("user_id") == expected_user


def test_get_prompt_studio_user_unauthenticated_401(monkeypatch):


    async def fake_get_request_user(*_args, **_kwargs):
        raise HTTPException(status_code=401, detail="Authentication required")

    monkeypatch.setattr(deps, "get_request_user", fake_get_request_user, raising=True)

    app = _make_app()
    client = TestClient(app)
    r = client.get("/ps/me")
    assert r.status_code == 401


def test_get_prompt_studio_user_test_mode_accepts_single_letter_y(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "y")

    app = _make_app()
    client = TestClient(app)
    r = client.get("/ps/me")

    assert r.status_code == 200, r.text
    assert r.json().get("user_id") == "test-user-123"


@pytest.mark.asyncio
async def test_prompt_studio_rate_limit_bypass_accepts_single_letter_y(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "y")

    allowed = await deps.check_rate_limit(
        operation="default",
        user_context={"user_id": "test-user"},
        security_config=deps.get_security_config(),
    )

    assert allowed is True


@pytest.mark.asyncio
async def test_prompt_studio_rate_limit_shared_limiter_failure_fails_closed(monkeypatch):
    if hasattr(deps.check_rate_limit, "_local_requests"):
        delattr(deps.check_rate_limit, "_local_requests")

    async def _raise_shared_limiter(*_args, **_kwargs):
        raise RuntimeError("shared limiter unavailable")

    monkeypatch.setattr(deps, "_authnz_check_rate_limit", _raise_shared_limiter, raising=True)
    monkeypatch.setattr(deps, "_PROMPT_STUDIO_RATE_LIMIT_SHIM_LOGGED", False, raising=True)

    sec = deps.get_security_config()
    ctx = {"user_id": "u1"}

    with pytest.raises(HTTPException) as exc_info_1:
        await deps.check_rate_limit(operation="optimize", user_context=ctx, security_config=sec)
    with pytest.raises(HTTPException) as exc_info_2:
        await deps.check_rate_limit(operation="optimize", user_context=ctx, security_config=sec)
    assert exc_info_1.value.status_code == 503
    assert exc_info_2.value.status_code == 503
    assert not hasattr(deps.check_rate_limit, "_local_requests")
