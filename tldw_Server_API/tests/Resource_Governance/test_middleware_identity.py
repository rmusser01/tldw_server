"""Ingress charges the validated principal; invalid credentials fall back to the IP bucket."""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ import auth_principal_resolver, jwt_service
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.Resource_Governance.governor import MemoryResourceGovernor
from tldw_Server_API.app.core.Resource_Governance.middleware_simple import RGSimpleMiddleware

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit]


class _Snap:
    route_map = {"by_path": {"/api/v1/*": "p"}, "by_tag": {}}
    tenant = {}
    policies = {}


class _Loader:
    def get_snapshot(self):
        return _Snap()

    def get_policy(self, pid):
        return {"requests": {"rpm": 1, "burst": 1.0}, "scopes": ["user", "api_key", "ip"]}


SESSION = get_settings().SINGLE_USER_SESSION_COOKIE_NAME


class _FakeJwt:
    def decode_access_token(self, token):
        if token != "a.valid.jwt":
            raise ValueError("bad signature")
        return {"sub": "42", "scope": "notes.read"}  # a scoped (virtual-key style) token


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("RG_POLICY_PATH", raising=False)
    calls = []

    async def fake_principal(request):
        calls.append(request.url.path)
        user = request.cookies.get(SESSION) or request.headers.get("X-API-KEY")
        if not user or not user.isdigit():
            raise HTTPException(status_code=401, detail="bad credentials")
        return AuthPrincipal(kind="user", user_id=int(user))

    monkeypatch.setattr(auth_principal_resolver, "get_auth_principal", fake_principal)
    monkeypatch.setattr(jwt_service, "get_jwt_service", lambda: _FakeJwt())
    monkeypatch.setattr(get_settings(), "AUTH_MODE", "multi_user")
    app = FastAPI()

    @app.get("/api/v1/thing")
    def thing() -> dict:
        return {"ok": True}

    app.add_middleware(RGSimpleMiddleware)
    app.state.rg_policy_loader = _Loader()
    app.state.rg_governor = MemoryResourceGovernor(policy_loader=_Loader())
    tc = TestClient(app)
    tc.principal_calls = calls
    return tc


def test_two_cookie_users_behind_one_ip_get_separate_buckets(client):
    assert client.get("/api/v1/thing", cookies={SESSION: "1"}).status_code == 200
    assert client.get("/api/v1/thing", cookies={SESSION: "1"}).status_code == 429
    assert client.get("/api/v1/thing", cookies={SESSION: "2"}).status_code == 200


def test_bearer_jwt_is_keyed_by_verified_subject_without_principal_resolution(client):
    auth = {"Authorization": "Bearer a.valid.jwt"}
    assert client.get("/api/v1/thing", headers=auth).status_code == 200
    assert client.get("/api/v1/thing", headers=auth).status_code == 429  # user:42 bucket
    assert client.principal_calls == []  # scoped-token route checks never run pre-routing


def test_invalid_jwt_charges_ip(client):
    assert client.get("/api/v1/thing", headers={"Authorization": "Bearer forged.jwt.x"}).status_code == 200
    assert client.get("/api/v1/thing").status_code == 429  # same anonymous IP bucket


def test_rotating_fake_tokens_share_the_ip_bucket(client):
    assert client.get("/api/v1/thing", headers={"X-API-KEY": "fake-a"}).status_code == 200
    assert client.get("/api/v1/thing", headers={"X-API-KEY": "fake-b"}).status_code == 429


def test_invalid_credentials_reach_the_route(client):
    # The middleware never answers 401 itself; the route's auth decides.
    assert client.get("/api/v1/thing", headers={"X-API-KEY": "fake"}).status_code == 200


def test_non_session_cookie_charges_ip_and_reaches_route(client):
    assert client.get("/api/v1/thing", cookies={"theme": "dark", "csrf_token": "t"}).status_code == 200
    assert client.principal_calls == []  # not a credential: never resolved
    assert client.get("/api/v1/thing").status_code == 429  # same anonymous IP bucket
