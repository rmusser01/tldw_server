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
    # Discriminator: if the JWT branch instead fell back to charging the IP bucket,
    # the two prior requests would have already exhausted it (rpm=1) and this would
    # also be 429. The anonymous IP bucket must be untouched by the user:42 charges.
    assert client.get("/api/v1/thing").status_code == 200


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


class _TenantSnap:
    route_map = {"by_path": {"/api/v1/*": "p"}, "by_tag": {}}
    tenant = {"enabled": True, "header": "X-TLDW-Tenant"}
    policies = {}


class _TenantLoader:
    def get_snapshot(self):
        return _TenantSnap()

    def get_policy(self, pid):
        return {"requests": {"rpm": 1, "burst": 1.0}, "scopes": ["tenant"]}


def test_tenant_scope_outranks_principal_resolution(monkeypatch):
    """Spec Sec2: tenant scoping keeps precedence over principal charging.

    A tenant-wide `scopes: [tenant]` cap must not become a per-user cap just
    because ingress can now validate the caller's credential.
    """
    monkeypatch.delenv("RG_POLICY_PATH", raising=False)

    async def fake_principal(request):
        user = request.headers.get("X-API-KEY")
        return AuthPrincipal(kind="user", user_id=int(user))

    monkeypatch.setattr(auth_principal_resolver, "get_auth_principal", fake_principal)
    monkeypatch.setattr(get_settings(), "AUTH_MODE", "multi_user")
    app = FastAPI()

    @app.get("/api/v1/thing")
    def thing() -> dict:
        return {"ok": True}

    app.add_middleware(RGSimpleMiddleware)
    app.state.rg_policy_loader = _TenantLoader()
    app.state.rg_governor = MemoryResourceGovernor(policy_loader=_TenantLoader())
    tc = TestClient(app)

    # Two different, individually-valid users in the same tenant.
    headers_a = {"X-API-KEY": "1", "X-TLDW-Tenant": "acme"}
    headers_b = {"X-API-KEY": "2", "X-TLDW-Tenant": "acme"}
    assert tc.get("/api/v1/thing", headers=headers_a).status_code == 200
    # If credential validation outranked tenant scoping, this would charge a
    # separate user:2 bucket and also return 200. It must instead share the
    # already-exhausted tenant:acme bucket (rpm=1).
    assert tc.get("/api/v1/thing", headers=headers_b).status_code == 429
