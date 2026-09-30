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
        user = request.cookies.get(SESSION) or request.headers.get("X-API-KEY")
        calls.append(user)
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


# --- Identity cache: a repeated credential is resolved (KDF, DB) once per TTL ---


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def monotonic(self) -> float:
        return self.t


def _charged_entities(client) -> set[str]:
    return {f"{scope}:{value}" for _pid, _cat, scope, value in client.app.state.rg_governor._buckets}


def test_repeated_valid_credential_is_resolved_once(client):
    client.get("/api/v1/thing", headers={"X-API-KEY": "7"})
    client.get("/api/v1/thing", headers={"X-API-KEY": "7"})
    assert client.principal_calls == ["7"]


def _from_ip(client, ip):
    return TestClient(client.app, client=(ip, 50000))


def test_repeated_invalid_credential_is_resolved_once_and_charges_ip(client):
    caller = _from_ip(client, "10.0.0.1")
    assert caller.get("/api/v1/thing", headers={"X-API-KEY": "fake"}).status_code == 200
    assert caller.get("/api/v1/thing", headers={"X-API-KEY": "fake"}).status_code == 429  # ip bucket spent
    assert client.principal_calls == ["fake"]
    assert _charged_entities(client) == {"ip:10.0.0.1"}


def test_cached_invalid_credential_is_never_promoted_to_an_entity(client):
    # The negative entry means "no principal", not "the first caller's ip:" bucket.
    _from_ip(client, "10.0.0.1").get("/api/v1/thing", headers={"X-API-KEY": "fake"})
    _from_ip(client, "10.0.0.2").get("/api/v1/thing", headers={"X-API-KEY": "fake"})
    assert client.principal_calls == ["fake"]
    assert _charged_entities(client) == {"ip:10.0.0.1", "ip:10.0.0.2"}


def test_valid_identity_expires_after_ttl(client, monkeypatch):
    from tldw_Server_API.app.core.Resource_Governance import middleware_simple

    clock = _Clock()
    monkeypatch.setattr(middleware_simple, "time", clock, raising=False)
    client.get("/api/v1/thing", headers={"X-API-KEY": "7"})
    clock.t += 59
    client.get("/api/v1/thing", headers={"X-API-KEY": "7"})
    assert client.principal_calls == ["7"]
    clock.t += 2
    client.get("/api/v1/thing", headers={"X-API-KEY": "7"})
    assert client.principal_calls == ["7", "7"]


def test_invalid_identity_expires_after_negative_ttl(client, monkeypatch):
    from tldw_Server_API.app.core.Resource_Governance import middleware_simple

    clock = _Clock()
    monkeypatch.setattr(middleware_simple, "time", clock, raising=False)
    client.get("/api/v1/thing", headers={"X-API-KEY": "fake"})
    clock.t += 29
    client.get("/api/v1/thing", headers={"X-API-KEY": "fake"})
    assert client.principal_calls == ["fake"]
    clock.t += 2
    client.get("/api/v1/thing", headers={"X-API-KEY": "fake"})
    assert client.principal_calls == ["fake", "fake"]


def test_identity_cache_cap_evicts_oldest(client, monkeypatch):
    from tldw_Server_API.app.core.Resource_Governance import middleware_simple

    monkeypatch.setattr(middleware_simple, "_IDENTITY_CACHE_MAX", 2, raising=False)
    for key in ("1", "2", "3", "3", "1"):
        client.get("/api/v1/thing", headers={"X-API-KEY": key})
    assert client.principal_calls == ["1", "2", "3", "1"]  # "1" was evicted by "3"


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
