"""Ingress charges the validated principal; invalid credentials fall back to the IP bucket."""

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ import auth_principal_resolver, jwt_service
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
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


def test_invalid_credentials_reach_the_route(monkeypatch):
    # The middleware never answers 401 itself; the route's own auth dependency
    # (get_request_user, same as real app routes) decides. RG's own identity
    # resolution (for bucket charging) stays faked, as in the other tests here;
    # only the route's auth is real, and it must be the thing returning 401.
    monkeypatch.delenv("RG_POLICY_PATH", raising=False)
    monkeypatch.delenv("SINGLE_USER_TEST_API_KEY", raising=False)

    async def no_rg_identity(request):
        """Stand in for RG's own (unrelated) identity resolution: always miss, so RG
        charges the IP bucket and never blocks the request before it reaches the route."""
        return None

    monkeypatch.setattr(auth_principal_resolver, "get_auth_principal", no_rg_identity)
    settings = get_settings()
    monkeypatch.setattr(settings, "AUTH_MODE", "single_user")
    monkeypatch.setattr(settings, "SINGLE_USER_API_KEY", "the-configured-real-key")

    class _GenerousLoader(_Loader):
        def get_policy(self, pid):
            """Return a high-rpm/burst policy so RG itself never denies here; this test is
            about the route's own auth dependency, not RG's rate limiting."""
            return {"requests": {"rpm": 1000, "burst": 10.0}, "scopes": ["user", "api_key", "ip"]}

    app = FastAPI()

    @app.get("/api/v1/thing")
    def thing(user=Depends(get_request_user)) -> dict:
        """Real route guarded by the same get_request_user dependency production routes
        use, so an invalid credential is rejected by real auth, not a hand-rolled stub."""
        return {"ok": True}

    app.add_middleware(RGSimpleMiddleware)
    app.state.rg_policy_loader = _GenerousLoader()
    app.state.rg_governor = MemoryResourceGovernor(policy_loader=_GenerousLoader())
    tc = TestClient(app)

    assert tc.get("/api/v1/thing", headers={"X-API-KEY": "fake"}).status_code == 401
    # Sanity: the real dependency isn't a blanket 401; a matching key reaches 200.
    assert tc.get("/api/v1/thing", headers={"X-API-KEY": "the-configured-real-key"}).status_code == 200


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
    assert _charged_entities(client) == {"ip:10.0.0.1", "ip:10.0.0.2"}


def test_cached_identity_is_per_client_ip(client):
    # validate_api_key enforces per-key allowed_ips, so another IP is another resolution.
    _from_ip(client, "10.0.0.1").get("/api/v1/thing", headers={"X-API-KEY": "7"})
    _from_ip(client, "10.0.0.2").get("/api/v1/thing", headers={"X-API-KEY": "7"})
    assert client.principal_calls == ["7", "7"]


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


def test_invalid_jwt_falls_back_to_api_key_like_route_auth(client):
    # get_auth_principal drops a failed bearer JWT for X-API-KEY, so ingress must too.
    client.get("/api/v1/thing", headers={"Authorization": "Bearer forged.jwt.x", "X-API-KEY": "7"})
    assert _charged_entities(client) == {"user:7"}


def test_cached_identity_is_per_authnz_client_ip(client, monkeypatch):
    # API-key allowed_ips checks AuthNZ's own client IP, which can differ from RG's.
    from tldw_Server_API.app.core.AuthNZ import ip_allowlist

    authnz_ips = iter(["10.0.0.1", "10.0.0.2"])
    monkeypatch.setattr(ip_allowlist, "resolve_client_ip", lambda _request, _settings=None: next(authnz_ips))
    client.get("/api/v1/thing", headers={"X-API-KEY": "7"})
    client.get("/api/v1/thing", headers={"X-API-KEY": "7"})
    assert client.principal_calls == ["7", "7"]


# --- Per-IP resolution budget: rotating credentials cannot buy a KDF per request ---


@pytest.fixture
def budget_of_two(monkeypatch):
    from tldw_Server_API.app.core.Resource_Governance import middleware_simple

    monkeypatch.setattr(middleware_simple, "_IDENTITY_RESOLVE_BUDGET_PER_MIN", 2, raising=False)


def test_resolution_budget_spent_charges_ip_without_resolving(client, budget_of_two):
    caller = _from_ip(client, "10.0.0.1")
    for key in ("fake-a", "fake-b", "3"):
        caller.get("/api/v1/thing", headers={"X-API-KEY": key})
    assert client.principal_calls == ["fake-a", "fake-b"]
    assert _charged_entities(client) == {"ip:10.0.0.1"}  # "3" is valid but was never resolved


def test_resolution_budget_is_per_ip(client, budget_of_two):
    for key in ("fake-a", "fake-b", "fake-c"):
        _from_ip(client, "10.0.0.1").get("/api/v1/thing", headers={"X-API-KEY": key})
    _from_ip(client, "10.0.0.2").get("/api/v1/thing", headers={"X-API-KEY": "4"})
    assert client.principal_calls == ["fake-a", "fake-b", "4"]
    assert "user:4" in _charged_entities(client)


def test_resolution_budget_resets_each_window(client, budget_of_two, monkeypatch):
    from tldw_Server_API.app.core.Resource_Governance import middleware_simple

    clock = _Clock()
    monkeypatch.setattr(middleware_simple, "time", clock, raising=False)
    for key in ("fake-a", "fake-b", "fake-c"):
        client.get("/api/v1/thing", headers={"X-API-KEY": key})
    clock.t += 60
    client.get("/api/v1/thing", headers={"X-API-KEY": "fake-d"})
    assert client.principal_calls == ["fake-a", "fake-b", "fake-d"]


def test_cache_hits_do_not_spend_resolution_budget(client, budget_of_two):
    for key in ("7", "7", "7", "7", "8"):
        client.get("/api/v1/thing", headers={"X-API-KEY": key})
    assert client.principal_calls == ["7", "8"]


def test_bearer_jwt_is_resolved_even_when_the_budget_is_spent(client, budget_of_two):
    # The JWT check is signature-only (no DB, no key derivation), so fake-key floods that
    # spend an IP's budget must not push JWT (WebUI) users into the shared IP bucket.
    caller = _from_ip(client, "10.0.0.1")
    for key in ("fake-a", "fake-b"):
        caller.get("/api/v1/thing", headers={"X-API-KEY": key})
    caller.get("/api/v1/thing", headers={"Authorization": "Bearer a.valid.jwt"})
    assert "user:42" in _charged_entities(client)


class _TenantSnap:
    route_map = {"by_path": {"/api/v1/*": "p"}, "by_tag": {}}
    tenant = {"enabled": True, "header": "X-TLDW-Tenant"}
    policies = {}


class _TenantLoader:
    def get_snapshot(self):
        return _TenantSnap()

    def get_policy(self, pid):
        return {"requests": {"rpm": 1, "burst": 1.0}, "scopes": ["tenant"]}


def _tenant_client(monkeypatch, principals=None, jwt_payload=None, tenant=True):
    """Ingress over a tenant-scoped (rpm=1) policy; principals maps X-API-KEY -> AuthPrincipal."""
    monkeypatch.delenv("RG_POLICY_PATH", raising=False)
    calls = []

    async def fake_principal(request):
        key = request.headers.get("X-API-KEY")
        calls.append(key)
        if key not in (principals or {}):
            raise HTTPException(status_code=401, detail="bad credentials")
        return principals[key]

    class _Jwt:
        def decode_access_token(self, token):
            return dict(jwt_payload or {})

    monkeypatch.setattr(auth_principal_resolver, "get_auth_principal", fake_principal)
    monkeypatch.setattr(jwt_service, "get_jwt_service", lambda: _Jwt())
    monkeypatch.setattr(get_settings(), "AUTH_MODE", "multi_user")
    loader = _TenantLoader() if tenant else _Loader()
    app = FastAPI()

    @app.get("/api/v1/thing")
    def thing() -> dict:
        return {"ok": True}

    app.add_middleware(RGSimpleMiddleware)
    app.state.rg_policy_loader = loader
    app.state.rg_governor = MemoryResourceGovernor(policy_loader=loader)
    tc = TestClient(app, client=("10.0.0.1", 50000))
    tc.principal_calls = calls
    return tc


def _member(user_id, *org_ids, active=None):
    return AuthPrincipal(kind="user", user_id=user_id, org_ids=list(org_ids), active_org_id=active)


def test_tenant_scope_outranks_principal_resolution(monkeypatch):
    """Spec Sec2: tenant scoping keeps precedence over principal charging.

    A tenant-wide `scopes: [tenant]` cap must not become a per-user cap just
    because ingress can now validate the caller's credential.
    """
    tc = _tenant_client(monkeypatch, {"1": _member(1, 7), "2": _member(2, 7)})

    # Two different, individually-valid users in the same tenant (org 7).
    assert tc.get("/api/v1/thing", headers={"X-API-KEY": "1", "X-TLDW-Tenant": "7"}).status_code == 200
    # If credential validation outranked tenant scoping, this would charge a
    # separate user:2 bucket and also return 200. It must instead share the
    # already-exhausted tenant:7 bucket (rpm=1).
    assert tc.get("/api/v1/thing", headers={"X-API-KEY": "2", "X-TLDW-Tenant": "7"}).status_code == 429


def test_rotating_tenant_header_on_one_ip_shares_one_bucket(monkeypatch):
    # TASK-13402: an unvalidated header never names a bucket; anonymous callers pay their IP.
    tc = _tenant_client(monkeypatch)
    assert tc.get("/api/v1/thing", headers={"X-TLDW-Tenant": "acme"}).status_code == 200
    assert tc.get("/api/v1/thing", headers={"X-TLDW-Tenant": "globex"}).status_code == 429
    assert tc.get("/api/v1/thing", headers={"X-TLDW-Tenant": "initech"}).status_code == 429
    assert _charged_entities(tc) == {"ip:10.0.0.1"}


def test_invalid_credential_with_tenant_header_charges_ip(monkeypatch):
    tc = _tenant_client(monkeypatch)
    tc.get("/api/v1/thing", headers={"X-API-KEY": "fake", "X-TLDW-Tenant": "acme"})
    assert _charged_entities(tc) == {"ip:10.0.0.1"}


def test_header_for_a_foreign_tenant_charges_the_principals_own_tenant(monkeypatch):
    tc = _tenant_client(monkeypatch, {"1": _member(1, 7, active=7)})
    tc.get("/api/v1/thing", headers={"X-API-KEY": "1", "X-TLDW-Tenant": "99"})
    assert _charged_entities(tc) == {"tenant:7"}


def test_header_for_a_foreign_tenant_charges_a_principal_without_tenant(monkeypatch):
    tc = _tenant_client(monkeypatch, {"1": _member(1)})
    tc.get("/api/v1/thing", headers={"X-API-KEY": "1", "X-TLDW-Tenant": "99"})
    assert _charged_entities(tc) == {"user:1"}


def test_header_for_a_member_tenant_selects_it(monkeypatch):
    tc = _tenant_client(monkeypatch, {"1": _member(1, 7, 8)})
    tc.get("/api/v1/thing", headers={"X-API-KEY": "1", "X-TLDW-Tenant": "8"})
    assert _charged_entities(tc) == {"tenant:8"}


def test_cached_identity_still_validates_each_tenant_header(monkeypatch):
    # The header is not part of the identity cache key, so the cache must keep the orgs.
    tc = _tenant_client(monkeypatch, {"1": _member(1, 7, 8)})
    for tenant_header in ("7", "8", "99"):
        tc.get("/api/v1/thing", headers={"X-API-KEY": "1", "X-TLDW-Tenant": tenant_header})
    assert tc.principal_calls == ["1"]
    assert _charged_entities(tc) == {"tenant:7", "tenant:8", "user:1"}


def test_jwt_tenant_comes_from_verified_org_claims(monkeypatch):
    tc = _tenant_client(monkeypatch, jwt_payload={"sub": "42", "org_ids": [7], "active_org_id": 7})
    auth = {"Authorization": "Bearer a.valid.jwt"}
    tc.get("/api/v1/thing", headers={**auth, "X-TLDW-Tenant": "99"})
    assert _charged_entities(tc) == {"tenant:7"}
    assert tc.principal_calls == []


def test_tenant_header_is_ignored_when_tenant_scoping_is_disabled(monkeypatch):
    tc = _tenant_client(monkeypatch, {"1": _member(1, 7, active=7)}, tenant=False)
    tc.get("/api/v1/thing", headers={"X-API-KEY": "1", "X-TLDW-Tenant": "7"})
    assert _charged_entities(tc) == {"user:1"}
