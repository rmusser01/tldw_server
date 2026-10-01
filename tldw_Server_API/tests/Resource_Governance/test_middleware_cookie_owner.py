"""Cookie ingress must reserve against its validated owner's existing quota."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ import auth_principal_resolver as resolver
from tldw_Server_API.app.core.AuthNZ.single_user_session import SingleUserSessionIdentity
from tldw_Server_API.app.core.Resource_Governance.governor import MemoryResourceGovernor
from tldw_Server_API.app.core.Resource_Governance.middleware_simple import RGSimpleMiddleware

# unit is the primary classification; rate_limit is a registered feature marker.
pytestmark = [pytest.mark.unit, pytest.mark.rate_limit]

GovernedCookieApp = tuple[FastAPI, list[str | None], MemoryResourceGovernor]


@pytest.fixture
def governed_cookie_app(monkeypatch: pytest.MonkeyPatch) -> GovernedCookieApp:
    """Build governed ingress with real quotas and stubbed cookie validation."""
    settings = SimpleNamespace(AUTH_MODE="single_user", SINGLE_USER_SESSION_COOKIE_NAME="custom_session")
    monkeypatch.setattr(resolver, "get_settings", lambda: settings)
    from tldw_Server_API.app.core.AuthNZ import settings as settings_module

    monkeypatch.setattr(settings_module, "get_settings", lambda: settings)
    validations: list[str | None] = []

    async def validate(request: Request) -> SingleUserSessionIdentity | None:
        token = request.cookies.get("custom_session")
        validations.append(token)
        if token not in {"session-a", "session-b"}:
            return None
        return SingleUserSessionIdentity(1, 1, datetime.now(timezone.utc) + timedelta(days=1))

    monkeypatch.setattr(resolver, "validate_single_user_session", validate)
    app = FastAPI()
    app.add_middleware(RGSimpleMiddleware)
    # Use the actual policy, but a frozen clock keeps quota exhaustion deterministic.
    path = Path(__file__).resolve().parents[2] / "Config_Files/resource_governor_policies.yaml"
    data = yaml.safe_load(path.read_text())
    policy = data["policies"]["character_chat.default"]
    snapshot = SimpleNamespace(route_map={"by_path": {"/api/v1/persona/*": "character_chat.default"}}, tenant={})
    loader = SimpleNamespace(get_snapshot=lambda: snapshot, get_policy=lambda _: policy)
    governor = MemoryResourceGovernor(policy_loader=loader, time_source=lambda: 100.0)
    app.state.rg_policy_loader = loader
    app.state.rg_governor = governor

    @app.get("/api/v1/persona/profiles")
    def profiles(principal=Depends(resolver.get_auth_principal)):
        return {"user_id": principal.user_id}

    @app.get("/ungoverned")
    def ungoverned():
        return {"ok": True}

    return app, validations, governor


async def test_cookie_sessions_share_owner_quota_and_cached_auth(governed_cookie_app: GovernedCookieApp) -> None:
    app, validations, governor = governed_cookie_app
    # character_chat.default is now rpm=300/burst=2.0 (capacity 600) per the
    # safety-net defaults (spec §3), so exhausting the per-user bucket takes
    # 600 requests instead of 60.
    with TestClient(app) as client:
        for index in range(600):
            response = client.get(
                "/api/v1/persona/profiles", headers={"Cookie": f"custom_session=session-{'a' if index % 2 else 'b'}"}
            )
            assert response.status_code == 200
        denied = client.get("/api/v1/persona/profiles", headers={"Cookie": "custom_session=session-b"})
    assert denied.status_code == 429
    # Each allowed request validates once: ingress on an identity-cache miss (the route
    # reuses its request-state AuthContext), the route itself on a hit. The 429 costs none.
    assert len(validations) == 600
    owner_quota = await governor.peek_with_policy("user:1", ["requests"], "character_chat.default")
    other_owner_quota = await governor.peek_with_policy("user:2", ["requests"], "character_chat.default")
    assert owner_quota["requests"]["remaining"] == 0
    assert other_owner_quota["requests"]["remaining"] == 600


async def test_invalid_cookie_returns_canonical_auth_failure(governed_cookie_app: GovernedCookieApp) -> None:
    app, validations, governor = governed_cookie_app
    with TestClient(app) as client:
        response = client.get("/api/v1/persona/profiles", headers={"Cookie": "custom_session=invalid"})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["detail"] == "Not authenticated (provide Bearer token or X-API-KEY)"
    # The middleware itself never answers 401 (ADR-056 / spec §2): it makes a best-effort
    # attempt to resolve the principal for ingress charging, and on failure falls back to
    # the IP entity and forwards to the route, whose own Depends(get_auth_principal)
    # re-validates the same cookie (a failure never reaches request state; ingress's
    # identity cache is its own) and returns the canonical 401. Hence two checks.
    assert validations == ["invalid", "invalid"]
    owner_quota = await governor.peek_with_policy("user:1", ["requests"], "character_chat.default")
    # capacity is rpm=300 * burst=2.0 = 600 under the safety-net defaults (spec §3).
    assert owner_quota["requests"]["remaining"] == 600
    ip_quota = await governor.peek_with_policy("ip:unknown", ["requests"], "character_chat.default")
    # The invalid cookie never resolves a principal, so ingress charges the anonymous
    # IP bucket instead of the (uninvolved) owner's quota.
    assert ip_quota["requests"]["remaining"] == 599


@pytest.mark.parametrize(
    "headers,expected_validations",
    [
        # No credentials at all: RG admits the anonymous entity (safety-net scope-mismatch
        # fix, spec §4), so the request now reaches the real auth dependency, which falls
        # back to (and exhausts) the cookie check before failing closed with 401.
        ({}, [None]),
        ({"Cookie": "unrelated=session-a"}, [None]),
        # Explicit headers, even empty ones, still take precedence over cookies: the
        # resolver never attempts cookie validation in these two cases.
        ({"Cookie": "custom_session=session-a", "Authorization": ""}, []),
        ({"Cookie": "custom_session=session-a", "X-API-KEY": ""}, []),
    ],
)
def test_cookie_preflight_preserves_absence_and_explicit_header_precedence(governed_cookie_app, headers, expected_validations):
    app, validations, _ = governed_cookie_app
    with TestClient(app) as client:
        response = client.get("/api/v1/persona/profiles", headers=headers)
    # Was 429: an anonymous entity kind (ip) not listed in character_chat.default's
    # scopes used to get no bucket at all and so was denied forever. Safety-net fix
    # (spec §4, scope mismatch) now charges it its own per-entity bucket, admitting it
    # into the real auth dependency, which correctly reports 401 (no credentials).
    assert response.status_code == 401
    assert validations == expected_validations


def test_ungoverned_cookie_does_not_trigger_authentication(governed_cookie_app):
    app, validations, _ = governed_cookie_app
    with TestClient(app) as client:
        response = client.get("/ungoverned", headers={"Cookie": "custom_session=invalid"})
    assert response.status_code == 200
    assert not validations


def test_cookie_preflight_charges_the_owner_in_multi_user_mode_too(governed_cookie_app):
    app, validations, _ = governed_cookie_app
    resolver.get_settings().AUTH_MODE = "multi_user"
    with TestClient(app) as client:
        response = client.get("/api/v1/persona/profiles", headers={"Cookie": "custom_session=session-a"})
    # Ingress identity resolution (ADR-056 / spec §2) is no longer gated to single-user
    # mode: RGSimpleMiddleware._principal_entity resolves the session cookie via
    # get_auth_principal in any AUTH_MODE, caching the AuthContext on request.state.
    # The route's own Depends(get_auth_principal) then reuses that cached context
    # instead of re-validating, so the cookie is only checked once even though both
    # ingress and the route reference the same dependency.
    assert response.status_code == 200
    assert response.json() == {"user_id": 1}
    assert validations == ["session-a"]


async def test_cookie_preflight_resolver_failure_falls_back_to_ip(
    governed_cookie_app: GovernedCookieApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ingress identity resolution is best-effort (ADR-056 / spec §2): it never blocks
    the request. A broken resolver charges the anonymous IP bucket instead of the
    owner's quota, and the route's own auth still runs (and here still succeeds,
    since the route's Depends captured the real, unpatched get_auth_principal).
    """
    app, _, governor = governed_cookie_app

    async def unavailable(request: Request) -> None:
        raise RuntimeError("auth unavailable")

    monkeypatch.setattr(resolver, "get_auth_principal", unavailable)
    with TestClient(app) as client:
        response = client.get("/api/v1/persona/profiles", headers={"Cookie": "custom_session=session-a"})
    assert response.status_code == 200
    owner_quota = await governor.peek_with_policy("user:1", ["requests"], "character_chat.default")
    # capacity is rpm=300 * burst=2.0 = 600 under the safety-net defaults (spec §3).
    assert owner_quota["requests"]["remaining"] == 600
    ip_quota = await governor.peek_with_policy("ip:unknown", ["requests"], "character_chat.default")
    assert ip_quota["requests"]["remaining"] == 599


def test_valid_cookie_does_not_bypass_missing_policy(governed_cookie_app):
    app, validations, _ = governed_cookie_app
    app.state.rg_policy_loader.get_policy = lambda _: {}
    with TestClient(app) as client:
        response = client.get("/api/v1/persona/profiles", headers={"Cookie": "custom_session=session-a"})
    # Was 429 with the error body's policy_id echoing "character_chat.default": an
    # unresolvable policy (the loader here returns {} for every id, including the
    # "default" fallback) used to deny forever. Safety-net fix (spec §4, unknown
    # policy) falls back further to the compiled-in BUILTIN_DEFAULT_POLICY, which
    # admits the request; the valid cookie is then authenticated normally.
    assert response.status_code == 200
    assert response.json() == {"user_id": 1}
    assert validations == ["session-a"]


@pytest.mark.parametrize("scopes", [["global", "ip"], ["user", "api_key", "ip"], ["entity"]])
def test_anonymous_policy_preserves_invalid_cookie_endpoint_behavior(governed_cookie_app, scopes):
    app, validations, governor = governed_cookie_app
    policy = {"requests": {"rpm": 60}, "scopes": scopes}
    app.state.rg_policy_loader.get_policy = lambda _: policy

    @app.get("/api/v1/persona/public")
    def public():
        return {"ok": True}

    with TestClient(app) as client:
        response = client.get("/api/v1/persona/public", headers={"Cookie": "custom_session=invalid"})
    assert response.status_code == 200
    # Ingress identity resolution (ADR-056 / spec §2) no longer depends on the policy's
    # scopes: it always makes a best-effort attempt to resolve the principal when a
    # session cookie is present, even though this anonymous-friendly policy does not
    # require an owner bucket. The failed attempt still falls back to charging the IP.
    assert validations == ["invalid"]


@pytest.mark.parametrize(
    "policy_name,path,method,payload",
    [
        ("health.default", "/api/v1/health", "GET", {"status": "ok"}),
        ("authnz.default", "/api/v1/auth/single-user/session", "DELETE", {"authenticated": False}),
    ],
)
def test_stale_cookie_health_and_idempotent_logout_keep_anonymous_admission(
    governed_cookie_app, policy_name, path, method, payload
):
    app, validations, _ = governed_cookie_app
    policy_path = Path(__file__).resolve().parents[2] / "Config_Files/resource_governor_policies.yaml"
    policy = yaml.safe_load(policy_path.read_text())["policies"][policy_name]
    app.state.rg_policy_loader.get_policy = lambda _: policy
    app.state.rg_policy_loader.get_snapshot().route_map["by_path"][path] = policy_name

    @app.api_route(path, methods=[method])
    def public():
        return payload

    with TestClient(app) as client:
        response = client.request(method, path, headers={"Cookie": "custom_session=stale"})
    assert response.status_code == 200
    assert response.json() == payload
    # Ingress identity resolution (ADR-056 / spec §2) always makes a best-effort attempt
    # to resolve the principal when a session cookie is present, even for policies that
    # admit anonymous traffic; the stale cookie fails validation and ingress falls back
    # to charging the IP entity.
    assert validations == ["stale"]
