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
    with TestClient(app) as client:
        for index in range(60):
            response = client.get(
                "/api/v1/persona/profiles", headers={"Cookie": f"custom_session=session-{'a' if index % 2 else 'b'}"}
            )
            assert response.status_code == 200
        denied = client.get("/api/v1/persona/profiles", headers={"Cookie": "custom_session=session-b"})
    assert denied.status_code == 429
    assert len(validations) == 61  # Canonical endpoint resolver reused the request cache.
    owner_quota = await governor.peek_with_policy("user:1", ["requests"], "character_chat.default")
    other_owner_quota = await governor.peek_with_policy("user:2", ["requests"], "character_chat.default")
    assert owner_quota["requests"]["remaining"] == 0
    assert other_owner_quota["requests"]["remaining"] == 60


async def test_invalid_cookie_returns_canonical_auth_failure(governed_cookie_app: GovernedCookieApp) -> None:
    app, validations, governor = governed_cookie_app
    with TestClient(app) as client:
        response = client.get("/api/v1/persona/profiles", headers={"Cookie": "custom_session=invalid"})
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["detail"] == "Not authenticated (provide Bearer token or X-API-KEY)"
    assert validations == ["invalid"]
    owner_quota = await governor.peek_with_policy("user:1", ["requests"], "character_chat.default")
    assert owner_quota["requests"]["remaining"] == 60


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Cookie": "unrelated=session-a"},
        {"Cookie": "custom_session=session-a", "Authorization": ""},
        {"Cookie": "custom_session=session-a", "X-API-KEY": ""},
    ],
)
def test_cookie_preflight_preserves_absence_and_explicit_header_precedence(governed_cookie_app, headers):
    app, validations, _ = governed_cookie_app
    with TestClient(app) as client:
        response = client.get("/api/v1/persona/profiles", headers=headers)
    assert response.status_code == 429
    assert not validations


def test_ungoverned_cookie_does_not_trigger_authentication(governed_cookie_app):
    app, validations, _ = governed_cookie_app
    with TestClient(app) as client:
        response = client.get("/ungoverned", headers={"Cookie": "custom_session=invalid"})
    assert response.status_code == 200
    assert not validations


def test_cookie_preflight_does_not_apply_in_multi_user_mode(governed_cookie_app):
    app, validations, _ = governed_cookie_app
    resolver.get_settings().AUTH_MODE = "multi_user"
    with TestClient(app) as client:
        response = client.get("/api/v1/persona/profiles", headers={"Cookie": "custom_session=session-a"})
    assert response.status_code == 429
    assert not validations


async def test_cookie_preflight_does_not_fail_open_on_resolver_failure(
    governed_cookie_app: GovernedCookieApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _, governor = governed_cookie_app

    async def unavailable(request: Request) -> None:
        raise RuntimeError("auth unavailable")

    monkeypatch.setattr(resolver, "get_auth_principal", unavailable)
    with TestClient(app) as client, pytest.raises(RuntimeError, match="auth unavailable"):
        client.get("/api/v1/persona/profiles", headers={"Cookie": "custom_session=session-a"})
    owner_quota = await governor.peek_with_policy("user:1", ["requests"], "character_chat.default")
    assert owner_quota["requests"]["remaining"] == 60


def test_valid_cookie_does_not_bypass_missing_policy(governed_cookie_app):
    app, validations, _ = governed_cookie_app
    app.state.rg_policy_loader.get_policy = lambda _: {}
    with TestClient(app) as client:
        response = client.get("/api/v1/persona/profiles", headers={"Cookie": "custom_session=session-a"})
    assert response.status_code == 429
    assert not validations
    assert response.json()["policy_id"] == "character_chat.default"


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
    assert not validations


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
    assert not validations
