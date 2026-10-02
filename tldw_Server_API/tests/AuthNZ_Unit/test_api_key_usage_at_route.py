"""API keys through RG ingress and route auth: usage (TASK-13403) and tenant (TASK-13402).

API-key usage is recorded once by route auth, not by RG ingress.

Ingress validates the key before routing, when the endpoint, action and scope are not
known yet, and may then answer 429. Usage must be recorded once, at route auth, with
those details, and never for a request ingress denied.
"""

from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import require_token_scope
from tldw_Server_API.app.core.AuthNZ import User_DB_Handling as user_mod
from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
from tldw_Server_API.app.core.AuthNZ.repos import users_repo as users_repo_mod
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.Resource_Governance.deps import derive_entity_key
from tldw_Server_API.app.core.Resource_Governance.governor import MemoryResourceGovernor
from tldw_Server_API.app.core.Resource_Governance.middleware_simple import RGSimpleMiddleware

pytestmark = [pytest.mark.unit]

KEY = "tldw_test.key"
DETAILS = {"endpoint_id": "unit.usage", "action": "call", "scope": "any", "path": "/api/v1/thing", "method": "GET"}


class _Snap:
    route_map = {"by_path": {"/api/v1/*": "p"}, "by_tag": {}}
    tenant = {}
    policies = {}


class _Loader:
    def get_snapshot(self):
        return _Snap()

    def get_policy(self, pid):
        return {"requests": {"rpm": 100, "burst": 1.0}, "scopes": ["user", "api_key", "ip"]}


class _DenyGovernor:
    async def reserve(self, request, op_id=None):
        return SimpleNamespace(allowed=False, retry_after=1, details={}), None


@pytest.fixture
def manager(monkeypatch):
    """The real APIKeyManager recording path, with key lookup and storage stubbed."""
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.delenv("RG_POLICY_PATH", raising=False)
    reset_settings()
    mgr = APIKeyManager()
    mgr._initialized = True
    mgr.settings = SimpleNamespace(API_KEY_AUDIT_LOG_USAGE=True, PII_REDACT_LOGS=False)
    mgr.validations, mgr.usage_updates, mgr.used_rows = [], [], []
    mgr.key_info = {"id": 7, "user_id": 42, "scope": "read", "org_id": 1, "team_id": None, "metadata": {}}
    mgr.memberships = [{"org_id": 1, "team_id": None}]

    async def verify(api_key, *_args):
        mgr.validations.append(api_key)
        return dict(mgr.key_info), None

    async def update_usage(key_id, ip_address=None):
        mgr.usage_updates.append(key_id)

    async def log_action(key_id, action, user_id=None, details=None):
        mgr.used_rows.append((action, details))

    async def get_manager():
        return mgr

    async def memberships(_user_id):
        return list(mgr.memberships)

    async def scoped_permissions(**kwargs):
        return SimpleNamespace(
            permissions=list(kwargs.get("base_permissions") or []),
            active_org_id=kwargs.get("active_org_id"),
            active_team_id=kwargs.get("active_team_id"),
        )

    async def no_db_pool():
        raise RuntimeError("no AuthNZ database in this test")

    class _Users:
        async def get_user_by_id(self, user_id):
            return {"id": user_id, "username": "api-user", "role": "user", "is_active": True, "is_verified": True}

    async def from_pool(cls):
        return _Users()

    monkeypatch.setattr(mgr, "_get_repo", lambda: None)
    monkeypatch.setattr(mgr, "_verify_new_format_key", verify)
    monkeypatch.setattr(mgr, "_verify_legacy_key", verify)
    monkeypatch.setattr(mgr, "_update_usage", update_usage)
    monkeypatch.setattr(mgr, "_log_action", log_action)
    monkeypatch.setattr(user_mod, "get_api_key_manager", get_manager)
    monkeypatch.setattr(auth_deps, "get_api_key_manager", get_manager)
    monkeypatch.setattr(user_mod, "_enrich_user_with_rbac", lambda *_a, **_k: (["user"], ["media.read"], False))
    monkeypatch.setattr(user_mod, "list_memberships_for_user", memberships)
    monkeypatch.setattr(user_mod, "apply_scoped_permissions", scoped_permissions)
    monkeypatch.setattr(users_repo_mod.AuthnzUsersRepo, "from_pool", classmethod(from_pool))
    monkeypatch.setattr(user_mod, "get_db_pool", no_db_pool)
    yield mgr
    reset_settings()


def _client(governor=None, route_auth=user_mod.get_request_user):
    app = FastAPI()
    guard = require_token_scope("any", require_if_present=True, endpoint_id="unit.usage", count_as="call")

    @app.get("/api/v1/thing", dependencies=[Depends(guard)])
    async def thing(_auth=Depends(route_auth)):  # noqa: B008
        return {"ok": True}

    async def fake_db_pool():
        return object()

    app.dependency_overrides[auth_deps.get_db_pool] = fake_db_pool
    app.dependency_overrides[auth_deps.get_jwt_service_dep] = lambda: object()
    app.dependency_overrides[auth_deps.get_session_manager_dep] = lambda: object()
    app.add_middleware(RGSimpleMiddleware)
    app.state.rg_policy_loader = _Loader()
    if governor is not None:
        app.state.rg_governor = governor
    return TestClient(app)


@pytest.mark.parametrize(
    "route_auth",
    [user_mod.get_request_user, auth_deps.get_auth_principal, auth_deps.get_current_user],
    ids=["get_request_user", "get_auth_principal", "get_current_user"],
)
def test_ingress_validated_key_records_usage_once_at_route_with_details(manager, route_auth):
    client = _client(MemoryResourceGovernor(policy_loader=_Loader()), route_auth)

    assert client.get("/api/v1/thing", headers={"X-API-KEY": KEY}).status_code == 200
    assert manager.usage_updates == [7]
    assert manager.used_rows == [("used", DETAILS)]


def test_request_denied_at_ingress_records_no_usage(manager):
    client = _client(_DenyGovernor())

    assert client.get("/api/v1/thing", headers={"X-API-KEY": KEY}).status_code == 429
    assert manager.validations == [KEY]  # ingress did validate the key
    assert manager.usage_updates == []
    assert manager.used_rows == []


def test_rg_disabled_records_usage_once_at_route(manager, monkeypatch):
    monkeypatch.setenv("RG_ENABLED", "false")
    client = _client(governor=None)

    assert client.get("/api/v1/thing", headers={"X-API-KEY": KEY}).status_code == 200
    assert manager.usage_updates == [7]
    assert manager.used_rows == [("used", DETAILS)]


def test_ingress_cache_hit_still_records_usage_once_per_request(manager):
    client = _client(MemoryResourceGovernor(policy_loader=_Loader()))

    for _ in range(2):
        assert client.get("/api/v1/thing", headers={"X-API-KEY": KEY}).status_code == 200
    assert manager.usage_updates == [7, 7]
    assert manager.used_rows == [("used", DETAILS), ("used", DETAILS)]


async def _non_user_auth_cache(request: Request) -> None:
    """Something replaced the cached user, so route auth skips its fast path and re-validates."""
    request.state._auth_user = {"id": 42}


@pytest.mark.parametrize("then_reuse_context", [False, True], ids=["revalidate", "revalidate-then-reuse"])
def test_route_revalidation_after_ingress_records_usage_once(manager, then_reuse_context):
    # Ingress deferred the usage. A fresh validation at route auth must record it itself
    # (the ingress flag is reset) and clear the pending entry, so a later reuse of the
    # context does not record it again.
    app = FastAPI()
    extra = [Depends(auth_deps.get_auth_principal)] if then_reuse_context else []

    @app.get("/api/v1/thing", dependencies=[Depends(_non_user_auth_cache), Depends(user_mod.get_request_user), *extra])
    async def thing():
        return {"ok": True}

    app.add_middleware(RGSimpleMiddleware)
    app.state.rg_policy_loader = _Loader()
    app.state.rg_governor = MemoryResourceGovernor(policy_loader=_Loader())

    assert TestClient(app).get("/api/v1/thing", headers={"X-API-KEY": KEY}).status_code == 200
    assert manager.validations == [KEY, KEY]  # ingress, then route auth again
    assert manager.usage_updates == [7]


class _TenantSnap(_Snap):
    tenant = {"enabled": True, "header": "X-TLDW-Tenant"}


class _TenantLoader(_Loader):
    def get_snapshot(self):
        return _TenantSnap()


def test_multi_org_key_without_active_org_gets_the_same_tenant_at_ingress_and_route(manager):
    # TASK-13402 review: ingress and route-level reservations must pick the caller's own
    # tenant in the same order (tenant claim, active org, then the key's first org).
    manager.key_info["org_id"] = None
    manager.memberships = [{"org_id": 1, "team_id": None}, {"org_id": 2, "team_id": None}]
    app = FastAPI()

    @app.get("/api/v1/thing")
    async def thing(request: Request, _user=Depends(user_mod.get_request_user)):  # noqa: B008
        return {"ingress": request.state.rg_ingress_entity, "route": derive_entity_key(request)}

    app.add_middleware(RGSimpleMiddleware)
    app.state.rg_policy_loader = _TenantLoader()
    app.state.rg_governor = MemoryResourceGovernor(policy_loader=_TenantLoader())
    client = TestClient(app)

    for _ in range(2):  # the second request is an ingress identity-cache hit
        body = client.get("/api/v1/thing", headers={"X-API-KEY": KEY}).json()
        assert body == {"ingress": "tenant:1", "route": "tenant:1"}
