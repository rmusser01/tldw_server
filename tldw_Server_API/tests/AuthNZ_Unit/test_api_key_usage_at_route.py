"""API-key usage is recorded once by route auth, not by RG ingress (TASK-13403).

Ingress validates the key before routing, when the endpoint, action and scope are not
known yet, and may then answer 429. Usage must be recorded once, at route auth, with
those details, and never for a request ingress denied.
"""

from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import require_token_scope
from tldw_Server_API.app.core.AuthNZ import User_DB_Handling as user_mod
from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
from tldw_Server_API.app.core.AuthNZ.repos import users_repo as users_repo_mod
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
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

    async def verify(api_key, *_args):
        mgr.validations.append(api_key)
        key_info = {"id": 7, "user_id": 42, "scope": "read", "org_id": 1, "team_id": None, "metadata": {}}
        return key_info, None

    async def update_usage(key_id, ip_address=None):
        mgr.usage_updates.append(key_id)

    async def log_action(key_id, action, user_id=None, details=None):
        mgr.used_rows.append((action, details))

    async def get_manager():
        return mgr

    async def memberships(_user_id):
        return [{"org_id": 1, "team_id": None}]

    async def scoped_permissions(**kwargs):
        return SimpleNamespace(permissions=list(kwargs.get("base_permissions") or []), active_org_id=1, active_team_id=None)

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
