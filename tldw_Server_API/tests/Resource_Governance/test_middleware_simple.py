import pytest
pytestmark = pytest.mark.rate_limit
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Resource_Governance.middleware_simple import RGSimpleMiddleware
from tldw_Server_API.app.core.Resource_Governance.governor import RGDecision


class _Snap:
    def __init__(self, route_map, tenant=None):
        self.route_map = route_map
        self.tenant = tenant or {}


class _Loader:
    def __init__(self, route_map, policies=None, tenant=None):
        self._snap = _Snap(route_map, tenant=tenant)
        self._policies = policies or {}

    def get_snapshot(self):

        return self._snap

    def get_policy(self, policy_id):
        return dict(self._policies.get(policy_id) or {})


class _Gov:
    def __init__(self):
        pass

    async def reserve(self, req, op_id=None):
        pid = (req.tags or {}).get("policy_id")
        # Any policy id starting with 'deny' will be denied
        if pid and pid.startswith("deny"):
            dec = RGDecision(
                allowed=False,
                retry_after=12,
                details={
                    "policy_id": pid,
                    "categories": {"requests": {"allowed": False, "retry_after": 12, "limit": 2}},
                },
            )
            return dec, None
        dec = RGDecision(
            allowed=True,
            retry_after=None,
            details={"policy_id": pid, "categories": {"requests": {"allowed": True, "limit": 2, "retry_after": 0}}},
        )
        return dec, "h1"

    async def commit(self, handle_id, actuals=None):
        return None


class _ExplodingGov:
    async def reserve(self, req, op_id=None):
        raise RuntimeError("resource governor unavailable")


class _CaptureGov:
    def __init__(self):
        self.requests = []

    async def reserve(self, req, op_id=None):
        self.requests.append(req)
        dec = RGDecision(
            allowed=True,
            retry_after=None,
            details={
                "policy_id": req.tags.get("policy_id"),
                "categories": {"requests": {"allowed": True, "limit": 2, "retry_after": 0}},
            },
        )
        return dec, "h-capture"

    async def commit(self, handle_id, actuals=None):
        return None


def _make_app(route_map):


    app = FastAPI()
    app.add_middleware(RGSimpleMiddleware)

    @app.get("/api/v1/chat/completions", tags=["chat"])
    async def chat_route():  # pragma: no cover - exercised via client
        return {"ok": True}

    @app.get("/api/v1/embeddings/vec")
    async def emb_route():  # pragma: no cover
        return {"ok": True}

    # Attach RG components
    app.state.rg_policy_loader = _Loader(route_map)
    app.state.rg_governor = _Gov()
    return app


@pytest.mark.asyncio
async def test_middleware_denies_with_retry_after_and_headers_by_tag():
    route_map = {"by_tag": {"chat": "deny.chat"}, "by_path": {"/api/v1/chat/*": "deny.chat"}}
    app = _make_app(route_map)
    with TestClient(app) as c:
        r = c.get("/api/v1/chat/completions")
        assert r.status_code == 429
        assert r.json().get("policy_id") == "deny.chat"
        # Headers present
        assert r.headers.get("Retry-After") == "12"
        assert r.headers.get("X-RateLimit-Limit") == "2"
        assert r.headers.get("X-RateLimit-Remaining") == "0"
        assert r.headers.get("X-RateLimit-Reset") == "12"


@pytest.mark.asyncio
async def test_middleware_denies_with_retry_after_by_path():
    route_map = {"by_path": {"/api/v1/embeddings*": "deny.emb"}}
    app = _make_app(route_map)
    with TestClient(app) as c:
        r = c.get("/api/v1/embeddings/vec")
        assert r.status_code == 429
        assert r.json().get("policy_id") == "deny.emb"
        assert r.headers.get("Retry-After") == "12"


@pytest.mark.asyncio
async def test_middleware_allows_when_policy_allows():
    route_map = {"by_tag": {"chat": "allow.chat"}, "by_path": {"/api/v1/chat/*": "allow.chat"}}
    app = _make_app(route_map)
    with TestClient(app) as c:
        r = c.get("/api/v1/chat/completions")
        assert r.status_code == 200
        assert r.json().get("ok") is True
        # Success-path rate-limit headers present
        assert r.headers.get("X-RateLimit-Limit") == "2"
        assert r.headers.get("X-RateLimit-Remaining") == "1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path, pattern, policy_id",
    [
        ("/api/v1/research/websearch", "/api/v1/research/*", "deny.research"),
        ("/api/v1/workflows/definitions", "/api/v1/workflows/*", "deny.workflows"),
        ("/api/v1/scheduler/workflows/status", "/api/v1/scheduler/workflows/*", "deny.workflows"),
        ("/api/v1/prompt-studio/projects", "/api/v1/prompt-studio/*", "deny.prompt_studio"),
        ("/api/v1/rag/search", "/api/v1/rag/*", "deny.rag"),
        ("/api/v1/media/process-videos", "/api/v1/media/*", "deny.media"),
    ],
)
async def test_middleware_resolves_new_domain_paths(path: str, pattern: str, policy_id: str):
    route_map = {"by_path": {pattern: policy_id}}
    app = _make_app(route_map)
    with TestClient(app) as c:
        r = c.get(path)
        assert r.status_code == 429
        assert r.json().get("policy_id") == policy_id


@pytest.mark.asyncio
async def test_middleware_fail_closed_on_reserve_error_when_policy_requires_it():
    app = FastAPI()
    app.add_middleware(RGSimpleMiddleware)

    @app.get("/api/v1/fail-closed")
    async def route():  # pragma: no cover - exercised via client
        return {"ok": True}

    route_map = {"by_path": {"/api/v1/fail-closed": "fail.closed"}}
    app.state.rg_policy_loader = _Loader(route_map, policies={"fail.closed": {"fail_mode": "fail_closed"}})
    app.state.rg_governor = _ExplodingGov()

    with TestClient(app) as c:
        r = c.get("/api/v1/fail-closed")

    assert r.status_code == 503
    assert r.json()["error"] == "resource_governance_unavailable"
    assert r.json()["policy_id"] == "fail.closed"


@pytest.mark.asyncio
async def test_middleware_uses_tenant_entity_when_tenant_scope_enabled():
    app = FastAPI()
    app.add_middleware(RGSimpleMiddleware)

    @app.get("/api/v1/tenant-scoped")
    async def route():  # pragma: no cover - exercised via client
        return {"ok": True}

    route_map = {"by_path": {"/api/v1/tenant-scoped": "tenant.only"}}
    gov = _CaptureGov()
    app.state.rg_policy_loader = _Loader(
        route_map,
        policies={"tenant.only": {"requests": {"rpm": 2}, "scopes": ["tenant"]}},
        tenant={"enabled": True, "header": "X-TLDW-Tenant"},
    )
    app.state.rg_governor = gov

    with TestClient(app) as c:
        r = c.get("/api/v1/tenant-scoped", headers={"X-TLDW-Tenant": "acme"})

    assert r.status_code == 200
    assert gov.requests
    assert gov.requests[0].entity == "tenant:acme"
