"""RGSimpleMiddleware enforces tag-only policies, before routing, through nested includes."""

import pytest
import yaml
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Resource_Governance.governor import MemoryResourceGovernor
from tldw_Server_API.app.core.Resource_Governance.middleware_simple import RGSimpleMiddleware
from tldw_Server_API.app.core.Resource_Governance.policy_loader import PolicyLoader, PolicyReloadConfig

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit, pytest.mark.asyncio]


async def _client(tmp_path, monkeypatch, policies, route_map):
    monkeypatch.delenv("RG_POLICY_PATH", raising=False)
    path = tmp_path / "rg.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "policies": policies, "route_map": route_map}), encoding="utf-8")
    loader = PolicyLoader(path, PolicyReloadConfig(enabled=False))
    await loader.load_once()

    inner = APIRouter(tags=["writing"])

    @inner.get("/docs/{doc_id}")
    def doc(doc_id: str) -> dict:
        return {"ok": True}

    middle = APIRouter()
    middle.include_router(inner, prefix="/writing")
    app = FastAPI()
    app.include_router(middle, prefix="/api/v1")

    @app.get("/api/v1/unmapped")
    def unmapped() -> dict:
        return {"ok": True}

    app.add_middleware(RGSimpleMiddleware)
    app.state.rg_policy_loader = loader
    app.state.rg_governor = MemoryResourceGovernor(policy_loader=loader)
    return TestClient(app)


async def test_tag_only_route_is_governed(tmp_path, monkeypatch):
    client = await _client(
        tmp_path,
        monkeypatch,
        policies={"tagpol": {"requests": {"rpm": 2, "burst": 1.0}, "scopes": ["ip"]}, "default": {"requests": {"rpm": 1000}}},
        route_map={"by_path": {}, "by_tag": {"writing": "tagpol"}},
    )
    codes = [client.get("/api/v1/writing/docs/1").status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    assert client.get("/api/v1/writing/docs/1").json()["policy_id"] == "tagpol"


async def test_unmapped_api_route_is_governed_by_default(tmp_path, monkeypatch):
    client = await _client(
        tmp_path,
        monkeypatch,
        policies={"default": {"requests": {"rpm": 1, "burst": 1.0}, "scopes": ["ip"]}},
        route_map={"by_path": {}, "by_tag": {}},
    )
    assert [client.get("/api/v1/unmapped").status_code for _ in range(2)] == [200, 429]
    assert client.get("/api/v1/unmapped").json()["policy_id"] == "default"
