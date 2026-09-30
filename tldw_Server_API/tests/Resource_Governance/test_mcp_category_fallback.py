import pytest

from tldw_Server_API.app.core.MCP_unified.auth import rate_limiter as mcp_rl
from tldw_Server_API.app.core.Resource_Governance.governor import RGDecision

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit, pytest.mark.asyncio]


class _SpyGov:
    def __init__(self):
        self.policy_ids = []

    async def reserve(self, req, op_id=None):
        self.policy_ids.append(req.tags["policy_id"])
        return RGDecision(allowed=True, retry_after=None, details={}), "h"

    async def commit(self, handle, actuals=None, op_id=None):
        return None


class _Loader:
    def get_policy(self, pid):
        return {"requests": {"rpm": 60}} if pid in {"mcp.default", "mcp.read"} else None


async def _run(monkeypatch, category):
    gov = _SpyGov()

    async def _get():
        return gov

    monkeypatch.setattr(mcp_rl, "_get_mcp_rg_governor", _get)
    monkeypatch.setattr(mcp_rl, "_rg_mcp_loader", _Loader())
    result = await mcp_rl._maybe_enforce_with_rg_mcp(key="user:1", category=category)
    return gov, result


async def test_undefined_category_uses_mcp_default(monkeypatch):
    gov, result = await _run(monkeypatch, "browser")
    assert gov.policy_ids == ["mcp.default"] and result["allowed"]


async def test_defined_category_keeps_its_policy(monkeypatch):
    gov, _ = await _run(monkeypatch, "read")
    assert gov.policy_ids == ["mcp.read"]
