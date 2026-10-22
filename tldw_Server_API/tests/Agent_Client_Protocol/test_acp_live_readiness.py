"""Actual health aggregation must distinguish an executable from a live runner."""

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints import agent_client_protocol as endpoints
from tldw_Server_API.app.core.Agent_Client_Protocol import agent_registry
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("probe", ["ok", "not_running", "error"])
async def test_health_requires_live_runner_probe(monkeypatch, probe):
    entry = agent_registry.AgentRegistryEntry(type="c04-test", name="Configured agent")
    monkeypatch.setattr(entry, "check_availability", lambda: {"is_configured": True, "status": "available"})
    monkeypatch.setattr(agent_registry, "get_agent_registry", lambda: SimpleNamespace(entries=[entry]))
    monkeypatch.setattr(endpoints, "_check_runner_binary", lambda: {"status": "ok", "source": "PATH"})

    async def runner():
        if probe == "error":
            raise RuntimeError("private-startup-detail")
        return SimpleNamespace(is_running=probe == "ok", agent_capabilities={})

    monkeypatch.setattr(endpoints, "get_runner_client", runner)
    result = await endpoints.acp_health(user=User(id=1, username="owner"))
    assert result["runner"]["status"] == "ok"  # nosec B101 - Installed binary remains separately attributable.
    assert result["runner_probe"]["status"] == probe  # nosec B101 - Actual probe outcome remains visible.
    assert result["overall"] == ("ok" if probe == "ok" else "unavailable")  # nosec B101 - Binary presence cannot certify execution readiness.
    if probe != "ok":
        assert "runner_cwd" in result["message"]  # nosec B101 - Actionable supported setup guidance.
        assert "private-startup-detail" not in str(result)  # nosec B101 - No raw startup error exposure.
