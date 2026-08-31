import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_LOGS
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Resource_Governance import MemoryResourceGovernor, RGRequest

pytestmark = pytest.mark.rate_limit


@pytest.mark.asyncio
async def test_prometheus_metrics_endpoint_includes_rg_series(monkeypatch):
    # Minimal app to avoid heavy imports; enable metrics routes
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")

    # Generate some RG metrics using memory backend (same process registry)
    pols = {"p": {"requests": {"rpm": 1}, "scopes": ["global", "user"]}}
    rg = MemoryResourceGovernor(policies=pols)
    e = "user:prom"
    d1, h1 = await rg.reserve(RGRequest(entity=e, categories={"requests": {"units": 1}}, tags={"policy_id": "p"}))
    assert d1.allowed and h1
    d2, _ = await rg.reserve(RGRequest(entity=e, categories={"requests": {"units": 1}}, tags={"policy_id": "p"}))
    assert not d2.allowed

    # Import app and fetch /metrics
    from tldw_Server_API.app.main import app

    async def _metrics_operator() -> AuthPrincipal:
        return AuthPrincipal(
            kind="user",
            user_id=1,
            subject="resource-governance-metrics-test",
            roles=["user"],
            permissions=[SYSTEM_LOGS],
        )

    previous_override = app.dependency_overrides.get(auth_deps.get_auth_principal)
    app.dependency_overrides[auth_deps.get_auth_principal] = _metrics_operator
    try:
        with TestClient(app) as c:
            r = c.get("/metrics")
            assert r.status_code == 200
            body = r.text
            # Assert RG decision counter is present and does not include entity label
            assert "rg_decisions_total" in body
            assert "entity=" not in body
            # Presence of our deny counter series
            assert "rg_denials_total" in body
    finally:
        if previous_override is None:
            app.dependency_overrides.pop(auth_deps.get_auth_principal, None)
        else:
            app.dependency_overrides[auth_deps.get_auth_principal] = previous_override
