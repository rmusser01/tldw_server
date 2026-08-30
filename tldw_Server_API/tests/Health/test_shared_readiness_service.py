from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import health
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import readiness_service
from tldw_Server_API.app.services.readiness_service import (
    ReadinessSnapshot,
    internal_readiness_payload,
    operator_readiness_payload,
)


def test_internal_projection_discards_all_detail() -> None:
    snapshot = ReadinessSnapshot(
        ready=False,
        reason="database_unavailable",
        details={"database": {"type": "postgresql"}, "providers_initialized": False},
    )
    assert internal_readiness_payload(snapshot) == {"status": "not_ready"}


def test_operator_projection_keeps_only_sanitized_snapshot_detail() -> None:
    snapshot = ReadinessSnapshot(
        ready=True,
        reason=None,
        details={"database": {"status": "healthy", "type": "postgresql"}},
    )
    assert operator_readiness_payload(snapshot) == {
        "status": "ready",
        "database": {"status": "healthy", "type": "postgresql"},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ("/ready", "/health/ready", "/api/v1/readyz", "/api/v1/health/ready"))
async def test_each_readiness_route_uses_the_shared_snapshot_once(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    from tldw_Server_API.app import main

    app = FastAPI()
    app.add_api_route("/ready", main.readiness_check, methods=["GET"])
    app.add_api_route("/health/ready", main.readiness_alias, methods=["GET"])
    app.include_router(health.router, prefix="/api/v1")
    app.dependency_overrides[auth_deps.get_auth_principal] = lambda: AuthPrincipal(
        kind="user", user_id=1, api_key_id=None, subject="test", token_type="access", jti=None,
        roles=["admin"], permissions=[], is_admin=True, org_ids=[], team_ids=[],
    )
    snapshot = ReadinessSnapshot(True, None, {"database": {"status": "healthy", "type": "sqlite"}})
    collect = AsyncMock(return_value=snapshot)
    monkeypatch.setattr(readiness_service, "collect_readiness_snapshot", collect)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(path)

    assert response.status_code == 200
    assert collect.await_count == 1


@pytest.mark.asyncio
async def test_public_liveness_never_collects_readiness_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app import main

    app = FastAPI()
    app.add_api_route("/health", main.health_check, methods=["GET"])
    collect = AsyncMock()
    monkeypatch.setattr(readiness_service, "collect_readiness_snapshot", collect)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert collect.await_count == 0
