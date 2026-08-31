from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import monitoring as monitoring_mod
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_LOGS
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal

MUTATING_MONITORING_REQUESTS = (
    ("POST", "/api/v1/monitoring/watchlists", {"name": "scope-check", "rules": []}),
    ("DELETE", "/api/v1/monitoring/watchlists/scope-check", None),
    ("POST", "/api/v1/monitoring/reload", None),
    ("POST", "/api/v1/monitoring/alerts/1/read", None),
    ("POST", "/api/v1/monitoring/alerts/1/acknowledge", None),
    ("DELETE", "/api/v1/monitoring/alerts/1", None),
)


def _read_scoped_monitoring_app() -> FastAPI:
    app = FastAPI()
    app.include_router(monitoring_mod.router, prefix="/api/v1")
    principal = AuthPrincipal(
        kind="api_key",
        user_id=41,
        api_key_id=73,
        subject="prometheus",
        token_type="api_key",
        jti=None,
        roles=[],
        permissions=[SYSTEM_LOGS],
        is_admin=False,
        org_ids=[],
        team_ids=[],
    )

    async def _principal(request: Request) -> AuthPrincipal:
        request.state._api_key_scope = "read"
        request.state.auth = AuthContext(
            principal=principal,
            ip="127.0.0.1",
            user_agent="scope-test",
            request_id="scope-test",
        )
        return principal

    app.dependency_overrides[auth_deps.get_auth_principal] = _principal
    return app


@pytest.mark.unit
@pytest.mark.parametrize(("method", "path", "payload"), MUTATING_MONITORING_REQUESTS)
def test_read_scoped_system_logs_key_cannot_mutate_monitoring_state(
    method: str,
    path: str,
    payload: dict[str, object] | None,
) -> None:
    app = _read_scoped_monitoring_app()

    with TestClient(app) as client:
        response = client.request(method, path, json=payload)

    assert response.status_code == 403
    assert "scope" in response.json()["detail"].lower()
