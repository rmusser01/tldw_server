from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import monitoring as monitoring_mod
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE, SYSTEM_LOGS
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal

MUTATING_MONITORING_REQUESTS = (
    ("POST", "/api/v1/monitoring/watchlists", "/api/v1/monitoring/watchlists", {"name": "scope-check", "rules": []}),
    ("DELETE", "/api/v1/monitoring/watchlists/{watchlist_id}", "/api/v1/monitoring/watchlists/scope-check", None),
    ("POST", "/api/v1/monitoring/reload", "/api/v1/monitoring/reload", None),
    ("POST", "/api/v1/monitoring/alerts/{alert_id}/read", "/api/v1/monitoring/alerts/1/read", None),
    ("POST", "/api/v1/monitoring/alerts/{alert_id}/acknowledge", "/api/v1/monitoring/alerts/1/acknowledge", None),
    ("DELETE", "/api/v1/monitoring/alerts/{alert_id}", "/api/v1/monitoring/alerts/1", None),
    (
        "PUT",
        "/api/v1/monitoring/notifications/settings",
        "/api/v1/monitoring/notifications/settings",
        {"enabled": True},
    ),
    (
        "POST",
        "/api/v1/monitoring/notifications/test",
        "/api/v1/monitoring/notifications/test",
        {"message": "scope-check"},
    ),
)
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _read_scoped_monitoring_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
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
        permissions=[SYSTEM_LOGS, SYSTEM_CONFIGURE],
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

    class _NotificationService:
        def get_settings(self) -> dict[str, object]:
            return {
                "enabled": False,
                "min_severity": "critical",
                "file": "Databases/monitoring_notifications.log",
            }

        def update_settings(self, **changes: object) -> dict[str, object]:
            return {**self.get_settings(), **changes}

        def notify(self, _alert: object) -> str:
            return "ok"

    monkeypatch.setattr(monitoring_mod, "get_notification_service", _NotificationService)
    return app


@pytest.mark.unit
@pytest.mark.parametrize(("method", "route_path", "request_path", "payload"), MUTATING_MONITORING_REQUESTS)
def test_read_scoped_system_logs_key_cannot_mutate_monitoring_state(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    route_path: str,
    request_path: str,
    payload: dict[str, object] | None,
) -> None:
    del route_path
    app = _read_scoped_monitoring_app(monkeypatch)

    with TestClient(app) as client:
        response = client.request(method, request_path, json=payload)

    assert response.status_code == 403
    assert "scope" in response.json()["detail"].lower()


def test_scope_regression_inventory_covers_every_monitoring_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _read_scoped_monitoring_app(monkeypatch)
    expected = {(method, route_path) for method, route_path, _request_path, _payload in MUTATING_MONITORING_REQUESTS}
    actual = {
        (method, route.path)
        for route in app.routes
        if route.path.startswith("/api/v1/monitoring/")
        for method in route.methods or set()
        if method in UNSAFE_METHODS
    }

    assert actual == expected
