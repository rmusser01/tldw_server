"""Caller discovery uses the same permission guards as the optional read routes."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_auth_principal,
    get_request_user,
    get_session_manager_dep,
)
from tldw_Server_API.app.api.v1.endpoints import (
    auth,
    monitoring,
    notifications,
    scheduled_tasks_control_plane,
    users,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

pytestmark = pytest.mark.unit

READS = {
    "can_read_scheduled_tasks": "/api/v1/scheduled-tasks",
    "can_read_notifications": "/api/v1/notifications",
    "can_read_monitoring_alerts": "/api/v1/monitoring/alerts",
}


@pytest.fixture
def app(monkeypatch):
    """Keep real route/guard execution; replace only unrelated data/rate boundaries."""
    application = FastAPI()
    for router in (
        users.router,
        auth.router,
        scheduled_tasks_control_plane.router,
        notifications.router,
        monitoring.router,
    ):
        application.include_router(router, prefix="/api/v1")
    for route in application.routes:
        for dependency in getattr(route, "dependencies", []):
            if getattr(dependency.dependency, "_tldw_rate_limit_resource", None):
                application.dependency_overrides[dependency.dependency] = lambda: None
    application.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=7)
    application.dependency_overrides[get_session_manager_dep] = lambda: None
    application.dependency_overrides[scheduled_tasks_control_plane.get_scheduled_tasks_control_plane_service] = (
        lambda: SimpleNamespace(list_tasks=AsyncMock(return_value={"items": [], "total": 0}))
    )
    application.dependency_overrides[notifications.get_collections_db_for_user] = lambda: SimpleNamespace(
        user_id=7,
        list_user_notifications=Mock(return_value=[]),
        count_user_notifications=Mock(return_value=0),
    )
    monkeypatch.setattr(notifications, "RemindersService", Mock())
    application.dependency_overrides[monitoring.get_topic_monitoring_db] = lambda: SimpleNamespace(
        list_alerts=Mock(return_value=[]),
    )
    monkeypatch.setattr(
        monitoring,
        "_get_monitoring_repo",
        AsyncMock(
            return_value=SimpleNamespace(
                list_alert_states=AsyncMock(return_value=[]),
                ensure_schema_ready_once=AsyncMock(),
            )
        ),
    )
    return application


@pytest.mark.parametrize(
    "permissions,roles,admin,expected",
    [
        (["media.read", "notes.write"], ["user"], False, (False, False, False)),
        (["tasks.read"], ["user"], False, (True, False, False)),
        (["notifications.read"], ["custom"], False, (False, True, False)),
        (["system.logs"], ["custom"], False, (False, False, True)),
        (["tasks.read", "notifications.read", "system.logs"], ["user"], False, (True, True, True)),
        ([], ["admin"], False, (True, True, True)),
        (["*"], ["user"], False, (True, True, True)),
        (["system.configure"], ["user"], False, (False, False, False)),
        ([], ["user"], True, (False, False, False)),
    ],
)
def test_decisions_match_real_protected_read_routes(app, permissions, roles, admin, expected):
    principal = AuthPrincipal(kind="user", user_id=7, permissions=permissions, roles=roles, is_admin=admin)
    app.dependency_overrides[get_auth_principal] = lambda: principal
    with TestClient(app) as client:
        response = client.get("/api/v1/users/me/capabilities?user_id=99")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.json() == {"user_id": 7, **dict(zip(READS, expected))}
        for capability, allowed in zip(READS, expected):
            protected = client.get(READS[capability])
            assert protected.status_code == (200 if allowed else 403), protected.text


def test_requires_authentication(app):
    async def unauthenticated():
        raise HTTPException(status_code=401, detail="Not authenticated")

    app.dependency_overrides[get_auth_principal] = unauthenticated
    with TestClient(app) as client:
        response = client.get("/api/v1/users/me/capabilities")
    assert response.status_code == 401


@pytest.mark.parametrize("error", [HTTPException(status_code=503, detail="unavailable"), RuntimeError("guard failed")])
def test_unexpected_guard_failure_is_not_a_denied_capability(app, monkeypatch, error):
    app.dependency_overrides[get_auth_principal] = lambda: AuthPrincipal(kind="user", user_id=7)
    monkeypatch.setattr(users, "RequirePermission", lambda _permission: AsyncMock(side_effect=error), raising=False)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/users/me/capabilities")
    assert response.status_code == getattr(error, "status_code", 500)


def test_discovery_preserves_profile_verification_and_legacy_410(app, monkeypatch):
    app.dependency_overrides[get_auth_principal] = lambda: AuthPrincipal(kind="user", user_id=7)
    monkeypatch.setattr(
        users.AuthnzUsersRepo,
        "from_pool",
        AsyncMock(
            return_value=SimpleNamespace(
                get_user_by_id=AsyncMock(return_value={"id": 7, "is_active": True, "is_verified": False}),
            )
        ),
    )
    monkeypatch.setenv("ENABLE_LEGACY_USER_ME_ENDPOINTS", "false")
    with TestClient(app) as client:
        assert client.get("/api/v1/users/me/capabilities").status_code == 200
        profile = client.get("/api/v1/users/me/profile")
        assert profile.status_code == 403
        assert profile.json()["detail"] == "Email verification required"
        assert client.get("/api/v1/auth/me").status_code == 410
        assert client.get("/api/v1/users/me").status_code == 410
