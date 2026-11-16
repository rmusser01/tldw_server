import pytest
from unittest.mock import patch

from fastapi import HTTPException, status
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.main import app
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, AuthContext


class _QStub:
    def get_queue_status(self):
        return {
            "queue_size": 1,
            "processing_count": 0,
            "max_queue_size": 100,
            "max_concurrent": 4,
            "total_processed": 10,
            "total_rejected": 2,
            "is_running": True,
        }

    def get_recent_activity(self, limit: int = 50):
        return [{"id": 1, "status": "ok", "limit": limit}]


async def _principal_override(request: Request):  # type: ignore[override]
    principal = AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject="admin",
        token_type="access",
        jti=None,
        roles=["admin"],
        permissions=["system.logs"],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )
    request.state.auth = AuthContext(
        principal=principal,
        ip=None,
        user_agent=None,
        request_id=None,
    )
    return principal


async def _limited_principal_override(request: Request):  # type: ignore[override]
    principal = AuthPrincipal(
        kind="user",
        user_id=2,
        api_key_id=None,
        subject="limited",
        token_type="access",
        jti=None,
        roles=["user"],
        permissions=[],
        is_admin=False,
        org_ids=[],
        team_ids=[],
    )
    request.state.auth = AuthContext(
        principal=principal,
        ip=None,
        user_agent=None,
        request_id=None,
    )
    return principal


async def _unauthenticated_principal(_request: Request):  # type: ignore[override]
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated for test",
    )


@pytest.mark.unit
def test_queue_status_endpoint_enabled():
    app.dependency_overrides[get_auth_principal] = _principal_override
    try:
        with TestClient(app) as client:
            with patch("tldw_Server_API.app.core.Chat.request_queue.get_request_queue", return_value=_QStub()):
                resp = client.get("/api/v1/chat/queue/status")
                assert resp.status_code == 200
                data = resp.json()
                assert data.get("enabled") is True
                assert data.get("queue_size") == 1
    finally:
        app.dependency_overrides.pop(get_auth_principal, None)


@pytest.mark.unit
def test_queue_status_endpoint_disabled():
    app.dependency_overrides[get_auth_principal] = _principal_override
    try:
        with TestClient(app) as client:
            with patch("tldw_Server_API.app.core.Chat.request_queue.get_request_queue", return_value=None):
                resp = client.get("/api/v1/chat/queue/status")
                assert resp.status_code == 200
                data = resp.json()
                assert data.get("enabled") is False
    finally:
        app.dependency_overrides.pop(get_auth_principal, None)


@pytest.mark.unit
def test_queue_status_requires_auth_returns_401():
    app.dependency_overrides[get_auth_principal] = _unauthenticated_principal
    try:
        with TestClient(app) as client:
            resp = client.get("/api/v1/chat/queue/status")
            assert resp.status_code == status.HTTP_401_UNAUTHORIZED
    finally:
        app.dependency_overrides.pop(get_auth_principal, None)


@pytest.mark.unit
def test_queue_status_requires_system_logs_permission_returns_403():
    app.dependency_overrides[get_auth_principal] = _limited_principal_override
    try:
        with TestClient(app) as client:
            resp = client.get("/api/v1/chat/queue/status")
            assert resp.status_code == status.HTTP_403_FORBIDDEN
    finally:
        app.dependency_overrides.pop(get_auth_principal, None)


@pytest.mark.unit
def test_queue_activity_requires_auth_and_permissions():
    app.dependency_overrides[get_auth_principal] = _principal_override
    try:
        with TestClient(app) as client:
            with patch("tldw_Server_API.app.core.Chat.request_queue.get_request_queue", return_value=_QStub()):
                resp = client.get("/api/v1/chat/queue/activity?limit=5")
                assert resp.status_code == 200
                data = resp.json()
                assert data.get("enabled") is True
                assert data.get("limit") == 5
    finally:
        app.dependency_overrides.pop(get_auth_principal, None)
