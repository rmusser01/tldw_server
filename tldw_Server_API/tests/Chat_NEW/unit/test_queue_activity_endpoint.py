import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.main import app
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, AuthContext


class _QStub:
    def __init__(self, items=None):
        self._items = items or [
            {"request_id": "r1", "client_id": "c1", "priority": 3, "streaming": False, "duration": 0.01, "result": "completed", "ts": 1.0},
            {"request_id": "r2", "client_id": "c2", "priority": 2, "streaming": True,  "duration": 0.02, "result": "stream_completed", "ts": 2.0},
        ]

    def get_recent_activity(self, limit=None):

        if limit is None:
            return list(self._items)
        return list(self._items)[-int(limit):]

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


@pytest.fixture
def auth_override():
    app.dependency_overrides[get_auth_principal] = _principal_override
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_auth_principal, None)


@pytest.mark.unit
def test_queue_activity_endpoint_enabled_default_limit(auth_override):
    with TestClient(app) as client:
        with patch("tldw_Server_API.app.core.Chat.request_queue.get_request_queue", return_value=_QStub()):
            resp = client.get("/api/v1/chat/queue/activity")
            assert resp.status_code == 200
            data = resp.json()
            assert data.get("enabled") is True
            assert isinstance(data.get("activity"), list)
            assert data.get("limit") == 50  # default


@pytest.mark.unit
def test_queue_activity_endpoint_enabled_custom_limit(auth_override):
    with TestClient(app) as client:
        with patch("tldw_Server_API.app.core.Chat.request_queue.get_request_queue", return_value=_QStub()):
            resp = client.get("/api/v1/chat/queue/activity?limit=2")
            assert resp.status_code == 200
            data = resp.json()
            assert data.get("enabled") is True
            assert data.get("limit") == 2
            assert len(data.get("activity")) <= 2


@pytest.mark.unit
def test_queue_activity_endpoint_disabled(auth_override):
    with TestClient(app) as client:
        with patch("tldw_Server_API.app.core.Chat.request_queue.get_request_queue", return_value=None):
            resp = client.get("/api/v1/chat/queue/activity?limit=5")
            assert resp.status_code == 200
            data = resp.json()
            assert data.get("enabled") is False


@pytest.mark.unit
def test_queue_activity_endpoint_limit_too_large(auth_override):
    with TestClient(app) as client:
        with patch("tldw_Server_API.app.core.Chat.request_queue.get_request_queue", return_value=_QStub()):
            resp = client.get("/api/v1/chat/queue/activity?limit=1001")
            assert resp.status_code == 400
            assert "limit" in resp.json().get("detail", "").lower()


@pytest.mark.unit
def test_queue_activity_endpoint_limit_too_small(auth_override):
    with TestClient(app) as client:
        with patch("tldw_Server_API.app.core.Chat.request_queue.get_request_queue", return_value=_QStub()):
            resp = client.get("/api/v1/chat/queue/activity?limit=0")
            assert resp.status_code == 400
            assert "limit" in resp.json().get("detail", "").lower()
