from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.endpoints import sharing
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User


def test_unsafe_shared_chat_route_is_absent_until_safe_replacement(monkeypatch):
    class _RepoStub:
        async def get_share(self, share_id: int) -> dict[str, object]:
            assert share_id == 12
            return {
                "id": share_id,
                "owner_user_id": 7,
                "workspace_id": "ws-shared",
                "is_revoked": False,
            }

    app = FastAPI()
    app.include_router(sharing.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: User(
        id=11,
        username="reviewer",
        email="reviewer@example.com",
        password_hash="hash",
    )
    monkeypatch.setattr(sharing, "_get_repo", lambda: _RepoStub())
    client = TestClient(app)

    response = client.post(
        "/api/v1/sharing/shared-with-me/12/chat",
        json={"query": "sentinel"},
        follow_redirects=False,
    )

    assert response.status_code == 404
    assert "location" not in response.headers
