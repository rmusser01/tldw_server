from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.endpoints import sharing
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User


def test_unsafe_shared_chat_contract_is_rejected_by_typed_fail_closed_replacement():
    class _AccessServiceStub:
        async def resolve(self, **_kwargs):
            raise AssertionError("malformed chat must fail before access resolution")

    app = FastAPI()
    app.include_router(sharing.router, prefix="/api/v1")
    app.dependency_overrides[get_request_user] = lambda: User(
        id=11,
        username="reviewer",
        email="reviewer@example.com",
        password_hash="hash",
    )
    app.dependency_overrides[sharing.get_shared_workspace_access_service] = (
        lambda: _AccessServiceStub()
    )
    client = TestClient(app)

    response = client.post(
        "/api/v1/sharing/shared-with-me/12/chat",
        json={"query": "sentinel"},
        follow_redirects=False,
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "code": "invalid_shared_chat_request",
            "message": "The shared chat request is invalid.",
            "retryable": False,
        }
    }
    assert "location" not in response.headers
