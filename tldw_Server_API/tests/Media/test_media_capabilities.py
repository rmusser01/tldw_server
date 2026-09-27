"""Media affordances report the same permissions enforced by mutations."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal


@pytest.mark.unit
@pytest.mark.parametrize("permissions,roles,can_delete", [
    (["media.read", "media.create"], ["user"], False),
    (["media.delete"], ["user"], True),
    ([], ["admin"], True),
])
def test_media_capabilities_match_delete_permission(permissions, roles, can_delete):
    from tldw_Server_API.app.api.v1.endpoints.media.capabilities import router
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/media")
    app.dependency_overrides[get_auth_principal] = lambda: AuthPrincipal(
        kind="user", user_id=2, permissions=permissions, roles=roles,
    )
    with TestClient(app) as client:
        response = client.get("/api/v1/media/capabilities")
    assert response.status_code == 200
    assert response.json() == {"can_delete": can_delete}
