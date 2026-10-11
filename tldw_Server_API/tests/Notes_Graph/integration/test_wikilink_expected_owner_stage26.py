"""Captured Notes lookup owners are checked before user data is loaded."""

from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import notes
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("path", ["/search", "/search/", "/wikilinks/resolve"])
@pytest.mark.parametrize("expected_owner", ["1", "2", None])
def test_lookup_checks_captured_owner_before_database(
    tmp_path, path: str, expected_owner: str | None
) -> None:
    """Finite authenticated-principal double, real dependency and temporary DB."""

    db = CharactersRAGDB(str(tmp_path / "lookup.db"), client_id="2")
    db.add_note(title="Owner B", content="Only owner B's note")
    app = FastAPI()
    app.include_router(notes.router, prefix="/api/v1/notes")
    database_read = Mock(return_value=db)

    def database():
        return database_read()

    async def principal() -> AuthPrincipal:
        return AuthPrincipal(kind="user", user_id=2, is_admin=True)

    async def user() -> User:
        return User(id=2, username="owner-b", is_active=True, is_admin=True)

    class Limiter:
        async def check_user_rate_limit(self, *_args):
            return True, {}

    app.dependency_overrides[auth_deps.get_auth_principal] = principal
    app.dependency_overrides[get_request_user] = user
    app.dependency_overrides[notes.get_chacha_db_for_user] = database
    app.dependency_overrides[notes.get_rate_limiter_dep] = lambda: Limiter()
    headers = (
        {"X-TLDW-Expected-User-ID": expected_owner}
        if expected_owner is not None
        else {}
    )
    try:
        with TestClient(app) as client:
            if path == "/wikilinks/resolve":
                response = client.post(
                    f"/api/v1/notes{path}", json={"titles": ["Owner B"]}, headers=headers
                )
            else:
                response = client.get(
                    f"/api/v1/notes{path}", params={"query": "Owner B", "title_only": True}, headers=headers
                )
        if expected_owner == "1":
            assert response.status_code == 412, response.text
            assert response.json()["detail"]["code"] == "request_config_scope_changed"
            assert response.headers["Cache-Control"] == "no-store"
            database_read.assert_not_called()
        else:
            assert response.status_code == 200, response.text
            database_read.assert_called_once()
            assert "Owner B" in response.text
    finally:
        app.dependency_overrides.clear()
        db.close_connection()
