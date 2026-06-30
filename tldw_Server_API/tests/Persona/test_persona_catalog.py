import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.unit

fastapi_app = FastAPI()
fastapi_app.include_router(persona_ep.router, prefix="/api/v1/persona")


@pytest.fixture()
def client_with_persona_user(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "persona_catalog.db"), client_id="persona-catalog-tests")

    async def override_user():
        return User(id=1, username="tester", email=None, is_active=True)

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    with TestClient(fastapi_app) as client:
        yield client
    fastapi_app.dependency_overrides.clear()
    db.close_connection()


def test_persona_catalog_requires_auth():
    with TestClient(fastapi_app) as c:
        r = c.get("/api/v1/persona/catalog")
        assert r.status_code == 401


def test_persona_session_requires_auth():
    with TestClient(fastapi_app) as c:
        r = c.post("/api/v1/persona/session", json={"persona_id": "research_assistant"})
        assert r.status_code == 401


def test_persona_catalog_smoke(client_with_persona_user: TestClient):
    r = client_with_persona_user.get("/api/v1/persona/catalog")
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload, list)
    assert payload
    assert payload[0]["id"] == "research_assistant"
    assert payload[0]["mode"] == "session_scoped"


def test_persona_catalog_returns_404_when_disabled(client_with_persona_user: TestClient, monkeypatch):
    monkeypatch.setattr(persona_ep, "is_persona_enabled", lambda: False)
    r = client_with_persona_user.get("/api/v1/persona/catalog")
    assert r.status_code == 404


def test_persona_session_returns_404_when_disabled(client_with_persona_user: TestClient, monkeypatch):
    monkeypatch.setattr(persona_ep, "is_persona_enabled", lambda: False)
    r = client_with_persona_user.post("/api/v1/persona/session", json={"persona_id": "research_assistant"})
    assert r.status_code == 404
