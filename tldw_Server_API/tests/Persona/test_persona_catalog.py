import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit

fastapi_app = FastAPI()
fastapi_app.include_router(persona_ep.router, prefix="/api/v1/persona")


@pytest.fixture()
def persona_db(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "persona_catalog.db"), client_id="persona-catalog-tests")
    yield db
    db.close_connection()


@pytest.fixture()
def client_with_persona_user(persona_db):

    async def override_user():
        return User(id=1, username="tester", email=None, is_active=True)

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_auth_principal] = lambda: AuthPrincipal(kind="user", user_id=1)
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: persona_db
    with TestClient(fastapi_app) as client:
        yield client
    fastapi_app.dependency_overrides.clear()


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


@pytest.mark.parametrize("expected_user_id", [None, "1"], ids=["unasserted", "matching"])
def test_persona_catalog_read_only_empty_does_not_create_default(
    client_with_persona_user: TestClient,
    persona_db: CharactersRAGDB,
    expected_user_id: str | None,
):
    headers = {} if expected_user_id is None else {"X-TLDW-Expected-User-ID": expected_user_id}
    response = client_with_persona_user.get("/api/v1/persona/catalog?ensure_default=false", headers=headers)

    assert response.status_code == 200, response.text
    assert response.json() == []
    assert persona_db.list_persona_profiles(user_id="1") == []
    assert persona_db.list_persona_policy_rules(persona_id="research_assistant", user_id="1") == []


@pytest.mark.parametrize("query", ["", "?ensure_default=true"], ids=["omitted", "explicit"])
@pytest.mark.parametrize("expected_user_id", [None, "1"], ids=["unasserted", "matching"])
def test_persona_catalog_preserves_default_creation(
    client_with_persona_user: TestClient,
    persona_db: CharactersRAGDB,
    query: str,
    expected_user_id: str | None,
):
    headers = {} if expected_user_id is None else {"X-TLDW-Expected-User-ID": expected_user_id}
    response = client_with_persona_user.get(f"/api/v1/persona/catalog{query}", headers=headers)

    assert response.status_code == 200, response.text
    assert [item["id"] for item in response.json()] == ["research_assistant"]
    assert persona_db.get_persona_profile("research_assistant", user_id="1") is not None
    assert persona_db.list_persona_policy_rules(persona_id="research_assistant", user_id="1")


def test_persona_catalog_read_only_preserves_active_owned_profile_projection(
    client_with_persona_user: TestClient,
    persona_db: CharactersRAGDB,
):
    for persona_id, user_id, active in (
        ("owned-active", "1", True),
        ("owned-inactive", "1", False),
        ("foreign", "2", True),
    ):
        persona_db.create_persona_profile(
            {"id": persona_id, "user_id": user_id, "name": persona_id, "is_active": active}
        )
    persona_db.replace_persona_policy_rules(
        persona_id="owned-active",
        user_id="1",
        rules=[{"rule_kind": "mcp_tool", "rule_name": "media.search", "allowed": True}],
    )
    profiles_before = persona_db.list_persona_profiles(user_id="1")
    policies_before = persona_db.list_persona_policy_rules(persona_id="owned-active", user_id="1")

    response = client_with_persona_user.get(
        "/api/v1/persona/catalog?ensure_default=false",
        headers={"X-TLDW-Expected-User-ID": "1"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert [item["id"] for item in payload] == ["owned-active"]
    assert payload[0]["default_tools"] == ["media.search"]
    assert payload[0]["buddy_summary"]["persona_name"] == "owned-active"
    assert persona_db.list_persona_profiles(user_id="1") == profiles_before
    assert persona_db.list_persona_policy_rules(persona_id="owned-active", user_id="1") == policies_before


@pytest.mark.parametrize("query", ["", "?ensure_default=true", "?ensure_default=false"])
@pytest.mark.parametrize("blocked_work", ["database", "feature"])
def test_persona_catalog_rejects_mismatched_user_before_work(
    client_with_persona_user: TestClient,
    monkeypatch,
    query: str,
    blocked_work: str,
):
    work_calls = []

    def unexpected_work():
        work_calls.append(blocked_work)
        raise HTTPException(status_code=500, detail="Work started before scope check")

    if blocked_work == "database":
        fastapi_app.dependency_overrides[get_chacha_db_for_user] = unexpected_work
    else:
        monkeypatch.setattr(persona_ep, "is_persona_enabled", unexpected_work)

    response = client_with_persona_user.get(f"/api/v1/persona/catalog{query}", headers={"X-TLDW-Expected-User-ID": "2"})

    assert response.status_code == 412, response.text
    assert work_calls == []
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["detail"]["code"] == "request_config_scope_changed"


def test_persona_catalog_returns_404_when_disabled(client_with_persona_user: TestClient, monkeypatch):
    monkeypatch.setattr(persona_ep, "is_persona_enabled", lambda: False)
    r = client_with_persona_user.get("/api/v1/persona/catalog")
    assert r.status_code == 404


def test_persona_session_returns_404_when_disabled(client_with_persona_user: TestClient, monkeypatch):
    monkeypatch.setattr(persona_ep, "is_persona_enabled", lambda: False)
    r = client_with_persona_user.post("/api/v1/persona/session", json={"persona_id": "research_assistant"})
    assert r.status_code == 404
