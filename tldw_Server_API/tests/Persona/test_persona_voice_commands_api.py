import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.unit

fastapi_app = FastAPI()
fastapi_app.include_router(persona_ep.router, prefix="/api/v1/persona")


def _client_for_user(user_id: int, db: CharactersRAGDB):
    async def override_user():
        return User(id=user_id, username=f"persona-user-{user_id}", email=None, is_active=True)

    fastapi_app.dependency_overrides[get_request_user] = override_user
    fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    return TestClient(fastapi_app)


@pytest.fixture()
def persona_db(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "persona_voice_commands_api.db"), client_id="persona-voice-commands-api-tests")
    yield db
    db.close_connection()


def _create_persona(client: TestClient, *, name: str) -> str:
    response = client.post(
        "/api/v1/persona/profiles",
        json={"name": name, "mode": "persistent_scoped"},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_persona_voice_command_crud_is_scoped_to_selected_persona(persona_db: CharactersRAGDB):
    with _client_for_user(1, persona_db) as client:
        persona_a = _create_persona(client, name="Voice Builder A")
        persona_b = _create_persona(client, name="Voice Builder B")

        created = client.post(
            f"/api/v1/persona/profiles/{persona_a}/voice-commands",
            json={
                "name": "Search Notes",
                "phrases": ["search notes for {topic}"],
                "action_type": "mcp_tool",
                "action_config": {"tool_name": "notes.search"},
                "priority": 10,
                "enabled": True,
                "requires_confirmation": False,
                "description": "Search saved notes by topic",
            },
        )
        assert created.status_code == 201, created.text
        created_payload = created.json()
        command_id = created_payload["id"]
        assert created_payload["persona_id"] == persona_a
        assert created_payload["connection_id"] is None

        list_a = client.get(f"/api/v1/persona/profiles/{persona_a}/voice-commands")
        assert list_a.status_code == 200, list_a.text
        list_a_payload = list_a.json()
        assert list_a_payload["total"] == 1
        assert list_a_payload["commands"][0]["id"] == command_id

        list_b = client.get(f"/api/v1/persona/profiles/{persona_b}/voice-commands")
        assert list_b.status_code == 200, list_b.text
        assert list_b.json()["total"] == 0

        updated = client.put(
            f"/api/v1/persona/profiles/{persona_a}/voice-commands/{command_id}",
            json={
                "name": "Search Notes Quickly",
                "phrases": ["search notes for {topic}", "find notes about {topic}"],
                "action_type": "mcp_tool",
                "action_config": {"tool_name": "notes.search"},
                "priority": 25,
                "enabled": True,
                "requires_confirmation": False,
                "description": "Search notes with parameter extraction",
            },
        )
        assert updated.status_code == 200, updated.text
        updated_payload = updated.json()
        assert updated_payload["name"] == "Search Notes Quickly"
        assert updated_payload["priority"] == 25
        assert updated_payload["persona_id"] == persona_a

        toggled = client.post(
            f"/api/v1/persona/profiles/{persona_a}/voice-commands/{command_id}/toggle",
            json={"enabled": False},
        )
        assert toggled.status_code == 200, toggled.text
        assert toggled.json()["enabled"] is False

        list_with_disabled = client.get(
            f"/api/v1/persona/profiles/{persona_a}/voice-commands?include_disabled=true"
        )
        assert list_with_disabled.status_code == 200, list_with_disabled.text
        assert list_with_disabled.json()["commands"][0]["enabled"] is False

        deleted = client.delete(f"/api/v1/persona/profiles/{persona_a}/voice-commands/{command_id}")
        assert deleted.status_code == 204, deleted.text

        after_delete = client.get(
            f"/api/v1/persona/profiles/{persona_a}/voice-commands?include_disabled=true"
        )
        assert after_delete.status_code == 200, after_delete.text
        assert after_delete.json()["total"] == 0

    fastapi_app.dependency_overrides.clear()


def test_persona_voice_command_rejects_mismatched_persona_payload(persona_db: CharactersRAGDB):
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Voice Builder")

        response = client.post(
            f"/api/v1/persona/profiles/{persona_id}/voice-commands",
            json={
                "persona_id": "other-persona",
                "name": "Search Notes",
                "phrases": ["search notes for {topic}"],
                "action_type": "mcp_tool",
                "action_config": {"tool_name": "notes.search"},
            },
        )
        assert response.status_code == 400, response.text

    fastapi_app.dependency_overrides.clear()


def test_persona_voice_command_reports_missing_connection_after_connection_delete(
    persona_db: CharactersRAGDB,
):
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Voice Builder With Connection")

        created_connection = client.post(
            f"/api/v1/persona/profiles/{persona_id}/connections",
            json={
                "name": "Webhook API",
                "base_url": "https://api.example.com/hooks",
                "auth_type": "none",
            },
        )
        assert created_connection.status_code == 201, created_connection.text
        connection_id = created_connection.json()["id"]

        created_command = client.post(
            f"/api/v1/persona/profiles/{persona_id}/voice-commands",
            json={
                "name": "Send Webhook",
                "phrases": ["send webhook for {query}"],
                "action_type": "custom",
                "connection_id": connection_id,
                "action_config": {
                    "action": "external_request",
                    "method": "POST",
                    "path": "search",
                },
                "priority": 10,
                "enabled": True,
                "requires_confirmation": True,
            },
        )
        assert created_command.status_code == 201, created_command.text
        created_command_payload = created_command.json()
        assert created_command_payload["connection_status"] == "ok"
        assert created_command_payload["connection_name"] == "Webhook API"

        deleted_connection = client.delete(
            f"/api/v1/persona/profiles/{persona_id}/connections/{connection_id}"
        )
        assert deleted_connection.status_code == 200, deleted_connection.text

        listed = client.get(
            f"/api/v1/persona/profiles/{persona_id}/voice-commands?include_disabled=true"
        )
        assert listed.status_code == 200, listed.text
        payload = listed.json()
        assert payload["total"] == 1
        assert payload["commands"][0]["connection_id"] == connection_id
        assert payload["commands"][0]["connection_status"] == "missing"
        assert payload["commands"][0]["connection_name"] is None

    fastapi_app.dependency_overrides.clear()


def test_persona_voice_command_routes_include_rate_limit_dependency():
    expected_routes = {
        ("/api/v1/persona/profiles/{persona_id}/voice-commands", "GET"),
        ("/api/v1/persona/profiles/{persona_id}/voice-commands", "POST"),
        ("/api/v1/persona/profiles/{persona_id}/voice-commands/{command_id}", "PUT"),
        (
            "/api/v1/persona/profiles/{persona_id}/voice-commands/{command_id}/toggle",
            "POST",
        ),
        ("/api/v1/persona/profiles/{persona_id}/voice-commands/{command_id}", "DELETE"),
        ("/api/v1/persona/profiles/{persona_id}/voice-commands/test", "POST"),
    }

    seen_routes: set[tuple[str, str]] = set()
    for route in iter_served_routes(fastapi_app.routes):
        if not isinstance(route.route, APIRoute):
            continue
        for method in route.methods:
            key = (route.path, method)
            if key not in expected_routes:
                continue
            seen_routes.add(key)
            dependencies = [dependency.call for dependency in route.dependant.dependencies]
            assert check_rate_limit in dependencies, key

    assert seen_routes == expected_routes
