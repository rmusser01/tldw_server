import json
from types import SimpleNamespace

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
from tldw_Server_API.app.core.Persona import connections as persona_connections


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
    db = CharactersRAGDB(str(tmp_path / "persona_connections_api.db"), client_id="persona-connections-api-tests")
    yield db
    db.close_connection()


def _create_persona(client: TestClient, *, name: str) -> str:
    response = client.post(
        "/api/v1/persona/profiles",
        json={"name": name, "mode": "persistent_scoped"},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _add_connection_memory_entry(
    db: CharactersRAGDB,
    *,
    user_id: int,
    persona_id: str,
    connection_id: str,
    content: dict,
) -> None:
    db.add_persona_memory_entry(
        {
            "id": connection_id,
            "persona_id": persona_id,
            "user_id": str(user_id),
            "memory_type": "persona_connection",
            "content": json.dumps(content),
            "salience": 0.0,
        }
    )


def test_persona_connections_create_and_list_are_scoped_and_redacted(persona_db: CharactersRAGDB):
    with _client_for_user(1, persona_db) as client:
        persona_a = _create_persona(client, name="Connection Builder A")
        persona_b = _create_persona(client, name="Connection Builder B")

        created = client.post(
            f"/api/v1/persona/profiles/{persona_a}/connections",
            json={
                "name": "Primary API",
                "base_url": "https://api.example.com/v1",
                "auth_type": "bearer",
                "headers_template": {"X-Client": "voice-builder"},
                "timeout_ms": 12000,
            },
        )
        assert created.status_code == 201, created.text
        created_payload = created.json()
        assert created_payload["persona_id"] == persona_a
        assert created_payload["name"] == "Primary API"
        assert created_payload["allowed_hosts"] == ["api.example.com"]
        assert created_payload["secret_configured"] is False
        assert created_payload["key_hint"] is None

        listed_a = client.get(f"/api/v1/persona/profiles/{persona_a}/connections")
        assert listed_a.status_code == 200, listed_a.text
        list_a_payload = listed_a.json()
        assert len(list_a_payload) == 1
        assert list_a_payload[0]["id"] == created_payload["id"]

        listed_b = client.get(f"/api/v1/persona/profiles/{persona_b}/connections")
        assert listed_b.status_code == 200, listed_b.text
        assert listed_b.json() == []

    fastapi_app.dependency_overrides.clear()


def test_persona_connection_update_rotates_fields_and_preserves_existing_secret(
    persona_db: CharactersRAGDB,
):
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Connection Editor")
        _add_connection_memory_entry(
            persona_db,
            user_id=1,
            persona_id=persona_id,
            connection_id="conn-editable",
            content={
                "name": "Primary API",
                "base_url": "https://api.example.com/v1",
                "auth_type": "bearer",
                "headers_template": {"X-Client": "voice-builder"},
                "timeout_ms": 12000,
                "allowed_hosts": ["api.example.com"],
                "secret_envelope": "opaque-secret-envelope",
                "secret_configured": True,
                "key_hint": "***1234",
            },
        )

        updated = client.put(
            f"/api/v1/persona/profiles/{persona_id}/connections/conn-editable",
            json={
                "name": "Primary API v2",
                "base_url": "https://hooks.example.net/incoming",
                "auth_type": "custom_header",
                "headers_template": {"X-Client": "garden-builder"},
                "timeout_ms": 20000,
            },
        )
        assert updated.status_code == 200, updated.text
        updated_payload = updated.json()
        assert updated_payload["name"] == "Primary API v2"
        assert updated_payload["base_url"] == "https://hooks.example.net/incoming"
        assert updated_payload["auth_type"] == "custom_header"
        assert updated_payload["headers_template"] == {"X-Client": "garden-builder"}
        assert updated_payload["allowed_hosts"] == ["hooks.example.net"]
        assert updated_payload["timeout_ms"] == 20000
        assert updated_payload["secret_configured"] is True
        assert updated_payload["key_hint"] == "***1234"

        listed = client.get(f"/api/v1/persona/profiles/{persona_id}/connections")
        assert listed.status_code == 200, listed.text
        assert listed.json()[0]["id"] == "conn-editable"
        assert listed.json()[0]["allowed_hosts"] == ["hooks.example.net"]

    fastapi_app.dependency_overrides.clear()


def test_persona_connection_delete_soft_deletes_entry(persona_db: CharactersRAGDB):
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Connection Deleter")
        created = client.post(
            f"/api/v1/persona/profiles/{persona_id}/connections",
            json={
                "name": "Webhook API",
                "base_url": "https://api.example.com/hooks",
                "auth_type": "none",
            },
        )
        assert created.status_code == 201, created.text
        connection_id = created.json()["id"]

        deleted = client.delete(
            f"/api/v1/persona/profiles/{persona_id}/connections/{connection_id}"
        )
        assert deleted.status_code == 200, deleted.text
        assert deleted.json() == {
            "status": "deleted",
            "persona_id": persona_id,
            "connection_id": connection_id,
        }

        listed = client.get(f"/api/v1/persona/profiles/{persona_id}/connections")
        assert listed.status_code == 200, listed.text
        assert listed.json() == []

    fastapi_app.dependency_overrides.clear()


def test_persona_connection_test_executes_request_with_redacted_preview(
    persona_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Connection Tester")
        _add_connection_memory_entry(
            persona_db,
            user_id=1,
            persona_id=persona_id,
            connection_id="conn-testable",
            content={
                "name": "Search API",
                "base_url": "https://api.example.com/v1",
                "auth_type": "bearer",
                "headers_template": {"X-Client": "voice-builder"},
                "timeout_ms": 12000,
                "allowed_hosts": ["api.example.com"],
                "secret_envelope": "opaque-secret-envelope",
                "secret_configured": True,
                "key_hint": "***5678",
            },
        )

        captured: dict[str, object] = {}

        class _MockHTTPResponse:
            status_code = 200
            headers = {"content-type": "application/json"}

            def json(self):
                return {"message": "Connection OK"}

            text = '{"message":"Connection OK"}'

        async def fake_afetch(**kwargs):
            captured.update(kwargs)
            return _MockHTTPResponse()

        monkeypatch.setattr(persona_ep, "afetch", fake_afetch, raising=False)
        monkeypatch.setattr(
            persona_connections,
            "evaluate_url_policy",
            lambda url, **kwargs: SimpleNamespace(allowed=True, reason=None),
            raising=False,
        )
        monkeypatch.setattr(
            persona_connections,
            "loads_envelope",
            lambda encrypted_blob: {"encrypted_blob": encrypted_blob},
            raising=False,
        )
        monkeypatch.setattr(
            persona_connections,
            "decrypt_byok_payload",
            lambda envelope: {"api_key": "secret-token"},
            raising=False,
        )

        tested = client.post(
            f"/api/v1/persona/profiles/{persona_id}/connections/conn-testable/test",
            json={
                "method": "POST",
                "path": "search",
                "payload": {"query": "whales"},
            },
        )
        assert tested.status_code == 200, tested.text
        tested_payload = tested.json()
        assert tested_payload["ok"] is True
        assert tested_payload["status_code"] == 200
        assert tested_payload["method"] == "POST"
        assert tested_payload["url"] == "https://api.example.com/v1/search"
        assert tested_payload["request_headers"]["Authorization"] == "[redacted]"
        assert tested_payload["request_headers"]["X-Client"] == "voice-builder"
        assert tested_payload["request_payload"] == {"query": "whales"}
        assert tested_payload["body_preview"] == {"message": "Connection OK"}
        assert captured["headers"]["Authorization"] == "Bearer secret-token"
        assert captured["url"] == "https://api.example.com/v1/search"
        assert captured["json"] == {"query": "whales"}

    fastapi_app.dependency_overrides.clear()


def test_persona_connection_test_derives_allowlist_from_base_url_when_missing(
    persona_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Connection Tester Derived Allowlist")
        _add_connection_memory_entry(
            persona_db,
            user_id=1,
            persona_id=persona_id,
            connection_id="conn-derived-allowlist",
            content={
                "name": "Search API",
                "base_url": "https://api.example.com/v1",
                "auth_type": "none",
                "headers_template": {"X-Client": "voice-builder"},
                "timeout_ms": 12000,
            },
        )

        captured_policy: dict[str, object] = {}

        class _MockHTTPResponse:
            status_code = 200
            headers = {"content-type": "application/json"}

            def json(self):
                return {"message": "Connection OK"}

            text = '{"message":"Connection OK"}'

        async def fake_afetch(**kwargs):
            return _MockHTTPResponse()

        def fake_evaluate_url_policy(url, **kwargs):
            captured_policy["url"] = url
            captured_policy["allowlist"] = kwargs.get("allowlist")
            return SimpleNamespace(allowed=True, reason=None)

        monkeypatch.setattr(persona_ep, "afetch", fake_afetch, raising=False)
        monkeypatch.setattr(
            persona_connections,
            "evaluate_url_policy",
            fake_evaluate_url_policy,
            raising=False,
        )

        tested = client.post(
            f"/api/v1/persona/profiles/{persona_id}/connections/conn-derived-allowlist/test",
            json={
                "method": "POST",
                "path": "search",
                "payload": {"query": "whales"},
            },
        )
        assert tested.status_code == 200, tested.text
        assert tested.json()["ok"] is True
        assert captured_policy == {
            "url": "https://api.example.com/v1/search",
            "allowlist": ["api.example.com"],
        }

    fastapi_app.dependency_overrides.clear()


def test_persona_connection_test_returns_error_for_invalid_stored_secret(
    persona_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    with _client_for_user(1, persona_db) as client:
        persona_id = _create_persona(client, name="Connection Tester Invalid Secret")
        _add_connection_memory_entry(
            persona_db,
            user_id=1,
            persona_id=persona_id,
            connection_id="conn-invalid-secret",
            content={
                "name": "Search API",
                "base_url": "https://api.example.com/v1",
                "auth_type": "bearer",
                "headers_template": {"X-Client": "voice-builder"},
                "timeout_ms": 12000,
                "allowed_hosts": ["api.example.com"],
                "secret_envelope": "{not-json",
                "secret_configured": True,
            },
        )

        async def fail_if_called(**kwargs):
            raise AssertionError("Outbound HTTP should not run when the stored secret is invalid")

        monkeypatch.setattr(persona_ep, "afetch", fail_if_called, raising=False)
        monkeypatch.setattr(
            persona_connections,
            "evaluate_url_policy",
            lambda url, **kwargs: SimpleNamespace(allowed=True, reason=None),
            raising=False,
        )

        tested = client.post(
            f"/api/v1/persona/profiles/{persona_id}/connections/conn-invalid-secret/test",
            json={
                "method": "POST",
                "path": "search",
                "payload": {"query": "whales"},
            },
        )
        assert tested.status_code == 200, tested.text
        tested_payload = tested.json()
        assert tested_payload["ok"] is False
        assert "invalid stored secret" in str(tested_payload["error"]).lower()

    fastapi_app.dependency_overrides.clear()


def test_persona_connection_routes_include_rate_limit_dependency():
    expected_routes = {
        ("/api/v1/persona/profiles/{persona_id}/connections", "GET"),
        ("/api/v1/persona/profiles/{persona_id}/connections", "POST"),
        ("/api/v1/persona/profiles/{persona_id}/connections/{connection_id}", "PUT"),
        ("/api/v1/persona/profiles/{persona_id}/connections/{connection_id}", "DELETE"),
        ("/api/v1/persona/profiles/{persona_id}/connections/{connection_id}/test", "POST"),
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
