import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.tests.test_config import TestConfig

AUTH_HEADERS = {"X-API-KEY": TestConfig.TEST_API_KEY}


@pytest.fixture(autouse=True)
def flashcards_template_test_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("READING_DIGEST_JOBS_WORKER_ENABLED", "0")
    monkeypatch.setenv("READING_DIGEST_SCHEDULER_ENABLED", "0")
    monkeypatch.setenv("TEST_MODE", "1")


def _build_test_app() -> FastAPI:
    from tldw_Server_API.app.api.v1.endpoints.flashcards import router as flashcards_router

    app = FastAPI()
    app.include_router(flashcards_router, prefix="/api/v1")
    return app


@pytest.fixture
def flashcards_db(tmp_path):
    db_path = tmp_path / "flashcards.db"
    db = CharactersRAGDB(str(db_path), client_id=f"test-{uuid.uuid4().hex[:6]}")
    yield db
    db.close_connection()


@pytest.fixture
def client_with_flashcards_db(flashcards_db: CharactersRAGDB):
    TestConfig.setup_test_environment()
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    app = _build_test_app()

    def override_get_db():
        logger.info("[TEST] override get_chacha_db_for_user -> flashcards_db")
        try:
            yield flashcards_db
        finally:
            pass

    async def override_user():
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    app.dependency_overrides[get_chacha_db_for_user] = override_get_db
    app.dependency_overrides[get_request_user] = override_user

    with TestClient(app, headers=AUTH_HEADERS) as client:
        yield client

    app.dependency_overrides.clear()
    TestConfig.reset_settings()


def test_flashcard_template_routes_use_stable_id(client_with_flashcards_db: TestClient):
    created = client_with_flashcards_db.post(
        "/api/v1/flashcards/templates",
        json={
            "name": "Vocabulary Definition",
            "model_type": "basic",
            "front_template": "What does {{term}} mean?",
            "back_template": "{{definition}}",
            "placeholder_definitions": [
                {
                    "key": "term",
                    "label": "Term",
                    "required": True,
                    "targets": ["front_template"],
                },
                {
                    "key": "definition",
                    "label": "Definition",
                    "required": True,
                    "targets": ["back_template"],
                },
            ],
        },
        headers=AUTH_HEADERS,
    )
    assert created.status_code == 200
    created_payload = created.json()
    template_id = created_payload["id"]

    renamed = client_with_flashcards_db.patch(
        f"/api/v1/flashcards/templates/{template_id}",
        json={
            "name": "Renamed",
            "expected_version": created_payload["version"],
        },
        headers=AUTH_HEADERS,
    )

    assert renamed.status_code == 200
    assert renamed.json()["id"] == template_id
    assert renamed.json()["name"] == "Renamed"


def test_flashcard_template_requires_front_template(client_with_flashcards_db: TestClient):
    response = client_with_flashcards_db.post(
        "/api/v1/flashcards/templates",
        json={
            "name": "Invalid",
            "model_type": "basic",
            "back_template": "Answer",
            "placeholder_definitions": [],
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_flashcard_template_rejects_whitespace_only_placeholder_key(
    client_with_flashcards_db: TestClient,
):
    response = client_with_flashcards_db.post(
        "/api/v1/flashcards/templates",
        json={
            "name": "Whitespace placeholder",
            "model_type": "basic",
            "front_template": "What does {{term}} mean?",
            "back_template": "{{definition}}",
            "placeholder_definitions": [
                {
                    "key": "   ",
                    "label": "Term",
                    "required": True,
                    "targets": ["front_template"],
                },
                {
                    "key": "definition",
                    "label": "Definition",
                    "required": True,
                    "targets": ["back_template"],
                },
            ],
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_flashcard_template_placeholder_targets_are_required_in_openapi_schema(
    client_with_flashcards_db: TestClient,
):
    response = client_with_flashcards_db.get("/openapi.json", headers=AUTH_HEADERS)

    assert response.status_code == 200
    schemas = response.json()["components"]["schemas"]
    placeholder_schema = schemas["FlashcardTemplatePlaceholderDefinition"]
    assert "targets" in placeholder_schema["required"]


def test_flashcard_template_update_missing_returns_not_found(client_with_flashcards_db: TestClient):
    response = client_with_flashcards_db.patch(
        "/api/v1/flashcards/templates/999999",
        json={
            "name": "Missing",
            "expected_version": 1,
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 404


def test_flashcard_template_update_rejects_invalid_merged_non_cloze_state(
    client_with_flashcards_db: TestClient,
):
    created = client_with_flashcards_db.post(
        "/api/v1/flashcards/templates",
        json={
            "name": "Cloze scaffold",
            "model_type": "cloze",
            "front_template": "{{c1::ATP}} powers the cell",
            "back_template": None,
            "placeholder_definitions": [],
        },
        headers=AUTH_HEADERS,
    )
    assert created.status_code == 200
    payload = created.json()

    response = client_with_flashcards_db.patch(
        f"/api/v1/flashcards/templates/{payload['id']}",
        json={
            "model_type": "basic",
            "expected_version": payload["version"],
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 400
    assert "back_template is required" in response.json()["detail"]


def test_flashcard_template_empty_update_is_noop(client_with_flashcards_db: TestClient):
    created = client_with_flashcards_db.post(
        "/api/v1/flashcards/templates",
        json={
            "name": "No-op patch",
            "model_type": "basic",
            "front_template": "What does {{term}} mean?",
            "back_template": "{{definition}}",
            "placeholder_definitions": [
                {
                    "key": "term",
                    "label": "Term",
                    "required": True,
                    "targets": ["front_template"],
                },
                {
                    "key": "definition",
                    "label": "Definition",
                    "required": True,
                    "targets": ["back_template"],
                },
            ],
        },
        headers=AUTH_HEADERS,
    )
    assert created.status_code == 200
    payload = created.json()

    response = client_with_flashcards_db.patch(
        f"/api/v1/flashcards/templates/{payload['id']}",
        json={"expected_version": payload["version"]},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    assert response.json()["version"] == payload["version"]
    assert response.json()["last_modified"] == payload["last_modified"]


def test_flashcard_template_delete_missing_returns_not_found(client_with_flashcards_db: TestClient):
    response = client_with_flashcards_db.delete(
        "/api/v1/flashcards/templates/999999",
        params={"expected_version": 1},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 404


def test_flashcard_template_repeat_delete_returns_not_found(client_with_flashcards_db: TestClient):
    created = client_with_flashcards_db.post(
        "/api/v1/flashcards/templates",
        json={
            "name": "Repeat delete",
            "model_type": "basic",
            "front_template": "What is {{term}}?",
            "back_template": "{{definition}}",
            "placeholder_definitions": [
                {
                    "key": "term",
                    "label": "Term",
                    "required": True,
                    "targets": ["front_template"],
                },
                {
                    "key": "definition",
                    "label": "Definition",
                    "required": True,
                    "targets": ["back_template"],
                },
            ],
        },
        headers=AUTH_HEADERS,
    )
    assert created.status_code == 200
    payload = created.json()

    first_delete = client_with_flashcards_db.delete(
        f"/api/v1/flashcards/templates/{payload['id']}",
        params={"expected_version": payload["version"]},
        headers=AUTH_HEADERS,
    )
    assert first_delete.status_code == 200

    second_delete = client_with_flashcards_db.delete(
        f"/api/v1/flashcards/templates/{payload['id']}",
        params={"expected_version": payload["version"] + 1},
        headers=AUTH_HEADERS,
    )
    assert second_delete.status_code == 404


def test_flashcard_template_update_after_delete_returns_not_found(client_with_flashcards_db: TestClient):
    created = client_with_flashcards_db.post(
        "/api/v1/flashcards/templates",
        json={
            "name": "Delete then patch",
            "model_type": "basic",
            "front_template": "What is {{term}}?",
            "back_template": "{{definition}}",
            "placeholder_definitions": [
                {
                    "key": "term",
                    "label": "Term",
                    "required": True,
                    "targets": ["front_template"],
                },
                {
                    "key": "definition",
                    "label": "Definition",
                    "required": True,
                    "targets": ["back_template"],
                },
            ],
        },
        headers=AUTH_HEADERS,
    )
    assert created.status_code == 200
    payload = created.json()

    deleted = client_with_flashcards_db.delete(
        f"/api/v1/flashcards/templates/{payload['id']}",
        params={"expected_version": payload["version"]},
        headers=AUTH_HEADERS,
    )
    assert deleted.status_code == 200

    response = client_with_flashcards_db.patch(
        f"/api/v1/flashcards/templates/{payload['id']}",
        json={
            "name": "Should not patch",
            "expected_version": payload["version"] + 1,
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 404
