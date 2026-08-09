# Tests/test_notes_restore.py
#
# Tests for the note restore endpoint: POST /api/v1/notes/{note_id}/restore
#
# Imports
import pytest
pytestmark = pytest.mark.unit
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from unittest.mock import call
from datetime import datetime, timezone
import uuid

# Local Imports
from tldw_Server_API.app.api.v1.endpoints import notes as notes_router_module
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDBError as Actual_CharactersRAGDBError,
    ConflictError as Actual_ConflictError,
    CharactersRAGDB,
)


# --- Mocked DB and Dependency Override ---
mock_chacha_db_instance = MagicMock(spec=CharactersRAGDB)


async def override_get_chacha_db_for_user():
    mock_chacha_db_instance.client_id = "test_api_client_for_user_db"
    return mock_chacha_db_instance


class _StubRateLimiter:
    """Simple stub that always allows requests for tests."""

    def __init__(self):
        self.enabled = True

    async def check_user_rate_limit(self, user_id: int, endpoint: str, role: str = "user"):
        return True, {"limit": 0, "remaining": None}


_stub_rate_limiter = _StubRateLimiter()


async def override_get_rate_limiter_dep():
    return _stub_rate_limiter


@pytest.fixture(scope="module")
def test_app():
    app = FastAPI()
    app.include_router(notes_router_module.router, prefix="/api/v1/notes", tags=["Notes"])
    app.dependency_overrides[notes_router_module.get_chacha_db_for_user] = override_get_chacha_db_for_user
    app.dependency_overrides[notes_router_module.get_rate_limiter_dep] = override_get_rate_limiter_dep

    async def _override_user():
        return User(id=1, username="tester", email="t@e.com", is_active=True, is_admin=True)

    app.dependency_overrides[notes_router_module.get_request_user] = _override_user
    return app


@pytest.fixture(scope="module")
def client(test_app: FastAPI):
    return TestClient(test_app)


@pytest.fixture(autouse=True)
def reset_db_mock_calls():
    mock_chacha_db_instance.reset_mock()
    mock_chacha_db_instance.restore_note.side_effect = None
    mock_chacha_db_instance.get_note_by_id.side_effect = None
    mock_chacha_db_instance.get_keywords_for_note.side_effect = None
    mock_chacha_db_instance.add_note.side_effect = None
    mock_chacha_db_instance.soft_delete_note.side_effect = None


def create_timestamped_data(base_data: dict, client_id: str) -> dict:
    """Helper to create note data with timestamps."""
    now = datetime.now(timezone.utc)
    default_data = {
        "created_at": now.isoformat(),
        "last_modified": now.isoformat(),
        "version": 1,
        "client_id": client_id,
        "deleted": False,
    }
    if "title" in base_data and "content" not in base_data:
        base_data.setdefault("content", "Default test content")
    return {**default_data, **base_data}


def create_keyword_data(keyword_id: int, keyword: str, client_id: str) -> dict:
    """Helper to create keyword data matching KeywordResponse schema."""
    now = datetime.now(timezone.utc)
    return {
        "id": keyword_id,
        "sync_id": str(uuid.uuid4()),
        "keyword": keyword,
        "created_at": now.isoformat(),
        "last_modified": now.isoformat(),
        "version": 1,
        "client_id": client_id,
        "deleted": False,
    }


# --- Test Cases for Restore Endpoint ---


def test_restore_note_success(client: TestClient):
    """Happy path: restore a deleted note, verify 200 + NoteResponse."""
    note_id_val = str(uuid.uuid4())
    expected_version = 2
    expected_db_client_id = "test_api_client_for_user_db"

    mock_chacha_db_instance.restore_note.return_value = True
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {
            "id": note_id_val,
            "title": "Restored Note",
            "content": "This note was restored",
            "version": expected_version + 1,
            "deleted": False,
        },
        expected_db_client_id,
    )
    mock_chacha_db_instance.get_keywords_for_note.return_value = []

    response = client.post(f"/api/v1/notes/{note_id_val}/restore?expected_version={expected_version}")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["id"] == note_id_val
    assert data["title"] == "Restored Note"
    assert data["deleted"] is False
    assert data["version"] == expected_version + 1

    mock_chacha_db_instance.restore_note.assert_called_once_with(
        note_id=note_id_val, expected_version=expected_version
    )
    assert mock_chacha_db_instance.get_note_by_id.call_count == 2
    mock_chacha_db_instance.get_note_by_id.assert_has_calls(
        [
            call(note_id=note_id_val, include_deleted=True),
            call(note_id_val),
        ]
    )
    mock_chacha_db_instance.get_keywords_for_note.assert_called_once_with(note_id_val)


def test_restore_note_not_found(client: TestClient):
    """Note doesn't exist -> 404."""
    note_id_val = str(uuid.uuid4())
    expected_version = 1

    mock_chacha_db_instance.restore_note.side_effect = Actual_ConflictError(
        f"Note ID {note_id_val} not found.",
        entity="notes",
        entity_id=note_id_val,
    )

    response = client.post(f"/api/v1/notes/{note_id_val}/restore?expected_version={expected_version}")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found" in response.json()["detail"].lower()


def test_restore_note_version_conflict(client: TestClient):
    """Wrong expected_version -> 409."""
    note_id_val = str(uuid.uuid4())
    wrong_version = 1
    actual_version = 3

    mock_chacha_db_instance.restore_note.side_effect = Actual_ConflictError(
        f"Restore for Note ID {note_id_val} failed: version mismatch (db has {actual_version}, client expected {wrong_version}).",
        entity="notes",
        entity_id=note_id_val,
    )

    response = client.post(f"/api/v1/notes/{note_id_val}/restore?expected_version={wrong_version}")

    assert response.status_code == status.HTTP_409_CONFLICT
    assert "modified since you last fetched" in response.json()["detail"]


def test_restore_note_already_active(client: TestClient):
    """Note not deleted -> 200 (idempotent behavior)."""
    note_id_val = str(uuid.uuid4())
    expected_version = 1
    expected_db_client_id = "test_api_client_for_user_db"

    # The DB method returns True for idempotency (note already active)
    mock_chacha_db_instance.restore_note.return_value = True
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {
            "id": note_id_val,
            "title": "Already Active Note",
            "content": "This note was never deleted",
            "version": expected_version,
            "deleted": False,
        },
        expected_db_client_id,
    )
    mock_chacha_db_instance.get_keywords_for_note.return_value = []

    response = client.post(f"/api/v1/notes/{note_id_val}/restore?expected_version={expected_version}")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["id"] == note_id_val
    assert data["deleted"] is False


def test_restore_note_missing_version_param(client: TestClient):
    """No expected_version query param -> 422 validation error."""
    note_id_val = str(uuid.uuid4())

    response = client.post(f"/api/v1/notes/{note_id_val}/restore")

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    # The error detail should mention the missing required query parameter
    assert "expected_version" in response.text.lower()


def test_restore_note_with_keywords(client: TestClient):
    """Restored note includes keywords in response."""
    note_id_val = str(uuid.uuid4())
    expected_version = 2
    expected_db_client_id = "test_api_client_for_user_db"

    mock_chacha_db_instance.restore_note.return_value = True
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {
            "id": note_id_val,
            "title": "Restored Note with Keywords",
            "content": "Content here",
            "version": expected_version + 1,
            "deleted": False,
        },
        expected_db_client_id,
    )
    mock_chacha_db_instance.get_keywords_for_note.return_value = [
        create_keyword_data(1, "important", expected_db_client_id),
        create_keyword_data(2, "research", expected_db_client_id),
    ]

    response = client.post(f"/api/v1/notes/{note_id_val}/restore?expected_version={expected_version}")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data["keywords"]) == 2
    assert data["keywords"][0]["keyword"] == "important"
    assert data["keywords"][1]["keyword"] == "research"


def test_restore_note_db_error(client: TestClient):
    """DB error during restore -> 500."""
    note_id_val = str(uuid.uuid4())
    expected_version = 1

    mock_chacha_db_instance.restore_note.side_effect = Actual_CharactersRAGDBError(
        "Database connection failed"
    )

    response = client.post(f"/api/v1/notes/{note_id_val}/restore?expected_version={expected_version}")

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "database error" in response.json()["detail"].lower()


def test_restore_note_not_found_after_restore(client: TestClient):
    """Restore succeeds but note not retrievable -> 404."""
    note_id_val = str(uuid.uuid4())
    expected_version = 2

    mock_chacha_db_instance.restore_note.return_value = True
    mock_chacha_db_instance.get_note_by_id.return_value = None

    response = client.post(f"/api/v1/notes/{note_id_val}/restore?expected_version={expected_version}")

    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "not found after restore" in response.json()["detail"]


# --- Integration Test: Delete and Restore Lifecycle ---


def test_delete_and_restore_note_lifecycle(client: TestClient):
    """
    Full lifecycle: Create -> Delete -> Restore.
    Tests that a note can be soft-deleted and then restored.
    """
    note_id_val = str(uuid.uuid4())
    expected_db_client_id = "test_api_client_for_user_db"

    # Step 1: Create note
    mock_chacha_db_instance.add_note.return_value = note_id_val
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {
            "id": note_id_val,
            "title": "Lifecycle Test Note",
            "content": "Testing delete and restore",
            "version": 1,
            "deleted": False,
        },
        expected_db_client_id,
    )
    mock_chacha_db_instance.get_keywords_for_note.return_value = []

    create_response = client.post(
        "/api/v1/notes/",
        json={"title": "Lifecycle Test Note", "content": "Testing delete and restore"},
    )
    assert create_response.status_code == status.HTTP_201_CREATED
    created_note = create_response.json()
    assert created_note["version"] == 1

    # Step 2: Delete note
    mock_chacha_db_instance.soft_delete_note.return_value = True

    delete_response = client.delete(
        f"/api/v1/notes/{note_id_val}",
        headers={"expected-version": "1"},
    )
    assert delete_response.status_code == status.HTTP_204_NO_CONTENT
    mock_chacha_db_instance.soft_delete_note.assert_called_once_with(
        note_id=note_id_val, expected_version=1
    )

    # Step 3: Restore note (version is now 2 after soft delete)
    mock_chacha_db_instance.restore_note.return_value = True
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {
            "id": note_id_val,
            "title": "Lifecycle Test Note",
            "content": "Testing delete and restore",
            "version": 3,  # Incremented after restore
            "deleted": False,
        },
        expected_db_client_id,
    )

    restore_response = client.post(
        f"/api/v1/notes/{note_id_val}/restore?expected_version=2"
    )
    assert restore_response.status_code == status.HTTP_200_OK
    restored_note = restore_response.json()
    assert restored_note["id"] == note_id_val
    assert restored_note["deleted"] is False
    assert restored_note["version"] == 3

    mock_chacha_db_instance.restore_note.assert_called_once_with(
        note_id=note_id_val, expected_version=2
    )


#
# End of test_notes_restore.py
