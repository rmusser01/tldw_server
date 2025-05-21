# Tests/test_notes_api_integration.py
#
#
# Imports
import pytest
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import uuid
#
# Third-Party Imports
#
# Local Imports
from tldw_Server_API.app.api.v1.schemas.notes_schemas import (
    NoteCreate, NoteUpdate, NoteResponse,
    KeywordCreate, KeywordResponse,
    NoteKeywordLinkResponse, KeywordsForNoteResponse, NotesForKeywordResponse
)
from tldw_Server_API.app.api.v1.endpoints import notes as notes_router_module
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDBError as Actual_CharactersRAGDBError,
    InputError as Actual_InputError,
    ConflictError as Actual_ConflictError,
    CharactersRAGDB # Import the class for spec if needed, or type hinting
)
#
########################################################################################################################
#
# Functions:

# FIXME - Write more tests
# Full CRUD for Keywords: Test create, get, list, delete, search for keywords.
# All Linking/Unlinking Operations:
#     unlink_note_from_keyword
#     get_notes_for_keyword
# Validation Tests:
#     Invalid limit / offset parameters (e.g., negative, too large).
#     Missing expected-version header where required.
#     Invalid UUID format for note_id.
#     Payloads that fail Pydantic validation (e.g., title too long, wrong data types).
# Authentication/Authorization (if applicable): If your get_chacha_db_for_user actually involves user authentication, you'd need to mock that or test with valid/invalid credentials.
# Concurrency Tests (Advanced): These are harder with TestClient but testing how the API handles ConflictError due to version mismatches is a good start.
# Empty States: Test listing notes/keywords when none exist.
# Pagination Edge Cases:
#     Requesting an offset that's beyond the total number of items.
#     Requesting limit 0 (if allowed, or test for error if not).

# Import exceptions that the API layer might expect from the DB layer
# These should ideally be the actual exceptions from ChaChaNotes_DB
# For this example, we use the dummy ones if defined in conftest or here for clarity.
# If you have the real exceptions, mock their paths.
MODULE_PATH_PREFIX = "tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB"

MockCharactersRAGDBError = type('CharactersRAGDBError', (Exception,), {})
MockInputError = type('InputError', (MockCharactersRAGDBError,), {})
MockConflictError = type('ConflictError', (MockCharactersRAGDBError,), {'entity': None, 'entity_id': None})

# --- Mocked DB and Dependency Override ---
mock_chacha_db_instance = MagicMock(spec=CharactersRAGDB)


async def override_get_chacha_db_for_user():
    mock_chacha_db_instance.client_id = "test_api_client_for_user_db"
    return mock_chacha_db_instance


@pytest.fixture(scope="module")
def test_app():
    app = FastAPI()
    app.include_router(notes_router_module.router, prefix="/api/v1/notes", tags=["Notes"])
    app.dependency_overrides[notes_router_module.get_chacha_db_for_user] = override_get_chacha_db_for_user
    return app


@pytest.fixture(scope="module")
def client(test_app: FastAPI):
    return TestClient(test_app)


@pytest.fixture(autouse=True)
def reset_db_mock_calls():
    mock_chacha_db_instance.reset_mock()
    mock_chacha_db_instance.add_note.side_effect = None
    mock_chacha_db_instance.get_note_by_id.side_effect = None
    mock_chacha_db_instance.update_note.side_effect = None
    mock_chacha_db_instance.soft_delete_note.side_effect = None
    mock_chacha_db_instance.list_notes.side_effect = None
    mock_chacha_db_instance.search_notes.side_effect = None
    mock_chacha_db_instance.add_keyword.side_effect = None
    mock_chacha_db_instance.get_keyword_by_id.side_effect = None
    mock_chacha_db_instance.link_note_to_keyword.side_effect = None
    mock_chacha_db_instance.get_keywords_for_note.side_effect = None


def create_timestamped_data(base_data: Dict[str, Any], client_id: str) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    default_data = {
        "created_at": now.isoformat(),
        "last_modified": now.isoformat(),
        "version": 1,
        "client_id": client_id,
        "deleted": False,
    }
    # Ensure required fields like 'content' for NoteResponse are present if not in base_data
    # For NoteResponse specifically
    if 'title' in base_data and 'content' not in base_data:  # Heuristic: if it looks like a note
        base_data.setdefault('content', 'Default test content')

    return {**default_data, **base_data}


# --- Test Cases ---

def test_create_note(client: TestClient):
    note_id_val = str(uuid.uuid4())
    note_create_payload = {"title": "New Note", "content": "Note content", "id": note_id_val}
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.add_note.return_value = note_id_val
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {"id": note_id_val, "title": "New Note", "content": "Note content"},  # Content included
        expected_db_client_id
    )
    response = client.post("/api/v1/notes/", json=note_create_payload)
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["title"] == "New Note"
    assert data["id"] == note_id_val
    assert data["client_id"] == expected_db_client_id
    mock_chacha_db_instance.add_note.assert_called_once_with(
        title="New Note", content="Note content", note_id=note_id_val
    )
    mock_chacha_db_instance.get_note_by_id.assert_called_once_with(note_id=note_id_val)


def test_create_note_db_error(client: TestClient):
    note_create_payload = {"title": "Error Note", "content": "Content"}
    mock_chacha_db_instance.add_note.side_effect = Actual_CharactersRAGDBError("DB connection failed")
    response = client.post("/api/v1/notes/", json=note_create_payload)
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "A database error occurred while processing your request for note." in response.json()["detail"]


def test_create_note_input_error(client: TestClient):
    valid_payload_for_api = {"title": "Valid Title", "content": "Content"}
    mock_chacha_db_instance.add_note.side_effect = Actual_InputError("DB says: Title cannot be empty")
    response = client.post("/api/v1/notes/", json=valid_payload_for_api)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "DB says: Title cannot be empty" in response.json()["detail"]


def test_create_note_conflict_error(client: TestClient):
    note_id_val = str(uuid.uuid4())
    note_create_payload = {"title": "Conflict Note", "content": "Content", "id": note_id_val}
    mock_chacha_db_instance.add_note.side_effect = Actual_ConflictError(
        message=f"Note with ID '{note_id_val}' already exists.", entity="note", entity_id=note_id_val
    )
    response = client.post("/api/v1/notes/", json=note_create_payload)
    assert response.status_code == status.HTTP_409_CONFLICT
    assert f"A conflict occurred with note (ID: {note_id_val})." in response.json()["detail"]


def test_get_note(client: TestClient):
    note_id_val = str(uuid.uuid4())
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {"id": note_id_val, "title": "Fetched Note", "content": "Content"},  # Content included
        expected_db_client_id
    )
    response = client.get(f"/api/v1/notes/{note_id_val}")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["id"] == note_id_val
    mock_chacha_db_instance.get_note_by_id.assert_called_once_with(note_id=note_id_val)


def test_get_note_not_found(client: TestClient):
    note_id_val = str(uuid.uuid4())
    mock_chacha_db_instance.get_note_by_id.return_value = None
    response = client.get(f"/api/v1/notes/{note_id_val}")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == "Note not found"


def test_list_notes(client: TestClient):
    note1_id, note2_id = str(uuid.uuid4()), str(uuid.uuid4())
    expected_db_client_id = "test_api_client_for_user_db"
    # Ensure 'content' is provided for NoteResponse schema
    mock_notes_data = [
        create_timestamped_data({"id": note1_id, "title": "Note 1", "content": "Content for Note 1"},
                                expected_db_client_id),
        create_timestamped_data({"id": note2_id, "title": "Note 2", "content": "Content for Note 2"},
                                expected_db_client_id)
    ]
    mock_chacha_db_instance.list_notes.return_value = mock_notes_data
    response = client.get("/api/v1/notes/?limit=10&offset=0")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) == 2
    assert data[0]["id"] == note1_id
    assert data[0]["content"] == "Content for Note 1"
    mock_chacha_db_instance.list_notes.assert_called_once_with(limit=10, offset=0)


def test_update_note(client: TestClient):
    note_id_val = str(uuid.uuid4())
    update_payload = {"title": "Updated Title", "content": "Newer Content"}
    expected_version_header = 1
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.update_note.return_value = True
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        base_data={"id": note_id_val, "title": "Updated Title", "content": "Newer Content",
                   "version": expected_version_header + 1},
        client_id=expected_db_client_id
    )
    response = client.put(
        f"/api/v1/notes/{note_id_val}", json=update_payload, headers={"expected-version": str(expected_version_header)}
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["version"] == expected_version_header + 1
    mock_chacha_db_instance.update_note.assert_called_once_with(
        note_id=note_id_val, update_data=update_payload, expected_version=expected_version_header
    )


def test_update_note_conflict(client: TestClient):
    note_id_val = str(uuid.uuid4())
    update_payload = {"title": "Updated Title"}
    wrong_version_header = 1
    mock_chacha_db_instance.update_note.side_effect = Actual_ConflictError(
        message="Version mismatch", entity="note", entity_id=note_id_val
    )
    response = client.put(
        f"/api/v1/notes/{note_id_val}", json=update_payload, headers={"expected-version": str(wrong_version_header)}
    )
    assert response.status_code == status.HTTP_409_CONFLICT
    assert "The resource has been modified since you last fetched it" in response.json()["detail"]


def test_update_note_no_fields(client: TestClient):
    note_id_val = str(uuid.uuid4())
    response = client.put(f"/api/v1/notes/{note_id_val}", json={}, headers={"expected-version": "1"})
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "No fields provided for update."


def test_delete_note(client: TestClient):
    note_id_val = str(uuid.uuid4())
    expected_version_header = 2
    mock_chacha_db_instance.soft_delete_note.return_value = True
    response = client.delete(f"/api/v1/notes/{note_id_val}", headers={"expected-version": str(expected_version_header)})
    assert response.status_code == status.HTTP_204_NO_CONTENT
    mock_chacha_db_instance.soft_delete_note.assert_called_once_with(
        note_id=note_id_val, expected_version=expected_version_header
    )


def test_delete_note_conflict(client: TestClient):
    note_id_val = str(uuid.uuid4())
    mock_chacha_db_instance.soft_delete_note.side_effect = Actual_ConflictError(
        message="Note version mismatch on delete", entity="note", entity_id=note_id_val
    )
    response = client.delete(f"/api/v1/notes/{note_id_val}", headers={"expected-version": "1"})
    assert response.status_code == status.HTTP_409_CONFLICT
    assert "The resource has been modified since you last fetched it" in response.json()["detail"]


def test_search_notes(client: TestClient):
    query_term, note_id_val = "important", str(uuid.uuid4())
    expected_db_client_id = "test_api_client_for_user_db"
    # Ensure 'content' is provided for NoteResponse schema
    mock_chacha_db_instance.search_notes.return_value = [
        create_timestamped_data(
            {"id": note_id_val, "title": "Important Note", "content": "This content is important."},  # Added content
            expected_db_client_id
        )
    ]
    response = client.get(f"/api/v1/notes/search/?query={query_term}&limit=5")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) == 1
    assert data[0]["title"] == "Important Note"
    assert data[0]["content"] == "This content is important."
    mock_chacha_db_instance.search_notes.assert_called_once_with(search_term=query_term, limit=5)


def test_create_keyword(client: TestClient):
    keyword_payload, keyword_id_val = {"keyword": "ProjectAlpha"}, 123
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.add_keyword.return_value = keyword_id_val
    mock_chacha_db_instance.get_keyword_by_id.return_value = create_timestamped_data(
        {"id": keyword_id_val, "keyword": "ProjectAlpha"}, expected_db_client_id
    )
    response = client.post("/api/v1/notes/keywords/", json=keyword_payload)
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["id"] == keyword_id_val
    mock_chacha_db_instance.add_keyword.assert_called_once_with(keyword_text="ProjectAlpha")


def test_get_keyword(client: TestClient):
    keyword_id_val = 123
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.get_keyword_by_id.return_value = create_timestamped_data(
        {"id": keyword_id_val, "keyword": "ProjectAlpha"}, expected_db_client_id
    )
    response = client.get(f"/api/v1/notes/keywords/{keyword_id_val}")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["id"] == keyword_id_val
    mock_chacha_db_instance.get_keyword_by_id.assert_called_once_with(keyword_id=keyword_id_val)


def test_get_keyword_not_found(client: TestClient):
    keyword_id_val = 999
    mock_chacha_db_instance.get_keyword_by_id.return_value = None
    response = client.get(f"/api/v1/notes/keywords/{keyword_id_val}")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == "Keyword not found"


def test_link_note_to_keyword(client: TestClient):
    note_id_val, keyword_id_val = str(uuid.uuid4()), 123
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {"id": note_id_val, "content": "linking note"}, client_id=expected_db_client_id  # Added content
    )
    mock_chacha_db_instance.get_keyword_by_id.return_value = create_timestamped_data(
        {"id": keyword_id_val, "keyword": "linking keyword"}, client_id=expected_db_client_id
    )
    mock_chacha_db_instance.link_note_to_keyword.return_value = True
    response = client.post(f"/api/v1/notes/{note_id_val}/keywords/{keyword_id_val}")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    mock_chacha_db_instance.link_note_to_keyword.assert_called_once_with(note_id=note_id_val, keyword_id=keyword_id_val)


def test_link_note_to_keyword_note_not_found(client: TestClient):
    note_id_val, keyword_id_val = str(uuid.uuid4()), 123
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.get_note_by_id.return_value = None
    mock_chacha_db_instance.get_keyword_by_id.return_value = create_timestamped_data(
        {"id": keyword_id_val, "keyword": "Some Keyword"}, client_id=expected_db_client_id
    )
    response = client.post(f"/api/v1/notes/{note_id_val}/keywords/{keyword_id_val}")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert f"Note with ID '{note_id_val}' not found" in response.json()["detail"]


def test_get_keywords_for_note(client: TestClient):
    note_id_val, k1_id, k2_id = str(uuid.uuid4()), 1, 2
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {"id": note_id_val, "content": "note with keywords"}, client_id=expected_db_client_id  # Added content
    )
    mock_chacha_db_instance.get_keywords_for_note.return_value = [
        create_timestamped_data({"id": k1_id, "keyword": "Tag1"}, expected_db_client_id),
        create_timestamped_data({"id": k2_id, "keyword": "Tag2"}, expected_db_client_id)
    ]
    response = client.get(f"/api/v1/notes/{note_id_val}/keywords/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["note_id"] == note_id_val
    assert len(data["keywords"]) == 2
    mock_chacha_db_instance.get_keywords_for_note.assert_called_once_with(note_id=note_id_val)

#
# End of test_notes_api_integration.py
########################################################################################################################
