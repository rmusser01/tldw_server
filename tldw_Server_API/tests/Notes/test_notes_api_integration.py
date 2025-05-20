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
#
########################################################################################################################
#
# Functions:

# Import exceptions that the API layer might expect from the DB layer
# These should ideally be the actual exceptions from ChaChaNotes_DB
# For this example, we use the dummy ones if defined in conftest or here for clarity.
# If you have the real exceptions, mock their paths.
MODULE_PATH_PREFIX = "tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB"

MockCharactersRAGDBError = type('CharactersRAGDBError', (Exception,), {})
MockInputError = type('InputError', (MockCharactersRAGDBError,), {})
MockConflictError = type('ConflictError', (MockCharactersRAGDBError,), {'entity': None, 'entity_id': None})

# --- Mocked DB and Dependency Override ---
mock_chacha_db_instance = MagicMock()


async def override_get_chacha_db_for_user():
    # Reset relevant mock states for each call if necessary, or do it in a fixture
    mock_chacha_db_instance.reset_mock()  # Reset call counts etc.
    # Simulate the client_id attribute that the API logs
    mock_chacha_db_instance.client_id = "test_api_client_for_user_db"
    return mock_chacha_db_instance


@pytest.fixture(scope="module")
def test_app():
    app = FastAPI()
    app.include_router(notes_router_module.router, prefix="/api/v1/notes", tags=["Notes"])

    # Override the dependency for all tests
    app.dependency_overrides[notes_router_module.get_chacha_db_for_user] = override_get_chacha_db_for_user
    return app


@pytest.fixture(scope="module")
def client(test_app):
    return TestClient(test_app)


# Helper to create consistent timestamped data
def create_timestamped_data(base_data: Dict[str, Any], client_id: str) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        **base_data,
        "created_at": now.isoformat(),
        "last_modified": now.isoformat(),
        "version": 1,
        "client_id": client_id,  # This should be the api_client_id
        "deleted": False,
    }


# --- Test Cases ---

# == Notes Endpoints ==
def test_create_note(client: TestClient):
    note_id_val = str(uuid.uuid4())
    note_create_payload = {"title": "New Note", "content": "Note content", "id": note_id_val}

    mock_chacha_db_instance.add_note.return_value = note_id_val
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {"id": note_id_val, "title": "New Note", "content": "Note content"},
        mock_chacha_db_instance.client_id
    )

    response = client.post("/api/v1/notes/", json=note_create_payload)

    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["title"] == "New Note"
    assert data["id"] == note_id_val
    mock_chacha_db_instance.add_note.assert_called_once_with(
        title="New Note", content="Note content", note_id=note_id_val
    )
    mock_chacha_db_instance.get_note_by_id.assert_called_once_with(note_id=note_id_val)


def test_create_note_db_error(client: TestClient):
    note_create_payload = {"title": "Error Note", "content": "Content"}
    mock_chacha_db_instance.add_note.side_effect = MockCharactersRAGDBError("DB connection failed")

    response = client.post("/api/v1/notes/", json=note_create_payload)
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "database error occurred" in response.json()["detail"]


def test_create_note_input_error(client: TestClient):
    note_create_payload = {"title": "", "content": "Content"}  # Invalid title
    # Pydantic usually catches this first. If it passes Pydantic but DB raises InputError:
    mock_chacha_db_instance.add_note.side_effect = MockInputError("Title cannot be empty (from DB)")

    response = client.post("/api/v1/notes/", json={"title": "Valid Pydantic", "content": "Content"})
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Title cannot be empty (from DB)" in response.json()["detail"]


def test_create_note_conflict_error(client: TestClient):
    note_id_val = str(uuid.uuid4())
    note_create_payload = {"title": "Conflict Note", "content": "Content", "id": note_id_val}
    mock_chacha_db_instance.add_note.side_effect = MockConflictError(
        "Note already exists", entity="note", entity_id=note_id_val
    )

    response = client.post("/api/v1/notes/", json=note_create_payload)
    assert response.status_code == status.HTTP_409_CONFLICT
    assert f"A conflict occurred with note (ID: {note_id_val})" in response.json()["detail"]


def test_get_note(client: TestClient):
    note_id_val = str(uuid.uuid4())
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {"id": note_id_val, "title": "Fetched Note", "content": "Content"},
        mock_chacha_db_instance.client_id
    )

    response = client.get(f"/api/v1/notes/{note_id_val}")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["id"] == note_id_val
    assert data["title"] == "Fetched Note"
    mock_chacha_db_instance.get_note_by_id.assert_called_once_with(note_id=note_id_val)


def test_get_note_not_found(client: TestClient):
    note_id_val = str(uuid.uuid4())
    mock_chacha_db_instance.get_note_by_id.return_value = None
    response = client.get(f"/api/v1/notes/{note_id_val}")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == "Note not found"


def test_list_notes(client: TestClient):
    note1_id = str(uuid.uuid4())
    note2_id = str(uuid.uuid4())
    mock_notes_data = [
        create_timestamped_data({"id": note1_id, "title": "Note 1", "content": "Content 1"},
                                mock_chacha_db_instance.client_id),
        create_timestamped_data({"id": note2_id, "title": "Note 2", "content": "Content 2"},
                                mock_chacha_db_instance.client_id)
    ]
    mock_chacha_db_instance.list_notes.return_value = mock_notes_data

    response = client.get("/api/v1/notes/?limit=10&offset=0")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) == 2
    assert data[0]["id"] == note1_id
    mock_chacha_db_instance.list_notes.assert_called_once_with(limit=10, offset=0)


def test_update_note(client: TestClient):
    note_id_val = str(uuid.uuid4())
    update_payload = {"title": "Updated Title"}
    expected_version = 1

    mock_chacha_db_instance.update_note.return_value = True
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {"id": note_id_val, "title": "Updated Title", "content": "Original Content", "version": expected_version + 1},
        mock_chacha_db_instance.client_id
    )

    response = client.put(
        f"/api/v1/notes/{note_id_val}",
        json=update_payload,
        headers={"expected-version": str(expected_version)}
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["title"] == "Updated Title"
    assert data["version"] == expected_version + 1
    mock_chacha_db_instance.update_note.assert_called_once_with(
        note_id=note_id_val, update_data=update_payload, expected_version=expected_version
    )


def test_update_note_conflict(client: TestClient):
    note_id_val = str(uuid.uuid4())
    update_payload = {"title": "Updated Title"}
    wrong_version = 1

    mock_chacha_db_instance.update_note.side_effect = MockConflictError("Version mismatch", entity="note",
                                                                        entity_id=note_id_val)

    response = client.put(
        f"/api/v1/notes/{note_id_val}",
        json=update_payload,
        headers={"expected-version": str(wrong_version)}
    )
    assert response.status_code == status.HTTP_409_CONFLICT
    assert "The resource has been modified" in response.json()["detail"].lower() or \
           f"A conflict occurred with note (ID: {note_id_val})" in response.json()["detail"]


def test_update_note_no_fields(client: TestClient):
    note_id_val = str(uuid.uuid4())
    response = client.put(
        f"/api/v1/notes/{note_id_val}",
        json={},  # Empty payload
        headers={"expected-version": "1"}
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "No fields provided for update."


def test_delete_note(client: TestClient):
    note_id_val = str(uuid.uuid4())
    expected_version = 2
    mock_chacha_db_instance.soft_delete_note.return_value = True

    response = client.delete(
        f"/api/v1/notes/{note_id_val}",
        headers={"expected-version": str(expected_version)}
    )
    assert response.status_code == status.HTTP_204_NO_CONTENT
    mock_chacha_db_instance.soft_delete_note.assert_called_once_with(
        note_id=note_id_val, expected_version=expected_version
    )


def test_delete_note_not_found(client: TestClient):
    note_id_val = str(uuid.uuid4())
    # Simulate the DB raising an error that gets translated to 404.
    # ConflictError for "not found" during delete (if version implies existence) is one way.
    # Or if soft_delete_note itself returns False and the API checks it (not current impl).
    # For now, let's assume ConflictError is used if the item doesn't exist for the given version.
    mock_chacha_db_instance.soft_delete_note.side_effect = MockConflictError(
        "Note not found or version mismatch", entity="note", entity_id=note_id_val
    )
    response = client.delete(
        f"/api/v1/notes/{note_id_val}",
        headers={"expected-version": "1"}
    )
    assert response.status_code == status.HTTP_409_CONFLICT  # As per current error handling
    # If the DB method was to raise a specific "Not Found Error", then it could be 404.
    # Current ChaChaDB likely raises ConflictError on version mismatch or if ID not found.


def test_search_notes(client: TestClient):
    query_term = "important"
    note_id_val = str(uuid.uuid4())
    mock_chacha_db_instance.search_notes.return_value = [
        create_timestamped_data(
            {"id": note_id_val, "title": "Important Note", "content": "This is important."},
            mock_chacha_db_instance.client_id
        )
    ]
    response = client.get(f"/api/v1/notes/search/?query={query_term}&limit=5")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) == 1
    assert data[0]["title"] == "Important Note"
    mock_chacha_db_instance.search_notes.assert_called_once_with(search_term=query_term, limit=5)


# == Keywords Endpoints (selected tests, others follow similar pattern) ==

def test_create_keyword(client: TestClient):
    keyword_payload = {"keyword": "ProjectAlpha"}
    keyword_id_val = 123

    mock_chacha_db_instance.add_keyword.return_value = keyword_id_val
    mock_chacha_db_instance.get_keyword_by_id.return_value = create_timestamped_data(
        {"id": keyword_id_val, "keyword": "ProjectAlpha"},
        mock_chacha_db_instance.client_id
    )

    response = client.post("/api/v1/notes/keywords/", json=keyword_payload)
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["keyword"] == "ProjectAlpha"
    assert data["id"] == keyword_id_val
    mock_chacha_db_instance.add_keyword.assert_called_once_with(keyword_text="ProjectAlpha")


def test_get_keyword(client: TestClient):
    keyword_id_val = 123
    mock_chacha_db_instance.get_keyword_by_id.return_value = create_timestamped_data(
        {"id": keyword_id_val, "keyword": "ProjectAlpha"},
        mock_chacha_db_instance.client_id
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


# == Linking Endpoints (selected tests) ==

def test_link_note_to_keyword(client: TestClient):
    note_id_val = str(uuid.uuid4())
    keyword_id_val = 123

    # Simulate note and keyword existing
    mock_chacha_db_instance.get_note_by_id.return_value = {"id": note_id_val, "title": "Some Note"}
    mock_chacha_db_instance.get_keyword_by_id.return_value = {"id": keyword_id_val, "keyword": "Some Keyword"}
    mock_chacha_db_instance.link_note_to_keyword.return_value = True

    response = client.post(f"/api/v1/notes/{note_id_val}/keywords/{keyword_id_val}")
    assert response.status_code == status.HTTP_200_OK  # Default for this endpoint if not specified otherwise
    data = response.json()
    assert data["success"] is True
    assert "linked" in data["message"]
    mock_chacha_db_instance.link_note_to_keyword.assert_called_once_with(note_id=note_id_val, keyword_id=keyword_id_val)


def test_link_note_to_keyword_note_not_found(client: TestClient):
    note_id_val = str(uuid.uuid4())
    keyword_id_val = 123
    mock_chacha_db_instance.get_note_by_id.return_value = None  # Note not found
    mock_chacha_db_instance.get_keyword_by_id.return_value = {"id": keyword_id_val, "keyword": "Some Keyword"}

    response = client.post(f"/api/v1/notes/{note_id_val}/keywords/{keyword_id_val}")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert f"Note with ID '{note_id_val}' not found" in response.json()["detail"]


def test_get_keywords_for_note(client: TestClient):
    note_id_val = str(uuid.uuid4())
    keyword1_id = 1
    keyword2_id = 2
    # Simulate note existing
    mock_chacha_db_instance.get_note_by_id.return_value = {"id": note_id_val, "title": "Some Note"}
    mock_chacha_db_instance.get_keywords_for_note.return_value = [
        create_timestamped_data({"id": keyword1_id, "keyword": "Tag1"}, mock_chacha_db_instance.client_id),
        create_timestamped_data({"id": keyword2_id, "keyword": "Tag2"}, mock_chacha_db_instance.client_id)
    ]

    response = client.get(f"/api/v1/notes/{note_id_val}/keywords/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["note_id"] == note_id_val
    assert len(data["keywords"]) == 2
    assert data["keywords"][0]["id"] == keyword1_id
    mock_chacha_db_instance.get_keywords_for_note.assert_called_once_with(note_id=note_id_val)

# Add more tests for other keyword endpoints (list, delete, search) and
# other linking endpoints (unlink, get_notes_for_keyword) following similar patterns.
# Remember to test edge cases, error conditions (400, 404, 409, 500),
# and validation of query parameters and headers.

#
# End of test_notes_api_integration.py
########################################################################################################################
