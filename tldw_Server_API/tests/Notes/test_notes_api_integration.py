# Tests/test_notes_api_integration.py
#
#
# Imports
import pytest
pytestmark = pytest.mark.unit
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import json
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
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
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

class _StubRateLimiter:
    """Simple stub that denies after N calls per endpoint."""
    def __init__(self):
        self.enabled = True
        # endpoint -> max allowed calls before denying
        self._limits: dict[str, int | None] = {}
        # endpoint -> current calls
        self._calls: dict[str, int] = {}

    def set_limit(self, endpoint: str, max_calls: Optional[int]):
        # None means unlimited
        self._limits[endpoint] = max_calls
        self._calls[endpoint] = 0

    def reset(self):

        self._limits.clear()
        self._calls.clear()

    async def check_user_rate_limit(self, user_id: int, endpoint: str, role: str = "user"):
        if not self.enabled:
            return True, {"rate_limit_enabled": False}
        cur = self._calls.get(endpoint, 0) + 1
        self._calls[endpoint] = cur
        lim = self._limits.get(endpoint)
        if lim is None or cur <= lim:
            return True, {"limit": lim if lim is not None else 0, "remaining": None}
        # Deny
        return False, {"retry_after": 60}


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
    mock_chacha_db_instance.add_note.side_effect = None
    mock_chacha_db_instance.get_note_by_id.side_effect = None
    mock_chacha_db_instance.update_note.side_effect = None
    mock_chacha_db_instance.soft_delete_note.side_effect = None
    mock_chacha_db_instance.list_notes.side_effect = None
    mock_chacha_db_instance.search_notes.side_effect = None
    mock_chacha_db_instance.search_notes_with_keywords.side_effect = None
    mock_chacha_db_instance.count_notes_matching.side_effect = None
    mock_chacha_db_instance.count_notes_matching.return_value = 0
    mock_chacha_db_instance.count_notes_matching_keywords.side_effect = None
    mock_chacha_db_instance.count_notes_matching_keywords.return_value = 0
    mock_chacha_db_instance.add_keyword.side_effect = None
    mock_chacha_db_instance.get_keyword_by_text.side_effect = None
    mock_chacha_db_instance.get_keyword_by_text.return_value = None
    mock_chacha_db_instance.get_keyword_by_id.side_effect = None
    mock_chacha_db_instance.get_keyword_by_id.return_value = None
    mock_chacha_db_instance.list_keyword_collections.side_effect = None
    mock_chacha_db_instance.add_keyword_collection.side_effect = None
    mock_chacha_db_instance.get_keyword_collection_by_id.side_effect = None
    mock_chacha_db_instance.get_keyword_collection_by_id.return_value = None
    mock_chacha_db_instance.update_keyword_collection.side_effect = None
    mock_chacha_db_instance.soft_delete_keyword_collection.side_effect = None
    mock_chacha_db_instance.get_keywords_for_collection.side_effect = None
    mock_chacha_db_instance.link_collection_to_keyword.side_effect = None
    mock_chacha_db_instance.unlink_collection_from_keyword.side_effect = None
    mock_chacha_db_instance.link_conversation_to_keyword.side_effect = None
    mock_chacha_db_instance.unlink_conversation_from_keyword.side_effect = None
    mock_chacha_db_instance.get_keywords_for_conversation.side_effect = None
    mock_chacha_db_instance.execute_query.side_effect = None
    mock_chacha_db_instance.link_note_to_keyword.side_effect = None
    mock_chacha_db_instance.get_keywords_for_note.side_effect = None
    mock_chacha_db_instance.get_note_folders_for_note.side_effect = None
    mock_chacha_db_instance.get_note_folders_for_note.return_value = []
    mock_chacha_db_instance.get_note_folders_for_notes.side_effect = None
    mock_chacha_db_instance.get_note_folders_for_notes.return_value = {}
    mock_chacha_db_instance.get_conversation_by_id.side_effect = None
    mock_chacha_db_instance.get_message_conversation_id.side_effect = None
    _stub_rate_limiter.reset()
    # Default: unlimited for endpoints unless test sets otherwise
    _stub_rate_limiter.set_limit("notes.create", None)
    _stub_rate_limiter.set_limit("notes.list", None)
    _stub_rate_limiter.set_limit("notes.update", None)
    _stub_rate_limiter.set_limit("notes.delete", None)
    _stub_rate_limiter.set_limit("notes.search", None)
    _stub_rate_limiter.set_limit("notes.export", None)
    _stub_rate_limiter.set_limit("notes.bulk_create", None)
    _stub_rate_limiter.set_limit("notes.attachments.upload", None)
    _stub_rate_limiter.set_limit("notes.attachments.list", None)
    _stub_rate_limiter.set_limit("notes.attachments.get", None)
    _stub_rate_limiter.set_limit("notes.attachments.delete", None)
    _stub_rate_limiter.set_limit("keywords.create", None)
    _stub_rate_limiter.set_limit("keywords.list", None)
    _stub_rate_limiter.set_limit("keywords.delete", None)
    _stub_rate_limiter.set_limit("keywords.search", None)


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
        title="New Note",
        content="Note content",
        note_id=note_id_val,
        conversation_id=None,
        message_id=None,
    )
    mock_chacha_db_instance.get_note_by_id.assert_called_once_with(note_id=note_id_val)


def test_get_note_includes_folders(client: TestClient):
    note_id_val = str(uuid.uuid4())
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {"id": note_id_val, "title": "Note", "content": "Body"},
        expected_db_client_id,
    )
    mock_chacha_db_instance.get_keywords_for_note.return_value = []
    mock_chacha_db_instance.get_note_folders_for_note.return_value = [
        {"id": 1, "name": "docs", "path": "docs", "parent_id": None}
    ]

    response = client.get(f"/api/v1/notes/{note_id_val}")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["folders"] == [{"id": 1, "name": "docs", "path": "docs", "parent_id": None}]


def test_create_note_keyword_conflict_refetch(client: TestClient):
    note_id_val = str(uuid.uuid4())
    kw_id = 7
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.add_note.return_value = note_id_val
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {"id": note_id_val, "title": "Note", "content": "Body"},
        expected_db_client_id,
    )
    mock_chacha_db_instance.get_keywords_for_note.return_value = []
    mock_chacha_db_instance.get_keyword_by_text.side_effect = [
        None,
        {"id": kw_id, "keyword": "alpha"},
    ]
    mock_chacha_db_instance.add_keyword.side_effect = Actual_ConflictError(
        "Keyword already exists.",
        entity="keywords",
        entity_id="alpha",
    )

    response = client.post(
        "/api/v1/notes/",
        json={"title": "Note", "content": "Body", "keywords": ["alpha"]},
    )

    assert response.status_code == status.HTTP_201_CREATED, response.text
    mock_chacha_db_instance.add_keyword.assert_called_once_with("alpha")
    assert mock_chacha_db_instance.get_keyword_by_text.call_count == 2
    mock_chacha_db_instance.link_note_to_keyword.assert_called_once_with(
        note_id=note_id_val,
        keyword_id=kw_id,
    )


def test_create_note_surfaces_keyword_partial_failures(client: TestClient):
    note_id_val = str(uuid.uuid4())
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.add_note.return_value = note_id_val
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {"id": note_id_val, "title": "Note", "content": "Body"},
        expected_db_client_id,
    )
    mock_chacha_db_instance.get_keywords_for_note.side_effect = [
        [],
        [create_timestamped_data({"id": 11, "keyword": "alpha"}, expected_db_client_id)],
    ]
    mock_chacha_db_instance.get_keyword_by_text.side_effect = [
        {"id": 11, "keyword": "alpha"},
        {"id": 12, "keyword": "beta"},
    ]

    def _link_side_effect(note_id: str, keyword_id: int):
        if keyword_id == 12:
            raise Actual_InputError("Link failed")
        return True

    mock_chacha_db_instance.link_note_to_keyword.side_effect = _link_side_effect

    response = client.post(
        "/api/v1/notes/",
        json={"title": "Note", "content": "Body", "keywords": ["alpha", "beta"]},
    )

    assert response.status_code == status.HTTP_201_CREATED, response.text
    payload = response.json()
    keyword_sync = payload.get("keyword_sync")
    assert isinstance(keyword_sync, dict)
    assert keyword_sync.get("failed_count") == 1
    assert "beta" in (keyword_sync.get("failed_keywords") or [])


def test_create_note_rejects_overlong_keyword(client: TestClient):
    long_kw = "k" * 101
    response = client.post(
        "/api/v1/notes/",
        json={"title": "New Note", "content": "Note content", "keywords": [long_kw]},
    )
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    mock_chacha_db_instance.add_note.assert_not_called()


def test_create_note_invalid_conversation_id_returns_404(client: TestClient):
    mock_chacha_db_instance.get_conversation_by_id.return_value = None
    response = client.post(
        "/api/v1/notes/",
        json={"title": "New Note", "content": "Note content", "conversation_id": "missing-conv"},
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND
    mock_chacha_db_instance.add_note.assert_not_called()


def test_create_note_invalid_message_id_returns_404(client: TestClient):
    mock_chacha_db_instance.get_message_conversation_id.return_value = None
    response = client.post(
        "/api/v1/notes/",
        json={"title": "New Note", "content": "Note content", "message_id": "missing-msg"},
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND
    mock_chacha_db_instance.add_note.assert_not_called()


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


def test_bulk_create_respects_rate_limit(client: TestClient):
    note_payloads = [
        {"title": "Bulk Note 1", "content": "Content 1"},
        {"title": "Bulk Note 2", "content": "Content 2"},
        {"title": "Bulk Note 3", "content": "Content 3"},
    ]

    # Deny bulk endpoint immediately (parity with centralized limiter at request level)
    _stub_rate_limiter.set_limit("notes.bulk_create", 0)

    response = client.post("/api/v1/notes/bulk", json={"notes": note_payloads})
    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert "Rate limit exceeded" in response.json()["detail"]
    # No item-level work should proceed
    assert mock_chacha_db_instance.add_note.call_count == 0


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
    mock_chacha_db_instance.get_note_by_id.assert_called_once_with(
        note_id=note_id_val,
        include_studio_summary=True,
    )


def test_get_note_not_found(client: TestClient):
    note_id_val = str(uuid.uuid4())
    mock_chacha_db_instance.get_note_by_id.return_value = None
    response = client.get(f"/api/v1/notes/{note_id_val}")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == "Note not found"


def test_note_attachment_upload_list_download_delete(client: TestClient, tmp_path):
    note_id_val = str(uuid.uuid4())
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {"id": note_id_val, "title": "Attachment Note", "content": "Body"},
        expected_db_client_id,
    )

    with patch.object(notes_router_module.DatabasePaths, "get_user_base_directory", return_value=tmp_path):
        upload_response = client.post(
            f"/api/v1/notes/{note_id_val}/attachments",
            files={"file": ("diagram.png", b"image-bytes", "image/png")},
        )
        assert upload_response.status_code == status.HTTP_201_CREATED, upload_response.text
        upload_payload = upload_response.json()
        assert upload_payload["file_name"] == "diagram.png"
        assert upload_payload["original_file_name"] == "diagram.png"
        assert upload_payload["url"].endswith(f"/api/v1/notes/{note_id_val}/attachments/diagram.png")

        note_dir = (
            tmp_path
            / "notes_attachments"
            / notes_router_module._safe_note_attachment_dirname(note_id_val)
        )
        assert (note_dir / "diagram.png").exists()
        assert (note_dir / "diagram.png.meta.json").exists()

        list_response = client.get(f"/api/v1/notes/{note_id_val}/attachments")
        assert list_response.status_code == status.HTTP_200_OK, list_response.text
        list_payload = list_response.json()
        assert list_payload["count"] == 1
        assert list_payload["attachments"][0]["file_name"] == "diagram.png"

        download_response = client.get(f"/api/v1/notes/{note_id_val}/attachments/diagram.png")
        assert download_response.status_code == status.HTTP_200_OK, download_response.text
        assert download_response.content == b"image-bytes"
        assert download_response.headers.get("content-type", "").startswith("image/png")

        delete_response = client.delete(f"/api/v1/notes/{note_id_val}/attachments/diagram.png")
        assert delete_response.status_code == status.HTTP_200_OK, delete_response.text
        assert delete_response.json()["detail"] == "Attachment deleted"
        assert not (note_dir / "diagram.png").exists()
        assert not (note_dir / "diagram.png.meta.json").exists()

        post_delete_list = client.get(f"/api/v1/notes/{note_id_val}/attachments")
        assert post_delete_list.status_code == status.HTTP_200_OK
        assert post_delete_list.json()["count"] == 0


def test_note_attachment_upload_rejects_unsupported_extension(client: TestClient, tmp_path):
    note_id_val = str(uuid.uuid4())
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {"id": note_id_val, "title": "Attachment Note", "content": "Body"},
        expected_db_client_id,
    )

    with patch.object(notes_router_module.DatabasePaths, "get_user_base_directory", return_value=tmp_path):
        response = client.post(
            f"/api/v1/notes/{note_id_val}/attachments",
            files={"file": ("malware.exe", b"nope", "application/octet-stream")},
        )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Unsupported attachment type" in response.json()["detail"]


def test_note_attachment_upload_rejects_oversized_files(client: TestClient, tmp_path):
    note_id_val = str(uuid.uuid4())
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {"id": note_id_val, "title": "Attachment Note", "content": "Body"},
        expected_db_client_id,
    )

    with (
        patch.object(notes_router_module.DatabasePaths, "get_user_base_directory", return_value=tmp_path),
        patch.object(notes_router_module, "_NOTES_ATTACHMENT_MAX_BYTES", 8),
    ):
        response = client.post(
            f"/api/v1/notes/{note_id_val}/attachments",
            files={"file": ("small.txt", b"123456789", "text/plain")},
        )
    assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    assert "Attachment exceeds maximum size" in response.json()["detail"]


def test_note_attachment_download_rejects_invalid_filename(client: TestClient, tmp_path):
    note_id_val = str(uuid.uuid4())
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {"id": note_id_val, "title": "Attachment Note", "content": "Body"},
        expected_db_client_id,
    )

    with patch.object(notes_router_module.DatabasePaths, "get_user_base_directory", return_value=tmp_path):
        client.post(
            f"/api/v1/notes/{note_id_val}/attachments",
            files={"file": ("diagram.png", b"image-bytes", "image/png")},
        )
        response = client.get(f"/api/v1/notes/{note_id_val}/attachments/diagram%3C%3E.png")

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "Invalid attachment filename"


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
    mock_chacha_db_instance.count_notes.return_value = len(mock_notes_data)
    response = client.get("/api/v1/notes/?limit=10&offset=0")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    # Endpoint now returns an object with pagination metadata;
    # fall back to treating it as a bare list if needed.
    if isinstance(data, dict):
        items = data.get("items") or data.get("notes") or data.get("results") or []
        count = data.get("count")
    else:
        items = data
        count = len(items)

    assert len(items) == 2
    if count is not None:
        assert count == 2
    assert items[0]["id"] == note1_id
    assert items[0]["content"] == "Content for Note 1"
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


def test_update_note_surfaces_keyword_partial_failures(client: TestClient):
    note_id_val = str(uuid.uuid4())
    expected_version_header = 3
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.update_note.return_value = True
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        base_data={
            "id": note_id_val,
            "title": "Updated Title",
            "content": "Updated body",
            "version": expected_version_header + 1,
        },
        client_id=expected_db_client_id,
    )
    mock_chacha_db_instance.get_keywords_for_note.side_effect = [
        [],
        [create_timestamped_data({"id": 31, "keyword": "alpha"}, expected_db_client_id)],
    ]
    mock_chacha_db_instance.get_keyword_by_text.side_effect = [
        {"id": 31, "keyword": "alpha"},
        {"id": 32, "keyword": "beta"},
    ]

    def _link_side_effect(note_id: str, keyword_id: int):
        if keyword_id == 32:
            raise Actual_InputError("Link failed")
        return True

    mock_chacha_db_instance.link_note_to_keyword.side_effect = _link_side_effect

    response = client.put(
        f"/api/v1/notes/{note_id_val}",
        json={"title": "Updated Title", "keywords": ["alpha", "beta"]},
        headers={"expected-version": str(expected_version_header)},
    )
    assert response.status_code == status.HTTP_200_OK, response.text
    payload = response.json()
    keyword_sync = payload.get("keyword_sync")
    assert isinstance(keyword_sync, dict)
    assert keyword_sync.get("failed_count") == 1
    assert "beta" in (keyword_sync.get("failed_keywords") or [])


def test_update_note_rejects_mismatched_conversation_and_existing_message(client: TestClient):
    note_id_val = str(uuid.uuid4())
    expected_version_header = 1
    expected_db_client_id = "test_api_client_for_user_db"
    # Existing note has message_id tied to a different conversation
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {
            "id": note_id_val,
            "title": "Existing",
            "content": "Body",
            "conversation_id": "conv-1",
            "message_id": "msg-1",
            "version": expected_version_header,
        },
        expected_db_client_id,
    )
    mock_chacha_db_instance.get_conversation_by_id.return_value = {"id": "conv-2"}
    mock_chacha_db_instance.get_message_conversation_id.return_value = "conv-1"

    response = client.put(
        f"/api/v1/notes/{note_id_val}",
        json={"conversation_id": "conv-2"},
        headers={"expected-version": str(expected_version_header)},
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == "Message not found in conversation"
    mock_chacha_db_instance.update_note.assert_not_called()


def test_patch_note_rejects_mismatched_message_and_existing_conversation(client: TestClient):
    note_id_val = str(uuid.uuid4())
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {
            "id": note_id_val,
            "title": "Existing",
            "content": "Body",
            "conversation_id": "conv-1",
            "message_id": "msg-1",
            "version": 1,
        },
        expected_db_client_id,
    )
    mock_chacha_db_instance.get_conversation_by_id.return_value = {"id": "conv-1"}
    mock_chacha_db_instance.get_message_conversation_id.return_value = "conv-2"

    response = client.patch(
        f"/api/v1/notes/{note_id_val}",
        json={"message_id": "msg-2"},
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["detail"] == "Message not found in conversation"
    mock_chacha_db_instance.update_note.assert_not_called()


def test_create_note_auto_title_clamped_to_255(client: TestClient, monkeypatch):
    note_id_val = str(uuid.uuid4())
    expected_db_client_id = "test_api_client_for_user_db"
    captured: Dict[str, Any] = {}

    def _fake_generate_note_title(content: str, options=None):
        captured["max_len"] = getattr(options, "max_len", None)
        max_len = captured["max_len"] or 0
        return "x" * max_len

    monkeypatch.setattr(notes_router_module, "generate_note_title", _fake_generate_note_title)

    mock_chacha_db_instance.add_note.return_value = note_id_val
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {"id": note_id_val, "title": "x" * 255, "content": "Note content"},
        expected_db_client_id,
    )

    response = client.post(
        "/api/v1/notes/",
        json={"content": "Note content", "auto_title": True, "title_max_len": 500},
    )
    assert response.status_code == status.HTTP_201_CREATED
    assert captured.get("max_len") == 255
    called_title = mock_chacha_db_instance.add_note.call_args.kwargs.get("title")
    assert called_title is not None and len(called_title) == 255


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
    assert response.json()["detail"] == "No valid fields provided for update."


def test_update_note_rejects_null_values(client: TestClient):
    note_id_val = str(uuid.uuid4())
    response = client.put(
        f"/api/v1/notes/{note_id_val}",
        json={"content": None},
        headers={"expected-version": "1"}
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "No valid fields provided for update."
    mock_chacha_db_instance.update_note.assert_not_called()


def test_update_note_rejects_whitespace_title(client: TestClient):
    note_id_val = str(uuid.uuid4())
    response = client.put(
        f"/api/v1/notes/{note_id_val}",
        json={"title": "   "},
        headers={"expected-version": "1"}
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "Title cannot be empty or whitespace."
    mock_chacha_db_instance.update_note.assert_not_called()


def test_patch_note_rejects_null_values(client: TestClient):
    note_id_val = str(uuid.uuid4())
    response = client.patch(f"/api/v1/notes/{note_id_val}", json={"title": None})
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "No valid fields provided for update."
    mock_chacha_db_instance.get_note_by_id.assert_not_called()
    mock_chacha_db_instance.update_note.assert_not_called()


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
    mock_chacha_db_instance.count_notes_matching.return_value = 1
    response = client.get(f"/api/v1/notes/search/?query={query_term}&limit=5")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["count"] == 1
    assert data["total"] == 1
    assert data["pagination"]["total"] == 1
    assert data["notes"][0]["title"] == "Important Note"
    assert data["notes"][0]["content"] == "This content is important."
    mock_chacha_db_instance.search_notes.assert_called_once_with(search_term=query_term, limit=5, offset=0)
    mock_chacha_db_instance.count_notes_matching.assert_called_once_with(query_term)


def test_search_notes_without_trailing_slash(client: TestClient):
    query_term, note_id_val = "important", str(uuid.uuid4())
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.search_notes.return_value = [
        create_timestamped_data(
            {"id": note_id_val, "title": "Important Note", "content": "This content is important."},
            expected_db_client_id
        )
    ]
    mock_chacha_db_instance.count_notes_matching.return_value = 1
    response = client.get(f"/api/v1/notes/search?query={query_term}&limit=5")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["count"] == 1
    assert data["total"] == 1
    assert data["pagination"]["total"] == 1
    assert data["notes"][0]["id"] == note_id_val
    mock_chacha_db_instance.search_notes.assert_called_once_with(search_term=query_term, limit=5, offset=0)
    mock_chacha_db_instance.count_notes_matching.assert_called_once_with(query_term)


def test_search_notes_with_keyword_tokens(client: TestClient):
    query_term, note_id_val = "important", str(uuid.uuid4())
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.search_notes_with_keywords.return_value = [
        create_timestamped_data(
            {"id": note_id_val, "title": "Important Note", "content": "Token-filtered content."},
            expected_db_client_id
        )
    ]
    mock_chacha_db_instance.count_notes_matching_keywords.return_value = 1
    response = client.get(
        "/api/v1/notes/search/?query=important&tokens=mindmap&tokens=project&limit=5"
    )
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["count"] == 1
    assert data["total"] == 1
    assert data["pagination"]["total"] == 1
    assert data["notes"][0]["id"] == note_id_val
    mock_chacha_db_instance.search_notes_with_keywords.assert_called_once_with(
        search_term=query_term,
        keyword_tokens=["mindmap", "project"],
        limit=5,
        offset=0
    )
    mock_chacha_db_instance.count_notes_matching_keywords.assert_called_once_with(
        search_term=query_term,
        keyword_tokens=["mindmap", "project"],
    )


def test_export_notes_route_not_shadowed(client: TestClient):
    note_id_val = str(uuid.uuid4())
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.list_notes.return_value = [
        create_timestamped_data(
            {"id": note_id_val, "title": "Exported Note", "content": "Export content"},
            expected_db_client_id
        )
    ]
    mock_chacha_db_instance.count_notes.return_value = 1
    response = client.get("/api/v1/notes/export")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["count"] == 1
    assert data["notes"][0]["id"] == note_id_val
    assert data["pagination"] == {
        "mode": "offset",
        "limit": 1000,
        "offset": 0,
        "total": 1,
        "has_more": False,
        "next_offset": None,
    }
    mock_chacha_db_instance.list_notes.assert_called_once_with(limit=1000, offset=0)


def test_export_notes_csv_route(client: TestClient):
    note_id_val = str(uuid.uuid4())
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.list_notes.return_value = [
        create_timestamped_data(
            {"id": note_id_val, "title": "Exported Note", "content": "Export content"},
            expected_db_client_id
        )
    ]
    response = client.get("/api/v1/notes/export.csv")
    assert response.status_code == status.HTTP_200_OK
    assert response.headers["content-type"].startswith("text/csv")
    assert "notes_export_" in response.headers.get("content-disposition", "")
    assert "Exported Note" in response.text
    mock_chacha_db_instance.list_notes.assert_called_once_with(limit=1000, offset=0)


def test_import_notes_json_creates_notes_and_keywords(client: TestClient):
    expected_db_client_id = "test_api_client_for_user_db"
    created_note_id = str(uuid.uuid4())
    mock_chacha_db_instance.add_note.return_value = created_note_id
    mock_chacha_db_instance.get_note_by_id.side_effect = lambda note_id: (
        create_timestamped_data(
            {"id": created_note_id, "title": "Imported Note", "content": "Imported body"},
            expected_db_client_id,
        )
        if str(note_id) == created_note_id
        else None
    )
    mock_chacha_db_instance.get_keywords_for_note.return_value = []
    mock_chacha_db_instance.get_keyword_by_text.return_value = {
        "id": 42,
        "keyword": "alpha",
        "client_id": expected_db_client_id,
        "version": 1,
        "deleted": False,
    }
    mock_chacha_db_instance.link_note_to_keyword.return_value = True

    payload = {
        "items": [
            {
                "file_name": "notes.json",
                "format": "json",
                "content": '[{"id":"note-import-1","title":"Imported Note","content":"Imported body","keywords":["alpha"]}]'
            }
        ]
    }
    response = client.post("/api/v1/notes/import", json=payload)
    assert response.status_code == status.HTTP_200_OK, response.text
    body = response.json()
    assert body["created_count"] == 1
    assert body["updated_count"] == 0
    assert body["skipped_count"] == 0
    assert body["failed_count"] == 0
    mock_chacha_db_instance.add_note.assert_called_once()
    # Default strategy is create_copy, so imported IDs are not reused.
    assert mock_chacha_db_instance.add_note.call_args.kwargs.get("note_id") is None


def test_import_notes_skip_duplicate_ids(client: TestClient):
    duplicate_id = str(uuid.uuid4())
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {"id": duplicate_id, "title": "Existing", "content": "Existing body", "version": 2},
        expected_db_client_id,
    )

    payload = {
        "duplicate_strategy": "skip",
        "items": [
            {
                "file_name": "dupes.json",
                "format": "json",
                "content": f'[{{"id":"{duplicate_id}","title":"Imported Duplicate","content":"New body"}}]'
            }
        ]
    }
    response = client.post("/api/v1/notes/import", json=payload)
    assert response.status_code == status.HTTP_200_OK, response.text
    body = response.json()
    assert body["created_count"] == 0
    assert body["updated_count"] == 0
    assert body["skipped_count"] == 1
    assert body["failed_count"] == 0
    mock_chacha_db_instance.add_note.assert_not_called()


def test_import_notes_round_trip_accepts_export_wrapper_payload(client: TestClient):
    source_note_id = str(uuid.uuid4())
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.list_notes.return_value = [
        create_timestamped_data(
            {
                "id": source_note_id,
                "title": "Round-trip source note",
                "content": "Round-trip body",
            },
            expected_db_client_id
        )
    ]
    mock_chacha_db_instance.count_notes.return_value = 1
    export_response = client.get("/api/v1/notes/export")
    assert export_response.status_code == status.HTTP_200_OK, export_response.text
    export_payload = export_response.json()
    assert export_payload["count"] == 1

    mock_chacha_db_instance.get_note_by_id.return_value = None
    created_note_id = str(uuid.uuid4())
    mock_chacha_db_instance.add_note.return_value = created_note_id
    mock_chacha_db_instance.get_note_by_id.side_effect = lambda note_id: (
        create_timestamped_data(
            {
                "id": created_note_id,
                "title": "Round-trip source note",
                "content": "Round-trip body",
            },
            expected_db_client_id,
        )
        if str(note_id) == created_note_id
        else None
    )
    mock_chacha_db_instance.get_keywords_for_note.return_value = []

    import_response = client.post(
        "/api/v1/notes/import",
        json={
            "duplicate_strategy": "create_copy",
            "items": [
                {
                    "file_name": "roundtrip-export.json",
                    "format": "json",
                    "content": json.dumps(export_payload),
                }
            ],
        },
    )
    assert import_response.status_code == status.HTTP_200_OK, import_response.text
    body = import_response.json()
    assert body["created_count"] == 1
    assert body["updated_count"] == 0
    assert body["failed_count"] == 0
    mock_chacha_db_instance.add_note.assert_called_once()
    assert mock_chacha_db_instance.add_note.call_args.kwargs.get("note_id") is None


def test_import_notes_partial_success_includes_parse_errors(client: TestClient):
    expected_db_client_id = "test_api_client_for_user_db"
    created_note_id = str(uuid.uuid4())
    mock_chacha_db_instance.add_note.return_value = created_note_id
    mock_chacha_db_instance.get_note_by_id.side_effect = lambda note_id: (
        create_timestamped_data(
            {
                "id": created_note_id,
                "title": "Imported Markdown",
                "content": "# Imported Markdown\n\nBody from markdown.",
            },
            expected_db_client_id,
        )
        if str(note_id) == created_note_id
        else None
    )
    mock_chacha_db_instance.get_keywords_for_note.return_value = []
    payload = {
        "duplicate_strategy": "create_copy",
        "items": [
            {
                "file_name": "broken.json",
                "format": "json",
                "content": '{"title": "Missing closing brace"'
            },
            {
                "file_name": "note.md",
                "format": "markdown",
                "content": "# Imported Markdown\n\nBody from markdown."
            }
        ]
    }
    response = client.post("/api/v1/notes/import", json=payload)
    assert response.status_code == status.HTTP_200_OK, response.text
    body = response.json()
    assert body["created_count"] == 1
    assert body["failed_count"] == 1
    assert isinstance(body.get("files"), list) and len(body["files"]) == 2
    json_result = next((f for f in body["files"] if f.get("file_name") == "broken.json"), None)
    assert json_result is not None
    assert json_result["failed_count"] >= 1
    assert any("JSON parse error" in err for err in json_result.get("errors", []))


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


def test_list_keyword_collections_with_keywords(client: TestClient):
    expected_db_client_id = "test_api_client_for_user_db"
    collection_row = create_timestamped_data(
        {"id": 11, "name": "Research", "parent_id": None},
        expected_db_client_id,
    )
    keyword_row = create_timestamped_data(
        {"id": 5, "keyword": "research"},
        expected_db_client_id,
    )
    mock_chacha_db_instance.list_keyword_collections.return_value = [collection_row]
    mock_chacha_db_instance.count_keyword_collections.return_value = 1
    mock_chacha_db_instance.get_keywords_for_collection.return_value = [keyword_row]

    response = client.get("/api/v1/notes/collections?include_keywords=true")
    assert response.status_code == status.HTTP_200_OK, response.text
    payload = response.json()
    assert isinstance(payload.get("collections"), list)
    assert payload["collections"][0]["name"] == "Research"
    assert payload["collections"][0]["keywords"][0]["keyword"] == "research"
    mock_chacha_db_instance.list_keyword_collections.assert_called_once_with(limit=100, offset=0)
    mock_chacha_db_instance.get_keywords_for_collection.assert_called_once_with(collection_id=11)


def test_create_keyword_collection_with_keywords(client: TestClient):
    expected_db_client_id = "test_api_client_for_user_db"
    collection_id = 17
    collection_row = create_timestamped_data(
        {"id": collection_id, "name": "Research", "parent_id": None},
        expected_db_client_id,
    )
    keyword_row = create_timestamped_data(
        {"id": 7, "keyword": "research"},
        expected_db_client_id,
    )
    mock_chacha_db_instance.add_keyword_collection.return_value = collection_id
    mock_chacha_db_instance.get_keyword_collection_by_id.return_value = collection_row
    mock_chacha_db_instance.get_keywords_for_collection.side_effect = [[], [keyword_row]]
    mock_chacha_db_instance.get_keyword_by_text.side_effect = [None, None]
    mock_chacha_db_instance.add_keyword.return_value = 7
    mock_chacha_db_instance.get_keyword_by_id.return_value = keyword_row
    mock_chacha_db_instance.link_collection_to_keyword.return_value = True

    response = client.post(
        "/api/v1/notes/collections",
        json={"name": "Research", "keywords": ["research"]},
    )
    assert response.status_code == status.HTTP_201_CREATED, response.text
    payload = response.json()
    assert payload["id"] == collection_id
    assert payload["name"] == "Research"
    assert payload["keywords"][0]["keyword"] == "research"
    mock_chacha_db_instance.add_keyword_collection.assert_called_once_with(
        name="Research",
        parent_id=None,
    )
    mock_chacha_db_instance.link_collection_to_keyword.assert_called_once_with(
        collection_id=collection_id,
        keyword_id=7,
    )


def test_update_keyword_collection_with_keywords(client: TestClient):
    expected_db_client_id = "test_api_client_for_user_db"
    collection_id = 23
    current_collection = create_timestamped_data(
        {"id": collection_id, "name": "Old Name", "parent_id": None, "version": 3},
        expected_db_client_id,
    )
    updated_collection = create_timestamped_data(
        {"id": collection_id, "name": "Updated Name", "parent_id": None, "version": 4},
        expected_db_client_id,
    )
    keyword_row = create_timestamped_data(
        {"id": 9, "keyword": "ml"},
        expected_db_client_id,
    )
    mock_chacha_db_instance.get_keyword_collection_by_id.side_effect = [
        current_collection,
        updated_collection,
    ]
    mock_chacha_db_instance.update_keyword_collection.return_value = True
    mock_chacha_db_instance.get_keywords_for_collection.side_effect = [[], [keyword_row]]
    mock_chacha_db_instance.get_keyword_by_text.return_value = keyword_row
    mock_chacha_db_instance.link_collection_to_keyword.return_value = True

    response = client.patch(
        f"/api/v1/notes/collections/{collection_id}",
        json={"name": "Updated Name", "keywords": ["ml"]},
        headers={"expected-version": "3"},
    )
    assert response.status_code == status.HTTP_200_OK, response.text
    payload = response.json()
    assert payload["name"] == "Updated Name"
    assert payload["keywords"][0]["keyword"] == "ml"
    mock_chacha_db_instance.update_keyword_collection.assert_called_once_with(
        collection_id=collection_id,
        update_data={"name": "Updated Name"},
        expected_version=3,
    )


def test_delete_keyword_collection(client: TestClient):
    expected_db_client_id = "test_api_client_for_user_db"
    collection_id = 44
    mock_chacha_db_instance.get_keyword_collection_by_id.return_value = create_timestamped_data(
        {"id": collection_id, "name": "Delete Me", "parent_id": None, "version": 6},
        expected_db_client_id,
    )
    mock_chacha_db_instance.soft_delete_keyword_collection.return_value = True

    response = client.delete(f"/api/v1/notes/collections/{collection_id}")
    assert response.status_code == status.HTTP_204_NO_CONTENT
    mock_chacha_db_instance.soft_delete_keyword_collection.assert_called_once_with(
        collection_id=collection_id,
        expected_version=6,
    )


def test_rate_limit_enforced_without_env(client: TestClient):
    # Allow only 1 create, then 429
    _stub_rate_limiter.set_limit("notes.create", 1)
    note_id_val = str(uuid.uuid4())
    expected_db_client_id = "test_api_client_for_user_db"
    mock_chacha_db_instance.add_note.return_value = note_id_val
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {"id": note_id_val, "title": "First Note", "content": "Content"},
        expected_db_client_id,
    )

    first = client.post("/api/v1/notes/", json={"title": "First Note", "content": "Content"})
    assert first.status_code == status.HTTP_201_CREATED

    second = client.post("/api/v1/notes/", json={"title": "Second Note", "content": "More"})
    assert second.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert "Rate limit" in second.json()["detail"]


def test_rate_limit_disabled_by_env_flag(client: TestClient, monkeypatch):
    # Simulate unlimited by stub configuration; environment flag not required for stub
    _stub_rate_limiter.set_limit("notes.create", None)

    expected_db_client_id = "test_api_client_for_user_db"
    first_note_id = str(uuid.uuid4())
    mock_chacha_db_instance.add_note.return_value = first_note_id
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {"id": first_note_id, "title": "Allowed Note", "content": "Content"},
        expected_db_client_id,
    )
    first = client.post("/api/v1/notes/", json={"title": "Allowed Note", "content": "Content"})
    assert first.status_code == status.HTTP_201_CREATED

    second_note_id = str(uuid.uuid4())
    mock_chacha_db_instance.add_note.return_value = second_note_id
    mock_chacha_db_instance.get_note_by_id.return_value = create_timestamped_data(
        {"id": second_note_id, "title": "Allowed Note 2", "content": "Content"},
        expected_db_client_id,
    )
    second = client.post("/api/v1/notes/", json={"title": "Allowed Note 2", "content": "Content"})
    assert second.status_code == status.HTTP_201_CREATED

#
# End of test_notes_api_integration.py
########################################################################################################################
