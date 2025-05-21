# tests/integration/api/v1/test_prompts_api.py
# Description:
#
# Imports
import os
from pathlib import Path
import pytest
import json
import base64
import urllib.parse
#
# Third-Party Imports
from fastapi.testclient import TestClient
#
# Local Imports
from tldw_Server_API.app.main import app as fastapi_app
from tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps import close_all_cached_prompts_db_instances
from tldw_Server_API.app.api.v1.endpoints.prompts import verify_token
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.config import settings
#
# Local Imports
#
#######################################################################################################################
#


# Functions:
# tests/integration/api/v1/test_prompts_api.py
# Description: Extensive integration tests for the Prompts API.
#
# Imports
import pytest
import json
import base64
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import patch, MagicMock

# Third-Party Imports
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

# Local Imports
from tldw_Server_API.app.main import app as fastapi_app
from tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps import (
    close_all_cached_prompts_db_instances,
    _get_prompts_db_path_for_user,  # For monkeypatching target
    get_prompts_db_for_user  # For potential mock DB injection
)
from tldw_Server_API.app.api.v1.endpoints.prompts import verify_token  # For dependency override
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.config import settings  # Canonical settings object
from tldw_Server_API.app.core.DB_Management.Prompts_DB import (
    PromptsDatabase,
    DatabaseError,
    InputError,
    ConflictError
)
from tldw_Server_API.app.api.v1.schemas import prompt_schemas as schemas

# For mocking file operations in export
from tldw_Server_API.app.core.Prompt_Management import Prompts_Interop

#######################################################################################################################
# Constants
#######################################################################################################################

API_V1_PROMPTS_PREFIX = "/api/v1/prompts"
TEST_USER_ID = 1  # Or derive from settings if needed
TEST_USERNAME = "testuser"
FIXED_TEST_API_TOKEN = "fixed_test_api_token_for_pytest_xyz123"  # Ensure this is complex enough


# Sample Payloads
def get_sample_prompt_payload(name_suffix: str = "") -> Dict[str, Any]:
    return {
        "name": f"API Test Prompt {name_suffix}".strip(),
        "author": "API Tester",
        "details": f"Details via API for {name_suffix}.".strip(),
        "system_prompt": "System instructions for API.",
        "user_prompt": "User query for API.",
        "keywords": ["api_test", "integration", name_suffix.lower().replace(" ", "_")] if name_suffix else ["api_test",
                                                                                                            "integration"]
    }


def get_sample_keyword_payload(text_suffix: str = "") -> Dict[str, str]:
    return {"keyword_text": f"api_keyword_{text_suffix}".strip()}


#######################################################################################################################
# Fixtures
#######################################################################################################################

@pytest.fixture(scope="session")
def test_user_instance() -> User:
    """Provides a consistent test user instance."""
    # In a real multi-user scenario, SINGLE_USER_FIXED_ID might not be relevant.
    # For testing, a fixed ID is fine.
    user_id = getattr(settings, "SINGLE_USER_FIXED_ID", TEST_USER_ID) if getattr(settings, "SINGLE_USER_MODE",
                                                                                 False) else TEST_USER_ID
    return User(id=user_id, username=TEST_USERNAME)


@pytest.fixture(scope="session")
def actual_api_token_value() -> str:
    """Returns the API token value that verify_token would expect."""
    # This should align with what settings.API_BEARER would be in a real scenario.
    # For testing, we can use a fixed value and monkeypatch settings.API_BEARER.
    return FIXED_TEST_API_TOKEN


@pytest.fixture(scope="function")
def client_env_setup(tmp_path: Path, monkeypatch, test_user_instance: User):
    """Sets up common environment mocks for client fixtures (DB path, user)."""

    # Mock _get_prompts_db_path_for_user to use tmp_path
    def mock_get_db_path(user: User, db_version: str = "v2") -> Path:
        user_db_dir = tmp_path / str(user.id) / "prompts_user_dbs"
        user_db_dir.mkdir(parents=True, exist_ok=True)
        return user_db_dir / f"user_prompts_{db_version}.sqlite"

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps._get_prompts_db_path_for_user",
        mock_get_db_path
    )

    # Override get_request_user dependency
    def override_get_request_user_dependency():
        return test_user_instance

    fastapi_app.dependency_overrides[get_request_user] = override_get_request_user_dependency

    yield  # Allows dependent fixtures to run

    # Teardown: remove overrides
    del fastapi_app.dependency_overrides[get_request_user]
    if callable(close_all_cached_prompts_db_instances):
        close_all_cached_prompts_db_instances()


@pytest.fixture(scope="function")
def client(client_env_setup):
    """
    Provides a TestClient with auth bypassed (verify_token always returns True).
    Uses a fresh, temporary database for each test function.
    """

    async def override_verify_token_dependency_always_true():
        return True

    fastapi_app.dependency_overrides[verify_token] = override_verify_token_dependency_always_true

    with TestClient(fastapi_app) as c:
        yield c

    # Teardown: remove verify_token override
    del fastapi_app.dependency_overrides[verify_token]


@pytest.fixture(scope="function")
def client_with_auth(client_env_setup, monkeypatch, actual_api_token_value: str):
    """
    Provides a TestClient where actual auth logic (verify_token) is tested.
    `settings.API_BEARER` is monkeypatched to `actual_api_token_value`.
    """
    original_api_bearer = getattr(settings, "API_BEARER", None)
    monkeypatch.setattr(settings, "API_BEARER", actual_api_token_value)

    # No override for verify_token, so the actual dependency will be called
    with TestClient(fastapi_app) as c:
        yield c

    # Teardown: restore original API_BEARER
    if original_api_bearer is not None:
        monkeypatch.setattr(settings, "API_BEARER", original_api_bearer)
    # else: if it was not set, it might be tricky to "unset" it without knowing Pydantic's behavior
    # For tests, it's usually fine if it remains set to the test value for subsequent non-auth tests
    # if they don't rely on its absence. Better to always have a default in settings.


@pytest.fixture
def auth_headers(actual_api_token_value: str) -> Dict[str, str]:
    """Provides authentication headers for tests where auth is NOT bypassed."""
    # The `verify_token` dependency is `Token: str = Header(None)`
    # It internally handles "Bearer " prefix if present.
    return {"Token": f"Bearer {actual_api_token_value}"}


@pytest.fixture
def no_auth_headers() -> Dict[str, str]:
    """Empty headers for testing unauthenticated requests."""
    return {}


@pytest.fixture
def invalid_auth_headers() -> Dict[str, str]:
    """Invalid authentication headers."""
    return {"Token": "Bearer invalidone"}


# Helper to create a prompt and return its ID and UUID
def create_prompt_utility(client: TestClient, payload_suffix: str = "", headers: Optional[Dict[str, str]] = None) -> \
Dict[str, Any]:
    if headers is None:  # Default to bypassed auth if client is the standard one
        headers = {
            "Token": "irrelevant_for_bypassed_auth"} if "client_with_auth" not in client.app.dependency_overrides else {}

    payload = get_sample_prompt_payload(payload_suffix)
    response = client.post(f"{API_V1_PROMPTS_PREFIX}/", json=payload, headers=headers)
    response.raise_for_status()  # Will raise for non-2xx
    assert response.status_code == status.HTTP_201_CREATED
    return response.json()


#######################################################################################################################
# Unit Tests (for helpers like verify_token)
#######################################################################################################################

@pytest.mark.asyncio
async def test_verify_token_success(monkeypatch, actual_api_token_value: str):
    monkeypatch.setattr(settings, "API_BEARER", actual_api_token_value)
    assert await verify_token(Token=f"Bearer {actual_api_token_value}") is True
    assert await verify_token(Token=actual_api_token_value) is True  # Without Bearer prefix


@pytest.mark.asyncio
async def test_verify_token_missing_token(monkeypatch, actual_api_token_value: str):
    monkeypatch.setattr(settings, "API_BEARER", actual_api_token_value)
    with pytest.raises(HTTPException) as exc_info:
        await verify_token(Token=None)
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Missing authentication token" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_token_invalid_token(monkeypatch, actual_api_token_value: str):
    monkeypatch.setattr(settings, "API_BEARER", actual_api_token_value)
    with pytest.raises(HTTPException) as exc_info:
        await verify_token(Token="Bearer invalid")
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Invalid authentication token" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_token_server_misconfigured(monkeypatch):
    # Simulate API_BEARER not being set
    original_bearer = getattr(settings, "API_BEARER", "exists")
    monkeypatch.setattr(settings, "API_BEARER", None)  # or delattr if appropriate and safe

    with pytest.raises(HTTPException) as exc_info:
        await verify_token(Token="anytoken")
    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Server authentication misconfigured" in exc_info.value.detail

    if original_bearer != "exists":  # Restore if it was there
        monkeypatch.setattr(settings, "API_BEARER", original_bearer)
    # If it was 'None' to begin with, this won't change it back if setattr can't set None back to "not set"
    # This part of teardown can be tricky with Pydantic. Best if settings always have a default.


#######################################################################################################################
# Authentication / Authorization Integration Tests
#######################################################################################################################

def test_unauthorized_access_no_token_provided(client_with_auth: TestClient, no_auth_headers: Dict[str, str]):
    response = client_with_auth.get(f"{API_V1_PROMPTS_PREFIX}/", headers=no_auth_headers)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED  # FastAPI default for missing Header
    assert "Missing authentication token" in response.json()["detail"]


def test_unauthorized_access_invalid_token_provided(client_with_auth: TestClient, invalid_auth_headers: Dict[str, str]):
    response = client_with_auth.get(f"{API_V1_PROMPTS_PREFIX}/", headers=invalid_auth_headers)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Invalid authentication token" in response.json()["detail"]


def test_authorized_access_valid_token_provided(client_with_auth: TestClient, auth_headers: Dict[str, str]):
    response = client_with_auth.get(f"{API_V1_PROMPTS_PREFIX}/", headers=auth_headers)
    # Expect 200 OK if endpoint is valid and auth passes, not 401/403
    assert response.status_code == status.HTTP_200_OK


#######################################################################################################################
# Prompt CRUD Integration Tests
#######################################################################################################################

class TestPromptEndpoints:

    def test_create_prompt_success(self, client: TestClient):
        payload = get_sample_prompt_payload("CreateSuccess")
        response = client.post(f"{API_V1_PROMPTS_PREFIX}/", json=payload)  # Auth bypassed
        assert response.status_code == status.HTTP_201_CREATED, response.text
        data = response.json()
        assert data["name"] == payload["name"]
        assert data["author"] == payload["author"]
        assert all(kw in data["keywords"] for kw in payload["keywords"])
        assert "id" in data and isinstance(data["id"], int)
        assert "uuid" in data and isinstance(data["uuid"], str)
        assert "version" in data and data["version"] == 1
        assert data["is_deleted"] is False

    def test_create_prompt_duplicate_name(self, client: TestClient):
        payload = get_sample_prompt_payload("DuplicateName")
        create_prompt_utility(client, "DuplicateName")  # First creation

        response = client.post(f"{API_V1_PROMPTS_PREFIX}/", json=payload)
        assert response.status_code == status.HTTP_409_CONFLICT, response.text
        assert "already exists" in response.json()["detail"]  # Based on PromptsDatabase behavior

    def test_create_prompt_invalid_input_empty_name(self, client: TestClient):
        payload = get_sample_prompt_payload("InvalidInput")
        payload["name"] = ""
        response = client.post(f"{API_V1_PROMPTS_PREFIX}/", json=payload)
        # Pydantic validation should catch this before DBInputError
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY, response.text

    def test_create_prompt_input_error_from_db(self, client: TestClient, monkeypatch):
        payload = get_sample_prompt_payload("DBInputError")

        # Mock the PromptsDatabase dependency for this specific test
        mock_db_instance = MagicMock(spec=PromptsDatabase)
        mock_db_instance.add_prompt.side_effect = InputError("DB validation failed")

        def override_get_prompts_db_for_input_error():
            return mock_db_instance

        original_override = fastapi_app.dependency_overrides.get(get_prompts_db_for_user)
        fastapi_app.dependency_overrides[get_prompts_db_for_user] = override_get_prompts_db_for_input_error

        response = client.post(f"{API_V1_PROMPTS_PREFIX}/", json=payload)
        assert response.status_code == status.HTTP_400_BAD_REQUEST, response.text
        assert "DB validation failed" in response.json()["detail"]

        # Restore original dependency override
        if original_override:
            fastapi_app.dependency_overrides[get_prompts_db_for_user] = original_override
        else:
            del fastapi_app.dependency_overrides[get_prompts_db_for_user]

    @pytest.mark.parametrize("identifier_type", ["id", "uuid", "name"])
    def test_get_prompt_by_identifier(self, client: TestClient, identifier_type: str):
        created_prompt = create_prompt_utility(client, f"GetBy{identifier_type.capitalize()}")

        identifier_value = created_prompt[identifier_type]
        if identifier_type == "name":
            identifier_value = urllib.parse.quote(created_prompt["name"])

        response = client.get(f"{API_V1_PROMPTS_PREFIX}/{identifier_value}")
        assert response.status_code == status.HTTP_200_OK, response.text
        data = response.json()
        assert data[identifier_type] == created_prompt[identifier_type]  # Compare with original, unquoted name
        assert data["name"] == created_prompt["name"]

    def test_get_prompt_not_found(self, client: TestClient):
        response_id = client.get(f"{API_V1_PROMPTS_PREFIX}/9999999")
        assert response_id.status_code == status.HTTP_404_NOT_FOUND, response_id.text

        response_uuid = client.get(f"{API_V1_PROMPTS_PREFIX}/00000000-0000-0000-0000-000000000000")
        assert response_uuid.status_code == status.HTTP_404_NOT_FOUND, response_uuid.text

        non_existent_name = "ThisNameShouldNotExistForSure123"
        response_name = client.get(f"{API_V1_PROMPTS_PREFIX}/{urllib.parse.quote(non_existent_name)}")
        assert response_name.status_code == status.HTTP_404_NOT_FOUND, response_name.text

    def test_list_prompts_basic_and_pagination(self, client: TestClient):
        # Clearer state: ensure no prompts before starting or use a unique user
        create_prompt_utility(client, "ListPrompt1")
        create_prompt_utility(client, "ListPrompt2")

        # Test default pagination
        response = client.get(f"{API_V1_PROMPTS_PREFIX}/")
        assert response.status_code == status.HTTP_200_OK, response.text
        data = response.json()
        assert "items" in data
        assert len(data["items"]) <= 10  # Default per_page
        assert data["total_items"] >= 2
        assert data["current_page"] == 1

        # Test custom pagination
        response_page2 = client.get(f"{API_V1_PROMPTS_PREFIX}/?page=2&per_page=1")
        assert response_page2.status_code == status.HTTP_200_OK
        data_page2 = response_page2.json()
        assert data_page2["total_items"] >= 2
        assert len(data_page2["items"]) <= 1
        if data_page2["total_items"] >= 2:  # Only check current_page if items exist for it
            assert data_page2["current_page"] == 2

    def test_list_prompts_empty(self, client: TestClient):
        # This test assumes a fresh DB for the user.
        # If prompts were created by other tests for the same user, this might fail.
        # The `client` fixture *should* provide a fresh DB for each test function.
        response = client.get(f"{API_V1_PROMPTS_PREFIX}/")
        assert response.status_code == status.HTTP_200_OK, response.text
        data = response.json()
        # Check if any prompts were created by this test function before this call.
        # If not, then items should be 0.
        # For a truly isolated test, ensure the DB is empty or query for specific items.
        # Assuming a clean slate for this test based on `client` fixture behavior:
        # assert len(data["items"]) == 0
        # assert data["total_items"] == 0
        # More robustly, just check structure:
        assert "items" in data and isinstance(data["items"], list)
        assert "total_pages" in data
        assert "current_page" in data
        assert "total_items" in data

    @pytest.mark.parametrize("page, per_page, expected_status", [
        (0, 10, status.HTTP_422_UNPROCESSABLE_ENTITY),  # page < 1
        (1, 0, status.HTTP_422_UNPROCESSABLE_ENTITY),  # per_page < 1
        (1, 101, status.HTTP_422_UNPROCESSABLE_ENTITY)  # per_page > 100
    ])
    def test_list_prompts_invalid_pagination_params(self, client: TestClient, page, per_page, expected_status):
        response = client.get(f"{API_V1_PROMPTS_PREFIX}/?page={page}&per_page={per_page}")
        assert response.status_code == expected_status

    def test_update_prompt_success(self, client: TestClient):
        created_prompt = create_prompt_utility(client, "ToUpdate")
        prompt_id_to_update = created_prompt["id"]

        update_payload = get_sample_prompt_payload("Updated")
        update_payload["keywords"].append("new_kw_after_update")

        response = client.put(
            f"{API_V1_PROMPTS_PREFIX}/{prompt_id_to_update}",
            json=update_payload
        )
        assert response.status_code == status.HTTP_200_OK, response.text
        data = response.json()
        assert data["name"] == update_payload["name"]
        assert data["author"] == update_payload["author"]
        assert "new_kw_after_update" in data["keywords"]
        assert data["id"] == prompt_id_to_update  # ID should remain the same if PUT updates based on identifier
        assert data["version"] == created_prompt["version"] + 1  # Assuming version increments

        # Verify with a GET
        get_response = client.get(f"{API_V1_PROMPTS_PREFIX}/{prompt_id_to_update}")
        assert get_response.status_code == status.HTTP_200_OK
        assert get_response.json()["name"] == update_payload["name"]

    def test_update_prompt_not_found(self, client: TestClient):
        update_payload = get_sample_prompt_payload("UpdateNotFound")
        response = client.put(f"{API_V1_PROMPTS_PREFIX}/999999", json=update_payload)
        assert response.status_code == status.HTTP_404_NOT_FOUND, response.text

    def test_update_prompt_name_conflict(self, client: TestClient):
        prompt1 = create_prompt_utility(client, "Prompt1ForConflict")
        prompt2 = create_prompt_utility(client, "Prompt2ForConflict")

        update_payload = get_sample_prompt_payload("UpdatedToConflict")
        update_payload["name"] = prompt1["name"]  # Try to rename prompt2 to prompt1's name

        response = client.put(f"{API_V1_PROMPTS_PREFIX}/{prompt2['id']}", json=update_payload)
        assert response.status_code == status.HTTP_409_CONFLICT, response.text
        assert "already exists" in response.json()["detail"]

    def test_delete_prompt_success_and_verify_soft_delete(self, client: TestClient):
        created_prompt = create_prompt_utility(client, "ToDelete")
        prompt_id_to_delete = created_prompt["id"]

        delete_response = client.delete(f"{API_V1_PROMPTS_PREFIX}/{prompt_id_to_delete}")
        assert delete_response.status_code == status.HTTP_204_NO_CONTENT, delete_response.text

        # Verify it's not found by default
        get_response_normal = client.get(f"{API_V1_PROMPTS_PREFIX}/{prompt_id_to_delete}")
        assert get_response_normal.status_code == status.HTTP_404_NOT_FOUND

        # Verify it IS found with include_deleted=true
        get_response_deleted = client.get(f"{API_V1_PROMPTS_PREFIX}/{prompt_id_to_delete}?include_deleted=true")
        assert get_response_deleted.status_code == status.HTTP_200_OK
        assert get_response_deleted.json()["is_deleted"] is True
        assert get_response_deleted.json()["id"] == prompt_id_to_delete

        # Verify it's not in the default list
        list_response = client.get(f"{API_V1_PROMPTS_PREFIX}/")
        assert list_response.status_code == status.HTTP_200_OK
        listed_ids = [item['id'] for item in list_response.json()['items']]
        assert prompt_id_to_delete not in listed_ids

        # Verify it IS in the list with include_deleted=true
        list_deleted_response = client.get(f"{API_V1_PROMPTS_PREFIX}/?include_deleted=true")
        assert list_deleted_response.status_code == status.HTTP_200_OK
        listed_deleted_ids = [item['id'] for item in list_deleted_response.json()['items']]
        assert prompt_id_to_delete in listed_deleted_ids

    def test_delete_prompt_not_found(self, client: TestClient):
        response = client.delete(f"{API_V1_PROMPTS_PREFIX}/999999")
        assert response.status_code == status.HTTP_404_NOT_FOUND, response.text

    def test_search_prompts(self, client: TestClient):
        unique_name_keyword = "SuperUniqueSearchName123"
        unique_details_keyword = "HyperUniqueSearchDetails456"
        shared_keyword = "CommonSearchTerm"

        create_prompt_utility(client, f"{unique_name_keyword} Alpha {shared_keyword}")
        p2_payload = get_sample_prompt_payload(f"Beta {shared_keyword}")
        p2_payload["details"] = unique_details_keyword
        client.post(f"{API_V1_PROMPTS_PREFIX}/", json=p2_payload).raise_for_status()

        # Search by name
        response_name = client.post(
            f"{API_V1_PROMPTS_PREFIX}/search",
            params={"search_query": unique_name_keyword, "search_fields": ["name"]}
        )
        assert response_name.status_code == status.HTTP_200_OK, response_name.text
        data_name = response_name.json()
        assert data_name["total_matches"] == 1
        assert unique_name_keyword in data_name["items"][0]["name"]

        # Search by details
        response_details = client.post(
            f"{API_V1_PROMPTS_PREFIX}/search",
            params={"search_query": unique_details_keyword, "search_fields": ["details"]}
        )
        assert response_details.status_code == status.HTTP_200_OK, response_details.text
        data_details = response_details.json()
        assert data_details["total_matches"] == 1
        assert unique_details_keyword in data_details["items"][0]["details"]

        # Search by shared keyword across default fields (if name/details are default) or all fields
        response_shared = client.post(
            f"{API_V1_PROMPTS_PREFIX}/search",
            params={"search_query": shared_keyword}  # No search_fields, use DB default
        )
        assert response_shared.status_code == status.HTTP_200_OK, response_shared.text
        data_shared = response_shared.json()
        assert data_shared["total_matches"] >= 2

        # Search with no results
        response_no_results = client.post(
            f"{API_V1_PROMPTS_PREFIX}/search",
            params={"search_query": "ThisWillYieldNoResultsForSure"}
        )
        assert response_no_results.status_code == status.HTTP_200_OK
        data_no_results = response_no_results.json()
        assert data_no_results["total_matches"] == 0
        assert len(data_no_results["items"]) == 0

    def test_search_prompts_invalid_query(self, client: TestClient):
        response = client.post(f"{API_V1_PROMPTS_PREFIX}/search", params={"search_query": ""})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY  # Query('') fails min_length=1


#######################################################################################################################
# Keyword CRUD Integration Tests
#######################################################################################################################

class TestKeywordEndpoints:

    def test_create_keyword_success(self, client: TestClient):
        payload = get_sample_keyword_payload("CreateSuccess")
        response = client.post(f"{API_V1_PROMPTS_PREFIX}/keywords/", json=payload)
        assert response.status_code == status.HTTP_201_CREATED, response.text
        data = response.json()
        assert data["keyword_text"] == payload["keyword_text"].lower()  # Assuming normalization to lowercase
        assert "id" in data
        assert "uuid" in data

    def test_create_keyword_duplicate_normalized(self, client: TestClient):
        payload = get_sample_keyword_payload("DuplicateKW")
        client.post(f"{API_V1_PROMPTS_PREFIX}/keywords/", json=payload).raise_for_status()

        # Try creating again (exact same or different case that normalizes to same)
        payload_alt_case = {"keyword_text": payload["keyword_text"].upper()}
        response = client.post(f"{API_V1_PROMPTS_PREFIX}/keywords/", json=payload_alt_case)

        # Behavior for duplicates depends on db.add_keyword:
        # If it raises ConflictError for existing normalized keyword:
        assert response.status_code == status.HTTP_409_CONFLICT, response.text
        # Or if it handles it gracefully (e.g. returns existing), status might be 200 or 201
        # Based on prompt.py error handling, 409 is expected for ConflictError
        # assert "already exists" in response.json()["detail"] # Or similar message

    def test_create_keyword_invalid_input(self, client: TestClient):
        response = client.post(f"{API_V1_PROMPTS_PREFIX}/keywords/", json={"keyword_text": ""})
        # Depends on Pydantic schema for KeywordCreate or InputError from DB
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY or response.status_code == status.HTTP_400_BAD_REQUEST

    def test_list_keywords(self, client: TestClient):
        kw1_payload = get_sample_keyword_payload("ListKW1")
        kw2_payload = get_sample_keyword_payload("ListKW2")
        client.post(f"{API_V1_PROMPTS_PREFIX}/keywords/", json=kw1_payload).raise_for_status()
        client.post(f"{API_V1_PROMPTS_PREFIX}/keywords/", json=kw2_payload).raise_for_status()

        response = client.get(f"{API_V1_PROMPTS_PREFIX}/keywords/")
        assert response.status_code == status.HTTP_200_OK, response.text
        data = response.json()
        assert isinstance(data, list)
        # Assuming keywords are normalized (e.g., lowercased)
        normalized_kw1 = kw1_payload["keyword_text"].lower()
        normalized_kw2 = kw2_payload["keyword_text"].lower()
        assert normalized_kw1 in data
        assert normalized_kw2 in data

    def test_list_keywords_empty(self, client: TestClient):
        # Assumes a fresh DB state for keywords for this user
        response = client.get(f"{API_V1_PROMPTS_PREFIX}/keywords/")
        assert response.status_code == status.HTTP_200_OK, response.text
        data = response.json()
        assert data == []  # Expect an empty list

    def test_delete_keyword_success(self, client: TestClient):
        keyword_payload = get_sample_keyword_payload("ToDeleteKW")
        create_resp = client.post(f"{API_V1_PROMPTS_PREFIX}/keywords/", json=keyword_payload)
        assert create_resp.status_code == status.HTTP_201_CREATED
        keyword_text_to_delete = create_resp.json()["keyword_text"]  # Use normalized text

        delete_response = client.delete(
            f"{API_V1_PROMPTS_PREFIX}/keywords/{urllib.parse.quote(keyword_text_to_delete)}")
        assert delete_response.status_code == status.HTTP_204_NO_CONTENT

        # Verify it's not in the list
        list_response = client.get(f"{API_V1_PROMPTS_PREFIX}/keywords/")
        assert keyword_text_to_delete not in list_response.json()

    def test_delete_keyword_not_found(self, client: TestClient):
        non_existent_keyword = "ThisKeywordDoesNotExist123"
        response = client.delete(f"{API_V1_PROMPTS_PREFIX}/keywords/{urllib.parse.quote(non_existent_keyword)}")
        assert response.status_code == status.HTTP_404_NOT_FOUND


#######################################################################################################################
# Export Endpoints Integration Tests
#######################################################################################################################

class TestExportEndpoints:

    @pytest.fixture(autouse=True)  # Apply this mock to all tests in this class
    def mock_file_ops(self, monkeypatch):
        # Mock os.path.exists, open, os.remove for export tests to avoid actual file I/O
        # and control content easily.
        self.mocked_file_content = b""
        self.temp_file_path = "mocked_temp_export_file.tmp"

        def mock_exists(path):
            return path == self.temp_file_path

        def mock_open(path, mode):
            if path == self.temp_file_path and mode == "rb":
                mock_file = MagicMock()
                mock_file.read.return_value = self.mocked_file_content
                mock_file.__enter__.return_value = mock_file  # For 'with open(...) as f:'
                mock_file.__exit__.return_value = None
                return mock_file
            # Fallback to actual open for other paths if necessary, or raise error
            raise FileNotFoundError(f"Mocked open received unexpected path: {path}")

        def mock_remove(path):
            if path == self.temp_file_path:
                return
            raise FileNotFoundError(f"Mocked remove received unexpected path: {path}")

        monkeypatch.setattr(os.path, "exists", mock_exists)
        # Python's built-in `open` is harder to mock globally for just this module.
        # It's better to mock it where it's used, i.e., in db_export_prompts_formatted
        # For simplicity, we'll assume db_export... returns content or a path that we then mock `open` for.
        # The endpoint code itself calls open(), so we mock it in `prompts.py`'s scope.
        monkeypatch.setattr("tldw_Server_API.app.api.v1.endpoints.prompts.open", mock_open, raising=False)
        monkeypatch.setattr(os, "remove", mock_remove)

        # Mock the interop functions to control their output directly
        self.mock_db_export_prompts_formatted = MagicMock()
        self.mock_db_export_prompt_keywords_to_csv = MagicMock()

        monkeypatch.setattr(Prompts_Interop, "db_export_prompts_formatted", self.mock_db_export_prompts_formatted)
        monkeypatch.setattr(Prompts_Interop, "db_export_prompt_keywords_to_csv",
                            self.mock_db_export_prompt_keywords_to_csv)

        # Allow access to self.mocked_file_content and self.temp_file_path in tests
        yield self

    def test_export_prompts_csv_success(self, client: TestClient):
        self.mocked_file_content = b"id,name\n1,TestPromptCSV"
        self.mock_db_export_prompts_formatted.return_value = ("Successfully exported CSV", self.temp_file_path)

        response = client.get(f"{API_V1_PROMPTS_PREFIX}/export?export_format=csv")
        assert response.status_code == status.HTTP_200_OK, response.text
        data = response.json()
        assert "Successfully exported CSV" in data["message"]
        assert data["file_content_b64"] is not None
        decoded_content = base64.b64decode(data["file_content_b64"]).decode('utf-8')
        assert "TestPromptCSV" in decoded_content
        self.mock_db_export_prompts_formatted.assert_called_once()  # Verify it was called

    def test_export_prompts_markdown_success(self, client: TestClient):
        self.mocked_file_content = b"# TestPromptMD"
        self.mock_db_export_prompts_formatted.return_value = ("Successfully exported Markdown", self.temp_file_path)

        response = client.get(f"{API_V1_PROMPTS_PREFIX}/export?export_format=markdown")
        assert response.status_code == status.HTTP_200_OK, response.text
        data = response.json()
        assert "Successfully exported Markdown" in data["message"]
        decoded_content = base64.b64decode(data["file_content_b64"]).decode('utf-8')
        assert "# TestPromptMD" in decoded_content

    def test_export_prompts_no_prompts_found(self, client: TestClient):
        self.mock_db_export_prompts_formatted.return_value = ("No prompts found matching criteria.",
                                                              "None")  # Or path that mock_exists returns False for

        response = client.get(f"{API_V1_PROMPTS_PREFIX}/export?export_format=csv")
        assert response.status_code == status.HTTP_200_OK  # Endpoint handles "None" path gracefully
        data = response.json()
        assert "No prompts found" in data["message"]
        assert data["file_content_b64"] is None

    def test_export_prompts_interop_failure(self, client: TestClient):
        self.mock_db_export_prompts_formatted.return_value = ("Export failed internally",
                                                              "None")  # Simulates failure where file isn't created

        response = client.get(f"{API_V1_PROMPTS_PREFIX}/export?export_format=csv")
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Export failed" in response.json()["detail"]

    def test_export_prompts_invalid_format(self, client: TestClient):
        # The interop function `db_export_prompts_formatted` should raise ValueError for this.
        self.mock_db_export_prompts_formatted.side_effect = ValueError("Invalid export format")

        response = client.get(f"{API_V1_PROMPTS_PREFIX}/export?export_format=xml")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid export format" in response.json()["detail"]

    def test_export_keywords_csv_success(self, client: TestClient):
        self.mocked_file_content = b"keyword,prompt_ids\ntest_kw,1;2"
        self.mock_db_export_prompt_keywords_to_csv.return_value = ("Successfully exported keywords",
                                                                   self.temp_file_path)

        response = client.get(f"{API_V1_PROMPTS_PREFIX}/keywords/export-csv")
        assert response.status_code == status.HTTP_200_OK, response.text
        data = response.json()
        assert "Successfully exported keywords" in data["message"]
        decoded_content = base64.b64decode(data["file_content_b64"]).decode('utf-8')
        assert "test_kw" in decoded_content
        self.mock_db_export_prompt_keywords_to_csv.assert_called_once()

    def test_export_keywords_no_keywords_found(self, client: TestClient):
        self.mock_db_export_prompt_keywords_to_csv.return_value = ("No active keywords found.", "None")

        response = client.get(f"{API_V1_PROMPTS_PREFIX}/keywords/export-csv")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "No active keywords found" in data["message"]
        assert data["file_content_b64"] is None


#######################################################################################################################
# Sync Log Endpoint Integration Tests
#######################################################################################################################

class TestSyncLogEndpoint:

    def test_get_sync_log_success_empty(self, client: TestClient):
        response = client.get(f"{API_V1_PROMPTS_PREFIX}/sync-log")
        assert response.status_code == status.HTTP_200_OK, response.text
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0  # Assuming fresh DB has no sync logs initially

    def test_get_sync_log_with_entries(self, client: TestClient, monkeypatch):
        # Mock the DB call for this as creating actual sync log entries can be complex
        mock_db_instance = MagicMock(spec=PromptsDatabase)
        mock_log_entry = {
            "change_id": 1, "table_name": "Prompts", "row_uuid": "some-uuid",
            "operation": "INSERT", "change_timestamp": "2023-01-01T10:00:00", "user_id": TEST_USER_ID
        }
        mock_db_instance.get_sync_log_entries.return_value = [mock_log_entry]

        def override_get_prompts_db_for_sync_log():
            return mock_db_instance

        original_override = fastapi_app.dependency_overrides.get(get_prompts_db_for_user)
        fastapi_app.dependency_overrides[get_prompts_db_for_user] = override_get_prompts_db_for_sync_log

        response = client.get(f"{API_V1_PROMPTS_PREFIX}/sync-log?since_change_id=0&limit=10")
        assert response.status_code == status.HTTP_200_OK, response.text
        data = response.json()
        assert len(data) == 1
        assert data[0]["change_id"] == mock_log_entry["change_id"]
        assert data[0]["table_name"] == mock_log_entry["table_name"]
        mock_db_instance.get_sync_log_entries.assert_called_with(since_change_id=0, limit=10)

        # Restore
        if original_override:
            fastapi_app.dependency_overrides[get_prompts_db_for_user] = original_override
        else:
            del fastapi_app.dependency_overrides[get_prompts_db_for_user]

    def test_get_sync_log_db_error(self, client: TestClient, monkeypatch):
        mock_db_instance = MagicMock(spec=PromptsDatabase)
        mock_db_instance.get_sync_log_entries.side_effect = DatabaseError("Sync log query failed")

        def override_get_prompts_db_for_sync_log_error():
            return mock_db_instance

        original_override = fastapi_app.dependency_overrides.get(get_prompts_db_for_user)
        fastapi_app.dependency_overrides[get_prompts_db_for_user] = override_get_prompts_db_for_sync_log_error

        response = client.get(f"{API_V1_PROMPTS_PREFIX}/sync-log")
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Database error" in response.json()["detail"]

        # Restore
        if original_override:
            fastapi_app.dependency_overrides[get_prompts_db_for_user] = original_override
        else:
            del fastapi_app.dependency_overrides[get_prompts_db_for_user]

# TODO: Add tests for Sync Log endpoint if it's not admin-only or mock admin user.
# TODO: Test edge cases for pagination, search with no results, various include_deleted flags.
# TODO: Test PromptUpdate logic carefully if a separate PATCH endpoint is added.

#
# End of test_prompts_api.py
#######################################################################################################################
