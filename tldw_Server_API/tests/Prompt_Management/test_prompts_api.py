# tests/integration/api/v1/test_prompts_api.py
# Description:
#
# Imports
import os
from datetime import datetime, timezone
from pathlib import Path
import pytest
import json
import base64
import urllib.parse
from unittest.mock import MagicMock
#
# Third-Party Imports
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps import Prompts_DB_Deps
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

# Fixture for a specific API token value for direct testing of verify_token
@pytest.fixture(scope="session")
def actual_test_api_key() -> str:
    return "this_is_the_actual_single_user_key_for_testing"

# Standalone tests for verify_token (if they are in test_prompts_api.py)
@pytest.mark.asyncio
async def test_verify_token_success_single_user_mode(monkeypatch, actual_test_api_key: str):
    # Simulate single-user mode and set the expected API key
    original_single_user_mode = settings.get("SINGLE_USER_MODE")
    original_api_key = settings.get("SINGLE_USER_API_KEY")

    monkeypatch.setitem(settings, "SINGLE_USER_MODE", True)
    monkeypatch.setitem(settings, "SINGLE_USER_API_KEY", actual_test_api_key)
    try:
        assert await verify_token(Token=actual_test_api_key) is True
    finally:
        # Restore original settings
        monkeypatch.setitem(settings, "SINGLE_USER_MODE", original_single_user_mode)
        if original_api_key is not None:
            monkeypatch.setitem(settings, "SINGLE_USER_API_KEY", original_api_key)
        else:
            monkeypatch.delitem(settings, "SINGLE_USER_API_KEY", raising=False)


@pytest.mark.asyncio
async def test_verify_token_missing_token_header_direct(): # Renamed for clarity
    with pytest.raises(HTTPException) as excinfo:
        await verify_token(Token=None) # FastAPI would pass None if Header is missing
    assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Missing authentication token" in excinfo.value.detail


@pytest.mark.asyncio
async def test_verify_token_invalid_token_single_user_mode(monkeypatch, actual_test_api_key: str):
    original_single_user_mode = settings.get("SINGLE_USER_MODE")
    original_api_key = settings.get("SINGLE_USER_API_KEY")

    monkeypatch.setitem(settings, "SINGLE_USER_MODE", True)
    monkeypatch.setitem(settings, "SINGLE_USER_API_KEY", actual_test_api_key)
    try:
        with pytest.raises(HTTPException) as excinfo:
            await verify_token(Token="completely-wrong-token")
        assert excinfo.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid authentication token" in excinfo.value.detail
    finally:
        monkeypatch.setitem(settings, "SINGLE_USER_MODE", original_single_user_mode)
        if original_api_key is not None:
            monkeypatch.setitem(settings, "SINGLE_USER_API_KEY", original_api_key)
        else:
            monkeypatch.delitem(settings, "SINGLE_USER_API_KEY", raising=False)

@pytest.mark.asyncio
async def test_verify_token_server_misconfigured_key_missing_single_user(monkeypatch):
    original_single_user_mode = settings.get("SINGLE_USER_MODE")
    original_api_key = settings.get("SINGLE_USER_API_KEY")

    monkeypatch.setitem(settings, "SINGLE_USER_MODE", True)
    monkeypatch.setitem(settings, "SINGLE_USER_API_KEY", None) # Simulate API key not set
    try:
        with pytest.raises(HTTPException) as excinfo:
            await verify_token(Token="any-token-will-do-for-this-check")
        assert excinfo.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Server authentication misconfigured (API key missing)" in excinfo.value.detail
    finally:
        monkeypatch.setitem(settings, "SINGLE_USER_MODE", original_single_user_mode)
        if original_api_key is not None:
            monkeypatch.setitem(settings, "SINGLE_USER_API_KEY", original_api_key)
        else:
            # If original was None or not present, restore that state
             monkeypatch.setitem(settings, "SINGLE_USER_API_KEY", None) # Or delitem if that's more appropriate for your settings load

@pytest.fixture(scope="session")
def test_api_token(actual_test_api_key: str): # Depends on the actual_test_api_key fixture
    if settings.get("SINGLE_USER_MODE"):
        # For single-user mode, tests should use a known, valid API key.
        # We can set this key in settings for the test session if it's not already there.
        # This assumes your settings["SINGLE_USER_API_KEY"] is the source of truth.
        # If settings() are reloaded per test function (via `client` fixture),
        # then settings["SINGLE_USER_API_KEY"] might default to "default-secret-key-for-single-user"
        # unless explicitly patched for the test.
        # Using actual_test_api_key makes it consistent.
        return actual_test_api_key
    # For multi-user mode, this is a placeholder, as real JWTs are complex.
    # The verify_token override in the `client` fixture handles this for endpoint tests.
    return "fixed_test_api_token_for_pytest_jwt_placeholder"

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
def client(test_user: User, test_api_token: str, tmp_path: Path, monkeypatch):
    # (Client fixture as corrected in the previous good response, ensuring
    # mock_get_prompts_db_path_for_user takes user_id: int,
    # settings["USER_DB_BASE_DIR"] is patched with setitem,
    # PromptsDBDepsModule.MAIN_USER_DATA_BASE_DIR is patched,
    # verify_token is overridden to return True for client tests,
    # client.headers is set to {"Token": test_api_token} )
    def mock_get_prompts_db_path_for_user(user_id: int, db_version: str = "v2") -> Path:
        user_db_dir = tmp_path / str(user_id) / "prompts_user_dbs"
        user_db_dir.mkdir(parents=True, exist_ok=True)
        if db_version == "v2":
            return user_db_dir / "user_prompts_v2.sqlite"
        return user_db_dir / f"user_prompts_{db_version}.sqlite"

    monkeypatch.setattr(Prompts_DB_Deps, "_get_prompts_db_path_for_user", mock_get_prompts_db_path_for_user)

    original_user_db_base_dir_in_settings = settings.get("USER_DB_BASE_DIR")
    monkeypatch.setitem(settings, "USER_DB_BASE_DIR", tmp_path)

    original_main_user_data_base_dir_in_module = getattr(Prompts_DB_Deps, "MAIN_USER_DATA_BASE_DIR", None)
    monkeypatch.setattr(Prompts_DB_Deps, "MAIN_USER_DATA_BASE_DIR", tmp_path)

    def override_get_request_user():
        return test_user

    async def override_verify_token_dependency_for_client_tests():
        return True

    original_overrides = fastapi_app.dependency_overrides.copy()
    fastapi_app.dependency_overrides[get_request_user] = override_get_request_user
    fastapi_app.dependency_overrides[verify_token] = override_verify_token_dependency_for_client_tests

    test_client_instance = TestClient(fastapi_app)
    test_client_instance.headers = {"Token": test_api_token}

    try:
        yield test_client_instance
    finally:
        fastapi_app.dependency_overrides = original_overrides
        close_all_cached_prompts_db_instances()
        if original_user_db_base_dir_in_settings is not None:
            monkeypatch.setitem(settings, "USER_DB_BASE_DIR", original_user_db_base_dir_in_settings)
        else:
            monkeypatch.delitem(settings, "USER_DB_BASE_DIR", raising=False)
        if original_main_user_data_base_dir_in_module is not None:
            monkeypatch.setattr(Prompts_DB_Deps, "MAIN_USER_DATA_BASE_DIR",
                                original_main_user_data_base_dir_in_module)


@pytest.fixture(scope="function")
def client_with_auth(client_env_setup, monkeypatch, actual_api_token_value: str):
    """
    Provides a TestClient where actual auth logic (verify_token) is tested.
    `settings.API_BEARER` is monkeypatched to `actual_api_token_value`.
    """
    original_api_bearer = getattr(settings, "API_BEARER", None)
    monkeypatch.setitem(settings, "SINGLE_USER_API_KEY", actual_api_token_value)

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
    monkeypatch.setitem(settings, "SINGLE_USER_API_KEY", actual_api_token_value)
    assert await verify_token(Token=f"Bearer {actual_api_token_value}") is True
    assert await verify_token(Token=actual_api_token_value) is True  # Without Bearer prefix


@pytest.mark.asyncio
async def test_verify_token_missing_token(monkeypatch, actual_api_token_value: str):
    monkeypatch.setitem(settings, "SINGLE_USER_API_KEY", actual_api_token_value)
    with pytest.raises(HTTPException) as exc_info:
        await verify_token(Token=None)
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Missing authentication token" in exc_info.value.detail


@pytest.mark.asyncio
async def test_verify_token_invalid_token(monkeypatch, actual_api_token_value: str):
    monkeypatch.setitem(settings, "SINGLE_USER_API_KEY", actual_api_token_value)
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
    def get_sample_prompt_payload(self, name_suffix: str = "") -> dict:
        return {
            "name": f"Test Prompt {name_suffix}",
            "author": "API Test Author",
            "details": "Some test details.",
            "system_prompt": "System prompt content.",
            "user_prompt": "User prompt content.",
            "keywords": [f"kw{name_suffix.lower()}", "common"]
        }

    def create_prompt_utility(self, client: TestClient, name_suffix: str) -> dict:
        payload = self.get_sample_prompt_payload(name_suffix)
        response = client.post(f"{API_V1_PROMPTS_PREFIX}/", json=payload)
        response.raise_for_status()  # Will raise for non-2xx
        return response.json()

    @pytest.mark.parametrize("identifier_type", ["id", "uuid", "name"])
    def test_get_prompt_by_identifier(self, client: TestClient, identifier_type: str):
        created_prompt = self.create_prompt_utility(client, f"GetBy{identifier_type.capitalize()}")  # Use self
        identifier_to_fetch = created_prompt[identifier_type]

        if identifier_type == "name":
            identifier_to_fetch = urllib.parse.quote(str(identifier_to_fetch))

        response = client.get(f"{API_V1_PROMPTS_PREFIX}/{identifier_to_fetch}")
        assert response.status_code == 200, response.text
        data = response.json()
        assert data[identifier_type] == created_prompt[identifier_type]
        assert data["name"] == created_prompt["name"]

    def test_create_prompt_success(self, client: TestClient):
        payload = get_sample_prompt_payload("CreateSuccess")
        response = client.post(f"{API_V1_PROMPTS_PREFIX}/", json=payload)
        assert response.status_code == status.HTTP_201_CREATED, response.text
        data = response.json()
        assert data["name"] == payload["name"]
        assert data["author"] == payload["author"]
        assert all(kw in data["keywords"] for kw in payload["keywords"])
        assert "id" in data and isinstance(data["id"], int)
        assert "uuid" in data and isinstance(data["uuid"], str)
        assert "version" in data and data["version"] == 1
        assert "deleted" in data and data["deleted"] is False # Check "deleted" field

    def test_create_prompt_duplicate_name(self, client: TestClient):
        payload = get_sample_prompt_payload("DuplicateName")
        response1 = client.post(f"{API_V1_PROMPTS_PREFIX}/", json=payload)
        assert response1.status_code == status.HTTP_201_CREATED, response1.text

        response2 = client.post(f"{API_V1_PROMPTS_PREFIX}/", json=payload) # Attempt to create again
        assert response2.status_code == status.HTTP_409_CONFLICT, response2.text
        assert "already exists" in response2.json()["detail"].lower()


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

        update_payload = get_sample_prompt_payload("Updated") # Gets a new name
        update_payload["name"] = "Updated Prompt Name Completely" # Ensure a different name for update
        update_payload["keywords"].append("new_kw_after_update")

        response = client.put(f"{API_V1_PROMPTS_PREFIX}/{prompt_id_to_update}", json=update_payload)
        assert response.status_code == status.HTTP_200_OK, response.text
        data = response.json()
        assert data["name"] == update_payload["name"] # Should reflect the new name
        assert data["author"] == update_payload["author"]
        assert "new_kw_after_update" in data["keywords"]
        assert data["id"] == prompt_id_to_update # ID should remain the same

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
        prompt2 = create_prompt_utility(client, "Prompt2ForConflict") # This will have ID 2 if DB is fresh per test

        update_payload = get_sample_prompt_payload("UpdatedToConflict")
        update_payload["name"] = prompt1["name"]  # Try to rename prompt2 to prompt1's name

        response = client.put(f"{API_V1_PROMPTS_PREFIX}/{prompt2['id']}", json=update_payload)
        assert response.status_code == status.HTTP_409_CONFLICT, response.text
        assert "already exists" in response.json()["detail"].lower()

    def test_delete_prompt_success_and_verify_soft_delete(self, client: TestClient):
        created_prompt = create_prompt_utility(client, "ToDelete")
        prompt_id_to_delete = created_prompt["id"]

        delete_response = client.delete(f"{API_V1_PROMPTS_PREFIX}/{prompt_id_to_delete}")
        assert delete_response.status_code == status.HTTP_204_NO_CONTENT, delete_response.text

        get_response_normal = client.get(f"{API_V1_PROMPTS_PREFIX}/{prompt_id_to_delete}")
        assert get_response_normal.status_code == status.HTTP_404_NOT_FOUND

        get_response_deleted = client.get(f"{API_V1_PROMPTS_PREFIX}/{prompt_id_to_delete}?include_deleted=true")
        assert get_response_deleted.status_code == status.HTTP_200_OK
        assert get_response_deleted.json()["deleted"] is True # Check "deleted" field

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

        duplicate_payload = {"keyword_text": "  DUPLICATEKW  "}  # Normalizes to the same
        response = client.post(f"{API_V1_PROMPTS_PREFIX}/keywords/", json=duplicate_payload)
        # If add_keyword now raises ConflictError for active duplicates
        assert response.status_code == status.HTTP_409_CONFLICT, response.text
        assert "already exists" in response.json()["detail"].lower()

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
    @pytest.fixture(autouse=True)
    def setup_mocks(self, monkeypatch, tmp_path): # Added tmp_path for temp file
        self.mock_db_export_prompts_formatted = MagicMock()
        self.mock_db_export_prompt_keywords_to_csv = MagicMock()

        monkeypatch.setattr("tldw_Server_API.app.api.v1.endpoints.prompts.db_export_prompts_formatted", self.mock_db_export_prompts_formatted)
        monkeypatch.setattr("tldw_Server_API.app.api.v1.endpoints.prompts.db_export_prompt_keywords_to_csv", self.mock_db_export_prompt_keywords_to_csv)

        self.mock_os_path_exists = MagicMock(return_value=True)
        self.mock_os_remove = MagicMock()
        monkeypatch.setattr("os.path.exists", self.mock_os_path_exists)
        monkeypatch.setattr("os.remove", self.mock_os_remove)

        # Use tmp_path provided by pytest for a unique temp file per test run
        self.temp_file_path = str(tmp_path / "test_export.tmp")

    def test_export_prompts_csv_success(self, client: TestClient):
        # Ensure the mocked temp file exists for the test
        with open(self.temp_file_path, "w") as f: f.write("id,name\n1,TestPromptCSV")
        self.mock_db_export_prompts_formatted.return_value = ("Successfully exported CSV", self.temp_file_path)
        self.mock_os_path_exists.return_value = True # Make sure it "exists"

        response = client.get(f"{API_V1_PROMPTS_PREFIX}/export?export_format=csv")
        assert response.status_code == status.HTTP_200_OK, response.text
        data = response.json()
        assert "file_content_b64" in data and data["file_content_b64"] is not None
        decoded_content = base64.b64decode(data["file_content_b64"]).decode('utf-8')
        assert "TestPromptCSV" in decoded_content
        self.mock_os_remove.assert_called_with(self.temp_file_path)

    def test_export_prompts_markdown_success(self, client: TestClient):
        with open(self.temp_file_path, "w") as f: f.write("# TestPromptMD")
        self.mock_db_export_prompts_formatted.return_value = ("Successfully exported Markdown", self.temp_file_path)
        self.mock_os_path_exists.return_value = True

        response = client.get(f"{API_V1_PROMPTS_PREFIX}/export?export_format=markdown")
        assert response.status_code == status.HTTP_200_OK, response.text
        data = response.json()
        assert "file_content_b64" in data and data["file_content_b64"] is not None
        decoded_content = base64.b64decode(data["file_content_b64"]).decode('utf-8')
        assert "TestPromptMD" in decoded_content
        self.mock_os_remove.assert_called_with(self.temp_file_path)

    def test_export_prompts_no_prompts_found(self, client: TestClient):
        self.mock_db_export_prompts_formatted.return_value = ("No prompts found matching criteria.", "None")
        self.mock_os_path_exists.return_value = False

        response = client.get(f"{API_V1_PROMPTS_PREFIX}/export?export_format=csv")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "No prompts found matching criteria."
        assert data["file_content_b64"] is None

    def test_export_prompts_interop_failure(self, client: TestClient):
        self.mock_db_export_prompts_formatted.return_value = ("Export failed internally", "None")
        self.mock_os_path_exists.return_value = False

        response = client.get(f"{API_V1_PROMPTS_PREFIX}/export?export_format=csv")
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Export failed" in response.json()["detail"] # Check the specific detail message

    def test_export_prompts_invalid_format(self, client: TestClient):
        self.mock_db_export_prompts_formatted.side_effect = ValueError("Unsupported export_format: xml")
        response = client.get(f"{API_V1_PROMPTS_PREFIX}/export?export_format=xml")
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Unsupported export_format: xml" in response.json()["detail"]

    def test_export_keywords_csv_success(self, client: TestClient):
        with open(self.temp_file_path, "w") as f: f.write("keyword,prompt_ids\ntest_kw,1;2")
        self.mock_db_export_prompt_keywords_to_csv.return_value = ("Successfully exported keywords", self.temp_file_path)
        self.mock_os_path_exists.return_value = True

        response = client.get(f"{API_V1_PROMPTS_PREFIX}/keywords/export-csv")
        assert response.status_code == status.HTTP_200_OK, response.text
        data = response.json()
        assert "Successfully exported keywords" in data["message"] # Match exact expected message
        assert "file_content_b64" in data and data["file_content_b64"] is not None
        self.mock_os_remove.assert_called_with(self.temp_file_path)

    def test_export_keywords_no_keywords_found(self, client: TestClient):
        self.mock_db_export_prompt_keywords_to_csv.return_value = ("No active keywords found.", "None")
        self.mock_os_path_exists.return_value = False

        response = client.get(f"{API_V1_PROMPTS_PREFIX}/keywords/export-csv")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "No active keywords found." # Match exact expected message
        assert data["file_content_b64"] is None


#######################################################################################################################
# Sync Log Endpoint Integration Tests
#######################################################################################################################

class TestSyncLogEndpoint:

    def test_get_sync_log_success_empty(self, client: TestClient, monkeypatch):
        mock_db = MagicMock(spec=PromptsDatabase)
        mock_db.get_sync_log_entries.return_value = []
        monkeypatch.setattr(Prompts_DB_Deps, "get_prompts_db_for_user", lambda: mock_db)  # Patch the dep source

        response = client.get(f"{API_V1_PROMPTS_PREFIX}/sync-log")
        assert response.status_code == status.HTTP_200_OK, response.text
        assert response.json() == []
        mock_db.get_sync_log_entries.assert_called_once_with(since_change_id=0, limit=100)

    def test_get_sync_log_with_entries(self, client: TestClient, monkeypatch):
        mock_db = MagicMock(spec=PromptsDatabase)
        mock_log_entry_from_db = {  # This is what db.get_sync_log_entries would return
            "change_id": 1,
            "entity": "Prompts",
            "entity_uuid": "some-uuid-123",
            "operation": "create",
            "timestamp": datetime.now(timezone.utc),  # Actual datetime object
            "client_id": "test_client_xyz",
            "version": 1,
            "payload": {"name": "Test Prompt Sync"}  # Payload as dict
        }
        mock_db.get_sync_log_entries.return_value = [mock_log_entry_from_db]

        monkeypatch.setattr(Prompts_DB_Deps, "get_prompts_db_for_user", lambda: mock_db)

        response = client.get(f"{API_V1_PROMPTS_PREFIX}/sync-log?since_change_id=0&limit=10")
        assert response.status_code == status.HTTP_200_OK, response.text
        data = response.json()
        assert len(data) == 1
        # Compare relevant fields, convert timestamp back if needed for exact match
        assert data[0]["change_id"] == mock_log_entry_from_db["change_id"]
        assert data[0]["entity_uuid"] == mock_log_entry_from_db["entity_uuid"]
        assert data[0]["payload"] == mock_log_entry_from_db["payload"]
        # Timestamp will be string in JSON, convert mock's for comparison or parse response's
        assert datetime.fromisoformat(data[0]["timestamp"].replace("Z", "+00:00")) == mock_log_entry_from_db[
            "timestamp"]

    def test_get_sync_log_db_error(self, client: TestClient, monkeypatch):
        mock_db = MagicMock(spec=PromptsDatabase)
        mock_db.get_sync_log_entries.side_effect = DatabaseError("Sync log query failed")
        monkeypatch.setattr(Prompts_DB_Deps, "get_prompts_db_for_user", lambda: mock_db)

        response = client.get(f"{API_V1_PROMPTS_PREFIX}/sync-log")
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Database error." in response.json()["detail"]  # Check specific message from endpoint

# TODO: Add tests for Sync Log endpoint if it's not admin-only or mock admin user.
# TODO: Test edge cases for pagination, search with no results, various include_deleted flags.
# TODO: Test PromptUpdate logic carefully if a separate PATCH endpoint is added.

#
# End of test_prompts_api.py
#######################################################################################################################
