# test_api_characters.py
import base64
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Generator
from io import BytesIO

#
# Third-party imports
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage, PngImagePlugin  # Corrected PIL import

# Third-party imports
from hypothesis import given, strategies as st, settings, HealthCheck, assume, event, Verbosity, note
import os
from unittest.mock import patch, MagicMock  # For unit tests
from loguru import logger

from tldw_Server_API.app.api.v1.schemas.character_schemas import CharacterUpdate, MAX_NAME_LENGTH
from tldw_Server_API.tests.Characters._ml_import_stubs import stub_heavy_ml_imports

#
# Local Imports
stub_heavy_ml_imports()

from tldw_Server_API.app.main import app as fastapi_app  # Your FastAPI app instance
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
    CharactersRAGDBError,
)
from tldw_Server_API.app.api.v1.endpoints import characters_endpoint as characters_api_module

#######################################################################################################################
#
# --- Constants ---
BASE_URL_V1 = "/api/v1"
# Ensure this matches the prefix in your app.include_router for the characters API
CHARACTERS_ENDPOINT_PREFIX = "/api/v1/characters"
CHARACTER_FOLDER_TAG_PREFIX = "__tldw_folder_id:"


# --- Helper Functions / Fixtures for Integration Tests ---


@pytest.fixture(scope="function")
def test_db(tmp_path) -> Generator[CharactersRAGDB, Any, None]:
    # Using a file-based database to avoid in-memory threading issues
    db_path = tmp_path / "test_characters.db"
    db_instance = CharactersRAGDB(str(db_path), client_id=f"db-client-test-{uuid.uuid4().hex[:6]}")
    yield db_instance
    db_instance.close_connection()


@pytest.fixture
def client(test_db: CharactersRAGDB) -> Generator[TestClient, Any, None]:
    """
    Provides a TestClient instance with the real DB dependency overridden
    for integration tests. Also handles CSRF token setup.
    """
    # This is where get_chacha_db_for_user is defined or imported in your actual app
    # For testing, we override the dependency that the *endpoints file* uses.
    # The path for dependency_overrides should be the actual dependency callable.
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
    from tldw_Server_API.app.core.config import settings as global_settings
    from tldw_Server_API.tests.test_config import TestConfig

    # Ensure auth env is set so SINGLE_USER_API_KEY matches headers
    try:
        TestConfig.setup_test_environment()
        # Reset settings to pick up env change if necessary
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

        reset_settings()
    except Exception:
        _ = None

    def override_get_db_for_test():
        logger.info("<<<<< OVERRIDE override_get_db_for_test CALLED >>>>>")
        try:
            yield test_db
        finally:
            pass  # test_db fixture handles its own close

    # Disable CSRF protection for tests
    original_csrf_setting = global_settings.get("CSRF_ENABLED", None)
    global_settings["CSRF_ENABLED"] = False

    fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_get_db_for_test
    with TestClient(fastapi_app) as c:
        # Set authentication header for single-user mode
        c.headers["X-API-KEY"] = TestConfig.TEST_API_KEY
        yield c

    # Restore original settings
    fastapi_app.dependency_overrides.clear()
    try:
        TestConfig.reset_settings()
    except Exception:
        _ = None
    if original_csrf_setting is None:
        global_settings.pop("CSRF_ENABLED", None)
    else:
        global_settings["CSRF_ENABLED"] = original_csrf_setting


@pytest.fixture
def client_with_csrf(test_db: CharactersRAGDB) -> Generator[TestClient, Any, None]:
    """
    Provides a TestClient instance with CSRF protection enabled.
    Use this fixture when you want to test CSRF token handling.
    """
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
    from tldw_Server_API.tests.test_config import TestConfig

    def override_get_db_for_test():
        logger.info("<<<<< OVERRIDE override_get_db_for_test WITH CSRF ENABLED >>>>>")
        try:
            yield test_db
        finally:
            pass

    fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_get_db_for_test

    with TestClient(fastapi_app) as c:
        # Set authentication header for single-user mode
        c.headers["X-API-KEY"] = TestConfig.TEST_API_KEY

        # First make a GET request to obtain CSRF token
        response = c.get("/api/v1/health")  # Or any GET endpoint
        csrf_token = response.cookies.get("csrf_token")

        # Add CSRF token to client's default headers if obtained
        if csrf_token:
            c.headers["X-CSRF-Token"] = csrf_token
            c.cookies["csrf_token"] = csrf_token

        yield c

    fastapi_app.dependency_overrides.clear()


def create_dummy_image_base64(width=10, height=10, image_format="PNG") -> str:
    img = PILImage.new("RGB", (width, height), color="red")
    buffered = BytesIO()
    img.save(buffered, format=image_format)
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


def create_sample_character_payload(name_suffix: str = "", **overrides) -> Dict[str, Any]:
    payload = {
        "name": f"Test Char API {name_suffix}{uuid.uuid4().hex[:6]}",
        "description": "A character for API testing.",
        "first_message": "Hello from API Test!",
        "tags": ["api", "test"],
        "image_base64": create_dummy_image_base64(),
    }
    payload.update(overrides)
    return payload


# --- Hypothesis Strategies for PBT ---
st_valid_api_text = st.text(
    min_size=1,
    max_size=50,
    alphabet=st.characters(min_codepoint=32, max_codepoint=126, blacklist_categories=("Cc", "Cs")),
)
st_optional_api_text = st.one_of(st.none(), st_valid_api_text)  # Allow empty string as well, if Pydantic model permits
st_api_json_list_or_str = st.one_of(
    st.none(),
    st.lists(st_valid_api_text, max_size=2, unique=True),
    st.lists(st_valid_api_text, max_size=2, unique=True).map(json.dumps),
)
st_api_json_dict_or_str = st.one_of(
    st.none(),
    st.dictionaries(
        st_valid_api_text, st.one_of(st_valid_api_text, st.integers(0, 100), st.booleans(), st.none()), max_size=2
    ),
    st.dictionaries(st_valid_api_text, st_valid_api_text, max_size=2).map(json.dumps),
)
st_base64_image_str = st.one_of(st.none(), st.just(create_dummy_image_base64()))


def _normalize_expected_tags_for_api(value: Any) -> list[str]:
    """Mirror API/DB tag normalization used by character create/update paths."""
    if value is None:
        return []

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [value]
        values = parsed if isinstance(parsed, list) else [value]
    elif isinstance(value, list):
        values = value
    else:
        values = [value]

    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        tag = str(item).strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
    return normalized


def st_character_create_payload_pbt():
    return st.builds(
        dict,
        name=st_valid_api_text,  # Name is mandatory for create
        description=st_optional_api_text,
        personality=st_optional_api_text,
        scenario=st_optional_api_text,
        system_prompt=st_optional_api_text,
        post_history_instructions=st_optional_api_text,
        first_message=st_optional_api_text,
        message_example=st_optional_api_text,
        creator_notes=st_optional_api_text,
        alternate_greetings=st_api_json_list_or_str,
        tags=st_api_json_list_or_str,
        creator=st_optional_api_text,
        character_version=st_optional_api_text,
        extensions=st_api_json_dict_or_str,
        image_base64=st_base64_image_str,
    ).filter(lambda x: x["name"] is not None and x["name"].strip() != "")


# Revised strategy for update payload to be less sparse
def st_character_update_payload_pbt():
    keys = [  # All fields that can be part of an update payload
        "name",
        "description",
        "personality",
        "scenario",
        "system_prompt",
        "post_history_instructions",
        "first_message",
        "message_example",
        "creator_notes",
        "alternate_greetings",
        "tags",
        "creator",
        "character_version",
        "extensions",
        "image_base64",
    ]
    # Strategies for values that are definitely not None (when chosen to be the "concrete" one)
    concrete_value_strategies = {
        "name": st_valid_api_text,
        "description": st_valid_api_text,
        # ... (fill for all keys, ensuring they don't generate None)
        "tags": st.lists(st_valid_api_text, min_size=1, max_size=2, unique=True),
        "image_base64": st.just(create_dummy_image_base64()),
    }
    # Strategies that can produce None (for other fields not chosen as concrete)
    optional_value_strategies = {
        "name": st_optional_api_text,
        "description": st_optional_api_text,
        # ... (fill for all keys)
        "tags": st_api_json_list_or_str,  # Can be None
        "image_base64": st_base64_image_str,  # Can be None
    }
    # Ensure all keys are in both strategy dicts for simplicity in lookup
    for k in keys:
        if k not in concrete_value_strategies:
            concrete_value_strategies[k] = st_valid_api_text  # Default concrete
        if k not in optional_value_strategies:
            optional_value_strategies[k] = st_optional_api_text  # Default optional

    @st.composite
    def at_least_one_concrete_field_payload(draw):
        # Draw a subset of keys to include in the update payload, must include at least one
        num_fields_to_update = draw(st.integers(min_value=1, max_value=len(keys)))
        selected_keys = draw(
            st.lists(st.sampled_from(keys), min_size=num_fields_to_update, max_size=num_fields_to_update, unique=True)
        )

        # From these selected keys, pick one to ensure it gets a concrete (non-None) value
        key_for_concrete_value = draw(st.sampled_from(selected_keys))

        payload = {}
        has_at_least_one_non_none = False
        for k_sel in selected_keys:
            if k_sel == key_for_concrete_value:
                val = draw(concrete_value_strategies[k_sel])
            else:
                val = draw(optional_value_strategies[k_sel])
            payload[k_sel] = val
            if val is not None:
                has_at_least_one_non_none = True

        assume(has_at_least_one_non_none)  # Ensure the generated payload is not all Nones
        return payload

    return at_least_one_concrete_field_payload()


# ================================= UNIT TESTS =================================
# Patch target should be where the function is LOOKED UP in the module under test.
# If characters.py (the API endpoint file) imports `create_new_character_from_data` from char_lib,
# then the patch target is 'tldw_Server_API.app.api.v1.endpoints.characters.create_new_character_from_data'

UNIT_TEST_PATCH_PREFIX = "tldw_Server_API.app.api.v1.endpoints.characters_endpoint"


@patch(f"{UNIT_TEST_PATCH_PREFIX}.create_new_character_from_data")
@patch(f"{UNIT_TEST_PATCH_PREFIX}.get_character_details")
def test_unit_create_character_success(mock_get_details: MagicMock, mock_create: MagicMock, client: TestClient):
    mock_create.return_value = 1
    mock_char_data = {
        "id": 1,
        "name": "Unit Test Char",
        "version": 1,
        "description": "Desc",
        "image": b"dummy",
        "alternate_greetings": ["Hi"],
        "tags": ["test"],
        "extensions": {"key": "val"},
    }
    mock_get_details.return_value = mock_char_data

    payload = {"name": "Unit Test Char", "description": "Desc"}
    response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=payload)

    assert response.status_code == 201, response.text
    data = response.json()
    assert data["id"] == 1
    assert data["name"] == "Unit Test Char"
    assert data["image_present"] is True
    mock_create.assert_called_once()
    assert mock_create.call_args[0][1]["name"] == payload["name"]  # db, character_payload
    mock_get_details.assert_called_once_with(mock_create.call_args[0][0], 1)


@patch(f"{UNIT_TEST_PATCH_PREFIX}.create_new_character_from_data")
def test_unit_create_character_conflict(mock_create: MagicMock, client: TestClient):
    mock_create.side_effect = ConflictError("Character with name 'Exists' already exists.")
    payload = {"name": "Exists", "description": "Desc"}
    response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=payload)
    assert response.status_code == 409, response.text
    assert "Character with name 'Exists' already exists." in response.json()["detail"]


@patch(f"{UNIT_TEST_PATCH_PREFIX}.create_new_character_from_data")
def test_unit_create_character_input_error_from_lib(mock_create: MagicMock, client: TestClient):
    # Test case where Pydantic validation passes, but the library function raises InputError
    mock_create.side_effect = InputError("Lib-level Invalid input for character.")
    payload = {
        "name": "ValidPydanticName",
        "description": "Desc",
        "image_base64": "invalid-b64!",
    }  # image_base64 will be caught by lib

    # Simulate that Pydantic validation for 'name' passes
    # The call to create_new_character_from_data inside the endpoint will raise InputError
    response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=payload)
    assert response.status_code == 400, response.text
    assert "Lib-level Invalid input for character." in response.json()["detail"]


def test_unit_create_character_pydantic_error(client: TestClient):  # No mock needed for Pydantic
    payload = {"name": "", "description": "Desc"}  # Pydantic CharacterCreate requires non-empty name
    response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=payload)
    assert response.status_code == 422  # Unprocessable Entity for Pydantic validation
    assert "String should have at least 1 character" in response.text


@patch(f"{UNIT_TEST_PATCH_PREFIX}.get_character_details")
def test_unit_get_character_success(mock_get_details: MagicMock, client: TestClient):
    mock_char_data = {"id": 1, "name": "Fetched Char", "version": 1, "image": None}
    mock_get_details.return_value = mock_char_data
    response = client.get(f"{CHARACTERS_ENDPOINT_PREFIX}/1")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["name"] == "Fetched Char"
    assert data["image_present"] is False
    mock_get_details.assert_called_once_with(mock_get_details.call_args[0][0], 1)


@patch(f"{UNIT_TEST_PATCH_PREFIX}.get_character_details")
def test_unit_get_character_not_found(mock_get_details: MagicMock, client: TestClient):
    mock_get_details.return_value = None
    response = client.get(f"{CHARACTERS_ENDPOINT_PREFIX}/999")
    assert response.status_code == 404, response.text
    assert "not found" in response.json()["detail"]


@patch(f"{UNIT_TEST_PATCH_PREFIX}.update_existing_character_details")
@patch(f"{UNIT_TEST_PATCH_PREFIX}.get_character_details")
def test_unit_update_character_success(mock_get_details: MagicMock, mock_update: MagicMock, client: TestClient):
    mock_get_details.side_effect = [
        {"id": 1, "name": "Old Name", "version": 1, "image": None},
        {"id": 1, "name": "New Name", "version": 2, "image": b"newimg"},
    ]
    mock_update.return_value = True

    payload = {"name": "New Name", "image_base64": create_dummy_image_base64()}
    response = client.put(f"{CHARACTERS_ENDPOINT_PREFIX}/1?expected_version=1", json=payload)

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["name"] == "New Name"
    assert data["version"] == 2
    assert data["image_present"] is True
    mock_update.assert_called_once()
    assert mock_update.call_args[0][2]["name"] == "New Name"  # update_payload dict
    assert "image_base64" in mock_update.call_args[0][2]  # Check if it was passed to lib as base64
    assert mock_update.call_args[0][3] == 1  # expected_version


@patch(f"{UNIT_TEST_PATCH_PREFIX}.get_character_details")  # Only get_character_details is called before version check
def test_unit_update_character_version_mismatch(mock_get_details: MagicMock, client: TestClient):
    mock_get_details.return_value = {"id": 1, "name": "Old Name", "version": 2}
    payload = {"description": "New Desc"}
    response = client.put(f"{CHARACTERS_ENDPOINT_PREFIX}/1?expected_version=1", json=payload)
    assert response.status_code == 409, response.text
    assert "Version mismatch" in response.json()["detail"]


@patch(f"{UNIT_TEST_PATCH_PREFIX}.delete_character_from_db")
@patch(f"{UNIT_TEST_PATCH_PREFIX}.get_character_details")
def test_unit_delete_character_success(mock_get_details: MagicMock, mock_delete: MagicMock, client: TestClient):
    mock_get_details.return_value = {"id": 1, "name": "ToDelete", "version": 1}
    mock_delete.return_value = True
    response = client.delete(f"{CHARACTERS_ENDPOINT_PREFIX}/1?expected_version=1")
    assert response.status_code == 200, response.text
    assert response.json()["message"] == "Character 'ToDelete' (ID: 1) soft-deleted."
    mock_delete.assert_called_once_with(mock_delete.call_args[0][0], 1, 1)


@patch(f"{UNIT_TEST_PATCH_PREFIX}.create_new_character_from_data")
@patch(f"{UNIT_TEST_PATCH_PREFIX}.get_character_details")
def test_unit_create_character_success(mock_get_details: MagicMock, mock_create: MagicMock, client: TestClient):
    mock_create.return_value = 1  # This is char_id
    # This mock_char_data is what get_character_details returns from DB
    mock_char_data_from_db = {
        "id": 1,
        "name": "Unit Test Char",
        "version": 1,
        "description": "Desc",
        "image": b"dummy_image_bytes",  # mock image bytes from DB
        "alternate_greetings": ["Hi"],
        "tags": ["test"],
        "extensions": {"key": "val"},
        # Add all other fields expected by _convert_db_char_to_response_model / CharacterResponse
        "personality": None,
        "scenario": None,
        "system_prompt": None,
        "post_history_instructions": None,
        "first_message": None,
        "message_example": None,
        "creator_notes": None,
        "creator": None,
        "character_version": None,
        "created_at": "2023-01-01T00:00:00",
        "updated_at": "2023-01-01T00:00:00",
        "deleted": 0,
    }
    mock_get_details.return_value = mock_char_data_from_db

    payload = {"name": "Unit Test Char", "description": "Desc"}  # API input
    response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=payload)

    assert response.status_code == 201, response.text
    data = response.json()  # This is CharacterResponse
    assert data["id"] == 1
    assert data["name"] == "Unit Test Char"
    assert data["image_present"] is True  # Because mock_char_data_from_db["image"] is present
    mock_create.assert_called_once()
    # mock_create is called with (db_obj, character_payload_dict)
    assert mock_create.call_args[0][1]["name"] == payload["name"]
    # mock_get_details is called with (db_obj, char_id)
    mock_get_details.assert_called_once_with(mock_create.call_args[0][0], 1)


@patch(f"{UNIT_TEST_PATCH_PREFIX}.WorldBookService")
@patch(f"{UNIT_TEST_PATCH_PREFIX}.get_character_details")
def test_unit_attach_world_book_permission_denied(
    mock_get_details: MagicMock,
    mock_world_book_service: MagicMock,
    client: TestClient,
):
    mock_get_details.return_value = {"id": 1, "name": "Permission Char"}
    service = mock_world_book_service.return_value
    service.get_world_book.return_value = {"id": 9, "name": "Restricted Book"}
    service.attach_to_character.side_effect = CharactersRAGDBError("permission denied")

    response = client.post(
        f"{CHARACTERS_ENDPOINT_PREFIX}/1/world-books",
        json={"world_book_id": 9, "enabled": True, "priority": 0},
    )

    assert response.status_code == 403, response.text
    assert "Insufficient permissions" in response.json()["detail"]


@patch(f"{UNIT_TEST_PATCH_PREFIX}.WorldBookService")
def test_unit_detach_world_book_permission_denied(
    mock_world_book_service: MagicMock,
    client: TestClient,
):
    service = mock_world_book_service.return_value
    service.detach_from_character.side_effect = CharactersRAGDBError("forbidden")

    response = client.delete(f"{CHARACTERS_ENDPOINT_PREFIX}/1/world-books/9")

    assert response.status_code == 403, response.text
    assert "Insufficient permissions" in response.json()["detail"]


@patch(f"{UNIT_TEST_PATCH_PREFIX}.WorldBookService")
@patch(f"{UNIT_TEST_PATCH_PREFIX}.get_character_details")
def test_unit_list_character_world_books_permission_denied(
    mock_get_details: MagicMock,
    mock_world_book_service: MagicMock,
    client: TestClient,
):
    mock_get_details.return_value = {"id": 1, "name": "Permission Char"}
    service = mock_world_book_service.return_value
    service.get_character_world_books.side_effect = CharactersRAGDBError("insufficient privilege")

    response = client.get(f"{CHARACTERS_ENDPOINT_PREFIX}/1/world-books")

    assert response.status_code == 403, response.text
    assert "Insufficient permissions" in response.json()["detail"]


# ============================= INTEGRATION TESTS ==============================


class TestCharacterAPIIntegration:

    def test_create_character_integration(self, client: TestClient, test_db: CharactersRAGDB):
        payload = create_sample_character_payload("Integration")
        response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=payload)
        assert response.status_code == 201, response.text
        data = response.json()
        char_id = data["id"]
        assert data["name"] == payload["name"]
        assert data["description"] == payload["description"]
        assert data["version"] == 1
        assert data["image_present"] is True

        db_char = test_db.get_character_card_by_id(char_id)
        assert db_char is not None
        assert db_char["name"] == payload["name"]
        assert db_char["image"] is not None

    def test_create_character_conflict_integration(self, client: TestClient):
        payload = create_sample_character_payload("Conflict")
        client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=payload)
        response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=payload)
        assert response.status_code == 409, response.text
        assert "already exists" in response.json()["detail"]

    def test_create_character_accepts_name_at_max_length_integration(self, client: TestClient):
        unique_prefix = f"MaxLen_{uuid.uuid4().hex}_"
        max_length_name = (unique_prefix + ("N" * MAX_NAME_LENGTH))[:MAX_NAME_LENGTH]
        payload = create_sample_character_payload(name=max_length_name)

        response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=payload)

        assert response.status_code == 201, response.text
        data = response.json()
        assert data["name"] == max_length_name
        assert len(data["name"]) == MAX_NAME_LENGTH

    def test_create_character_rejects_name_over_max_length_integration(self, client: TestClient):
        unique_prefix = f"TooLong_{uuid.uuid4().hex}_"
        over_limit_name = (unique_prefix + ("N" * (MAX_NAME_LENGTH + 1)))[: MAX_NAME_LENGTH + 1]
        payload = create_sample_character_payload(name=over_limit_name)

        response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=payload)

        assert response.status_code == 422, response.text
        detail = response.json().get("detail", [])
        assert isinstance(detail, list)
        assert any("name" in [str(part) for part in (item.get("loc") or [])] for item in detail)
        assert any(
            "at most" in str(item.get("msg", "")).lower()
            and str(MAX_NAME_LENGTH) in str(item.get("msg", ""))
            for item in detail
        )

    def test_create_character_bad_image_data_integration(self, client: TestClient):
        payload = create_sample_character_payload("BadImage", image_base64="not_a_valid_base64_string")
        response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=payload)
        assert response.status_code == 400, response.text
        assert "Invalid image_base64 data" in response.json()["detail"]

    def test_get_character_integration(self, client: TestClient, test_db: CharactersRAGDB):
        payload = create_sample_character_payload("GetMe")
        create_response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=payload)
        assert create_response.status_code == 201, create_response.text
        char_id = create_response.json()["id"]

        response = client.get(f"{CHARACTERS_ENDPOINT_PREFIX}/{char_id}")
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["name"] == payload["name"]
        assert data["image_base64"] is not None

    def test_get_character_not_found_integration(self, client: TestClient):
        response = client.get(f"{CHARACTERS_ENDPOINT_PREFIX}/99999")
        assert response.status_code == 404, response.text

    def test_list_characters_integration(self, client: TestClient, test_db: CharactersRAGDB):
        # Clear previous test data for cleaner list assertion, or ensure unique names
        # For robust testing against existing data, ensure very unique names or count before/after
        initial_chars_response = client.get(f"{CHARACTERS_ENDPOINT_PREFIX}/")
        assume(initial_chars_response.status_code == 200)
        initial_count = len(initial_chars_response.json())

        name_a = f"List_Integ_A_{uuid.uuid4().hex[:6]}"
        name_b = f"List_Integ_B_{uuid.uuid4().hex[:6]}"
        res_a = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=create_sample_character_payload(name=name_a))
        assert res_a.status_code == 201
        res_b = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=create_sample_character_payload(name=name_b))
        assert res_b.status_code == 201

        response = client.get(f"{CHARACTERS_ENDPOINT_PREFIX}/")
        assert response.status_code == 200, response.text
        data = response.json()

        # Check count if DB is guaranteed empty before these two additions
        # assert len(data) == initial_count + 2

        # Filter for names created in this test to be robust against other data
        current_test_names = {name_a, name_b}
        found_names = {item["name"] for item in data if item["name"] in current_test_names}
        assert len(found_names) == 2
        assert name_a in found_names
        assert name_b in found_names

    def test_query_characters_integration(self, client: TestClient, test_db: CharactersRAGDB):
        creator = f"Creator_{uuid.uuid4().hex[:6]}"
        name_a = f"QueryA_{uuid.uuid4().hex[:6]}"
        name_b = f"QueryB_{uuid.uuid4().hex[:6]}"
        name_c = f"QueryC_{uuid.uuid4().hex[:6]}"

        resp_a = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/",
            json=create_sample_character_payload(name=name_a, creator=creator)
        )
        resp_b = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/",
            json=create_sample_character_payload(
                name=name_b,
                creator=creator,
                extensions={"tldw": {"favorite": True}},
            )
        )
        resp_c = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/",
            json=create_sample_character_payload(name=name_c, creator="other")
        )
        assert resp_a.status_code == 201, resp_a.text
        assert resp_b.status_code == 201, resp_b.text
        assert resp_c.status_code == 201, resp_c.text

        # Create one conversation to validate has_conversations filtering.
        char_with_conversation = int(resp_a.json()["id"])
        conv_id = test_db.add_conversation(
            {"character_id": char_with_conversation, "title": f"Conv {uuid.uuid4().hex[:4]}"}
        )
        assert conv_id is not None

        page_response = client.get(
            f"{CHARACTERS_ENDPOINT_PREFIX}/query?page=1&page_size=2&sort_by=name&sort_order=asc"
        )
        assert page_response.status_code == 200, page_response.text
        page_data = page_response.json()
        assert "items" in page_data
        assert "total" in page_data
        assert page_data["page"] == 1
        assert page_data["page_size"] == 2
        assert isinstance(page_data["has_more"], bool)
        assert len(page_data["items"]) <= 2
        assert page_data["total"] >= 3

        creator_response = client.get(
            f"{CHARACTERS_ENDPOINT_PREFIX}/query?page=1&page_size=20&creator={creator}"
        )
        assert creator_response.status_code == 200, creator_response.text
        creator_items = creator_response.json()["items"]
        assert len(creator_items) >= 2
        assert all((item.get("creator") or "").lower() == creator.lower() for item in creator_items)

        favorite_only_response = client.get(
            f"{CHARACTERS_ENDPOINT_PREFIX}/query?page=1&page_size=20&favorite_only=true"
        )
        assert favorite_only_response.status_code == 200, favorite_only_response.text
        favorite_ids = {
            int(item["id"]) for item in favorite_only_response.json()["items"] if "id" in item
        }
        assert int(resp_b.json()["id"]) in favorite_ids
        assert int(resp_a.json()["id"]) not in favorite_ids

        has_conv_response = client.get(
            f"{CHARACTERS_ENDPOINT_PREFIX}/query?page=1&page_size=20&has_conversations=true"
        )
        assert has_conv_response.status_code == 200, has_conv_response.text
        has_conv_ids = {
            int(item["id"]) for item in has_conv_response.json()["items"] if "id" in item
        }
        assert char_with_conversation in has_conv_ids

        no_conv_response = client.get(
            f"{CHARACTERS_ENDPOINT_PREFIX}/query?page=1&page_size=20&has_conversations=false"
        )
        assert no_conv_response.status_code == 200, no_conv_response.text
        no_conv_ids = {
            int(item["id"]) for item in no_conv_response.json()["items"] if "id" in item
        }
        assert char_with_conversation not in no_conv_ids

        char_deleted_id = int(resp_c.json()["id"])
        char_deleted_version = int(resp_c.json()["version"])
        delete_response = client.delete(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{char_deleted_id}?expected_version={char_deleted_version}"
        )
        assert delete_response.status_code == 200, delete_response.text

        deleted_only_response = client.get(
            f"{CHARACTERS_ENDPOINT_PREFIX}/query?page=1&page_size=20&deleted_only=true"
        )
        assert deleted_only_response.status_code == 200, deleted_only_response.text
        deleted_only_ids = {
            int(item["id"])
            for item in deleted_only_response.json()["items"]
            if "id" in item
        }
        assert char_deleted_id in deleted_only_ids
        assert char_with_conversation not in deleted_only_ids

    def test_query_characters_image_payload_controls(self, client: TestClient):
        creator = f"CreatorImagePayload_{uuid.uuid4().hex[:6]}"
        created = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/",
            json=create_sample_character_payload(
                name=f"ImagePayload_{uuid.uuid4().hex[:6]}",
                creator=creator,
            ),
        )
        assert created.status_code == 201, created.text
        created_id = int(created.json()["id"])

        default_response = client.get(
            f"{CHARACTERS_ENDPOINT_PREFIX}/query?page=1&page_size=20&creator={creator}"
        )
        assert default_response.status_code == 200, default_response.text
        default_items = default_response.json()["items"]
        default_item = next(
            (item for item in default_items if int(item.get("id", -1)) == created_id),
            None,
        )
        assert default_item is not None
        assert default_item.get("image_base64") is None
        assert default_item.get("image_present") is True

        include_response = client.get(
            f"{CHARACTERS_ENDPOINT_PREFIX}/query?page=1&page_size=20&creator={creator}&include_image_base64=true"
        )
        assert include_response.status_code == 200, include_response.text
        include_items = include_response.json()["items"]
        include_item = next(
            (item for item in include_items if int(item.get("id", -1)) == created_id),
            None,
        )
        assert include_item is not None
        assert isinstance(include_item.get("image_base64"), str)
        assert len(include_item["image_base64"]) > 0
        assert include_item.get("image_present") is True

    def test_query_characters_reserved_folder_tag_filters_integration(self, client: TestClient):
        folder_a = f"{CHARACTER_FOLDER_TAG_PREFIX}alpha_{uuid.uuid4().hex[:6]}"
        folder_b = f"{CHARACTER_FOLDER_TAG_PREFIX}beta_{uuid.uuid4().hex[:6]}"
        shared_tag = f"shared_{uuid.uuid4().hex[:6]}"

        response_a = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/",
            json=create_sample_character_payload(
                name=f"FolderA_{uuid.uuid4().hex[:6]}",
                tags=[shared_tag, folder_a],
            ),
        )
        response_b = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/",
            json=create_sample_character_payload(
                name=f"FolderB_{uuid.uuid4().hex[:6]}",
                tags=[folder_a],
            ),
        )
        response_c = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/",
            json=create_sample_character_payload(
                name=f"FolderC_{uuid.uuid4().hex[:6]}",
                tags=[shared_tag, folder_b],
            ),
        )
        assert response_a.status_code == 201, response_a.text
        assert response_b.status_code == 201, response_b.text
        assert response_c.status_code == 201, response_c.text

        folder_only_response = client.get(
            f"{CHARACTERS_ENDPOINT_PREFIX}/query",
            params=[
                ("page", 1),
                ("page_size", 50),
                ("tags", folder_a),
            ],
        )
        assert folder_only_response.status_code == 200, folder_only_response.text
        folder_only_ids = {
            int(item["id"])
            for item in folder_only_response.json()["items"]
            if "id" in item
        }
        assert int(response_a.json()["id"]) in folder_only_ids
        assert int(response_b.json()["id"]) in folder_only_ids
        assert int(response_c.json()["id"]) not in folder_only_ids

        match_all_response = client.get(
            f"{CHARACTERS_ENDPOINT_PREFIX}/query",
            params=[
                ("page", 1),
                ("page_size", 50),
                ("tags", folder_a),
                ("tags", shared_tag),
                ("match_all_tags", "true"),
            ],
        )
        assert match_all_response.status_code == 200, match_all_response.text
        match_all_ids = {
            int(item["id"])
            for item in match_all_response.json()["items"]
            if "id" in item
        }
        assert int(response_a.json()["id"]) in match_all_ids
        assert int(response_b.json()["id"]) not in match_all_ids
        assert int(response_c.json()["id"]) not in match_all_ids

    def test_create_and_update_enforce_single_folder_token_integration(self, client: TestClient):
        folder_1 = f"{CHARACTER_FOLDER_TAG_PREFIX}f1_{uuid.uuid4().hex[:6]}"
        folder_2 = f"{CHARACTER_FOLDER_TAG_PREFIX}f2_{uuid.uuid4().hex[:6]}"
        folder_3 = f"{CHARACTER_FOLDER_TAG_PREFIX}f3_{uuid.uuid4().hex[:6]}"
        folder_4 = f"{CHARACTER_FOLDER_TAG_PREFIX}f4_{uuid.uuid4().hex[:6]}"

        created = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/",
            json=create_sample_character_payload(
                name=f"FolderNormalize_{uuid.uuid4().hex[:6]}",
                tags=["alpha", folder_1, folder_2, "beta"],
            ),
        )
        assert created.status_code == 201, created.text
        created_data = created.json()
        created_tags = created_data.get("tags") or []
        folder_tags_after_create = [tag for tag in created_tags if str(tag).startswith(CHARACTER_FOLDER_TAG_PREFIX)]
        assert folder_tags_after_create == [folder_2]
        assert "alpha" in created_tags
        assert "beta" in created_tags

        update_response = client.put(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{created_data['id']}?expected_version={created_data['version']}",
            json={"tags": ["gamma", folder_3, folder_4]},
        )
        assert update_response.status_code == 200, update_response.text
        updated_data = update_response.json()
        updated_tags = updated_data.get("tags") or []
        folder_tags_after_update = [tag for tag in updated_tags if str(tag).startswith(CHARACTER_FOLDER_TAG_PREFIX)]
        assert folder_tags_after_update == [folder_4]
        assert "gamma" in updated_tags
        assert folder_3 not in updated_tags

    def test_manage_character_tags_operations_integration(self, client: TestClient, test_db: CharactersRAGDB):
        char_a = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/",
            json=create_sample_character_payload(
                name=f"TagOpsA_{uuid.uuid4().hex[:6]}",
                tags=["legacy", "shared"],
            ),
        )
        char_b = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/",
            json=create_sample_character_payload(
                name=f"TagOpsB_{uuid.uuid4().hex[:6]}",
                tags=["legacy"],
            ),
        )
        char_c = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/",
            json=create_sample_character_payload(
                name=f"TagOpsC_{uuid.uuid4().hex[:6]}",
                tags=["shared"],
            ),
        )
        assert char_a.status_code == 201, char_a.text
        assert char_b.status_code == 201, char_b.text
        assert char_c.status_code == 201, char_c.text

        char_a_id = int(char_a.json()["id"])
        char_b_id = int(char_b.json()["id"])
        char_c_id = int(char_c.json()["id"])

        rename_response = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/tags/operations",
            json={
                "operation": "rename",
                "source_tag": "legacy",
                "target_tag": "modern",
            },
        )
        assert rename_response.status_code == 200, rename_response.text
        rename_data = rename_response.json()
        assert rename_data["updated_count"] == 2
        assert set(rename_data["updated_character_ids"]) == {char_a_id, char_b_id}

        renamed_a = test_db.get_character_card_by_id(char_a_id)
        renamed_b = test_db.get_character_card_by_id(char_b_id)
        assert renamed_a is not None
        assert renamed_b is not None
        assert renamed_a["tags"] == ["modern", "shared"]
        assert renamed_b["tags"] == ["modern"]

        merge_response = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/tags/operations",
            json={
                "operation": "merge",
                "source_tag": "modern",
                "target_tag": "shared",
            },
        )
        assert merge_response.status_code == 200, merge_response.text
        merge_data = merge_response.json()
        assert merge_data["updated_count"] == 2
        assert set(merge_data["updated_character_ids"]) == {char_a_id, char_b_id}

        merged_a = test_db.get_character_card_by_id(char_a_id)
        merged_b = test_db.get_character_card_by_id(char_b_id)
        assert merged_a is not None
        assert merged_b is not None
        assert merged_a["tags"] == ["shared"]
        assert merged_b["tags"] == ["shared"]

        delete_response = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/tags/operations",
            json={
                "operation": "delete",
                "source_tag": "shared",
            },
        )
        assert delete_response.status_code == 200, delete_response.text
        delete_data = delete_response.json()
        assert delete_data["updated_count"] >= 3
        assert char_c_id in delete_data["updated_character_ids"]

        deleted_a = test_db.get_character_card_by_id(char_a_id)
        deleted_b = test_db.get_character_card_by_id(char_b_id)
        deleted_c = test_db.get_character_card_by_id(char_c_id)
        assert deleted_a is not None
        assert deleted_b is not None
        assert deleted_c is not None
        assert deleted_a["tags"] == []
        assert deleted_b["tags"] == []
        assert deleted_c["tags"] == []

    def test_manage_character_tags_requires_target_for_rename_integration(self, client: TestClient):
        response = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/tags/operations",
            json={
                "operation": "rename",
                "source_tag": "legacy",
            },
        )
        assert response.status_code == 422, response.text

    def test_create_character_strips_whitespace_from_tags_integration(
        self, client: TestClient
    ):
        response = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/",
            json=create_sample_character_payload(
                name=f"TagTrim_{uuid.uuid4().hex[:6]}",
                tags=["  alpha  "],
            ),
        )

        assert response.status_code == 201, response.text
        assert response.json()["tags"] == ["alpha"]

    def test_character_world_book_attachment_lifecycle_integration(
        self, client: TestClient
    ):
        character_response = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/",
            json=create_sample_character_payload(name=f"WBChar_{uuid.uuid4().hex[:6]}"),
        )
        assert character_response.status_code == 201, character_response.text
        character_id = int(character_response.json()["id"])

        world_book_response = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/world-books",
            json={
                "name": f"WB_{uuid.uuid4().hex[:6]}",
                "description": "Attachment integration test",
                "scan_depth": 3,
                "token_budget": 500,
                "recursive_scanning": False,
                "enabled": True,
            },
        )
        assert world_book_response.status_code == 201, world_book_response.text
        world_book_id = int(world_book_response.json()["id"])

        attach_response = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{character_id}/world-books",
            json={"world_book_id": world_book_id, "enabled": True, "priority": 2},
        )
        assert attach_response.status_code == 200, attach_response.text
        attach_data = attach_response.json()
        assert int(attach_data["world_book_id"]) == world_book_id
        assert attach_data["attachment_enabled"] is True
        assert int(attach_data["attachment_priority"]) == 2

        list_response = client.get(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{character_id}/world-books"
        )
        assert list_response.status_code == 200, list_response.text
        attached_ids = {
            int(item["world_book_id"]) for item in list_response.json() if "world_book_id" in item
        }
        assert world_book_id in attached_ids

        detach_response = client.delete(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{character_id}/world-books/{world_book_id}"
        )
        assert detach_response.status_code == 200, detach_response.text
        detach_payload = detach_response.json()
        assert int(detach_payload["world_book_id"]) == world_book_id
        assert int(detach_payload["detached_from_character_id"]) == character_id
        assert "character_id" not in detach_payload

        detached_list_response = client.get(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{character_id}/world-books"
        )
        assert detached_list_response.status_code == 200, detached_list_response.text
        detached_ids = {
            int(item["world_book_id"])
            for item in detached_list_response.json()
            if "world_book_id" in item
        }
        assert world_book_id not in detached_ids

    def test_world_book_delete_and_entry_delete_report_explicit_resource_ids(
        self, client: TestClient
    ):
        world_book_response = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/world-books",
            json={
                "name": f"WBDelete_{uuid.uuid4().hex[:6]}",
                "description": "Delete response integration test",
                "scan_depth": 3,
                "token_budget": 500,
                "recursive_scanning": False,
                "enabled": True,
            },
        )
        assert world_book_response.status_code == 201, world_book_response.text
        world_book_id = int(world_book_response.json()["id"])

        entry_response = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/world-books/{world_book_id}/entries",
            json={
                "keywords": ["delete-response-keyword"],
                "content": "Delete response entry content",
                "priority": 1,
                "enabled": True,
                "case_sensitive": False,
                "regex_match": False,
                "whole_word_match": True,
                "metadata": {"source": "integration"},
            },
        )
        assert entry_response.status_code == 201, entry_response.text
        entry_id = int(entry_response.json()["id"])

        entry_delete_response = client.delete(
            f"{CHARACTERS_ENDPOINT_PREFIX}/world-books/entries/{entry_id}"
        )
        assert entry_delete_response.status_code == 200, entry_delete_response.text
        entry_delete_payload = entry_delete_response.json()
        assert int(entry_delete_payload["entry_id"]) == entry_id
        assert "character_id" not in entry_delete_payload

        world_book_delete_response = client.delete(
            f"{CHARACTERS_ENDPOINT_PREFIX}/world-books/{world_book_id}"
        )
        assert world_book_delete_response.status_code == 200, world_book_delete_response.text
        world_book_delete_payload = world_book_delete_response.json()
        assert int(world_book_delete_payload["world_book_id"]) == world_book_id
        assert "character_id" not in world_book_delete_payload

    def test_character_world_book_attachment_missing_references_integration(
        self, client: TestClient
    ):
        character_response = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/",
            json=create_sample_character_payload(name=f"MissingWB_{uuid.uuid4().hex[:6]}"),
        )
        assert character_response.status_code == 201, character_response.text
        character_id = int(character_response.json()["id"])

        missing_world_book_id = 999_999_991
        attach_missing_book_response = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{character_id}/world-books",
            json={"world_book_id": missing_world_book_id, "enabled": True, "priority": 0},
        )
        assert attach_missing_book_response.status_code == 404
        assert "World book with ID" in attach_missing_book_response.json().get("detail", "")

        missing_character_id = 999_999_992
        list_missing_character_response = client.get(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{missing_character_id}/world-books"
        )
        assert list_missing_character_response.status_code == 404
        assert "Character with ID" in list_missing_character_response.json().get("detail", "")

    def test_update_character_integration(self, client: TestClient, test_db: CharactersRAGDB):
        create_payload = create_sample_character_payload("UpdateBase")
        create_response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=create_payload)
        assert create_response.status_code == 201, create_response.text
        create_resp_json = create_response.json()

        char_id = create_resp_json["id"]
        original_version = create_resp_json["version"]

        update_payload = {
            "name": "Updated Character Name",
            "description": "Updated description.",
            "tags": ["newtag"],
            "image_base64": None,
        }
        response = client.put(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{char_id}?expected_version={original_version}", json=update_payload
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["name"] == "Updated Character Name"
        assert data["description"] == "Updated description."
        assert data["tags"] == ["newtag"]
        assert data["version"] == original_version + 1
        assert data["image_present"] is False

        db_char = test_db.get_character_card_by_id(char_id)
        assert db_char is not None
        assert db_char["name"] == "Updated Character Name"
        assert db_char["image"] is None

    def test_get_character_versions_integration(self, client: TestClient):
        create_payload = create_sample_character_payload(
            "VersionList",
            description="Version 1 description",
        )
        create_response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=create_payload)
        assert create_response.status_code == 201, create_response.text
        created = create_response.json()
        char_id = int(created["id"])

        update_one = client.put(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{char_id}?expected_version={created['version']}",
            json={"description": "Version 2 description"},
        )
        assert update_one.status_code == 200, update_one.text
        updated_one = update_one.json()

        update_two = client.put(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{char_id}?expected_version={updated_one['version']}",
            json={"description": "Version 3 description"},
        )
        assert update_two.status_code == 200, update_two.text

        versions_response = client.get(f"{CHARACTERS_ENDPOINT_PREFIX}/{char_id}/versions?limit=10")
        assert versions_response.status_code == 200, versions_response.text
        payload = versions_response.json()
        assert payload["total"] >= 3

        items = payload["items"]
        versions = [int(item["version"]) for item in items]
        assert versions == sorted(versions, reverse=True)
        assert {1, 2, 3}.issubset(set(versions))
        operations = {str(item.get("operation", "")) for item in items}
        assert "create" in operations
        assert "update" in operations

    def test_get_character_version_diff_integration(self, client: TestClient):
        create_payload = create_sample_character_payload(
            "VersionDiff",
            description="Original description",
            tags=["alpha"],
        )
        create_response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=create_payload)
        assert create_response.status_code == 201, create_response.text
        created = create_response.json()
        char_id = int(created["id"])

        update_one = client.put(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{char_id}?expected_version={created['version']}",
            json={"description": "Updated description"},
        )
        assert update_one.status_code == 200, update_one.text
        updated_one = update_one.json()

        update_two = client.put(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{char_id}?expected_version={updated_one['version']}",
            json={"tags": ["alpha", "beta"]},
        )
        assert update_two.status_code == 200, update_two.text

        from_version = int(created["version"])
        to_version = int(update_two.json()["version"])
        diff_response = client.get(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{char_id}/versions/diff"
            f"?from_version={from_version}&to_version={to_version}"
        )
        assert diff_response.status_code == 200, diff_response.text
        diff_payload = diff_response.json()

        assert int(diff_payload["character_id"]) == char_id
        assert int(diff_payload["from_entry"]["version"]) == from_version
        assert int(diff_payload["to_entry"]["version"]) == to_version
        assert int(diff_payload["changed_count"]) >= 1
        changed_fields = {field["field"] for field in diff_payload["changed_fields"]}
        assert "description" in changed_fields
        assert "tags" in changed_fields

    def test_revert_character_to_previous_version_integration(self, client: TestClient):
        create_payload = create_sample_character_payload(
            "VersionRevert",
            description="Baseline description",
        )
        create_response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=create_payload)
        assert create_response.status_code == 201, create_response.text
        created = create_response.json()
        char_id = int(created["id"])
        version_one = int(created["version"])

        update_response = client.put(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{char_id}?expected_version={version_one}",
            json={"description": "Mutated description"},
        )
        assert update_response.status_code == 200, update_response.text
        updated = update_response.json()
        version_two = int(updated["version"])
        assert updated["description"] == "Mutated description"

        revert_response = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{char_id}/revert",
            json={"target_version": version_one},
        )
        assert revert_response.status_code == 200, revert_response.text
        reverted = revert_response.json()
        assert reverted["description"] == "Baseline description"
        assert int(reverted["version"]) == version_two + 1

        versions_response = client.get(f"{CHARACTERS_ENDPOINT_PREFIX}/{char_id}/versions?limit=10")
        assert versions_response.status_code == 200, versions_response.text
        versions_payload = versions_response.json()
        latest_entry = versions_payload["items"][0]
        assert int(latest_entry["version"]) == int(reverted["version"])
        latest_payload = latest_entry.get("payload", {})
        assert latest_payload.get("description") == "Baseline description"

    def test_update_character_version_conflict_integration(self, client: TestClient):
        create_payload = create_sample_character_payload("VersionConflict")
        create_response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=create_payload)
        assert create_response.status_code == 201, create_response.text
        create_resp_json = create_response.json()
        char_id = create_resp_json["id"]

        update_payload = {"description": "New Description"}
        response = client.put(f"{CHARACTERS_ENDPOINT_PREFIX}/{char_id}?expected_version=99", json=update_payload)
        assert response.status_code == 409, response.text
        assert "Version mismatch" in response.json()["detail"]

    def test_delete_character_integration(self, client: TestClient, test_db: CharactersRAGDB):
        create_payload = create_sample_character_payload("ToDelete")
        create_response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=create_payload)
        assert create_response.status_code == 201, create_response.text
        create_resp_json = create_response.json()
        char_id = create_resp_json["id"]
        original_version = create_resp_json["version"]

        response = client.delete(f"{CHARACTERS_ENDPOINT_PREFIX}/{char_id}?expected_version={original_version}")
        assert response.status_code == 200, response.text
        assert response.json()["character_id"] == char_id

        db_char = test_db.get_character_card_by_id(char_id)
        assert db_char is None

        conn = test_db.get_connection()
        deleted_record = conn.execute(
            "SELECT deleted, version FROM character_cards WHERE id = ?", (char_id,)
        ).fetchone()
        assert deleted_record is not None
        assert deleted_record["deleted"] == 1
        assert deleted_record["version"] == original_version + 1

    def test_restore_character_integration(self, client: TestClient, test_db: CharactersRAGDB):
        create_payload = create_sample_character_payload("RestoreMe")
        create_response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=create_payload)
        assert create_response.status_code == 201, create_response.text
        created = create_response.json()
        char_id = int(created["id"])

        delete_response = client.delete(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{char_id}?expected_version={created['version']}"
        )
        assert delete_response.status_code == 200, delete_response.text

        conn = test_db.get_connection()
        deleted_record = conn.execute(
            "SELECT deleted, version FROM character_cards WHERE id = ?",
            (char_id,),
        ).fetchone()
        assert deleted_record is not None
        assert int(deleted_record["deleted"]) == 1
        deleted_version = int(deleted_record["version"])
        # Close the test-thread SQLite connection before issuing the restore request.
        # Requests run on a different thread with their own connection.
        test_db.close_connection()

        restore_response = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{char_id}/restore?expected_version={deleted_version}"
        )
        assert restore_response.status_code == 200, restore_response.text
        restored = restore_response.json()
        assert int(restored["id"]) == char_id
        assert restored["name"] == created["name"]
        assert int(restored["version"]) == deleted_version + 1

        restored_db = test_db.get_character_card_by_id(char_id)
        assert restored_db is not None
        assert int(restored_db["deleted"]) == 0

    def test_restore_character_active_row_returns_409(self, client: TestClient):
        create_response = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/",
            json=create_sample_character_payload("RestoreActiveAPI"),
        )
        assert create_response.status_code == 201, create_response.text
        created = create_response.json()

        response = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{created['id']}/restore?expected_version={created['version']}"
        )

        assert response.status_code == 409, response.text
        assert "already active" in response.json()["detail"].lower()

    def test_restore_character_version_conflict_integration(
        self, client: TestClient, test_db: CharactersRAGDB
    ):
        create_payload = create_sample_character_payload("RestoreConflict")
        create_response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=create_payload)
        assert create_response.status_code == 201, create_response.text
        created = create_response.json()
        char_id = int(created["id"])

        delete_response = client.delete(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{char_id}?expected_version={created['version']}"
        )
        assert delete_response.status_code == 200, delete_response.text

        conn = test_db.get_connection()
        deleted_record = conn.execute(
            "SELECT version FROM character_cards WHERE id = ?",
            (char_id,),
        ).fetchone()
        assert deleted_record is not None
        deleted_version = int(deleted_record["version"])
        # Ensure test-thread connection is closed before restore request.
        test_db.close_connection()

        response = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{char_id}/restore?expected_version={deleted_version + 1}"
        )
        assert response.status_code == 409, response.text
        assert "version mismatch" in response.json()["detail"].lower()

    def test_restore_character_not_found_returns_conflict(self, client: TestClient):
        response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/999999/restore?expected_version=1")
        assert response.status_code == 409, response.text
        assert "not found" in response.json()["detail"].lower()

    def test_restore_character_outside_retention_window_integration(
        self, client: TestClient, test_db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("CHARACTERS_RESTORE_RETENTION_DAYS", "1")

        create_payload = create_sample_character_payload("RestoreExpired")
        create_response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=create_payload)
        assert create_response.status_code == 201, create_response.text
        created = create_response.json()
        char_id = int(created["id"])

        delete_response = client.delete(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{char_id}?expected_version={created['version']}"
        )
        assert delete_response.status_code == 200, delete_response.text

        conn = test_db.get_connection()
        deleted_record = conn.execute(
            "SELECT version FROM character_cards WHERE id = ?",
            (char_id,),
        ).fetchone()
        assert deleted_record is not None
        deleted_version = int(deleted_record["version"])

        expired_deleted_at = (
            datetime.now(timezone.utc) - timedelta(days=3)
        ).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        conn.execute(
            "UPDATE character_cards SET last_modified = ? WHERE id = ?",
            (expired_deleted_at, char_id),
        )
        conn.commit()
        test_db.close_connection()

        response = client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/{char_id}/restore?expected_version={deleted_version}"
        )
        assert response.status_code == 409, response.text
        detail = response.json().get("detail", "")
        assert "restore window expired" in detail.lower()
        assert "could only be restored until" in detail.lower()

    def test_search_character_integration(self, client: TestClient, test_db: CharactersRAGDB):
        unique_name_search = f"SearchableNameAPI_{uuid.uuid4().hex[:6]}"
        desc_keyword = f"unique_keyword_search_api_{uuid.uuid4().hex[:4]}"

        client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/",
            json=create_sample_character_payload(name=unique_name_search, description=f"Has {desc_keyword}"),
        )
        client.post(
            f"{CHARACTERS_ENDPOINT_PREFIX}/",
            json=create_sample_character_payload(name=f"OtherSearch_{uuid.uuid4().hex[:6]}"),
        )

        response = client.get(f"{CHARACTERS_ENDPOINT_PREFIX}/search/?query={unique_name_search}")
        assert response.status_code == 200, response.text
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == unique_name_search

        response_keyword = client.get(f"{CHARACTERS_ENDPOINT_PREFIX}/search/?query={desc_keyword}*")
        assert response_keyword.status_code == 200, response_keyword.text
        data_keyword = response_keyword.json()
        assert len(data_keyword) == 1
        assert data_keyword[0]["name"] == unique_name_search

    def test_import_character_png_integration(self, client: TestClient, test_db: CharactersRAGDB):
        char_name_for_png = f"PNG Import Char {uuid.uuid4().hex[:4]}"
        dummy_card_data = {
            "spec": "chara_card_v2",
            "spec_version": "2.0",
            "data": {
                "name": char_name_for_png,
                "description": "Imported from PNG.",
                "personality": "Test",
                "scenario": "Test",
                "first_mes": "Hello from PNG!",
                "mes_example": "Example",
            },
        }
        chara_json_str = json.dumps(dummy_card_data)
        chara_base64 = base64.b64encode(chara_json_str.encode("utf-8")).decode("utf-8")

        img = PILImage.new("RGB", (60, 30), color="blue")
        png_info = PngImagePlugin.PngInfo()  # Corrected usage
        png_info.add_text("chara", chara_base64)

        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format="PNG", pnginfo=png_info)
        img_byte_arr.seek(0)

        files = {"character_file": (f"{char_name_for_png}.png", img_byte_arr, "image/png")}
        response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/import", files=files)

        assert response.status_code == 201, response.text
        data_wrapper = response.json()
        assert "character" in data_wrapper and data_wrapper["character"] is not None
        data = data_wrapper["character"]
        assert data["name"] == char_name_for_png
        assert data["description"] == "Imported from PNG."

        db_char = test_db.get_character_card_by_name(char_name_for_png)
        assert db_char is not None
        assert db_char["description"] == "Imported from PNG."

    @pytest.mark.parametrize("file_ext", ["yaml", "yml"])
    def test_import_character_yaml_integration(
        self,
        client: TestClient,
        test_db: CharactersRAGDB,
        file_ext: str,
    ):
        char_name_for_yaml = f"YAML Import Char {uuid.uuid4().hex[:4]}"
        yaml_payload = "\n".join(
            [
                f"name: {char_name_for_yaml}",
                "description: Imported from YAML endpoint.",
                "personality: Structured",
                "scenario: API integration coverage",
                "first_mes: Hello from YAML!",
                "mes_example: \"User: Hi\\nCharacter: Hello\"",
                "tags:",
                "  - yaml",
                "  - api",
            ]
        )
        files = {
            "character_file": (
                f"{char_name_for_yaml}.{file_ext}",
                yaml_payload.encode("utf-8"),
                "text/yaml",
            )
        }

        response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/import", files=files)

        assert response.status_code == 201, response.text
        data_wrapper = response.json()
        assert "character" in data_wrapper and data_wrapper["character"] is not None
        data = data_wrapper["character"]
        assert data["name"] == char_name_for_yaml
        assert data["description"] == "Imported from YAML endpoint."
        assert "yaml" in (data.get("tags") or [])

        db_char = test_db.get_character_card_by_name(char_name_for_yaml)
        assert db_char is not None
        assert db_char["description"] == "Imported from YAML endpoint."

    def test_import_character_rejects_unsupported_extension(self, client: TestClient):
        files = {
            "character_file": (
                "unsupported.csv",
                b"name,description\nbad,format\n",
                "text/csv",
            )
        }
        response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/import", files=files)

        assert response.status_code == 400, response.text
        detail = response.json().get("detail", "")
        assert "not allowed" in detail.lower()
        assert "allowed:" in detail.lower()
        assert ".json" in detail
        assert ".yaml" in detail
        assert ".yml" in detail

    def test_import_supported_extension_message_order_is_stable(self):
        expected = ", ".join(sorted(characters_api_module.ALLOWED_EXTENSIONS))
        assert characters_api_module._format_allowed_extensions() == expected

    def test_import_malformed_yaml_falls_back_to_plain_text_character(
        self,
        client: TestClient,
    ):
        malformed_yaml = "---\nname: [missing\n---"
        files = {
            "character_file": (
                "malformed.yaml",
                malformed_yaml.encode("utf-8"),
                "text/yaml",
            )
        }
        response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/import", files=files)

        assert response.status_code == 201, response.text
        data_wrapper = response.json()
        assert "character" in data_wrapper and data_wrapper["character"] is not None
        data = data_wrapper["character"]
        assert data.get("name")
        assert malformed_yaml in (data.get("description") or "")
        assert "plain-text" in (data.get("tags") or [])

    def test_export_character_v2_format_integration(self, client: TestClient):
        create_payload = create_sample_character_payload(
            "ExportV2",
            description="V2 export description",
            personality="V2 export personality",
            scenario="V2 export scenario",
            first_message="V2 first message",
            message_example="User: Hi\nCharacter: Hello",
            creator_notes="V2 notes",
            system_prompt="V2 system prompt",
            post_history_instructions="V2 post-history",
            alternate_greetings=["Alt 1", "Alt 2"],
            tags=["v2", "export"],
            creator="QA",
            character_version="2.4",
        )
        create_response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=create_payload)
        assert create_response.status_code == 201, create_response.text
        character_id = create_response.json()["id"]

        export_response = client.get(f"{CHARACTERS_ENDPOINT_PREFIX}/{character_id}/export?format=v2")
        assert export_response.status_code == 200, export_response.text
        data = export_response.json()

        assert data["spec"] == "chara_card_v2"
        assert data["spec_version"] == "2.0"
        assert "data" in data and isinstance(data["data"], dict)
        assert data["data"]["name"] == create_payload["name"]
        assert data["data"]["description"] == create_payload["description"]
        assert data["data"]["personality"] == create_payload["personality"]
        assert data["data"]["scenario"] == create_payload["scenario"]
        assert data["data"]["first_mes"] == create_payload["first_message"]
        assert data["data"]["mes_example"] == create_payload["message_example"]
        assert data["data"]["creator_notes"] == create_payload["creator_notes"]
        assert data["data"]["system_prompt"] == create_payload["system_prompt"]
        assert data["data"]["post_history_instructions"] == create_payload["post_history_instructions"]
        assert data["data"]["alternate_greetings"] == create_payload["alternate_greetings"]
        assert data["data"]["tags"] == create_payload["tags"]
        assert data["data"]["creator"] == create_payload["creator"]
        assert data["data"]["character_version"] == create_payload["character_version"]
        assert isinstance(data["data"]["extensions"], dict)
        assert data["data"]["char_image"]
        assert data["character_image"] == data["data"]["char_image"]


# ======================= PROPERTY-BASED TESTS (API) =========================

_PBT_DEBUG = os.getenv("PBT_DEBUG", "").lower() in {"1", "true", "yes", "y", "on"}
_PBT_RELAX = os.getenv("PBT_RELAX", "").lower() in {"1", "true", "yes", "y", "on"}


@settings(
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.function_scoped_fixture,
    ],
    max_examples=25,
    verbosity=Verbosity.verbose if _PBT_DEBUG else Verbosity.normal,
)
@given(payload=st_character_create_payload_pbt())
def test_pbt_create_character_api(client: TestClient, test_db: CharactersRAGDB, payload: Dict[str, Any]):
    # Ensure name is unique for each Hypothesis example if db is shared across examples within one PBT run
    # (pytest fixtures with scope="function" are setup once per test function, not per hypothesis example)
    payload["name"] = f"{payload['name']}_{uuid.uuid4().hex[:8]}"

    # Avoid creating a character with a name that might already exist from a *previous* PBT example
    # if the DB is not perfectly clean for each example (it is not, for Hypothesis)
    if test_db.get_character_card_by_name(payload["name"]):
        if _PBT_DEBUG:
            event("skip:create-collision")
            note(f"Create collision on name={payload['name']}")
        assume(False)  # Skip this example if name collides from previous example in same PBT run

    response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=payload)

    assert response.status_code == 201, f"Payload: {payload}, Response: {response.text}"
    data = response.json()  # This is CharacterResponse
    assert data["name"] == payload["name"]

    # Check all fields from payload against the response
    for key, value in payload.items():
        if key == "image_base64":
            assert data["image_present"] is (value is not None)
        elif key in ["alternate_greetings", "tags", "extensions"]:
            expected_val = value
            if isinstance(value, str):  # If payload sent JSON string
                try:
                    expected_val = json.loads(value)
                except json.JSONDecodeError:  # Should be caught by Pydantic if invalid
                    # If Pydantic allows non-JSON string for these fields and they are returned as such:
                    pass  # expected_val remains the string
            # If expected_val is None and API returns default empty list/dict:
            if expected_val is None and key in data and (data[key] == [] or data[key] == {}):
                pass  # This is acceptable if API behavior is to default None to empty collection
            else:
                if key == "tags":
                    expected_val = _normalize_expected_tags_for_api(expected_val)
                assert data.get(key) == expected_val, f"Mismatch for {key}"
        elif value is not None:  # For other simple fields that were provided
            assert data.get(key) == value, f"Mismatch for {key}"
        elif value is None:  # If payload field was None
            # Check if the API response field is also None or a suitable default (e.g., "" for optional strings)
            api_val = data.get(key)
            assert (
                api_val is None or api_val == ""
            ), f"Mismatch for {key} (expected None or empty string, got {api_val})"

    db_char = test_db.get_character_card_by_id(data["id"])
    assert db_char is not None
    assert db_char["name"] == payload["name"]
    if payload.get("image_base64"):
        assert db_char["image"] is not None
    else:
        assert db_char["image"] is None


@settings(
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.function_scoped_fixture,
        HealthCheck.filter_too_much,
    ],
    max_examples=25,
    verbosity=Verbosity.verbose if _PBT_DEBUG else Verbosity.normal,
)
@given(initial_payload_gen=st_character_create_payload_pbt(), update_payload_diff_gen=st_character_update_payload_pbt())
def test_pbt_update_character_api(
    client: TestClient,
    test_db: CharactersRAGDB,
    initial_payload_gen: Dict[str, Any],
    update_payload_diff_gen: Dict[str, Any],
):
    # --- Create initial character ---
    initial_payload = initial_payload_gen.copy()  # Avoid modifying the generated dict directly
    initial_payload["name"] = f"{initial_payload['name']}_{uuid.uuid4().hex[:8]}"
    if test_db.get_character_card_by_name(initial_payload["name"]):
        if _PBT_DEBUG:
            event("skip:init-collision")
            note(f"Initial collision on name={initial_payload['name']}")
        assume(False)  # Avoid collision for initial creation

    create_response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=initial_payload)
    if create_response.status_code != 201:
        if _PBT_DEBUG:
            event("skip:create-failed")
            note(f"Create failed: status={create_response.status_code} body={create_response.text}")
    assume(create_response.status_code == 201)  # If creation fails, skip this example
    created_char_data = create_response.json()  # This is CharacterResponse
    char_id = created_char_data["id"]
    current_version = created_char_data["version"]

    # --- Prepare and perform update ---
    update_payload_diff = update_payload_diff_gen.copy()

    # Handle potential name update and uniqueness
    if "name" in update_payload_diff and update_payload_diff["name"] is not None:
        # Ensure new name is valid (non-empty) and unique if it's being changed
        if not str(update_payload_diff["name"]).strip():  # Invalid name (empty or whitespace)
            if _PBT_DEBUG:
                event("skip:update-invalid-name-empty")
                note(f"Update invalid name: {update_payload_diff['name']!r}")
            if _PBT_RELAX:
                # Replace with a minimal valid name
                update_payload_diff["name"] = "X"
            else:
                assume(False)  # Pydantic should catch this, but we can assume valid generated name here

        updated_unique_name = f"{update_payload_diff['name']}_{uuid.uuid4().hex[:8]}"
        existing_with_new_name = test_db.get_character_card_by_name(updated_unique_name)
        if existing_with_new_name and existing_with_new_name["id"] != char_id:
            if _PBT_DEBUG:
                event("skip:update-name-collision")
                note(f"Update name collision on {updated_unique_name}")
            if _PBT_RELAX:
                # Force uniqueness by appending more entropy
                updated_unique_name = f"{updated_unique_name}_{uuid.uuid4().hex[:4]}"
            else:
                assume(False)  # Avoid conflict with another existing character
        update_payload_diff["name"] = updated_unique_name
    elif "name" in update_payload_diff and update_payload_diff["name"] is None:
        # Name is a required field in the database - cannot be NULL
        # Skip this test case as it would violate DB constraints
        if _PBT_DEBUG:
            event("skip:update-name-null")
            note("Update payload attempted to set name=None")
        if _PBT_RELAX:
            update_payload_diff.pop("name", None)
        else:
            assume(False)

    # Use Pydantic model to see what `exclude_unset=True` would do.
    # This helps determine which fields were *actually* intended for update.
    try:
        pydantic_update_model = CharacterUpdate.model_validate(update_payload_diff)
    except Exception as e:
        if _PBT_DEBUG:
            event("skip:update-pydantic-invalid")
            note(f"Pydantic validation failed: {e} for payload={update_payload_diff}")
        assume(False)  # Skip this example as it's not a valid update payload for the API
        return

    payload_sent_to_lib = pydantic_update_model.model_dump(exclude_unset=True)

    if not payload_sent_to_lib:  # If all fields were unset or default after Pydantic processing
        if _PBT_DEBUG:
            event("skip:update-empty-after-validate")
            note(f"Payload empty after Pydantic processing; original={update_payload_diff}")
        if _PBT_RELAX:
            # Inject a harmless change to ensure non-empty update
            update_payload_diff["description"] = "pbt-relax"
            pydantic_update_model = CharacterUpdate.model_validate(update_payload_diff)
            payload_sent_to_lib = pydantic_update_model.model_dump(exclude_unset=True)
        else:
            assume(False)  # This update wouldn't change anything, skip.
            # Note: `st_character_update_payload_pbt` tries to ensure at least one field.

    update_response = client.put(
        f"{CHARACTERS_ENDPOINT_PREFIX}/{char_id}?expected_version={current_version}", json=update_payload_diff
    )  # Send original generated diff

    if update_response.status_code == 422:  # Pydantic validation error from API
        if _PBT_DEBUG:
            event("skip:update-422")
            note(f"Update 422: payload={update_payload_diff}, response={update_response.text}")
        assume(False)  # Generated data was invalid for the model, skip successful assertion part
        return

    assert (
        update_response.status_code == 200
    ), f"Update failed. Initial: {created_char_data['name']}, UpdatePayload: {update_payload_diff}, SentToLib: {payload_sent_to_lib}, Response: {update_response.text}"

    updated_data_api = update_response.json()  # This is CharacterResponse

    # --- Assertions ---
    assert updated_data_api["id"] == char_id
    assert updated_data_api["version"] == current_version + 1

    # Verify each field in the response
    for resp_key, resp_value in updated_data_api.items():
        if resp_key in [
            "id",
            "version",
            "image_base64",
            "updated_at",
            "last_modified",
        ]:  # Server-managed metadata checked elsewhere or expected to change on update.
            continue

        if resp_key == "image_present":
            if "image_base64" in payload_sent_to_lib:
                assert resp_value is (payload_sent_to_lib["image_base64"] is not None)
            else:  # image_base64 was not part of the update
                assert resp_value == created_char_data["image_present"]
            continue

        # If the key was in the actual data sent to the library function for update
        if resp_key in payload_sent_to_lib:
            expected_value = payload_sent_to_lib[resp_key]
            # Handle special cases where the API transforms None values
            if resp_key in ["alternate_greetings", "tags"] and expected_value is None:
                expected_value = []  # API converts None to empty list for these fields
            elif resp_key == "extensions" and expected_value is None:
                expected_value = {}  # API converts None to empty dict for extensions
            if resp_key == "tags":
                expected_value = _normalize_expected_tags_for_api(expected_value)
            # The `payload_sent_to_lib` should have Python objects if JSON strings were parsed by Pydantic
            assert (
                resp_value == expected_value
            ), f"Mismatch for updated key '{resp_key}'. API: {resp_value}, Expected (post-Pydantic): {expected_value}"
        else:
            # Key was not in the update payload, so it should be same as original character
            assert resp_value == created_char_data.get(
                resp_key
            ), f"Mismatch for non-updated key '{resp_key}'. API: {resp_value}, Original: {created_char_data.get(resp_key)}"

    # Optional: Double check against DB
    db_char_after_update = test_db.get_character_card_by_id(char_id)
    assert db_char_after_update is not None
    assert db_char_after_update["version"] == current_version + 1
    if "name" in payload_sent_to_lib:
        assert db_char_after_update["name"] == payload_sent_to_lib["name"]
    if "image_base64" in payload_sent_to_lib:
        assert (db_char_after_update["image"] is not None) == (payload_sent_to_lib["image_base64"] is not None)
