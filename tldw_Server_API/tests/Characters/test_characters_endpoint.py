# test_api_characters.py
import base64
import json
import uuid
from typing import List, Dict, Any, Optional, Generator
from io import BytesIO
#
# Third-party imports
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage, PngImagePlugin  # Corrected PIL import
# Third-party imports
from hypothesis import given, strategies as st, settings, HealthCheck, assume
from unittest.mock import patch, MagicMock  # For unit tests
from loguru import logger

from tldw_Server_API.app.api.v1.schemas.character_schemas import CharacterUpdate
#
# Local Imports
from tldw_Server_API.app.main import app  # Your FastAPI app instance
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
    CharactersRAGDBError
)
from tldw_Server_API.app.api.v1.endpoints import characters_endpoint as characters_api_module
#######################################################################################################################
#
# --- Constants ---
BASE_URL_V1 = "/api/v1"
# Ensure this matches the prefix in your app.include_router for the characters API
CHARACTERS_ENDPOINT_PREFIX = "/api/v1/characters"


# --- Helper Functions / Fixtures for Integration Tests ---

@pytest.fixture(scope="function")
def test_db() -> Generator[CharactersRAGDB, Any, None]:
    # Using a unique client_id for the test DB instance
    db_instance = CharactersRAGDB(":memory:", client_id=f"db-client-test-{uuid.uuid4().hex[:6]}")
    yield db_instance
    db_instance.close_connection()


@pytest.fixture
def client(test_db: CharactersRAGDB) -> Generator[TestClient, Any, None]:
    """
    Provides a TestClient instance with the real DB dependency overridden
    for integration tests.
    """
    # This is where get_chacha_db_for_user is defined or imported in your actual app
    # For testing, we override the dependency that the *endpoints file* uses.
    # The path for dependency_overrides should be the actual dependency callable.
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    def override_get_db_for_test():
        logger.info("<<<<< OVERRIDE override_get_db_for_test CALLED >>>>>")
        try:
            yield test_db
        finally:
            pass  # test_db fixture handles its own close

    app.dependency_overrides[get_chacha_db_for_user] = override_get_db_for_test
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def create_dummy_image_base64(width=10, height=10, image_format="PNG") -> str:
    img = PILImage.new('RGB', (width, height), color='red')
    buffered = BytesIO()
    img.save(buffered, format=image_format)
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


def create_sample_character_payload(name_suffix: str = "", **overrides) -> Dict[str, Any]:
    payload = {
        "name": f"Test Char API {name_suffix}{uuid.uuid4().hex[:6]}",
        "description": "A character for API testing.",
        "first_message": "Hello from API Test!",
        "tags": ["api", "test"],
        "image_base64": create_dummy_image_base64()
    }
    payload.update(overrides)
    return payload


# --- Hypothesis Strategies for PBT ---
st_valid_api_text = st.text(min_size=1, max_size=50, alphabet=st.characters(min_codepoint=32, max_codepoint=126,
                                                                            blacklist_categories=('Cc', 'Cs')))
st_optional_api_text = st.one_of(st.none(), st_valid_api_text)  # Allow empty string as well, if Pydantic model permits
st_api_json_list_or_str = st.one_of(
    st.none(),
    st.lists(st_valid_api_text, max_size=2, unique=True),
    st.lists(st_valid_api_text, max_size=2, unique=True).map(json.dumps)
)
st_api_json_dict_or_str = st.one_of(
    st.none(),
    st.dictionaries(st_valid_api_text, st.one_of(st_valid_api_text, st.integers(0, 100), st.booleans(), st.none()),
                    max_size=2),
    st.dictionaries(st_valid_api_text, st_valid_api_text, max_size=2).map(json.dumps)
)
st_base64_image_str = st.one_of(st.none(), st.just(create_dummy_image_base64()))


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
        image_base64=st_base64_image_str
    ).filter(lambda x: x["name"] is not None and x["name"].strip() != "")


# Revised strategy for update payload to be less sparse
def st_character_update_payload_pbt():
    keys = [  # All fields that can be part of an update payload
        "name", "description", "personality", "scenario", "system_prompt",
        "post_history_instructions", "first_message", "message_example",
        "creator_notes", "alternate_greetings", "tags", "creator",
        "character_version", "extensions", "image_base64"
    ]
    # Strategies for values that are definitely not None (when chosen to be the "concrete" one)
    concrete_value_strategies = {
        "name": st_valid_api_text,
        "description": st_valid_api_text,
        # ... (fill for all keys, ensuring they don't generate None)
        "tags": st.lists(st_valid_api_text, min_size=1, max_size=2, unique=True),
        "image_base64": st.just(create_dummy_image_base64())
    }
    # Strategies that can produce None (for other fields not chosen as concrete)
    optional_value_strategies = {
        "name": st_optional_api_text,
        "description": st_optional_api_text,
        # ... (fill for all keys)
        "tags": st_api_json_list_or_str,  # Can be None
        "image_base64": st_base64_image_str  # Can be None
    }
    # Ensure all keys are in both strategy dicts for simplicity in lookup
    for k in keys:
        if k not in concrete_value_strategies: concrete_value_strategies[k] = st_valid_api_text  # Default concrete
        if k not in optional_value_strategies: optional_value_strategies[k] = st_optional_api_text  # Default optional

    @st.composite
    def at_least_one_concrete_field_payload(draw):
        # Draw a subset of keys to include in the update payload, must include at least one
        num_fields_to_update = draw(st.integers(min_value=1, max_value=len(keys)))
        selected_keys = draw(
            st.lists(st.sampled_from(keys), min_size=num_fields_to_update, max_size=num_fields_to_update, unique=True))

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

UNIT_TEST_PATCH_PREFIX = 'tldw_Server_API.app.api.v1.endpoints.characters_endpoint'


@patch(f'{UNIT_TEST_PATCH_PREFIX}.create_new_character_from_data')
@patch(f'{UNIT_TEST_PATCH_PREFIX}.get_character_details')
def test_unit_create_character_success(mock_get_details: MagicMock, mock_create: MagicMock, client: TestClient):
    mock_create.return_value = 1
    mock_char_data = {
        "id": 1, "name": "Unit Test Char", "version": 1, "description": "Desc",
        "image": b"dummy", "alternate_greetings": ["Hi"], "tags": ["test"], "extensions": {"key": "val"}
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


@patch(f'{UNIT_TEST_PATCH_PREFIX}.create_new_character_from_data')
def test_unit_create_character_conflict(mock_create: MagicMock, client: TestClient):
    mock_create.side_effect = ConflictError("Character with name 'Exists' already exists.")
    payload = {"name": "Exists", "description": "Desc"}
    response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=payload)
    assert response.status_code == 409, response.text
    assert "Character with name 'Exists' already exists." in response.json()["detail"]


@patch(f'{UNIT_TEST_PATCH_PREFIX}.create_new_character_from_data')
def test_unit_create_character_input_error_from_lib(mock_create: MagicMock, client: TestClient):
    # Test case where Pydantic validation passes, but the library function raises InputError
    mock_create.side_effect = InputError("Lib-level Invalid input for character.")
    payload = {"name": "ValidPydanticName", "description": "Desc",
               "image_base64": "invalid-b64!"}  # image_base64 will be caught by lib

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


@patch(f'{UNIT_TEST_PATCH_PREFIX}.get_character_details')
def test_unit_get_character_success(mock_get_details: MagicMock, client: TestClient):
    mock_char_data = {"id": 1, "name": "Fetched Char", "version": 1, "image": None}
    mock_get_details.return_value = mock_char_data
    response = client.get(f"{CHARACTERS_ENDPOINT_PREFIX}/1")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["name"] == "Fetched Char"
    assert data["image_present"] is False
    mock_get_details.assert_called_once_with(mock_get_details.call_args[0][0], 1)


@patch(f'{UNIT_TEST_PATCH_PREFIX}.get_character_details')
def test_unit_get_character_not_found(mock_get_details: MagicMock, client: TestClient):
    mock_get_details.return_value = None
    response = client.get(f"{CHARACTERS_ENDPOINT_PREFIX}/999")
    assert response.status_code == 404, response.text
    assert "not found" in response.json()["detail"]


@patch(f'{UNIT_TEST_PATCH_PREFIX}.update_existing_character_details')
@patch(f'{UNIT_TEST_PATCH_PREFIX}.get_character_details')
def test_unit_update_character_success(mock_get_details: MagicMock, mock_update: MagicMock, client: TestClient):
    mock_get_details.side_effect = [
        {"id": 1, "name": "Old Name", "version": 1, "image": None},
        {"id": 1, "name": "New Name", "version": 2, "image": b"newimg"}
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


@patch(f'{UNIT_TEST_PATCH_PREFIX}.get_character_details')  # Only get_character_details is called before version check
def test_unit_update_character_version_mismatch(mock_get_details: MagicMock, client: TestClient):
    mock_get_details.return_value = {"id": 1, "name": "Old Name", "version": 2}
    payload = {"description": "New Desc"}
    response = client.put(f"{CHARACTERS_ENDPOINT_PREFIX}/1?expected_version=1", json=payload)
    assert response.status_code == 409, response.text
    assert "Version mismatch" in response.json()["detail"]


@patch(f'{UNIT_TEST_PATCH_PREFIX}.delete_character_from_db')
@patch(f'{UNIT_TEST_PATCH_PREFIX}.get_character_details')
def test_unit_delete_character_success(mock_get_details: MagicMock, mock_delete: MagicMock, client: TestClient):
    mock_get_details.return_value = {"id": 1, "name": "ToDelete", "version": 1}
    mock_delete.return_value = True
    response = client.delete(f"{CHARACTERS_ENDPOINT_PREFIX}/1?expected_version=1")
    assert response.status_code == 200, response.text
    assert response.json()["message"] == "Character 'ToDelete' (ID: 1) soft-deleted."
    mock_delete.assert_called_once_with(mock_delete.call_args[0][0], 1, 1)


@patch(f'{UNIT_TEST_PATCH_PREFIX}.create_new_character_from_data')
@patch(f'{UNIT_TEST_PATCH_PREFIX}.get_character_details')
def test_unit_create_character_success(mock_get_details: MagicMock, mock_create: MagicMock, client: TestClient):
    mock_create.return_value = 1 # This is char_id
    # This mock_char_data is what get_character_details returns from DB
    mock_char_data_from_db = {
        "id": 1, "name": "Unit Test Char", "version": 1, "description": "Desc",
        "image": b"dummy_image_bytes", # mock image bytes from DB
        "alternate_greetings": ["Hi"], "tags": ["test"], "extensions": {"key": "val"},
        # Add all other fields expected by _convert_db_char_to_response_model / CharacterResponse
        "personality": None, "scenario": None, "system_prompt": None,
        "post_history_instructions": None, "first_message": None,
        "message_example": None, "creator_notes": None, "creator": None,
        "character_version": None, "created_at": "2023-01-01T00:00:00", "updated_at": "2023-01-01T00:00:00",
        "deleted": 0
    }
    mock_get_details.return_value = mock_char_data_from_db

    payload = {"name": "Unit Test Char", "description": "Desc"} # API input
    response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=payload)

    assert response.status_code == 201, response.text
    data = response.json() # This is CharacterResponse
    assert data["id"] == 1
    assert data["name"] == "Unit Test Char"
    assert data["image_present"] is True # Because mock_char_data_from_db["image"] is present
    mock_create.assert_called_once()
    # mock_create is called with (db_obj, character_payload_dict)
    assert mock_create.call_args[0][1]["name"] == payload["name"]
    # mock_get_details is called with (db_obj, char_id)
    mock_get_details.assert_called_once_with(mock_create.call_args[0][0], 1)


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

    def test_update_character_integration(self, client: TestClient, test_db: CharactersRAGDB):
        create_payload = create_sample_character_payload("UpdateBase")
        create_response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=create_payload)
        assert create_response.status_code == 201, create_response.text
        create_resp_json = create_response.json()

        char_id = create_resp_json["id"]
        original_version = create_resp_json["version"]

        update_payload = {
            "name": "Updated Character Name", "description": "Updated description.",
            "tags": ["newtag"], "image_base64": None
        }
        response = client.put(f"{CHARACTERS_ENDPOINT_PREFIX}/{char_id}?expected_version={original_version}",
                              json=update_payload)
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
        deleted_record = conn.execute("SELECT deleted, version FROM character_cards WHERE id = ?",
                                      (char_id,)).fetchone()
        assert deleted_record is not None
        assert deleted_record["deleted"] == 1
        assert deleted_record["version"] == original_version + 1

    def test_search_character_integration(self, client: TestClient, test_db: CharactersRAGDB):
        unique_name_search = f"SearchableNameAPI_{uuid.uuid4().hex[:6]}"
        desc_keyword = f"unique_keyword_search_api_{uuid.uuid4().hex[:4]}"

        client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/",
                    json=create_sample_character_payload(name=unique_name_search, description=f"Has {desc_keyword}"))
        client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/",
                    json=create_sample_character_payload(name=f"OtherSearch_{uuid.uuid4().hex[:6]}"))

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
            "spec": "chara_card_v2", "spec_version": "2.0",
            "data": {"name": char_name_for_png, "description": "Imported from PNG.",
                     "personality": "Test", "scenario": "Test",
                     "first_mes": "Hello from PNG!", "mes_example": "Example"}
        }
        chara_json_str = json.dumps(dummy_card_data)
        chara_base64 = base64.b64encode(chara_json_str.encode('utf-8')).decode('utf-8')

        img = PILImage.new('RGB', (60, 30), color='blue')
        png_info = PngImagePlugin.PngInfo()  # Corrected usage
        png_info.add_text("chara", chara_base64)

        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='PNG', pnginfo=png_info)
        img_byte_arr.seek(0)

        files = {'character_file': (f'{char_name_for_png}.png', img_byte_arr, 'image/png')}
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


# ======================= PROPERTY-BASED TESTS (API) =========================

@settings(deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.function_scoped_fixture],
          max_examples=25)  # Reduced for faster debugging, increase later
@given(payload=st_character_create_payload_pbt())
def test_pbt_create_character_api(client: TestClient, test_db: CharactersRAGDB, payload: Dict[str, Any]):
    # Ensure name is unique for each Hypothesis example if db is shared across examples within one PBT run
    # (pytest fixtures with scope="function" are setup once per test function, not per hypothesis example)
    payload["name"] = f"{payload['name']}_{uuid.uuid4().hex[:8]}"

    # Avoid creating a character with a name that might already exist from a *previous* PBT example
    # if the DB is not perfectly clean for each example (it is not, for Hypothesis)
    if test_db.get_character_card_by_name(payload["name"]):
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
                assert data.get(key) == expected_val, f"Mismatch for {key}"
        elif value is not None:  # For other simple fields that were provided
            assert data.get(key) == value, f"Mismatch for {key}"
        elif value is None:  # If payload field was None
            # Check if the API response field is also None or a suitable default (e.g., "" for optional strings)
            api_val = data.get(key)
            assert api_val is None or api_val == "", f"Mismatch for {key} (expected None or empty string, got {api_val})"

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
        HealthCheck.too_slow, HealthCheck.data_too_large,
        HealthCheck.function_scoped_fixture, HealthCheck.filter_too_much
    ],
    max_examples=25  # Start with fewer examples
)
@given(initial_payload_gen=st_character_create_payload_pbt(), update_payload_diff_gen=st_character_update_payload_pbt())
def test_pbt_update_character_api(client: TestClient, test_db: CharactersRAGDB,
                                  initial_payload_gen: Dict[str, Any],
                                  update_payload_diff_gen: Dict[str, Any]):
    # --- Create initial character ---
    initial_payload = initial_payload_gen.copy()  # Avoid modifying the generated dict directly
    initial_payload["name"] = f"{initial_payload['name']}_{uuid.uuid4().hex[:8]}"
    if test_db.get_character_card_by_name(initial_payload["name"]):
        assume(False)  # Avoid collision for initial creation

    create_response = client.post(f"{CHARACTERS_ENDPOINT_PREFIX}/", json=initial_payload)
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
            assume(False)  # Pydantic should catch this, but we can assume valid generated name here

        updated_unique_name = f"{update_payload_diff['name']}_{uuid.uuid4().hex[:8]}"
        existing_with_new_name = test_db.get_character_card_by_name(updated_unique_name)
        if existing_with_new_name and existing_with_new_name['id'] != char_id:
            assume(False)  # Avoid conflict with another existing character
        update_payload_diff["name"] = updated_unique_name
    elif "name" in update_payload_diff and update_payload_diff["name"] is None:
        # Allowing name to be set to None. Your CharacterUpdate Pydantic model and DB schema must allow this.
        # If name is required, Pydantic would raise a 422, and this test path wouldn't be hit with status 200.
        pass

    # Use Pydantic model to see what `exclude_unset=True` would do.
    # This helps determine which fields were *actually* intended for update.
    try:
        pydantic_update_model = CharacterUpdate.model_validate(update_payload_diff)
    except Exception:  # If Pydantic validation fails for the generated payload
        assume(False)  # Skip this example as it's not a valid update payload for the API
        return

    payload_sent_to_lib = pydantic_update_model.model_dump(exclude_unset=True)

    if not payload_sent_to_lib:  # If all fields were unset or default after Pydantic processing
        assume(False)  # This update wouldn't change anything, skip.
        # Note: `st_character_update_payload_pbt` tries to ensure at least one field.

    update_response = client.put(f"{CHARACTERS_ENDPOINT_PREFIX}/{char_id}?expected_version={current_version}",
                                 json=update_payload_diff)  # Send original generated diff

    if update_response.status_code == 422:  # Pydantic validation error from API
        assume(False)  # Generated data was invalid for the model, skip successful assertion part
        return

    assert update_response.status_code == 200, \
        f"Update failed. Initial: {created_char_data['name']}, UpdatePayload: {update_payload_diff}, SentToLib: {payload_sent_to_lib}, Response: {update_response.text}"

    updated_data_api = update_response.json()  # This is CharacterResponse

    # --- Assertions ---
    assert updated_data_api["id"] == char_id
    assert updated_data_api["version"] == current_version + 1

    # Verify each field in the response
    for resp_key, resp_value in updated_data_api.items():
        if resp_key in ["id", "version", "image_base64"]:  # Already checked or handled by image_present
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
            # The `payload_sent_to_lib` should have Python objects if JSON strings were parsed by Pydantic
            assert resp_value == expected_value, f"Mismatch for updated key '{resp_key}'. API: {resp_value}, Expected (post-Pydantic): {expected_value}"
        else:
            # Key was not in the update payload, so it should be same as original character
            assert resp_value == created_char_data.get(
                resp_key), f"Mismatch for non-updated key '{resp_key}'. API: {resp_value}, Original: {created_char_data.get(resp_key)}"

    # Optional: Double check against DB
    db_char_after_update = test_db.get_character_card_by_id(char_id)
    assert db_char_after_update is not None
    assert db_char_after_update["version"] == current_version + 1
    if "name" in payload_sent_to_lib:
        assert db_char_after_update["name"] == payload_sent_to_lib["name"]
    if "image_base64" in payload_sent_to_lib:
        assert (db_char_after_update["image"] is not None) == (payload_sent_to_lib["image_base64"] is not None)