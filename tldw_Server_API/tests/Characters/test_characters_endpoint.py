#
#
# Imports
import base64
import binascii
import json
import sys

import pytest  # Using pytest for its features, though not strictly required by "no conftest"
import builtins  # For patching builtins.hasattr
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, patch, ANY, AsyncMock
#
# Third-Party Imports
from fastapi import FastAPI, UploadFile, Depends
from fastapi.testclient import TestClient
from starlette import status

# --- SUT (System Under Test) Imports ---
try:
    from tldw_Server_API.app.api.v1.endpoints import characters as characters_module
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError, InputError
    # Pydantic models from the SUT (optional for tests focusing on API I/O, but good for clarity)
    # from tldw_Server_API.app.api.v1.endpoints.characters import CharacterCreate, CharacterUpdate, CharacterResponse
except ImportError as e:
    # This block allows the test file to be syntactically correct even if SUT paths are initially wrong.
    # Tests will fail at runtime if these imports are truly broken.
    print(
        f"Critical Error: Could not import SUT modules: {e}. Please check PYTHONPATH and file paths. Tests will likely fail.")
#
########################################################################################################################
#
# --- Test Setup: FastAPI app and TestClient ---

# Global mock DB instance. This will be reset for each test by the dependency override.
mock_db_instance = MagicMock(spec=CharactersRAGDB)

# Debug print block from original, kept for context if needed
invalid_string = "this is not valid base64 data!@#"


# print(f"\n[DEBUG] In test: python version: {sys.version}")
# print(f"[DEBUG] In test: base64 module path: {base64.__file__}")
# try:
#     print(f"[DEBUG] In test: Attempting b64decode directly with: '{invalid_string}'")
#     decoded_data = base64.b64decode(invalid_string)
#     print(f"[DEBUG] In test: Direct b64decode DID NOT FAIL. Result: {decoded_data[:50]}") # Should not be reached
# except binascii.Error as e:
#     print(f"[DEBUG] In test: Direct b64decode FAILED AS EXPECTED: {type(e).__name__}: {e}")
# except Exception as e:
#     print(f"[DEBUG] In test: Direct b64decode failed with UNEXPECTED error: {type(e).__name__}: {e}")


async def override_get_chacha_db_for_tests():
    """
    FastAPI dependency override. Called for each request that depends on get_chacha_db_for_user.
    Provides a freshly reset mock DB instance for each test, ensuring test isolation.
    """
    # Reset the main mock object's call history, return_value, side_effect etc.
    # Using return_value=True and side_effect=True ensures these are reset to defaults.
    mock_db_instance.reset_mock(return_value=True, side_effect=True)

    # Explicitly re-assign known methods to new MagicMocks.
    # This is the most robust way to ensure that any `side_effect` (like iterators)
    # or `return_value` set on these child mocks in a previous test is cleared.
    # Adding `name` to mocks helps in debugging if a mock is unexpectedly called.
    mock_db_instance.get_character_card_by_id = MagicMock(name="get_character_card_by_id_mock")
    mock_db_instance.get_character_card_by_name = MagicMock(name="get_character_card_by_name_mock")
    mock_db_instance.add_character_card = MagicMock(name="add_character_card_mock")
    mock_db_instance.update_character_card = MagicMock(name="update_character_card_mock")
    # soft_delete_character_card is critical for delete tests, including those that check its existence.
    # Ensure it starts as a MagicMock. Tests might replace or check it.
    mock_db_instance.soft_delete_character_card = MagicMock(name="soft_delete_character_card_mock")
    mock_db_instance.list_character_cards = MagicMock(name="list_character_cards_mock")
    # Add any other CharactersRAGDB methods that are called by the character endpoints
    # and might have their state (return_value/side_effect) modified by tests.

    return mock_db_instance


# Create a FastAPI application instance for testing.
test_app = FastAPI()

# Apply the dependency override.
if 'get_chacha_db_for_user' in globals() and callable(
        get_chacha_db_for_user) and get_chacha_db_for_user.__name__ != '<lambda>':  # Check it's not a dummy
    test_app.dependency_overrides[get_chacha_db_for_user] = override_get_chacha_db_for_tests
else:
    print("Warning: `get_chacha_db_for_user` not found, not callable, or is a dummy from SUT imports. "
          "Dependency override for the database might be ineffective. Ensure SUT imports are correct.")

# Include the router from characters.py into the test application.
if hasattr(characters_module, 'router') and characters_module.router is not None:
    test_app.include_router(characters_module.router, prefix="/api/v1/characters", tags=["Characters"])
else:
    print("Warning: `characters_module.router` not found or is None. Endpoints will not be available for testing. "
          "Ensure SUT imports are correct and `characters.py` defines `router`.")

# TestClient instance using the configured test_app. This is used to make simulated HTTP requests.
client = TestClient(test_app)

# --- Test Data and Helper Functions ---
BASE_API_URL = "/api/v1/characters"
SAMPLE_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="  # 1x1 black PNG
SAMPLE_IMAGE_BYTES = base64.b64decode(SAMPLE_IMAGE_BASE64)


def get_sample_character_db_dict(
        id: int,
        name: str,
        version: int = 1,
        description: Optional[str] = "A test character",
        image_bytes: Optional[bytes] = None,
        tags: Optional[List[str]] = None,
        alternate_greetings: Optional[List[str]] = None,
        extensions: Optional[Dict[str, Any]] = None,
        **kwargs  # For other CharacterBase fields
) -> Dict[str, Any]:
    """
    Helper to create a dictionary mimicking a character record as it might be retrieved from the database.
    It handles converting Python lists/dicts for JSON-like fields into JSON strings,
    as the SUT's `_convert_db_char_to_response_model` expects to parse these if they come as strings from DB.
    However, for direct use as mock DB return, Python objects are fine for JSON fields if DB driver handles it.
    The _convert_db_char_to_response_model is robust enough to handle Python objects or JSON strings for these.
    """
    data = {
        "id": id,
        "name": name,
        "version": version,
        "description": description,
        "personality": kwargs.get("personality", "Test personality"),
        "scenario": kwargs.get("scenario", "Test scenario"),
        "system_prompt": kwargs.get("system_prompt", "Test system prompt"),
        "post_history_instructions": kwargs.get("post_history_instructions", "Test post history instructions"),
        "first_message": kwargs.get("first_message", "Hello test!"),
        "message_example": kwargs.get("message_example", "<START>USER: Hi\nASSISTANT: Hello Test<END>"),
        "creator_notes": kwargs.get("creator_notes", "Test creator notes"),
        "creator": kwargs.get("creator", "Test Creator"),
        "character_version": kwargs.get("character_version", "1.0"),
        "image": image_bytes,  # In DB representation, 'image' is bytes
        # For mock DB return, these can be Python objects. _convert_db_char_to_response_model handles it.
        # If simulating a DB that strictly returns JSON strings for these fields, use json.dumps.
        "tags": tags if tags is not None else [],  # Kept as Python list
        "alternate_greetings": alternate_greetings if alternate_greetings is not None else [],  # Kept as Python list
        "extensions": extensions if extensions is not None else {}  # Kept as Python dict
    }
    # If testing the scenario where DB returns JSON strings for these fields, you would do:
    # "tags": json.dumps(tags if tags is not None else []),
    # "alternate_greetings": json.dumps(alternate_greetings if alternate_greetings is not None else []),
    # "extensions": json.dumps(extensions if extensions is not None else {})
    return data


# Path for patching the `import_and_save_character_from_file` function.
# This function is imported and used within the `characters.py` module.
# So, we patch it *where it is looked up*, which is in `characters_module`.
PATCH_IMPORT_LIB_PATH = f"{characters_module.__name__}.import_and_save_character_from_file"


# --- Test Classes Grouped by Endpoint ---

class TestImportCharacter:
    """Tests for the POST /import endpoint."""

    @patch(PATCH_IMPORT_LIB_PATH)
    def test_import_character_from_file_success_new(self, mock_import_lib_func):
        char_id = 1
        char_name = "Imported Char"
        mock_import_lib_func.return_value = char_id

        db_char_data = get_sample_character_db_dict(id=char_id, name=char_name, image_bytes=SAMPLE_IMAGE_BYTES)
        mock_db_instance.get_character_card_by_id.return_value = db_char_data

        mock_file_content = b"dummy file content for character card"
        files = {'character_file': ('char.png', mock_file_content, 'image/png')}

        response = client.post(f"{BASE_API_URL}/import", files=files)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["message"] == f"Character '{char_name}' processed successfully."
        assert data["character"]["id"] == char_id
        assert data["character"]["name"] == char_name
        assert data["character"]["image_present"] is True
        assert data["character"]["image_base64"] == SAMPLE_IMAGE_BASE64

        mock_import_lib_func.assert_called_once_with(mock_db_instance, mock_file_content)
        mock_db_instance.get_character_card_by_id.assert_called_once_with(char_id)

    @patch(PATCH_IMPORT_LIB_PATH)
    def test_import_character_from_file_already_exists(self, mock_import_lib_func):
        existing_char_id = 2
        existing_char_name = "Existing Imported Char"
        mock_import_lib_func.return_value = existing_char_id

        db_char_data = get_sample_character_db_dict(id=existing_char_id, name=existing_char_name)
        mock_db_instance.get_character_card_by_id.return_value = db_char_data

        files = {'character_file': ('char.json', b'{"name": "Existing Char"}', 'application/json')}
        response = client.post(f"{BASE_API_URL}/import", files=files)

        assert response.status_code == status.HTTP_201_CREATED  # Or 200 OK, depending on desired SUT behavior
        data = response.json()
        assert data["message"] == f"Character '{existing_char_name}' processed successfully."
        assert data["character"]["id"] == existing_char_id

    @patch(PATCH_IMPORT_LIB_PATH)
    def test_import_character_empty_file(self, mock_import_lib_func):
        files = {'character_file': ('empty.txt', b'', 'text/plain')}
        response = client.post(f"{BASE_API_URL}/import", files=files)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Uploaded file is empty" in response.json()["detail"]  # Adjusted for SUT message
        mock_import_lib_func.assert_not_called()

    @patch(PATCH_IMPORT_LIB_PATH)
    def test_import_character_lib_fails_returns_none(self, mock_import_lib_func):
        mock_import_lib_func.return_value = None
        files = {'character_file': ('invalid.png', b'invalid content', 'image/png')}
        response = client.post(f"{BASE_API_URL}/import", files=files)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Failed to import character from file" in response.json()["detail"]


class TestListCharacters:
    """Tests for the GET / endpoint."""

    def test_list_characters_empty(self):
        mock_db_instance.list_character_cards.return_value = []
        response = client.get(f"{BASE_API_URL}/")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == []
        mock_db_instance.list_character_cards.assert_called_once_with(limit=100, offset=0)

    def test_list_characters_with_data(self):
        char1_db = get_sample_character_db_dict(id=1, name="Char1", image_bytes=SAMPLE_IMAGE_BYTES, tags=["a"])
        char2_db = get_sample_character_db_dict(id=2, name="Char2", tags=["b", "c"])
        mock_db_instance.list_character_cards.return_value = [char1_db, char2_db]

        response = client.get(f"{BASE_API_URL}/")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] == 1
        assert data[0]["name"] == "Char1"
        assert data[0]["image_present"] is True
        assert data[0]["tags"] == ["a"]
        assert data[1]["id"] == 2
        assert data[1]["name"] == "Char2"
        assert data[1]["image_present"] is False
        assert data[1]["tags"] == ["b", "c"]

    def test_list_characters_with_pagination(self):
        mock_db_instance.list_character_cards.return_value = []
        response = client.get(f"{BASE_API_URL}/?limit=10&offset=5")
        assert response.status_code == status.HTTP_200_OK
        mock_db_instance.list_character_cards.assert_called_once_with(limit=10, offset=5)

    def test_list_characters_db_error(self):
        mock_db_instance.list_character_cards.side_effect = CharactersRAGDBError("DB List Error")
        response = client.get(f"{BASE_API_URL}/")
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Database error: DB List Error" in response.json()["detail"]


class TestCreateCharacter:
    """Tests for the POST / endpoint (create character)."""

    def test_create_character_success(self):
        char_create_payload = {
            "name": "New Char", "description": "A shiny new character.",
            "personality": "Optimistic", "tags": ["new", "test"],  # Send as list
            "alternate_greetings": ["Hi!", "Greetings!"],  # Send as list
            "extensions": {"source": "test_suite"},  # Send as dict
            "image_base64": SAMPLE_IMAGE_BASE64
        }
        # SUT might also accept JSON strings for these due to `parse_json_string` validator,
        # but sending Python types directly is cleaner for testing model binding logic.
        # The original test sent JSON strings for some fields; this is also a valid test case for the validator.
        # For this specific test, direct Python types are fine.

        created_char_id = 101
        mock_db_instance.get_character_card_by_name.return_value = None
        mock_db_instance.add_character_card.return_value = created_char_id

        db_char_data_after_creation = get_sample_character_db_dict(
            id=created_char_id, name=char_create_payload["name"], description=char_create_payload["description"],
            personality=char_create_payload["personality"], tags=char_create_payload["tags"],
            alternate_greetings=char_create_payload["alternate_greetings"],
            extensions=char_create_payload["extensions"],
            image_bytes=SAMPLE_IMAGE_BYTES, version=1
        )
        mock_db_instance.get_character_card_by_id.return_value = db_char_data_after_creation

        response = client.post(f"{BASE_API_URL}/", json=char_create_payload)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["id"] == created_char_id
        assert data["name"] == char_create_payload["name"]
        assert data["tags"] == ["new", "test"]
        assert data["alternate_greetings"] == ["Hi!", "Greetings!"]
        assert data["extensions"] == {"source": "test_suite"}
        assert data["image_present"] is True
        assert data["image_base64"] == SAMPLE_IMAGE_BASE64

        mock_db_instance.get_character_card_by_name.assert_called_once_with(char_create_payload["name"])

        actual_call_args_dict = mock_db_instance.add_character_card.call_args[0][0]
        assert actual_call_args_dict["name"] == char_create_payload["name"]
        assert actual_call_args_dict["image"] == SAMPLE_IMAGE_BYTES
        assert actual_call_args_dict["tags"] == ["new", "test"]
        assert actual_call_args_dict["alternate_greetings"] == ["Hi!", "Greetings!"]
        assert actual_call_args_dict["extensions"] == {"source": "test_suite"}
        assert actual_call_args_dict.get("scenario") is None

    def test_create_character_name_conflict(self):
        char_create_data = {"name": "Existing Char"}  # Minimal valid payload for CharacterCreate
        # SUT uses db.get_character_card_by_name, ensure this returns something
        mock_db_instance.get_character_card_by_name.return_value = get_sample_character_db_dict(id=99,
                                                                                                name="Existing Char")

        response = client.post(f"{BASE_API_URL}/", json=char_create_data)
        assert response.status_code == status.HTTP_409_CONFLICT
        assert "Character with name 'Existing Char' already exists" in response.json()["detail"]

    def test_create_character_missing_name(self):
        response = client.post(f"{BASE_API_URL}/", json={"description": "A character without a name"})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_character_invalid_image_base64(self):
        char_create_data = {"name": "CharWithBadImage", "image_base64": "this is not valid base64 data!@#"}
        mock_db_instance.get_character_card_by_name.return_value = None  # Assume name is not a conflict

        # The SUT's _prepare_char_data_for_db uses base64.b64decode(..., validate=True)
        # which will raise an error for this string.
        response = client.post(f"{BASE_API_URL}/", json=char_create_data)
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        # The SUT's error detail for b64decode includes the specific error message from binascii.Error
        # e.g., "Invalid base64-encoded string: number of data characters (<x_bin_448>is not a multiple of 4"
        # or "Non-base64 digit found"
        assert "Invalid image_base64 data" in response.json()["detail"]


class TestGetCharacterById:
    """Tests for the GET /{character_id} endpoint."""

    def test_get_character_success(self):
        char_id = 1
        db_char_data = get_sample_character_db_dict(id=char_id, name="Test Char", image_bytes=SAMPLE_IMAGE_BYTES,
                                                    tags=["retrieved"])
        mock_db_instance.get_character_card_by_id.return_value = db_char_data

        response = client.get(f"{BASE_API_URL}/{char_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == char_id
        assert data["name"] == "Test Char"
        assert data["image_present"] is True
        assert data["image_base64"] == SAMPLE_IMAGE_BASE64
        assert data["tags"] == ["retrieved"]
        mock_db_instance.get_character_card_by_id.assert_called_once_with(char_id)

    def test_get_character_not_found(self):
        char_id = 999
        mock_db_instance.get_character_card_by_id.return_value = None
        response = client.get(f"{BASE_API_URL}/{char_id}")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert f"Character with ID {char_id} not found" in response.json()["detail"]

    def test_get_character_invalid_id_format(self):
        response = client.get(f"{BASE_API_URL}/notaninteger")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_get_character_invalid_id_value_zero(self):
        response = client.get(f"{BASE_API_URL}/0")  # Path(gt=0)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestUpdateCharacter:
    """Tests for the PUT /{character_id} endpoint."""

    def test_update_character_success(self):
        char_id = 1
        current_version = 1
        current_char_db_data = get_sample_character_db_dict(
            id=char_id, name="Old Name", version=current_version, description="Old Desc"
        )
        updated_char_db_data = get_sample_character_db_dict(
            id=char_id, name="New Name", version=current_version + 1,
            description="New Desc", image_bytes=SAMPLE_IMAGE_BYTES
        )

        mock_db_instance.get_character_card_by_id.side_effect = [current_char_db_data, updated_char_db_data]
        mock_db_instance.get_character_card_by_name.return_value = None  # New name not conflicting
        mock_db_instance.update_character_card.return_value = True

        update_payload = {
            "name": "New Name", "description": "New Desc", "image_base64": SAMPLE_IMAGE_BASE64
        }
        response = client.put(f"{BASE_API_URL}/{char_id}", json=update_payload)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "New Name"
        assert data["description"] == "New Desc"
        assert data["version"] == current_version + 1
        assert data["image_present"] is True
        assert data["image_base64"] == SAMPLE_IMAGE_BASE64

        assert mock_db_instance.get_character_card_by_id.call_count == 2
        mock_db_instance.get_character_card_by_name.assert_called_once_with("New Name")

        expected_db_update_data = {
            "name": "New Name", "description": "New Desc", "image": SAMPLE_IMAGE_BYTES
        }
        mock_db_instance.update_character_card.assert_called_once_with(
            character_id=char_id,
            card_data=expected_db_update_data,
            expected_version=current_version
        )

    def test_update_character_remove_image(self):
        char_id = 1
        current_version = 1
        current_db = get_sample_character_db_dict(id=char_id, name="CharWithImage", version=current_version,
                                                  image_bytes=SAMPLE_IMAGE_BYTES)
        updated_db = get_sample_character_db_dict(id=char_id, name="CharWithImage", version=current_version + 1,
                                                  image_bytes=None)

        mock_db_instance.get_character_card_by_id.side_effect = [current_db, updated_db]
        mock_db_instance.update_character_card.return_value = True

        response = client.put(f"{BASE_API_URL}/{char_id}", json={"image_base64": None})
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["image_present"] is False
        assert data["image_base64"] is None

        mock_db_instance.update_character_card.assert_called_once_with(
            character_id=char_id, card_data={"image": None}, expected_version=current_version
        )

    def test_update_character_no_updatable_fields(self):
        char_id = 1
        current_char_db_dict = get_sample_character_db_dict(id=char_id, name="NoChangeChar")

        # mock_db_instance is reset by override_get_chacha_db_for_tests
        mock_db_instance.get_character_card_by_id.return_value = current_char_db_dict
        # update_character_card mock is already a fresh MagicMock from the override

        response = client.put(f"{BASE_API_URL}/{char_id}", json={})

        assert response.status_code == status.HTTP_200_OK  # SUT returns current data
        assert response.json()["name"] == "NoChangeChar"
        mock_db_instance.update_character_card.assert_not_called()

    def test_update_character_version_conflict(self):
        char_id = 1
        current_version = 1
        current_char_db_dict = get_sample_character_db_dict(id=char_id, name="MyChar", version=current_version)

        mock_db_instance.get_character_card_by_id.return_value = current_char_db_dict
        mock_db_instance.update_character_card.side_effect = ConflictError("Version mismatch during update")

        response = client.put(f"{BASE_API_URL}/{char_id}", json={"description": "Trying new description"})
        assert response.status_code == status.HTTP_409_CONFLICT
        assert "Version mismatch during update" in response.json()["detail"]  # SUT uses str(e) for detail


class TestDeleteCharacter:
    """Tests for the DELETE /{character_id} endpoint."""

    def test_delete_character_success(self):
        char_id = 1
        char_version = 2
        char_name = "Char About To Be Deleted"
        char_to_delete_db_data = get_sample_character_db_dict(id=char_id, name=char_name, version=char_version)

        mock_db_instance.get_character_card_by_id.return_value = char_to_delete_db_data
        # soft_delete_character_card is a fresh mock from override, set its return_value
        mock_db_instance.soft_delete_character_card.return_value = True

        response = client.delete(f"{BASE_API_URL}/{char_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == f"Character with ID {char_id} ('{char_name}') marked as deleted successfully."
        assert data["character_id"] == char_id

        mock_db_instance.get_character_card_by_id.assert_called_once_with(char_id)
        mock_db_instance.soft_delete_character_card.assert_called_once_with(char_id, expected_version=char_version)

    def test_delete_character_not_found(self):
        char_id = 999
        mock_db_instance.get_character_card_by_id.return_value = None
        # soft_delete_character_card mock will exist but won't be called

        response = client.delete(f"{BASE_API_URL}/{char_id}")
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert f"Character with ID {char_id} not found for deletion" in response.json()["detail"]  # SUT message
        mock_db_instance.soft_delete_character_card.assert_not_called()

    def test_delete_character_version_conflict(self):
        char_id = 1
        char_db_data = get_sample_character_db_dict(id=char_id, name="ConflictDelete", version=1)

        mock_db_instance.get_character_card_by_id.return_value = char_db_data
        mock_db_instance.soft_delete_character_card.side_effect = ConflictError("Version mismatch on delete attempt")

        response = client.delete(f"{BASE_API_URL}/{char_id}")
        assert response.status_code == status.HTTP_409_CONFLICT
        assert "Cannot delete character: Version mismatch on delete attempt" in response.json()[
            "detail"]  # SUT detail includes str(e)

    def test_delete_character_db_method_not_implemented(self):
        char_id = 1
        char_to_delete_db_data = get_sample_character_db_dict(id=char_id, name="NoDeleteMethodChar", version=1)

        mock_db_instance.get_character_card_by_id.return_value = char_to_delete_db_data

        # Store original hasattr to restore it later (though patch context manager handles this)
        original_hasattr = builtins.hasattr

        def mock_hasattr_for_delete_test(obj, name):
            # Important: Compare obj identity with the mock_db_instance provided by the dependency override
            # This ensures we only affect our DB mock.
            # The `mock_db_instance` global should be the one injected.
            if obj is mock_db_instance and name == 'soft_delete_character_card':
                return False
            return original_hasattr(obj, name)

        with patch('builtins.hasattr', new=mock_hasattr_for_delete_test):
            response = client.delete(f"{BASE_API_URL}/{char_id}")

        assert response.status_code == status.HTTP_501_NOT_IMPLEMENTED
        assert "Character deletion functionality is not available on the server." in response.json()["detail"]

        # Ensure the actual delete method (if it somehow existed despite hasattr patch) wasn't called
        # mock_db_instance.soft_delete_character_card should be the fresh mock from override, so it exists.
        # The hasattr patch should prevent it from being called.
        mock_db_instance.soft_delete_character_card.assert_not_called()

#
# End of test_characters_endpoint.py
########################################################################################################################
