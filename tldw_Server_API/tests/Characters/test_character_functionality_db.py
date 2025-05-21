# test_character_functionality.py
#
#
# Imports
import time
from typing import Any, Generator
import pytest
import sqlite3
import json
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os
#
# Third-party imports
#
# Local imports
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    InputError,
    ConflictError,
    CharactersRAGDBError,
    # SchemaError, # Not directly tested unless schema initialization fails
)
#
########################################################################################################################
#
# --- Constants for tests ---
TEST_CLIENT_ID = "test-client-pytest"


# --- Helper function for DB instance ---
@pytest.fixture
def db() -> Generator[CharactersRAGDB, Any, None]:
    """Provides a fresh in-memory CharactersRAGDB instance for each test."""
    database = CharactersRAGDB(":memory:", client_id=TEST_CLIENT_ID)
    yield database
    database.close_connection()  # Ensure connection is closed after test


# --- Helper for sample character card data ---
def sample_card_data(name="Test Character", **kwargs) -> dict:
    """Generates sample character card data with defaults."""
    data = {
        "name": name,
        "description": "A character for testing purposes.",
        "personality": "Brave and curious.",
        "scenario": "Exploring ancient ruins.",
        "system_prompt": "You are this test character.",
        "image": b"sample_image_data_bytes",
        "post_history_instructions": "Keep track of decisions.",
        "first_message": "Greetings, traveler!",
        "message_example": "User: Hello\nAI: Hi there!",
        "creator_notes": "Test card notes.",
        "alternate_greetings": ["Hi!", "Hey!", "Salutations!"],  # List
        "tags": ["test", "sample-char"],  # List
        "creator": "Pytest Fixture",
        "character_version": "1.0.0",
        "extensions": {"custom_data": "value1", "setting": True}  # Dict
    }
    data.update(kwargs)
    return data


# --- Helper to check ISO timestamps ---
def is_recent_iso_timestamp(timestamp_str: str, tolerance_seconds: int = 5) -> bool:
    """Checks if a Z-formatted ISO timestamp string is recent."""
    if not timestamp_str:
        return False
    try:
        # Convert 'Z' to +00:00 for fromisoformat
        dt_obj = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        if dt_obj.tzinfo is None or dt_obj.tzinfo.utcoffset(dt_obj) is None:
            return False  # Must be timezone-aware (UTC)
        now_utc = datetime.now(timezone.utc)
        return (now_utc - timedelta(seconds=tolerance_seconds)) <= dt_obj <= (
                    now_utc + timedelta(seconds=1))  # Allow a bit future for clock skew
    except ValueError:
        return False


class TestCharacterCardAddition:
    def test_add_minimal_character(self, db: CharactersRAGDB):
        card_name = "Minimal Char"
        card_id = db.add_character_card({"name": card_name})
        assert card_id is not None
        assert isinstance(card_id, int)

        retrieved_card = db.get_character_card_by_id(card_id)
        assert retrieved_card is not None
        assert retrieved_card["name"] == card_name
        assert retrieved_card["description"] is None  # Default for unspecified fields

    def test_add_full_character(self, db: CharactersRAGDB):
        data = sample_card_data(name="Full Char")
        card_id = db.add_character_card(data)
        assert card_id is not None

        retrieved = db.get_character_card_by_id(card_id)
        assert retrieved is not None
        for key, value in data.items():
            if key == "alternate_greetings" or key == "tags":  # Compare as sets due to potential order change
                assert set(retrieved[key]) == set(value)
            elif key == "extensions":
                assert retrieved[key] == value  # Dicts compare fine
            else:
                assert retrieved[key] == value

    def test_add_character_verify_autofilled_fields(self, db: CharactersRAGDB):
        card_name = "AutoFields Char"
        card_id = db.add_character_card({"name": card_name})
        assert card_id is not None

        retrieved = db.get_character_card_by_id(card_id)
        assert retrieved is not None
        assert retrieved["id"] == card_id
        assert retrieved["deleted"] == 0
        assert retrieved["version"] == 1
        assert retrieved["client_id"] == TEST_CLIENT_ID
        assert is_recent_iso_timestamp(retrieved["created_at"])
        assert retrieved["created_at"] == retrieved["last_modified"]

    def test_add_character_missing_name_raises_input_error(self, db: CharactersRAGDB):
        with pytest.raises(InputError, match="Required field 'name' is missing or empty."):
            db.add_character_card({"description": "A char without a name."})
        with pytest.raises(InputError, match="Required field 'name' is missing or empty."):
            db.add_character_card({"name": ""})  # Empty name

    def test_add_duplicate_name_raises_conflict_error(self, db: CharactersRAGDB):
        card_name = "Duplicate Name Char"
        db.add_character_card({"name": card_name})  # Add first one
        with pytest.raises(ConflictError, match=f"Character card with name '{card_name}' already exists."):
            db.add_character_card({"name": card_name})  # Try adding again

    # JSON field input variations
    def test_add_character_with_list_json_fields(self, db: CharactersRAGDB):
        data = sample_card_data(
            alternate_greetings=["Yo!"],
            tags=["tag_list"],
            extensions={"ext_key": ["val1", "val2"]}
        )
        card_id = db.add_character_card(data)
        retrieved = db.get_character_card_by_id(card_id)
        assert retrieved["alternate_greetings"] == ["Yo!"]
        assert retrieved["tags"] == ["tag_list"]
        assert retrieved["extensions"] == {"ext_key": ["val1", "val2"]}

    def test_add_character_with_set_json_fields(self, db: CharactersRAGDB):
        # Sets are converted to lists for JSON storage
        alt_greetings_set = {"Hola", "Bonjour"}
        tags_set = {"tag_set1", "tag_set2"}
        data = sample_card_data(
            name="SetJSONChar",
            alternate_greetings=alt_greetings_set,
            tags=tags_set
        )
        card_id = db.add_character_card(data)
        retrieved = db.get_character_card_by_id(card_id)
        assert isinstance(retrieved["alternate_greetings"], list)
        assert set(retrieved["alternate_greetings"]) == alt_greetings_set
        assert isinstance(retrieved["tags"], list)
        assert set(retrieved["tags"]) == tags_set

    def test_add_character_with_string_json_fields(self, db: CharactersRAGDB):
        # Strings that are valid JSON
        alt_greetings_json_str = '["Hola JSON", "Bonjour JSON"]'
        tags_json_str = '["tag_json_str1", "tag_json_str2"]'
        extensions_json_str = '{"key_json": "val_json"}'
        data = sample_card_data(
            name="StringJSONChar",
            alternate_greetings=alt_greetings_json_str,
            tags=tags_json_str,
            extensions=extensions_json_str
        )
        card_id = db.add_character_card(data)
        retrieved = db.get_character_card_by_id(card_id)
        assert retrieved["alternate_greetings"] == json.loads(alt_greetings_json_str)
        assert retrieved["tags"] == json.loads(tags_json_str)
        assert retrieved["extensions"] == json.loads(extensions_json_str)

    def test_add_character_with_invalid_string_for_json_field_becomes_none(self, db: CharactersRAGDB, caplog):
        # Behavior: if a string is passed for a JSON field but it's not valid JSON,
        # add_character_card stores it as is.
        # get_character_card_by_id then tries to json.loads it, fails, logs warning, and returns None for that field.
        invalid_json_str = "this is not a valid json array"
        data = sample_card_data(name="InvalidStringJSONChar", tags=invalid_json_str)
        card_id = db.add_character_card(data)

        # Check raw storage (optional, more for debugging the DB behavior)
        conn = db.get_connection()
        raw_tags = conn.execute("SELECT tags FROM character_cards WHERE id = ?", (card_id,)).fetchone()['tags']
        assert raw_tags == invalid_json_str  # Stored as the invalid string

        retrieved = db.get_character_card_by_id(card_id)
        assert retrieved["tags"] is None  # Deserialization fails, field becomes None
        assert f"Failed to decode JSON for field 'tags' in row (ID: {card_id})" in caplog.text

    def test_add_character_with_none_json_fields(self, db: CharactersRAGDB):
        data = sample_card_data(
            name="NoneJSONChar",
            alternate_greetings=None,
            tags=None,
            extensions=None
        )
        card_id = db.add_character_card(data)
        retrieved = db.get_character_card_by_id(card_id)
        assert retrieved["alternate_greetings"] is None
        assert retrieved["tags"] is None
        assert retrieved["extensions"] is None


class TestCharacterCardRetrieval:
    @pytest.fixture
    def populated_db(self, db: CharactersRAGDB) -> CharactersRAGDB:
        self.char1_data = sample_card_data(name="Char One", description="First character")
        self.char2_data = sample_card_data(name="Char Two", description="Second character", tags=["alpha", "beta"])
        self.char3_data = sample_card_data(name="Char Three DELETED", description="Third, will be deleted")

        self.char1_id = db.add_character_card(self.char1_data)
        self.char2_id = db.add_character_card(self.char2_data)
        char3_id_temp = db.add_character_card(self.char3_data)
        # Soft delete char3
        db.soft_delete_character_card(char3_id_temp, expected_version=1)
        return db

    def test_get_character_by_id(self, populated_db: CharactersRAGDB):
        retrieved = populated_db.get_character_card_by_id(self.char1_id)
        assert retrieved is not None
        assert retrieved["name"] == self.char1_data["name"]
        assert retrieved["description"] == self.char1_data["description"]
        assert set(retrieved["tags"]) == set(self.char1_data["tags"])  # example_card_data has default tags

    def test_get_character_by_id_non_existent(self, db: CharactersRAGDB):
        assert db.get_character_card_by_id(99999) is None

    def test_get_character_by_id_soft_deleted(self, populated_db: CharactersRAGDB):
        # char3_id was added and then deleted in populated_db fixture
        # To get its ID, we might need to query with deleted=1 or know its name
        char3_deleted = populated_db.execute_query("SELECT id FROM character_cards WHERE name = ? AND deleted = 1",
                                                   (self.char3_data["name"],)).fetchone()
        assert char3_deleted is not None, "Deleted char not found even with direct query"

        assert populated_db.get_character_card_by_id(char3_deleted["id"]) is None

    def test_get_character_by_name(self, populated_db: CharactersRAGDB):
        retrieved = populated_db.get_character_card_by_name(self.char2_data["name"])
        assert retrieved is not None
        assert retrieved["id"] == self.char2_id
        assert retrieved["description"] == self.char2_data["description"]
        assert set(retrieved["tags"]) == set(self.char2_data["tags"])

    def test_get_character_by_name_non_existent(self, db: CharactersRAGDB):
        assert db.get_character_card_by_name("NonExistent Name") is None

    def test_get_character_by_name_soft_deleted(self, populated_db: CharactersRAGDB):
        assert populated_db.get_character_card_by_name(self.char3_data["name"]) is None

    def test_list_characters_empty(self, db: CharactersRAGDB):
        assert db.list_character_cards() == []

    def test_list_characters_populated(self, populated_db: CharactersRAGDB):
        cards = populated_db.list_character_cards()
        assert len(cards) == 2  # Char One, Char Two (Char Three is deleted)
        card_names = [c["name"] for c in cards]
        assert self.char1_data["name"] in card_names
        assert self.char2_data["name"] in card_names
        assert self.char3_data["name"] not in card_names

    def test_list_characters_pagination_and_order(self, populated_db: CharactersRAGDB):
        # Order is by name. "Char One", "Char Two"
        cards_limit1 = populated_db.list_character_cards(limit=1)
        assert len(cards_limit1) == 1
        assert cards_limit1[0]["name"] == "Char One"  # Assuming "Char One" < "Char Two"

        cards_offset1 = populated_db.list_character_cards(limit=1, offset=1)
        assert len(cards_offset1) == 1
        assert cards_offset1[0]["name"] == "Char Two"

        cards_all = populated_db.list_character_cards(limit=10)  # Get all
        assert len(cards_all) == 2
        assert cards_all[0]["name"] < cards_all[1]["name"]  # Check order


class TestCharacterCardUpdate:
    @pytest.fixture
    def char_id_for_update(self, db: CharactersRAGDB) -> int:
        card_id = db.add_character_card(sample_card_data(name="Updateable Char"))
        assert card_id is not None
        return card_id

    def test_update_character_successful(self, db: CharactersRAGDB, char_id_for_update: int):
        original_card = db.get_character_card_by_id(char_id_for_update)
        assert original_card is not None
        original_version = original_card["version"]
        original_last_modified = original_card["last_modified"]

        update_payload = {
            "name": "Updated Name",
            "description": "Updated description.",
            "image": b"new_image_data",
            "tags": ["updated", "new_tag"],
            "extensions": {"new_key": "new_value", "setting": False}
        }

        # Brief pause to ensure last_modified changes
        time.sleep(0.01)

        assert db.update_character_card(char_id_for_update, update_payload, expected_version=original_version)

        updated_card = db.get_character_card_by_id(char_id_for_update)
        assert updated_card is not None
        assert updated_card["name"] == "Updated Name"
        assert updated_card["description"] == "Updated description."
        assert updated_card["image"] == b"new_image_data"
        assert set(updated_card["tags"]) == {"updated", "new_tag"}
        assert updated_card["extensions"] == {"new_key": "new_value", "setting": False}

        assert updated_card["version"] == original_version + 1
        assert updated_card["last_modified"] > original_last_modified
        assert is_recent_iso_timestamp(updated_card["last_modified"])
        assert updated_card["client_id"] == TEST_CLIENT_ID  # Should be updated to current client

    def test_update_character_version_mismatch(self, db: CharactersRAGDB, char_id_for_update: int):
        with pytest.raises(ConflictError, match="Update failed: version mismatch"):
            db.update_character_card(char_id_for_update, {"name": "Won't Update"}, expected_version=99)

    def test_update_character_non_existent(self, db: CharactersRAGDB):
        with pytest.raises(ConflictError, match="Record not found in character_cards."):
            db.update_character_card(99999, {"name": "No Char"}, expected_version=1)

    def test_update_character_soft_deleted(self, db: CharactersRAGDB, char_id_for_update: int):
        db.soft_delete_character_card(char_id_for_update, expected_version=1)  # version becomes 2
        with pytest.raises(ConflictError, match="Record is soft-deleted in character_cards."):
            db.update_character_card(char_id_for_update, {"name": "No Update"},
                                     expected_version=2)  # Trying to update deleted

    def test_update_name_to_duplicate_conflict(self, db: CharactersRAGDB, char_id_for_update: int):
        existing_name = "Existing Unique Name"
        db.add_character_card({"name": existing_name})  # Create another card

        with pytest.raises(ConflictError,
                           match=f"Cannot update character card ID {char_id_for_update}: name '{existing_name}' already exists."):
            db.update_character_card(char_id_for_update, {"name": existing_name}, expected_version=1)

    def test_update_character_empty_data_is_noop_true(self, db: CharactersRAGDB, char_id_for_update: int):
        # card_data = {} should return True without DB interaction
        original_card = db.get_character_card_by_id(char_id_for_update)
        assert db.update_character_card(char_id_for_update, {},
                                        expected_version=original_card["version"])  # Version not checked here

        after_noop_card = db.get_character_card_by_id(char_id_for_update)
        # Ensure no changes happened
        assert after_noop_card["version"] == original_card["version"]
        assert after_noop_card["last_modified"] == original_card["last_modified"]

    def test_update_character_with_only_ignored_fields_touches_record(self, db: CharactersRAGDB,
                                                                      char_id_for_update: int):
        original_card = db.get_character_card_by_id(char_id_for_update)
        original_version = original_card["version"]
        original_last_modified = original_card["last_modified"]

        # Update with data that contains only fields not in updatable_direct_fields or _CHARACTER_CARD_JSON_FIELDS
        # e.g. 'id' or some made-up field
        update_payload = {"id": 999, "made_up_field": "value"}

        time.sleep(0.01)

        assert db.update_character_card(char_id_for_update, update_payload, expected_version=original_version)

        touched_card = db.get_character_card_by_id(char_id_for_update)
        assert touched_card["version"] == original_version + 1  # Version bumps
        assert touched_card["last_modified"] > original_last_modified  # last_modified updates
        assert is_recent_iso_timestamp(touched_card["last_modified"])
        # Other fields should remain unchanged
        assert touched_card["name"] == original_card["name"]


class TestCharacterCardSoftDelete:
    @pytest.fixture
    def char_to_delete_id(self, db: CharactersRAGDB) -> int:
        card_id = db.add_character_card(sample_card_data(name="Deletable Char"))
        assert card_id is not None
        return card_id

    def test_soft_delete_character_successful(self, db: CharactersRAGDB, char_to_delete_id: int):
        original_card = db.get_character_card_by_id(char_to_delete_id)
        assert original_card is not None
        original_version = original_card["version"]

        time.sleep(0.01)

        assert db.soft_delete_character_card(char_to_delete_id, expected_version=original_version)

        # Verify it's gone from normal retrieval
        assert db.get_character_card_by_id(char_to_delete_id) is None
        assert db.get_character_card_by_name("Deletable Char") is None

        # Verify using a direct query that it's marked deleted and version/timestamp updated
        conn = db.get_connection()
        deleted_record_row = conn.execute("SELECT * FROM character_cards WHERE id = ?", (char_to_delete_id,)).fetchone()
        assert deleted_record_row is not None
        deleted_record = dict(deleted_record_row)

        assert deleted_record["deleted"] == 1
        assert deleted_record["version"] == original_version + 1
        assert is_recent_iso_timestamp(deleted_record["last_modified"])
        assert deleted_record["last_modified"] > original_card["last_modified"]
        assert deleted_record["client_id"] == TEST_CLIENT_ID

    def test_soft_delete_character_version_mismatch(self, db: CharactersRAGDB, char_to_delete_id: int):
        with pytest.raises(ConflictError, match="Soft delete for Character ID .* failed: version mismatch"):
            db.soft_delete_character_card(char_to_delete_id, expected_version=99)

    def test_soft_delete_non_existent(self, db: CharactersRAGDB):
        with pytest.raises(ConflictError, match="Record not found in character_cards."):
            db.soft_delete_character_card(99999, expected_version=1)

    def test_soft_delete_already_deleted_idempotent(self, db: CharactersRAGDB, char_to_delete_id: int):
        db.soft_delete_character_card(char_to_delete_id, expected_version=1)  # First delete, version becomes 2

        # Try deleting again. soft_delete_character_card has logic to check if already deleted.
        assert db.soft_delete_character_card(char_to_delete_id,
                                             expected_version=2)  # Expected version based on previous state

        # Check its state, version should not have incremented again from the idempotent call
        conn = db.get_connection()
        deleted_record_row = conn.execute("SELECT version FROM character_cards WHERE id = ?",
                                          (char_to_delete_id,)).fetchone()
        assert deleted_record_row["version"] == 2  # Still version 2


class TestCharacterCardSearch:
    @pytest.fixture
    def search_db(self, db: CharactersRAGDB) -> CharactersRAGDB:
        db.add_character_card(
            sample_card_data(name="Gandalf the Grey", description="A wise old wizard.", personality="Kind, powerful",
                             scenario="Fellowship journey"))
        db.add_character_card(sample_card_data(name="Aragorn Son of Arathorn", description="Ranger, heir of Isildur.",
                                               system_prompt="You are Strider."))
        db.add_character_card(sample_card_data(name="Bilbo Baggins", description="A hobbit who found a magic ring."))

        # Add a character to be deleted
        deleted_char_id = db.add_character_card(
            sample_card_data(name="Saruman the White", description="Corrupted wizard."))
        db.soft_delete_character_card(deleted_char_id, 1)
        return db

    def test_search_character_by_name(self, search_db: CharactersRAGDB):
        results = search_db.search_character_cards("Gandalf")
        assert len(results) == 1
        assert results[0]["name"] == "Gandalf the Grey"

    def test_search_character_by_description(self, search_db: CharactersRAGDB):
        results = search_db.search_character_cards("wizard")  # Gandalf, (Saruman - deleted)
        assert len(results) == 1
        assert results[0]["name"] == "Gandalf the Grey"

        results_ring = search_db.search_character_cards("ring")  # Bilbo
        assert len(results_ring) == 1
        assert results_ring[0]["name"] == "Bilbo Baggins"

    def test_search_character_by_personality(self, search_db: CharactersRAGDB):
        results = search_db.search_character_cards("Kind")  # Gandalf
        assert len(results) == 1
        assert results[0]["name"] == "Gandalf the Grey"

    def test_search_character_by_scenario(self, search_db: CharactersRAGDB):
        results = search_db.search_character_cards("journey")  # Gandalf
        assert len(results) == 1
        assert results[0]["name"] == "Gandalf the Grey"

    def test_search_character_by_system_prompt(self, search_db: CharactersRAGDB):
        results = search_db.search_character_cards("Strider")  # Aragorn
        assert len(results) == 1
        assert results[0]["name"] == "Aragorn Son of Arathorn"

    def test_search_character_multiple_terms(self, search_db: CharactersRAGDB):
        results = search_db.search_character_cards("Arathorn Ranger")  # Aragorn
        assert len(results) == 1
        assert results[0]["name"] == "Aragorn Son of Arathorn"

        results_gandalf = search_db.search_character_cards("Grey wizard")  # Gandalf
        assert len(results_gandalf) == 1
        assert results_gandalf[0]["name"] == "Gandalf the Grey"

    def test_search_character_no_results(self, search_db: CharactersRAGDB):
        results = search_db.search_character_cards("ZzzzzNonExistentTermZzzzz")
        assert len(results) == 0

    def test_search_character_limit_results(self, search_db: CharactersRAGDB):
        # Add more "wizard" characters if needed, or search a common term
        results_limit1 = search_db.search_character_cards(
            "description:wizard OR description:Ranger OR description:hobbit", limit=1)  # "description:" to target field
        assert len(results_limit1) == 1

        results_limit2 = search_db.search_character_cards(
            "description:wizard OR description:Ranger OR description:hobbit", limit=2)
        assert len(results_limit2) == 2

    def test_search_character_excludes_soft_deleted(self, search_db: CharactersRAGDB):
        results = search_db.search_character_cards("Saruman")
        assert len(results) == 0  # Saruman was soft-deleted

        results_corrupted = search_db.search_character_cards("Corrupted")
        assert len(results_corrupted) == 0