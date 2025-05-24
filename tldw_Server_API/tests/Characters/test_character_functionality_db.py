# test_character_functionality_db.py
#
#
# Imports
import time
from typing import Any, Generator, List, Dict, Optional, Union
import pytest
import sqlite3
import json
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
import os
import logging  # For caplog
#
# Third-party imports
from hypothesis import given, strategies as st, settings, HealthCheck, assume, reproduce_failure
from hypothesis.errors import InvalidArgument
#
# Local imports
# Assuming the DB library is in a discoverable path. Adjust if necessary.
# For example, if tldw_Server_API is a top-level package in your project:
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    InputError,
    ConflictError,
    CharactersRAGDBError,
    SchemaError,
)

#
########################################################################################################################
#
# --- Constants for tests ---
TEST_CLIENT_ID = "test-client-pytest"
TEST_CLIENT_ID_ALT = "test-client-pytest-alt"


# --- Helper function for DB instance ---
@pytest.fixture
def db() -> Generator[CharactersRAGDB, Any, None]:
    """Provides a fresh in-memory CharactersRAGDB instance for each test."""
    database = CharactersRAGDB(":memory:", client_id=TEST_CLIENT_ID)
    yield database
    database.close_connection()  # Ensure connection is closed after test


@pytest.fixture
def file_db(tmp_path: Path) -> Generator[CharactersRAGDB, Any, None]:
    """Provides a file-based CharactersRAGDB instance for tests needing persistence/file ops."""
    db_file = tmp_path / "test_app.db"
    database = CharactersRAGDB(db_file, client_id=TEST_CLIENT_ID + "_file")
    yield database
    database.close_connection()
    # tmp_path fixture handles cleanup of the directory and file


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
        "alternate_greetings": ["Hi!", "Hey!", "Salutations!"],
        "tags": ["test", "sample-char"],
        "creator": "Pytest Fixture",
        "character_version": "1.0.0",
        "extensions": {"custom_data": "value1", "setting": True}
    }
    data.update(kwargs)
    return data


# --- Helper for sample conversation data ---
def sample_conversation_data(character_id: int, **kwargs) -> dict:
    data = {
        "character_id": character_id,
        "title": "Test Conversation",
        "rating": 4,
        "root_id": str(uuid.uuid4()),  # Provide a default root_id
    }
    data.update(kwargs)
    if 'id' not in data:  # if id not provided, use root_id as id often
        data['id'] = data['root_id']
    return data


# --- Helper for sample message data ---
def sample_message_data(conversation_id: str, sender: str = "user", **kwargs) -> dict:
    data = {
        "conversation_id": conversation_id,
        "sender": sender,
        "content": "This is a test message.",
    }
    data.update(kwargs)
    return data


# --- Helper for sample note data ---
def sample_note_data(title="Test Note", **kwargs) -> dict:
    data = {
        "title": title,
        "content": "This is the content of the test note."
    }
    data.update(kwargs)
    return data


# --- Helper to check ISO timestamps ---
def is_recent_iso_timestamp(timestamp_str: Optional[str], tolerance_seconds: int = 10) -> bool:  # Increased tolerance
    """Checks if a Z-formatted ISO timestamp string is recent."""
    if not timestamp_str:
        return False
    try:
        dt_obj = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        if dt_obj.tzinfo is None or dt_obj.tzinfo.utcoffset(dt_obj) is None:
            return False
        now_utc = datetime.now(timezone.utc)
        # Allow a bit more future for clock skew or slow test runner
        return (now_utc - timedelta(seconds=tolerance_seconds)) <= dt_obj <= (
                now_utc + timedelta(seconds=tolerance_seconds))
    except ValueError:
        return False


# --- Helper to check sync log entries ---
def check_sync_log_entry(db: CharactersRAGDB, entity: str, entity_id: Any, operation: str,
                         expected_version: Optional[int] = None,  # Can be None for linking table ops
                         client_id: str = TEST_CLIENT_ID,
                         check_payload_details: bool = True):
    log_entries = db.get_sync_log_entries(since_change_id=0)  # Get all

    relevant_logs = [
        log for log in log_entries
        if log['entity'] == entity and str(log['entity_id']) == str(entity_id) and log['operation'] == operation
    ]

    assert relevant_logs, f"No sync log entry found for {entity} ID {entity_id} with operation {operation}"

    last_log = relevant_logs[-1]  # Assume the last one is the most recent for this specific operation

    assert last_log['operation'] == operation
    if expected_version is not None:
        assert last_log['version'] == expected_version
    assert last_log['client_id'] == client_id  # Or the specific client_id used for the operation
    assert is_recent_iso_timestamp(last_log['timestamp'])

    assert isinstance(last_log['payload'], dict)
    # For linking tables, entity_id in sync_log is composite like "conv_id_kw_id"
    # For main tables, payload['id'] should match entity_id
    if entity not in ["conversation_keywords", "collection_keywords", "note_keywords"]:
        assert str(last_log['payload'].get('id')) == str(entity_id)

    if check_payload_details:
        if operation != 'delete':  # Delete payload is minimal for main entities
            if expected_version is not None:
                assert last_log['payload'].get('version') == expected_version
            # client_id might not be in payload for linking tables if not versioned
            if entity not in ["conversation_keywords", "collection_keywords", "note_keywords"]:
                assert last_log['payload'].get('client_id') == client_id
        elif entity not in ["conversation_keywords", "collection_keywords", "note_keywords"]:  # Main entity delete
            # The schema for 'delete' payload only includes id, deleted, last_modified, version, client_id
            assert last_log['payload'].get('deleted') == True or last_log['payload'].get('deleted') == 1

    return last_log  # Return the log entry for further specific checks if needed


# --- Hypothesis Strategies ---
# Define a strategy for generating valid text for names, descriptions, etc.
# Avoid null characters or other problematic unicode for SQLite FTS or general handling.
# Control characters (Cc) and surrogate pairs (Cs) are often problematic.
# Limit codepoints to common printable ranges if issues arise.
valid_text_chars = st.characters(
    min_codepoint=32,  # Space
    max_codepoint=126,  # ~ Tilde (common ASCII)
    # Optionally add more ranges, e.g., common Latin-1, Cyrillic, CJK, emoji if your FTS tokenizer handles them.
    # For now, stick to basic printable ASCII to avoid tokenizer complexities in tests.
    blacklist_categories=('Cc', 'Cs')  # Control characters, Surrogates
)

st_valid_name = st.text(min_size=1, max_size=100, alphabet=valid_text_chars)
st_description_text = st.text(max_size=500, alphabet=valid_text_chars)  # Can be empty
st_optional_text = st.one_of(st.none(), st_description_text)
st_uuid_str = st.uuids().map(str)
st_image_binary = st.one_of(st.none(), st.binary(min_size=0, max_size=256))  # Keep test images small
st_json_list_or_str = st.one_of(
    st.none(),
    st.lists(st_valid_name, max_size=3, unique=True),  # unique for sets/tags like behavior
    st.lists(st_valid_name, max_size=3, unique=True).map(json.dumps)  # pre-serialized JSON string
)
st_json_dict_or_str = st.one_of(
    st.none(),
    st.dictionaries(st_valid_name, st.one_of(st_valid_name, st.integers(), st.booleans(), st.none()), max_size=3),
    st.dictionaries(st_valid_name, st.one_of(st_valid_name, st.integers(), st.booleans(), st.none()), max_size=3).map(
        json.dumps)  # pre-serialized JSON string
)


def st_character_card_payload():
    return st.fixed_dictionaries({
        "description": st_optional_text,
        "personality": st_optional_text,
        "scenario": st_optional_text,
        "system_prompt": st_optional_text,
        "image": st_image_binary,
        "post_history_instructions": st_optional_text,
        "first_message": st_optional_text,
        "message_example": st_optional_text,
        "creator_notes": st_optional_text,
        "alternate_greetings": st_json_list_or_str,
        "tags": st_json_list_or_str,
        "creator": st_optional_text,
        "character_version": st_optional_text,
        "extensions": st_json_dict_or_str
    })


# --- Test Classes ---

class TestDBInitializationAndTransactions:
    def test_db_initialization_memory(self, db: CharactersRAGDB):
        assert db.is_memory_db
        assert db.client_id == TEST_CLIENT_ID
        # Schema version check
        conn = db.get_connection()
        version_row = conn.execute("SELECT version FROM db_schema_version WHERE schema_name = ?",
                                   (db._SCHEMA_NAME,)).fetchone()
        assert version_row is not None
        assert version_row['version'] == db._CURRENT_SCHEMA_VERSION

    def test_db_initialization_file(self, file_db: CharactersRAGDB):
        assert not file_db.is_memory_db
        assert file_db.db_path.exists()
        assert file_db.client_id == TEST_CLIENT_ID + "_file"
        conn = file_db.get_connection()
        version_row = conn.execute("SELECT version FROM db_schema_version WHERE schema_name = ?",
                                   (file_db._SCHEMA_NAME,)).fetchone()
        assert version_row is not None
        assert version_row['version'] == file_db._CURRENT_SCHEMA_VERSION

    def test_empty_client_id_raises_value_error(self):
        with pytest.raises(ValueError, match="Client ID cannot be empty or None."):
            CharactersRAGDB(":memory:", client_id="")
        with pytest.raises(ValueError, match="Client ID cannot be empty or None."):
            CharactersRAGDB(":memory:", client_id=None)  # type: ignore

    def test_transaction_commit(self, db: CharactersRAGDB):
        char_name = "CommitTestChar"
        try:
            with db.transaction() as conn:
                conn.execute("INSERT INTO character_cards (name, client_id, version) VALUES (?, ?, 1)",
                             (char_name, db.client_id))
            # Transaction committed
        except Exception as e:
            pytest.fail(f"Transaction failed unexpectedly: {e}")

        retrieved = db.get_character_card_by_name(char_name)
        assert retrieved is not None
        assert retrieved["name"] == char_name

    def test_transaction_rollback(self, db: CharactersRAGDB):
        char_name = "RollbackTestChar"

        # Add and commit "ConflictingName" *before* the main transaction block
        conflicting_name_val = "ConflictingName_RollbackTest"  # Make it unique for this test
        db.add_character_card({"name": conflicting_name_val})
        initial_count = len(db.list_character_cards())  # Should be 1 now
        assert initial_count == 1

        with pytest.raises(sqlite3.IntegrityError):
            with db.transaction() as conn:
                unique_char_name_for_rollback = f"{char_name}_{uuid.uuid4()}"
                conn.execute("INSERT INTO character_cards (name, client_id, version) VALUES (?, ?, 1)",
                             (unique_char_name_for_rollback, db.client_id))
                # This will cause the IntegrityError because conflicting_name_val already exists
                conn.execute("INSERT INTO character_cards (name, client_id, version) VALUES (?, ?, 1)",
                             (conflicting_name_val, db.client_id))

        # Transaction should have rolled back unique_char_name_for_rollback
        # The unique_char_name_for_rollback was never committed
        all_cards = db.list_character_cards()
        assert not any(c['name'] == unique_char_name_for_rollback for c in all_cards)

        # Only "ConflictingName_RollbackTest" should remain from what this test interacted with
        # Plus any cards added by other PBT examples if the DB isn't perfectly fresh (though it should be per function)
        # The initial_count already accounts for "ConflictingName_RollbackTest"
        assert len(db.list_character_cards()) == initial_count
        assert db.get_character_card_by_name(conflicting_name_val) is not None

    def test_close_connection_with_uncommitted_transaction_file_db(self, file_db: CharactersRAGDB, caplog):
        # Use file_db for this test
        db_path = file_db.db_path_str  # Get the path of the file DB

        conn = file_db.get_connection()
        conn.execute("BEGIN")
        conn.execute("INSERT INTO character_cards (name, client_id, version) VALUES (?, ?, 1)",
                     ("UncommittedCharFile", file_db.client_id))

        file_db.close_connection()  # This should attempt rollback and log

        assert "is in an uncommitted transaction during close. Attempting rollback." in caplog.text

        # Re-open the SAME file database
        db_reopened = CharactersRAGDB(db_path, client_id=TEST_CLIENT_ID_ALT)
        assert db_reopened.get_character_card_by_name("UncommittedCharFile") is None
        db_reopened.close_connection()


class TestCharacterCardAddition:
    def test_add_minimal_character(self, db: CharactersRAGDB):
        card_name = "Minimal Char"
        card_id = db.add_character_card({"name": card_name})
        assert card_id is not None
        assert isinstance(card_id, int)

        retrieved_card = db.get_character_card_by_id(card_id)
        assert retrieved_card is not None
        assert retrieved_card["name"] == card_name
        assert retrieved_card["description"] is None
        check_sync_log_entry(db, "character_cards", card_id, "create", expected_version=1)

    def test_add_full_character(self, db: CharactersRAGDB):
        data = sample_card_data(name="Full Char")
        card_id = db.add_character_card(data)
        assert card_id is not None

        retrieved = db.get_character_card_by_id(card_id)
        assert retrieved is not None
        for key, value in data.items():
            if key in ["alternate_greetings", "tags"]:
                assert set(retrieved[key] or []) == set(value or [])
            elif key == "extensions":
                assert retrieved[key] == value
            else:
                assert retrieved[key] == value
        check_sync_log_entry(db, "character_cards", card_id, "create", expected_version=1)

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
            db.add_character_card({"name": ""})

    def test_add_duplicate_name_raises_conflict_error(self, db: CharactersRAGDB):
        card_name = "Duplicate Name Char"
        db.add_character_card({"name": card_name})
        with pytest.raises(ConflictError, match=f"Character card with name '{card_name}' already exists."):
            db.add_character_card({"name": card_name})

    def test_add_character_with_list_json_fields(self, db: CharactersRAGDB):
        data = sample_card_data(
            name="ListJSON",
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
        alt_greetings_set = {"Hola", "Bonjour"}
        tags_set = {"tag_set1", "tag_set2"}
        data = sample_card_data(
            name="SetJSONChar",
            alternate_greetings=alt_greetings_set,  # type: ignore
            tags=tags_set  # type: ignore
        )
        card_id = db.add_character_card(data)
        retrieved = db.get_character_card_by_id(card_id)
        assert isinstance(retrieved["alternate_greetings"], list)
        assert set(retrieved["alternate_greetings"]) == alt_greetings_set
        assert isinstance(retrieved["tags"], list)
        assert set(retrieved["tags"]) == tags_set

    def test_add_character_with_string_json_fields(self, db: CharactersRAGDB):
        alt_greetings_json_str = '["Hola JSON", "Bonjour JSON"]'
        tags_json_str = '["tag_json_str1", "tag_json_str2"]'
        extensions_json_str = '{"key_json": "val_json"}'
        data = sample_card_data(
            name="StringJSONChar",
            alternate_greetings=alt_greetings_json_str,  # type: ignore
            tags=tags_json_str,  # type: ignore
            extensions=extensions_json_str  # type: ignore
        )
        card_id = db.add_character_card(data)
        retrieved = db.get_character_card_by_id(card_id)
        assert retrieved["alternate_greetings"] == json.loads(alt_greetings_json_str)
        assert retrieved["tags"] == json.loads(tags_json_str)
        assert retrieved["extensions"] == json.loads(extensions_json_str)

    def test_add_character_with_invalid_string_for_json_field_becomes_none(self, db: CharactersRAGDB, caplog):
        invalid_json_str = "this is not a valid json array"
        data = sample_card_data(name="InvalidStringJSONChar", tags=invalid_json_str)  # type: ignore
        card_id = db.add_character_card(data)

        conn = db.get_connection()
        raw_tags_row = conn.execute("SELECT tags FROM character_cards WHERE id = ?", (card_id,)).fetchone()
        assert raw_tags_row is not None
        assert raw_tags_row['tags'] == invalid_json_str

        retrieved = db.get_character_card_by_id(card_id)
        assert retrieved["tags"] is None
        assert f"Failed to decode JSON for field 'tags' in row (ID: {card_id})" in caplog.text
        check_sync_log_entry(db, "character_cards", card_id, "create", expected_version=1)

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

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.function_scoped_fixture], max_examples=50)
    @given(name=st_valid_name, card_details=st_character_card_payload())
    def test_pbt_add_character_card(self, db: CharactersRAGDB, name: str, card_details: dict):
        full_data = {"name": name, **card_details}

        # Ensure name uniqueness for this test example run
        # If tests run on the same DB instance across hypothesis examples, this is needed.
        # Pytest fixtures usually provide a fresh DB for each test *function*, not each *example*.
        # To handle this, either make names unique or clear DB, or skip if exists.
        # Simplest is to append a UUID to the name for PBT add tests.
        pbt_unique_name = f"{name}_{uuid.uuid4()}"
        full_data["name"] = pbt_unique_name

        card_id = db.add_character_card(full_data)
        assert card_id is not None
        retrieved = db.get_character_card_by_id(card_id)
        assert retrieved is not None
        assert retrieved["name"] == pbt_unique_name
        assert retrieved["version"] == 1
        assert retrieved["client_id"] == TEST_CLIENT_ID
        check_sync_log_entry(db, "character_cards", card_id, "create", expected_version=1)

        for key in ["alternate_greetings", "tags", "extensions"]:
            original_value = full_data.get(key)
            retrieved_value = retrieved.get(key)
            if isinstance(original_value, str):
                try:
                    expected_deserialized = json.loads(original_value)
                    assert retrieved_value == expected_deserialized
                except json.JSONDecodeError:
                    assert retrieved_value is None
            elif original_value is None:
                assert retrieved_value is None
            elif isinstance(original_value, list) and key in ["tags", "alternate_greetings"]:
                assert set(retrieved_value or []) == set(original_value or [])
            else:  # dicts for extensions
                assert retrieved_value == original_value


class TestCharacterCardRetrieval:
    char1_data: Dict
    char2_data: Dict
    char3_data: Dict
    char1_id: int
    char2_id: int

    @pytest.fixture
    def populated_db(self, db: CharactersRAGDB) -> CharactersRAGDB:
        self.char1_data = sample_card_data(name="Char One", description="First character", tags=["c1tag"])
        self.char2_data = sample_card_data(name="Char Two", description="Second character", tags=["alpha", "beta"])
        self.char3_data = sample_card_data(name="Char Three DELETED", description="Third, will be deleted")

        self.char1_id = db.add_character_card(self.char1_data)  # type: ignore
        self.char2_id = db.add_character_card(self.char2_data)  # type: ignore
        char3_id_temp = db.add_character_card(self.char3_data)
        assert char3_id_temp is not None
        db.soft_delete_character_card(char3_id_temp, expected_version=1)
        return db

    def test_get_character_by_id(self, populated_db: CharactersRAGDB):
        retrieved = populated_db.get_character_card_by_id(self.char1_id)
        assert retrieved is not None
        assert retrieved["name"] == self.char1_data["name"]
        assert retrieved["description"] == self.char1_data["description"]
        assert set(retrieved["tags"] or []) == set(self.char1_data["tags"] or [])

    def test_get_character_by_id_non_existent(self, db: CharactersRAGDB):
        assert db.get_character_card_by_id(99999) is None

    def test_get_character_by_id_soft_deleted(self, populated_db: CharactersRAGDB):
        char3_deleted_row = populated_db.execute_query("SELECT id FROM character_cards WHERE name = ? AND deleted = 1",
                                                       (self.char3_data["name"],)).fetchone()
        assert char3_deleted_row is not None, "Deleted char not found even with direct query"
        assert populated_db.get_character_card_by_id(char3_deleted_row["id"]) is None

    def test_get_character_by_name(self, populated_db: CharactersRAGDB):
        retrieved = populated_db.get_character_card_by_name(self.char2_data["name"])
        assert retrieved is not None
        assert retrieved["id"] == self.char2_id
        assert retrieved["description"] == self.char2_data["description"]
        assert set(retrieved["tags"] or []) == set(self.char2_data["tags"] or [])

    def test_get_character_by_name_non_existent(self, db: CharactersRAGDB):
        assert db.get_character_card_by_name("NonExistent Name") is None

    def test_get_character_by_name_soft_deleted(self, populated_db: CharactersRAGDB):
        assert populated_db.get_character_card_by_name(self.char3_data["name"]) is None

    def test_list_characters_empty(self, db: CharactersRAGDB):
        assert db.list_character_cards() == []

    def test_list_characters_populated(self, populated_db: CharactersRAGDB):
        cards = populated_db.list_character_cards()
        assert len(cards) == 2
        card_names = [c["name"] for c in cards]
        assert self.char1_data["name"] in card_names
        assert self.char2_data["name"] in card_names
        assert self.char3_data["name"] not in card_names

    def test_list_characters_pagination_and_order(self, populated_db: CharactersRAGDB):
        cards_limit1 = populated_db.list_character_cards(limit=1)
        assert len(cards_limit1) == 1
        # Order is by name. "Char One", "Char Two"
        assert cards_limit1[0]["name"] == "Char One"

        cards_offset1 = populated_db.list_character_cards(limit=1, offset=1)
        assert len(cards_offset1) == 1
        assert cards_offset1[0]["name"] == "Char Two"

        cards_all = populated_db.list_character_cards(limit=10)
        assert len(cards_all) == 2
        assert cards_all[0]["name"] < cards_all[1]["name"]


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
        time.sleep(0.01)
        assert db.update_character_card(char_id_for_update, update_payload, expected_version=original_version)

        updated_card = db.get_character_card_by_id(char_id_for_update)
        assert updated_card is not None
        assert updated_card["name"] == "Updated Name"
        assert updated_card["description"] == "Updated description."
        assert updated_card["image"] == b"new_image_data"
        assert set(updated_card["tags"] or []) == {"updated", "new_tag"}
        assert updated_card["extensions"] == {"new_key": "new_value", "setting": False}
        assert updated_card["version"] == original_version + 1
        assert updated_card["last_modified"] > original_last_modified
        check_sync_log_entry(db, "character_cards", char_id_for_update, "update", expected_version=original_version + 1)

    def test_update_character_no_actual_field_change_touches_record(self, db: CharactersRAGDB, char_id_for_update: int):
        original_card = db.get_character_card_by_id(char_id_for_update)
        assert original_card is not None
        original_version = original_card["version"]
        original_last_modified = original_card["last_modified"]

        update_payload = {"name": original_card["name"], "description": original_card["description"]}
        time.sleep(0.01)
        assert db.update_character_card(char_id_for_update, update_payload, expected_version=original_version)

        updated_card = db.get_character_card_by_id(char_id_for_update)
        assert updated_card is not None
        assert updated_card["version"] == original_version + 1
        assert updated_card["last_modified"] > original_last_modified
        check_sync_log_entry(db, "character_cards", char_id_for_update, "update", expected_version=original_version + 1)

    def test_update_character_version_mismatch(self, db: CharactersRAGDB, char_id_for_update: int):
        with pytest.raises(ConflictError, match="Update failed: version mismatch"):
            db.update_character_card(char_id_for_update, {"name": "Won't Update"}, expected_version=99)

    def test_update_character_non_existent(self, db: CharactersRAGDB):
        with pytest.raises(ConflictError, match="Record not found in character_cards."):
            db.update_character_card(99999, {"name": "No Char"}, expected_version=1)

    def test_update_character_soft_deleted(self, db: CharactersRAGDB, char_id_for_update: int):
        db.soft_delete_character_card(char_id_for_update, expected_version=1)
        with pytest.raises(ConflictError, match="Record is soft-deleted in character_cards."):
            db.update_character_card(char_id_for_update, {"name": "No Update"}, expected_version=2)

    def test_update_name_to_duplicate_conflict(self, db: CharactersRAGDB, char_id_for_update: int):
        existing_name = "Existing Unique Name"
        db.add_character_card({"name": existing_name})

        with pytest.raises(ConflictError,
                           match=f"Cannot update character card ID {char_id_for_update}: name '{existing_name}' already exists."):
            db.update_character_card(char_id_for_update, {"name": existing_name}, expected_version=1)

    def test_update_character_empty_data_is_noop_true(self, db: CharactersRAGDB, char_id_for_update: int):
        original_card = db.get_character_card_by_id(char_id_for_update)
        assert original_card is not None
        assert db.update_character_card(char_id_for_update, {}, expected_version=original_card["version"])

        after_noop_card = db.get_character_card_by_id(char_id_for_update)
        assert after_noop_card["version"] == original_card["version"]
        assert after_noop_card["last_modified"] == original_card["last_modified"]

    def test_update_character_with_only_ignored_fields_touches_record(self, db: CharactersRAGDB,
                                                                      char_id_for_update: int):
        original_card = db.get_character_card_by_id(char_id_for_update)
        assert original_card is not None
        original_version = original_card["version"]
        original_last_modified = original_card["last_modified"]
        update_payload = {"id": 999, "made_up_field": "value"}
        time.sleep(0.01)
        assert db.update_character_card(char_id_for_update, update_payload, expected_version=original_version)

        touched_card = db.get_character_card_by_id(char_id_for_update)
        assert touched_card["version"] == original_version + 1
        assert touched_card["last_modified"] > original_last_modified
        assert is_recent_iso_timestamp(touched_card["last_modified"])
        assert touched_card["name"] == original_card["name"]
        check_sync_log_entry(db, "character_cards", char_id_for_update, "update", expected_version=original_version + 1)

    @settings(deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.function_scoped_fixture], max_examples=50)
    @given(initial_name_suffix=st_valid_name, update_payload=st_character_card_payload())
    def test_pbt_update_character_card(self, db: CharactersRAGDB, initial_name_suffix: str, update_payload: dict):
        unique_initial_name = f"PBT_Update_{initial_name_suffix}_{uuid.uuid4()}"
        card_id = db.add_character_card({"name": unique_initial_name})
        assert card_id is not None
        original_card = db.get_character_card_by_id(card_id)
        assert original_card is not None

        # Handle potential name conflict in update_payload
        new_name_in_payload = update_payload.get("name")
        if new_name_in_payload:  # 'name' is not in st_character_card_payload, but good to be defensive
            pbt_update_name = f"{new_name_in_payload}_{uuid.uuid4()}"
            update_payload["name"] = pbt_update_name
            # If another card with pbt_update_name exists (unlikely with UUID)
            conflicting_card = db.get_character_card_by_name(pbt_update_name)
            if conflicting_card and conflicting_card["id"] != card_id:
                with pytest.raises(ConflictError):
                    db.update_character_card(card_id, update_payload, original_card["version"])
                return

        if not update_payload:
            assert db.update_character_card(card_id, update_payload, original_card["version"])
            noop_card = db.get_character_card_by_id(card_id)
            assert noop_card["version"] == original_card["version"]
            return

        assert db.update_character_card(card_id, update_payload, original_card["version"])
        updated_card = db.get_character_card_by_id(card_id)
        assert updated_card is not None
        assert updated_card["version"] == original_card["version"] + 1

        for key, value in update_payload.items():
            if key in updated_card:
                retrieved_value = updated_card[key]
                if isinstance(value, str) and key in db._CHARACTER_CARD_JSON_FIELDS:
                    try:
                        expected_deserialized = json.loads(value)
                        assert retrieved_value == expected_deserialized
                    except json.JSONDecodeError:
                        assert retrieved_value is None
                elif isinstance(value, list) and key in ["tags", "alternate_greetings"]:
                    assert set(retrieved_value or []) == set(value or [])
                else:
                    assert retrieved_value == value
        check_sync_log_entry(db, "character_cards", card_id, "update", expected_version=original_card["version"] + 1)


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
        original_last_modified = original_card["last_modified"]
        time.sleep(0.01)
        assert db.soft_delete_character_card(char_to_delete_id, expected_version=original_version)

        assert db.get_character_card_by_id(char_to_delete_id) is None
        assert db.get_character_card_by_name("Deletable Char") is None

        conn = db.get_connection()
        deleted_record_row = conn.execute("SELECT * FROM character_cards WHERE id = ?", (char_to_delete_id,)).fetchone()
        assert deleted_record_row is not None
        deleted_record = dict(deleted_record_row)

        assert deleted_record["deleted"] == 1
        assert deleted_record["version"] == original_version + 1
        assert is_recent_iso_timestamp(deleted_record["last_modified"])
        assert deleted_record["last_modified"] > original_last_modified
        check_sync_log_entry(db, "character_cards", char_to_delete_id, "delete", expected_version=original_version + 1)

    def test_soft_delete_character_version_mismatch(self, db: CharactersRAGDB, char_to_delete_id: int):
        with pytest.raises(ConflictError, match="Soft delete for Character ID .* failed: version mismatch"):
            db.soft_delete_character_card(char_to_delete_id, expected_version=99)

    def test_soft_delete_non_existent(self, db: CharactersRAGDB):
        with pytest.raises(ConflictError, match="Record not found in character_cards."):
            db.soft_delete_character_card(99999, expected_version=1)

    def test_soft_delete_already_deleted_idempotent(self, db: CharactersRAGDB, char_to_delete_id: int):
        db.soft_delete_character_card(char_to_delete_id, expected_version=1)

        assert db.soft_delete_character_card(char_to_delete_id, expected_version=2)

        conn = db.get_connection()
        deleted_record_row = conn.execute("SELECT version FROM character_cards WHERE id = ?",
                                          (char_to_delete_id,)).fetchone()
        assert deleted_record_row["version"] == 2


class TestCharacterCardSearch:
    @pytest.fixture
    def search_db(self, db: CharactersRAGDB) -> CharactersRAGDB:
        db.add_character_card(
            sample_card_data(name="Gandalf the Grey", description="A wise old wizard.", personality="Kind, powerful",
                             scenario="Fellowship journey"))
        db.add_character_card(sample_card_data(name="Aragorn Son of Arathorn", description="Ranger, heir of Isildur.",
                                               system_prompt="You are Strider."))
        db.add_character_card(sample_card_data(name="Bilbo Baggins", description="A hobbit who found a magic ring."))

        deleted_char_id = db.add_character_card(
            sample_card_data(name="Saruman the White", description="Corrupted wizard."))
        assert deleted_char_id is not None
        db.soft_delete_character_card(deleted_char_id, 1)
        return db

    def test_search_character_by_name(self, search_db: CharactersRAGDB):
        results = search_db.search_character_cards("Gandalf")
        assert len(results) == 1
        assert results[0]["name"] == "Gandalf the Grey"

    def test_search_character_by_description(self, search_db: CharactersRAGDB):
        results = search_db.search_character_cards("wizard")
        assert len(results) == 1
        assert results[0]["name"] == "Gandalf the Grey"

        results_ring = search_db.search_character_cards("ring")
        assert len(results_ring) == 1
        assert results_ring[0]["name"] == "Bilbo Baggins"

    def test_search_character_by_personality(self, search_db: CharactersRAGDB):
        results = search_db.search_character_cards("Kind")
        assert len(results) == 1
        assert results[0]["name"] == "Gandalf the Grey"

    def test_search_character_by_scenario(self, search_db: CharactersRAGDB):
        results = search_db.search_character_cards("journey")
        assert len(results) == 1
        assert results[0]["name"] == "Gandalf the Grey"

    def test_search_character_by_system_prompt(self, search_db: CharactersRAGDB):
        results = search_db.search_character_cards("Strider")
        assert len(results) == 1
        assert results[0]["name"] == "Aragorn Son of Arathorn"

    def test_search_character_multiple_terms(self, search_db: CharactersRAGDB):
        results = search_db.search_character_cards("Arathorn Ranger")
        assert len(results) == 1
        assert results[0]["name"] == "Aragorn Son of Arathorn"

        results_gandalf = search_db.search_character_cards("Grey wizard")
        assert len(results_gandalf) == 1
        assert results_gandalf[0]["name"] == "Gandalf the Grey"

    def test_search_character_no_results(self, search_db: CharactersRAGDB):
        results = search_db.search_character_cards("ZzzzzNonExistentTermZzzzz")
        assert len(results) == 0

    def test_search_character_limit_results(self, search_db: CharactersRAGDB):
        results_limit1 = search_db.search_character_cards(
            "description:wizard OR description:Ranger OR description:hobbit", limit=1)
        assert len(results_limit1) == 1

        results_limit2 = search_db.search_character_cards(
            "description:wizard OR description:Ranger OR description:hobbit", limit=2)
        assert len(results_limit2) == 2

    def test_search_character_excludes_soft_deleted(self, search_db: CharactersRAGDB):
        results = search_db.search_character_cards("Saruman")
        assert len(results) == 0

        results_corrupted = search_db.search_character_cards("Corrupted")
        assert len(results_corrupted) == 0


# --- NEW TEST CLASSES FOR OTHER ENTITIES ---

@pytest.fixture
def char_id_for_conv(db: CharactersRAGDB) -> int:
    """Provides a character_id to associate conversations with."""
    char_id = db.add_character_card(sample_card_data(name="ConvChar"))
    assert char_id is not None
    return char_id


class TestConversationCRUD:
    def test_add_conversation(self, db: CharactersRAGDB, char_id_for_conv: int):
        data = sample_conversation_data(character_id=char_id_for_conv, title="My First Chat")
        conv_id = db.add_conversation(data)
        assert conv_id is not None
        assert isinstance(conv_id, str)

        retrieved = db.get_conversation_by_id(conv_id)
        assert retrieved is not None
        assert retrieved["character_id"] == char_id_for_conv
        assert retrieved["title"] == "My First Chat"
        assert retrieved["version"] == 1
        assert retrieved["client_id"] == TEST_CLIENT_ID
        assert is_recent_iso_timestamp(retrieved["created_at"])
        check_sync_log_entry(db, "conversations", conv_id, "create", expected_version=1)

    def test_add_conversation_missing_char_id_fails(self, db: CharactersRAGDB):
        with pytest.raises(InputError, match="Required field 'character_id' is missing"):
            db.add_conversation({"title": "No Char Convo", "root_id": str(uuid.uuid4())})

    def test_add_conversation_invalid_char_id_fails(self, db: CharactersRAGDB):
        with pytest.raises(CharactersRAGDBError):  # Wraps sqlite3.IntegrityError
            db.add_conversation(sample_conversation_data(character_id=99999))

    def test_add_conversation_duplicate_id_fails(self, db: CharactersRAGDB, char_id_for_conv: int):
        conv_id = str(uuid.uuid4())
        db.add_conversation(sample_conversation_data(character_id=char_id_for_conv, id=conv_id, root_id=conv_id))
        with pytest.raises(ConflictError, match=f"Conversation with ID '{conv_id}' already exists."):
            db.add_conversation(sample_conversation_data(character_id=char_id_for_conv, id=conv_id, root_id=conv_id))

    def test_get_conversations_for_character(self, db: CharactersRAGDB, char_id_for_conv: int):
        db.add_conversation(sample_conversation_data(character_id=char_id_for_conv, title="C1"))
        db.add_conversation(sample_conversation_data(character_id=char_id_for_conv, title="C2"))

        convs = db.get_conversations_for_character(char_id_for_conv)
        assert len(convs) == 2

        other_char_id = db.add_character_card(sample_card_data(name="OtherChar"))
        assert other_char_id is not None
        db.add_conversation(sample_conversation_data(character_id=other_char_id, title="C3"))

        convs_char1 = db.get_conversations_for_character(char_id_for_conv)
        assert len(convs_char1) == 2
        convs_char2 = db.get_conversations_for_character(other_char_id)
        assert len(convs_char2) == 1
        assert convs_char2[0]["title"] == "C3"

    def test_update_conversation(self, db: CharactersRAGDB, char_id_for_conv: int):
        conv_id = db.add_conversation(sample_conversation_data(character_id=char_id_for_conv, title="Original Title"))
        assert conv_id
        original_conv = db.get_conversation_by_id(conv_id)
        assert original_conv

        update_payload = {"title": "Updated Chat Title", "rating": 5}
        time.sleep(0.01)
        assert db.update_conversation(conv_id, update_payload, original_conv["version"])

        updated_conv = db.get_conversation_by_id(conv_id)
        assert updated_conv is not None
        assert updated_conv["title"] == "Updated Chat Title"
        assert updated_conv["rating"] == 5
        assert updated_conv["version"] == original_conv["version"] + 1
        assert updated_conv["last_modified"] > original_conv["last_modified"]
        check_sync_log_entry(db, "conversations", conv_id, "update", expected_version=original_conv["version"] + 1)

    def test_update_conversation_invalid_rating(self, db: CharactersRAGDB, char_id_for_conv: int):
        conv_id = db.add_conversation(sample_conversation_data(character_id=char_id_for_conv))
        assert conv_id
        original_conv = db.get_conversation_by_id(conv_id)
        assert original_conv

        with pytest.raises(InputError, match="Rating must be between 1 and 5"):
            db.update_conversation(conv_id, {"rating": 0}, original_conv["version"])
        with pytest.raises(InputError, match="Rating must be between 1 and 5"):
            db.update_conversation(conv_id, {"rating": 6}, original_conv["version"])

    def test_soft_delete_conversation(self, db: CharactersRAGDB, char_id_for_conv: int):
        conv_id = db.add_conversation(sample_conversation_data(character_id=char_id_for_conv))
        assert conv_id
        original_conv = db.get_conversation_by_id(conv_id)
        assert original_conv

        assert db.soft_delete_conversation(conv_id, original_conv["version"])
        assert db.get_conversation_by_id(conv_id) is None
        check_sync_log_entry(db, "conversations", conv_id, "delete", expected_version=original_conv["version"] + 1)

        assert db.soft_delete_conversation(conv_id, original_conv["version"] + 1)

    def test_search_conversations_by_title(self, db: CharactersRAGDB, char_id_for_conv: int):
        db.add_conversation(sample_conversation_data(character_id=char_id_for_conv, title="Alpha Chat"))
        db.add_conversation(sample_conversation_data(character_id=char_id_for_conv, title="Beta Test Chat"))
        deleted_conv_id = db.add_conversation(
            sample_conversation_data(character_id=char_id_for_conv, title="Gamma Chat ToDelete"))
        assert deleted_conv_id
        db.soft_delete_conversation(deleted_conv_id, 1)

        results = db.search_conversations_by_title("Alpha")
        assert len(results) == 1
        assert results[0]["title"] == "Alpha Chat"

        results_chat = db.search_conversations_by_title("Chat")
        assert len(results_chat) == 2

        results_gamma = db.search_conversations_by_title("Gamma")
        assert len(results_gamma) == 0


@pytest.fixture
def conv_id_for_msg(db: CharactersRAGDB, char_id_for_conv: int) -> str:
    conv_id = db.add_conversation(sample_conversation_data(character_id=char_id_for_conv))
    assert conv_id
    return conv_id


class TestMessageCRUD:
    def test_add_message_text_only(self, db: CharactersRAGDB, conv_id_for_msg: str):
        data = sample_message_data(conversation_id=conv_id_for_msg, content="Hello world")
        msg_id = db.add_message(data)
        assert msg_id is not None

        retrieved = db.get_message_by_id(msg_id)
        assert retrieved is not None
        assert retrieved["conversation_id"] == conv_id_for_msg
        assert retrieved["content"] == "Hello world"
        assert retrieved["image_data"] is None
        assert retrieved["version"] == 1
        check_sync_log_entry(db, "messages", msg_id, "create", expected_version=1)

    def test_add_message_image_only(self, db: CharactersRAGDB, conv_id_for_msg: str):
        data = {
            "conversation_id": conv_id_for_msg,
            "sender": "user",
            "content": "",
            "image_data": b"test_image",
            "image_mime_type": "image/png"
        }
        msg_id = db.add_message(data)
        assert msg_id is not None

        retrieved = db.get_message_by_id(msg_id)
        assert retrieved is not None
        assert retrieved["image_data"] == b"test_image"
        assert retrieved["image_mime_type"] == "image/png"
        assert retrieved["content"] == ""
        check_sync_log_entry(db, "messages", msg_id, "create", expected_version=1)

    def test_add_message_text_and_image(self, db: CharactersRAGDB, conv_id_for_msg: str):
        data = sample_message_data(conversation_id=conv_id_for_msg, content="Look at this!",
                                   image_data=b"another_image", image_mime_type="image/jpeg")
        msg_id = db.add_message(data)
        assert msg_id is not None
        retrieved = db.get_message_by_id(msg_id)
        assert retrieved["content"] == "Look at this!"
        assert retrieved["image_data"] == b"another_image"
        check_sync_log_entry(db, "messages", msg_id, "create", expected_version=1)

    def test_add_message_no_content_or_image_fails(self, db: CharactersRAGDB, conv_id_for_msg: str):
        # Case 1: 'content' is None, 'image_data' is missing -> should hit the specific "must have text or image"
        with pytest.raises(InputError, match="Message must have text content or image data."):
            db.add_message({"conversation_id": conv_id_for_msg, "sender": "user", "content": None})  # type: ignore

        # Case 2: 'content' key is missing entirely -> should hit "Required field 'content' is missing"
        with pytest.raises(InputError, match="Required field 'content' is missing for message."):
            db.add_message({"conversation_id": conv_id_for_msg, "sender": "user"})

    def test_add_message_image_without_mime_type_fails(self, db: CharactersRAGDB, conv_id_for_msg: str):
        with pytest.raises(InputError, match="image_mime_type is required if image_data is provided."):
            db.add_message(
                {"conversation_id": conv_id_for_msg, "sender": "user", "content": "dummy", "image_data": b"img"})

    def test_add_message_to_deleted_conversation_fails(self, db: CharactersRAGDB, char_id_for_conv: int):
        deleted_conv_id = db.add_conversation(sample_conversation_data(character_id=char_id_for_conv))
        assert deleted_conv_id
        db.soft_delete_conversation(deleted_conv_id, 1)

        with pytest.raises(InputError,
                           match=f"Cannot add message: Conversation ID '{deleted_conv_id}' not found or deleted."):
            db.add_message(sample_message_data(conversation_id=deleted_conv_id))

    def test_get_messages_for_conversation_order(self, db: CharactersRAGDB, conv_id_for_msg: str):
        # Timestamps need to be distinct for reliable order testing
        # Ensure timestamps are ISO format strings
        ts1 = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat().replace('+00:00', 'Z')
        db.add_message(sample_message_data(conversation_id=conv_id_for_msg, content="Msg1", timestamp=ts1))
        time.sleep(0.02)  # Ensure different timestamps
        ts2 = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
        db.add_message(sample_message_data(conversation_id=conv_id_for_msg, content="Msg2", timestamp=ts2))

        msgs_asc = db.get_messages_for_conversation(conv_id_for_msg, order_by_timestamp="ASC")
        assert len(msgs_asc) == 2
        assert msgs_asc[0]["content"] == "Msg1"
        assert msgs_asc[1]["content"] == "Msg2"

        msgs_desc = db.get_messages_for_conversation(conv_id_for_msg, order_by_timestamp="DESC")
        assert len(msgs_desc) == 2
        assert msgs_desc[0]["content"] == "Msg2"
        assert msgs_desc[1]["content"] == "Msg1"

    def test_update_message_content(self, db: CharactersRAGDB, conv_id_for_msg: str):
        msg_id = db.add_message(sample_message_data(conversation_id=conv_id_for_msg, content="Original Content"))
        assert msg_id
        original_msg = db.get_message_by_id(msg_id)
        assert original_msg

        update_payload = {"content": "Updated Content", "ranking": 10}
        time.sleep(0.01)
        assert db.update_message(msg_id, update_payload, original_msg["version"])

        updated_msg = db.get_message_by_id(msg_id)
        assert updated_msg is not None
        assert updated_msg["content"] == "Updated Content"
        assert updated_msg["ranking"] == 10
        assert updated_msg["version"] == original_msg["version"] + 1
        check_sync_log_entry(db, "messages", msg_id, "update", expected_version=original_msg["version"] + 1)

    def test_update_message_image(self, db: CharactersRAGDB, conv_id_for_msg: str):
        msg_id = db.add_message(sample_message_data(conversation_id=conv_id_for_msg,
                                                    image_data=b"old_img", image_mime_type="image/png"))
        assert msg_id
        original_msg = db.get_message_by_id(msg_id)
        assert original_msg

        update_payload = {"image_data": b"new_img", "image_mime_type": "image/jpeg"}
        assert db.update_message(msg_id, update_payload, original_msg["version"])
        updated_msg = db.get_message_by_id(msg_id)
        assert updated_msg is not None
        assert updated_msg["image_data"] == b"new_img"
        assert updated_msg["image_mime_type"] == "image/jpeg"

        # Clear image
        updated_msg_v2 = db.get_message_by_id(msg_id)  # Get latest version
        assert updated_msg_v2 is not None
        assert db.update_message(msg_id, {"image_data": None}, updated_msg_v2["version"])
        cleared_msg = db.get_message_by_id(msg_id)
        assert cleared_msg is not None
        assert cleared_msg["image_data"] is None
        assert cleared_msg["image_mime_type"] is None

    def test_soft_delete_message(self, db: CharactersRAGDB, conv_id_for_msg: str):
        msg_id = db.add_message(sample_message_data(conversation_id=conv_id_for_msg))
        assert msg_id
        original_msg = db.get_message_by_id(msg_id)
        assert original_msg

        assert db.soft_delete_message(msg_id, original_msg["version"])
        assert db.get_message_by_id(msg_id) is None
        check_sync_log_entry(db, "messages", msg_id, "delete", expected_version=original_msg["version"] + 1)

    def test_search_messages_by_content(self, db: CharactersRAGDB, conv_id_for_msg: str, char_id_for_conv: int):
        db.add_message(sample_message_data(conversation_id=conv_id_for_msg, content="Unique keyword search test"))
        db.add_message(sample_message_data(conversation_id=conv_id_for_msg, content="Another message for testing"))

        results = db.search_messages_by_content("keyword")
        assert len(results) >= 1  # Could be more if other tests added "keyword"
        assert any(r["content"] == "Unique keyword search test" for r in results)

        other_conv_id = db.add_conversation(sample_conversation_data(character_id=char_id_for_conv))
        assert other_conv_id is not None
        db.add_message(sample_message_data(conversation_id=other_conv_id, content="Keyword in another convo"))

        results_specific_conv = db.search_messages_by_content("keyword", conversation_id=conv_id_for_msg)
        assert len(results_specific_conv) == 1
        assert results_specific_conv[0]["conversation_id"] == conv_id_for_msg

        results_all_keyword = db.search_messages_by_content("keyword")
        assert len(results_all_keyword) >= 2  # At least the two we added


class TestKeywordAndCollectionCRUD:
    def test_add_keyword(self, db: CharactersRAGDB):
        kw_id = db.add_keyword("TestKeyword")
        assert kw_id is not None
        retrieved = db.get_keyword_by_id(kw_id)
        assert retrieved is not None
        assert retrieved["keyword"] == "TestKeyword"
        check_sync_log_entry(db, "keywords", kw_id, "create", expected_version=1)

        original_log_count = len(db.get_sync_log_entries())
        db.soft_delete_keyword(kw_id, 1)
        check_sync_log_entry(db, "keywords", kw_id, "delete", expected_version=2)
        assert len(db.get_sync_log_entries()) == original_log_count + 1  # Ensure only one new log for delete

        time.sleep(0.01)
        new_kw_id = db.add_keyword("TestKeyword")
        assert new_kw_id == kw_id

        undeleted_kw = db.get_keyword_by_id(kw_id)
        assert undeleted_kw is not None
        assert undeleted_kw["deleted"] == 0
        assert undeleted_kw["version"] == 3
        # Undelete is an update, so check for "update" operation
        check_sync_log_entry(db, "keywords", kw_id, "update", expected_version=3)

    def test_add_keyword_collection(self, db: CharactersRAGDB):
        coll_id = db.add_keyword_collection("TestCollection")
        assert coll_id is not None
        retrieved = db.get_keyword_collection_by_id(coll_id)
        assert retrieved is not None
        assert retrieved["name"] == "TestCollection"
        check_sync_log_entry(db, "keyword_collections", coll_id, "create", expected_version=1)

        parent_id = coll_id
        child_coll_id = db.add_keyword_collection("ChildCollection", parent_id=parent_id)
        assert child_coll_id is not None
        child_retrieved = db.get_keyword_collection_by_id(child_coll_id)
        assert child_retrieved is not None
        assert child_retrieved["parent_id"] == parent_id

    def test_update_keyword_collection(self, db: CharactersRAGDB):
        coll_id = db.add_keyword_collection("OriginalCollName")
        assert coll_id
        original_coll = db.get_keyword_collection_by_id(coll_id)
        assert original_coll

        parent_coll_id = db.add_keyword_collection("ParentColl")
        assert parent_coll_id

        update_payload = {"name": "UpdatedCollName", "parent_id": parent_coll_id}
        assert db.update_keyword_collection(coll_id, update_payload, original_coll["version"])

        updated_coll = db.get_keyword_collection_by_id(coll_id)
        assert updated_coll is not None
        assert updated_coll["name"] == "UpdatedCollName"
        assert updated_coll["parent_id"] == parent_coll_id
        assert updated_coll["version"] == original_coll["version"] + 1
        check_sync_log_entry(db, "keyword_collections", coll_id, "update",
                             expected_version=original_coll["version"] + 1)

    def test_soft_delete_keyword(self, db: CharactersRAGDB):
        kw_id = db.add_keyword("ToDeleteKW")
        assert kw_id
        kw = db.get_keyword_by_id(kw_id)
        assert kw
        assert db.soft_delete_keyword(kw_id, kw["version"])
        assert db.get_keyword_by_id(kw_id) is None
        check_sync_log_entry(db, "keywords", kw_id, "delete", kw["version"] + 1)

    def test_soft_delete_keyword_collection(self, db: CharactersRAGDB):
        coll_id = db.add_keyword_collection("ToDeleteColl")
        assert coll_id
        coll = db.get_keyword_collection_by_id(coll_id)
        assert coll
        assert db.soft_delete_keyword_collection(coll_id, coll["version"])
        assert db.get_keyword_collection_by_id(coll_id) is None
        check_sync_log_entry(db, "keyword_collections", coll_id, "delete", coll["version"] + 1)

    def test_search_keywords(self, db: CharactersRAGDB):
        db.add_keyword("SearchableKey")
        db.add_keyword("AnotherKey")
        # Use prefix search for FTS
        results = db.search_keywords("Searchable*")
        assert len(results) == 1
        assert results[0]['keyword'] == "SearchableKey"

    def test_search_keyword_collections(self, db: CharactersRAGDB):
        db.add_keyword_collection("SearchableCollection")
        db.add_keyword_collection("AnotherCollection")
        # Use prefix search for FTS
        results = db.search_keyword_collections("SearchableColl*")
        assert len(results) == 1
        assert results[0]['name'] == "SearchableCollection"


class TestNoteCRUD:
    def test_add_note(self, db: CharactersRAGDB):
        note_data = sample_note_data(title="My Test Note")
        note_id = db.add_note(**note_data)
        assert note_id is not None

        retrieved = db.get_note_by_id(note_id)
        assert retrieved is not None
        assert retrieved["title"] == "My Test Note"
        assert retrieved["content"] == note_data["content"]
        assert retrieved["version"] == 1
        check_sync_log_entry(db, "notes", note_id, "create", expected_version=1)

    def test_update_note(self, db: CharactersRAGDB):
        note_id = db.add_note(**sample_note_data(title="Original Note Title"))
        assert note_id
        original_note = db.get_note_by_id(note_id)
        assert original_note

        update_payload = {"title": "Updated Note Title ", "content": "New content."}  # Title with trailing space
        assert db.update_note(note_id, update_payload, original_note["version"])

        updated_note = db.get_note_by_id(note_id)
        assert updated_note is not None
        assert updated_note["title"] == "Updated Note Title"  # Should be stripped
        assert updated_note["content"] == "New content."
        assert updated_note["version"] == original_note["version"] + 1
        check_sync_log_entry(db, "notes", note_id, "update", expected_version=original_note["version"] + 1)

    def test_soft_delete_note(self, db: CharactersRAGDB):
        note_id = db.add_note(**sample_note_data(title="Note to Delete"))
        assert note_id
        note = db.get_note_by_id(note_id)
        assert note
        assert db.soft_delete_note(note_id, note["version"])
        assert db.get_note_by_id(note_id) is None
        check_sync_log_entry(db, "notes", note_id, "delete", note["version"] + 1)

    def test_list_notes(self, db: CharactersRAGDB):
        db.add_note(title="NoteA", content="...")
        time.sleep(0.01)
        db.add_note(title="NoteB", content="...")
        notes = db.list_notes()
        assert len(notes) == 2
        assert notes[0]["title"] == "NoteB"  # Ordered by last_modified DESC

    def test_search_notes(self, db: CharactersRAGDB):
        db.add_note(title="AlphaNote", content="Content with unique_search_term_alpha")
        db.add_note(title="BetaNote", content="Content with unique_search_term_beta")
        results = db.search_notes("unique_search_term_alpha")
        assert len(results) == 1
        assert results[0]["title"] == "AlphaNote"
        results_title = db.search_notes("BetaNote")  # Search by title
        assert len(results_title) == 1
        assert results_title[0]["title"] == "BetaNote"


class TestLinkingTables:
    conv_id: str
    kw_id1: int
    kw_id2: int
    note_id: str
    coll_id: int

    @pytest.fixture
    def setup_for_linking(self, db: CharactersRAGDB, char_id_for_conv: int, conv_id_for_msg: str):
        self.conv_id = conv_id_for_msg
        kw_id1_opt = db.add_keyword("LinkKW1")
        assert kw_id1_opt is not None;
        self.kw_id1 = kw_id1_opt
        kw_id2_opt = db.add_keyword("LinkKW2")
        assert kw_id2_opt is not None;
        self.kw_id2 = kw_id2_opt
        note_id_opt = db.add_note(title="LinkNote", content="...")
        assert note_id_opt is not None;
        self.note_id = note_id_opt
        coll_id_opt = db.add_keyword_collection("LinkColl")
        assert coll_id_opt is not None;
        self.coll_id = coll_id_opt
        return db

    def test_link_unlink_conversation_keyword(self, setup_for_linking: CharactersRAGDB):
        db = setup_for_linking
        assert db.link_conversation_to_keyword(self.conv_id, self.kw_id1)
        linked_kws = db.get_keywords_for_conversation(self.conv_id)
        assert len(linked_kws) == 1 and linked_kws[0]["id"] == self.kw_id1
        check_sync_log_entry(db, "conversation_keywords", f"{self.conv_id}_{self.kw_id1}", "create", expected_version=1,
                             check_payload_details=False)

        assert not db.link_conversation_to_keyword(self.conv_id, self.kw_id1)

        assert db.unlink_conversation_from_keyword(self.conv_id, self.kw_id1)
        assert len(db.get_keywords_for_conversation(self.conv_id)) == 0
        check_sync_log_entry(db, "conversation_keywords", f"{self.conv_id}_{self.kw_id1}", "delete", expected_version=1,
                             check_payload_details=False)

        assert not db.unlink_conversation_from_keyword(self.conv_id, self.kw_id2)

    def test_link_unlink_note_keyword(self, setup_for_linking: CharactersRAGDB):
        db = setup_for_linking
        assert db.link_note_to_keyword(self.note_id, self.kw_id1)
        linked_kws = db.get_keywords_for_note(self.note_id)
        assert len(linked_kws) == 1 and linked_kws[0]["id"] == self.kw_id1
        check_sync_log_entry(db, "note_keywords", f"{self.note_id}_{self.kw_id1}", "create", expected_version=1,
                             check_payload_details=False)

        assert db.unlink_note_from_keyword(self.note_id, self.kw_id1)
        assert len(db.get_keywords_for_note(self.note_id)) == 0
        check_sync_log_entry(db, "note_keywords", f"{self.note_id}_{self.kw_id1}", "delete", expected_version=1,
                             check_payload_details=False)

    def test_link_unlink_collection_keyword(self, setup_for_linking: CharactersRAGDB):
        db = setup_for_linking
        assert db.link_collection_to_keyword(self.coll_id, self.kw_id1)
        linked_kws = db.get_keywords_for_collection(self.coll_id)
        assert len(linked_kws) == 1 and linked_kws[0]["id"] == self.kw_id1
        check_sync_log_entry(db, "collection_keywords", f"{self.coll_id}_{self.kw_id1}", "create", expected_version=1,
                             check_payload_details=False)

        assert db.unlink_collection_from_keyword(self.coll_id, self.kw_id1)
        assert len(db.get_keywords_for_collection(self.coll_id)) == 0
        check_sync_log_entry(db, "collection_keywords", f"{self.coll_id}_{self.kw_id1}", "delete", expected_version=1,
                             check_payload_details=False)

    def test_get_linked_items_with_deleted_target(self, setup_for_linking: CharactersRAGDB):
        db = setup_for_linking
        db.link_conversation_to_keyword(self.conv_id, self.kw_id1)
        db.link_conversation_to_keyword(self.conv_id, self.kw_id2)

        kw1 = db.get_keyword_by_id(self.kw_id1)
        assert kw1
        db.soft_delete_keyword(self.kw_id1, kw1["version"])

        linked_kws = db.get_keywords_for_conversation(self.conv_id)
        assert len(linked_kws) == 1
        assert linked_kws[0]["id"] == self.kw_id2


class TestSyncLog:
    def test_get_sync_log_entries_empty(self, db: CharactersRAGDB):
        logs = db.get_sync_log_entries()
        assert len(logs) == 0

    def test_get_sync_log_entries_populated(self, db: CharactersRAGDB):
        card_id = db.add_character_card(sample_card_data(name="LogChar1"))
        assert card_id

        logs = db.get_sync_log_entries()
        assert len(logs) == 1
        assert logs[0]["entity"] == "character_cards"
        assert logs[0]["entity_id"] == str(card_id)
        assert logs[0]["operation"] == "create"

        db.update_character_card(card_id, {"description": "new desc"}, 1)
        logs_after_update = db.get_sync_log_entries()
        assert len(logs_after_update) == 2
        assert logs_after_update[1]["operation"] == "update"
        assert logs_after_update[1]["version"] == 2

    def test_get_sync_log_entries_since_change_id(self, db: CharactersRAGDB):
        db.add_character_card(sample_card_data(name="LogCharA"))
        first_log_id = db.get_latest_sync_log_change_id()

        db.add_character_card(sample_card_data(name="LogCharB"))

        logs_since_first = db.get_sync_log_entries(since_change_id=first_log_id)
        assert len(logs_since_first) == 1
        assert logs_since_first[0]["payload"]["name"] == "LogCharB"

    def test_get_sync_log_entries_limit_and_entity_type(self, db: CharactersRAGDB, char_id_for_conv: int):
        # char_id_for_conv fixture already added one character "ConvChar"
        db.add_character_card(sample_card_data(name="C1"))  # Second char entry
        db.add_character_card(sample_card_data(name="C2"))  # Third char entry
        db.add_conversation(sample_conversation_data(character_id=char_id_for_conv, title="Conv1"))  # First conv entry

        all_logs_limit2 = db.get_sync_log_entries(limit=2)  # Gets first 2 logs overall
        assert len(all_logs_limit2) == 2
        # The first two logs will be for "ConvChar" and "C1"
        assert all_logs_limit2[0]["payload"]["name"] == "ConvChar"
        assert all_logs_limit2[1]["payload"]["name"] == "C1"

        char_logs = db.get_sync_log_entries(entity_type="character_cards")
        assert len(char_logs) == 3  # "ConvChar", "C1", "C2"
        assert all(log["entity"] == "character_cards" for log in char_logs)

        conv_logs = db.get_sync_log_entries(entity_type="conversations")
        assert len(conv_logs) == 1
        assert conv_logs[0]["entity"] == "conversations"
        assert conv_logs[0]["payload"]["title"] == "Conv1"

    def test_latest_sync_log_change_id(self, db: CharactersRAGDB):
        assert db.get_latest_sync_log_change_id() == 0

        db.add_character_card(sample_card_data(name="AnyChar"))
        latest_id = db.get_latest_sync_log_change_id()
        assert latest_id > 0

        conn = db.get_connection()
        max_id_raw_row = conn.execute("SELECT MAX(change_id) FROM sync_log").fetchone()
        assert max_id_raw_row is not None
        assert latest_id == max_id_raw_row[0]