# test_chacha_notes_db.py
#
#
# Imports
import pytest
import sqlite3
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import os  # For :memory: check
#
# Third-Party Imports
#
# Local Imports
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    SchemaError,
    InputError,
    ConflictError
)
#
#######################################################################################################################
#
# Functions:

# --- Fixtures ---

@pytest.fixture
def client_id():
    return "test_client_001"


@pytest.fixture
def db_path(tmp_path):
    """Provides a temporary path for the database file for each test."""
    return tmp_path / "test_db.sqlite"


@pytest.fixture
def db_instance(db_path, client_id):
    """Creates a DB instance for each test, ensuring a fresh database."""
    if db_path.exists():
        db_path.unlink()  # Ensure clean state if a previous test failed
    db = CharactersRAGDB(db_path, client_id)
    yield db
    db.close_connection()
    if db_path.exists():  # Clean up after test
        db_path.unlink()


@pytest.fixture
def mem_db_instance(client_id):
    """Creates an in-memory DB instance."""
    db = CharactersRAGDB(":memory:", client_id)
    yield db
    db.close_connection()


# --- Helper Functions ---
def get_current_utc_timestamp_iso():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def _create_sample_card_data(name_suffix="", client_id_override=None):
    return {
        "name": f"Test Character {name_suffix}",
        "description": "A test character.",
        "personality": "Testy",
        "scenario": "A test scenario.",
        "image": b"testimagebytes",
        "first_message": "Hello, test!",
        "alternate_greetings": json.dumps(["Hi", "Hey"]),  # Ensure JSON strings for direct use
        "tags": json.dumps(["test", "sample"]),
        "extensions": json.dumps({"custom_field": "value"}),
        "client_id": client_id_override  # For testing specific client_id scenarios
    }


# --- Test Cases ---

class TestDBInitialization:
    def test_db_creation(self, db_path, client_id):
        assert not db_path.exists()
        db = CharactersRAGDB(db_path, client_id)
        assert db_path.exists()
        assert db.client_id == client_id

        # Check schema version
        conn = db.get_connection()
        version = \
        conn.execute("SELECT version FROM db_schema_version WHERE schema_name = ?", (db._SCHEMA_NAME,)).fetchone()[
            'version']
        assert version == db._CURRENT_SCHEMA_VERSION
        db.close_connection()

    def test_in_memory_db(self, client_id):
        db = CharactersRAGDB(":memory:", client_id)
        assert db.is_memory_db
        assert db.client_id == client_id
        # Check schema version for in-memory
        conn = db.get_connection()
        version = \
        conn.execute("SELECT version FROM db_schema_version WHERE schema_name = ?", (db._SCHEMA_NAME,)).fetchone()[
            'version']
        assert version == db._CURRENT_SCHEMA_VERSION
        db.close_connection()

    def test_missing_client_id(self, db_path):
        with pytest.raises(ValueError, match="Client ID cannot be empty or None."):
            CharactersRAGDB(db_path, "")
        with pytest.raises(ValueError, match="Client ID cannot be empty or None."):
            CharactersRAGDB(db_path, None)

    def test_reopen_db(self, db_path, client_id):
        db1 = CharactersRAGDB(db_path, client_id)
        v1 = db1._get_db_version(db1.get_connection())
        db1.close_connection()

        db2 = CharactersRAGDB(db_path, "another_client")
        v2 = db2._get_db_version(db2.get_connection())
        assert v1 == v2
        assert v2 == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        db2.close_connection()

    def test_schema_newer_than_code(self, db_path, client_id):
        db = CharactersRAGDB(db_path, client_id)
        conn = db.get_connection()
        # Manually set a newer version
        conn.execute("UPDATE db_schema_version SET version = ? WHERE schema_name = ?",
                     (CharactersRAGDB._CURRENT_SCHEMA_VERSION + 1, CharactersRAGDB._SCHEMA_NAME))
        conn.commit()
        db.close_connection()

        with pytest.raises(SchemaError, match="is newer than supported by code"):
            CharactersRAGDB(db_path, client_id)


class TestCharacterCards:
    def test_add_character_card(self, db_instance: CharactersRAGDB):
        card_data = _create_sample_card_data("Add")
        card_id = db_instance.add_character_card(card_data)
        assert card_id is not None
        assert isinstance(card_id, int)

        retrieved = db_instance.get_character_card_by_id(card_id)
        assert retrieved is not None
        assert retrieved["name"] == card_data["name"]
        assert retrieved["description"] == card_data["description"]
        assert retrieved["image"] == card_data["image"]  # BLOB check
        assert isinstance(retrieved["alternate_greetings"], list)  # Check deserialization
        assert retrieved["alternate_greetings"] == json.loads(card_data["alternate_greetings"])
        assert retrieved["client_id"] == db_instance.client_id  # Ensure instance client_id is used
        assert retrieved["version"] == 1
        assert not retrieved["deleted"]

    def test_add_character_card_missing_name(self, db_instance: CharactersRAGDB):
        card_data = _create_sample_card_data("MissingName")
        del card_data["name"]
        with pytest.raises(InputError, match="Required field 'name' is missing"):
            db_instance.add_character_card(card_data)

    def test_add_character_card_duplicate_name(self, db_instance: CharactersRAGDB):
        card_data = _create_sample_card_data("Duplicate")
        db_instance.add_character_card(card_data)
        with pytest.raises(ConflictError, match=f"Character card with name '{card_data['name']}' already exists"):
            db_instance.add_character_card(card_data)

    def test_get_character_card_by_id_not_found(self, db_instance: CharactersRAGDB):
        assert db_instance.get_character_card_by_id(999) is None

    def test_get_character_card_by_name(self, db_instance: CharactersRAGDB):
        card_data = _create_sample_card_data("ByName")
        db_instance.add_character_card(card_data)
        retrieved = db_instance.get_character_card_by_name(card_data["name"])
        assert retrieved is not None
        assert retrieved["description"] == card_data["description"]

    def test_list_character_cards(self, db_instance: CharactersRAGDB):
        assert db_instance.list_character_cards() == []
        card_data1 = _create_sample_card_data("List1")
        card_data2 = _create_sample_card_data("List2")
        db_instance.add_character_card(card_data1)
        db_instance.add_character_card(card_data2)
        cards = db_instance.list_character_cards()
        assert len(cards) == 2
        assert cards[0]["name"] == card_data1["name"] or cards[1]["name"] == card_data1["name"]

    def test_update_character_card(self, db_instance: CharactersRAGDB):
        card_data = _create_sample_card_data("Update")
        card_id = db_instance.add_character_card(card_data)

        update_payload = {"description": "Updated Description", "personality": "More Testy"}
        updated = db_instance.update_character_card(card_id, update_payload)
        assert updated is True

        retrieved = db_instance.get_character_card_by_id(card_id)
        assert retrieved["description"] == "Updated Description"
        assert retrieved["personality"] == "More Testy"
        assert retrieved["name"] == card_data["name"]  # Unchanged
        assert retrieved["version"] == 2
        assert retrieved["client_id"] == db_instance.client_id  # Should be updated by the instance

    def test_update_character_card_version_conflict(self, db_instance: CharactersRAGDB, client_id):
        card_id = db_instance.add_character_card(_create_sample_card_data("VersionConflict"))

        # Simulate another client's update
        conn = db_instance.get_connection()
        conn.execute("UPDATE character_cards SET version = 2, client_id = 'other_client' WHERE id = ?", (card_id,))
        conn.commit()

        update_payload = {"description": "Conflict Update"}
        with pytest.raises(ConflictError, match="was modified by another client"):
            db_instance.update_character_card(card_id, update_payload)

    def test_update_character_card_not_found(self, db_instance: CharactersRAGDB):
        with pytest.raises(ConflictError, match="not found for update"):  # Raised by _get_record_version_and_bump
            db_instance.update_character_card(999, {"description": "Not Found"})

    def test_soft_delete_character_card(self, db_instance: CharactersRAGDB):
        card_id = db_instance.add_character_card(_create_sample_card_data("Delete"))

        deleted = db_instance.soft_delete_character_card(card_id)
        assert deleted is True

        retrieved = db_instance.get_character_card_by_id(card_id)
        assert retrieved is None  # Should not be found by normal get

        # Verify it's in DB but marked deleted
        conn = db_instance.get_connection()
        raw_retrieved = conn.execute("SELECT * FROM character_cards WHERE id = ?", (card_id,)).fetchone()
        assert raw_retrieved is not None
        assert raw_retrieved["deleted"] == 1
        assert raw_retrieved["version"] == 2  # Version bumped

        # Test idempotent delete
        assert db_instance.soft_delete_character_card(card_id) is True

    def test_search_character_cards(self, db_instance: CharactersRAGDB):
        card1_data = _create_sample_card_data("Searchable Alpha")
        card1_data["description"] = "Unique keyword: ZYX"
        card2_data = _create_sample_card_data("Searchable Beta")
        card2_data["system_prompt"] = "Contains ZYX too"
        card3_data = _create_sample_card_data("Unsearchable")
        db_instance.add_character_card(card1_data)
        db_instance.add_character_card(card2_data)
        db_instance.add_character_card(card3_data)

        results = db_instance.search_character_cards("ZYX")
        assert len(results) == 2
        names = [r["name"] for r in results]
        assert card1_data["name"] in names
        assert card2_data["name"] in names

        # Test search after delete
        card1_id = db_instance.get_character_card_by_name(card1_data["name"])["id"]
        db_instance.soft_delete_character_card(card1_id)
        results_after_delete = db_instance.search_character_cards("ZYX")
        assert len(results_after_delete) == 1
        assert results_after_delete[0]["name"] == card2_data["name"]


class TestConversationsAndMessages:
    @pytest.fixture
    def char_id(self, db_instance):
        return db_instance.add_character_card(_create_sample_card_data("ConvChar"))

    def test_add_conversation(self, db_instance: CharactersRAGDB, char_id):
        conv_data = {
            "id": str(uuid.uuid4()),
            "character_id": char_id,
            "title": "Test Conversation"
        }
        conv_id = db_instance.add_conversation(conv_data)
        assert conv_id == conv_data["id"]

        retrieved = db_instance.get_conversation_by_id(conv_id)
        assert retrieved is not None
        assert retrieved["title"] == "Test Conversation"
        assert retrieved["character_id"] == char_id
        assert retrieved["root_id"] == conv_id  # Default root_id
        assert retrieved["version"] == 1
        assert retrieved["client_id"] == db_instance.client_id

    def test_add_conversation_duplicate_id(self, db_instance: CharactersRAGDB, char_id):
        conv_id_val = str(uuid.uuid4())
        conv_data = {"id": conv_id_val, "character_id": char_id, "title": "First"}
        db_instance.add_conversation(conv_data)

        conv_data_dup = {"id": conv_id_val, "character_id": char_id, "title": "Duplicate"}
        with pytest.raises(ConflictError, match=f"Conversation with ID '{conv_id_val}' already exists"):
            db_instance.add_conversation(conv_data_dup)

    def test_add_message(self, db_instance: CharactersRAGDB, char_id):
        conv_id = db_instance.add_conversation({"character_id": char_id, "title": "MsgConv"})

        msg_data = {
            "conversation_id": conv_id,
            "sender": "user",
            "content": "Hello there!"
        }
        msg_id = db_instance.add_message(msg_data)
        assert msg_id is not None

        retrieved_msg = db_instance.get_message_by_id(msg_id)
        assert retrieved_msg is not None
        assert retrieved_msg["sender"] == "user"
        assert retrieved_msg["content"] == "Hello there!"
        assert retrieved_msg["conversation_id"] == conv_id
        assert retrieved_msg["version"] == 1
        assert retrieved_msg["client_id"] == db_instance.client_id

        # Test adding message to non-existent conversation
        msg_data_bad_conv = {
            "conversation_id": str(uuid.uuid4()),
            "sender": "user",
            "content": "Test"
        }
        with pytest.raises(InputError, match="Cannot add message: Conversation ID .* not found or deleted"):
            db_instance.add_message(msg_data_bad_conv)

    def test_get_messages_for_conversation_ordering(self, db_instance: CharactersRAGDB, char_id):
        conv_id = db_instance.add_conversation({"character_id": char_id, "title": "OrderedMsgConv"})
        msg1_id = db_instance.add_message(
            {"conversation_id": conv_id, "sender": "user", "content": "First", "timestamp": "2023-01-01T10:00:00Z"})
        msg2_id = db_instance.add_message(
            {"conversation_id": conv_id, "sender": "ai", "content": "Second", "timestamp": "2023-01-01T10:01:00Z"})

        messages_asc = db_instance.get_messages_for_conversation(conv_id, order_by_timestamp="ASC")
        assert len(messages_asc) == 2
        assert messages_asc[0]["id"] == msg1_id
        assert messages_asc[1]["id"] == msg2_id

        messages_desc = db_instance.get_messages_for_conversation(conv_id, order_by_timestamp="DESC")
        assert len(messages_desc) == 2
        assert messages_desc[0]["id"] == msg2_id
        assert messages_desc[1]["id"] == msg1_id

    def test_update_conversation(self, db_instance: CharactersRAGDB, char_id):
        conv_id = db_instance.add_conversation({"character_id": char_id, "title": "UpdateConv"})
        db_instance.update_conversation(conv_id, {"title": "New Title", "rating": 5})
        retrieved = db_instance.get_conversation_by_id(conv_id)
        assert retrieved["title"] == "New Title"
        assert retrieved["rating"] == 5
        assert retrieved["version"] == 2

    def test_soft_delete_conversation_and_messages(self, db_instance: CharactersRAGDB, char_id):
        # Setup: Conversation with messages
        conv_id = db_instance.add_conversation({"character_id": char_id, "title": "DeleteConv"})
        msg1_id = db_instance.add_message({"conversation_id": conv_id, "sender": "user", "content": "Msg1"})

        # Soft delete conversation
        db_instance.soft_delete_conversation(conv_id)
        assert db_instance.get_conversation_by_id(conv_id) is None

        # Messages should NOT be deleted by ON DELETE CASCADE with soft delete
        # They are still associated with the soft-deleted conversation
        msg1 = db_instance.get_message_by_id(msg1_id)
        assert msg1 is not None
        assert msg1["conversation_id"] == conv_id

        # FTS search for conversation should not find it
        results = db_instance.search_conversations_by_title("DeleteConv")
        assert len(results) == 0

    def test_search_messages_by_content_FIXED_JOIN(self, db_instance: CharactersRAGDB, char_id):
        # This test specifically validates the FTS join fix for messages (TEXT PK)
        conv_id = db_instance.add_conversation({"character_id": char_id, "title": "MessageSearchConv"})
        msg1_data = {"id": str(uuid.uuid4()), "conversation_id": conv_id, "sender": "user",
                     "content": "UniqueMessageContentAlpha"}
        msg2_data = {"id": str(uuid.uuid4()), "conversation_id": conv_id, "sender": "ai", "content": "Another phrase"}

        db_instance.add_message(msg1_data)
        db_instance.add_message(msg2_data)

        results = db_instance.search_messages_by_content("UniqueMessageContentAlpha")
        assert len(results) == 1
        assert results[0]["id"] == msg1_data["id"]
        assert results[0]["content"] == msg1_data["content"]

        # Test search within a specific conversation
        results_conv_specific = db_instance.search_messages_by_content("UniqueMessageContentAlpha",
                                                                       conversation_id=conv_id)
        assert len(results_conv_specific) == 1
        assert results_conv_specific[0]["id"] == msg1_data["id"]

        # Test search for content in another conversation (should not be found if conv_id is specified)
        other_conv_id = db_instance.add_conversation({"character_id": char_id, "title": "Other MessageSearchConv"})
        db_instance.add_message({"id": str(uuid.uuid4()), "conversation_id": other_conv_id, "sender": "user",
                                 "content": "UniqueMessageContentAlpha In Other"})

        results_other_conv = db_instance.search_messages_by_content("UniqueMessageContentAlpha",
                                                                    conversation_id=other_conv_id)
        assert len(results_other_conv) == 1
        assert results_other_conv[0]["content"] == "UniqueMessageContentAlpha In Other"

        results_original_conv_again = db_instance.search_messages_by_content("UniqueMessageContentAlpha",
                                                                             conversation_id=conv_id)
        assert len(results_original_conv_again) == 1
        assert results_original_conv_again[0]["id"] == msg1_data["id"]


class TestNotes:
    def test_add_note(self, db_instance: CharactersRAGDB):
        note_id = db_instance.add_note("Test Note Title", "This is the content of the note.")
        assert isinstance(note_id, str)  # UUID

        retrieved = db_instance.get_note_by_id(note_id)
        assert retrieved is not None
        assert retrieved["title"] == "Test Note Title"
        assert retrieved["content"] == "This is the content of the note."
        assert retrieved["version"] == 1
        assert not retrieved["deleted"]

    def test_add_note_empty_title(self, db_instance: CharactersRAGDB):
        with pytest.raises(InputError, match="Note title cannot be empty."):
            db_instance.add_note("", "Content")

    def test_add_note_duplicate_id(self, db_instance: CharactersRAGDB):
        fixed_id = str(uuid.uuid4())
        db_instance.add_note("First Note", "Content1", note_id=fixed_id)
        with pytest.raises(ConflictError, match=f"Note with ID '{fixed_id}' already exists."):
            db_instance.add_note("Second Note", "Content2", note_id=fixed_id)

    def test_update_note(self, db_instance: CharactersRAGDB):
        note_id = db_instance.add_note("Original Title", "Original Content")
        db_instance.update_note(note_id, {"title": "Updated Title", "content": "Updated Content"})

        retrieved = db_instance.get_note_by_id(note_id)
        assert retrieved["title"] == "Updated Title"
        assert retrieved["content"] == "Updated Content"
        assert retrieved["version"] == 2

    def test_list_notes(self, db_instance: CharactersRAGDB):
        assert db_instance.list_notes() == []
        id1 = db_instance.add_note("Note A", "Content A")
        id2 = db_instance.add_note("Note B", "Content B")
        notes = db_instance.list_notes()
        assert len(notes) == 2
        # Default order is last_modified DESC
        assert notes[0]["id"] == id2  # Note B was added last
        assert notes[1]["id"] == id1

    def test_search_notes(self, db_instance: CharactersRAGDB):
        db_instance.add_note("Alpha Note", "Contains a keyword ZYX")
        db_instance.add_note("Beta Note", "Another one with ZYX in title")
        db_instance.add_note("Gamma Note", "Nothing special")

        results = db_instance.search_notes("ZYX")
        assert len(results) == 2


class TestKeywordsAndCollections:
    def test_add_keyword(self, db_instance: CharactersRAGDB):
        keyword_id = db_instance.add_keyword("  TestKeyword  ")  # Test stripping
        assert keyword_id is not None
        retrieved = db_instance.get_keyword_by_id(keyword_id)
        assert retrieved["keyword"] == "TestKeyword"
        assert retrieved["version"] == 1

    def test_add_keyword_duplicate_active(self, db_instance: CharactersRAGDB):
        db_instance.add_keyword("UniqueKeyword")
        with pytest.raises(ConflictError, match="'UniqueKeyword' already exists and is active"):
            db_instance.add_keyword("UniqueKeyword")

    def test_add_keyword_undelete(self, db_instance: CharactersRAGDB):
        keyword_id = db_instance.add_keyword("ToDeleteAndReadd")
        db_instance.soft_delete_keyword(keyword_id)  # v2, deleted

        # Adding same keyword should undelete and update
        new_keyword_id = db_instance.add_keyword("ToDeleteAndReadd")
        assert new_keyword_id == keyword_id  # Should be the same ID

        retrieved = db_instance.get_keyword_by_id(keyword_id)
        assert retrieved is not None
        assert not retrieved["deleted"]
        assert retrieved["version"] == 3  # Initial add (v1), soft_delete (v2), undelete/update (v3)

    def test_add_keyword_collection(self, db_instance: CharactersRAGDB):
        coll_id = db_instance.add_keyword_collection("My Collection")
        assert coll_id is not None
        retrieved = db_instance.get_keyword_collection_by_id(coll_id)
        assert retrieved["name"] == "My Collection"
        assert retrieved["parent_id"] is None

        child_coll_id = db_instance.add_keyword_collection("Child Collection", parent_id=coll_id)
        retrieved_child = db_instance.get_keyword_collection_by_id(child_coll_id)
        assert retrieved_child["parent_id"] == coll_id

    def test_link_conversation_to_keyword(self, db_instance: CharactersRAGDB):
        char_id = db_instance.add_character_card(_create_sample_card_data("LinkChar"))
        conv_id = db_instance.add_conversation({"character_id": char_id, "title": "LinkConv"})
        kw_id = db_instance.add_keyword("Linkable")

        assert db_instance.link_conversation_to_keyword(conv_id, kw_id) is True
        keywords = db_instance.get_keywords_for_conversation(conv_id)
        assert len(keywords) == 1
        assert keywords[0]["id"] == kw_id

        # Test idempotency
        assert db_instance.link_conversation_to_keyword(conv_id, kw_id) is False  # Already linked

        # Test unlinking
        assert db_instance.unlink_conversation_from_keyword(conv_id, kw_id) is True
        assert len(db_instance.get_keywords_for_conversation(conv_id)) == 0
        assert db_instance.unlink_conversation_from_keyword(conv_id, kw_id) is False  # Already unlinked

    # Similar tests for other link types:
    # link_collection_to_keyword, link_note_to_keyword


class TestSyncLog:
    def test_sync_log_entry_on_add_character(self, db_instance: CharactersRAGDB):
        initial_log_count = len(db_instance.get_sync_log_entries())
        card_data = _create_sample_card_data("SyncLogChar")
        card_id = db_instance.add_character_card(card_data)

        log_entries = db_instance.get_sync_log_entries(since_change_id=0)  # Get all
        assert len(log_entries) > initial_log_count

        # Find the relevant log entry (usually the last one for a simple add)
        char_log_entry = None
        for entry in reversed(log_entries):
            if entry["entity"] == "character_cards" and entry["entity_id"] == str(card_id) and entry[
                "operation"] == "create":
                char_log_entry = entry
                break

        assert char_log_entry is not None
        assert char_log_entry["payload"]["name"] == card_data["name"]
        assert char_log_entry["payload"]["version"] == 1
        assert char_log_entry["client_id"] == db_instance.client_id

    def test_sync_log_entry_on_update_character(self, db_instance: CharactersRAGDB):
        card_id = db_instance.add_character_card(_create_sample_card_data("SyncUpdateChar"))
        latest_change_id = db_instance.get_latest_sync_log_change_id()

        db_instance.update_character_card(card_id, {"description": "Updated for Sync"})

        new_entries = db_instance.get_sync_log_entries(since_change_id=latest_change_id)
        assert len(new_entries) >= 1  # Could be more if other triggers fired

        update_log_entry = None
        for entry in new_entries:
            if entry["entity"] == "character_cards" and entry["entity_id"] == str(card_id) and entry[
                "operation"] == "update":
                update_log_entry = entry
                break

        assert update_log_entry is not None
        assert update_log_entry["payload"]["description"] == "Updated for Sync"
        assert update_log_entry["payload"]["version"] == 2  # Version bumped

    def test_sync_log_entry_on_soft_delete_character(self, db_instance: CharactersRAGDB):
        card_id = db_instance.add_character_card(_create_sample_card_data("SyncDeleteChar"))
        latest_change_id = db_instance.get_latest_sync_log_change_id()

        db_instance.soft_delete_character_card(card_id)

        new_entries = db_instance.get_sync_log_entries(since_change_id=latest_change_id)
        delete_log_entry = None
        for entry in new_entries:
            if entry["entity"] == "character_cards" and entry["entity_id"] == str(card_id) and entry[
                "operation"] == "delete":
                delete_log_entry = entry
                break

        assert delete_log_entry is not None
        assert delete_log_entry["payload"]["deleted"] == True  # Or 1, depending on JSON conversion
        assert delete_log_entry["payload"]["version"] == 2

    def test_sync_log_for_link_tables(self, db_instance: CharactersRAGDB):
        char_id = db_instance.add_character_card(_create_sample_card_data("SyncLinkChar"))
        conv_id = db_instance.add_conversation({"character_id": char_id, "title": "SyncLinkConv"})
        kw_id = db_instance.add_keyword("SyncLinkable")

        latest_change_id = db_instance.get_latest_sync_log_change_id()
        db_instance.link_conversation_to_keyword(conv_id, kw_id)

        new_entries = db_instance.get_sync_log_entries(since_change_id=latest_change_id)
        link_log_entry = None
        expected_entity_id = f"{conv_id}_{kw_id}"
        for entry in new_entries:
            if entry["entity"] == "conversation_keywords" and entry["entity_id"] == expected_entity_id and entry[
                "operation"] == "create":
                link_log_entry = entry
                break

        assert link_log_entry is not None
        assert link_log_entry["payload"]["conversation_id"] == conv_id
        assert link_log_entry["payload"]["keyword_id"] == kw_id

        latest_change_id_after_link = db_instance.get_latest_sync_log_change_id()
        db_instance.unlink_conversation_from_keyword(conv_id, kw_id)
        new_entries_unlink = db_instance.get_sync_log_entries(since_change_id=latest_change_id_after_link)
        unlink_log_entry = None
        for entry in new_entries_unlink:
            if entry["entity"] == "conversation_keywords" and entry["entity_id"] == expected_entity_id and entry[
                "operation"] == "delete":
                unlink_log_entry = entry
                break
        assert unlink_log_entry is not None


class TestTransactions:
    def test_transaction_commit(self, db_instance: CharactersRAGDB):
        card_data1 = _create_sample_card_data("Trans1")
        card_data2 = _create_sample_card_data("Trans2")

        with db_instance.transaction():
            id1 = db_instance.add_character_card(card_data1)  # Uses its own nested transaction, fine.
            # For a pure transaction test, we'd use conn.execute directly inside
            conn = db_instance.get_connection()  # Get connection from context
            conn.execute(
                "INSERT INTO character_cards (name, client_id, last_modified, version) VALUES (?, ?, ?, ?)",
                (card_data2['name'], db_instance.client_id, get_current_utc_timestamp_iso(), 1)
            )
            id2_name = card_data2['name']

        retrieved1 = db_instance.get_character_card_by_id(id1)
        retrieved2 = db_instance.get_character_card_by_name(id2_name)
        assert retrieved1 is not None
        assert retrieved2 is not None

    def test_transaction_rollback(self, db_instance: CharactersRAGDB):
        card_data = _create_sample_card_data("TransRollback")
        initial_count = len(db_instance.list_character_cards())

        with pytest.raises(sqlite3.IntegrityError):  # Or your custom ConflictError if name is already taken
            with db_instance.transaction() as conn:  # Get conn from context
                # First insert (will be part of transaction)
                conn.execute(
                    "INSERT INTO character_cards (name, client_id, last_modified, version) VALUES (?, ?, ?, ?)",
                    (card_data['name'], db_instance.client_id, get_current_utc_timestamp_iso(), 1)
                )
                # Second insert that causes an error (e.g., duplicate unique name)
                conn.execute(
                    "INSERT INTO character_cards (name, client_id, last_modified, version) VALUES (?, ?, ?, ?)",
                    (card_data['name'], db_instance.client_id, get_current_utc_timestamp_iso(), 1)
                )

        # Check that the first insert was rolled back
        assert len(db_instance.list_character_cards()) == initial_count
        assert db_instance.get_character_card_by_name(card_data["name"]) is None

# More tests can be added for:
# - Specific FTS trigger behavior (though search tests cover them indirectly)
# - Behavior of ON DELETE CASCADE / ON UPDATE CASCADE where applicable (e.g., true deletion of character should cascade to conversations IF hard delete was used and schema supported it)
# - More complex conflict scenarios with multiple clients (harder to simulate perfectly in unit tests without multiple DB instances writing to the same file).
# - All permutations of linking and unlinking for all link tables.
# - All specific error conditions for each method (e.g. InputError for various fields).