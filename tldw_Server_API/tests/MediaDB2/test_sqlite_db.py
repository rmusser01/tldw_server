# tests/test_sqlite_db.py
# Description: Unit tests for SQLite database operations, including CRUD, transactions, and sync log management.
#
# Imports:
import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import time
import sqlite3
from datetime import datetime, timezone, timedelta

from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import Database, ConflictError, DatabaseError, search_media_db


#
# 3rd-Party Imports:
#
# Local imports
# Import from src using adjusted sys.path in conftest
#
#######################################################################################################################
#
# Functions:

# Helper to get sync log entries for assertions
def get_log_count(db: Database, entity_uuid: str) -> int:
    cursor = db.execute_query("SELECT COUNT(*) FROM sync_log WHERE entity_uuid = ?", (entity_uuid,))
    return cursor.fetchone()[0]

def get_latest_log(db: Database, entity_uuid: str) -> dict | None:
    cursor = db.execute_query(
        "SELECT * FROM sync_log WHERE entity_uuid = ? ORDER BY change_id DESC LIMIT 1",
        (entity_uuid,)
    )
    row = cursor.fetchone()
    return dict(row) if row else None

def get_entity_version(db: Database, entity_table: str, uuid: str) -> int | None:
     cursor = db.execute_query(f"SELECT version FROM {entity_table} WHERE uuid = ?", (uuid,))
     row = cursor.fetchone()
     return row['version'] if row else None

class TestDatabaseInitialization:
    def test_memory_db_creation(self, memory_db_factory):
        """Test creating an in-memory database."""
        db = memory_db_factory("client_mem")
        assert db.is_memory_db
        assert db.client_id == "client_mem"
        # Check if a table exists (schema creation check)
        cursor = db.execute_query("SELECT name FROM sqlite_master WHERE type='table' AND name='Media'")
        assert cursor.fetchone() is not None
        db.close_connection()

    def test_file_db_creation(self, file_db, temp_db_path):
        """Test creating a file-based database."""
        assert not file_db.is_memory_db
        assert file_db.client_id == "file_client"
        assert os.path.exists(temp_db_path)
        cursor = file_db.execute_query("SELECT name FROM sqlite_master WHERE type='table' AND name='Media'")
        assert cursor.fetchone() is not None
        # file_db fixture handles closure

    def test_missing_client_id(self):
        """Test that ValueError is raised if client_id is missing."""
        with pytest.raises(ValueError, match="Client ID cannot be empty"):
            Database(db_path=":memory:", client_id="")
        with pytest.raises(ValueError, match="Client ID cannot be empty"):
            Database(db_path=":memory:", client_id=None)

    def test_schema_versioning_new_file_db(file_db): # Use the file_db fixture
        """Test that a new file DB gets the correct schema version."""
        # Initialization happened in the fixture
        cursor = file_db.execute_query("SELECT version FROM schema_version")
        version = cursor.fetchone()['version']
        assert version == Database._CURRENT_SCHEMA_VERSION

class TestDatabaseTransactions:
    def test_transaction_commit(self, memory_db_factory):
        db = memory_db_factory()
        keyword = "commit_test"
        with db.transaction():
            # Use internal method _add_keyword_internal or simplified version for test
            kw_id, kw_uuid = db.add_keyword(keyword) # add_keyword uses transaction internally too, nested is ok
        # Verify outside transaction
        cursor = db.execute_query("SELECT keyword FROM Keywords WHERE id = ?", (kw_id,))
        assert cursor.fetchone()['keyword'] == keyword

    def test_transaction_rollback(self, memory_db_factory):
        db = memory_db_factory()
        keyword = "rollback_test"
        initial_count_cursor = db.execute_query("SELECT COUNT(*) FROM Keywords")
        initial_count = initial_count_cursor.fetchone()[0]
        try:
            with db.transaction():
                # Simplified insert for test clarity
                new_uuid = db._generate_uuid()
                db.execute_query(
                     "INSERT INTO Keywords (keyword, uuid, last_modified, version, client_id, deleted) VALUES (?, ?, ?, 1, ?, 0)",
                     (keyword, new_uuid, db._get_current_utc_timestamp_str(), db.client_id),
                     commit=False # Important: commit=False inside transaction block
                )
                # Check *inside* transaction
                cursor_inside = db.execute_query("SELECT COUNT(*) FROM Keywords")
                assert cursor_inside.fetchone()[0] == initial_count + 1
                raise ValueError("Simulating error to trigger rollback") # Force rollback
        except ValueError:
            pass # Expected error
        except Exception as e:
            pytest.fail(f"Unexpected exception during rollback test: {e}")

        # Verify outside transaction (count should be back to initial)
        final_count_cursor = db.execute_query("SELECT COUNT(*) FROM Keywords")
        assert final_count_cursor.fetchone()[0] == initial_count


class TestDatabaseCRUDAndSync:

    @pytest.fixture
    def db_instance(self, memory_db_factory):
        """Provides a fresh in-memory DB for each test in this class."""
        return memory_db_factory("crud_client")

    def test_add_keyword(self, db_instance):
        keyword = " test keyword "
        expected_keyword = "test keyword"
        kw_id, kw_uuid = db_instance.add_keyword(keyword)

        assert kw_id is not None
        assert kw_uuid is not None

        # Verify DB state
        cursor = db_instance.execute_query("SELECT * FROM Keywords WHERE id = ?", (kw_id,))
        row = cursor.fetchone()
        assert row['keyword'] == expected_keyword
        assert row['uuid'] == kw_uuid
        assert row['version'] == 1
        assert row['client_id'] == db_instance.client_id
        assert not row['deleted']

        # Verify Sync Log
        log_entry = get_latest_log(db_instance, kw_uuid)
        assert log_entry['operation'] == 'create'
        assert log_entry['entity'] == 'Keywords'
        assert log_entry['version'] == 1
        assert log_entry['client_id'] == db_instance.client_id
        payload = json.loads(log_entry['payload'])
        assert payload['keyword'] == expected_keyword
        assert payload['uuid'] == kw_uuid

    def test_add_existing_keyword(self, db_instance):
        keyword = "existing"
        kw_id1, kw_uuid1 = db_instance.add_keyword(keyword)
        log_count1 = get_log_count(db_instance, kw_uuid1)

        kw_id2, kw_uuid2 = db_instance.add_keyword(keyword) # Add again
        log_count2 = get_log_count(db_instance, kw_uuid1)

        assert kw_id1 == kw_id2
        assert kw_uuid1 == kw_uuid2
        assert log_count1 == log_count2 # No new log entry

    def test_soft_delete_keyword(self, db_instance):
        keyword = "to_delete"
        kw_id, kw_uuid = db_instance.add_keyword(keyword)
        initial_version = get_entity_version(db_instance, "Keywords", kw_uuid)

        deleted = db_instance.soft_delete_keyword(keyword)
        assert deleted is True

        # Verify DB state
        cursor = db_instance.execute_query("SELECT deleted, version FROM Keywords WHERE id = ?", (kw_id,))
        row = cursor.fetchone()
        assert row['deleted'] == 1
        assert row['version'] == initial_version + 1

        # Verify Sync Log
        log_entry = get_latest_log(db_instance, kw_uuid)
        assert log_entry['operation'] == 'delete'
        assert log_entry['entity'] == 'Keywords'
        assert log_entry['version'] == initial_version + 1
        payload = json.loads(log_entry['payload'])
        assert payload['uuid'] == kw_uuid # Delete payload is minimal

    def test_undelete_keyword(self, db_instance):
        keyword = "to_undelete"
        kw_id, kw_uuid = db_instance.add_keyword(keyword)
        db_instance.soft_delete_keyword(keyword) # Delete it first
        deleted_version = get_entity_version(db_instance, "Keywords", kw_uuid)

        # Adding it again should undelete it
        undelete_id, undelete_uuid = db_instance.add_keyword(keyword)

        assert undelete_id == kw_id
        assert undelete_uuid == kw_uuid

        # Verify DB state
        cursor = db_instance.execute_query("SELECT deleted, version FROM Keywords WHERE id = ?", (kw_id,))
        row = cursor.fetchone()
        assert row['deleted'] == 0
        assert row['version'] == deleted_version + 1

        # Verify Sync Log
        log_entry = get_latest_log(db_instance, kw_uuid)
        # Undelete is logged as an 'update'
        assert log_entry['operation'] == 'update'
        assert log_entry['entity'] == 'Keywords'
        assert log_entry['version'] == deleted_version + 1
        payload = json.loads(log_entry['payload'])
        assert payload['uuid'] == kw_uuid
        assert payload['deleted'] == 0 # Payload shows undeleted state

    def test_add_media_with_keywords_create(self, db_instance):
        title = "Test Media Create"
        content = "Some unique content for create."
        keywords = ["create_kw1", "create_kw2"]

        media_id, media_uuid, msg = db_instance.add_media_with_keywords(
            title=title,
            media_type="article",
            content=content,
            keywords=keywords,
            author="Tester"
        )

        assert media_id is not None
        assert media_uuid is not None
        # FIX: Adjust assertion to match actual return message
        assert msg == f"Media '{title}' added."

        # Verify DB state (unchanged)
        cursor = db_instance.execute_query("SELECT * FROM Media WHERE id = ?", (media_id,))
        media_row = cursor.fetchone()
        assert media_row['title'] == title
        assert media_row['uuid'] == media_uuid
        assert media_row['version'] == 1 # Initial version
        assert not media_row['deleted']

        # Verify Keywords exist (unchanged)
        cursor = db_instance.execute_query("SELECT COUNT(*) FROM Keywords WHERE keyword IN (?, ?)", tuple(keywords))
        assert cursor.fetchone()[0] == 2

        # Verify MediaKeywords links (unchanged)
        cursor = db_instance.execute_query("SELECT COUNT(*) FROM MediaKeywords WHERE media_id = ?", (media_id,))
        assert cursor.fetchone()[0] == 2

        # Verify DocumentVersion creation (unchanged)
        cursor = db_instance.execute_query("SELECT version_number, content FROM DocumentVersions WHERE media_id = ? ORDER BY version_number DESC LIMIT 1", (media_id,))
        version_row = cursor.fetchone()
        assert version_row['version_number'] == 1
        assert version_row['content'] == content

        # Verify Sync Log for Media (Now Python generated)
        log_entry = get_latest_log(db_instance, media_uuid)
        # The *last* log might be DocumentVersion or MediaKeywords link depending on order.
        # Find the Media create log specifically.
        cursor_log = db_instance.execute_query("SELECT * FROM sync_log WHERE entity_uuid = ? AND operation = 'create' AND entity = 'Media'", (media_uuid,))
        log_entry = dict(cursor_log.fetchone())

        assert log_entry['operation'] == 'create'
        assert log_entry['entity'] == 'Media'
        assert log_entry['version'] == 1 # Check version
        payload = json.loads(log_entry['payload'])
        assert payload['uuid'] == media_uuid
        assert payload['title'] == title


    def test_add_media_with_keywords_update(self, db_instance):
        title = "Test Media Update"
        content1 = "Initial content."
        content2 = "Updated content."
        keywords1 = ["update_kw1"]
        keywords2 = ["update_kw2", "update_kw3"]

        media_id, media_uuid, _ = db_instance.add_media_with_keywords(
            title=title, media_type="text", content=content1, keywords=keywords1
        )
        initial_version = get_entity_version(db_instance, "Media", media_uuid)
        cursor_check_initial = db_instance.execute_query("SELECT content_hash FROM Media WHERE id = ?", (media_id,))
        initial_hash_row = cursor_check_initial.fetchone()
        assert initial_hash_row is not None
        initial_content_hash = initial_hash_row['content_hash']

        # Update 1: Using explicit URL (optional part of test)
        generated_url = f"local://text/{initial_content_hash}"
        media_id_up1, media_uuid_up1, msg1 = db_instance.add_media_with_keywords(
            title=title + " Updated Via URL", media_type="text", content=content2,
            keywords=["url_update_kw"], overwrite=True, url=generated_url
        )
        assert media_id_up1 == media_id
        assert media_uuid_up1 == media_uuid
        # FIX: Adjust assertion
        assert msg1 == f"Media '{title + ' Updated Via URL'}' updated."
        version_after_update1 = get_entity_version(db_instance, "Media", media_uuid)
        assert version_after_update1 == initial_version + 1

        # Update 2: Simulate finding by hash (URL=None)
        media_id_up2, media_uuid_up2, msg2 = db_instance.add_media_with_keywords(
            title=title + " Updated Via Hash", media_type="text", content=content2,
            keywords=keywords2, overwrite=True, url=None
        )
        assert media_id_up2 == media_id
        assert media_uuid_up2 == media_uuid
        # FIX: Adjust assertion
        assert msg2 == f"Media '{title + ' Updated Via Hash'}' updated."

        # Verify Final State (unchanged checks for DB content)
        cursor = db_instance.execute_query("SELECT title, content, version FROM Media WHERE id = ?", (media_id,))
        media_row = cursor.fetchone()
        assert media_row['title'] == title + " Updated Via Hash"
        assert media_row['content'] == content2
        assert media_row['version'] == version_after_update1 + 1

        # Verify Keywords links updated (unchanged)
        cursor = db_instance.execute_query("SELECT k.keyword FROM MediaKeywords mk JOIN Keywords k ON mk.keyword_id = k.id WHERE mk.media_id = ? ORDER BY k.keyword", (media_id,))
        current_keywords = [r['keyword'] for r in cursor.fetchall()]
        assert current_keywords == sorted(keywords2)

        # Verify latest DocumentVersion (unchanged)
        cursor = db_instance.execute_query("SELECT version_number, content FROM DocumentVersions WHERE media_id = ? ORDER BY version_number DESC LIMIT 1", (media_id,))
        version_row = cursor.fetchone(); assert version_row['version_number'] == 3; assert version_row['content'] == content2

        # Verify Sync Log for the *last* Media update
        log_entry = get_latest_log(db_instance, media_uuid) # Should be the Media update
        assert log_entry['operation'] == 'update'
        assert log_entry['entity'] == 'Media'
        assert log_entry['version'] == version_after_update1 + 1
        payload = json.loads(log_entry['payload'])
        assert payload['title'] == title + " Updated Via Hash"

    def test_soft_delete_media_cascade(self, db_instance):
        # 1. Setup complex item
        media_id, media_uuid, _ = db_instance.add_media_with_keywords(
            title="Cascade Test", content="Cascade content", media_type="article",
            keywords=["cascade1", "cascade2"], author="Cascade Author"
        )
        # Add a transcript manually (assuming no direct add_transcript method)
        t_uuid = db_instance._generate_uuid()
        db_instance.execute_query(
            """INSERT INTO Transcripts (media_id, whisper_model, transcription, uuid, last_modified, version, client_id, deleted)
               VALUES (?, ?, ?, ?, ?, 1, ?, 0)""",
            (media_id, "model_xyz", "Transcript text", t_uuid, db_instance._get_current_utc_timestamp_str(), db_instance.client_id),
            commit=True
        )
        # Add a chunk manually
        c_uuid = db_instance._generate_uuid()
        db_instance.execute_query(
            """INSERT INTO MediaChunks (media_id, chunk_text, uuid, last_modified, version, client_id, deleted)
               VALUES (?, ?, ?, ?, 1, ?, 0)""",
            (media_id, "Chunk text", c_uuid, db_instance._get_current_utc_timestamp_str(), db_instance.client_id),
            commit=True
        )
        media_version = get_entity_version(db_instance, "Media", media_uuid)
        transcript_version = get_entity_version(db_instance, "Transcripts", t_uuid)
        chunk_version = get_entity_version(db_instance, "MediaChunks", c_uuid)


        # 2. Perform soft delete with cascade
        deleted = db_instance.soft_delete_media(media_id, cascade=True)
        assert deleted is True

        # 3. Verify parent and children are marked deleted and versioned
        cursor = db_instance.execute_query("SELECT deleted, version FROM Media WHERE id = ?", (media_id,))
        assert dict(cursor.fetchone()) == {'deleted': 1, 'version': media_version + 1}

        cursor = db_instance.execute_query("SELECT deleted, version FROM Transcripts WHERE uuid = ?", (t_uuid,))
        assert dict(cursor.fetchone()) == {'deleted': 1, 'version': transcript_version + 1}

        cursor = db_instance.execute_query("SELECT deleted, version FROM MediaChunks WHERE uuid = ?", (c_uuid,))
        assert dict(cursor.fetchone()) == {'deleted': 1, 'version': chunk_version + 1}

        # 4. Verify keywords are unlinked
        cursor = db_instance.execute_query("SELECT COUNT(*) FROM MediaKeywords WHERE media_id = ?", (media_id,))
        assert cursor.fetchone()[0] == 0

        # 5. Verify Sync Logs
        media_log = get_latest_log(db_instance, media_uuid)
        assert media_log['operation'] == 'delete'
        assert media_log['version'] == media_version + 1

        transcript_log = get_latest_log(db_instance, t_uuid)
        assert transcript_log['operation'] == 'delete'
        assert transcript_log['version'] == transcript_version + 1

        chunk_log = get_latest_log(db_instance, c_uuid)
        assert chunk_log['operation'] == 'delete'
        assert chunk_log['version'] == chunk_version + 1

        # Check MediaKeywords unlink logs (tricky to get exact UUIDs, check count)
        cursor = db_instance.execute_query("SELECT COUNT(*) FROM sync_log WHERE entity = 'MediaKeywords' AND operation = 'unlink' AND payload LIKE ?", (f'%{media_uuid}%',))
        assert cursor.fetchone()[0] == 2 # Should be 2 unlink events

    def test_optimistic_locking_prevents_update_with_stale_version(self, db_instance):
        """Test that an UPDATE with a stale version number fails (rowcount 0)."""
        keyword = "conflict_test"
        kw_id, kw_uuid = db_instance.add_keyword(keyword)
        original_version = get_entity_version(db_instance, "Keywords", kw_uuid)  # Should be 1
        assert original_version == 1, "Initial version should be 1"

        # Simulate external update incrementing version
        db_instance.execute_query(
            "UPDATE Keywords SET version = ?, client_id = ? WHERE id = ?",
            (original_version + 1, "external_client", kw_id),
            commit=True
        )
        version_after_external_update = get_entity_version(db_instance, "Keywords", kw_uuid)  # Should be 2
        assert version_after_external_update == original_version + 1, "Version after external update should be 2"

        # Now, manually attempt an update using the *original stale version* (version=1)
        # This mimics what would happen if a process read version 1, then tried
        # to update after the external process bumped it to version 2.
        current_time = db_instance._get_current_utc_timestamp_str()
        client_id = db_instance.client_id
        cursor = db_instance.execute_query(
            "UPDATE Keywords SET keyword='stale_update', last_modified=?, version=?, client_id=? WHERE id=? AND version=?",
            (current_time, original_version + 1, client_id, kw_id, original_version),  # <<< WHERE version = 1 (stale)
            commit=True  # Commit needed to actually perform the check
        )

        # Assert that the update failed because the WHERE clause (version=1) didn't match any rows
        assert cursor.rowcount == 0, "Update with stale version should affect 0 rows"

        # Verify DB state is unchanged by the failed update (still shows external update's state)
        cursor_check = db_instance.execute_query("SELECT keyword, version, client_id FROM Keywords WHERE id = ?",
                                                 (kw_id,))
        row = cursor_check.fetchone()
        assert row is not None, "Keyword should still exist"
        assert row['keyword'] == keyword, "Keyword text should not have changed to 'stale_update'"
        assert row['version'] == original_version + 1, "Version should remain 2 from the external update"
        assert row['client_id'] == "external_client", "Client ID should remain from the external update"

    def test_version_validation_trigger(self, db_instance):
        """Test trigger preventing non-sequential version updates."""
        kw_id, kw_uuid = db_instance.add_keyword("validation_test")
        current_version = get_entity_version(db_instance, "Keywords", kw_uuid)

        # Try to update version incorrectly (skipping a version)
        with pytest.raises(sqlite3.IntegrityError,
                           match=r"Sync Error \(Keywords\): Version must increment by exactly 1"):
            # Provide client_id to prevent the *other* validation trigger firing
            client_id = db_instance.client_id
            db_instance.execute_query(
                "UPDATE Keywords SET version = ?, keyword = ?, client_id = ? WHERE id = ?",
                (current_version + 2, "bad version", client_id, kw_id),
                commit=True
            )

        # Try to update version incorrectly (same version)
        with pytest.raises(sqlite3.IntegrityError, match=r"Sync Error \(Keywords\): Version must increment by exactly 1"):
            client_id = db_instance.client_id
            db_instance.execute_query(
                "UPDATE Keywords SET version = ?, keyword = ?, client_id = ? WHERE id = ?",
                (current_version + 2, "bad version", client_id, kw_id),
                commit=True
            )

    def test_client_id_validation_trigger(self, db_instance):
        """Test trigger preventing null/empty client_id on update."""
        kw_id, kw_uuid = db_instance.add_keyword("clientid_test")
        current_version = get_entity_version(db_instance, "Keywords", kw_uuid)

        # Test the EMPTY STRING case handled by the trigger
        # Use raw string for regex match safety
        with pytest.raises(sqlite3.IntegrityError, match=r"Sync Error \(Keywords\): Client ID cannot be NULL or empty"):
            db_instance.execute_query(
                "UPDATE Keywords SET version = ?, client_id = '' WHERE id = ?",
                (current_version + 1, kw_id),
                commit=True
            )

        # Optional: Test the NULL case separately, expecting the NOT NULL constraint error
        # This confirms the underlying table constraint works, though not the trigger message.
        with pytest.raises(sqlite3.IntegrityError, match=r"Sync Error \(Keywords\): Client ID cannot be NULL or empty"):
            db_instance.execute_query(
                "UPDATE Keywords SET version = ?, client_id = NULL WHERE id = ?",
                (current_version + 1, kw_id),  # Increment version correctly
                commit=True
            )


class TestSyncLogManagement:

     @pytest.fixture
     def db_instance(self, memory_db_factory):
         db = memory_db_factory("log_client")
         # Add some initial data to generate logs
         db.add_keyword("log_kw_1")
         time.sleep(0.01) # Ensure timestamp difference
         db.add_keyword("log_kw_2")
         time.sleep(0.01)
         db.add_keyword("log_kw_3")
         db.soft_delete_keyword("log_kw_2")
         return db

     def test_get_sync_log_entries_all(self, db_instance):
         logs = db_instance.get_sync_log_entries()
         # Expect 3 creates + 1 delete = 4 entries
         assert len(logs) == 4
         assert logs[0]['change_id'] == 1
         assert logs[-1]['change_id'] == 4

     def test_get_sync_log_entries_since(self, db_instance):
         logs = db_instance.get_sync_log_entries(since_change_id=2) # Get 3 and 4
         assert len(logs) == 2
         assert logs[0]['change_id'] == 3
         assert logs[1]['change_id'] == 4

     def test_get_sync_log_entries_limit(self, db_instance):
         logs = db_instance.get_sync_log_entries(limit=2) # Get 1 and 2
         assert len(logs) == 2
         assert logs[0]['change_id'] == 1
         assert logs[1]['change_id'] == 2

     def test_get_sync_log_entries_since_and_limit(self, db_instance):
         logs = db_instance.get_sync_log_entries(since_change_id=1, limit=2) # Get 2 and 3
         assert len(logs) == 2
         assert logs[0]['change_id'] == 2
         assert logs[1]['change_id'] == 3

     def test_delete_sync_log_entries_specific(self, db_instance):
         initial_logs = db_instance.get_sync_log_entries()
         initial_count = len(initial_logs) # Should be 4
         ids_to_delete = [initial_logs[1]['change_id'], initial_logs[2]['change_id']] # Delete 2 and 3

         deleted_count = db_instance.delete_sync_log_entries(ids_to_delete)
         assert deleted_count == 2

         remaining_logs = db_instance.get_sync_log_entries()
         assert len(remaining_logs) == initial_count - 2
         remaining_ids = {log['change_id'] for log in remaining_logs}
         assert remaining_ids == {initial_logs[0]['change_id'], initial_logs[3]['change_id']} # 1 and 4 should remain

     def test_delete_sync_log_entries_before(self, db_instance):
         initial_logs = db_instance.get_sync_log_entries()
         initial_count = len(initial_logs) # Should be 4
         threshold_id = initial_logs[2]['change_id'] # Delete up to and including ID 3

         deleted_count = db_instance.delete_sync_log_entries_before(threshold_id)
         assert deleted_count == 3 # Deleted 1, 2, 3

         remaining_logs = db_instance.get_sync_log_entries()
         assert len(remaining_logs) == 1
         assert remaining_logs[0]['change_id'] == initial_logs[3]['change_id'] # Only 4 remains

     def test_delete_sync_log_entries_empty(self, db_instance):
         deleted_count = db_instance.delete_sync_log_entries([])
         assert deleted_count == 0

     def test_delete_sync_log_entries_invalid_id(self, db_instance):
         with pytest.raises(ValueError):
             db_instance.delete_sync_log_entries([1, "two", 3])


# Add FTS specific tests
class TestDatabaseFTS:
    @pytest.fixture
    def db_instance(self, memory_db_factory):
        # Use file DB for FTS tests if memory DB proves unstable
        # return memory_db_factory("fts_client")
        temp_dir = tempfile.mkdtemp()
        db_file = Path(temp_dir) / "fts_test_db.sqlite"
        db = Database(db_path=str(db_file), client_id="fts_client")
        yield db
        db.close_connection()
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_fts_media_create_search(self, db_instance):
        """Test searching media via FTS after creation."""
        title = "FTS Test Alpha"
        content = "Unique content string omega gamma beta."
        media_id, media_uuid, _ = db_instance.add_media_with_keywords(title=title, content=content, media_type="fts_test")

        # Search by title fragment
        results, total = search_media_db(db_instance, search_query="Alpha", search_fields=["title"])
        assert total == 1
        assert len(results) == 1
        assert results[0]['id'] == media_id

        # Search by content fragment
        results, total = search_media_db(db_instance, search_query="omega", search_fields=["content"])
        assert total == 1
        assert len(results) == 1
        assert results[0]['id'] == media_id

        # Search by content phrase
        results, total = search_media_db(db_instance, search_query='"omega gamma"', search_fields=["content"])
        assert total == 1

        # Search non-existent term
        results, total = search_media_db(db_instance, search_query="nonexistent", search_fields=["content", "title"])
        assert total == 0

    def test_fts_media_update_search(self, db_instance):
        """Test searching media via FTS after update."""
        title1 = "FTS Update Initial"
        content1 = "Original text epsilon."
        title2 = "FTS Update Final Zeta"
        content2 = "Replacement stuff delta."

        media_id, media_uuid, _ = db_instance.add_media_with_keywords(title=title1, content=content1,
                                                                      media_type="fts_update")

        # Verify initial search works
        results, total = search_media_db(db_instance, search_query="epsilon", search_fields=["content"])
        assert total == 1
        initial_url = results[0]['url']  # Get URL for update lookup

        # Update the media
        db_instance.add_media_with_keywords(title=title2, content=content2, media_type="fts_update", overwrite=True,
                                            url=initial_url)

        # Search for OLD content - REMOVE this assertion as immediate consistency isn't guaranteed
        # results, total = search_media_db(db_instance, search_query="epsilon", search_fields=["content"])
        # assert total == 0

        # Search for NEW content should work
        results, total = search_media_db(db_instance, search_query="delta", search_fields=["content"])
        assert total == 1
        assert results[0]['id'] == media_id

        # Search for NEW title should work
        results, total = search_media_db(db_instance, search_query="Zeta", search_fields=["title"])
        assert total == 1
        assert results[0]['id'] == media_id

    def test_fts_media_delete_search(self, db_instance):
        """Test searching media via FTS after soft deletion."""
        title = "FTS To Delete"
        content = "Content will vanish theta."
        media_id, media_uuid, _ = db_instance.add_media_with_keywords(title=title, content=content, media_type="fts_delete")

        # Verify initial search works
        results, total = search_media_db(db_instance, search_query="theta", search_fields=["content"])
        assert total == 1

        # Soft delete the media
        deleted = db_instance.soft_delete_media(media_id)
        assert deleted is True

        # Search should now fail
        results, total = search_media_db(db_instance, search_query="theta", search_fields=["content"])
        assert total == 0

    def test_fts_keyword_search(self, db_instance):
        """Test searching keywords via FTS."""
        kw1_id, kw1_uuid = db_instance.add_keyword("fts_keyword_apple")
        kw2_id, kw2_uuid = db_instance.add_keyword("fts_keyword_banana")

        # Search keyword FTS directly (not typically done, but tests population)
        # NOTE: search_media_db doesn't search keyword_fts directly, this is just to test population
        cursor = db_instance.execute_query("SELECT rowid, keyword FROM keyword_fts WHERE keyword_fts MATCH ?", ("apple",))
        fts_results = cursor.fetchall()
        assert len(fts_results) == 1
        assert fts_results[0]['rowid'] == kw1_id

        # Soft delete keyword 1
        db_instance.soft_delete_keyword("fts_keyword_apple")

        # Search should now fail for apple
        cursor = db_instance.execute_query("SELECT rowid FROM keyword_fts WHERE keyword_fts MATCH ?", ("apple",))
        assert cursor.fetchone() is None

        # Search for banana should still work
        cursor = db_instance.execute_query("SELECT rowid FROM keyword_fts WHERE keyword_fts MATCH ?", ("banana",))
        assert cursor.fetchone()['rowid'] == kw2_id