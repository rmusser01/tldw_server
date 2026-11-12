# tests/test_sync_server.py
# Description: This file contains unit tests for the server-side synchronization processor, specifically focusing on applying client changes and handling conflicts. The tests ensure that the server correctly processes incoming changes, resolves conflicts, and maintains data integrity in the database.
#
# Imports
import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

#
# 3rd-party Libraries
#
# Local Imports
from tldw_Server_API.app.core.Sync.server_sync_processor import ServerSyncProcessor
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

#
#######################################################################################################################
#
# Functions:


# Helper function from client tests (can be moved to conftest or shared utils)
def create_mock_log_entry(change_id, entity, uuid, op, client, version, payload_dict, ts="2023-01-01T12:00:00Z"):
    # Slight modification for testing: payload might not always exist in input dict format
    payload_str = json.dumps(payload_dict) if payload_dict is not None else None
    return {
         "change_id": change_id, "entity": entity, "entity_uuid": uuid,
         "operation": op, "timestamp": ts, "client_id": client,
         "version": version, "payload": payload_str # Ensure payload is string for server input
     }

def get_entity_state(db: MediaDatabase, entity: str, uuid: str) -> dict | None:
    cursor = db.execute_query(f"SELECT * FROM `{entity}` WHERE uuid = ?", (uuid,))  # nosec B608
    row = cursor.fetchone()
    return dict(row) if row else None


def get_media_keyword_link_count(db: MediaDatabase, media_uuid: str, keyword_uuid: str) -> int:
    cursor = db.execute_query(
        """
        SELECT COUNT(*) AS cnt
        FROM MediaKeywords mk
        JOIN Media m ON m.id = mk.media_id
        JOIN Keywords k ON k.id = mk.keyword_id
        WHERE m.uuid = ? AND k.uuid = ?
        """,
        (media_uuid, keyword_uuid),
    )
    row = cursor.fetchone()
    return int(row["cnt"]) if row else 0


@pytest.fixture(scope="function")
def server_user_db(memory_db_factory):
    """Provides a fresh DB instance representing a user's DB on the server."""
    db = memory_db_factory("SERVER") # Use server's client ID when acting
    yield db # Use yield to allow for potential cleanup if needed
    # Optional cleanup: ensure connection is closed after test
    try:
        db.close_connection()
    except Exception as e:
        print(f"Warning: Error closing DB connection in fixture teardown: {e}")

@pytest.fixture(scope="function")
def server_processor(server_user_db):
    """Provides an initialized ServerSyncProcessor instance."""
    # Ensure the server_user_db passed here is the fresh, function-scoped one
    return ServerSyncProcessor(db=server_user_db, user_id="test_user_1", requesting_client_id="client_sender_1")


class TestServerSyncProcessorApply:

    # Remove @pytest.mark.asyncio and async/await
    def test_apply_client_create_success(self, server_processor, server_user_db):
        """Test server applying a 'create' change from a client."""
        kw_uuid = "client-create-uuid"
        client_change = create_mock_log_entry(
            change_id=5, entity="Keywords", uuid=kw_uuid, op="create",
            client="client_sender_1", version=1,
            payload_dict={"uuid": kw_uuid, "keyword": "client_created"},
            ts="2023-11-01T09:00:00Z"
        )

        # Call synchronous method
        success, errors = server_processor.apply_client_changes_batch([client_change])

        assert success is True
        assert not errors

        state = get_entity_state(server_user_db, "Keywords", kw_uuid)
        assert state is not None
        assert state['keyword'] == "client_created"
        assert state['version'] == 1
        assert state['client_id'] == "client_sender_1"
        assert not state['deleted']
        assert state['last_modified'] > "2023-11-01T09:00:00Z"

    # Remove @pytest.mark.asyncio and async/await
    def test_apply_client_update_success(self, server_processor, server_user_db):
        """Test server applying an 'update' change from a client."""
        kw_uuid = "client-update-uuid"
        server_processor.db.execute_query(
              "INSERT INTO Keywords (uuid, keyword, version, client_id, last_modified, deleted) VALUES (?, ?, 1, ?, ?, 0)",
              (kw_uuid, "server_v1", "other_client", "2023-11-01T08:00:00Z"), commit=True
         )
        assert get_entity_state(server_user_db, "Keywords", kw_uuid)['version'] == 1

        client_change = create_mock_log_entry(
              change_id=6, entity="Keywords", uuid=kw_uuid, op="update",
              client="client_sender_1", version=2,
              payload_dict={"keyword": "client_updated_v2"},
              ts="2023-11-01T10:00:00Z"
         )

        # Call synchronous method
        success, errors = server_processor.apply_client_changes_batch([client_change])

        assert success is True
        assert not errors

        state = get_entity_state(server_user_db, "Keywords", kw_uuid)
        assert state['keyword'] == "client_updated_v2"
        assert state['version'] == 2
        assert state['client_id'] == "client_sender_1"
        assert state['last_modified'] > "2023-11-01T10:00:00Z"

    def test_apply_client_media_update_refreshes_fts_index(self, server_processor, server_user_db):
        """Media updates received via sync should refresh the FTS index."""
        media_id, media_uuid, _ = server_user_db.add_media_with_keywords(
            title="sync media old title",
            media_type="article",
            content="legacy_sync_marker",
            keywords=[],
        )

        initial_results, initial_total = MediaDatabase.search_media_db(
            server_user_db,
            search_query="legacy_sync_marker",
            search_fields=["content"],
        )
        assert initial_total == 1
        assert initial_results[0]["id"] == media_id

        client_change = create_mock_log_entry(
            change_id=24,
            entity="Media",
            uuid=media_uuid,
            op="update",
            client="client_sender_1",
            version=2,
            payload_dict={
                "title": "sync media new title",
                "content": "fresh_sync_token_123",
            },
            ts="2023-11-01T10:15:00Z",
        )

        success, errors = server_processor.apply_client_changes_batch([client_change])

        assert success is True
        assert not errors

        updated_results, updated_total = MediaDatabase.search_media_db(
            server_user_db,
            search_query="fresh_sync_token_123",
            search_fields=["content"],
        )
        assert updated_total == 1
        assert updated_results[0]["id"] == media_id

    def test_apply_client_media_update_rolls_back_when_fts_refresh_fails(
        self,
        server_processor,
        server_user_db,
        monkeypatch,
    ):
        """FTS refresh failures should rollback the sync write atomically."""
        media_id, media_uuid, _ = server_user_db.add_media_with_keywords(
            title="rollback old title",
            media_type="article",
            content="rollback_old_content",
            keywords=[],
        )

        def _raise_fts_failure(*_args, **_kwargs):
            raise DatabaseError("synthetic fts refresh failure")

        monkeypatch.setattr(
            server_processor.db,
            "sync_refresh_fts_for_entity",
            _raise_fts_failure,
            raising=False,
        )

        client_change = create_mock_log_entry(
            change_id=25,
            entity="Media",
            uuid=media_uuid,
            op="update",
            client="client_sender_1",
            version=2,
            payload_dict={
                "title": "rollback new title",
                "content": "rollback_new_content",
            },
            ts="2023-11-01T10:20:00Z",
        )

        success, errors = server_processor.apply_client_changes_batch([client_change])

        assert success is False
        assert errors
        state = get_entity_state(server_user_db, "Media", media_uuid)
        assert state is not None
        assert state["id"] == media_id
        assert state["title"] == "rollback old title"
        assert state["content"] == "rollback_old_content"
        assert state["version"] == 1

    def test_apply_client_mediakeywords_link_success(self, server_processor, server_user_db):
        """MediaKeywords link should be handled without generic uuid/version lookup."""
        media_id, media_uuid, _ = server_user_db.add_media_with_keywords(
            title="server sync link media",
            media_type="article",
            content="server media body",
            keywords=[],
        )
        keyword_id, keyword_uuid = server_user_db.add_keyword("server-sync-link-keyword")
        assert media_id is not None
        assert keyword_id is not None

        link_change = create_mock_log_entry(
            change_id=20,
            entity="MediaKeywords",
            uuid=f"{media_uuid}_{keyword_uuid}",
            op="link",
            client="client_sender_1",
            version=1,
            payload_dict={"media_uuid": media_uuid, "keyword_uuid": keyword_uuid},
            ts="2023-11-01T10:30:00Z",
        )

        success, errors = server_processor.apply_client_changes_batch([link_change])

        assert success is True
        assert not errors
        assert get_media_keyword_link_count(server_user_db, media_uuid, keyword_uuid) == 1

    def test_apply_client_mediakeywords_unlink_success(self, server_processor, server_user_db):
        """MediaKeywords unlink should remove the link idempotently."""
        media_id, media_uuid, _ = server_user_db.add_media_with_keywords(
            title="server sync unlink media",
            media_type="article",
            content="server media body",
            keywords=[],
        )
        keyword_id, keyword_uuid = server_user_db.add_keyword("server-sync-unlink-keyword")
        assert media_id is not None
        assert keyword_id is not None

        server_user_db.execute_query(
            "INSERT INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)",
            (media_id, keyword_id),
            commit=True,
        )
        assert get_media_keyword_link_count(server_user_db, media_uuid, keyword_uuid) == 1

        unlink_change = create_mock_log_entry(
            change_id=21,
            entity="MediaKeywords",
            uuid=f"{media_uuid}_{keyword_uuid}",
            op="unlink",
            client="client_sender_1",
            version=2,
            payload_dict={"media_uuid": media_uuid, "keyword_uuid": keyword_uuid},
            ts="2023-11-01T10:31:00Z",
        )

        success, errors = server_processor.apply_client_changes_batch([unlink_change])

        assert success is True
        assert not errors
        assert get_media_keyword_link_count(server_user_db, media_uuid, keyword_uuid) == 0

    def test_apply_client_change_rejects_invalid_entity_operation(self, server_processor):
        """Unsupported entity/operation combinations should be rejected cleanly."""
        invalid_change = create_mock_log_entry(
            change_id=22,
            entity="Keywords",
            uuid="invalid-op-uuid",
            op="link",
            client="client_sender_1",
            version=1,
            payload_dict={"keyword": "invalid"},
            ts="2023-11-01T10:32:00Z",
        )

        success, errors = server_processor.apply_client_changes_batch([invalid_change])

        assert success is False
        assert errors
        assert any("only valid for MediaKeywords" in err for err in errors)

    def test_apply_client_change_rejects_disallowed_entity(self, server_processor):
        """Entities outside the server /sync/send allowlist should be rejected."""
        invalid_change = create_mock_log_entry(
            change_id=23,
            entity="Transcripts",
            uuid="invalid-entity-uuid",
            op="create",
            client="client_sender_1",
            version=1,
            payload_dict={"uuid": "invalid-entity-uuid", "transcription": "not supported by sync send"},
            ts="2023-11-01T10:33:00Z",
        )

        success, errors = server_processor.apply_client_changes_batch([invalid_change])

        assert success is False
        assert errors
        assert any("Unsupported sync entity 'Transcripts'" in err for err in errors)

    # Remove @pytest.mark.asyncio and async/await
    def test_apply_idempotency_on_server(self, server_processor, server_user_db):
        """Test server correctly handles receiving the same change twice."""
        kw_uuid = "server-idem-uuid"
        client_change = create_mock_log_entry(5, "Keywords", kw_uuid, "create", "c1", 1, {"keyword":"idem1"}, "ts1")

        # Apply first time (sync call)
        success1, errors1 = server_processor.apply_client_changes_batch([client_change])
        assert success1 is True
        assert get_entity_state(server_user_db, "Keywords", kw_uuid)['version'] == 1

        # Apply second time (sync call)
        success2, errors2 = server_processor.apply_client_changes_batch([client_change])
        assert success2 is True
        assert not errors2
        assert get_entity_state(server_user_db, "Keywords", kw_uuid)['version'] == 1

    # Remove @pytest.mark.asyncio and async/await
    def test_apply_old_change_on_server(self, server_processor, server_user_db):
        """Test server correctly skips a change older than its state."""
        kw_uuid = "server-old-uuid"
        server_processor.db.execute_query(
              "INSERT INTO Keywords (uuid, keyword, version, client_id, last_modified, deleted) VALUES (?, ?, 2, ?, ?, 0)",
              (kw_uuid, "server_v2", "other_client", "2023-11-01T11:00:00Z"), commit=True
         )
        assert get_entity_state(server_user_db, "Keywords", kw_uuid)['version'] == 2

        client_change_v1 = create_mock_log_entry(
              change_id=3, entity="Keywords", uuid=kw_uuid, op="update",
              client="c1", version=1,
              payload_dict={"keyword": "client_v1_ignored"},
              ts="2023-11-01T09:30:00Z"
         )

        # Call synchronous method
        success, errors = server_processor.apply_client_changes_batch([client_change_v1])

        assert success is True
        assert not errors
        state = get_entity_state(server_user_db, "Keywords", kw_uuid)
        assert state['version'] == 2
        assert state['keyword'] == "server_v2"


class TestServerSyncProcessorConflict:

    def test_server_conflict_client_wins_lww(self, server_processor, server_user_db):

        """Server detects conflict, incoming client change wins LWW."""
        kw_uuid = "server-conflict-client-wins"
        ts_v1 = "2023-11-01T12:00:00.000Z"
        server_processor.db.execute_query(
          "INSERT INTO Keywords (uuid, keyword, version, client_id, last_modified, deleted) VALUES (?, ?, 1, ?, ?, 0)",
          (kw_uuid, "server_v1", "other_client", ts_v1), commit=True
        )
        # USE CONSISTENT ISO FORMAT
        ts_server_v2 = "2023-11-01T12:00:10.000Z"
        server_processor.db.execute_query(
          "UPDATE Keywords SET keyword='server_v2_concurrent', version=2, last_modified=? WHERE uuid=?",
          (ts_server_v2, kw_uuid), commit=True
        )
        assert get_entity_state(server_user_db, "Keywords", kw_uuid)['version'] == 2

        client_change = create_mock_log_entry(
            change_id=10, entity="Keywords", uuid=kw_uuid, op="update",
            client="client_sender_1", version=2,
            payload_dict={"keyword": "client_v2_conflicting"},
            ts="2023-11-01T12:00:15.000Z"  # Consistent format
        )

        # --- Mock Setup ---
        # Generate mock datetime object
        mock_now_dt_object = datetime.now(timezone.utc).replace(year=2023, month=11, day=1, hour=12, minute=0,
                                                                second=20, microsecond=123000)
        # Format it using the *exact same* strftime as the main code
        server_authoritative_time_str = mock_now_dt_object.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        # server_authoritative_time_str should be "2023-11-01T12:00:20.123Z"

        # Patching logic remains the same
        with patch('tldw_Server_API.app.core.Sync.server_sync_processor.datetime') as mock_datetime_module:
            mock_datetime_module.now.return_value = mock_now_dt_object
            # Provide strptime if needed (the code *does* use it now for parsing)
            mock_datetime_module.strptime = datetime.strptime

            # Call synchronous method INSIDE the patch context
            success, errors = server_processor.apply_client_changes_batch([client_change])
        # --- End patch context ---

        # --- Assertions ---
        assert success is True, f"Expected success=True, got False. Errors: {errors}"
        assert not errors

        state = get_entity_state(server_user_db, "Keywords", kw_uuid)
        assert state is not None
        assert state['keyword'] == "client_v2_conflicting"
        assert state['version'] == 3  # Incremented because client won (forced update)
        assert state['client_id'] == "client_sender_1"
        # Timestamp should match the mocked authoritative *string*
        assert state['last_modified'] == server_authoritative_time_str

    def test_server_conflict_server_wins_lww(self, server_processor, server_user_db):

        """Server detects conflict, existing server state wins LWW."""
        kw_uuid = "server-conflict-server-wins"
        # USE CONSISTENT ISO FORMAT
        ts_v1 = "2023-11-01T13:00:00.000Z"
        server_processor.db.execute_query(
             "INSERT INTO Keywords (uuid, keyword, version, client_id, last_modified, deleted) VALUES (?, ?, 1, ?, ?, 0)",
             (kw_uuid, "server_v1_sw", "other_client", ts_v1), commit=True
         )
        # USE CONSISTENT ISO FORMAT (Server's winning timestamp)
        ts_server_v2 = "2023-11-01T13:00:20.000Z"
        server_processor.db.execute_query(
             "UPDATE Keywords SET keyword='server_v2_wins_concurrent', version=2, client_id='server_updater', last_modified=? WHERE uuid=?",
             (ts_server_v2, kw_uuid), commit=True
         )
        assert get_entity_state(server_user_db, "Keywords", kw_uuid)['version'] == 2

        client_change = create_mock_log_entry(
             change_id=11, entity="Keywords", uuid=kw_uuid, op="update",
             client="client_sender_1", version=2,
             payload_dict={"keyword": "client_v2_loses"},
             ts="2023-11-01T13:00:10.000Z"  # Consistent format
         )

        # --- Mock Setup ---
        # Mocked time is *earlier* than ts_server_v2
        mock_now_dt_object = datetime.now(timezone.utc).replace(year=2023, month=11, day=1, hour=13, minute=0,
                                                                 second=15, microsecond=456000)

        # Patching logic remains the same
        with patch('tldw_Server_API.app.core.Sync.server_sync_processor.datetime') as mock_datetime_module:
            mock_datetime_module.now.return_value = mock_now_dt_object
            mock_datetime_module.strptime = datetime.strptime

            # Call synchronous method INSIDE the patch context
            success, errors = server_processor.apply_client_changes_batch([client_change])
        # --- End patch context ---

        # --- Assertions ---
        assert success is True, f"Expected success=True, got False. Errors: {errors}"
        assert not errors  # Should succeed by skipping the client change

        state = get_entity_state(server_user_db, "Keywords", kw_uuid)
        assert state is not None
        # State should remain unchanged because server won LWW
        assert state['keyword'] == "server_v2_wins_concurrent"
        assert state['version'] == 2
        assert state['client_id'] == "server_updater"
        # Timestamp should be the server's winning timestamp string
        assert state['last_modified'] == ts_server_v2

    def test_server_conflict_server_wins_lww_with_plus00_offset_timestamp(self, server_processor, server_user_db):
        """Server conflict resolution should accept +00:00 server timestamps."""
        kw_uuid = "server-conflict-server-wins-plus00"
        ts_v1 = "2023-11-01T14:00:00.000Z"
        server_processor.db.execute_query(
            "INSERT INTO Keywords (uuid, keyword, version, client_id, last_modified, deleted) VALUES (?, ?, 1, ?, ?, 0)",
            (kw_uuid, "server_v1_plus00", "other_client", ts_v1),
            commit=True,
        )
        ts_server_v2 = "2023-11-01T14:00:20+00:00"
        server_processor.db.execute_query(
            "UPDATE Keywords SET keyword='server_v2_plus00', version=2, client_id='server_updater', last_modified=? WHERE uuid=?",
            (ts_server_v2, kw_uuid),
            commit=True,
        )

        client_change = create_mock_log_entry(
            change_id=12,
            entity="Keywords",
            uuid=kw_uuid,
            op="update",
            client="client_sender_1",
            version=2,
            payload_dict={"keyword": "client_v2_loses_plus00"},
            ts="2023-11-01T14:00:10.000Z",
        )

        mock_now_dt_object = datetime.now(timezone.utc).replace(
            year=2023,
            month=11,
            day=1,
            hour=14,
            minute=0,
            second=15,
            microsecond=111000,
        )
        with patch("tldw_Server_API.app.core.Sync.server_sync_processor.datetime") as mock_datetime_module:
            mock_datetime_module.now.return_value = mock_now_dt_object
            mock_datetime_module.strptime = datetime.strptime
            success, errors = server_processor.apply_client_changes_batch([client_change])

        assert success is True, f"Expected success=True, got False. Errors: {errors}"
        assert not errors

        state = get_entity_state(server_user_db, "Keywords", kw_uuid)
        assert state is not None
        assert state["keyword"] == "server_v2_plus00"
        assert state["version"] == 2
        assert state["client_id"] == "server_updater"
        assert state["last_modified"] == ts_server_v2

    def test_server_conflict_client_wins_lww_with_non_utc_offset_timestamp(self, server_processor, server_user_db):
        """Server conflict resolution should normalize non-UTC offsets before LWW compare."""
        kw_uuid = "server-conflict-client-wins-offset"
        ts_v1 = "2023-11-01T11:00:00.000Z"
        server_processor.db.execute_query(
            "INSERT INTO Keywords (uuid, keyword, version, client_id, last_modified, deleted) VALUES (?, ?, 1, ?, ?, 0)",
            (kw_uuid, "server_v1_offset", "other_client", ts_v1),
            commit=True,
        )
        ts_server_v2 = "2023-11-01T13:00:20+02:00"  # Equivalent to 11:00:20Z
        server_processor.db.execute_query(
            "UPDATE Keywords SET keyword='server_v2_offset', version=2, client_id='server_updater', last_modified=? WHERE uuid=?",
            (ts_server_v2, kw_uuid),
            commit=True,
        )

        client_change = create_mock_log_entry(
            change_id=13,
            entity="Keywords",
            uuid=kw_uuid,
            op="update",
            client="client_sender_1",
            version=2,
            payload_dict={"keyword": "client_v2_wins_offset"},
            ts="2023-11-01T11:00:21.000Z",
        )

        mock_now_dt_object = datetime.now(timezone.utc).replace(
            year=2023,
            month=11,
            day=1,
            hour=11,
            minute=0,
            second=25,
            microsecond=250000,
        )
        expected_authoritative_time_str = mock_now_dt_object.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        with patch("tldw_Server_API.app.core.Sync.server_sync_processor.datetime") as mock_datetime_module:
            mock_datetime_module.now.return_value = mock_now_dt_object
            mock_datetime_module.strptime = datetime.strptime
            success, errors = server_processor.apply_client_changes_batch([client_change])

        assert success is True, f"Expected success=True, got False. Errors: {errors}"
        assert not errors

        state = get_entity_state(server_user_db, "Keywords", kw_uuid)
        assert state is not None
        assert state["keyword"] == "client_v2_wins_offset"
        assert state["version"] == 3
        assert state["client_id"] == "client_sender_1"
        assert state["last_modified"] == expected_authoritative_time_str

    def test_resolve_server_conflict_accepts_authoritative_plus00_offset(self, server_processor):
        """Conflict resolver should parse authoritative timestamps using +00:00 offsets."""
        kw_uuid = "server-conflict-authoritative-plus00"
        server_processor.db.execute_query(
            "INSERT INTO Keywords (uuid, keyword, version, client_id, last_modified, deleted) VALUES (?, ?, 2, ?, ?, 0)",
            (kw_uuid, "server_v2_authoritative", "server_updater", "2023-11-01T14:00:20+00:00"),
            commit=True,
        )

        client_change = create_mock_log_entry(
            change_id=14,
            entity="Keywords",
            uuid=kw_uuid,
            op="update",
            client="client_sender_1",
            version=2,
            payload_dict={"keyword": "client_v2_loses_authoritative"},
            ts="2023-11-01T14:00:10.000Z",
        )

        with server_processor.db.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT version, client_id, last_modified FROM Keywords WHERE uuid = ?",
                (kw_uuid,),
            )
            server_record_info = cursor.fetchone()
            resolved, error_msg = server_processor._resolve_server_conflict_sync(
                cursor=cursor,
                client_change=client_change,
                server_record_info=server_record_info,
                current_server_time_str="2023-11-01T14:00:15+00:00",
            )

        assert resolved is True
        assert error_msg is None

#
# End of test_sync_server.py
#######################################################################################################################
