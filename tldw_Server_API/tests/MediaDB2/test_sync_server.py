# tests/test_sync_server.py
# Description: This file contains unit tests for the server-side synchronization processor, specifically focusing on applying client changes and handling conflicts. The tests ensure that the server correctly processes incoming changes, resolves conflicts, and maintains data integrity in the database.
#
# Imports
from datetime import timezone, datetime

import pytest
import json
import time
from unittest.mock import MagicMock, patch
#
# 3rd-party Libraries
#
# Local Imports
from tldw_Server_API.app.api.v1.endpoints.sync import ServerSyncProcessor
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import Database
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

def get_entity_state(db: Database, entity: str, uuid: str) -> dict | None:
     cursor = db.execute_query(f"SELECT * FROM `{entity}` WHERE uuid = ?", (uuid,))
     row = cursor.fetchone()
     return dict(row) if row else None

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
        ts_v1 = "2023-11-01T12:00:00Z"
        server_processor.db.execute_query(
          "INSERT INTO Keywords (uuid, keyword, version, client_id, last_modified, deleted) VALUES (?, ?, 1, ?, ?, 0)",
          (kw_uuid, "server_v1", "other_client", ts_v1), commit=True
        )
        ts_server_v2 = "2023-11-01T12:00:10Z"
        server_processor.db.execute_query(
          "UPDATE Keywords SET keyword='server_v2_concurrent', version=2, last_modified=? WHERE uuid=?",
          (ts_server_v2, kw_uuid), commit=True
        )
        assert get_entity_state(server_user_db, "Keywords", kw_uuid)['version'] == 2

        client_change = create_mock_log_entry(
          change_id=10, entity="Keywords", uuid=kw_uuid, op="update",
          client="client_sender_1", version=2,
          payload_dict={"keyword": "client_v2_conflicting"},
        )

        # --- Mock Setup using 'with patch' inside the test ---
        server_authoritative_time_str = "2023-11-01 12:00:20.123"  # String for parsing later
        mock_now_dt_object = datetime.strptime(server_authoritative_time_str, '%Y-%m-%d %H:%M:%S.%f').replace(
          tzinfo=timezone.utc)

        # Use 'with patch' targeting the specific attribute within the sync module
        with patch('tldw_Server_API.app.api.v1.endpoints.sync.datetime.now') as mock_dt_now:
          mock_dt_now.return_value = mock_now_dt_object  # Mock now() to return a real dt object

        # Call synchronous method INSIDE the patch context
        success, errors = server_processor.apply_client_changes_batch([client_change])
        # --- End patch context ---

        # --- Assertions ---
        assert success is True
        assert not errors

        state = get_entity_state(server_user_db, "Keywords", kw_uuid)
        assert state['keyword'] == "client_v2_conflicting"
        assert state['version'] == 3
        assert state['client_id'] == "client_sender_1"
        expected_stored_time = mock_now_dt_object.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + 'Z'
        assert state['last_modified'] == expected_stored_time

    def test_server_conflict_server_wins_lww(self, server_processor, server_user_db):
         """Server detects conflict, existing server state wins LWW."""
         kw_uuid = "server-conflict-server-wins"
         ts_v1 = "2023-11-01T13:00:00Z"
         server_processor.db.execute_query(
             "INSERT INTO Keywords (uuid, keyword, version, client_id, last_modified, deleted) VALUES (?, ?, 1, ?, ?, 0)",
             (kw_uuid, "server_v1_sw", "other_client", ts_v1), commit=True
         )
         ts_server_v2 = "2023-11-01T13:00:20Z"  # Server's winning timestamp
         server_processor.db.execute_query(
             "UPDATE Keywords SET keyword='server_v2_wins_concurrent', version=2, client_id='server_updater', last_modified=? WHERE uuid=?",
             (ts_server_v2, kw_uuid), commit=True
         )
         assert get_entity_state(server_user_db, "Keywords", kw_uuid)['version'] == 2

         client_change = create_mock_log_entry(
             change_id=11, entity="Keywords", uuid=kw_uuid, op="update",
             client="client_sender_1", version=2,
             payload_dict={"keyword": "client_v2_loses"},
         )

         # --- Mock Setup using 'with patch' inside the test ---
         server_authoritative_time_str = "2023-11-01 13:00:15.456"
         mock_now_dt_object = datetime.strptime(server_authoritative_time_str, '%Y-%m-%d %H:%M:%S.%f').replace(
             tzinfo=timezone.utc)

         with patch('tldw_Server_API.app.api.v1.endpoints.sync.datetime.now') as mock_dt_now:
             mock_dt_now.return_value = mock_now_dt_object
             success, errors = server_processor.apply_client_changes_batch([client_change])
         # --- End patch context ---

         # --- Assertions ---
         assert success is True
         assert not errors

         state = get_entity_state(server_user_db, "Keywords", kw_uuid)
         assert state['keyword'] == "server_v2_wins_concurrent"
         assert state['version'] == 2
         assert state['client_id'] == "server_updater"
         assert state['last_modified'] == ts_server_v2

#
# End of test_sync_server.py
#######################################################################################################################
