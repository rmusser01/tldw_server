# tests/test_sync_server.py
# Description: This file contains unit tests for the server-side synchronization processor, specifically focusing on applying client changes and handling conflicts. The tests ensure that the server correctly processes incoming changes, resolves conflicts, and maintains data integrity in the database.
#
# Imports
import pytest
import json
import time
from unittest.mock import MagicMock, patch

from tldw_Server_API.app.api.v1.endpoints.sync import ServerSyncProcessor
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import Database


#
# 3rd-party imports
#
# Local Imports
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

    @pytest.mark.asyncio # Mark test as async if processor methods are async
    async def test_apply_client_create_success(self, server_processor, server_user_db):
        """Test server applying a 'create' change from a client."""
        kw_uuid = "client-create-uuid"
        client_change = create_mock_log_entry(
            change_id=5, # Client's local change ID
            entity="Keywords", uuid=kw_uuid, op="create",
            client="client_sender_1", version=1,
            payload_dict={"uuid": kw_uuid, "keyword": "client_created"},
            ts="2023-11-01T09:00:00Z" # Client timestamp (will be overridden)
        )

        success, errors = await server_processor.apply_client_changes_batch([client_change])

        assert success is True
        assert not errors

        # Verify server DB state
        state = get_entity_state(server_user_db, "Keywords", kw_uuid)
        assert state is not None
        assert state['keyword'] == "client_created"
        assert state['version'] == 1
        assert state['client_id'] == "client_sender_1" # Originating client
        assert not state['deleted']
        # Check that server timestamp was used (hard to assert exact value without mocking time)
        assert state['last_modified'] > "2023-11-01T09:00:00Z"

    @pytest.mark.asyncio
    async def test_apply_client_update_success(self, server_processor, server_user_db):
         """Test server applying an 'update' change from a client."""
         # 1. Setup initial state on server DB (V1)
         kw_uuid = "client-update-uuid"
         # Manually insert V1 state as if synced before
         server_processor.db.execute_query(
              "INSERT INTO Keywords (uuid, keyword, version, client_id, last_modified, deleted) VALUES (?, ?, 1, ?, ?, 0)",
              (kw_uuid, "server_v1", "other_client", "2023-11-01T08:00:00Z"), commit=True
         )
         assert get_entity_state(server_user_db, "Keywords", kw_uuid)['version'] == 1

         # 2. Client sends update (V2 based on V1)
         client_change = create_mock_log_entry(
              change_id=6, entity="Keywords", uuid=kw_uuid, op="update",
              client="client_sender_1", version=2,
              payload_dict={"keyword": "client_updated_v2"},
              ts="2023-11-01T10:00:00Z"
         )

         success, errors = await server_processor.apply_client_changes_batch([client_change])

         assert success is True
         assert not errors

         # 3. Verify server DB state
         state = get_entity_state(server_user_db, "Keywords", kw_uuid)
         assert state['keyword'] == "client_updated_v2"
         assert state['version'] == 2
         assert state['client_id'] == "client_sender_1"
         assert state['last_modified'] > "2023-11-01T10:00:00Z" # Server time

    @pytest.mark.asyncio
    async def test_apply_idempotency_on_server(self, server_processor, server_user_db):
         """Test server correctly handles receiving the same change twice."""
         kw_uuid = "server-idem-uuid"
         client_change = create_mock_log_entry(5, "Keywords", kw_uuid, "create", "c1", 1, {"keyword":"idem1"}, "ts1")

         # Apply first time
         success1, errors1 = await server_processor.apply_client_changes_batch([client_change])
         assert success1 is True
         assert get_entity_state(server_user_db, "Keywords", kw_uuid)['version'] == 1

         # Apply second time
         success2, errors2 = await server_processor.apply_client_changes_batch([client_change])
         assert success2 is True # Should still report success (idempotent skip)
         assert not errors2
         # Version should remain 1
         assert get_entity_state(server_user_db, "Keywords", kw_uuid)['version'] == 1

    @pytest.mark.asyncio
    async def test_apply_old_change_on_server(self, server_processor, server_user_db):
         """Test server correctly skips a change older than its state."""
         # 1. Setup server state (V2)
         kw_uuid = "server-old-uuid"
         server_processor.db.execute_query(
              "INSERT INTO Keywords (uuid, keyword, version, client_id, last_modified, deleted) VALUES (?, ?, 2, ?, ?, 0)",
              (kw_uuid, "server_v2", "other_client", "2023-11-01T11:00:00Z"), commit=True
         )
         assert get_entity_state(server_user_db, "Keywords", kw_uuid)['version'] == 2

         # 2. Client sends change representing V1 state
         client_change_v1 = create_mock_log_entry(
              change_id=3, entity="Keywords", uuid=kw_uuid, op="update",
              client="c1", version=1, # Version is older than server
              payload_dict={"keyword": "client_v1_ignored"},
              ts="2023-11-01T09:30:00Z"
         )

         success, errors = await server_processor.apply_client_changes_batch([client_change_v1])

         assert success is True # Idempotent skip is success
         assert not errors
         # Server state should remain V2
         state = get_entity_state(server_user_db, "Keywords", kw_uuid)
         assert state['version'] == 2
         assert state['keyword'] == "server_v2"


class TestServerSyncProcessorConflict:

     @pytest.mark.asyncio
     async def test_server_conflict_client_wins_lww(self, server_processor, server_user_db):
          """Server detects conflict, incoming client change wins LWW."""
          # 1. Setup server state (V1, base timestamp)
          kw_uuid = "server-conflict-client-wins"
          ts_v1 = "2023-11-01T12:00:00Z"
          server_processor.db.execute_query(
               "INSERT INTO Keywords (uuid, keyword, version, client_id, last_modified, deleted) VALUES (?, ?, 1, ?, ?, 0)",
               (kw_uuid, "server_v1", "other_client", ts_v1), commit=True
          )

          # 2. Simulate concurrent server update (V2, timestamp V1 + 10s)
          ts_server_v2 = "2023-11-01T12:00:10Z"
          server_processor.db.execute_query(
               "UPDATE Keywords SET keyword='server_v2_concurrent', version=2, last_modified=? WHERE uuid=?",
               (ts_server_v2, kw_uuid), commit=True
          )
          assert get_entity_state(server_user_db, "Keywords", kw_uuid)['version'] == 2

          # 3. Client sends conflicting change (also V2, based on V1)
          client_change = create_mock_log_entry(
               change_id=10, entity="Keywords", uuid=kw_uuid, op="update",
               client="client_sender_1", version=2, # Based on V1
               payload_dict={"keyword": "client_v2_conflicting"},
               ts="2023-11-01T12:00:05Z" # Client's original time doesn't matter for LWW here
          )

          # 4. Process - Mock server time to be AFTER server_db_timestamp (ts_server_v2)
          with patch('tldw_Server_API.app.api.v1.endpoints.sync.datetime') as mock_dt:
                # Ensure the mocked datetime object behaves like the real one enough
                mock_now = MagicMock()
                mock_now.strftime.return_value = "2023-11-01T12:00:20Z"  # The desired server authoritative timestamp
                mock_dt.now.return_value = mock_now

                success, errors = await server_processor.apply_client_changes_batch([client_change])

                assert success is True # Conflict resolved successfully
                assert not errors

                # 5. Verify state - Client change was forcefully applied
                state = get_entity_state(server_user_db, "Keywords", kw_uuid)
                assert state['keyword'] == "client_v2_conflicting"
                assert state['version'] == 2 # Version from client change applied
                assert state['client_id'] == "client_sender_1"
                assert state['last_modified'] == "2023-11-01T12:00:20Z" # Server authoritative time

     @pytest.mark.asyncio
     async def test_server_conflict_server_wins_lww(self, server_processor, server_user_db):
          """Server detects conflict, existing server state wins LWW."""
           # 1. Setup server state (V1, base timestamp)
          kw_uuid = "server-conflict-server-wins"
          ts_v1 = "2023-11-01T13:00:00Z"
          server_processor.db.execute_query(
               "INSERT INTO Keywords (uuid, keyword, version, client_id, last_modified, deleted) VALUES (?, ?, 1, ?, ?, 0)",
               (kw_uuid, "server_v1_sw", "other_client", ts_v1), commit=True
          )

          # 2. Simulate concurrent server update (V2, timestamp V1 + 20s) - This is the 'winning' state
          ts_server_v2 = "2023-11-01T13:00:20Z"
          server_processor.db.execute_query(
               "UPDATE Keywords SET keyword='server_v2_wins_concurrent', version=2, last_modified=? WHERE uuid=?",
               (ts_server_v2, kw_uuid), commit=True
          )
          assert get_entity_state(server_user_db, "Keywords", kw_uuid)['version'] == 2

          # 3. Client sends conflicting change (also V2, based on V1)
          client_change = create_mock_log_entry(
               change_id=11, entity="Keywords", uuid=kw_uuid, op="update",
               client="client_sender_1", version=2, # Based on V1
               payload_dict={"keyword": "client_v2_loses"},
               ts="2023-11-01T13:00:05Z" # Client original time
          )

          # 4. Process - Mock server time to be BEFORE the winning server state's timestamp
          with patch('tldw_Server_API.app.api.v1.endpoints.sync.datetime') as mock_dt:
                mock_now = MagicMock()
                mock_now.strftime.return_value = "2023-11-01T13:00:15Z"  # The desired server authoritative timestamp
                mock_dt.now.return_value = mock_now

                success, errors = await server_processor.apply_client_changes_batch([client_change])

                assert success is True # Conflict resolved successfully (by skipping)
                assert not errors

                # 5. Verify state - Server state remains unchanged
                state = get_entity_state(server_user_db, "Keywords", kw_uuid)
                assert state['keyword'] == "server_v2_wins_concurrent" # Existing server state kept
                assert state['version'] == 2
                assert state['client_id'] == "other_client" # From the winning server update
                assert state['last_modified'] == ts_server_v2

#
# End of test_sync_server.py
#######################################################################################################################
