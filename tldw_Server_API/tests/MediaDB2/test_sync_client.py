# tests/test_sync_client.py
# Description: Unit tests for the ClientSyncEngine class, focusing on state management, push/pull operations, and conflict resolution.
#
# Imports
from datetime import datetime, timedelta
import pytest
import json
import os
from unittest.mock import patch, MagicMock, call
#
# 3rd-Party Imports
import requests
#
# Local Imports
from tldw_Server_API.tests.MediaDB2.test_sqlite_db import get_entity_version
from tldw_Server_API.app.core.Sync.Sync_Client import ClientSyncEngine
#
#######################################################################################################################
#
# Functions:

# --- Fixtures specific to client engine testing ---

@pytest.fixture
def client_db(memory_db_factory):
    """Provides a fresh client DB instance."""
    # Use a specific client ID matching potential state file content
    return memory_db_factory("client_test_eng")

@pytest.fixture
def client_state_file(temp_state_file):
    """Provides path to temp state file and ensures it starts empty or with defaults."""
    # Ensure file is empty before test if it exists from fixture creation
    if os.path.exists(temp_state_file):
        os.remove(temp_state_file)
    # Optionally write default empty state
    # with open(temp_state_file, 'w') as f:
    #    json.dump({'last_local_log_id_sent': 0, 'last_server_log_id_processed': 0}, f)
    return temp_state_file

@pytest.fixture
def sync_engine(client_db, client_state_file):
    """Provides an initialized ClientSyncEngine instance."""
    # Use a dummy server URL for unit tests, mocking network calls
    engine = ClientSyncEngine(
        db_instance=client_db,
        server_api_url="http://mock-server.test",
        client_id=client_db.client_id, # Use client_id from the db instance
        state_file=client_state_file
    )
    return engine

# Helper function to create mock sync log entries
def create_mock_log_entry(change_id, entity, uuid, op, client, version, payload_dict, ts="2023-01-01T12:00:00Z"):
    return {
        "change_id": change_id, "entity": entity, "entity_uuid": uuid,
        "operation": op, "timestamp": ts, "client_id": client,
        "version": version, "payload": json.dumps(payload_dict)
    }

# --- Test Class ---

class TestClientSyncEngineState:
    def test_state_load_no_file(self, client_db, temp_state_file):
        """Test initialization when state file doesn't exist."""
        engine = ClientSyncEngine(client_db, "http://mock", "c1", temp_state_file)
        assert engine.last_local_log_id_sent == 0
        assert engine.last_server_log_id_processed == 0
        # Check if file was created with defaults
        assert os.path.exists(temp_state_file)
        with open(temp_state_file, 'r') as f:
            state = json.load(f)
            assert state == {'last_local_log_id_sent': 0, 'last_server_log_id_processed': 0}


    def test_state_load_existing_file(self, client_db, temp_state_file):
        """Test initialization with a pre-existing state file."""
        initial_state = {'last_local_log_id_sent': 10, 'last_server_log_id_processed': 25}
        with open(temp_state_file, 'w') as f:
            json.dump(initial_state, f)

        engine = ClientSyncEngine(client_db, "http://mock", "c1", temp_state_file)
        assert engine.last_local_log_id_sent == 10
        assert engine.last_server_log_id_processed == 25

    def test_state_save(self, sync_engine):
        """Test saving the state updates the file."""
        sync_engine.last_local_log_id_sent = 15
        sync_engine.last_server_log_id_processed = 30
        sync_engine._save_sync_state()

        with open(sync_engine.state_file, 'r') as f:
            state = json.load(f)
            assert state == {'last_local_log_id_sent': 15, 'last_server_log_id_processed': 30}

    def test_state_load_corrupt_file(self, client_db, temp_state_file):
         """Test handling of corrupt state file."""
         with open(temp_state_file, 'w') as f:
              f.write("this is not json")
         engine = ClientSyncEngine(client_db, "http://mock", "c1", temp_state_file)
         # Should default to 0 and log an error (check logs manually or capture logs)
         assert engine.last_local_log_id_sent == 0
         assert engine.last_server_log_id_processed == 0
         # Check if file was overwritten with defaults
         with open(temp_state_file, 'r') as f:
            state = json.load(f)
            assert state == {'last_local_log_id_sent': 0, 'last_server_log_id_processed': 0}


class TestClientSyncEnginePush:

    @patch('tldw_Server_API.app.core.Sync.Sync_Client.requests.post') # Mock the post method used in sync_client
    def test_push_no_local_changes(self, mock_post, sync_engine):
        """Test push phase when there are no local changes."""
        sync_engine._push_local_changes()
        mock_post.assert_not_called() # Network call should not happen
        assert sync_engine.last_local_log_id_sent == 0 # State unchanged

    @patch('tldw_Server_API.app.core.Sync.Sync_Client.requests.post')
    def test_push_successful(self, mock_post, sync_engine, client_db):
        """Test successful push of local changes."""
        # 1. Create local changes
        kw_id, kw_uuid = client_db.add_keyword("push_test_kw")
        # (Assume add_keyword creates log entry 1)

        # 2. Configure mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None # Simulate HTTP 2xx
        mock_post.return_value = mock_response

        # 3. Run push
        sync_engine._push_local_changes()

        # 4. Assertions
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        sent_payload = kwargs['json']
        assert sent_payload['client_id'] == sync_engine.client_id
        assert len(sent_payload['changes']) == 1
        assert sent_payload['changes'][0]['change_id'] == 1 # Assuming it's the first log entry
        assert sent_payload['changes'][0]['entity_uuid'] == kw_uuid
        assert sent_payload['last_processed_server_id'] == 0

        # State should be updated
        assert sync_engine.last_local_log_id_sent == 1

    @patch('tldw_Server_API.app.core.Sync.Sync_Client.requests.post')
    def test_push_http_error(self, mock_post, sync_engine, client_db):
        """Test push phase when server returns an HTTP error."""
        client_db.add_keyword("push_fail_kw")  # Log entry 1

        mock_http_response = MagicMock(spec=requests.Response)  # Mock the actual response object
        mock_http_response.status_code = 500
        mock_http_response.text = "Internal Server Error Text"
        mock_http_error = requests.exceptions.HTTPError("Server Error", response=mock_http_response)

        # Configure the mock post *return value* (the response obj)
        # Then configure its raise_for_status method to raise the error we just created
        mock_post_response = MagicMock()
        mock_post_response.raise_for_status.side_effect = mock_http_error
        mock_post.return_value = mock_post_response

        # Run push - expect it to catch the error internally and log
        sync_engine._push_local_changes()

        mock_post.assert_called_once()
        # State should NOT be updated on error
        assert sync_engine.last_local_log_id_sent == 0


class TestClientSyncEnginePullApply:

    @patch('tldw_Server_API.app.core.Sync.Sync_Client.requests.get')
    def test_pull_no_remote_changes(self, mock_get, sync_engine):
        """Test pull phase when server has no new changes."""
        # Configure mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        # Server response indicates no changes, but its latest ID is 50
        mock_response.json.return_value = {"changes": [], "latest_change_id": 50}
        mock_get.return_value = mock_response

        sync_engine._pull_and_apply_remote_changes()

        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert kwargs['params']['since_change_id'] == 0 # Initial state
        # State should fast-forward to server's latest known ID
        assert sync_engine.last_server_log_id_processed == 50

    @patch('tldw_Server_API.app.core.Sync.Sync_Client.requests.get')
    def test_pull_and_apply_create_success(self, mock_get, sync_engine, client_db):
        """Test pulling and applying a 'create' change successfully."""
        kw_uuid = "uuid-from-server-create"
        kw_name = "server_created_keyword"
        server_change = create_mock_log_entry(
            change_id=101, # Server's change ID
            entity="Keywords", uuid=kw_uuid, op="create",
            client="other_client", version=1,
            payload_dict={"uuid": kw_uuid, "keyword": kw_name, "deleted": False, "client_id": "other_client"},
            ts="2023-10-28T10:00:00Z" # Assume server adds this authoritative timestamp
        )
        # Configure mock response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"changes": [server_change], "latest_change_id": 101}
        mock_get.return_value = mock_response

        # Run pull & apply
        sync_engine._pull_and_apply_remote_changes()

        # Assertions
        mock_get.assert_called_once()
        # Verify DB state
        cursor = client_db.execute_query("SELECT * FROM Keywords WHERE uuid = ?", (kw_uuid,))
        row = cursor.fetchone()
        assert row is not None
        assert row['keyword'] == kw_name
        assert row['version'] == 1
        assert row['client_id'] == "other_client" # Originating client ID stored
        assert row['last_modified'] == "2023-10-28T10:00:00Z" # Server timestamp applied
        assert not row['deleted']
        # Verify state update
        assert sync_engine.last_server_log_id_processed == 101

    @patch('tldw_Server_API.app.core.Sync.Sync_Client.requests.get')
    def test_pull_and_apply_update_success(self, mock_get, sync_engine, client_db):
        """Test pulling and applying an 'update' change successfully."""
        # 1. Setup initial state locally
        kw_uuid = "uuid-for-update"
        client_db.add_keyword("initial_name") # Assume this creates version 1, kw_uuid = '...' - need the actual uuid
        cursor = client_db.execute_query("SELECT uuid FROM Keywords WHERE keyword = ?", ("initial_name",))
        kw_uuid = cursor.fetchone()['uuid']
        assert get_entity_version(client_db, "Keywords", kw_uuid) == 1

        # 2. Prepare server change (update to version 2)
        server_change = create_mock_log_entry(
            change_id=102, entity="Keywords", uuid=kw_uuid, op="update",
            client="other_client", version=2,
            payload_dict={"uuid": kw_uuid, "keyword": "updated_name", "client_id": "other_client", "version": 2},
            ts="2023-10-28T11:00:00Z"
        )
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"changes": [server_change], "latest_change_id": 102}
        mock_get.return_value = mock_response

        # Set state to expect this change
        sync_engine.last_server_log_id_processed = 101

        # 3. Run pull & apply
        sync_engine._pull_and_apply_remote_changes()

        # 4. Assertions
        cursor = client_db.execute_query("SELECT keyword, version, last_modified FROM Keywords WHERE uuid = ?", (kw_uuid,))
        row = cursor.fetchone()
        assert row['keyword'] == "updated_name"
        assert row['version'] == 2
        assert row['last_modified'] == "2023-10-28T11:00:00Z"
        assert sync_engine.last_server_log_id_processed == 102

    @patch('tldw_Server_API.app.core.Sync.Sync_Client.requests.get')
    def test_pull_and_apply_delete_success(self, mock_get, sync_engine, client_db):
        """Test pulling and applying a 'delete' change successfully."""
        # 1. Setup initial state locally
        kw_uuid = "uuid-for-delete"
        client_db.add_keyword("to_be_deleted")
        kw_uuid = client_db.execute_query("SELECT uuid FROM Keywords WHERE keyword = ?", ("to_be_deleted",)).fetchone()['uuid']
        assert get_entity_version(client_db, "Keywords", kw_uuid) == 1

        # 2. Prepare server change (delete, version 2)
        server_change = create_mock_log_entry(
            change_id=103, entity="Keywords", uuid=kw_uuid, op="delete",
            client="other_client", version=2,
            payload_dict={"uuid": kw_uuid, "version": 2, "client_id": "other_client"}, # Minimal delete payload
            ts="2023-10-28T12:00:00Z"
        )
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"changes": [server_change], "latest_change_id": 103}
        mock_get.return_value = mock_response
        sync_engine.last_server_log_id_processed = 102

        # 3. Run pull & apply
        sync_engine._pull_and_apply_remote_changes()

        # 4. Assertions
        cursor = client_db.execute_query("SELECT deleted, version, last_modified FROM Keywords WHERE uuid = ?", (kw_uuid,))
        row = cursor.fetchone()
        assert row['deleted'] == 1
        assert row['version'] == 2
        assert row['last_modified'] == "2023-10-28T12:00:00Z"
        assert sync_engine.last_server_log_id_processed == 103

    @patch('tldw_Server_API.app.core.Sync.Sync_Client.requests.get')
    def test_apply_idempotency(self, mock_get, sync_engine, client_db):
        """Test that applying the same change twice has no adverse effect."""
        # 1. Apply a change once
        kw_uuid = "uuid-idempotent"
        server_change = create_mock_log_entry(101, "Keywords", kw_uuid, "create", "other", 1, {"keyword":"idem"}, "ts1")
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"changes": [server_change], "latest_change_id": 101}
        mock_get.return_value = mock_response
        sync_engine._pull_and_apply_remote_changes()
        assert get_entity_version(client_db, "Keywords", kw_uuid) == 1
        assert sync_engine.last_server_log_id_processed == 101

        # 2. Mock receiving the *same* change again
        # Note: Real server would likely filter this, but test client robustness
        mock_response.json.return_value = {"changes": [server_change], "latest_change_id": 101} # Same change ID
        # Don't reset sync_engine.last_server_log_id_processed, it's already 101

        # 3. Run apply again
        sync_engine._pull_and_apply_remote_changes()

        # 4. Assertions
        # State should not have changed, no error should occur
        assert get_entity_version(client_db, "Keywords", kw_uuid) == 1 # Still version 1
        assert sync_engine.last_server_log_id_processed == 101 # Should not advance


    @patch('tldw_Server_API.app.core.Sync.Sync_Client.requests.get')
    def test_apply_old_change(self, mock_get, sync_engine, client_db):
        """Test that applying a change older than local state is skipped."""
        # 1. Setup local state (V2)
        kw_uuid = "uuid-old-change"
        client_db.add_keyword("v1")  # Create V1
        kw_uuid = client_db.execute_query("SELECT uuid FROM Keywords WHERE keyword=?", ("v1",)).fetchone()['uuid']
        # Manually update to V2 locally - Use client_db here:
        client_db.execute_query("UPDATE Keywords SET keyword='v2', version=2, last_modified=? WHERE uuid=?",
                                (client_db._get_current_utc_timestamp_str(), kw_uuid),  # Use client_db
                                commit=True)
        assert get_entity_version(client_db, "Keywords", kw_uuid) == 2

         # 2. Prepare server change (representing V1 state)
        server_change_v1 = create_mock_log_entry(101, "Keywords", kw_uuid, "update", "other", 1, {"keyword":"v1_server"}, "ts_old")
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"changes": [server_change_v1], "latest_change_id": 101}
        mock_get.return_value = mock_response

        # 3. Run pull & apply
        sync_engine._pull_and_apply_remote_changes()

        # 4. Assertions
        # Local state should remain V2
        assert get_entity_version(client_db, "Keywords", kw_uuid) == 2
        cursor=client_db.execute_query("SELECT keyword FROM Keywords where uuid=?", (kw_uuid,))
        assert cursor.fetchone()['keyword'] == 'v2'
        # State should advance to server's latest ID even if change was skipped
        assert sync_engine.last_server_log_id_processed == 101


class TestClientSyncEngineConflict:

     @patch('tldw_Server_API.app.core.Sync.Sync_Client.requests.get')
     def test_conflict_detected_and_remote_wins_lww(self, mock_get, sync_engine, client_db):
          """Test conflict where remote change wins based on LWW timestamp."""
          # 1. Setup initial synced state (V1)
          kw_uuid = "uuid-conflict-remote-wins"
          client_db.add_keyword("synced_v1")
          kw_uuid = client_db.execute_query("SELECT uuid FROM Keywords WHERE keyword=?", ("synced_v1",)).fetchone()['uuid']
          ts_v1 = client_db.execute_query("SELECT last_modified FROM Keywords WHERE uuid=?", (kw_uuid,)).fetchone()[0]
          assert get_entity_version(client_db, "Keywords", kw_uuid) == 1

          # 2. Simulate local concurrent change (V2 - Local)
          ts_local_v2 = (datetime.fromisoformat(ts_v1.replace("Z","+00:00")) + timedelta(seconds=10)).strftime('%Y-%m-%dT%H:%M:%SZ')
          client_db.execute_query(
               "UPDATE Keywords SET keyword='local_v2', version=2, last_modified=?, client_id=? WHERE uuid=?",
               (ts_local_v2, sync_engine.client_id, kw_uuid), commit=True
          )
          assert get_entity_version(client_db, "Keywords", kw_uuid) == 2

          # 3. Prepare conflicting server change (also V2, based on V1, but with later timestamp)
          ts_server_v2 = (datetime.fromisoformat(ts_v1.replace("Z","+00:00")) + timedelta(seconds=20)).strftime('%Y-%m-%dT%H:%M:%SZ')
          server_change = create_mock_log_entry(
               change_id=102, entity="Keywords", uuid=kw_uuid, op="update",
               client="other_client", version=2, # Based on V1
               payload_dict={"keyword": "server_v2_wins"},
               ts=ts_server_v2 # Server's authoritative timestamp > local timestamp
          )
          mock_response = MagicMock()
          mock_response.raise_for_status.return_value = None
          mock_response.json.return_value = {"changes": [server_change], "latest_change_id": 102}
          mock_get.return_value = mock_response
          sync_engine.last_server_log_id_processed = 101 # Assume V1 was processed

          # 4. Run pull & apply
          sync_engine._pull_and_apply_remote_changes()

          # 5. Assertions
          # Conflict should be detected, LWW resolution should apply server change
          cursor = client_db.execute_query("SELECT keyword, version, last_modified FROM Keywords WHERE uuid = ?",
                                           (kw_uuid,))
          row = cursor.fetchone()
          assert row['keyword'] == "server_v2_wins"  # Server change applied
          # assert row['version'] == 2 # OLD Assertion: Expects version to stay 2
          assert row['version'] == 3  # NEW Assertion: Expects version to be incremented by force_apply logic
          assert row['last_modified'] == ts_server_v2  # Server timestamp applied
          assert sync_engine.last_server_log_id_processed == 102


     @patch('tldw_Server_API.app.core.Sync.Sync_Client.requests.get')
     def test_conflict_detected_and_local_wins_lww(self, mock_get, sync_engine, client_db):
           """Test conflict where local change wins based on LWW timestamp."""
           # 1. Setup initial synced state (V1)
           kw_uuid = "uuid-conflict-local-wins"
           client_db.add_keyword("synced_v1_lw")
           kw_uuid = client_db.execute_query("SELECT uuid FROM Keywords WHERE keyword=?", ("synced_v1_lw",)).fetchone()['uuid']
           ts_v1 = client_db.execute_query("SELECT last_modified FROM Keywords WHERE uuid=?", (kw_uuid,)).fetchone()[0]
           assert get_entity_version(client_db, "Keywords", kw_uuid) == 1

           # 2. Simulate local concurrent change (V2 - Local) with later timestamp
           ts_local_v2 = (datetime.fromisoformat(ts_v1.replace("Z","+00:00")) + timedelta(seconds=30)).strftime('%Y-%m-%dT%H:%M:%SZ')
           client_db.execute_query(
               "UPDATE Keywords SET keyword='local_v2_wins', version=2, last_modified=?, client_id=? WHERE uuid=?",
               (ts_local_v2, sync_engine.client_id, kw_uuid), commit=True
           )
           assert get_entity_version(client_db, "Keywords", kw_uuid) == 2

           # 3. Prepare conflicting server change (also V2, based on V1, but with earlier timestamp)
           ts_server_v2 = (datetime.fromisoformat(ts_v1.replace("Z","+00:00")) + timedelta(seconds=5)).strftime('%Y-%m-%dT%H:%M:%SZ')
           server_change = create_mock_log_entry(
               change_id=102, entity="Keywords", uuid=kw_uuid, op="update",
               client="other_client", version=2, # Based on V1
               payload_dict={"keyword": "server_v2_loses"},
               ts=ts_server_v2 # Server's authoritative timestamp < local timestamp
           )
           mock_response = MagicMock()
           mock_response.raise_for_status.return_value = None
           mock_response.json.return_value = {"changes": [server_change], "latest_change_id": 102}
           mock_get.return_value = mock_response
           sync_engine.last_server_log_id_processed = 101 # Assume V1 was processed

           # 4. Run pull & apply
           sync_engine._pull_and_apply_remote_changes()

           # 5. Assertions
           # Conflict should be detected, LWW resolution should *skip* server change
           cursor = client_db.execute_query("SELECT keyword, version, last_modified FROM Keywords WHERE uuid = ?", (kw_uuid,))
           row = cursor.fetchone()
           assert row['keyword'] == "local_v2_wins" # Local change retained
           assert row['version'] == 2
           assert row['last_modified'] == ts_local_v2 # Local timestamp retained
           assert sync_engine.last_server_log_id_processed == 102 # State still advances

#
# End of test_sync_client.py
#######################################################################################################################
