# tests/test_sync_client.py
import sqlite3

import pytest
import os
import json
import time
import uuid
from unittest.mock import MagicMock, call # Can use unittest.mock directly or pytest-mock's mocker

# Import necessary components from your modules
# Assuming they are in the parent directory or PYTHONPATH is set
import sys

import requests

from tldw_cli.tldw_app.DB.Media_DB import Database, DatabaseError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tldw_cli.tldw_app.DB.Sync_Client import ClientSyncEngine, SYNC_BATCH_SIZE  # Your client sync engine



# Explanation:
#     Fixtures: temp_db_path, temp_state_file, db_instance, sync_engine set up a clean environment for each test. The db_instance uses a unique client_id per test.
#     mocker: The mocker fixture from pytest-mock is used to patch requests.post and requests.get. This intercepts network calls and allows us to control the server's responses.
#     create_mock_response: A helper to build mock requests.Response objects with specific status codes and JSON data. Crucially, it simulates raise_for_status() behavior.
#     generate_sync_log_entry: Creates dictionary structures mimicking what the server would send back, including the authoritative server_timestamp.
#     get_db_state: A helper to query the database directly and check the state of a specific record after a sync operation.
#     Test Structure: Tests are grouped logically (Initialization, Push, Pull/Apply, Conflicts, etc.). Each test focuses on a specific scenario.
#     Arrange-Act-Assert: Tests generally follow this pattern:
#         Arrange: Set up the initial DB state (if needed), set the engine's state markers, and mock the server responses (mocker.patch).
#         Act: Call sync_engine.run_sync_cycle().
#         Assert: Check the final DB state (get_db_state), the engine's state markers (sync_engine.last_...), and verify that the mocks (requests.post/get) were called as expected (assert_called_once, assert_not_called, check call_args).
#     Conflict Tests: These explicitly set different last_modified timestamps locally vs. the server_timestamp in the mocked remote change to test the LWW logic in _resolve_conflict.
#     Transactionality Test: Simulates a batch where one change is designed to fail, then verifies that none of the changes in that batch were committed to the DB.
# Added:
#   Entity Tests (test_apply_remote_create_media, etc.): These follow the pattern of the Keywords tests but use payloads relevant to Media and DocumentVersions, checking that specific fields are applied correctly. The Media test also verifies that the new prev_version / merge_parent_uuid columns are handled if present in the payload (set to None in this case).
#   Edge Cases (test_apply_payload_edge_cases): Creates a Media record with a payload containing None, empty strings, and True boolean to verify they are stored correctly as NULL, '', and 1 respectively in SQLite.
#   Complex Conflicts (test_conflict_local_delete_remote_update, test_conflict_local_update_remote_delete): These tests carefully construct scenarios where local and remote changes conflict. They set specific timestamps to force either the local or remote change to win based on the LWW strategy (using server_timestamp) and assert the final state of the record (deleted or updated).
#   Batching Tests (test_push_large_batch, test_pull_large_batch):
#   Push test creates more local changes than SYNC_BATCH_SIZE, mocks successful server responses, runs the sync cycle twice, and asserts that requests.post was called twice with the correct number of changes in each batch and that the last_local_log_id_sent marker is updated correctly.
#   Pull test generates mock server changes split across two batches, uses mock_get.side_effect to return these batches sequentially, runs the sync cycle multiple times, and asserts that the DB state reflects the applied changes incrementally and the last_server_log_id_processed marker advances correctly.
#   Initial Sync (test_initial_sync_empty_client_db): Starts with the default empty DB provided by the fixture, mocks a server response containing initial create operations, runs the sync, and asserts that the items now exist locally and the processed marker is updated.



# --- Fixtures ---

@pytest.fixture
def temp_db_path(tmp_path):
    """Provides a path to a temporary database file for each test."""
    # tmp_path is a pytest fixture providing a temporary directory unique to each test function
    db_file = tmp_path / "test_client_media.db"
    return str(db_file)

@pytest.fixture
def temp_state_file(tmp_path):
    """Provides a path to a temporary state file."""
    state_file = tmp_path / "client_sync_state.json"
    return str(state_file)


@pytest.fixture
def db_instance(temp_db_path):
    """Creates a Database instance with a unique client ID for testing."""
    client_id = f"test_client_{uuid.uuid4().hex[:8]}"
    db = Database(db_path=temp_db_path, client_id=client_id)
    yield db # provide the db instance to the test
    # Teardown: close connection after test finishes
    try:
        db.close_connection()
    except Exception as e:
        print(f"Error closing test DB connection: {e}")


@pytest.fixture
def sync_engine(db_instance, temp_state_file):
    """Creates a ClientSyncEngine instance for testing."""
    # Use a known fake server URL
    engine = ClientSyncEngine(
        db_instance=db_instance,
        server_api_url="http://fake-server.test",
        client_id=db_instance.client_id,
        state_file=temp_state_file
    )
    return engine

# --- Helper Functions ---

def create_mock_response(status_code=200, json_data=None, text_data=""):
    """Creates a mock requests.Response object."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.text = text_data
    if status_code >= 400:
        # Make raise_for_status raise an exception for error codes
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_resp)
    else:
        # Ensure raise_for_status does nothing for success codes
        mock_resp.raise_for_status.return_value = None
    return mock_resp

def generate_sync_log_entry(
    change_id: int, entity: str, entity_uuid: str, operation: str, version: int,
    payload: dict, client_id: str = "client_A", timestamp: str = None, server_timestamp: str = None
    ) -> dict:
    """Helper to create realistic sync log entries for mocking server responses."""
    ts = timestamp or time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
    sts = server_timestamp or ts # Default server ts to client ts if not provided
    return {
        "change_id": change_id,
        "entity": entity,
        "entity_uuid": entity_uuid,
        "operation": operation,
        "timestamp": ts, # Client's original timestamp
        "server_timestamp": sts, # Server's authoritative timestamp
        "client_id": client_id, # Originating client
        "version": version,
        "payload": json.dumps(payload) # Payload must be a JSON string
    }

def get_db_state(db: Database, entity: str, entity_uuid: str) -> dict | None:
    """Helper to fetch the current state of a record from the DB."""
    try:
        # Use execute_query which returns a cursor
        cursor = db.execute_query(f"SELECT * FROM `{entity}` WHERE uuid = ?", (entity_uuid,))
        row = cursor.fetchone()
        return dict(row) if row else None
    except (DatabaseError, sqlite3.Error) as e:
        pytest.fail(f"Helper get_db_state failed for {entity}/{entity_uuid}: {e}")


# --- Test Class ---

class TestClientSyncEngine:

    # --- Initialization and State Tests ---

    def test_initialization_creates_state_file(self, sync_engine, temp_state_file):
        """Verify state file is created if it doesn't exist."""
        assert os.path.exists(temp_state_file)
        with open(temp_state_file, 'r') as f:
            state = json.load(f)
        assert state == {'last_local_log_id_sent': 0, 'last_server_log_id_processed': 0}

    def test_initialization_loads_existing_state(self, db_instance, temp_state_file):
        """Verify existing state is loaded correctly."""
        # Pre-populate state file
        initial_state = {'last_local_log_id_sent': 10, 'last_server_log_id_processed': 25}
        with open(temp_state_file, 'w') as f:
            json.dump(initial_state, f)

        engine = ClientSyncEngine(db_instance, "http://fake-server.test", db_instance.client_id, temp_state_file)
        assert engine.last_local_log_id_sent == 10
        assert engine.last_server_log_id_processed == 25

    def test_save_state_updates_file(self, sync_engine, temp_state_file):
        """Verify state file is updated correctly."""
        sync_engine.last_local_log_id_sent = 5
        sync_engine.last_server_log_id_processed = 15
        sync_engine._save_sync_state()

        with open(temp_state_file, 'r') as f:
            state = json.load(f)
        assert state == {'last_local_log_id_sent': 5, 'last_server_log_id_processed': 15}

    # --- Push Logic Tests ---

    def test_push_no_local_changes(self, sync_engine, mocker):
        """Verify no push request is made if the local sync log is empty or up-to-date."""
        mock_post = mocker.patch('requests.post')
        sync_engine.run_sync_cycle() # Assuming pull is mocked or returns nothing
        mock_post.assert_not_called()

    def test_push_one_local_change(self, sync_engine, db_instance, mocker):
        """Verify a single local change is pushed correctly."""
        # 1. Create a local change (triggers sync_log entry)
        kw_text = "local_push_test"
        kw_uuid = str(uuid.uuid4())
        db_instance.execute_query(
            "INSERT INTO Keywords (keyword, uuid, last_modified, version, client_id, deleted) VALUES (?, ?, datetime('now'), 1, ?, 0)",
            (kw_text, kw_uuid, db_instance.client_id), commit=True
        )
        local_logs = db_instance.get_sync_log_entries(since_change_id=0)
        assert len(local_logs) == 1
        expected_change_id = local_logs[0]['change_id']

        # 2. Mock the server response for the push
        mock_post = mocker.patch('requests.post', return_value=create_mock_response(200))
        mocker.patch('requests.get', return_value=create_mock_response(200, json_data={"changes": [], "latest_change_id": 0})) # Mock pull

        # 3. Run sync cycle
        sync_engine.run_sync_cycle()

        # 4. Assertions
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://fake-server.test/sync/send"
        sent_payload = kwargs['json']
        assert sent_payload['client_id'] == db_instance.client_id
        assert len(sent_payload['changes']) == 1
        assert sent_payload['changes'][0]['entity'] == 'Keywords'
        assert sent_payload['changes'][0]['entity_uuid'] == kw_uuid
        assert sent_payload['changes'][0]['operation'] == 'create'
        assert sent_payload['changes'][0]['version'] == 1

        assert sync_engine.last_local_log_id_sent == expected_change_id

    def test_push_network_error(self, sync_engine, db_instance, mocker):
        """Verify state is not updated on network error during push."""
        # Create a local change
        db_instance.add_keyword("local_net_error_kw")
        assert sync_engine.last_local_log_id_sent == 0

        # Mock network error
        mock_post = mocker.patch('requests.post', side_effect=requests.exceptions.ConnectionError("Fake connection error"))
        mock_get = mocker.patch('requests.get') # To check if pull is skipped

        # Run sync cycle
        sync_engine.run_sync_cycle()

        # Assertions
        mock_post.assert_called_once()
        mock_get.assert_not_called() # Pull should be skipped on network error during push
        assert sync_engine.last_local_log_id_sent == 0 # State not updated

    def test_push_server_error(self, sync_engine, db_instance, mocker):
        """Verify state is not updated on server error (500) during push."""
        # Create a local change
        db_instance.add_keyword("local_server_error_kw")
        assert sync_engine.last_local_log_id_sent == 0

        # Mock server error response
        mock_post = mocker.patch('requests.post', return_value=create_mock_response(500, text_data="Internal Server Error"))
        # Mock pull to return successfully (pull should still be attempted after non-network push error)
        mock_get = mocker.patch('requests.get', return_value=create_mock_response(200, json_data={"changes": [], "latest_change_id": 0}))

        # Run sync cycle
        sync_engine.run_sync_cycle()

        # Assertions
        mock_post.assert_called_once()
        mock_get.assert_called_once() # Pull is attempted
        assert sync_engine.last_local_log_id_sent == 0 # State not updated

    # --- Pull & Apply Logic Tests ---

    def test_pull_no_remote_changes(self, sync_engine, mocker):
        """Verify state update when server has no new changes but is ahead."""
        sync_engine.last_server_log_id_processed = 5 # Client is at 5
        mock_post = mocker.patch('requests.post') # Assume no local changes
        # Server response: no changes, but its latest ID is 10
        mock_get = mocker.patch('requests.get', return_value=create_mock_response(200, json_data={"changes": [], "latest_change_id": 10}))

        sync_engine.run_sync_cycle()

        mock_post.assert_not_called()
        mock_get.assert_called_once()
        assert sync_engine.last_server_log_id_processed == 10 # State updated to server's latest

    def test_apply_remote_create(self, sync_engine, db_instance, mocker):
        """Verify applying a 'create' operation from the server."""
        kw_uuid = str(uuid.uuid4())
        payload = {"keyword": "remote_create_kw", "uuid": kw_uuid, "deleted": False} # Add required fields
        change = generate_sync_log_entry(
            change_id=1, entity="Keywords", entity_uuid=kw_uuid, operation="create",
            version=1, payload=payload, client_id="server_or_other_client"
        )
        mock_post = mocker.patch('requests.post')
        mock_get = mocker.patch('requests.get', return_value=create_mock_response(200, json_data={"changes": [change], "latest_change_id": 1}))

        # Check DB before
        assert get_db_state(db_instance, "Keywords", kw_uuid) is None

        sync_engine.run_sync_cycle()

        # Check DB after
        db_state = get_db_state(db_instance, "Keywords", kw_uuid)
        assert db_state is not None
        assert db_state['keyword'] == "remote_create_kw"
        assert db_state['version'] == 1
        assert db_state['client_id'] == "server_or_other_client" # Check originating client ID
        assert db_state['deleted'] == 0

        assert sync_engine.last_server_log_id_processed == 1

    def test_apply_remote_update(self, sync_engine, db_instance, mocker):
        """Verify applying an 'update' operation from the server (no conflict)."""
        # 1. Setup initial state locally
        kw_uuid = str(uuid.uuid4())
        db_instance.execute_query(
            "INSERT INTO Keywords (keyword, uuid, last_modified, version, client_id, deleted) VALUES (?, ?, datetime('now'), 1, ?, 0)",
            ("initial_local_kw", kw_uuid, db_instance.client_id), commit=True
        )
        # Assume client has processed up to change 0 from server
        sync_engine.last_server_log_id_processed = 0

        # 2. Prepare remote change (updating the keyword)
        payload = {"keyword": "remote_updated_kw"} # Only include changed field in payload
        change = generate_sync_log_entry(
            change_id=1, entity="Keywords", entity_uuid=kw_uuid, operation="update",
            version=2, payload=payload, client_id="other_client" # Version 2 based on version 1
        )
        mock_post = mocker.patch('requests.post')
        mock_get = mocker.patch('requests.get', return_value=create_mock_response(200, json_data={"changes": [change], "latest_change_id": 1}))

        # 3. Run sync
        sync_engine.run_sync_cycle()

        # 4. Check DB state
        db_state = get_db_state(db_instance, "Keywords", kw_uuid)
        assert db_state is not None
        assert db_state['keyword'] == "remote_updated_kw"
        assert db_state['version'] == 2 # Version incremented
        assert db_state['client_id'] == "other_client" # Originating client
        assert db_state['deleted'] == 0

        assert sync_engine.last_server_log_id_processed == 1

    def test_apply_remote_delete(self, sync_engine, db_instance, mocker):
        """Verify applying a 'delete' (soft delete) operation."""
        # Setup initial state
        kw_uuid = str(uuid.uuid4())
        db_instance.execute_query(
            "INSERT INTO Keywords (keyword, uuid, last_modified, version, client_id, deleted) VALUES (?, ?, datetime('now'), 1, ?, 0)",
            ("kw_to_delete", kw_uuid, db_instance.client_id), commit=True
        )
        sync_engine.last_server_log_id_processed = 0

        # Prepare remote delete change
        # Payload for delete is minimal in trigger, but server might add more context if needed
        payload = {"uuid": kw_uuid} # Minimal payload sufficient for identification
        change = generate_sync_log_entry(
            change_id=1, entity="Keywords", entity_uuid=kw_uuid, operation="delete",
            version=2, payload=payload, client_id="deleter_client"
        )
        mock_post = mocker.patch('requests.post')
        mock_get = mocker.patch('requests.get', return_value=create_mock_response(200, json_data={"changes": [change], "latest_change_id": 1}))

        # Run sync
        sync_engine.run_sync_cycle()

        # Check DB state (should be soft deleted)
        db_state = get_db_state(db_instance, "Keywords", kw_uuid)
        assert db_state is not None
        assert db_state['deleted'] == 1 # Soft deleted
        assert db_state['version'] == 2 # Version incremented
        assert db_state['client_id'] == "deleter_client"

        assert sync_engine.last_server_log_id_processed == 1

    def test_apply_idempotency_duplicate_change(self, sync_engine, db_instance, mocker):
        """Verify applying the same change twice is handled correctly."""
        # Setup: Apply a change once
        kw_uuid = str(uuid.uuid4())
        payload1 = {"keyword": "idempotency_test_kw"}
        change1 = generate_sync_log_entry(
            change_id=1, entity="Keywords", entity_uuid=kw_uuid, operation="create", version=1, payload=payload1
        )
        mock_post = mocker.patch('requests.post')
        mock_get1 = mocker.patch('requests.get', return_value=create_mock_response(200, json_data={"changes": [change1], "latest_change_id": 1}))

        sync_engine.run_sync_cycle()
        assert sync_engine.last_server_log_id_processed == 1
        db_state1 = get_db_state(db_instance, "Keywords", kw_uuid)
        assert db_state1['version'] == 1

        # Act: Run sync again, server sends the *same* change ID 1
        # Need to reset mock_get specifically for the second call
        mocker.patch('requests.get', return_value=create_mock_response(200, json_data={"changes": [change1], "latest_change_id": 1}))
        # Mock post again if needed, or ensure no local changes exist
        mocker.patch('requests.post')

        sync_engine.run_sync_cycle()

        # Assert: State didn't change, log ID still 1 (as no *new* changes processed)
        db_state2 = get_db_state(db_instance, "Keywords", kw_uuid)
        assert db_state2['version'] == 1 # Version unchanged
        assert db_state2['keyword'] == "idempotency_test_kw"
        assert sync_engine.last_server_log_id_processed == 1 # Still 1

    def test_apply_idempotency_old_change(self, sync_engine, db_instance, mocker):
        """Verify applying a change with version <= local version is skipped."""
        # Setup: Local state is already at version 2
        kw_uuid = str(uuid.uuid4())
        db_instance.execute_query(
            "INSERT INTO Keywords (keyword, uuid, last_modified, version, client_id, deleted) VALUES (?, ?, datetime('now'), 2, ?, 0)",
            ("local_v2_kw", kw_uuid, db_instance.client_id), commit=True
        )
        sync_engine.last_server_log_id_processed = 0 # Server sending change based on older state

        # Act: Server sends an update for version 1 (older than local)
        payload = {"keyword": "remote_v1_update"}
        change = generate_sync_log_entry(
            change_id=1, entity="Keywords", entity_uuid=kw_uuid, operation="update", version=1, payload=payload
        )
        mock_post = mocker.patch('requests.post')
        mock_get = mocker.patch('requests.get', return_value=create_mock_response(200, json_data={"changes": [change], "latest_change_id": 1}))

        sync_engine.run_sync_cycle()

        # Assert: DB state unchanged, log ID updated (as server change *was* processed/skipped)
        db_state = get_db_state(db_instance, "Keywords", kw_uuid)
        assert db_state['version'] == 2 # Still version 2
        assert db_state['keyword'] == "local_v2_kw" # Local keyword remains
        assert sync_engine.last_server_log_id_processed == 1 # Updated as change was handled (skipped)

    # --- Conflict Tests (LWW) ---

    def test_conflict_remote_wins_lww(self, sync_engine, db_instance, mocker):
        """Remote update conflicts with local, remote timestamp is newer."""
         # 1. Local state: version 2, timestamp T1
        kw_uuid = str(uuid.uuid4())
        t1 = time.time()
        ts1 = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t1))
        db_instance.execute_query(
            "INSERT INTO Keywords (keyword, uuid, last_modified, version, client_id, deleted) VALUES (?, ?, ?, 2, ?, 0)",
            ("local_v2_change", kw_uuid, ts1, db_instance.client_id), commit=True
        )
        sync_engine.last_server_log_id_processed = 0 # Assume server is sending based on V1

        # 2. Remote change: version 2 (also based on V1), timestamp T2 (T2 > T1)
        t2 = t1 + 10 # Ensure remote timestamp is newer
        ts2_server = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t2))
        payload = {"keyword": "remote_v2_newer_wins"}
        change = generate_sync_log_entry(
            change_id=1, entity="Keywords", entity_uuid=kw_uuid, operation="update",
            version=2, payload=payload, client_id="other_client", server_timestamp=ts2_server
        )
        mock_post = mocker.patch('requests.post')
        mock_get = mocker.patch('requests.get', return_value=create_mock_response(200, json_data={"changes": [change], "latest_change_id": 1}))

        # 3. Run sync
        sync_engine.run_sync_cycle()

        # 4. Assert: Remote change applied (forced), version=2, keyword updated
        db_state = get_db_state(db_instance, "Keywords", kw_uuid)
        assert db_state is not None
        assert db_state['keyword'] == "remote_v2_newer_wins" # Remote change applied
        assert db_state['version'] == 2 # Version remains 2 (as remote was also V2)
        assert db_state['last_modified'] == ts2_server # Server timestamp applied
        assert db_state['client_id'] == "other_client" # Originating client ID applied
        assert sync_engine.last_server_log_id_processed == 1

    def test_conflict_local_wins_lww(self, sync_engine, db_instance, mocker):
        """Remote update conflicts with local, local timestamp is newer."""
        # 1. Local state: version 2, timestamp T2 (Newer)
        kw_uuid = str(uuid.uuid4())
        t2 = time.time()
        ts2 = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t2))
        db_instance.execute_query(
            "INSERT INTO Keywords (keyword, uuid, last_modified, version, client_id, deleted) VALUES (?, ?, ?, 2, ?, 0)",
            ("local_v2_newer_wins", kw_uuid, ts2, db_instance.client_id), commit=True
        )
        sync_engine.last_server_log_id_processed = 0

        # 2. Remote change: version 2 (based on V1), timestamp T1 (Older)
        t1 = t2 - 10
        ts1_server = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t1))
        payload = {"keyword": "remote_v2_older_loses"}
        change = generate_sync_log_entry(
            change_id=1, entity="Keywords", entity_uuid=kw_uuid, operation="update",
            version=2, payload=payload, client_id="other_client", server_timestamp=ts1_server
        )
        mock_post = mocker.patch('requests.post')
        mock_get = mocker.patch('requests.get', return_value=create_mock_response(200, json_data={"changes": [change], "latest_change_id": 1}))

        # 3. Run sync
        sync_engine.run_sync_cycle()

        # 4. Assert: Remote change skipped, local state preserved
        db_state = get_db_state(db_instance, "Keywords", kw_uuid)
        assert db_state is not None
        assert db_state['keyword'] == "local_v2_newer_wins" # Local change preserved
        assert db_state['version'] == 2 # Local version preserved
        assert db_state['last_modified'] == ts2 # Local timestamp preserved
        assert db_state['client_id'] == db_instance.client_id # Local client ID preserved
        assert sync_engine.last_server_log_id_processed == 1 # Marker updated as change was handled (skipped)


    # --- Transactionality Tests ---

    def test_batch_apply_rolls_back_on_error(self, sync_engine, db_instance, mocker):
        """Verify entire batch rolls back if one change fails."""
        # 1. Setup: Create kw1 locally
        kw1_uuid = str(uuid.uuid4())
        db_instance.add_keyword("kw1_initial")

        # 2. Remote changes:
        #    - Change 1: Update kw1 (Valid)
        #    - Change 2: Create kw2 (Valid)
        #    - Change 3: Update non-existent entity (Will fail)
        #    - Change 4: Create kw3 (Should not be applied)
        kw2_uuid = str(uuid.uuid4())
        kw3_uuid = str(uuid.uuid4())
        non_existent_uuid = str(uuid.uuid4())

        changes = [
            generate_sync_log_entry(1, "Keywords", kw1_uuid, "update", 2, {"keyword": "kw1_updated"}),
            generate_sync_log_entry(2, "Keywords", kw2_uuid, "create", 1, {"keyword": "kw2_created"}),
            generate_sync_log_entry(3, "Keywords", non_existent_uuid, "update", 1, {"keyword": "fail_update"}, server_timestamp=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(time.time()+5))), # Update non-existent - optimistic lock should fail -> ConflictError
            generate_sync_log_entry(4, "Keywords", kw3_uuid, "create", 1, {"keyword": "kw3_should_not_exist"})
        ]
        mock_post = mocker.patch('requests.post')
        mock_get = mocker.patch('requests.get', return_value=create_mock_response(200, json_data={"changes": changes, "latest_change_id": 4}))

        # 3. Run sync
        sync_engine.run_sync_cycle()

        # 4. Assertions
        #    - State marker NOT updated because batch failed
        assert sync_engine.last_server_log_id_processed == 0
        #    - kw1 should NOT be updated (rolled back)
        kw1_state = get_db_state(db_instance, "Keywords", kw1_uuid)
        assert kw1_state['keyword'] == "kw1_initial"
        assert kw1_state['version'] == 1
        #    - kw2 should NOT exist (rolled back)
        assert get_db_state(db_instance, "Keywords", kw2_uuid) is None
        #    - kw3 should NOT exist (rolled back)
        assert get_db_state(db_instance, "Keywords", kw3_uuid) is None

    # --- MediaKeywords Tests ---
    def test_apply_remote_link_unlink(self, sync_engine, db_instance, mocker):
        """Test linking and unlinking MediaKeywords."""
        # Setup: Create Media and Keyword locally
        media_uuid = str(uuid.uuid4())
        kw_uuid = str(uuid.uuid4())
        db_instance.execute_query(
            "INSERT INTO Media (title, type, content_hash, uuid, version, client_id) VALUES (?, ?, ?, ?, 1, ?)",
            ("link_media", "test", uuid.uuid4().hex, media_uuid, db_instance.client_id), commit=True
        )
        db_instance.execute_query(
            "INSERT INTO Keywords (keyword, uuid, version, client_id) VALUES (?, ?, 1, ?)",
            ("link_kw", kw_uuid, db_instance.client_id), commit=True
        )
        media_id = get_db_state(db_instance, "Media", media_uuid)['id']
        kw_id = get_db_state(db_instance, "Keywords", kw_uuid)['id']

        # Changes: Link then Unlink
        link_payload = {"media_uuid": media_uuid, "keyword_uuid": kw_uuid}
        unlink_payload = link_payload # Same payload identifies the link
        changes = [
            generate_sync_log_entry(1, "MediaKeywords", f"{media_uuid}_{kw_uuid}", "link", 1, link_payload),
            generate_sync_log_entry(2, "MediaKeywords", f"{media_uuid}_{kw_uuid}", "unlink", 1, unlink_payload) # Version for junction table less defined, using 1
        ]
        mock_post = mocker.patch('requests.post')
        mock_get = mocker.patch('requests.get', return_value=create_mock_response(200, json_data={"changes": changes, "latest_change_id": 2}))

        # Check initial state (no link)
        cursor = db_instance.execute_query("SELECT COUNT(*) FROM MediaKeywords WHERE media_id = ? AND keyword_id = ?", (media_id, kw_id))
        assert cursor.fetchone()[0] == 0

        # Run sync
        sync_engine.run_sync_cycle()

        # Check final state (link created then deleted)
        cursor = db_instance.execute_query("SELECT COUNT(*) FROM MediaKeywords WHERE media_id = ? AND keyword_id = ?", (media_id, kw_id))
        assert cursor.fetchone()[0] == 0
        assert sync_engine.last_server_log_id_processed == 2

    def test_apply_remote_create_media(self, sync_engine, db_instance, mocker):
        """Verify applying a 'create' operation for Media entity."""
        media_uuid = str(uuid.uuid4())
        content_hash = uuid.uuid4().hex  # Simulate content hash
        payload = {
            "uuid": media_uuid,
            "title": "Remote Media Title",
            "type": "article",
            "content": "Remote media content.",
            "content_hash": content_hash,
            "author": "Remote Author",
            "url": f"http://remote.test/{media_uuid}",
            "deleted": False,
            "is_trash": False,
            # Include new conflict-ready columns if present in payload (even if None)
            "prev_version": None,
            "merge_parent_uuid": None
        }
        change = generate_sync_log_entry(
            change_id=5, entity="Media", entity_uuid=media_uuid, operation="create",
            version=1, payload=payload, client_id="media_creator"
        )
        mocker.patch('requests.post')  # Assume no local changes to push
        mocker.patch('requests.get',
                     return_value=create_mock_response(200, json_data={"changes": [change], "latest_change_id": 5}))

        assert get_db_state(db_instance, "Media", media_uuid) is None
        sync_engine.run_sync_cycle()

        db_state = get_db_state(db_instance, "Media", media_uuid)
        assert db_state is not None
        assert db_state['title'] == "Remote Media Title"
        assert db_state['type'] == "article"
        assert db_state['content_hash'] == content_hash
        assert db_state['author'] == "Remote Author"
        assert db_state['url'] == f"http://remote.test/{media_uuid}"
        assert db_state['version'] == 1
        assert db_state['client_id'] == "media_creator"
        assert db_state['deleted'] == 0
        assert db_state['is_trash'] == 0
        # Verify new columns are stored correctly (as None in this case)
        assert db_state['prev_version'] is None
        assert db_state['merge_parent_uuid'] is None
        assert sync_engine.last_server_log_id_processed == 5

    def test_apply_remote_update_media_partial(self, sync_engine, db_instance, mocker):
        """Verify applying an update with only a subset of Media fields."""
        # 1. Setup initial state
        media_uuid = str(uuid.uuid4())
        initial_hash = uuid.uuid4().hex
        db_instance.execute_query(
            "INSERT INTO Media (uuid, title, type, content_hash, version, client_id, deleted) VALUES (?, ?, ?, ?, 1, ?, 0)",
            (media_uuid, "Initial Media", "article", initial_hash, db_instance.client_id), commit=True
        )
        sync_engine.last_server_log_id_processed = 0

        # 2. Prepare remote change (updating only title and author)
        payload = {
            "title": "Updated Media Title",
            "author": "Updated Author"
            # Other fields like type, content_hash are NOT in the payload
        }
        change = generate_sync_log_entry(
            change_id=6, entity="Media", entity_uuid=media_uuid, operation="update",
            version=2, payload=payload, client_id="media_updater"
        )
        mocker.patch('requests.post')
        mocker.patch('requests.get',
                     return_value=create_mock_response(200, json_data={"changes": [change], "latest_change_id": 6}))

        # 3. Run sync
        sync_engine.run_sync_cycle()

        # 4. Check DB state
        db_state = get_db_state(db_instance, "Media", media_uuid)
        assert db_state is not None
        assert db_state['title'] == "Updated Media Title"  # Updated
        assert db_state['author'] == "Updated Author"  # Updated
        assert db_state['type'] == "article"  # Unchanged (was not in payload)
        assert db_state['content_hash'] == initial_hash  # Unchanged (was not in payload)
        assert db_state['version'] == 2  # Incremented
        assert db_state['client_id'] == "media_updater"  # Updated
        assert sync_engine.last_server_log_id_processed == 6

    def test_apply_remote_create_doc_version(self, sync_engine, db_instance, mocker):
        """Verify applying 'create' for DocumentVersions."""
        # 1. Setup parent Media record
        media_uuid = str(uuid.uuid4())
        db_instance.execute_query(
            "INSERT INTO Media (uuid, title, type, content_hash, version, client_id) VALUES (?, ?, ?, ?, 1, ?)",
            (media_uuid, "Media For DocVersion", "doc", uuid.uuid4().hex, db_instance.client_id), commit=True
        )
        media_id = get_db_state(db_instance, "Media", media_uuid)['id']

        # 2. Prepare remote change
        docver_uuid = str(uuid.uuid4())
        payload = {
            "uuid": docver_uuid,
            "media_uuid": media_uuid,
            # Need parent UUID to find parent ID locally (though not used by apply logic itself)
            "version_number": 1,  # Logical version number for this media
            "content": "Document version 1 content.",
            "prompt": "Initial prompt",
            "analysis_content": "Initial analysis",
            "deleted": False
        }
        change = generate_sync_log_entry(
            change_id=7, entity="DocumentVersions", entity_uuid=docver_uuid, operation="create",
            version=1, payload=payload, client_id="docver_creator"  # Sync version is 1
        )
        mocker.patch('requests.post')
        mocker.patch('requests.get',
                     return_value=create_mock_response(200, json_data={"changes": [change], "latest_change_id": 7}))

        # 3. Run sync
        sync_engine.run_sync_cycle()

        # 4. Check DB state
        # Need to query DocumentVersions using the known media_id and version_number or uuid
        cursor = db_instance.execute_query("SELECT * FROM DocumentVersions WHERE uuid = ?", (docver_uuid,))
        db_state = dict(cursor.fetchone())

        assert db_state is not None
        assert db_state['media_id'] == media_id
        assert db_state['version_number'] == 1
        assert db_state['content'] == "Document version 1 content."
        assert db_state['prompt'] == "Initial prompt"
        assert db_state['analysis_content'] == "Initial analysis"
        assert db_state['version'] == 1  # Sync version
        assert db_state['client_id'] == "docver_creator"
        assert db_state['deleted'] == 0
        assert sync_engine.last_server_log_id_processed == 7

    # --- Tests for Edge Case Payloads ---

    def test_apply_payload_edge_cases(self, sync_engine, db_instance, mocker):
        """Test applying payloads with None, empty strings, and booleans."""
        media_uuid = str(uuid.uuid4())
        content_hash = uuid.uuid4().hex
        payload = {
            "uuid": media_uuid,
            "title": "",  # Empty string
            "type": "test_edge",
            "content": "Some content",
            "content_hash": content_hash,
            "author": None,  # Null value
            "url": None,  # Null value
            "transcription_model": None,  # Null value
            "deleted": False,
            "is_trash": True,  # Boolean True
            "trash_date": None,  # Null date
            "prev_version": 1,  # Example value
            "merge_parent_uuid": None  # Null UUID
        }
        change = generate_sync_log_entry(
            change_id=8, entity="Media", entity_uuid=media_uuid, operation="create",
            version=2, payload=payload, client_id="edge_case_creator"  # Assume based on v1
        )
        mocker.patch('requests.post')
        mocker.patch('requests.get',
                     return_value=create_mock_response(200, json_data={"changes": [change], "latest_change_id": 8}))

        sync_engine.run_sync_cycle()

        db_state = get_db_state(db_instance, "Media", media_uuid)
        assert db_state is not None
        assert db_state['title'] == ""  # Stored as empty string
        assert db_state['author'] is None  # Stored as NULL
        assert db_state['url'] is None  # Stored as NULL
        assert db_state['transcription_model'] is None  # Stored as NULL
        assert db_state['is_trash'] == 1  # Stored as 1
        assert db_state['trash_date'] is None  # Stored as NULL
        assert db_state['prev_version'] == 1  # Stored correctly
        assert db_state['merge_parent_uuid'] is None  # Stored as NULL
        assert db_state['version'] == 2
        assert db_state['client_id'] == "edge_case_creator"
        assert sync_engine.last_server_log_id_processed == 8

    # --- Tests for Complex Conflicts ---

    def test_conflict_local_delete_remote_update(self, sync_engine, db_instance, mocker):
        """Local deletes V1, Remote updates V2 (based on V1)."""
        # 1. Setup initial V1 state
        kw_uuid = str(uuid.uuid4())
        t1 = time.time() - 20
        ts1 = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t1))
        db_instance.execute_query(
            "INSERT INTO Keywords (keyword, uuid, last_modified, version, client_id, deleted) VALUES (?, ?, ?, 1, ?, 0)",
            ("conflict_target", kw_uuid, ts1, "original_client"), commit=True
        )

        # 2. Simulate local delete (creates V2 locally, timestamp T2)
        t2 = t1 + 10
        ts2_local = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t2))
        db_instance.execute_query(
            "UPDATE Keywords SET deleted = 1, last_modified = ?, version = 2, client_id = ? WHERE uuid = ?",
            (ts2_local, db_instance.client_id, kw_uuid), commit=True
        )
        # Don't push this local change yet for the test setup
        sync_engine.last_local_log_id_sent = db_instance.get_sync_log_entries(since_change_id=0)[-1]['change_id']
        sync_engine.last_server_log_id_processed = 0  # Assume client hasn't processed server changes based on V1

        # 3. Remote update V2 (based on V1), with timestamp T3
        t3 = t1 + 5  # T3 is OLDER than local delete T2
        ts3_server_older = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t3))
        payload_update = {"keyword": "remote_update_vs_local_delete"}
        remote_update = generate_sync_log_entry(
            change_id=9, entity="Keywords", entity_uuid=kw_uuid, operation="update",
            version=2, payload=payload_update, client_id="updater", server_timestamp=ts3_server_older
        )

        mocker.patch('requests.post')  # Assume no other local changes to push
        mock_get = mocker.patch('requests.get', return_value=create_mock_response(200,
                                                                                  json_data={"changes": [remote_update],
                                                                                             "latest_change_id": 9}))

        # 4. Run sync
        sync_engine.run_sync_cycle()

        # 5. Assert: Conflict detected. Local delete (T2) wins LWW vs remote update (T3). Record remains deleted.
        db_state = get_db_state(db_instance, "Keywords", kw_uuid)
        assert db_state is not None
        assert db_state['deleted'] == 1  # Still deleted
        assert db_state['version'] == 2  # Local version
        assert db_state['last_modified'] == ts2_local  # Local timestamp
        assert db_state['client_id'] == db_instance.client_id  # Local client ID
        assert sync_engine.last_server_log_id_processed == 9  # Processed marker updated

        # --- Now test scenario where remote update timestamp is NEWER ---
        # Reset processed marker
        sync_engine.last_server_log_id_processed = 0
        sync_engine._save_sync_state()

        t4 = t2 + 5  # T4 is NEWER than local delete T2
        ts4_server_newer = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t4))
        remote_update_newer = generate_sync_log_entry(
            change_id=10, entity="Keywords", entity_uuid=kw_uuid, operation="update",
            version=2, payload=payload_update, client_id="updater", server_timestamp=ts4_server_newer
        )
        mocker.patch('requests.get', return_value=create_mock_response(200, json_data={"changes": [remote_update_newer],
                                                                                       "latest_change_id": 10}))

        sync_engine.run_sync_cycle()

        # Assert: Conflict detected. Remote update (T4) wins LWW. Record is force-updated (and implicitly undeleted by the update).
        db_state_new = get_db_state(db_instance, "Keywords", kw_uuid)
        assert db_state_new is not None
        assert db_state_new['deleted'] == 0  # Undeleted by winning update
        assert db_state_new['keyword'] == "remote_update_vs_local_delete"
        assert db_state_new['version'] == 2  # Version becomes 2 (from remote change)
        assert db_state_new['last_modified'] == ts4_server_newer  # Remote server timestamp
        assert db_state_new['client_id'] == "updater"  # Remote client ID
        assert sync_engine.last_server_log_id_processed == 10

    def test_conflict_local_update_remote_delete(self, sync_engine, db_instance, mocker):
        """Local updates to V2, Remote deletes V2 (based on V1)."""
        # 1. Setup initial V1 state
        kw_uuid = str(uuid.uuid4())
        t1 = time.time() - 20
        ts1 = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t1))
        db_instance.execute_query(
            "INSERT INTO Keywords (keyword, uuid, last_modified, version, client_id, deleted) VALUES (?, ?, ?, 1, ?, 0)",
            ("conflict_target2", kw_uuid, ts1, "original_client"), commit=True
        )

        # 2. Simulate local update (creates V2 locally, timestamp T2)
        t2 = t1 + 10
        ts2_local = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t2))
        db_instance.execute_query(
            "UPDATE Keywords SET keyword = ?, last_modified = ?, version = 2, client_id = ? WHERE uuid = ?",
            ("local_update_v2", ts2_local, db_instance.client_id, kw_uuid), commit=True
        )
        sync_engine.last_local_log_id_sent = db_instance.get_sync_log_entries(since_change_id=0)[-1]['change_id']
        sync_engine.last_server_log_id_processed = 0

        # 3. Remote delete V2 (based on V1), with timestamp T3
        t3 = t1 + 5  # T3 is OLDER than local update T2
        ts3_server_older = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t3))
        payload_delete = {"uuid": kw_uuid}
        remote_delete = generate_sync_log_entry(
            change_id=11, entity="Keywords", entity_uuid=kw_uuid, operation="delete",
            version=2, payload=payload_delete, client_id="deleter", server_timestamp=ts3_server_older
        )

        mocker.patch('requests.post')
        mock_get = mocker.patch('requests.get', return_value=create_mock_response(200,
                                                                                  json_data={"changes": [remote_delete],
                                                                                             "latest_change_id": 11}))

        # 4. Run sync
        sync_engine.run_sync_cycle()

        # 5. Assert: Conflict detected. Local update (T2) wins LWW vs remote delete (T3). Record remains updated.
        db_state = get_db_state(db_instance, "Keywords", kw_uuid)
        assert db_state is not None
        assert db_state['deleted'] == 0  # Not deleted
        assert db_state['keyword'] == "local_update_v2"  # Local update kept
        assert db_state['version'] == 2
        assert db_state['last_modified'] == ts2_local
        assert db_state['client_id'] == db_instance.client_id
        assert sync_engine.last_server_log_id_processed == 11

        # --- Now test scenario where remote delete timestamp is NEWER ---
        sync_engine.last_server_log_id_processed = 0
        sync_engine._save_sync_state()

        t4 = t2 + 5  # T4 is NEWER than local update T2
        ts4_server_newer = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(t4))
        remote_delete_newer = generate_sync_log_entry(
            change_id=12, entity="Keywords", entity_uuid=kw_uuid, operation="delete",
            version=2, payload=payload_delete, client_id="deleter", server_timestamp=ts4_server_newer
        )
        mocker.patch('requests.get', return_value=create_mock_response(200, json_data={"changes": [remote_delete_newer],
                                                                                       "latest_change_id": 12}))

        sync_engine.run_sync_cycle()

        # Assert: Conflict detected. Remote delete (T4) wins LWW. Record is force-deleted.
        db_state_new = get_db_state(db_instance, "Keywords", kw_uuid)
        assert db_state_new is not None
        assert db_state_new['deleted'] == 1  # Is deleted
        assert db_state_new['version'] == 2  # Version becomes 2 (from remote delete)
        assert db_state_new['last_modified'] == ts4_server_newer  # Remote server timestamp
        assert db_state_new['client_id'] == "deleter"  # Remote client ID
        assert sync_engine.last_server_log_id_processed == 12

    # --- Batching Tests ---

    def test_push_large_batch(self, sync_engine, db_instance, mocker):
        """Verify pushing changes halts at SYNC_BATCH_SIZE and resumes."""
        num_changes = SYNC_BATCH_SIZE + 5
        # Create local changes
        for i in range(num_changes):
            db_instance.add_keyword(f"batch_push_kw_{i}")

        local_logs = db_instance.get_sync_log_entries(since_change_id=0)
        assert len(local_logs) == num_changes

        # Mock server responses
        mock_post = mocker.patch('requests.post', return_value=create_mock_response(200))
        # Mock pull to return nothing for simplicity
        mocker.patch('requests.get',
                     return_value=create_mock_response(200, json_data={"changes": [], "latest_change_id": 0}))

        # Run first sync cycle - should push only BATCH_SIZE
        sync_engine.run_sync_cycle()
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert len(kwargs['json']['changes']) == SYNC_BATCH_SIZE
        first_batch_last_id = kwargs['json']['changes'][-1]['change_id']
        assert sync_engine.last_local_log_id_sent == first_batch_last_id

        # Run second sync cycle - should push the remaining changes
        mock_post.reset_mock()  # Reset call count for next assertion
        sync_engine.run_sync_cycle()
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert len(kwargs['json']['changes']) == num_changes - SYNC_BATCH_SIZE  # Remaining 5
        second_batch_last_id = kwargs['json']['changes'][-1]['change_id']
        assert sync_engine.last_local_log_id_sent == second_batch_last_id
        assert second_batch_last_id == local_logs[-1]['change_id']

    def test_pull_large_batch(self, sync_engine, db_instance, mocker):
        """Verify pulling and applying changes across multiple batches."""
        num_changes = SYNC_BATCH_SIZE + 7
        changes_batch1 = []
        changes_batch2 = []
        all_uuids = []

        # Generate mock server changes
        for i in range(num_changes):
            kw_uuid = str(uuid.uuid4())
            all_uuids.append(kw_uuid)
            change = generate_sync_log_entry(
                change_id=i + 1,  # Change IDs 1 to N
                entity="Keywords", entity_uuid=kw_uuid, operation="create",
                version=1, payload={"keyword": f"batch_pull_kw_{i}"}
            )
            if i < SYNC_BATCH_SIZE:
                changes_batch1.append(change)
            else:
                changes_batch2.append(change)

        # Mock server responses - first call returns batch 1, second call returns batch 2
        mock_post = mocker.patch('requests.post')
        mock_get = mocker.patch('requests.get')
        mock_get.side_effect = [
            create_mock_response(200, json_data={"changes": changes_batch1, "latest_change_id": num_changes}),
            create_mock_response(200, json_data={"changes": changes_batch2, "latest_change_id": num_changes}),
            create_mock_response(200, json_data={"changes": [], "latest_change_id": num_changes})
            # Third call sees no more changes
        ]

        # Run first sync cycle - processes batch 1
        sync_engine.run_sync_cycle()
        assert sync_engine.last_server_log_id_processed == changes_batch1[-1]['change_id']
        assert get_db_state(db_instance, "Keywords",
                            changes_batch1[-1]['entity_uuid']) is not None  # Last item of batch 1 exists
        assert get_db_state(db_instance, "Keywords",
                            changes_batch2[0]['entity_uuid']) is None  # First item of batch 2 doesn't exist yet

        # Run second sync cycle - processes batch 2
        sync_engine.run_sync_cycle()
        assert sync_engine.last_server_log_id_processed == changes_batch2[-1]['change_id']  # Marker updated to last ID
        assert get_db_state(db_instance, "Keywords",
                            changes_batch2[0]['entity_uuid']) is not None  # First item of batch 2 now exists
        assert get_db_state(db_instance, "Keywords",
                            changes_batch2[-1]['entity_uuid']) is not None  # Last item of batch 2 now exists

        # Run third sync cycle - no more changes
        sync_engine.run_sync_cycle()
        assert sync_engine.last_server_log_id_processed == num_changes  # Still at the last ID
        assert mock_get.call_count == 3

    # --- Initial Sync Test ---

    def test_initial_sync_empty_client_db(self, sync_engine, db_instance, mocker):
        """Test client starting with empty DB pulling initial state."""
        # Client state starts at 0, DB is empty (assured by fixture)
        assert sync_engine.last_local_log_id_sent == 0
        assert sync_engine.last_server_log_id_processed == 0
        cursor = db_instance.execute_query("SELECT COUNT(*) FROM Media")
        assert cursor.fetchone()[0] == 0
        cursor = db_instance.execute_query("SELECT COUNT(*) FROM Keywords")
        assert cursor.fetchone()[0] == 0

        # Mock server response with initial data
        media1_uuid = str(uuid.uuid4())
        kw1_uuid = str(uuid.uuid4())
        changes = [
            generate_sync_log_entry(1, "Keywords", kw1_uuid, "create", 1, {"keyword": "initial_kw"}),
            generate_sync_log_entry(2, "Media", media1_uuid, "create", 1, {
                "title": "Initial Media", "type": "init", "content_hash": uuid.uuid4().hex, "uuid": media1_uuid
            })
        ]
        mock_post = mocker.patch('requests.post')
        mock_get = mocker.patch('requests.get', return_value=create_mock_response(200, json_data={"changes": changes,
                                                                                                  "latest_change_id": 2}))

        # Run sync
        sync_engine.run_sync_cycle()

        # Assert: Items created locally, state updated
        assert get_db_state(db_instance, "Keywords", kw1_uuid) is not None
        assert get_db_state(db_instance, "Media", media1_uuid) is not None
        assert get_db_state(db_instance, "Media", media1_uuid)['title'] == "Initial Media"
        assert sync_engine.last_server_log_id_processed == 2
        assert sync_engine.last_local_log_id_sent == 0  # No local changes were pushed

# Add more tests for:
# - Different entities (Media, Transcripts, etc.) ensuring all fields are handled.
# - Edge cases in payloads (null values, empty strings, boolean fields).
# - More complex conflict scenarios (e.g., delete vs update).
# - Handling of `prev_version` and `merge_parent_uuid` if/when that logic is added.
# - Large batches (pushing/pulling > SYNC_BATCH_SIZE).
# - Initial sync (client starts with empty DB and pulls all history).