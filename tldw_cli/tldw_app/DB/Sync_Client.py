# sync_client.py
import time
import requests
import json
import logging
import os
import sqlite3 # For specific error types
from typing import List, Dict, Optional, Tuple
#
# Third-Party Imports
#
# Local Imports
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import Database, ConflictError, DatabaseError, InputError
try:
    from Media_DB import Database, ConflictError, DatabaseError, InputError
except ImportError:
    print("ERROR: Could not import the 'Media_DB' library. Make sure Media_DB.py is accessible.")
#
#######################################################################################################################
#
# Functions:

# --- Logging Setup ---
# Configure this as needed for your client application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ClientSyncEngine")

# --- Configuration ---
# These should ideally come from a config file or environment variables
SERVER_API_URL = "http://127.0.0.1:5000" # Replace with your actual server URL
SYNC_ENDPOINT_SEND = "/sync/send"
SYNC_ENDPOINT_GET = "/sync/get"
STATE_FILE = "client_sync_state.json" # File to store last sync IDs
CLIENT_ID = "client_abc_123" # Needs to be unique per client instance
DATABASE_PATH = f"./client_dbs/{CLIENT_ID}_media.db" # Example: One DB per client
SYNC_BATCH_SIZE = 50 # How many changes to send/receive at once
SYNC_INTERVAL_SECONDS = 60 # How often to run the sync cycle automatically


class ClientSyncEngine:
    """
    Manages the synchronization process for a client's local database
    with a central server.
    """

    def __init__(self, db_instance: Database, server_api_url: str, client_id: str, state_file: str):
        if not isinstance(db_instance, Database):
             raise TypeError("db_instance must be a valid Database object.")

        self.db = db_instance
        self.server_api_url = server_api_url.rstrip('/') # Ensure no trailing slash
        self.client_id = client_id
        self.state_file = state_file

        # Load persistent sync state
        self.last_local_log_id_sent: int = 0
        self.last_server_log_id_processed: int = 0
        self._load_sync_state()

        logger.info(f"ClientSyncEngine initialized for client '{self.client_id}'.")
        logger.info(f"  DB Path: {self.db.db_path_str}")
        logger.info(f"  Server URL: {self.server_api_url}")
        logger.info(f"  Initial State: Last Sent={self.last_local_log_id_sent}, Last Processed={self.last_server_log_id_processed}")

    # --- State Management ---

    def _load_sync_state(self):
        """Loads the last sync IDs from the state file."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.last_local_log_id_sent = state.get('last_local_log_id_sent', 0)
                    self.last_server_log_id_processed = state.get('last_server_log_id_processed', 0)
                    logger.debug(f"Loaded sync state from {self.state_file}")
            else:
                logger.info(f"State file {self.state_file} not found, starting from scratch.")
                # Ensure the initial save happens if the file doesn't exist
                self._save_sync_state()
        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.error(f"Error loading sync state from {self.state_file}: {e}. Starting from scratch.", exc_info=True)
            self.last_local_log_id_sent = 0
            self.last_server_log_id_processed = 0
            # Attempt to save a clean initial state
            self._save_sync_state()


    def _save_sync_state(self):
        """Saves the current sync IDs to the state file."""
        state = {
            'last_local_log_id_sent': self.last_local_log_id_sent,
            'last_server_log_id_processed': self.last_server_log_id_processed
        }
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.state_file) or '.', exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=4)
            logger.debug(f"Saved sync state to {self.state_file}: {state}")
        except IOError as e:
            logger.error(f"Error saving sync state to {self.state_file}: {e}", exc_info=True)

    # --- Core Sync Logic ---

    def run_sync_cycle(self):
        """Performs one full sync cycle: push local changes, then pull remote changes."""
        logger.info(f"Starting sync cycle [Client ID: {self.client_id}]...")
        network_error = False
        try:
            # 1. Push Local Changes
            self._push_local_changes()

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during push phase: {e}")
            network_error = True # Don't proceed to pull if push failed due to network
        except Exception as e:
            logger.error(f"Unexpected error during push phase: {e}", exc_info=True)
            # Decide if we should attempt pull phase even if push had non-network error

        if not network_error:
            try:
                # 2. Pull and Apply Remote Changes
                self._pull_and_apply_remote_changes()

            except requests.exceptions.RequestException as e:
                logger.error(f"Network error during pull phase: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during pull/apply phase: {e}", exc_info=True)
        else:
            logger.warning("Skipping pull phase due to earlier network error during push.")

        logger.info(f"Sync cycle finished. Current state: Last Sent={self.last_local_log_id_sent}, Last Processed={self.last_server_log_id_processed}")

    def _push_local_changes(self):
        """Fetches local changes from sync_log and sends them to the server."""
        logger.debug(f"Checking for local changes since log ID {self.last_local_log_id_sent}")
        try:
            local_changes = self.db.get_sync_log_entries(
                since_change_id=self.last_local_log_id_sent,
                limit=SYNC_BATCH_SIZE
            )
        except DatabaseError as e:
            logger.error(f"Failed to get local sync log entries: {e}", exc_info=True)
            return # Don't proceed if we can't read the log

        if not local_changes:
            logger.info("No local changes to push.")
            return

        logger.info(f"Found {len(local_changes)} local changes to push.")

        payload = {
            'client_id': self.client_id,
            'changes': local_changes,
            # Include the last processed server ID so server knows our state
            'last_processed_server_id': self.last_server_log_id_processed
        }

        try:
            # TODO: Add authentication headers (e.g., Bearer token)
            # headers = {'Authorization': f'Bearer {your_auth_token}'}
            headers = {'Content-Type': 'application/json'}
            full_url = f"{self.server_api_url}{SYNC_ENDPOINT_SEND}"
            logger.debug(f"Posting {len(local_changes)} changes to {full_url}")

            response = requests.post(full_url, json=payload, headers=headers, timeout=45)
            response.raise_for_status() # Raises HTTPError for 4xx/5xx responses

            # If successful, update the marker
            new_last_sent = local_changes[-1]['change_id']
            self.last_local_log_id_sent = new_last_sent
            self._save_sync_state()
            logger.info(f"Successfully pushed {len(local_changes)} changes. Last sent ID updated to {new_last_sent}.")

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error pushing changes: {e.response.status_code} - {e.response.text}")
            # Do NOT update last_local_log_id_sent if push fails
        # Let RequestException (network errors) be caught by run_sync_cycle

    def _pull_and_apply_remote_changes(self):
        """Pulls changes from the server and applies them locally."""
        logger.info(f"Pulling remote changes since server log ID {self.last_server_log_id_processed}.")

        try:
            # TODO: Add authentication headers
            headers = {'Accept': 'application/json'}
            params = {
                'client_id': self.client_id,
                'since_change_id': self.last_server_log_id_processed
            }
            full_url = f"{self.server_api_url}{SYNC_ENDPOINT_GET}"
            logger.debug(f"Getting changes from {full_url} with params {params}")

            response = requests.get(full_url, params=params, headers=headers, timeout=45)
            response.raise_for_status()

            sync_data = response.json()
            remote_changes = sync_data.get('changes', [])
            # Server should tell us its latest ID, even if no changes sent for us
            server_latest_id = sync_data.get('latest_change_id', self.last_server_log_id_processed)

            if not remote_changes:
                logger.info("No new remote changes received.")
                # Fast-forward marker if server is ahead but had no changes *for us*
                if server_latest_id > self.last_server_log_id_processed:
                    logger.info(f"Updating processed ID to server's latest known ID: {server_latest_id}")
                    self.last_server_log_id_processed = server_latest_id
                    self._save_sync_state()
                return

            logger.info(f"Received {len(remote_changes)} remote changes from server.")

            # Apply the received changes
            processed_successfully = self._apply_remote_changes_batch(remote_changes)

            if processed_successfully:
                # Update marker to the ID of the last successfully processed change
                new_last_processed = remote_changes[-1]['change_id']
                self.last_server_log_id_processed = new_last_processed
                self._save_sync_state()
                logger.info(f"Successfully processed remote changes. Last processed ID updated to {new_last_processed}.")
            else:
                logger.error("Failed to apply one or more remote changes in the batch. Last processed ID NOT updated.")
                # Consider: Implement partial batch success handling? (More complex)

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error pulling changes: {e.response.status_code} - {e.response.text}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON response from server: {e}")
        # Let RequestException (network errors) be caught by run_sync_cycle


    def _apply_remote_changes_batch(self, changes: List[Dict]) -> bool:
        """
        Applies a batch of ordered changes received from the server within a single transaction.
        Returns True if the entire batch was applied successfully (or skipped idempotently), False otherwise.
        """
        all_applied_or_skipped = True
        # Ensure changes are sorted by server's change_id just in case
        changes.sort(key=lambda x: x['change_id'])

        try:
            with self.db.transaction() as conn: # Use the library's transaction context
                cursor = conn.cursor() # Get cursor from the transaction-managed connection
                for change in changes:
                    try:
                        # Check if change_id is as expected (optional robustness check)
                        # if change['change_id'] <= self.last_server_log_id_processed: continue

                        self._apply_single_change(cursor, change)

                    except ConflictError as cf_err: # Catch conflicts raised by _apply_single_change
                         logger.warning(f"Conflict occurred applying change ID {change['change_id']}: {cf_err}. Attempting resolution.")
                         resolved = self._resolve_conflict(cursor, change, cf_err)
                         if not resolved:
                              logger.error(f"Conflict resolution failed for change ID {change['change_id']}. Rolling back batch.")
                              all_applied_or_skipped = False
                              raise cf_err # Re-raise to trigger transaction rollback

                    except (DatabaseError, sqlite3.Error, json.JSONDecodeError, KeyError, InputError) as item_error:
                        logger.error(f"Failed to apply change ID {change['change_id']} ({change['entity']} {change['operation']}): {item_error}", exc_info=True)
                        all_applied_or_skipped = False
                        raise item_error # Re-raise to trigger transaction rollback

            # If the loop completes without exceptions, the transaction commits here.
            logger.info("Batch of remote changes applied transactionally.")

        except (DatabaseError, sqlite3.Error, ConflictError) as e:
            logger.error(f"Transaction rolled back while applying remote changes batch: {e}")
            all_applied_or_skipped = False
        except Exception as e:
             logger.error(f"Unexpected error during remote changes batch application: {e}", exc_info=True)
             all_applied_or_skipped = False

        return all_applied_or_skipped

    def _apply_single_change(self, cursor: sqlite3.Cursor, change: Dict):
        """
        Applies a single change record. Raises ConflictError if optimistic lock fails.
        This is called within the transaction managed by _apply_remote_changes_batch.
        """
        entity = change['entity']
        entity_uuid = change['entity_uuid']
        operation = change['operation']
        remote_version = change['version']
        remote_client_id = change['client_id']
        # Use server's timestamp; fall back to original if missing (shouldn't happen ideally)
        authoritative_timestamp = change.get('server_timestamp', change['timestamp'])
        payload_str = change.get('payload')
        if not payload_str:
             # Handle link/unlink which might not have payload in simple cases
             if entity == "MediaKeywords" and operation in ['link', 'unlink']:
                  payload = {} # Or handle based on uuids if needed
             else:
                  logger.error(f"Missing payload for change ID {change['change_id']} ({entity} {operation})")
                  raise ValueError("Change record missing payload")
        else:
            payload = json.loads(payload_str)

        # --- Idempotency Check ---
        # Check local version *before* executing the change
        cursor.execute(f"SELECT version FROM {entity} WHERE uuid = ?", (entity_uuid,))
        local_record = cursor.fetchone()
        local_version = local_record[0] if local_record else 0 # SQLite Row access by index

        # If remote version isn't newer, or it's a create for existing UUID (shouldn't happen often), skip.
        if remote_version <= local_version:
            # Also check if it's a 'create' operation trying to overwrite an existing UUID
            if operation == 'create' and local_version > 0:
                 logger.warning(f"Skipping remote 'create' for existing {entity} UUID {entity_uuid} (RemoteVer: {remote_version}, LocalVer: {local_version})")
            else:
                 logger.debug(f"Skipping old/duplicate change for {entity} UUID {entity_uuid} (RemoteVer: {remote_version}, LocalVer: {local_version})")
            return # Successfully skipped

        # --- Execute the change ---
        # Pass force_apply=False for standard application with optimistic locking
        logger.debug(f"Attempting to apply: {operation} on {entity} UUID {entity_uuid} (RemoteVer: {remote_version}, LocalVer: {local_version})")
        self._execute_change_sql(cursor, entity, operation, payload, entity_uuid, remote_version, remote_client_id, authoritative_timestamp, force_apply=False)


    def _resolve_conflict(self, cursor: sqlite3.Cursor, change: Dict, conflict_error: ConflictError) -> bool:
        """
        Attempts to resolve a conflict based on the chosen strategy (LWW).
        Returns True if resolved (applied or skipped), False if resolution failed.
        """
        entity = change['entity']
        entity_uuid = change['entity_uuid']
        operation = change['operation']
        remote_version = change['version']
        remote_client_id = change['client_id']
        authoritative_timestamp = change.get('server_timestamp', change['timestamp'])
        payload_str = change.get('payload')
        payload = json.loads(payload_str) if payload_str else {}

        # --- LWW Conflict Resolution ---
        try:
            cursor.execute(f"SELECT last_modified FROM {entity} WHERE uuid = ?", (entity_uuid,))
            local_ts_row = cursor.fetchone()
            # Use a very old timestamp if record somehow doesn't exist (shouldn't happen in conflict)
            local_timestamp = local_ts_row[0] if local_ts_row else '1970-01-01 00:00:00'

            # Compare server's authoritative timestamp with the local record's timestamp
            if authoritative_timestamp >= local_timestamp:
                logger.warning(f"  Resolving Conflict (LWW): Remote change wins (RemoteTS: {authoritative_timestamp} >= LocalTS: {local_timestamp}). Forcing apply.")
                # Force apply the change, overwriting local concurrent changes
                self._execute_change_sql(cursor, entity, operation, payload, entity_uuid, remote_version, remote_client_id, authoritative_timestamp, force_apply=True)
                return True # Resolved by applying remote change
            else:
                logger.warning(f"  Resolving Conflict (LWW): Local change wins (RemoteTS: {authoritative_timestamp} < LocalTS: {local_timestamp}). Skipping remote change.")
                # Skip applying the remote change; local state is kept
                return True # Resolved by skipping remote change

        except Exception as e:
            logger.error(f"Error during LWW conflict resolution for {entity} {entity_uuid}: {e}", exc_info=True)
            return False # Resolution failed

    def _execute_change_sql(self, cursor: sqlite3.Cursor, entity: str, operation: str, payload: Dict, uuid: str,
                            version: int, client_id: str, timestamp: str, force_apply: bool = False):
        """
        Generates and executes SQL to apply a single change operation locally.
        Raises ConflictError if optimistic lock fails (when force_apply=False).
        Raises DatabaseError or sqlite3.Error on other DB issues.
        Raises ValueError for invalid operations or missing data.
        """
        logger.debug(
            f"Executing SQL for: Op='{operation}', Entity='{entity}', UUID='{uuid}', Ver='{version}', Force='{force_apply}'")

        # --- Special handling for MediaKeywords junction table ---
        if entity == "MediaKeywords":
            return self._execute_media_keyword_sql(cursor, operation, payload)

        # --- Standard handling for main entities ---
        table_columns = self._get_table_columns(cursor, entity)
        if not table_columns:
            # Error logged in helper
            raise DatabaseError(f"Cannot proceed: Failed to get columns for entity '{entity}'.")

        # Base optimistic lock clause (used for update/delete unless forcing)
        optimistic_lock_sql = ""
        optimistic_lock_param = []
        expected_base_version = version - 1
        if not force_apply and operation in ['update', 'delete']:
            # Only apply lock if the expected base version is > 0 (i.e., not creating)
            if expected_base_version > 0:
                optimistic_lock_sql = " AND version = ?"
                optimistic_lock_param = [expected_base_version]
            elif operation == 'update':
                # This is an update targeting version 1 - means it should be an insert or was already created
                # If forcing, we proceed. If not forcing, this state is unusual but might occur if
                # a 'create' was missed. We'll allow the UPDATE WHERE uuid=? to proceed,
                # but it might affect 0 rows if the record truly doesn't exist.
                logger.debug(f"Applying update for version 1 (no prior version lock) for {entity} {uuid}")
            # Delete operations inherently target existing records, so the lock applies if expected_base > 0

        sql = ""
        params_tuple = tuple()

        # --- Handle Create ---
        if operation == 'create':
            cols_sql = []
            placeholders_sql = []
            params_list = []
            # Merge payload with core sync metadata that needs to be explicitly inserted
            core_sync_meta = {'uuid': uuid, 'last_modified': timestamp, 'version': version, 'client_id': client_id}
            # Default 'deleted' to 0 for creates if not specified otherwise
            all_data = {**payload, **core_sync_meta, 'deleted': payload.get('deleted', 0)}

            for col in table_columns:
                if col == 'id': continue  # Skip auto-increment primary key

                if col in all_data:
                    value = all_data[col]
                    # Handle boolean conversion for SQLite (stores as 0/1)
                    if isinstance(value, bool):
                        value = 1 if value else 0
                    cols_sql.append(f"`{col}`")  # Use backticks for safety
                    placeholders_sql.append("?")
                    params_list.append(value)
                # Add default handling here if certain columns MUST have a value
                # elif col == 'some_required_col':
                #    cols_sql.append(f"`{col}`"); placeholders_sql.append("?"); params_list.append(DEFAULT_VALUE)

            if not cols_sql:
                raise ValueError(
                    f"Cannot build INSERT for {entity} {uuid}: No columns determined from payload and schema.")

            sql = f"INSERT INTO `{entity}` ({', '.join(cols_sql)}) VALUES ({', '.join(placeholders_sql)})"
            params_tuple = tuple(params_list)

        # --- Handle Update ---
        elif operation == 'update':
            set_clauses = []
            params_list = []
            for col in table_columns:
                # Update only fields present in the payload, excluding keys/uuid
                if col in payload and col not in ['id', 'uuid']:
                    value = payload[col]
                    if isinstance(value, bool): value = 1 if value else 0
                    set_clauses.append(f"`{col}` = ?")
                    params_list.append(value)

            # Always update core sync metadata and deleted status on any update operation
            set_clauses.extend([
                "`last_modified` = ?", "`version` = ?", "`client_id` = ?",
                "`deleted` = ?"  # Explicitly set deleted status based on payload
            ])
            params_list.extend([
                timestamp, version, client_id,
                1 if payload.get('deleted', False) else 0  # Default to not deleted if key missing
            ])

            if not set_clauses:
                # This condition should ideally not be met because metadata always updates
                logger.warning(f"Update operation for {entity} {uuid} resulted in no SET clauses (unexpected).")
                # We might just return here, effectively skipping the update if only metadata changed
                # but the payload was empty. Or proceed to just update metadata. Let's proceed.
                # This warning indicates potential upstream issue where empty payloads are sent for updates.
                # raise ValueError(f"Cannot build UPDATE for {entity} {uuid}: No fields to update.")
                pass  # Proceed to execute metadata-only update

            sql = f"UPDATE `{entity}` SET {', '.join(set_clauses)} WHERE uuid = ?"
            params_list.append(uuid)
            sql += optimistic_lock_sql
            params_list.extend(optimistic_lock_param)
            params_tuple = tuple(params_list)

        # --- Handle Delete (Soft Delete) ---
        elif operation == 'delete':
            # Delete is an update setting the deleted flag and sync meta
            sql = f"UPDATE `{entity}` SET deleted = 1, last_modified = ?, version = ?, client_id = ? WHERE uuid = ?"
            params_list = [timestamp, version, client_id, uuid]
            sql += optimistic_lock_sql
            params_list.extend(optimistic_lock_param)
            params_tuple = tuple(params_list)

        # --- Invalid Operation ---
        else:
            raise ValueError(f"Unsupported operation '{operation}' received for entity '{entity}'")

        # --- Execute SQL ---
        try:
            logger.debug(f"Executing SQL: {sql} | Params: {params_tuple}")
            cursor.execute(sql, params_tuple)

            # --- Check Row Count for Optimistic Lock Failures (Update/Delete only) ---
            # Only check if we expected to apply a lock (not forcing, not creating)
            if operation in ['update', 'delete'] and not force_apply and optimistic_lock_sql:
                if cursor.rowcount == 0:
                    # If rowcount is 0, the WHERE clause (uuid + version lock) didn't match.
                    logger.warning(
                        f"Optimistic lock failed for {entity} UUID {uuid} (Op: {operation}, Expected Base Ver: {expected_base_version}). Rowcount 0.")
                    # Raise ConflictError to be caught by _apply_single_change or _resolve_conflict
                    raise ConflictError(
                        f"Optimistic lock failed applying change.",
                        entity=entity,
                        identifier=uuid
                    )
                else:
                    logger.debug(
                        f"Optimistic lock successful for {entity} {uuid} (Op: {operation}). Rowcount {cursor.rowcount}.")

        except sqlite3.IntegrityError as ie:
            # Could be unique constraint violation (e.g., trying to create duplicate UUID/hash,
            # or update fails FK constraint if a parent was deleted unexpectedly)
            logger.error(f"Integrity error applying change for {entity} {uuid}: {ie}", exc_info=True)
            raise DatabaseError(f"Integrity error applying change: {ie}") from ie
        except sqlite3.Error as e:
            # Catch other specific SQLite errors
            logger.error(f"SQLite error applying change for {entity} {uuid}: {e}", exc_info=True)
            raise  # Re-raise to be caught as DatabaseError or sqlite3.Error upstream

    def _execute_media_keyword_sql(self, cursor: sqlite3.Cursor, operation: str, payload: Dict):
        """Handles SQL execution specifically for the MediaKeywords junction table."""
        media_uuid = payload.get('media_uuid')
        keyword_uuid = payload.get('keyword_uuid')

        if not media_uuid or not keyword_uuid:
            raise ValueError(f"Missing media_uuid or keyword_uuid for MediaKeywords operation '{operation}'")

        # Look up local integer IDs using the provided UUIDs
        cursor.execute("SELECT id FROM Media WHERE uuid = ?", (media_uuid,))
        media_rec = cursor.fetchone()
        cursor.execute("SELECT id FROM Keywords WHERE uuid = ?", (keyword_uuid,))
        kw_rec = cursor.fetchone()

        if not media_rec:
            logger.warning(
                f"Skipping MediaKeywords {operation}: Parent Media record with UUID {media_uuid} not found locally.")
            return  # Skip if parent Media doesn't exist locally
        if not kw_rec:
            logger.warning(
                f"Skipping MediaKeywords {operation}: Parent Keyword record with UUID {keyword_uuid} not found locally.")
            return  # Skip if parent Keyword doesn't exist locally

        media_id_local = media_rec[0]  # Access by index for SQLite Row
        keyword_id_local = kw_rec[0]  # Access by index for SQLite Row

        sql = ""
        params_tuple = tuple()

        if operation == 'link':
            # Use INSERT OR IGNORE for idempotency. If the link already exists, it does nothing.
            sql = "INSERT OR IGNORE INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)"
            params_tuple = (media_id_local, keyword_id_local)
            logger.debug(f"Executing SQL: {sql} | Params: {params_tuple}")
            cursor.execute(sql, params_tuple)

        elif operation == 'unlink':
            # DELETE is naturally idempotent. If the link doesn't exist, it does nothing.
            sql = "DELETE FROM MediaKeywords WHERE media_id = ? AND keyword_id = ?"
            params_tuple = (media_id_local, keyword_id_local)
            logger.debug(f"Executing SQL: {sql} | Params: {params_tuple}")
            cursor.execute(sql, params_tuple)

        else:
            raise ValueError(f"Unsupported operation '{operation}' for MediaKeywords entity")

    # --- Helper to get table columns (cached for efficiency) ---
    _column_cache = {}

    def _get_table_columns(self, cursor: sqlite3.Cursor, table_name: str) -> Optional[List[str]]:
        """Gets column names for a table, using a simple cache."""
        if table_name in self._column_cache:
            return self._column_cache[table_name]
        try:
            # Ensure table name is safe (basic check, consider more robust quoting/validation if needed)
            if not table_name.replace('_', '').isalnum():
                raise ValueError(f"Invalid characters in table name: {table_name}")

            cursor.execute(f"PRAGMA table_info(`{table_name}`)")  # Use backticks
            columns = [row[1] for row in cursor.fetchall()]  # Column name is at index 1
            if columns:
                self._column_cache[table_name] = columns
                logger.debug(f"Cached columns for table '{table_name}': {columns}")
                return columns
            else:
                logger.error(f"Could not retrieve columns for table: {table_name}")
                return None
        except sqlite3.Error as e:
            logger.error(f"SQLite error getting columns for table {table_name}: {e}")
            return None
        except ValueError as e:
            logger.error(f"Error getting columns for table {table_name}: {e}")
            return None


# --- Example Usage ---
if __name__ == "__main__":
    logger.info("Setting up Client Sync Engine example...")

    # Ensure client directory exists
    os.makedirs(os.path.dirname(DATABASE_PATH) or '.', exist_ok=True)

    # Initialize db to None outside the try block
    db: Optional[Database] = None
    engine: Optional[ClientSyncEngine] = None

    try:
        # Initialize the database (creates schema if needed)
        db = Database(db_path=DATABASE_PATH, client_id=CLIENT_ID)
        logger.info("Database object created.")

        # Initialize the sync engine
        engine = ClientSyncEngine(
            db_instance=db,
            server_api_url=SERVER_API_URL,
            client_id=CLIENT_ID,
            state_file=STATE_FILE
        )
        logger.info("ClientSyncEngine object created.")

        # --- Simulate a local change ---
        try:
            logger.info("Simulating a local change (adding a keyword)...")
            # Use the Database instance methods directly for local changes
            db.add_keyword("test_sync_keyword") # 'db' is guaranteed to be Database object here if no exception occurred above
            logger.info("Local change presumably made and logged by sync_log trigger.")
        except Exception as e:
            logger.error(f"Error making simulated local change: {e}")


        # --- Run a sync cycle ---
        if engine: # Check if engine was successfully created
            logger.info("Running a sync cycle...")
            engine.run_sync_cycle()
            logger.info("Sync cycle complete.")
        else:
            logger.warning("Sync engine not initialized, skipping sync cycle.")

        # --- Simulate running periodically ---
        # In a real app, this would be in a loop or scheduled task
        # print(f"\nWill run next sync cycle in {SYNC_INTERVAL_SECONDS} seconds...")
        # time.sleep(SYNC_INTERVAL_SECONDS)
        # if engine: engine.run_sync_cycle()

    except Exception as main_err:
        logger.error(f"Critical error during client sync setup or execution: {main_err}", exc_info=True)

    finally:
        # Clean up DB connection if necessary
        # Now we just check if 'db' was successfully assigned an object
        if db:
             try:
                  db.close_connection() # Close the connection for this main thread
                  logger.info("Closed main thread DB connection.")
             except Exception as close_err:
                  logger.error(f"Error closing DB connection: {close_err}")
        else:
            logger.warning("DB object was not successfully initialized, no connection to close.")