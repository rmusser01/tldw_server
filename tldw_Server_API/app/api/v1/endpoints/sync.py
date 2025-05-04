# Server_API/app/api/v1/endpoints/sync-endpoint.py
# Description: This code provides a FastAPI endpoint for all Sync operations.
#
# Imports
import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Optional, List, Tuple
#
# 3rd-party imports
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
    UploadFile
)
# API Rate Limiter/Caching via Redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_db_for_user
#
# Local Imports
from tldw_Server_API.app.api.v1.schemas.sync_server_models import ClientChangesPayload, ServerChangesResponse, \
    SyncLogEntry
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
#
# DB Mgmt
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import DatabaseError, ConflictError, Database, InputError
from tldw_Server_API.app.core.Sync.Sync_Client import SYNC_BATCH_SIZE
#
#
#######################################################################################################################
#
# Functions:

# All functions below are endpoints callable via HTTP requests and the corresponding code executed as a result of it.
#
# The router is a FastAPI object that allows us to define multiple endpoints under a single prefix.
# Create a new router instance
router = APIRouter()

# Explanation and Key Server-Side Aspects:
#     FastAPI Structure: Uses APIRouter, Pydantic models (SyncLogEntry, ClientChangesPayload, ServerChangesResponse), and dependency injection (Depends) for authentication and database access.
#     User-Scoped DB: The get_db_for_user dependency is critical. It ensures that all operations within an endpoint call correctly target the database belonging to the authenticated user.
#
#     /sync/send Endpoint:
#         Receives changes via ClientChangesPayload.
#         Authenticates the user and gets their Database instance.
#         Uses ServerSyncProcessor to handle the batch application.
#         The processor uses a transaction to apply all changes atomically.
#         Authoritative Timestamp: It captures datetime.now(timezone.utc) once at the start of processing the batch (server_authoritative_timestamp) and uses this timestamp when calling _execute_server_change_sql.
#         Conflict Resolution (LWW): The _resolve_server_conflict method compares the server_authoritative_timestamp with the last_modified timestamp already present in the user's DB record. If the server's current time is newer or equal, the incoming change wins and is forcefully applied; otherwise, the existing server state wins, and the incoming change is skipped.
#         Originating Client ID: _execute_server_change_sql correctly uses the originating_client_id from the change record when updating the client_id column in the user's DB.
#         Returns success (200) or error (500/409).
#
#     /sync/get Endpoint:
#         Authenticates the user and gets their Database instance.
#         Queries the user's sync_log for changes after the since_change_id provided by the client.
#         Filters out echo: The SQL query includes AND client_id != ? to avoid sending changes back to the client that originally made them.
#         Includes the latest_change_id from the user's log in the response so the client knows the server's current state for that user.
#         Returns the list of changes and the latest ID.
#     ServerSyncProcessor: Encapsulates the server-side application logic, mirroring the structure of the ClientSyncEngine but adapted for the server's role (using authoritative timestamps, handling requests for a specific user).
#     _execute_server_change_sql: This is intentionally almost identical to the client's version, leveraging the shared library and schema. The key difference is the source of the timestamp and potentially client_id parameters passed into it.
#     Async Considerations: While FastAPI is async, the underlying sqlite_db.py library uses synchronous operations with check_same_thread=False. For low-to-moderate load, FastAPI handles this by running sync code in a thread pool. For very high concurrency, switching the DB library and these endpoints to use asyncio and an async DB driver (aiosqlite) would be necessary for optimal performance, but significantly increases complexity. The current implementation will work but might block the event loop under heavy load.

####################################################################################################
#
# --- FastAPI Endpoint Definitions ---

@router.post("/send",
             status_code=status.HTTP_200_OK,
             summary="Receive changes from a client")
async def receive_changes_from_client(
    payload: ClientChangesPayload,
    user_id: User = Depends(get_request_user),
    db: Database = Depends(get_db_for_user)
):
    """
    Receives a batch of sync log entries from a client, applies them
    to the user's database on the server synchronously in a thread pool.
    """
    requesting_client_id = payload.client_id
    if not payload.changes:
        logger.info(f"[{user_id.username}] Received empty change batch from client {requesting_client_id}.")
        return {"status": "success", "message": "No changes received."}

    logger.info(f"[{user_id.username}] Received {len(payload.changes)} changes from client {requesting_client_id}. Applying to DB: {db.db_path_str}")

    processor = ServerSyncProcessor(db=db, user_id=user_id.username, requesting_client_id=requesting_client_id)
    try:
        # Use .model_dump() for Pydantic v2+, or .dict() for v1
        changes_as_dicts = [change.model_dump() for change in payload.changes]
    except AttributeError:
        changes_as_dicts = [change.dict() for change in payload.changes] # Fallback for Pydantic v1

    try:
        # --- Run the SYNCHRONOUS method in a thread pool ---
        success, errors = await asyncio.to_thread(
            processor.apply_client_changes_batch,
            changes_as_dicts
        )
        # --- End Thread Pool Execution ---

        if not success:
            detail = {"message": "Failed to apply changes atomically.", "errors": errors}
            logger.error(f"[{user_id.username}] Failed processing batch from {requesting_client_id}: {errors}")
            # Use 400 Bad Request if errors suggest bad client data, 500 otherwise
            error_code = status.HTTP_400_BAD_REQUEST if any("payload" in e.lower() or "value" in e.lower() for e in errors) else status.HTTP_500_INTERNAL_SERVER_ERROR
            raise HTTPException(status_code=error_code, detail=detail)

        logger.info(f"[{user_id.username}] Successfully processed batch from client {requesting_client_id}.")
        return {"status": "success"}

    except Exception as e:
        # Catch unexpected errors during thread execution or processing
        logger.exception(f"[{user_id.username}] Unexpected error in /send endpoint from client {requesting_client_id}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {e}")


@router.get("/get",
            response_model=ServerChangesResponse,
            summary="Send changes back to a client")
async def send_changes_to_client(
    client_id: str,
    since_change_id: int = 0,
    user_id: User = Depends(get_request_user),
    db: Database = Depends(get_db_for_user)
):
    """
    Sends sync log entries from the user's server-side database back to
    the requesting client, starting after the `since_change_id`.
    Filters out changes that originated from the requesting client.
    Uses asyncio.to_thread for DB access.
    """
    logger.info(f"[{user_id.username}] Client '{client_id}' requesting changes since server log ID {since_change_id} from DB: {db.db_path_str}.")

    def _get_changes_sync():
        """Synchronous helper to fetch data"""
        changes_raw_list = []
        latest_id = 0
        try:
            # Note: Using the db object directly, assuming its methods are thread-safe
            # or that run_in_executor handles the context correctly.
            query = """
                SELECT change_id, entity, entity_uuid, operation, timestamp, client_id, version, payload
                FROM sync_log
                WHERE change_id > ? AND client_id != ?
                ORDER BY change_id ASC
                LIMIT ?
            """
            params = (since_change_id, client_id, SYNC_BATCH_SIZE)
            # Assuming db.execute_query is synchronous and thread-safe
            cursor_changes = db.execute_query(query, params)
            changes_raw_list = cursor_changes.fetchall() # This fetch happens synchronously

            cursor_latest = db.execute_query("SELECT MAX(change_id) FROM sync_log")
            latest_row = cursor_latest.fetchone()
            latest_id = latest_row[0] if latest_row and latest_row[0] is not None else 0
            return changes_raw_list, latest_id
        except (DatabaseError, sqlite3.Error) as db_err:
            logger.error(f"[{user_id.username}] Sync DB error in _get_changes_sync for client {client_id}: {db_err}", exc_info=True)
            # Re-raise to be caught by the main handler
            raise

    try:
        # --- Run the SYNCHRONOUS DB fetching in a thread pool ---
        changes_raw, latest_server_id = await asyncio.to_thread(_get_changes_sync)
        # --- End Thread Pool Execution ---

        changes_models: List[SyncLogEntry] = []
        for row_dict in changes_raw: # Assuming fetchall returns list of dicts/Rows
             try:
                  # Convert row (which should be dict-like if row_factory is set) to SyncLogEntry model
                  entry = SyncLogEntry(**dict(row_dict))
                  changes_models.append(entry)
             except Exception as pydantic_err:
                  logger.error(f"[{user_id.username}] Error validating sync log entry data (ID: {row_dict.get('change_id', 'N/A')}) against model: {pydantic_err}", exc_info=True)
                  continue # Skip bad entry

        logger.info(f"[{user_id.username}] Sending {len(changes_models)} changes to client '{client_id}'. Server latest ID: {latest_server_id}.")

        return ServerChangesResponse(
            changes=changes_models,
            latest_change_id=latest_server_id
        )

    except (DatabaseError, sqlite3.Error) as e: # Catch errors raised from sync helper
        logger.error(f"Database error getting changes for user '{user_id.username}', client '{client_id}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve changes from database.")
    except Exception as e: # Catch unexpected errors
        logger.error(f"Unexpected server error getting changes for user '{user_id.username}', client '{client_id}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {e}")

#
# End of API endpoints
#####################################################################################


# --- Server-Side Processing Logic ---
# Encapsulate server-side logic similar to ClientSyncEngine
class ServerSyncProcessor:
    """Handles applying changes received from a client to the server's user DB."""

    def __init__(self, db: Database, user_id: str, requesting_client_id: str):
        self.db = db
        self.user_id = user_id
        self.requesting_client_id = requesting_client_id # Client making the current API call
        logger.info(f"ServerSyncProcessor initialized for user '{self.user_id}', request from '{self.requesting_client_id}'.")

    # --- SYNCHRONOUS BATCH APPLICATION ---
    def apply_client_changes_batch(self, changes: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Applies a batch of ordered changes received from a client within a single transaction.
        Returns (success_status, list_of_error_messages).
        THIS RUNS SYNCHRONOUSLY (intended to be called via asyncio.to_thread).
        """
        all_applied_or_skipped = True
        errors = []
        changes.sort(key=lambda x: x['change_id'])
        server_authoritative_timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

        try:
            with self.db.transaction() as conn: # Use SYNCHRONOUS context manager
                cursor = conn.cursor()
                for change in changes:
                    try:
                        success, error_msg = self._apply_single_client_change_sync( # Call SYNC helper
                            cursor,
                            change,
                            server_authoritative_timestamp
                        )
                        if not success:
                            full_error = f"Failed Change ID {change.get('change_id', 'N/A')}: {error_msg}"
                            logger.error(f"[{self.user_id}] {full_error}")
                            errors.append(full_error)
                            all_applied_or_skipped = False
                            raise DatabaseError(f"Failed to apply change {change.get('change_id', 'N/A')}, rolling back batch.")

                    # Catch specific errors from _apply_single to ensure rollback
                    except (DatabaseError, sqlite3.Error, ConflictError, json.JSONDecodeError, KeyError, InputError, ValueError) as item_error:
                        error_msg = f"Failed Processing Change ID {change.get('change_id','?')}: {item_error}"
                        logger.error(f"[{self.user_id}] {error_msg}", exc_info=True)
                        if error_msg not in errors: errors.append(error_msg) # Avoid duplicate if already logged
                        all_applied_or_skipped = False
                        raise # Re-raise to trigger transaction rollback

            # Transaction commits here if no exceptions raised

        except (DatabaseError, sqlite3.Error, ConflictError) as e:
            logger.error(f"Transaction rolled back applying changes for user {self.user_id} from client {self.requesting_client_id}: {e}")
            if not errors: errors.append(f"Transaction rolled back: {e}")
            all_applied_or_skipped = False
        except Exception as e:
            logger.error(f"Unexpected error applying client changes batch for user {self.user_id}: {e}", exc_info=True)
            errors.append(f"Unexpected server error: {e}")
            all_applied_or_skipped = False

        return all_applied_or_skipped, errors

    # --- SYNCHRONOUS SINGLE CHANGE APPLICATION ---
    def _apply_single_client_change_sync(self, cursor: sqlite3.Cursor, change: Dict, current_server_time_str: str) -> Tuple[bool, Optional[str]]:
        """
        Applies a single change from a client synchronously. Handles conflict detection/resolution.
        Returns (success, error_message). Runs within the transaction from apply_client_changes_batch.
        """
        entity = change['entity']
        entity_uuid = change['entity_uuid']
        operation = change['operation']
        client_version = change['version']
        originating_client_id = change['client_id']
        payload_str = change.get('payload')
        payload = {}
        error_msg = None

        # Payload validation
        if not payload_str and not (entity == "MediaKeywords" and operation in ['link', 'unlink']):
            error_msg = f"Missing payload for incoming change: {change.get('change_id', 'N/A')}"
            logger.error(f"[{self.user_id}] {error_msg}")
            return False, error_msg
        try:
            if payload_str: payload = json.loads(payload_str)
        except json.JSONDecodeError as e:
             error_msg = f"Failed to decode payload for change ID {change.get('change_id', 'N/A')}: {e}"
             logger.error(f"[{self.user_id}] {error_msg}", exc_info=True)
             return False, error_msg

        try:
            # --- Fetch current server state ---
            cursor.execute(f"SELECT version, client_id, last_modified FROM `{entity}` WHERE uuid = ?", (entity_uuid,))
            server_record_info = cursor.fetchone() # Sync fetchone
            server_version = server_record_info[0] if server_record_info else 0
            server_client_id = server_record_info[1] if server_record_info else None

            logger.debug(f"[{self.user_id}] Applying change: Op={operation}, Entity={entity}, UUID={entity_uuid}, ClientVer={client_version}, ServerVer={server_version}, OrigClient={originating_client_id}, ServerClient={server_client_id}")

            # --- Idempotency/Conflict Check ---
            if client_version <= server_version:
                if client_version < server_version or (client_version == server_version and originating_client_id == server_client_id):
                    logger.debug(f"[{self.user_id}] Server skipping old/duplicate change for {entity} UUID {entity_uuid}...")
                    return True, None # Skip
                elif client_version == server_version and originating_client_id != server_client_id:
                    logger.warning(f"[{self.user_id}] Server detected conflict for {entity} UUID {entity_uuid}. ClientVer==ServerVer, different clients.")
                    # Resolve conflict using sync helper
                    resolved, error_msg = self._resolve_server_conflict_sync(cursor, change, server_record_info, current_server_time_str)
                    return resolved, error_msg
                elif operation == 'create' and server_version > 0:
                    logger.warning(f"[{self.user_id}] Server skipping client 'create' for existing {entity} UUID {entity_uuid}...")
                    return True, None # Skip
                else:
                    logger.error(f"[{self.user_id}] Unhandled state in server idempotency check...")
                    return True, None # Skip defensively

            # --- Apply the change (if client_version > server_version) ---
            logger.debug(f"[{self.user_id}] Applying change with ClientVer {client_version} > ServerVer {server_version}")
            self._execute_server_change_sql_sync( # Call sync version
                cursor, entity, operation, payload, entity_uuid,
                client_version, originating_client_id, current_server_time_str,
                force_apply=False
            )
            return True, None # Success

        except ConflictError as cf_err:
            # Optimistic lock failed during direct apply
            logger.warning(f"[{self.user_id}] Optimistic lock failed applying change for {entity} {entity_uuid}. Attempting resolution.")
            # Fetch current state again for resolution
            cursor.execute(f"SELECT version, client_id, last_modified FROM `{entity}` WHERE uuid = ?", (entity_uuid,))
            current_server_state = cursor.fetchone() # Sync fetchone
            resolved, error_msg = self._resolve_server_conflict_sync( # Call sync version
                 cursor, change, current_server_state, current_server_time_str
            )
            return resolved, error_msg

        # Keep specific exception handling
        except (DatabaseError, sqlite3.Error, json.JSONDecodeError, KeyError, InputError, ValueError) as e:
            error_msg = f"Error applying single change for {entity} {entity_uuid}: {e}"
            logger.error(f"[{self.user_id}] {error_msg}", exc_info=True)
            return False, error_msg # Return failure status and message
        except Exception as e:
            error_msg = f"Unexpected error in _apply_single_client_change: {e}"
            logger.critical(f"[{self.user_id}] {error_msg}", exc_info=True)
            return False, error_msg # Return failure status and message

    # --- SYNCHRONOUS CONFLICT RESOLUTION ---
    def _resolve_server_conflict_sync(self, cursor: sqlite3.Cursor, client_change: Dict, server_record_info: Optional[tuple], current_server_time_str: str) -> Tuple[bool, Optional[str]]:
        """
        Resolves conflict synchronously using LWW based on parsed timestamps.
        Returns (resolved_successfully, error_message).
        """
        entity = client_change['entity']
        entity_uuid = client_change['entity_uuid']
        operation = client_change['operation']
        client_version = client_change['version']
        originating_client_id = client_change['client_id']
        payload_str = client_change.get('payload')
        payload = json.loads(payload_str) if payload_str else {}
        error_msg = None # Initialize

        if server_record_info is None:
            logger.error(f"[{self.user_id}] Cannot resolve conflict for {entity} {entity_uuid}: server record info is missing.")
            return False, "Cannot resolve conflict without server record state."

        server_version = server_record_info[0]
        server_last_modified_str = server_record_info[2] if len(server_record_info) > 2 and isinstance(
            server_record_info[2], str) else '1970-01-01T00:00:00Z'

        # --- LWW Resolution (using datetime objects) ---
        try:
            # Define the EXPECTED standard formats
            iso_format_frac_z = '%Y-%m-%dT%H:%M:%S.%fZ'  # Primary format
            iso_format_no_frac_z = '%Y-%m-%dT%H:%M:%SZ'  # Fallback for older data / no fractional

            server_dt = None
            authoritative_dt = None

            # --- Try parsing server timestamp ---
            try:
                # Try with fractional seconds first
                server_dt = datetime.strptime(server_last_modified_str, iso_format_frac_z)
            except ValueError:
                try:
                    # Fallback to parsing without fractional seconds
                    server_dt = datetime.strptime(server_last_modified_str, iso_format_no_frac_z)
                except ValueError:
                    # If both fail, raise a clearer error
                    raise ValueError(
                        f"Could not parse server timestamp '{server_last_modified_str}' with formats '{iso_format_frac_z}' or '{iso_format_no_frac_z}'")

            # --- Try parsing authoritative timestamp ---
            # (It should match iso_format_frac_z generated by strftime)
            try:
                authoritative_dt = datetime.strptime(current_server_time_str, iso_format_frac_z)
            except ValueError:
                # Add fallback just in case strftime behaves unexpectedly
                try:
                    authoritative_dt = datetime.strptime(current_server_time_str, iso_format_no_frac_z)
                except ValueError:
                    raise ValueError(
                        f"Could not parse authoritative timestamp '{current_server_time_str}' with formats '{iso_format_frac_z}' or '{iso_format_no_frac_z}'")

            # Ensure timezone aware (strptime with %Z *should* handle UTC, but explicit is safer)
            # If %Z parsing worked, tzinfo might already be set. replace() only sets if naive.
            server_dt = server_dt.replace(tzinfo=timezone.utc)
            authoritative_dt = authoritative_dt.replace(tzinfo=timezone.utc)

        except ValueError as parse_err:
            error_msg = f"Timestamp parsing error during conflict resolution: {parse_err}"
            logger.error(
                f"[{self.user_id}] {error_msg} (Server='{server_last_modified_str}', Authoritative='{current_server_time_str}')",
                exc_info=True)
            return False, error_msg  # Return failure

            # --- Compare datetime objects ---
        logger.debug(
            f"[{self.user_id}] Comparing Timestamps: ServerDT={server_dt}, AuthoritativeDT={authoritative_dt}")  # Add debug log
        if server_dt >= authoritative_dt:
            logger.warning(
                f"[{self.user_id}] Resolving Conflict (LWW): Server wins (ServerDT {server_dt} >= AuthoritativeDT {authoritative_dt}). Skipping client change.")
            return True, None  # Resolved by skipping client change
        else:
            # Client wins - proceed with force apply
            logger.warning(
                f"[{self.user_id}] Resolving Conflict (LWW): Client change wins (ServerDT {server_dt} < AuthoritativeDT {authoritative_dt}). Forcing apply.")
            try:
                # Ensure correct arguments are passed, especially the timestamp string
                self._execute_server_change_sql_sync(
                    cursor, entity, operation, payload, entity_uuid,
                    client_version, originating_client_id, current_server_time_str,
                    # Pass the authoritative string timestamp
                    force_apply=True
                )
                return True, None
            # ... (Keep existing exception handling for force apply errors) ...
            except (DatabaseError, sqlite3.Error, ValueError) as e:
                error_msg = f"Failed to force apply winning change: {e}"
                logger.error(
                    f"[{self.user_id}] Error forcing update during LWW conflict resolution for {entity} {entity_uuid}: {e}",
                    exc_info=True)
                return False, error_msg
            except Exception as e:
                error_msg = f"Unexpected server error during resolution force apply: {e}"
                logger.critical(f"[{self.user_id}] Unexpected error during conflict resolution force apply: {e}",
                                exc_info=True)
                return False, error_msg

    # --- SYNCHRONOUS SQL EXECUTION ---
    def _execute_server_change_sql_sync(self, cursor: sqlite3.Cursor, entity: str, operation: str, payload: Dict, uuid: str,
                                     client_version: int, originating_client_id: str, server_timestamp: str,
                                     force_apply: bool = False):
        """Executes SQL synchronously on server DB."""
        logger.debug(f"[{self.user_id}] Executing Server SQL (Sync): Op='{operation}', Entity='{entity}', UUID='{uuid}', ClientVer='{client_version}', Force='{force_apply}'")

        # --- Special handling for MediaKeywords ---
        if entity == "MediaKeywords":
            return self._execute_server_media_keyword_sql_sync(cursor, operation, payload)

        # --- Standard handling ---
        table_columns = self._get_table_columns_sync(cursor, entity)
        if not table_columns:
            raise DatabaseError(f"Cannot proceed: Failed to get columns for entity '{entity}'.")

        # --- Version Calculation (Server Side - Sync) ---
        current_server_version = 0
        cursor.execute(f"SELECT version FROM `{entity}` WHERE uuid = ?", (uuid,))
        current_rec = cursor.fetchone()
        if current_rec:
            current_server_version = current_rec[0]

        target_sql_version = client_version
        if force_apply:
            target_sql_version = current_server_version + 1
            logger.debug(f"[{self.user_id}] Forcing apply (Sync): Setting target SQL version to {target_sql_version}")

        # --- Optimistic Lock (Sync - same logic) ---
        optimistic_lock_sql = ""
        optimistic_lock_param = []
        expected_base_version = client_version - 1
        if not force_apply and operation in ['update', 'delete']:
             if expected_base_version > 0:
                 optimistic_lock_sql = " AND version = ?"
                 optimistic_lock_param = [expected_base_version]
             else:
                 logger.debug(f"[{self.user_id}] Applying {operation} (Sync) based on client version {client_version}...")

        sql = ""
        params_tuple = tuple()
        sql_executed_in_fts_block = False # Flag to prevent double execution

        # --- SQL Generation (Sync) ---
        if operation == 'create':
            target_sql_version = 1
            cols_sql, placeholders_sql, params_list = [], [], []
            core_sync_meta = {'uuid': uuid, 'last_modified': server_timestamp, 'version': target_sql_version, 'client_id': originating_client_id}
            all_data = {**payload, **core_sync_meta, 'deleted': payload.get('deleted', 0)}
            for col in table_columns:
                if col == 'id': continue
                if col in all_data:
                    value = all_data[col]
                    if isinstance(value, bool): value = 1 if value else 0
                    cols_sql.append(f"`{col}`")
                    placeholders_sql.append("?")
                    params_list.append(value)
            if not cols_sql: raise ValueError(f"Cannot build INSERT for {entity} {uuid}: No columns")
            sql = f"INSERT OR IGNORE INTO `{entity}` ({', '.join(cols_sql)}) VALUES ({', '.join(placeholders_sql)})"
            params_tuple = tuple(params_list)

        elif operation == 'update':
            set_clauses, params_list = [], []
            for col in table_columns:
                 if col in payload and col not in ['id', 'uuid']:
                     value = payload[col]
                     if isinstance(value, bool): value = 1 if value else 0
                     set_clauses.append(f"`{col}` = ?")
                     params_list.append(value)

            set_clauses.extend(["`last_modified` = ?", "`version` = ?", "`client_id` = ?", "`deleted` = ?"])
            params_list.extend([server_timestamp, target_sql_version, originating_client_id, 1 if payload.get('deleted', False) else 0])

            if not set_clauses:
                 logger.warning(f"[{self.user_id}] Update operation for {entity} {uuid} resulted in no SET clauses.")
                 return # Skip

            where_clause = " WHERE uuid = ?"
            where_params = [uuid]
            if not force_apply:
                 where_clause += optimistic_lock_sql
                 where_params.extend(optimistic_lock_param)

            sql = f"UPDATE `{entity}` SET {', '.join(set_clauses)}{where_clause}"
            params_tuple = tuple(params_list + where_params)

        elif operation == 'delete':
            where_clause = " WHERE uuid = ?"
            where_params = [uuid]
            if not force_apply:
                 where_clause += optimistic_lock_sql
                 where_params.extend(optimistic_lock_param)
            sql = f"UPDATE `{entity}` SET deleted = 1, last_modified = ?, version = ?, client_id = ?{where_clause}"
            params_tuple = tuple([server_timestamp, target_sql_version, originating_client_id] + where_params)

        else:
            raise ValueError(f"Unsupported operation '{operation}' from client for entity '{entity}'")

        # --- Execute SQL (Sync) ---
        try:
            # --- Manual FTS Update (Server Side - Sync) ---
            # FIXME: FTS update logic is disabled for now. Uncomment and adjust as needed.
            # Check if FTS needs update *before* main execution if possible, to execute together
            # fts_update_needed = (operation in ['create', 'update']
            #                       and entity in ['Media', 'Keywords']
            #                       and any(k in payload for k in ['title', 'content', 'keyword']))
            #
            # if fts_update_needed:
            #     # Try executing main SQL, then FTS update
            #     logger.debug(f"[{self.user_id}] Executing Server SQL (Sync) [Pre-FTS]: {sql} | Params: {params_tuple}")
            #     cursor.execute(sql, params_tuple) # Execute main first
            #     sql_executed_in_fts_block = True # Mark as executed
            #
            #     # Check optimistic lock *after* main execution
            #     if not force_apply and operation in ['update', 'delete'] and optimistic_lock_sql and cursor.rowcount == 0:
            #          logger.warning(f"[{self.user_id}] Optimistic lock failed applying client change (Sync)...")
            #          raise ConflictError(f"Optimistic lock failed applying client change.", entity=entity, identifier=uuid)
            #     elif not force_apply and operation in ['update', 'delete'] and optimistic_lock_sql :
            #          logger.debug(f"[{self.user_id}] Optimistic lock successful applying client change (Sync). Rowcount {cursor.rowcount}.")
            #
            #
            #     # If main update successful, update FTS
            #     self._update_fts_manually_sync(cursor, entity, payload, uuid) # Call sync helper
            #
            # else:
            #     # If no FTS needed, execute main SQL normally
            #      logger.debug(f"[{self.user_id}] Executing Server SQL (Sync): {sql} | Params: {params_tuple}")
            #      cursor.execute(sql, params_tuple)
            #      # Optimistic Lock Check
            #      if not force_apply and operation in ['update', 'delete'] and optimistic_lock_sql:
            #           if cursor.rowcount == 0:
            #                logger.warning(f"[{self.user_id}] Optimistic lock failed applying client change (Sync)...")
            #                raise ConflictError(f"Optimistic lock failed applying client change.", entity=entity, identifier=uuid)
            #           else:
            #                logger.debug(f"[{self.user_id}] Optimistic lock successful applying client change (Sync). Rowcount {cursor.rowcount}.")

            # --- TEMP: Always execute main SQL outside of FTS check for now ---
            logger.debug(f"[{self.user_id}] Executing Server SQL (Sync) (FTS Disabled): {sql} | Params: {params_tuple}")
            cursor.execute(sql, params_tuple)
            # Optimistic Lock Check
            if not force_apply and operation in ['update', 'delete'] and optimistic_lock_sql:
                if cursor.rowcount == 0:
                    logger.warning(f"[{self.user_id}] Optimistic lock failed applying client change (Sync)...")
                    raise ConflictError(f"Optimistic lock failed applying client change.", entity=entity, identifier=uuid)
                else:
                    logger.debug(f"[{self.user_id}] Optimistic lock successful applying client change (Sync). Rowcount {cursor.rowcount}.")
            # --- END TEMP ---

        except sqlite3.IntegrityError as ie:
            logger.error(f"[{self.user_id}] Integrity error applying client change (Sync) for {entity} {uuid}: {ie}", exc_info=True)
            raise DatabaseError(f"Integrity error applying client change: {ie}") from ie
        except sqlite3.Error as e:
            logger.error(f"[{self.user_id}] SQLite error applying client change (Sync) for {entity} {uuid}: {e}", exc_info=True)
            raise

    # --- Define SYNCHRONOUS versions of helpers ---
    _server_column_cache = {}
    def _get_table_columns_sync(self, cursor: sqlite3.Cursor, table_name: str) -> Optional[List[str]]:
         """Synchronous version for getting table columns."""
         if table_name in self._server_column_cache: return self._server_column_cache[table_name]
         try:
              if not table_name.replace('_','').isalnum(): raise ValueError(f"Invalid table name: {table_name}")
              cursor.execute(f"PRAGMA table_info(`{table_name}`)")
              columns_info = cursor.fetchall() # Sync fetchall
              columns = [row[1] for row in columns_info]
              if columns:
                   self._server_column_cache[table_name] = columns
                   logger.debug(f"[{self.user_id}] Cached columns for server table '{table_name}': {columns}")
                   return columns
              else: logger.error(f"[{self.user_id}] Could not retrieve columns for server table: {table_name}"); return None
         except (sqlite3.Error, ValueError) as e: logger.error(f"[{self.user_id}] Error getting columns for server table {table_name}: {e}"); return None

    def _execute_server_media_keyword_sql_sync(self, cursor: sqlite3.Cursor, operation: str, payload: Dict):
        """Synchronous version for MediaKeywords SQL execution."""
        media_uuid = payload.get('media_uuid')
        keyword_uuid = payload.get('keyword_uuid')
        if not media_uuid or not keyword_uuid: raise ValueError(f"Missing UUIDs for MediaKeywords {operation}")
        cursor.execute("SELECT id FROM Media WHERE uuid = ?", (media_uuid,))
        media_rec = cursor.fetchone() # Sync
        cursor.execute("SELECT id FROM Keywords WHERE uuid = ?", (keyword_uuid,))
        kw_rec = cursor.fetchone() # Sync
        if not media_rec: logger.warning(f"[{self.user_id}] Skipping Server MediaKeywords {operation}: Media UUID {media_uuid} not found."); return
        if not kw_rec: logger.warning(f"[{self.user_id}] Skipping Server MediaKeywords {operation}: Keyword UUID {keyword_uuid} not found."); return
        media_id_local, keyword_id_local = media_rec[0], kw_rec[0]
        if operation == 'link':
             sql = "INSERT OR IGNORE INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)"
             params_tuple = (media_id_local, keyword_id_local)
             logger.debug(f"[{self.user_id}] Executing Server SQL (Sync): {sql} | Params: {params_tuple}")
             cursor.execute(sql, params_tuple) # Sync
        elif operation == 'unlink':
             sql = "DELETE FROM MediaKeywords WHERE media_id = ? AND keyword_id = ?"
             params_tuple = (media_id_local, keyword_id_local)
             logger.debug(f"[{self.user_id}] Executing Server SQL (Sync): {sql} | Params: {params_tuple}")
             cursor.execute(sql, params_tuple) # Sync
        else: raise ValueError(f"Unsupported server operation '{operation}' for MediaKeywords entity")


    def _update_fts_manually_sync(self, cursor: sqlite3.Cursor, entity: str, payload: Dict, uuid: str):
         """Updates the corresponding FTS table manually on the server (sync version)."""
         if entity == 'Media':
             if 'title' in payload or 'content' in payload:
                 cursor.execute("SELECT id, title, content FROM Media WHERE uuid = ?", (uuid,))
                 media_row = cursor.fetchone() # Sync
                 if media_row:
                     media_id, current_title, current_content = media_row
                     new_title = payload.get('title', current_title)
                     new_content = payload.get('content', current_content)
                     try:
                          logger.debug(f"[{self.user_id}] Manually updating media_fts (Sync) for Media UUID {uuid} (ID: {media_id})")
                          cursor.execute("UPDATE media_fts SET title = ?, content = ? WHERE rowid = ?", (new_title, new_content, media_id)) # Sync
                          if cursor.rowcount == 0:
                              logger.warning(f"[{self.user_id}] FTS row not found for update (Media ID: {media_id}), attempting insert.")
                              cursor.execute("INSERT INTO media_fts (rowid, title, content) VALUES (?, ?, ?)", (media_id, new_title, new_content)) # Sync
                     except sqlite3.Error as fts_err: logger.error(f"[{self.user_id}] Failed to manually update media_fts (Sync) for Media ID {media_id}: {fts_err}", exc_info=True)
                 else: logger.warning(f"[{self.user_id}] Cannot update media_fts (Sync): Media record not found for UUID {uuid}")
         elif entity == 'Keywords':
             if 'keyword' in payload:
                 cursor.execute("SELECT id FROM Keywords WHERE uuid = ?", (uuid,))
                 kw_row = cursor.fetchone() # Sync
                 if kw_row:
                     kw_id = kw_row[0]
                     new_keyword = payload['keyword']
                     try:
                          logger.debug(f"[{self.user_id}] Manually updating keyword_fts (Sync) for Keyword UUID {uuid} (ID: {kw_id})")
                          cursor.execute("UPDATE keyword_fts SET keyword = ? WHERE rowid = ?", (new_keyword, kw_id)) # Sync
                          if cursor.rowcount == 0:
                              logger.warning(f"[{self.user_id}] FTS row not found for update (Keyword ID: {kw_id}), attempting insert.")
                              cursor.execute("INSERT INTO keyword_fts (rowid, keyword) VALUES (?, ?)", (kw_id, new_keyword)) # Sync
                     except sqlite3.Error as fts_err: logger.error(f"[{self.user_id}] Failed to manually update keyword_fts (Sync) for Keyword ID {kw_id}: {fts_err}", exc_info=True)
                 else: logger.warning(f"[{self.user_id}] Cannot update keyword_fts (Sync): Keyword record not found for UUID {uuid}")

#
# End of sync-endpoint.py
#######################################################################################################################
