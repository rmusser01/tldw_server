# Server_API/app/api/v1/endpoints/prompts.py
# Description: This code provides a FastAPI endpoint for all Sync operations.
#
# Imports
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
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import redis
import requests
# API Rate Limiter/Caching via Redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from loguru import logger
from starlette.responses import JSONResponse

from tldw_Server_API.app.api.v1.schemas.sync_server_models import ClientChangesPayload, ServerChangesResponse
#
# Local Imports
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_db_for_user, verify_token_and_get_user
from tldw_Server_API.app.core.Chat.Chat_Functions import (
    get_character_names,
    get_conversation_name,
    alert_token_budget_exceeded,
)
#
# DB Mgmt
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import DatabaseError, ConflictError, Database, InputError
from tldw_Server_API.app.services.ephemeral_store import ephemeral_storage
from tldw_cli.tldw_app.DB.Sync_Client import SYNC_BATCH_SIZE

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


@router.post("/send",
             status_code=status.HTTP_200_OK, # Default success code
             summary="Receive changes from a client",
             )
async def receive_changes_from_client(
    payload: ClientChangesPayload, # Use Pydantic model for validation
    user_id: str = Depends(verify_token_and_get_user), # Get user ID from auth
    db: Database = Depends(get_db_for_user) # Get DB instance for this user
):
    """
    Receives a batch of sync log entries from a client, applies them
    to the user's database on the server, handling conflicts (LWW).
    """
    # Optional: Verify payload.client_id matches authenticated user/device context if needed
    requesting_client_id = payload.client_id # The device sending the changes
    if not payload.changes:
        logger.info(f"Received empty change batch from client {requesting_client_id} (user {user_id}).")
        return {"status": "success", "message": "No changes received."}

    logger.info(f"Received {len(payload.changes)} changes from client {requesting_client_id} (user {user_id}). Applying to DB: {db.db_path_str}")

    # --- Process Changes ---
    # Use a helper class or function to encapsulate the processing logic
    processor = ServerSyncProcessor(db=db, user_id=user_id, requesting_client_id=requesting_client_id)
    success, errors = await processor.apply_client_changes_batch(payload.changes)

    if success:
        logger.info(f"Successfully processed batch from client {requesting_client_id} for user {user_id}.")
        return {"status": "success"}
    else:
        # If processing failed, the transaction should have rolled back.
        # Return an error to the client.
        logger.error(f"Failed to process batch from client {requesting_client_id} for user {user_id}. Errors: {errors}")
        # Use 500 for server-side processing failure, or 409 Conflict if specific conflicts couldn't be auto-resolved
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, # Or status.HTTP_409_CONFLICT
            detail={"message": "Failed to apply changes atomically.", "errors": errors}
        )


@router.get("/get",
            response_model=ServerChangesResponse, # Use Pydantic model for response structure
            summary="Send changes back to a client",
            )
async def send_changes_to_client(
    client_id: str, # The ID of the specific client device requesting changes
    since_change_id: int = 0, # Last server log ID the client processed
    user_id: str = Depends(verify_token_and_get_user),
    db: Database = Depends(get_db_for_user)
):
    """
    Sends sync log entries from the user's server-side database back to
    the requesting client, starting after the `since_change_id`.
    Filters out changes that originated from the requesting client.
    """
    logger.info(f"Client '{client_id}' (user {user_id}) requesting changes since server log ID {since_change_id} from DB: {db.db_path_str}.")

    try:
        # Fetch changes from this user's sync_log on the server
        # Filter out changes that originated from the *requesting* client_id
        query = """
            SELECT change_id, entity, entity_uuid, operation, timestamp, client_id, version, payload
            FROM sync_log
            WHERE change_id > ? AND client_id != ?
            ORDER BY change_id ASC
            LIMIT ?
        """
        params = (since_change_id, client_id, SYNC_BATCH_SIZE)
        cursor = db.execute_query(query, params)
        changes_raw = cursor.fetchall()
        # Convert rows to SyncLogEntry model compatible dicts
        changes_list = [dict(row) for row in changes_raw]

        # Get the overall latest change ID in this user's log on the server
        cursor_latest = db.execute_query("SELECT MAX(change_id) FROM sync_log")
        latest_server_id = cursor_latest.fetchone()[0] or 0 # fetchone() returns tuple or None

        logger.info(f"Sending {len(changes_list)} changes to client '{client_id}'. Server latest ID for user '{user_id}' is {latest_server_id}.")

        return ServerChangesResponse(
            changes=changes_list,
            latest_change_id=latest_server_id
        )

    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Database error getting changes for user '{user_id}', client '{client_id}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve changes")
    except Exception as e:
        logger.error(f"Unexpected server error getting changes for user '{user_id}', client '{client_id}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")


# --- Server-Side Processing Logic ---
# Encapsulate server-side logic similar to ClientSyncEngine
class ServerSyncProcessor:
    """Handles applying changes received from a client to the server's user DB."""

    def __init__(self, db: Database, user_id: str, requesting_client_id: str):
        self.db = db
        self.user_id = user_id
        self.requesting_client_id = requesting_client_id # Client making the current API call

    async def apply_client_changes_batch(self, changes: List[Dict]) -> Tuple[bool, List[str]]:
        """
        Applies a batch of ordered changes received from a client within a single transaction.
        Returns (success_status, list_of_error_messages).
        """
        all_applied_or_skipped = True
        errors = []
        # Ensure changes are sorted by client's change_id (for sequential processing)
        changes.sort(key=lambda x: x['change_id'])
        # Determine authoritative timestamp for the *entire batch*
        server_authoritative_timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

        try:
            with self.db.transaction() as conn:
                cursor = conn.cursor()
                for change in changes:
                    try:
                        # Process a single change within the transaction
                        await self._apply_single_client_change(
                            cursor,
                            change,
                            server_authoritative_timestamp
                        )
                    except ConflictError as cf_err:
                        logger.warning(f"Conflict occurred applying change ID {change['change_id']} from client {change['client_id']}: {cf_err}. Attempting server resolution.")
                        resolved = await self._resolve_server_conflict(
                            cursor,
                            change,
                            server_authoritative_timestamp,
                            cf_err
                        )
                        if not resolved:
                            error_msg = f"Conflict resolution failed for change ID {change['change_id']}."
                            logger.error(error_msg)
                            errors.append(error_msg)
                            all_applied_or_skipped = False
                            raise cf_err # Trigger transaction rollback

                    except (DatabaseError, sqlite3.Error, json.JSONDecodeError, KeyError, InputError, ValueError) as item_error:
                        error_msg = f"Failed to apply change ID {change['change_id']} ({change.get('entity','?')}/{change.get('operation','?')}): {item_error}"
                        logger.error(error_msg, exc_info=True)
                        errors.append(error_msg)
                        all_applied_or_skipped = False
                        raise item_error # Trigger transaction rollback

            # Transaction commits here if no exceptions were raised

        except (DatabaseError, sqlite3.Error, ConflictError) as e:
            # Catch errors that caused rollback
            logger.error(f"Transaction rolled back applying changes for user {self.user_id} from client {self.requesting_client_id}: {e}")
            if not errors: errors.append(str(e)) # Add the rollback cause if no specific item errors were logged
            all_applied_or_skipped = False
        except Exception as e:
            logger.error(f"Unexpected error applying client changes batch for user {self.user_id}: {e}", exc_info=True)
            errors.append(f"Unexpected server error: {e}")
            all_applied_or_skipped = False

        return all_applied_or_skipped, errors

    async def _apply_single_client_change(self, cursor: sqlite3.Cursor, change: Dict, server_timestamp: str):
        """
        Applies a single change record received from a client.
        Raises ConflictError if optimistic lock fails against server state.
        """
        entity = change['entity']
        entity_uuid = change['entity_uuid']
        operation = change['operation']
        remote_version = change['version']
        originating_client_id = change['client_id'] # The client that *made* the change
        # remote_timestamp = change['timestamp'] # Client's original timestamp - mainly for info
        payload_str = change.get('payload')
        payload = json.loads(payload_str) if payload_str else {}

        # --- Idempotency Check (against SERVER state) ---
        cursor.execute(f"SELECT version FROM `{entity}` WHERE uuid = ?", (entity_uuid,))
        server_record = cursor.fetchone()
        server_version = server_record[0] if server_record else 0

        if remote_version <= server_version:
             if operation == 'create' and server_version > 0:
                  logger.warning(f"Server skipping client 'create' for existing {entity} UUID {entity_uuid} (RemoteVer: {remote_version}, ServerVer: {server_version}) from client {originating_client_id}")
             else:
                  logger.debug(f"Server skipping old/duplicate change for {entity} UUID {entity_uuid} (RemoteVer: {remote_version}, ServerVer: {server_version}) from client {originating_client_id}")
             return # Successfully skipped

        # --- Execute the change using server's logic ---
        # Pass force_apply=False for standard application with optimistic locking
        # Use the server's authoritative timestamp and the originating client's ID
        logger.debug(f"Server attempting to apply: {operation} on {entity} UUID {entity_uuid} (RemoteVer: {remote_version}, ServerVer: {server_version}) from client {originating_client_id}")

        # Use the shared _execute_sql method, adapted for server context
        await self._execute_server_change_sql(
            cursor=cursor,
            entity=entity,
            operation=operation,
            payload=payload,
            uuid=entity_uuid,
            version=remote_version, # Apply the version from the client's change log
            client_id=originating_client_id, # Record who originally made the change
            timestamp=server_timestamp, # Use SERVER's authoritative timestamp
            force_apply=False
        )
        # If _execute_server_change_sql encounters an optimistic lock failure (rowcount 0),
        # it will raise ConflictError, which is caught by apply_client_changes_batch

    async def _resolve_server_conflict(self, cursor: sqlite3.Cursor, change: Dict, server_timestamp: str, conflict_error: ConflictError) -> bool:
        """
        Attempts to resolve a conflict detected on the server (LWW).
        Returns True if resolved (applied or skipped), False otherwise.
        """
        entity = change['entity']
        entity_uuid = change['entity_uuid']
        operation = change['operation']
        remote_version = change['version']
        originating_client_id = change['client_id']
        payload_str = change.get('payload')
        payload = json.loads(payload_str) if payload_str else {}

        # --- LWW Conflict Resolution (Server's perspective) ---
        try:
            # Get the timestamp of the record currently in the server's DB for this user
            cursor.execute(f"SELECT last_modified FROM `{entity}` WHERE uuid = ?", (entity_uuid,))
            server_db_record = cursor.fetchone()
            server_db_timestamp = server_db_record[0] if server_db_record else '1970-01-01 00:00:00'

            # Compare the server's *current* authoritative timestamp with the timestamp in the DB
            if server_timestamp >= server_db_timestamp:
                logger.warning(f"  Server Resolving Conflict (LWW): Incoming change wins (ServerNowTS: {server_timestamp} >= ServerDB_TS: {server_db_timestamp}). Forcing apply for {entity} {entity_uuid}.")
                # Force apply the change, overwriting the conflicting server state
                await self._execute_server_change_sql(
                    cursor=cursor, entity=entity, operation=operation, payload=payload,
                    uuid=entity_uuid, version=remote_version, client_id=originating_client_id,
                    timestamp=server_timestamp, force_apply=True
                )
                return True # Resolved by applying remote change forcefully
            else:
                logger.warning(f"  Server Resolving Conflict (LWW): Server state wins (ServerNowTS: {server_timestamp} < ServerDB_TS: {server_db_timestamp}). Skipping incoming change for {entity} {entity_uuid}.")
                # Skip applying the incoming change; server's current state is kept
                return True # Resolved by skipping remote change

        except Exception as e:
            logger.error(f"Error during server LWW conflict resolution for {entity} {entity_uuid}: {e}", exc_info=True)
            return False # Resolution failed


    async def _execute_server_change_sql(self, cursor: sqlite3.Cursor, entity: str, operation: str, payload: Dict, uuid: str, version: int, client_id: str, timestamp: str, force_apply: bool = False):
        """
        Generates and executes SQL to apply a single change operation on the server DB.
        This is almost identical to the client's _execute_change_sql but separated
        for clarity and potential minor server-specific adaptations.
        Crucially, it uses the server's authoritative `timestamp`.
        """
        # This reuses the *exact same logic* as the client's _execute_change_sql
        # because the schema and library are shared. We just need to ensure the
        # correct parameters (especially server timestamp) are passed down.

        logger.debug(f"Server Executing SQL: Op='{operation}', Entity='{entity}', UUID='{uuid}', Ver='{version}', OrigClient='{client_id}', ServerTS='{timestamp}', Force='{force_apply}'")

        # --- Special handling for MediaKeywords junction table ---
        if entity == "MediaKeywords":
            # Re-use the MediaKeyword helper logic
            # Note: This helper needs to be adapted or duplicated if it's not async,
            # or called using run_in_executor if the DB library is synchronous.
            # Assuming helper is synchronous for now:
            self._execute_media_keyword_sql_sync(cursor, operation, payload) # Use sync version
            return

        # --- Standard handling for main entities ---
        table_columns = self._get_table_columns_sync(cursor, entity) # Use sync version
        if not table_columns:
            raise DatabaseError(f"Cannot proceed: Failed to get columns for entity '{entity}'.")

        # --- Build SQL and Params (Identical logic to client's _execute_change_sql) ---
        optimistic_lock_sql = ""
        optimistic_lock_param = []
        expected_base_version = version - 1
        if not force_apply and operation in ['update', 'delete']:
             if expected_base_version > 0:
                  optimistic_lock_sql = " AND version = ?"
                  optimistic_lock_param = [expected_base_version]
             # (Same logic for version 1 updates as in client)

        sql = ""
        params_tuple = tuple()

        if operation == 'create':
            # (Identical SQL generation logic as client's _execute_change_sql for create)
            cols_sql = []; placeholders_sql = []; params_list = []
            core_sync_meta = {'uuid': uuid, 'last_modified': timestamp, 'version': version, 'client_id': client_id}
            all_data = {**payload, **core_sync_meta, 'deleted': payload.get('deleted', 0)}
            for col in table_columns:
                 if col == 'id': continue
                 if col in all_data:
                      value = all_data[col]
                      if isinstance(value, bool): value = 1 if value else 0
                      cols_sql.append(f"`{col}`"); placeholders_sql.append("?"); params_list.append(value)
                 elif col == 'deleted':
                      cols_sql.append(f"`{col}`"); placeholders_sql.append("?"); params_list.append(0)
            if not cols_sql: raise ValueError(f"No columns for INSERT on {entity}")
            sql = f"INSERT INTO `{entity}` ({', '.join(cols_sql)}) VALUES ({', '.join(placeholders_sql)})"
            params_tuple = tuple(params_list)

        elif operation == 'update':
            # (Identical SQL generation logic as client's _execute_change_sql for update)
            set_clauses = []; params_list = []
            for col in table_columns:
                 if col in payload and col not in ['id', 'uuid']:
                      value = payload[col]
                      if isinstance(value, bool): value = 1 if value else 0
                      set_clauses.append(f"`{col}` = ?"); params_list.append(value)
            set_clauses.extend(["`last_modified` = ?", "`version` = ?", "`client_id` = ?", "`deleted` = ?"])
            params_list.extend([timestamp, version, client_id, 1 if payload.get('deleted', False) else 0])
            sql = f"UPDATE `{entity}` SET {', '.join(set_clauses)} WHERE uuid = ?"
            params_list.append(uuid)
            sql += optimistic_lock_sql; params_list.extend(optimistic_lock_param)
            params_tuple = tuple(params_list)

        elif operation == 'delete':
            # (Identical SQL generation logic as client's _execute_change_sql for delete)
            sql = f"UPDATE `{entity}` SET deleted = 1, last_modified = ?, version = ?, client_id = ? WHERE uuid = ?"
            params_list = [timestamp, version, client_id, uuid]
            sql += optimistic_lock_sql; params_list.extend(optimistic_lock_param)
            params_tuple = tuple(params_list)

        else:
            raise ValueError(f"Unsupported operation '{operation}' for entity '{entity}'")

        # --- Execute SQL ---
        try:
            logger.debug(f"Server Executing SQL: {sql} | Params: {params_tuple}")
            # IMPORTANT: Assume cursor.execute is synchronous based on the provided library
            cursor.execute(sql, params_tuple)

            # --- Check Row Count (Identical logic to client) ---
            if operation in ['update', 'delete'] and not force_apply and optimistic_lock_sql:
                if cursor.rowcount == 0:
                    logger.warning(f"Server Optimistic lock failed for {entity} UUID {uuid} (Op: {operation}, Expected Base Ver: {expected_base_version}). Rowcount 0.")
                    raise ConflictError(
                        f"Optimistic lock failed applying change on server.",
                        entity=entity, identifier=uuid
                    )
                else:
                     logger.debug(f"Server Optimistic lock successful for {entity} {uuid} (Op: {operation}). Rowcount {cursor.rowcount}.")

        except sqlite3.IntegrityError as ie:
             logger.error(f"Server Integrity error applying change for {entity} {uuid}: {ie}", exc_info=True)
             raise DatabaseError(f"Server Integrity error applying change: {ie}") from ie
        except sqlite3.Error as e:
             logger.error(f"Server SQLite error applying change for {entity} {uuid}: {e}", exc_info=True)
             raise


    # --- Synchronous versions of helpers needed if DB lib is sync ---
    # (These are direct copies/adaptations from the client code)
    def _execute_media_keyword_sql_sync(self, cursor: sqlite3.Cursor, operation: str, payload: Dict):
        """Synchronous version for MediaKeywords SQL execution."""
        media_uuid = payload.get('media_uuid')
        keyword_uuid = payload.get('keyword_uuid')
        if not media_uuid or not keyword_uuid: raise ValueError(f"Missing UUIDs for MediaKeywords {operation}")
        cursor.execute("SELECT id FROM Media WHERE uuid = ?", (media_uuid,))
        media_rec = cursor.fetchone();
        cursor.execute("SELECT id FROM Keywords WHERE uuid = ?", (keyword_uuid,))
        kw_rec = cursor.fetchone()
        if not media_rec or not kw_rec: logger.warning(f"Server skipping MediaKeywords {operation}: Parent not found locally for user {self.user_id}."); return
        media_id_local, keyword_id_local = media_rec[0], kw_rec[0]
        if operation == 'link': cursor.execute("INSERT OR IGNORE INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)", (media_id_local, keyword_id_local))
        elif operation == 'unlink': cursor.execute("DELETE FROM MediaKeywords WHERE media_id = ? AND keyword_id = ?", (media_id_local, keyword_id_local))
        else: raise ValueError(f"Unsupported operation '{operation}' for MediaKeywords")

    _column_cache = {}
    def _get_table_columns_sync(self, cursor: sqlite3.Cursor, table_name: str) -> Optional[List[str]]:
         """Synchronous version for getting table columns."""
         if table_name in self._column_cache: return self._column_cache[table_name]
         try:
              if not table_name.replace('_','').isalnum(): raise ValueError(f"Invalid table name: {table_name}")
              cursor.execute(f"PRAGMA table_info(`{table_name}`)")
              columns = [row[1] for row in cursor.fetchall()]
              if columns: self._column_cache[table_name] = columns; return columns
              else: logger.error(f"Could not retrieve columns for table: {table_name}"); return None
         except (sqlite3.Error, ValueError) as e: logger.error(f"Error getting columns for table {table_name}: {e}"); return None

#
# End of media.py
#######################################################################################################################
