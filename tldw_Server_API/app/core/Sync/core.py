# sync_library/core.py
import sqlite3
import logging
import threading
from typing import List, Optional, Dict, Tuple, Set
from datetime import datetime, timezone

# Adjust imports based on your project structure
from PoC_Version.App_Function_Libraries.DB.SQLite_DB import Database, DatabaseError

from .models import SyncLogEntry, parse_timestamp
from .exceptions import SyncError, ConflictError, TransportError, ApplyError, StateError
from .transport import SyncTransport
from .conflict import ConflictResolver
from .state import SyncStateManager
from .db_utils import get_entity_config, get_local_row_by_uuid, get_local_id_from_uuid

logger = logging.getLogger(__name__)

class SyncManager:
    """Orchestrates the database synchronization process."""

    def __init__(self,
                 db_instance: Database,
                 transport: SyncTransport,
                 resolver: ConflictResolver,
                 state_manager: SyncStateManager,
                 client_id: str):
        """
        Initializes the SyncManager.

        Args:
            db_instance: An initialized Database object for the local DB.
            transport: An object implementing the SyncTransport interface.
            resolver: An object implementing the ConflictResolver interface.
            state_manager: An object for managing persistent sync state.
            client_id: The unique identifier for this client instance.
        """
        if not isinstance(db_instance, Database): raise TypeError("db_instance must be a Database object")
        if not isinstance(transport, SyncTransport): raise TypeError("transport must be a SyncTransport object")
        if not isinstance(resolver, ConflictResolver): raise TypeError("resolver must be a ConflictResolver object")
        if not isinstance(state_manager, SyncStateManager): raise TypeError("state_manager must be a SyncStateManager object")
        if not client_id: raise ValueError("client_id cannot be empty")

        self.db = db_instance
        self.transport = transport
        self.resolver = resolver
        self.state_manager = state_manager
        self.client_id = client_id
        self._sync_lock = threading.Lock() # Prevents concurrent sync runs on this instance
        self._deferred_links: List[SyncLogEntry] = [] # Store link/unlink ops that need retry
        self._max_defer_retries = 3 # Max times to retry deferred links per sync cycle

        logger.info(f"SyncManager initialized for client_id: {self.client_id}")

    def _get_local_changes(self, last_sent_id: int) -> List[SyncLogEntry]:
        """Retrieves local changes from sync_log newer than the last sent ID."""
        query = "SELECT * FROM sync_log WHERE change_id > ? ORDER BY change_id ASC"
        try:
            cursor = self.db.execute_query(query, (last_sent_id,))
            rows = cursor.fetchall()
            changes = [SyncLogEntry.from_row(row) for row in rows]
            logger.info(f"Retrieved {len(changes)} local changes since change_id {last_sent_id}.")
            return changes
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Failed to retrieve local changes: {e}", exc_info=True)
            raise SyncError(f"Could not get local changes: {e}") from e

    def _apply_remote_changes(self, remote_changes: List[SyncLogEntry]) -> Optional[datetime]:
        """
        Applies a list of remote changes locally, handling conflicts.

        Returns:
            The UTC timestamp of the latest successfully applied change, or None if no changes applied.
        """
        if not remote_changes:
            logger.info("No remote changes to apply.")
            return None

        # Sort changes by timestamp, then change_id (if available/sent) or client_id as tie-breaker
        remote_changes.sort(key=lambda x: (x.timestamp, x.client_id, x.change_id))

        latest_applied_ts: Optional[datetime] = None
        applied_count = 0
        skipped_count = 0
        deferred_count = 0
        self._deferred_links = [] # Clear deferred from previous runs

        logger.info(f"Starting application of {len(remote_changes)} remote changes.")

        try:
            # Use a single transaction for the entire batch
            with self.db.transaction() as conn:
                # --- Initial Pass ---
                for entry in remote_changes:
                    try:
                        applied = self._apply_change(entry, conn)
                        if applied:
                            applied_count += 1
                            # Keep track of the latest timestamp *successfully* applied
                            if latest_applied_ts is None or entry.timestamp > latest_applied_ts:
                                latest_applied_ts = entry.timestamp
                        else:
                             skipped_count +=1 # Either kept local or deferred
                    except Exception as e: # Catch errors during individual change application
                         logger.error(f"Error applying change_id {entry.change_id} (UUID: {entry.entity_uuid}): {e}", exc_info=True)
                         # Wrap error with more context
                         raise ApplyError(f"Failed during apply loop: {e}", change_id=entry.change_id, entity_uuid=entry.entity_uuid) from e

                # --- Retry Deferred Links ---
                retries = 0
                while self._deferred_links and retries < self._max_defer_retries:
                    retries += 1
                    logger.info(f"Retrying {len(self._deferred_links)} deferred link/unlink operations (Retry #{retries})...")
                    deferred_to_retry = self._deferred_links[:] # Copy list
                    self._deferred_links = [] # Clear before retry pass
                    deferred_this_pass = 0

                    for entry in deferred_to_retry:
                        try:
                             applied = self._apply_change(entry, conn) # Retry applying
                             if applied:
                                 applied_count += 1
                                 if latest_applied_ts is None or entry.timestamp > latest_applied_ts:
                                     latest_applied_ts = entry.timestamp
                                 logger.debug(f"Successfully applied deferred link/unlink for {entry.entity_uuid}")
                             else:
                                 # Still couldn't apply (e.g., parents still missing), keep deferred unless max retries hit
                                 deferred_this_pass += 1
                                 if retries < self._max_defer_retries:
                                      self._deferred_links.append(entry)
                                 else:
                                      skipped_count += 1 # Count as skipped after max retries
                                      logger.warning(f"Max retries reached for deferred link/unlink (UUID: {entry.entity_uuid}). Skipping.")

                        except Exception as e:
                            logger.error(f"Error applying deferred change_id {entry.change_id} (UUID: {entry.entity_uuid}): {e}", exc_info=True)
                            raise ApplyError(f"Failed during deferred apply loop: {e}", change_id=entry.change_id, entity_uuid=entry.entity_uuid) from e
                    deferred_count = deferred_this_pass # Update count for logging

            # Transaction committed successfully if no exceptions raised
            logger.info(f"Finished applying remote changes. Applied: {applied_count}, Skipped/Kept Local: {skipped_count}, Deferred (final): {deferred_count}.")
            return latest_applied_ts

        except (DatabaseError, sqlite3.Error, ApplyError) as e:
             # Transaction automatically rolled back by context manager
             logger.error(f"Transaction failed during apply_remote_changes: {e}", exc_info=False) # Log less verbosely on expected failure
             raise SyncError(f"Failed to apply remote changes: {e}") from e
        except Exception as e:
             logger.error(f"Unexpected error during apply_remote_changes: {e}", exc_info=True)
             raise SyncError(f"Unexpected error applying remote changes: {e}") from e
        finally:
             self._deferred_links = [] # Ensure deferred list is clear after attempt


    def _apply_change(self, entry: SyncLogEntry, conn: sqlite3.Connection) -> bool:
        """Applies a single remote change, handling conflicts. Returns True if applied, False otherwise."""
        logger.debug(f"Applying change: {entry.entity} {entry.operation} UUID: {entry.entity_uuid} Time: {entry.timestamp} Client: {entry.client_id} Version: {entry.version}")

        # --- Handle Link/Unlink Separately ---
        if entry.entity == 'MediaKeywords':
            return self._apply_link_change(entry, conn)

        # --- Handle Create/Update/Delete for Normal Entities ---
        config = get_entity_config(entry.entity)
        if not config:
            logger.warning(f"Skipping change for unknown entity type: {entry.entity}")
            return False # Cannot apply

        table_name = config['table']
        uuid_col = config['uuid_col']

        # Fetch current local state (if exists)
        local_row = get_local_row_by_uuid(self.db, entry.entity, entry.entity_uuid)

        # Determine action based on conflict resolution
        resolution = self.resolver.resolve(local_row, entry)

        if resolution == 'keep_local':
            logger.debug(f"Resolution for {entry.entity_uuid}: Keep Local.")
            return False # Change was not applied

        elif resolution == 'apply_remote':
            logger.debug(f"Resolution for {entry.entity_uuid}: Apply Remote.")
            try:
                # Construct SQL based on operation and payload
                # Use INSERT ... ON CONFLICT for simplicity
                if entry.operation in ['create', 'update']:
                    if not entry.payload:
                         logger.error(f"Cannot apply {entry.operation} for {entry.entity_uuid}: Payload is missing.")
                         return False # Cannot apply without data

                    payload = entry.payload.copy() # Work on a copy

                    # --- Payload Adaptation & Column Mapping ---
                    # Remove fields that don't exist in the target table or are local-only
                    # Get target table columns
                    cursor = conn.execute(f"PRAGMA table_info({table_name})")
                    valid_columns = {col['name'] for col in cursor.fetchall()}
                    # Remove local ID if present in payload (shouldn't be, but defensive)
                    payload.pop('id', None)
                    # Remove local-only fields explicitly (sync trigger should already exclude them)
                    if entry.entity == 'Media':
                        payload.pop('vector_embedding', None)
                        payload.pop('chunking_status', None)
                        payload.pop('vector_processing', None)
                    if entry.entity == 'UnvectorizedMediaChunks':
                         payload.pop('is_processed', None)
                         payload.pop('last_modified_orig', None)

                    # Add/ensure core sync columns are present from the log entry itself
                    payload[uuid_col] = entry.entity_uuid
                    payload['last_modified'] = entry.timestamp.strftime('%Y-%m-%d %H:%M:%S') # Format for DB
                    payload['version'] = entry.version
                    payload['client_id'] = entry.client_id
                    # Handle 'deleted' flag explicitly based on operation
                    payload['deleted'] = 1 if entry.operation == 'delete' else payload.get('deleted', 0) # Use payload value if present for create/update, else default 0

                    # Filter payload keys to only include valid columns
                    columns_to_set = {k for k in payload.keys() if k in valid_columns}
                    if not columns_to_set:
                        logger.error(f"Cannot apply {entry.operation} for {entry.entity_uuid}: No valid columns found in payload after filtering.")
                        return False

                    set_clauses = [f"{col} = :{col}" for col in columns_to_set]
                    # Prepare dict for parameters, ensuring None is handled correctly
                    params = {col: payload.get(col) for col in columns_to_set}

                    # Use INSERT ON CONFLICT (requires UNIQUE constraint on UUID)
                    sql = f"""
                        INSERT INTO {table_name} ({', '.join(columns_to_set)})
                        VALUES ({', '.join(':' + col for col in columns_to_set)})
                        ON CONFLICT({uuid_col}) DO UPDATE SET
                            {', '.join(set_clauses)}
                        WHERE excluded.last_modified <= :last_modified -- Apply only if newer or equal (LWW logic handled by resolver, this is safety)
                          -- Optional: Add excluded.version check if needed for more complex resolution
                    """
                    conn.execute(sql, params)
                    logger.debug(f"Applied INSERT/UPDATE for {entry.entity_uuid}")
                    return True

                elif entry.operation == 'delete':
                     # Set deleted flag and update sync meta
                     sql = f"""
                         UPDATE {table_name}
                         SET deleted = 1,
                             last_modified = ?,
                             version = ?,
                             client_id = ?
                         WHERE {uuid_col} = ?
                     """
                     # Apply only if the remote delete is newer than local last_modified
                     # LWW resolver should handle this, but WHERE clause adds safety
                     if local_row and parse_timestamp(local_row.get('last_modified')) > entry.timestamp:
                          logger.debug(f"Skipping remote delete for {entry.entity_uuid} as local is newer.")
                          return False # Keep local (newer)

                     conn.execute(sql, (
                         entry.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                         entry.version,
                         entry.client_id,
                         entry.entity_uuid
                     ))
                     logger.debug(f"Applied DELETE (set deleted=1) for {entry.entity_uuid}")
                     return True
                else:
                    logger.error(f"Unknown operation '{entry.operation}' for {entry.entity_uuid}. Cannot apply.")
                    return False

            except (sqlite3.Error, ValueError, KeyError) as e:
                 logger.error(f"Failed to execute SQL for change {entry.entity_uuid}: {e}", exc_info=True)
                 # Re-raise as ApplyError? Yes, helps identify problematic change.
                 raise ApplyError(f"SQL execution failed: {e}", change_id=entry.change_id, entity_uuid=entry.entity_uuid) from e

        elif resolution == 'error':
            logger.error(f"Conflict resolution error for {entry.entity_uuid}. Skipping change.")
            # Potentially raise an exception or log more details
            return False
        else:
            # Should not happen if resolver returns valid strings
            logger.error(f"Unknown resolution type '{resolution}' for {entry.entity_uuid}. Skipping change.")
            return False

    def _apply_link_change(self, entry: SyncLogEntry, conn: sqlite3.Connection) -> bool:
        """Applies a 'link' or 'unlink' operation for MediaKeywords."""
        if not entry.payload or 'media_uuid' not in entry.payload or 'keyword_uuid' not in entry.payload:
            logger.error(f"Invalid or missing payload for MediaKeywords {entry.operation} (UUID: {entry.entity_uuid}).")
            return False # Cannot apply

        media_uuid = entry.payload['media_uuid']
        keyword_uuid = entry.payload['keyword_uuid']

        # Get local IDs for the parent entities
        media_id = get_local_id_from_uuid(self.db, "Media", media_uuid)
        keyword_id = get_local_id_from_uuid(self.db, "Keywords", keyword_uuid)

        if media_id is None or keyword_id is None:
             # One or both parents don't exist locally yet, defer this operation
             logger.warning(f"Deferring MediaKeywords {entry.operation} for {entry.entity_uuid}: Parent Media ({media_uuid}={media_id}) or Keyword ({keyword_uuid}={keyword_id}) not found locally.")
             if entry not in self._deferred_links: # Avoid duplicates if retrying
                  self._deferred_links.append(entry)
             return False # Not applied yet

        # Parents exist, proceed with link/unlink
        try:
            if entry.operation == 'link':
                sql = "INSERT OR IGNORE INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)"
                conn.execute(sql, (media_id, keyword_id))
                logger.debug(f"Applied LINK for Media ID {media_id} and Keyword ID {keyword_id}")
                return True
            elif entry.operation == 'unlink':
                sql = "DELETE FROM MediaKeywords WHERE media_id = ? AND keyword_id = ?"
                conn.execute(sql, (media_id, keyword_id))
                logger.debug(f"Applied UNLINK for Media ID {media_id} and Keyword ID {keyword_id}")
                return True
            else:
                logger.error(f"Unknown operation '{entry.operation}' for MediaKeywords.")
                return False
        except sqlite3.Error as e:
            logger.error(f"Failed SQL for MediaKeywords {entry.operation} ({media_id}, {keyword_id}): {e}", exc_info=True)
            raise ApplyError(f"SQL execution failed for link/unlink: {e}", change_id=entry.change_id, entity_uuid=entry.entity_uuid) from e


    def synchronize(self):
        """Performs a full sync cycle: fetch remote, apply local, send local."""
        if not self._sync_lock.acquire(blocking=False):
            logger.warning("Synchronization is already in progress. Skipping this run.")
            return

        logger.info(f"Starting synchronization cycle for client: {self.client_id}")
        try:
            # 1. Get current sync state
            last_remote_ts, last_local_id = self.state_manager.get_sync_cursors()

            # 2. Fetch remote changes
            logger.info(f"Fetching remote changes since {last_remote_ts}...")
            remote_changes = self.transport.fetch_changes(self.client_id, last_remote_ts)

            # 3. Apply remote changes
            if remote_changes:
                 logger.info(f"Applying {len(remote_changes)} remote changes...")
                 latest_applied_ts = self._apply_remote_changes(remote_changes)
                 # Update state only if changes were successfully applied
                 if latest_applied_ts is not None:
                      self.state_manager.update_last_processed_remote_ts(latest_applied_ts)
                 else:
                      logger.info("No remote changes were successfully applied.")
            else:
                 logger.info("No new remote changes fetched.")

            # 4. Get local changes
            logger.info(f"Getting local changes since change_id {last_local_id}...")
            local_changes = self._get_local_changes(last_local_id)

            # 5. Send local changes
            if local_changes:
                logger.info(f"Sending {len(local_changes)} local changes...")
                # Determine the change_id of the last change being sent
                max_sent_change_id = max(change.change_id for change in local_changes) if local_changes else last_local_id

                send_success = self.transport.send_changes(self.client_id, local_changes)
                if send_success:
                    # Update local state only if send was successful
                    self.state_manager.update_last_sent_local_change_id(max_sent_change_id)
                    logger.info(f"Successfully sent local changes up to change_id {max_sent_change_id}.")
                else:
                     logger.warning("Failed to send local changes. Will retry next cycle.")
                     # Do not update last_sent_local_change_id
            else:
                logger.info("No new local changes to send.")

            logger.info(f"Synchronization cycle completed for client: {self.client_id}")

        except (SyncError, TransportError, ApplyError, StateError, DatabaseError) as e:
             # Log errors from different stages
             logger.error(f"Synchronization cycle failed: {type(e).__name__} - {e}", exc_info=True)
             # Depending on the error, might need specific recovery logic
        except Exception as e:
             # Catch unexpected errors
             logger.critical(f"Unexpected critical error during synchronization: {e}", exc_info=True)
        finally:
            self._sync_lock.release()
            logger.debug("Sync lock released.")