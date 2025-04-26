# sync_library/conflict.py
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import logging

from .models import SyncLogEntry

logger = logging.getLogger(__name__)

class ConflictResolver(ABC):
    """Abstract base class for conflict resolution strategies."""

    @abstractmethod
    def resolve(self, local_row_data: Optional[Dict[str, Any]], remote_change: SyncLogEntry) -> str:
        """
        Determines the outcome when a remote change conflicts with local data.

        Args:
            local_row_data: Current state of the local row (dict from DB query), or None if it doesn't exist locally.
            remote_change: The incoming SyncLogEntry from the remote.

        Returns:
            'apply_remote': Apply the remote change.
            'keep_local': Keep the local version, ignore the remote change.
            'error': An unresolvable conflict or error occurred.
        """
        pass

class LastWriteWinsStrategy(ConflictResolver):
    """Resolves conflicts using Last Write Wins based on timestamp, then client_id."""

    def resolve(self, local_row_data: Optional[Dict[str, Any]], remote_change: SyncLogEntry) -> str:
        entity_uuid = remote_change.entity_uuid
        remote_ts = remote_change.timestamp
        remote_op = remote_change.operation

        if local_row_data is None:
            # Local record doesn't exist
            if remote_op == 'delete':
                logger.debug(f"Conflict resolution (UUID: {entity_uuid}): Local nonexistent, remote DELETE. Outcome: Keep Local (ignore).")
                return 'keep_local' # Nothing to delete
            else: # create or update (undelete)
                logger.debug(f"Conflict resolution (UUID: {entity_uuid}): Local nonexistent, remote {remote_op}. Outcome: Apply Remote.")
                return 'apply_remote'

        # Local record exists
        # Safely get local timestamp, defaulting to epoch if missing (shouldn't happen with schema)
        local_ts_str = local_row_data.get('last_modified')
        local_ts = parse_timestamp(local_ts_str) or datetime.fromtimestamp(0, timezone.utc)
        local_deleted = bool(local_row_data.get('deleted', 0))
        local_client_id = local_row_data.get('client_id', '')
        remote_client_id = remote_change.client_id

        logger.debug(f"Conflict check (UUID: {entity_uuid}): Remote TS={remote_ts}, Local TS={local_ts}, Remote Op={remote_op}, Local Deleted={local_deleted}")

        if remote_ts > local_ts:
            logger.debug(f"Conflict resolution (UUID: {entity_uuid}): Remote TS > Local TS. Outcome: Apply Remote.")
            return 'apply_remote'
        elif remote_ts < local_ts:
            logger.debug(f"Conflict resolution (UUID: {entity_uuid}): Remote TS < Local TS. Outcome: Keep Local.")
            return 'keep_local'
        else:
            # Timestamps are equal, apply tie-breaking logic
            logger.debug(f"Conflict resolution (UUID: {entity_uuid}): Timestamps equal. Tie-breaking...")

            # Rule 1: Delete operations often win or are idempotent if timestamps match
            if remote_op == 'delete':
                 if local_deleted:
                     logger.debug(f"Conflict resolution (UUID: {entity_uuid}): TS equal, remote DELETE, local already deleted. Outcome: Keep Local (no change).")
                     return 'keep_local' # Already deleted locally
                 else:
                     logger.debug(f"Conflict resolution (UUID: {entity_uuid}): TS equal, remote DELETE, local not deleted. Outcome: Apply Remote (delete).")
                     return 'apply_remote' # Apply the delete

            # Rule 2: Undelete operations win if timestamps match and local is deleted
            if remote_op != 'delete' and local_deleted:
                 logger.debug(f"Conflict resolution (UUID: {entity_uuid}): TS equal, remote UPDATE/CREATE, local deleted. Outcome: Apply Remote (undelete).")
                 return 'apply_remote'

            # Rule 3: Tie-break based on client_id (e.g., higher client ID wins)
            if remote_client_id > local_client_id:
                logger.debug(f"Conflict resolution (UUID: {entity_uuid}): TS equal, Tie-break Client ID: Remote ('{remote_client_id}') > Local ('{local_client_id}'). Outcome: Apply Remote.")
                return 'apply_remote'
            else:
                logger.debug(f"Conflict resolution (UUID: {entity_uuid}): TS equal, Tie-break Client ID: Remote ('{remote_client_id}') <= Local ('{local_client_id}'). Outcome: Keep Local.")
                return 'keep_local'