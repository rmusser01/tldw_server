# sync_library/state.py
import json
import os
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple

from .exceptions import StateError
from .models import parse_timestamp # Use the parser from models

logger = logging.getLogger(__name__)
DEFAULT_STATE_FILE = ".sync_state.json"

class SyncStateManager:
    """Manages persistent sync state using a JSON file."""

    def __init__(self, state_file_path: str = DEFAULT_STATE_FILE):
        self.state_file_path = state_file_path
        logger.info(f"Sync state manager initialized with file: {self.state_file_path}")

    def _load_state(self) -> Dict:
        """Loads state from the JSON file."""
        if not os.path.exists(self.state_file_path):
            logger.warning(f"Sync state file not found: {self.state_file_path}. Returning default state.")
            return {} # Default empty state
        try:
            with open(self.state_file_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
                logger.debug(f"Loaded sync state: {state}")
                return state
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error loading sync state from {self.state_file_path}: {e}", exc_info=True)
            raise StateError(f"Failed to load sync state: {e}") from e

    def _save_state(self, state: Dict):
        """Saves state to the JSON file."""
        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(self.state_file_path) or '.', exist_ok=True)
            with open(self.state_file_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"Saved sync state: {state}")
        except IOError as e:
            logger.error(f"Error saving sync state to {self.state_file_path}: {e}", exc_info=True)
            raise StateError(f"Failed to save sync state: {e}") from e

    def get_sync_cursors(self) -> Tuple[Optional[datetime], int]:
        """
        Gets the last processed remote timestamp and last sent local change ID.

        Returns:
            Tuple[Optional[datetime], int]: (last_processed_remote_ts_utc, last_sent_local_change_id)
                                            Returns (None, 0) if no state exists.
        """
        state = self._load_state()
        ts_str = state.get('last_processed_remote_ts_utc')
        last_ts = parse_timestamp(ts_str) if ts_str else None
        last_id = state.get('last_sent_local_change_id', 0)
        logger.info(f"Retrieved sync cursors: Last Remote TS = {last_ts}, Last Local ID = {last_id}")
        return last_ts, last_id

    def update_last_processed_remote_ts(self, timestamp: Optional[datetime]):
        """Updates the timestamp of the last successfully processed remote change."""
        if timestamp is None:
            logger.warning("Attempted to update last processed remote timestamp with None.")
            return
        # Ensure timestamp is UTC
        if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) != timezone.utc.utcoffset(None):
             logger.warning(f"Received non-UTC timestamp {timestamp}, converting to UTC for state.")
             timestamp = timestamp.astimezone(timezone.utc)

        state = self._load_state()
        # Store as ISO string for JSON compatibility
        state['last_processed_remote_ts_utc'] = timestamp.isoformat().replace('+00:00', 'Z')
        self._save_state(state)
        logger.info(f"Updated last processed remote timestamp to: {timestamp}")

    def update_last_sent_local_change_id(self, change_id: int):
        """Updates the change_id of the last successfully sent local change."""
        if not isinstance(change_id, int) or change_id < 0:
             logger.error(f"Invalid change_id provided: {change_id}. Must be non-negative integer.")
             raise ValueError("Invalid change_id")

        state = self._load_state()
        current_last_id = state.get('last_sent_local_change_id', 0)
        if change_id > current_last_id: # Only update if it's newer
            state['last_sent_local_change_id'] = change_id
            self._save_state(state)
            logger.info(f"Updated last sent local change ID to: {change_id}")
        elif change_id < current_last_id:
            logger.warning(f"Attempted to set last sent change ID to {change_id}, which is less than current {current_last_id}. Ignoring.")