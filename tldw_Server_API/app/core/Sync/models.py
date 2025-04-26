# sync_library/models.py
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import logging

# Assuming logger setup elsewhere or use default
logger = logging.getLogger(__name__)

# Helper to parse SQLite timestamps (assuming they are stored as UTC strings)
def parse_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    if not ts_str:
        return None
    try:
        # Try ISO format first (more common)
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        # Ensure it's timezone-aware (assume UTC if naive)
        if dt.tzinfo is None:
             dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        try:
            # Fallback for space separator
            dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
            dt = dt.replace(tzinfo=timezone.utc) # Assume UTC
            return dt
        except ValueError:
             logger.warning(f"Could not parse timestamp string: {ts_str}", exc_info=True)
             return None

@dataclass
class SyncLogEntry:
    change_id: int
    entity: str
    entity_uuid: str
    operation: str # 'create', 'update', 'delete', 'link', 'unlink'
    timestamp: datetime # Store as datetime obj, assumed UTC
    client_id: str
    version: int
    payload_str: Optional[str] = field(repr=False) # Keep raw payload string for debugging
    payload: Optional[Dict[str, Any]] = field(default=None) # Parsed payload

    def __post_init__(self):
        """Parse the JSON payload after initialization."""
        if self.payload_str and self.payload is None:
            try:
                self.payload = json.loads(self.payload_str)
                # Optional: Convert specific payload fields (like dates) if needed
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON payload for change_id {self.change_id}, uuid {self.entity_uuid}: {self.payload_str[:100]}...")
                self.payload = None # Indicate parsing failure

    @classmethod
    def from_row(cls, row: sqlite3.Row):
        """Creates a SyncLogEntry from a database row."""
        timestamp = parse_timestamp(row['timestamp'])
        if timestamp is None:
            logger.error(f"Failed to parse timestamp '{row['timestamp']}' for change_id {row['change_id']}. Using epoch as fallback.")
            # Fallback or raise error? Using epoch might be risky. Raising might be better.
            # Let's use epoch for now but log prominently.
            timestamp = datetime.fromtimestamp(0, timezone.utc)

        return cls(
            change_id=row['change_id'],
            entity=row['entity'],
            entity_uuid=row['entity_uuid'],
            operation=row['operation'],
            timestamp=timestamp,
            client_id=row['client_id'],
            version=row['version'],
            payload_str=row['payload'] # Store raw string
            # Let __post_init__ handle parsing payload_str
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the entry to a dictionary suitable for transport (e.g., JSON)."""
        return {
            "change_id": self.change_id, # Usually local only, might not send
            "entity": self.entity,
            "entity_uuid": self.entity_uuid,
            "operation": self.operation,
            "timestamp": self.timestamp.isoformat().replace('+00:00', 'Z'), # ISO 8601 UTC format
            "client_id": self.client_id,
            "version": self.version,
            "payload": self.payload # Send parsed payload
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
         """Creates an entry from a dictionary received via transport."""
         timestamp = parse_timestamp(data.get("timestamp"))
         if timestamp is None:
             raise ValueError(f"Invalid or missing timestamp in received data: {data.get('timestamp')}")

         # Payload might already be a dict if coming from JSON
         payload = data.get("payload")
         payload_str_representation = json.dumps(payload) if payload else None

         return cls(
            change_id=data.get("change_id", 0), # change_id might not be sent/relevant from remote
            entity=data["entity"],
            entity_uuid=data["entity_uuid"],
            operation=data["operation"],
            timestamp=timestamp,
            client_id=data["client_id"],
            version=data["version"],
            payload=payload,
            payload_str=payload_str_representation # Reconstruct string if needed
        )