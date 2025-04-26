# sync_library/db_utils.py
import sqlite3
from typing import Optional, Dict, Any
import logging

# Adjust import if your Database class is elsewhere
from PoC_Version.App_Function_Libraries.DB.SQLite_DB import Database, DatabaseError

logger = logging.getLogger(__name__)

# Mapping from sync log entity names to actual table names and UUID column
# Expand this as needed
ENTITY_CONFIG = {
    "Media": {"table": "Media", "uuid_col": "uuid"},
    "Keywords": {"table": "Keywords", "uuid_col": "uuid"},
    "Transcripts": {"table": "Transcripts", "uuid_col": "uuid"},
    "MediaChunks": {"table": "MediaChunks", "uuid_col": "uuid"},
    "UnvectorizedMediaChunks": {"table": "UnvectorizedMediaChunks", "uuid_col": "uuid"},
    "DocumentVersions": {"table": "DocumentVersions", "uuid_col": "uuid"},
    # MediaKeywords is handled differently (link/unlink)
}

def get_entity_config(entity: str) -> Optional[Dict[str, str]]:
    """Gets table name and uuid column for a given entity type."""
    config = ENTITY_CONFIG.get(entity)
    if not config:
        logger.error(f"Unknown entity type encountered: {entity}")
    return config

def get_local_row_by_uuid(db: Database, entity: str, uuid: str) -> Optional[Dict[str, Any]]:
    """Fetches the current local row data for a given entity UUID."""
    config = get_entity_config(entity)
    if not config:
        return None

    table_name = config['table']
    uuid_col = config['uuid_col']

    # Construct query safely
    # Use * for simplicity, or list specific columns if preferred
    # Important: Fetches all columns, including sync metadata like version, last_modified, deleted
    query = f"SELECT * FROM {table_name} WHERE {uuid_col} = ?"

    try:
        cursor = db.execute_query(query, (uuid,))
        row = cursor.fetchone()
        return dict(row) if row else None
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error fetching local row for {entity} UUID {uuid}: {e}", exc_info=True)
        return None # Indicate error or inability to fetch

def get_local_id_from_uuid(db: Database, entity: str, uuid: str) -> Optional[int]:
    """Gets the local primary key (id) for a given entity UUID."""
    config = get_entity_config(entity)
    if not config:
        return None

    table_name = config['table']
    uuid_col = config['uuid_col']

    query = f"SELECT id FROM {table_name} WHERE {uuid_col} = ? LIMIT 1"
    try:
        cursor = db.execute_query(query, (uuid,))
        row = cursor.fetchone()
        return row['id'] if row else None
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error fetching local ID for {entity} UUID {uuid}: {e}", exc_info=True)
        return None