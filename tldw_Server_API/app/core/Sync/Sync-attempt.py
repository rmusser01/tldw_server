import json
import uuid
from datetime import datetime
import logging
import sqlite3
from typing import List, Dict, Any, Optional, Tuple

from tldw_Server_API.app.core.DB_Management.Media_DB import Database

# Assume 'Database' class exists from SQLite_DB.py
# Assume an API client exists to talk to the server (e.g., 'remote_api_client')

# --- Configuration ---
LOCAL_CLIENT_ID = str(uuid.uuid4()) # Generate or load a persistent client ID for this device
CONFLICT_STRATEGY = 'Merge' # Options: 'Merge', 'ServerWins', 'ClientWins', 'Manual'
LAST_SYNC_TS_FILE = 'last_sync_ts.txt' # Store the timestamp persistently

def get_last_sync_ts() -> str:
    try:
        with open(LAST_SYNC_TS_FILE, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        return '1970-01-01 00:00:00' # Default for first sync

def save_last_sync_ts(timestamp: str):
    with open(LAST_SYNC_TS_FILE, 'w') as f:
        f.write(timestamp)

def get_local_changes(db_instance: Database, last_sync_ts: str) -> List[Dict]:
    """Fetch local changes from sync_log since last sync."""
    query = "SELECT * FROM sync_log WHERE timestamp > ? ORDER BY timestamp ASC"
    try:
        cursor = db_instance.execute_query(query, (last_sync_ts,))
        changes = [dict(row) for row in cursor.fetchall()]
        logging.info(f"Found {len(changes)} local changes since {last_sync_ts}")
        return changes
    except Exception as e:
        logging.error(f"Error fetching local changes: {e}", exc_info=True)
        return []

# Placeholder for API interaction
async def send_changes_to_server(changes: List[Dict]) -> bool:
    # Replace with actual API call using your HTTP client (e.g., httpx, requests)
    logging.info(f"Sending {len(changes)} changes to server...")
    # response = await remote_api_client.post('/sync/push', json=changes)
    # return response.status_code == 200
    await asyncio.sleep(0.1) # Simulate network
    print(f"SIMULATED: Sent {len(changes)} changes.")
    return True # Simulate success

async def get_changes_from_server(last_sync_ts: str, client_id: str) -> Tuple[List[Dict], str]:
    # Replace with actual API call
    logging.info(f"Fetching changes from server since {last_sync_ts} (excluding client {client_id})...")
    # response = await remote_api_client.get('/sync/pull', params={'since': last_sync_ts, 'exclude_client': client_id})
    # if response.status_code == 200:
    #   data = response.json()
    #   return data.get('changes', []), data.get('server_timestamp', last_sync_ts)
    # else:
    #   return [], last_sync_ts
    await asyncio.sleep(0.1) # Simulate network
    print(f"SIMULATED: Fetched 0 changes from server.")
    # Simulate server returning its current time as the new potential sync point
    new_sync_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] # Example format
    return [], new_sync_ts # Simulate no changes received, return current time

def apply_changes_locally(db_instance: Database, changes: List[Dict], local_client_id: str):
    """Apply changes received from the server locally, handling conflicts."""
    applied_count = 0
    skipped_count = 0
    conflict_count = 0
    max_processed_ts = '1970-01-01 00:00:00'

    for change in changes:
        entity = change['entity']
        entity_uuid = change['entity_uuid']
        operation = change['operation']
        payload_str = change['payload']
        change_version = change['version']
        change_ts = change['timestamp']
        change_client_id = change['client_id']

        try:
            payload = json.loads(payload_str) if payload_str else {}

            # --- Handle Relationship Changes Separately ---
            if entity == 'MediaKeywords':
                handle_media_keyword_sync(db_instance, operation, payload)
                applied_count += 1
                max_processed_ts = max(max_processed_ts, change_ts)
                continue # Move to next change

            # --- Handle Entity Changes (Media, Keywords, etc.) ---
            local_record = get_local_record_by_uuid(db_instance, entity, entity_uuid)

            if operation == 'create':
                if local_record:
                    logging.warning(f"Conflict (CREATE): Record {entity} UUID {entity_uuid} already exists locally.")
                    # Treat as an update conflict
                    handle_conflict(db_instance, local_record, change, local_client_id)
                    conflict_count += 1
                else:
                    logging.debug(f"Applying CREATE for {entity} UUID {entity_uuid}")
                    insert_local_record(db_instance, entity, payload)
                    applied_count += 1

            elif operation == 'delete':
                if local_record and not local_record['deleted']:
                    logging.debug(f"Applying DELETE for {entity} UUID {entity_uuid}")
                    mark_local_record_deleted(db_instance, entity, entity_uuid, change)
                    applied_count += 1
                else:
                    # Already deleted or doesn't exist locally, skip
                    logging.debug(f"Skipping DELETE for {entity} UUID {entity_uuid} (already deleted or non-existent)")
                    skipped_count += 1

            elif operation == 'update':
                if not local_record:
                    # Record doesn't exist locally (maybe deleted locally?)
                    # Decide: Insert it? Or ignore? For now, ignore.
                    logging.warning(f"Skipping UPDATE for non-existent local {entity} UUID {entity_uuid}")
                    skipped_count += 1
                elif local_record['deleted']:
                     # Local record is marked deleted, incoming is an update.
                     # Server probably didn't get the delete yet. Ignore server update for now.
                     logging.warning(f"Skipping UPDATE for locally deleted {entity} UUID {entity_uuid}")
                     skipped_count += 1
                else:
                    # --- Conflict Check ---
                    local_version = local_record['version']
                    if change_version > local_version:
                        # No conflict, incoming change is newer
                        logging.debug(f"Applying UPDATE for {entity} UUID {entity_uuid} (Incoming version {change_version} > Local {local_version})")
                        update_local_record(db_instance, entity, payload)
                        applied_count += 1
                    elif change_version == local_version:
                        # Same version - possible concurrent edit or duplicate log entry. Use timestamp.
                        if change_ts > local_record['last_modified']:
                             logging.debug(f"Applying UPDATE for {entity} UUID {entity_uuid} (Same version {change_version}, incoming timestamp newer)")
                             update_local_record(db_instance, entity, payload)
                             applied_count += 1
                        else:
                             logging.debug(f"Skipping UPDATE for {entity} UUID {entity_uuid} (Same version {change_version}, local timestamp newer or equal)")
                             skipped_count += 1
                    else: # change_version < local_version
                        # CONFLICT! Local changes are newer based on version.
                        logging.warning(f"Conflict (UPDATE): Local version {local_version} > Incoming {change_version} for {entity} UUID {entity_uuid}")
                        handle_conflict(db_instance, local_record, change, local_client_id)
                        conflict_count += 1

            # Update max processed timestamp only if successfully processed or intentionally skipped
            if operation in ['create', 'delete', 'update']: # Exclude skipped due to non-existence etc. unless intended
                 max_processed_ts = max(max_processed_ts, change_ts)

        except Exception as e:
            logging.error(f"Error applying change for {entity} UUID {entity_uuid}: {e}", exc_info=True)
            # Decide: stop sync, skip change, log to failed changes table?
            skipped_count += 1 # For now, just skip the problematic change

    logging.info(f"Finished applying remote changes. Applied: {applied_count}, Skipped: {skipped_count}, Conflicts: {conflict_count}")
    return max_processed_ts

# --- Helper functions for DB interaction during sync ---

def get_local_record_by_uuid(db_instance: Database, entity: str, uuid_val: str) -> Optional[Dict]:
    """Fetch a local record including sync metadata."""
    # Map entity name to table name (add other tables)
    table_map = {
        'Media': 'Media',
        'Keywords': 'Keywords',
        'DocumentVersions': 'DocumentVersions',
        'Transcripts': 'Transcripts',
        # ... add other synced tables
    }
    table_name = table_map.get(entity)
    if not table_name:
        logging.error(f"Unknown entity type '{entity}' in get_local_record_by_uuid")
        return None

    query = f"SELECT *, CAST(deleted AS INTEGER) AS deleted_int FROM {table_name} WHERE uuid = ?"
    try:
        cursor = db_instance.execute_query(query, (uuid_val,))
        row = cursor.fetchone()
        if row:
            record = dict(row)
            record['deleted'] = bool(record.pop('deleted_int', 0)) # Ensure 'deleted' is boolean
            return record
        return None
    except Exception as e:
        logging.error(f"Error fetching local record {entity}/{uuid_val}: {e}")
        return None

def insert_local_record(db_instance: Database, entity: str, payload: Dict):
    """Insert a record using data from the sync payload."""
    # WARNING: This requires mapping payload keys to table columns and using the correct SQL INSERT.
    # Needs careful implementation for each entity type.
    # IMPORTANT: Use payload's version, last_modified, client_id, deleted.
    table_map = { 'Media': 'Media', 'Keywords': 'Keywords', 'DocumentVersions': 'DocumentVersions', 'Transcripts': 'Transcripts', } #...
    table_name = table_map.get(entity)
    if not table_name: return

    # Construct columns and placeholders dynamically (SAFER: define explicitly per table)
    # Example for Media (adjust columns based on trigger payload)
    if entity == 'Media':
        cols = ['uuid', 'url', 'title', 'type', 'content', 'author', 'ingestion_date',
                'transcription_model', 'is_trash', 'content_hash', 'last_modified',
                'version', 'client_id', 'deleted']
        placeholders = ', '.join(['?'] * len(cols))
        values = [
            payload.get('uuid'), payload.get('url'), payload.get('title'), payload.get('type'),
            payload.get('content'), payload.get('author'), payload.get('ingestion_date'),
            payload.get('transcription_model'), payload.get('is_trash', 0), # Default is_trash if missing
            payload.get('content_hash'), payload.get('last_modified'),
            payload.get('version'), payload.get('client_id'), payload.get('deleted', 0)
        ]
        # Find parent media_id for DocumentVersions etc. if needed using media_uuid from payload
        # Example: if entity == 'DocumentVersions': values.insert(index_of_media_id, get_media_id_from_uuid(db, payload['media_uuid']))

        query = f"INSERT OR REPLACE INTO {table_name} ({', '.join(cols)}) VALUES ({placeholders})"
        try:
            db_instance.execute_query(query, tuple(values), commit=True)
            logging.debug(f"Inserted/Replaced {entity} UUID {payload.get('uuid')}")
        except Exception as e:
            logging.error(f"Error inserting {entity} UUID {payload.get('uuid')}: {e}")
            raise # Re-raise to be caught by apply_changes_locally

    # Add similar blocks for Keywords, DocumentVersions, etc.

def update_local_record(db_instance: Database, entity: str, payload: Dict):
    """Update a local record using data from the sync payload."""
    # WARNING: Similar to insert, needs careful implementation per entity.
    # IMPORTANT: Update sync metadata fields (version, last_modified, client_id, deleted) from payload.
    table_map = { 'Media': 'Media', 'Keywords': 'Keywords', 'DocumentVersions': 'DocumentVersions', 'Transcripts': 'Transcripts', } #...
    table_name = table_map.get(entity)
    if not table_name: return

    uuid_val = payload.get('uuid')
    if not uuid_val: return

    # Construct SET clause dynamically (SAFER: define explicitly per table)
    # Example for Media (adjust columns based on trigger payload)
    if entity == 'Media':
        set_clauses = [
            "url = ?", "title = ?", "type = ?", "content = ?", "author = ?", "ingestion_date = ?",
            "transcription_model = ?", "is_trash = ?", "content_hash = ?", "last_modified = ?",
            "version = ?", "client_id = ?", "deleted = ?"
        ]
        values = [
            payload.get('url'), payload.get('title'), payload.get('type'), payload.get('content'),
            payload.get('author'), payload.get('ingestion_date'), payload.get('transcription_model'),
            payload.get('is_trash', 0), payload.get('content_hash'), payload.get('last_modified'),
            payload.get('version'), payload.get('client_id'), payload.get('deleted', 0)
        ]
        values.append(uuid_val) # For the WHERE clause

        query = f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE uuid = ?"
        try:
            db_instance.execute_query(query, tuple(values), commit=True)
            logging.debug(f"Updated {entity} UUID {uuid_val}")
        except Exception as e:
            logging.error(f"Error updating {entity} UUID {uuid_val}: {e}")
            raise # Re-raise

     # Add similar blocks for Keywords, DocumentVersions, etc.


def mark_local_record_deleted(db_instance: Database, entity: str, uuid_val: str, change_payload: Dict):
    """Mark a local record as deleted based on sync log entry."""
    table_map = { 'Media': 'Media', 'Keywords': 'Keywords', 'DocumentVersions': 'DocumentVersions', 'Transcripts': 'Transcripts', } #...
    table_name = table_map.get(entity)
    if not table_name: return

    # Update deleted flag and sync metadata from the 'delete' log entry
    query = f"UPDATE {table_name} SET deleted = 1, version = ?, last_modified = ?, client_id = ? WHERE uuid = ?"
    params = (
        change_payload['version'],
        change_payload['timestamp'], # Use the timestamp from the log entry
        change_payload['client_id'], # Use the client_id from the log entry
        uuid_val
    )
    try:
        db_instance.execute_query(query, params, commit=True)
        logging.debug(f"Marked {entity} UUID {uuid_val} as deleted.")
    except Exception as e:
        logging.error(f"Error marking {entity} UUID {uuid_val} as deleted: {e}")
        raise # Re-raise


def handle_media_keyword_sync(db_instance: Database, operation: str, payload: Dict):
    """Handle linking/unlinking based on MediaKeywords sync log."""
    media_uuid = payload.get('media_uuid')
    keyword_uuid = payload.get('keyword_uuid')
    if not media_uuid or not keyword_uuid:
        logging.error("Missing media/keyword UUID in MediaKeywords sync payload.")
        return

    # Need to get internal integer IDs from UUIDs
    media_id = get_internal_id_from_uuid(db_instance, 'Media', media_uuid)
    keyword_id = get_internal_id_from_uuid(db_instance, 'Keywords', keyword_uuid)

    if not media_id or not keyword_id:
         logging.warning(f"Could not find internal IDs for MediaUUID {media_uuid} or KeywordUUID {keyword_uuid}. Skipping {operation}.")
         return

    if operation == 'link':
        query = "INSERT OR IGNORE INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)"
        params = (media_id, keyword_id)
        try:
            db_instance.execute_query(query, params, commit=True)
            logging.debug(f"Applied LINK Media {media_id} <-> Keyword {keyword_id}")
        except Exception as e:
            logging.error(f"Error applying LINK for Media {media_id} / Keyword {keyword_id}: {e}")
    elif operation == 'unlink':
        query = "DELETE FROM MediaKeywords WHERE media_id = ? AND keyword_id = ?"
        params = (media_id, keyword_id)
        try:
            db_instance.execute_query(query, params, commit=True)
            logging.debug(f"Applied UNLINK Media {media_id} <-> Keyword {keyword_id}")
        except Exception as e:
            logging.error(f"Error applying UNLINK for Media {media_id} / Keyword {keyword_id}: {e}")

def get_internal_id_from_uuid(db_instance: Database, entity: str, uuid_val: str) -> Optional[int]:
    """Helper to get integer ID from UUID."""
    table_map = { 'Media': 'Media', 'Keywords': 'Keywords', } #... add others if needed
    table_name = table_map.get(entity)
    if not table_name or not uuid_val: return None
    query = f"SELECT id FROM {table_name} WHERE uuid = ?"
    try:
        cursor = db_instance.execute_query(query, (uuid_val,))
        result = cursor.fetchone()
        return result[0] if result else None
    except:
        return None # Ignore errors for this helper

def handle_conflict(db_instance: Database, local_record: Dict, incoming_change: Dict, local_client_id: str):
    """Apply conflict resolution strategy."""
    global CONFLICT_STRATEGY # Access the global setting (or pass as arg)
    entity = incoming_change['entity']
    uuid_val = incoming_change['entity_uuid']
    payload = json.loads(incoming_change['payload'])

    logging.info(f"Handling conflict for {entity} UUID {uuid_val} using strategy: {CONFLICT_STRATEGY}")

    if CONFLICT_STRATEGY == 'ServerWins': # Assumes incoming is from server
        logging.info(f"Conflict resolution: ServerWins. Applying incoming change for {uuid_val}.")
        update_local_record(db_instance, entity, payload) # Overwrite local with incoming
    elif CONFLICT_STRATEGY == 'ClientWins': # Assumes incoming is from server
        logging.info(f"Conflict resolution: ClientWins. Ignoring incoming change for {uuid_val}.")
        # Do nothing, keep local version
    elif CONFLICT_STRATEGY == 'Manual':
        logging.info(f"Conflict resolution: Manual. Flagging {uuid_val} for manual review.")
        # Optional: Add a flag to the local record
        # mark_for_manual_resolution(db_instance, entity, uuid_val)
    elif CONFLICT_STRATEGY == 'Merge':
        logging.info(f"Conflict resolution: Merge. Attempting merge for {uuid_val}.")
        merged_payload = attempt_merge(local_record, payload)
        if merged_payload:
            # Apply merged result, bump version, set local client_id
            merged_payload['version'] = local_record['version'] + 1 # Bump local version
            merged_payload['last_modified'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            merged_payload['client_id'] = local_client_id # Mark change as local merge
            update_local_record(db_instance, entity, merged_payload)
            # IMPORTANT: The trigger *will* fire for this update, creating a new sync_log entry
            # for the merged state, which is correct.
            logging.info(f"Merge successful for {uuid_val}. Applied merged result.")
        else:
            # Merge failed or not possible, fallback to manual? Or ServerWins?
            logging.warning(f"Automatic merge failed for {uuid_val}. Flagging for manual review (or apply fallback).")
            # mark_for_manual_resolution(db_instance, entity, uuid_val) # Example fallback
    else:
        logging.error(f"Unknown conflict strategy: {CONFLICT_STRATEGY}. Ignoring conflict for {uuid_val}.")


def attempt_merge(local_data: Dict, incoming_payload: Dict) -> Optional[Dict]:
    """
    Attempts to merge fields based on simple rules.
    Returns merged payload or None if automatic merge isn't possible/defined.
    """
    # --- Define Merge Rules per field ---
    # Example rules:
    # - Keep the non-null value if one is null
    # - Keep the longer string for 'content' or 'title'?
    # - Keep the latest timestamp for date fields?
    # - Concatenate list-like fields (e.g., keywords - though keywords are handled separately)
    # - For most fields, maybe default to incoming (ServerWins for that field) or local (ClientWins for field)

    merged = local_data.copy() # Start with local data
    conflict_found = False

    # Iterate through keys present in incoming payload that also exist locally
    for key, incoming_value in incoming_payload.items():
        if key in merged and key not in ['uuid', 'version', 'last_modified', 'client_id', 'deleted']: # Skip metadata
            local_value = merged[key]
            if local_value != incoming_value:
                # Field conflict! Apply rule.
                if key == 'title':
                    # Rule: Keep the incoming title (ServerWins for title)
                    merged[key] = incoming_value
                elif key == 'content':
                     # Rule: Keep the longer content? Or flag? Let's flag for now.
                     if incoming_value is not None and (local_value is None or len(str(incoming_value)) > len(str(local_value))):
                         merged[key] = incoming_value # Keep longer
                     # Otherwise keep local
                     pass # Keep local if longer or incoming is None
                elif key == 'is_trash':
                     # Rule: If either is True, keep True?
                     merged[key] = local_value or incoming_value
                # ... other fields ...
                else:
                    # Default rule: Keep incoming value (ServerWins for this field)
                    merged[key] = incoming_value
                    # Or signal merge failure if no rule exists: conflict_found = True; break

    if conflict_found:
        return None # Indicate automatic merge failed
    else:
        # Update metadata fields (version, etc.) outside this function
        # after calling update_local_record
        return merged

# --- Main Sync Function ---
import asyncio # Use asyncio if API calls are async

async def run_sync_cycle(db_instance: Database):
    """Performs one full sync cycle: push local changes, pull remote changes."""
    logging.info("Starting sync cycle...")
    last_sync_ts = get_last_sync_ts()
    current_cycle_start_time = datetime.now() # Use for next sync timestamp

    # 1. Get local changes
    local_changes = get_local_changes(db_instance, last_sync_ts)

    # 2. Push local changes to server
    if local_changes:
        push_success = await send_changes_to_server(local_changes)
        if not push_success:
            logging.error("Failed to push changes to server. Aborting sync cycle.")
            return
        logging.info("Successfully pushed local changes.")
    else:
        logging.info("No local changes to push.")

    # 3. Pull remote changes from server
    # The server response should ideally include the timestamp up to which it provided changes
    remote_changes, server_sync_point_ts = await get_changes_from_server(last_sync_ts, LOCAL_CLIENT_ID)

    # 4. Apply remote changes locally
    if remote_changes:
        logging.info(f"Received {len(remote_changes)} changes from server.")
        max_processed_ts = apply_changes_locally(db_instance, remote_changes, LOCAL_CLIENT_ID)
        # Use the server's sync point or the max processed TS, whichever is *earlier*? Safest.
        # Or use the server_sync_point_ts as the definitive marker for the next request.
        next_sync_ts = server_sync_point_ts
    else:
        logging.info("No remote changes received.")
        # If no remote changes, we can potentially update sync ts to when we started *this* cycle
        # Or better, use the timestamp provided by the server indicating its current state.
        next_sync_ts = server_sync_point_ts # Use the timestamp server gave us

    # 5. Update last sync timestamp
    save_last_sync_ts(next_sync_ts)
    logging.info(f"Sync cycle completed. Next sync will be after {next_sync_ts}")


# --- Example of how to run it ---
# async def main():
#     db = Database(get_database_path('your_media_db.db')) # Use your path function
#     # You might run this periodically
#     await run_sync_cycle(db)

# if __name__ == "__main__":
#      # Setup logging
#      # ...
#      asyncio.run(main())