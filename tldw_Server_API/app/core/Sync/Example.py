import logging
import time
from ..Sync import (
    SyncManager,
    HttpApiTransport,
    LastWriteWinsStrategy,
    SyncStateManager,
)
# Assuming your Database class is accessible
from PoC_Version.App_Function_Libraries.DB.SQLite_DB import Database

# --- Configuration ---
DB_PATH = "path/to/your/user.db"
CLIENT_ID = "unique_client_identifier_123" # Get this persistently
API_BASE_URL = "http://your-sync-server.com/api/"
STATE_FILE = f".sync_state_{CLIENT_ID}.json" # Client-specific state file

# --- Setup ---

try:
    db = Database(DB_PATH)
    transport = HttpApiTransport(API_BASE_URL)
    resolver = LastWriteWinsStrategy()
    state_manager = SyncStateManager(STATE_FILE)

    sync_manager = SyncManager(
        db_instance=db,
        transport=transport,
        resolver=resolver,
        state_manager=state_manager,
        client_id=CLIENT_ID
    )

    # --- Run Sync Periodically ---
    while True: # In a real app, use a scheduler or trigger
        logging.info("Starting scheduled sync...")
        try:
            sync_manager.synchronize()
            logging.info("Sync finished.")
        except Exception as e:
             logging.error(f"Sync run failed: {e}", exc_info=True)

        time.sleep(60 * 5) # Sync every 5 minutes

except Exception as e:
    logging.critical(f"Failed to initialize sync components: {e}", exc_info=True)


