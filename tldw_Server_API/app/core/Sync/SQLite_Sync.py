# SQLite_Sync.py
## Description: This file contains functions relating to synchronizing a client's Media & Chat DBs with the server's instance and vice versa.
#
# Imports
#
# 3rd-party Libraries
#
# Local Imports
#
#######################################################################################################################
#
# Static Variables
#
# Functions:
import logging
import sqlite3
from datetime import datetime, timedelta

from tldw_Server_API.app.core.DB_Management.Media_DB import Database


def clear_old_sync_logs(db_instance: Database, days_to_keep: int = 30):
    """
    Deletes entries from the sync_log table older than a specified number of days.

    Args:
        db_instance: The Database instance.
        days_to_keep: How many days of sync history to retain.
    """
    if not isinstance(db_instance, Database):
        logging.error("clear_old_sync_logs: Invalid Database instance provided.")
        raise TypeError("A valid Database instance must be provided.")
    if not isinstance(days_to_keep, int) or days_to_keep < 0:
        logging.warning(f"Invalid days_to_keep value ({days_to_keep}), defaulting to 30.")
        days_to_keep = 30

    try:
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_timestamp_str = cutoff_date.strftime('%Y-%m-%d %H:%M:%S')

        logging.info(f"Clearing sync_log entries older than {days_to_keep} days (before {cutoff_timestamp_str})...")

        # Use the execute_query method with commit=True
        # Pass parameters correctly as a tuple
        query = "DELETE FROM sync_log WHERE timestamp <= ?"
        params = (cutoff_timestamp_str,)
        cursor = db_instance.execute_query(query, params, commit=True)

        deleted_count = cursor.rowcount
        logging.info(f"Successfully cleared {deleted_count} old sync_log entries.")
        return deleted_count

    except sqlite3.Error as e:
        logging.error(f"Database error clearing old sync_log entries: {e}", exc_info=True)
        # Depending on severity, you might want to raise an exception
        # raise DatabaseError(f"Failed to clear sync log: {e}") from e
        return -1 # Indicate error
    except Exception as e:
        logging.error(f"Unexpected error clearing old sync_log entries: {e}", exc_info=True)
        # raise DatabaseError(f"Unexpected error clearing sync log: {e}") from e
        return -1 # Indicate error

# Example Usage (e.g., in a scheduled task or maintenance routine):
# db = Database(db_path)
# cleared_count = clear_old_sync_logs(db, days_to_keep=60)
# if cleared_count >= 0:
#     print(f"Cleaned up {cleared_count} log entries.")
# else:
#     print("Error cleaning sync log.")


#
## End of SQLite_Sync.py
########################################################################################################################
