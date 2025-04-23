# test_utils.py
# Description: Shared functions used across unit tests.
#
# Imports
import os
from contextlib import contextmanager
import tempfile
import logging # Added for potential logging
#
# Local Imports
from tldw_Server_API.app.core.DB_Management.Media_DB import Database
#
########################################################################################################################
# Functions:

@contextmanager
def temp_db():
    """
    Creates a temporary directory and yields an initialized Database instance
    pointing to a test.db file within that directory.
    Relies on Database class's __init__ to handle schema creation.
    """
    db_instance = None # Initialize to None
    # Create a temporary directory; it will be deleted automatically
    with tempfile.TemporaryDirectory() as temp_dir:
        # Build a database path inside the temporary directory.
        db_path = os.path.join(temp_dir, "test.db")
        logging.debug(f"Creating temporary DB at: {db_path}")
        # --- Initialize Database instance - This calls _ensure_schema internally ---
        db_instance = Database(db_path)
        try:
            # --- REMOVED: create_tables(db) ---

            # Optionally, verify the schema AFTER initialization
            # verify_media_db_schema(db_instance) # Keep if desired

            # Yield the fresh database instance.
            yield db_instance
        finally:
            # Ensure the connection associated with this instance is closed
            if db_instance:
                logging.debug(f"Closing connection for temporary DB: {db_path}")
                db_instance.close_connection() # Use the instance's close method

def verify_media_db_schema(db):
    """Ensure critical columns exist in Media table."""
    # Make sure this function uses the instance's connection method
    conn = None
    try:
        conn = db.get_connection() # Get connection from the instance
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(Media)")
        columns = [col['name'] for col in cursor.fetchall()] # Access by name using Row factory

        critical_columns = {'id', 'author', 'title', 'content', 'type', 'content_hash'} # Update if needed
        missing = critical_columns - set(columns)

        if missing:
            # Raise a more specific error or log it
            logging.error(f"Schema Verification Failed! Media table missing columns: {missing}")
            raise RuntimeError(f"Media table missing columns: {missing}")
        else:
            logging.debug("verify_media_db_schema passed.")
    except Exception as e:
         logging.error(f"Error during schema verification: {e}", exc_info=True)
         raise # Re-raise the exception
    # No finally block needed to close conn, as get_connection manages thread-local connection


def create_test_media(db: Database, title: str, content: str, content_hash: str = "test_hash"):
    """Inserts a test document media item."""
    # Now just insert:
    # Ensure all NOT NULL columns are provided (like content_hash)
    db.execute_query(
        "INSERT INTO Media (title, type, content, author, content_hash) VALUES (?, ?, ?, ?, ?)",
        (title, "document", content, "Test Author", content_hash),
        commit=True # Commit this specific insert
    )
    # Get the ID of the inserted row
    cursor = db.execute_query("SELECT last_insert_rowid();")
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        raise RuntimeError("Failed to retrieve last insert rowid after creating test media.")


# End of test_utils.py
########################################################################################################################
