# test_utils.py
# Description: Shared functions used across unit tests.
#
# Imports
import os
import sqlite3
import uuid
from contextlib import contextmanager
import tempfile
import logging # Added for potential logging
from pathlib import Path

#
# Local Imports
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase, DatabaseError


#
########################################################################################################################
# Functions:

@contextmanager
def temp_db(client_id: str = None):
    """Provides a temporary, function-scoped database instance."""
    if client_id is None:
        # Generate a unique client ID for each test function instance
        client_id = f"test_client_{uuid.uuid4()}"

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / f"test_{client_id}.db"
        db = None
        try:
            logging.debug(f"Creating temp DB instance for test: {db_path} (Client: {client_id})")
            # --- Ensure this uses the CORRECT imported Database class ---
            db = MediaDatabase(db_path=db_path, client_id=client_id)
            # -------------------------------------------------------------
            yield db
        except (DatabaseError, sqlite3.Error) as e:
             logging.error(f"Failed to create/initialize temp DB {db_path}: {e}", exc_info=True)
             # Re-raise to fail the test setup clearly
             raise RuntimeError(f"Failed temp_db setup for {db_path}: {e}") from e
        finally:
            if db:
                logging.debug(f"Closing temp DB connection for test: {db_path}")
                try:
                    # Attempt to gracefully close the connection
                    db.close_connection()
                except Exception as close_err:
                    logging.warning(f"Error closing temp DB {db_path} during cleanup: {close_err}")
            # TemporaryDirectory context manager handles directory removal
            logging.debug(f"Temporary directory {temp_dir} will be removed.")

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


def create_test_media(db: MediaDatabase, title: str, content: str, content_hash: str = "test_hash"):
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
