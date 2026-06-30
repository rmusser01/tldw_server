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

import pytest

#
# Local Imports
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.backends.factory import reset_managed_sqlite_backends


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
            try:
                reset_managed_sqlite_backends(sqlite_targets=[str(db_path)], mode="hard")
            except Exception as reset_err:
                logging.warning(f"Error resetting temp DB backend {db_path} during cleanup: {reset_err}")
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
    media_uuid = str(uuid.uuid4())
    last_modified = db._get_current_utc_timestamp_str()
    client_id = db.client_id or "test_client"
    cursor = db.execute_query(
        "INSERT INTO Media (title, type, content, author, content_hash, uuid, last_modified, client_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (title, "document", content, "Test Author", content_hash, media_uuid, last_modified, client_id),
        commit=True,  # Commit this specific insert
    )
    media_id = getattr(cursor, "lastrowid", None)
    if media_id:
        return media_id
    result = db.execute_query(
        "SELECT id FROM Media WHERE uuid = ?",
        (media_uuid,),
    ).fetchone()
    if result:
        return result["id"] if isinstance(result, dict) else result[0]
    raise RuntimeError("Failed to retrieve media id after creating test media.")


def skip_if_transcription_model_unavailable(model_name: str) -> None:
    """Skip test if the specified transcription model is not available."""
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files import (
        check_transcription_model_status,
    )

    status = check_transcription_model_status(model_name)
    if not status.get("available") and not status.get("usable"):
        pytest.skip(status.get("message", "Transcription model not available"))


def skip_if_whisper_model_not_cached_locally(model_name: str) -> None:
    """Skip smoke tests unless the resolved Whisper model is already cached locally."""
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files import (
        check_transcription_model_status,
    )

    status = check_transcription_model_status(model_name)
    resolved_model = status.get("model") or model_name
    if not status.get("available"):
        pytest.skip(
            status.get(
                "message",
                f"Whisper model {resolved_model} is not cached locally; skipping smoke test",
            )
        )


# End of test_utils.py
########################################################################################################################
