# test_utils.py
# Description: Shared functions used across unit tests.
#
# Imports
import os
from contextlib import contextmanager
import tempfile
#
# 3rd-party Libraries
#
# Local Imports
from tldw_Server_API.app.core.DB_Management.SQLite_DB import Database, create_tables


#
########################################################################################################################
#
# Functions:


@contextmanager
def temp_db():
    # Create a temporary directory; it will be deleted automatically
    with tempfile.TemporaryDirectory() as temp_dir:
        # Build a database path inside the temporary directory.
        db_path = os.path.join(temp_dir, "test.db")
        db = Database(db_path)
        try:
            # Create tables using your current schema.
            create_tables(db)
            # Optionally, verify the schema.
            verify_media_db_schema(db)
            # Yield the fresh database instance.
            yield db
        finally:
            db.close_connection()

def verify_media_db_schema(db):
    """Ensure critical columns exist in Media table."""
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(Media)")
        columns = [col[1] for col in cursor.fetchall()]

        critical_columns = {'author', 'title', 'content'}
        missing = critical_columns - set(columns)

        if missing:
            raise RuntimeError(f"Media table missing columns: {missing}")

def create_test_media(db: Database, title: str, content: str):
    # Do NOT re-create the table; assume create_tables() has run
    # Now just insert:
    db.execute_query('''
        INSERT INTO Media (title, type, content, author)
        VALUES (?, ?, ?, ?)
    ''', (title, "document", content, "Test Author"))

    # Get the ID of the inserted row
    result = db.execute_query("SELECT last_insert_rowid()")
    return result[0][0]

#
# End of test_utils.py
########################################################################################################################