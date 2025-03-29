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
    """Context manager for a temporary DB using the standard schema."""
    db_path = tempfile.mktemp()
    db = Database(db_path)

    try:
        create_tables(db)
        verify_media_db_schema(db)
        yield db
    finally:
        db.close_connection()
        if os.path.exists(db_path):
            os.unlink(db_path)

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