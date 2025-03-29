# test_utils.py
# Description: Shared functions used across unit tests.
#
# Imports
from contextlib import contextmanager
import tempfile
#
# 3rd-party Libraries
#
# Local Imports
from tldw_Server_API.app.core.DB_Management.SQLite_DB import Database
#
########################################################################################################################
#
# Functions:

@contextmanager
def temp_db():
    """Context manager for temporary database"""
    db_path = tempfile.mktemp()
    db = Database(db_path)
    try:
        yield db
    finally:
        db.close_connection()
        import os
        if os.path.exists(db_path):
            os.unlink(db_path)

# In test_utils.py
def create_test_media(db: Database, title: str, content: str, media_type: str = "document"):
    db.execute_query('''
        INSERT INTO Media (title, type, content)
        VALUES (?, ?, ?)
    ''', (title, media_type, content))
    media_id = db.execute_query("SELECT last_insert_rowid()")[0][0]
    return media_id

#
# End of test_utils.py
########################################################################################################################