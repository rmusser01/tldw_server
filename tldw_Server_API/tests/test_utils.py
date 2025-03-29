# test_utils.py
# Description: Shared functions used across unit tests.
#
# Imports
import tempfile
from contextlib import contextmanager
#
# 3rd-party Libraries
#
# Local Imports
from your_app.db import Database
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

def create_test_media(db, title="Test Media", content="Test content"):
    """Helper to create test media"""
    db.execute_query('''
        INSERT INTO Media (title, type, content)
        VALUES (?, ?, ?)
    ''', (title, "document", content))
    return db.execute_query("SELECT last_insert_rowid()")[0][0]

#
# End of test_utils.py
########################################################################################################################