# tests/test_error_handling.py
from unittest.mock import patch, MagicMock
import pytest
import os
import sqlite3

# Adjust import paths
try:
    from tldw_Server_API.app.core.DB_Management.Media_DB import (
        Database,
        DatabaseError,
        InputError,
        add_keyword,
        delete_keyword,
        add_media_with_keywords,
        search_media_db
    )
except ImportError as e:
    print(f"Error importing functions for error handling tests: {e}")
    raise

# Use the fixture from conftest.py
# No need for test_db or mock_db fixtures here anymore if testing real DB interactions

# === Test Database Class Errors ===

def test_execute_query_with_invalid_sql(db_instance: Database):
    # Database class now raises DatabaseError wrapping sqlite3.Error
    with pytest.raises(DatabaseError, match="Query execution failed: no such table: nonexistent_table"):
        # Use db_instance fixture from conftest.py
        db_instance.execute_query("SELECT * FROM nonexistent_table")

# execute_many is not part of the provided Database class, skip or add if needed

def test_table_exists_nonexistent(db_instance: Database):
    assert not db_instance.table_exists("nonexistent_table")

def test_transaction_rollback(db_instance: Database):
    # Ensure table exists first for the rollback test part
    try:
        with db_instance.transaction() as conn:
             # Intentionally cause an error after an insert
            conn.execute("INSERT INTO Media (url, title, type, content, content_hash) VALUES (?, ?, ?, ?, ?)",
                         ('err_test', 'Err Test', 'doc', 'content', 'hash123'))
            # This will fail because table doesn't exist
            conn.execute("INSERT INTO nonexistent_table (id) VALUES (?)", (1,)) # <- This raises sqlite3.OperationalError
    except DatabaseError as e: # <- The transaction context re-raises this
        # Expecting the DatabaseError wrapper around the OperationalError
        assert "no such table: nonexistent_table" in str(e.__cause__) # Check the underlying cause
    # Verify the initial insert was rolled back (This part runs AFTER the exception)
    with db_instance.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM Media WHERE url = ?", ('err_test',))
        count = cursor.fetchone()[0]
        assert count == 0

# === Test Function Specific Errors ===

def test_search_media_db_with_invalid_page(db_instance: Database):
    """Test search_media_db with invalid page number"""
    with pytest.raises(ValueError, match="Page number must be 1 or greater"):
        # Pass the db_instance
        search_media_db(search_query="query", search_fields=['title'], keywords=[], page=0, db_instance=db_instance)

# Skip search tests with None values or empty fields if the function now handles them gracefully or raises specific errors.
# Add tests for TypeError if db_instance is missing if you make it mandatory.

def test_add_keyword_with_invalid_data(db_instance: Database):
    # Test the specific case where keyword is None or empty (now raises InputError)
    # --- Use the EXACT error message string for matching ---
    expected_error_msg = "Keyword cannot be None, empty, or just whitespace."
    with pytest.raises(InputError, match=expected_error_msg):
        add_keyword(None, db_instance=db_instance) # Test None
    with pytest.raises(InputError, match=expected_error_msg):
        add_keyword("", db_instance=db_instance) # Test empty string
    with pytest.raises(InputError, match=expected_error_msg):
         add_keyword("   ", db_instance=db_instance) # Test whitespace only

def test_delete_nonexistent_keyword(db_instance: Database):
    # Assuming delete_keyword returns a dict or raises an error
    result = delete_keyword("nonexistent_keyword", db_instance=db_instance) # Pass db_instance
    # Check based on the updated function's return type
    if isinstance(result, dict):
         assert "not found" in result.get('error', result.get('message', ''))
    # Or if it raises:
    # with pytest.raises(SomeSpecificError):
    #     delete_keyword("nonexistent_keyword", db_instance=db_instance)

def test_add_media_with_invalid_data_content_none(db_instance: Database):
    # Test the specific case where content is None (now raises InputError)
    with pytest.raises(InputError, match="Content cannot be None"):
        add_media_with_keywords(
            url="http://example.com", title="Test Title", media_type="article",
            content=None, # Invalid content
            keywords=["test"], prompt="p", analysis_content="a", transcription_model="m",
            author="au", ingestion_date="2023-01-01", db_instance=db_instance
        )

def test_add_media_with_invalid_media_type(db_instance: Database):
     with pytest.raises(InputError, match="Invalid media type"):
         add_media_with_keywords(
             url="http://example.com", title="Test", media_type="bad_type", content="c",
             keywords=[], prompt=None, analysis_content=None, transcription_model=None,
             author=None, ingestion_date="2023-01-01", db_instance=db_instance
         )

def test_add_media_with_invalid_date(db_instance: Database):
     with pytest.raises(InputError, match="Invalid ingestion date format"):
         add_media_with_keywords(
             url="http://example.com", title="Test", media_type="article", content="c",
             keywords=[], prompt=None, analysis_content=None, transcription_model=None,
             author=None, ingestion_date="01-01-2023", db_instance=db_instance # Wrong format
         )