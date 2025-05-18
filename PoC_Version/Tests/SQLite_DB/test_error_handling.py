# tests/test_error_handling.py
from unittest.mock import patch, MagicMock

import pytest
import os
import sqlite3
from PoC_Version.App_Function_Libraries.DB.SQLite_DB import Database, DatabaseError, InputError, add_keyword, delete_keyword, \
    add_media_with_keywords, search_media_db


@pytest.fixture
def test_db(tmp_path):
    """Create a test database file"""
    db_path = tmp_path / "test.db"
    db = Database(str(db_path))

    # Initialize database with required tables
    with patch('App_Function_Libraries.DB.SQLite_DB.Database.table_exists', return_value=False):
        with db.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test (
                    id INTEGER PRIMARY KEY
                )
            """)

    yield db

    db.close_connection()
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except PermissionError:
            print(f"Warning: Unable to delete {db_path}. It may still be in use.")


@pytest.fixture
def mock_db():
    """Create a mock database for search tests"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch('App_Function_Libraries.DB.DB_Manager.db.get_connection', return_value=mock_conn):
        yield mock_conn, mock_cursor


def test_execute_query_with_invalid_sql(test_db):
    with pytest.raises(sqlite3.Error):
        test_db.execute_query("INSERT INTO nonexistent_table (column) VALUES (?)", (1,))


def test_execute_many_with_invalid_data(test_db):
    with pytest.raises(sqlite3.Error):
        test_db.execute_many("INSERT INTO nonexistent_table (column) VALUES (?)", [(1,), (2,)])


def test_table_exists_nonexistent(test_db):
    assert not test_db.table_exists("nonexistent_table")


def test_transaction_rollback(test_db):
    with test_db.get_connection() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY)")

    with pytest.raises(sqlite3.OperationalError):
        with test_db.transaction() as conn:
            conn.execute("INSERT INTO test (id) VALUES (?)", (1,))
            conn.execute("INSERT INTO nonexistent_table (id) VALUES (?)", (1,))

    result = test_db.execute_query("SELECT COUNT(*) FROM test")
    assert result[0][0] == 0


def test_search_media_db_with_invalid_page(mock_db):
    """Test search_media_db with invalid page number"""
    with pytest.raises(ValueError, match="Page number must be 1 or greater."):
        search_media_db("query", ['title'], "", page=0)


def test_search_media_db_with_none_values(mock_db):
    """Test search_media_db with None values"""
    mock_conn, mock_cursor = mock_db
    mock_cursor.fetchall.return_value = []

    result = search_media_db(None, ['title'], "")
    assert isinstance(result, list)


def test_search_media_db_with_empty_search_fields(mock_db):
    """Test search_media_db with empty search fields list"""
    mock_conn, mock_cursor = mock_db
    mock_cursor.fetchall.return_value = []

    result = search_media_db("query", [], "")
    assert isinstance(result, list)


def test_search_media_db_with_invalid_fields(mock_db):
    """Test search_media_db with invalid search fields"""
    mock_conn, mock_cursor = mock_db
    mock_cursor.execute.side_effect = sqlite3.OperationalError

    with pytest.raises(sqlite3.OperationalError):
        search_media_db("query", ['nonexistent_field'], "")


def test_add_keyword_with_invalid_data():
    with pytest.raises(AttributeError):
        add_keyword(None)


def test_delete_nonexistent_keyword():
    result = delete_keyword("nonexistent_keyword")
    assert "not found" in result


def test_add_media_with_invalid_data():
    with pytest.raises(InputError):
        add_media_with_keywords(None, None, None, None, None, None, None, None, None, None)

