# tests/test_sqlite_db.py
import os
import sqlite3
import time
import threading # Keep for threading test
import pytest
import tempfile # Use tempfile for the test DB path

# Adjust import path
try:
    from app.db.database_setup import Database, DatabaseError
    # Remove create_tables import if it's not used externally
    # from app.db.database_setup import create_tables
    # Keep Utils import if needed for path logic, but Database class handles path now
    # from app.utils import Utils # Example path
except ImportError as e:
    print(f"Error importing Database class for core tests: {e}")
    raise


# Use the db_instance fixture from conftest.py for isolation per test
# Remove the module-scoped db fixture and reset_db fixture

def test_database_connection(db_instance: Database):
    # Connection is implicitly tested by the fixture setup
    # We can add an explicit check
    conn = db_instance.get_connection()
    assert conn is not None
    # Execute a simple query to ensure connection is live
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    assert cursor.fetchone()[0] == 1

def test_execute_query_select(db_instance: Database):
    # Assumes Media table exists from __init__
    # Insert some data first
    media_id = None
    with db_instance.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO Media (url, title, type, content, content_hash) VALUES (?,?,?,?,?)",
                       ('test_select', 'Select Test', 'doc', 'content', 'hash_select'))
        media_id = cursor.lastrowid

    # Now execute select query
    results = db_instance.execute_query("SELECT title FROM Media WHERE id = ?", (media_id,))
    # execute_query now returns cursor by default - fetch from it
    result_row = results.fetchone()
    assert result_row is not None
    assert result_row['title'] == 'Select Test' # Access by column name

def test_execute_query_insert_commit(db_instance: Database):
     # Use execute_query with commit=True
     cursor = db_instance.execute_query(
         "INSERT INTO Media (url, title, type, content, content_hash) VALUES (?,?,?,?,?)",
         ('test_commit', 'Commit Test', 'doc', 'content', 'hash_commit'),
         commit=True # Test the commit flag
     )
     assert cursor is not None # Should return cursor

     # Verify data is present in a new connection/transaction
     results = db_instance.execute_query("SELECT title FROM Media WHERE url = ?", ('test_commit',))
     assert results.fetchone()['title'] == 'Commit Test'


def test_database_error_invalid_sql(db_instance: Database):
    # Test invalid SQL causing OperationalError, wrapped in DatabaseError
    with pytest.raises(DatabaseError) as exc_info:
        db_instance.execute_query("SELECT * FROM non_existent_table")
    # Check the underlying cause if needed
    assert isinstance(exc_info.value.__cause__, sqlite3.OperationalError)
    assert "no such table" in str(exc_info.value.__cause__)

def test_transaction_commit(db_instance: Database):
    media_id = None
    with db_instance.transaction() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO Media (url, title, type, content, content_hash) VALUES (?,?,?,?,?)",
                       ('test_trans_commit', 'Trans Commit', 'doc', 'content', 'hash_trans_commit'))
        media_id = cursor.lastrowid
    # Transaction automatically commits on successful exit

    # Verify outside transaction
    results = db_instance.execute_query("SELECT title FROM Media WHERE id = ?", (media_id,))
    assert results.fetchone()['title'] == 'Trans Commit'

def test_transaction_rollback_on_exception(db_instance: Database):
    media_id_should_not_exist = None
    try:
        with db_instance.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO Media (url, title, type, content, content_hash) VALUES (?,?,?,?,?)",
                           ('test_trans_rollback', 'Trans Rollback', 'doc', 'content', 'hash_trans_rollback'))
            media_id_should_not_exist = cursor.lastrowid
            raise ValueError("Simulated error to trigger rollback") # Simulate error
    except ValueError as e:
        # Catch the simulated error
        assert "Simulated error" in str(e)
    except Exception as e:
        pytest.fail(f"Caught unexpected exception: {e}")


    # Verify the insert was rolled back
    results = db_instance.execute_query("SELECT COUNT(*) FROM Media WHERE url = ?", ('test_trans_rollback',))
    assert results.fetchone()[0] == 0
    # Also check if the ID was potentially reused (shouldn't happen with rollback)
    if media_id_should_not_exist:
         results_id = db_instance.execute_query("SELECT COUNT(*) FROM Media WHERE id = ?", (media_id_should_not_exist,))
         assert results_id.fetchone()[0] == 0


# execute_many is not part of the provided Database class, skip or add if needed

def test_table_exists(db_instance: Database):
    # Tables should be created by __init__
    assert db_instance.table_exists('Media') == True
    assert db_instance.table_exists('Keywords') == True
    assert db_instance.table_exists('DocumentVersions') == True
    assert db_instance.table_exists('non_existent_table') == False

def test_close_connection(db_instance: Database):
    conn = db_instance.get_connection()
    assert conn is not None
    db_instance.close_connection()
    # Check internal state if possible, or try using connection again
    assert not hasattr(db_instance._local, 'conn') or db_instance._local.conn is None
    # Trying to get connection again should create a new one
    conn2 = db_instance.get_connection()
    assert conn2 is not None
    assert conn2 != conn # Should be a new connection object

# Remove test_create_tables if it's just testing internal schema setup
# def test_create_tables(db): ...

def test_multiple_connections_threading(db_instance: Database):
    # Test that multiple threads can get and use connections
    errors = []
    def worker():
        try:
            # Each thread gets its own connection via thread-local storage
            conn = db_instance.get_connection()
            cursor = conn.cursor()
            # Perform a simple read/write unique to the thread
            thread_id = threading.get_ident()
            url = f"thread_test_{thread_id}"
            cursor.execute("INSERT INTO Media (url, title, type, content, content_hash) VALUES (?,?,?,?,?)",
                           (url, f"Thread {thread_id}", 'doc', 'c', f'h_{thread_id}'))
            conn.commit() # Commit within the thread's transaction
            cursor.execute("SELECT title FROM Media WHERE url = ?", (url,))
            assert cursor.fetchone()['title'] == f"Thread {thread_id}"
            # print(f"Thread {thread_id} finished successfully.") # Optional debug
        except Exception as e:
            errors.append(e)
            print(f"Error in thread {threading.get_ident()}: {e}")
        # finally:
            # Closing connection here might interfere if same thread is reused quickly?
            # Let the main fixture handle cleanup.
            # db_instance.close_connection() # Maybe don't close here

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors occurred in threads: {errors}"

# Renamed test_connection_closure to avoid duplication
# def test_connection_closure(db): ... # Covered by test_close_connection