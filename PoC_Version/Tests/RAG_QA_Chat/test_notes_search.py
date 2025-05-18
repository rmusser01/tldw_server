# test_notes_search.py
# pytest test file that tests the search_notes_titles function from the DB_Manager module.
#
# Imports
import os
import sys
import pytest
from unittest.mock import MagicMock
import sqlite3
from datetime import datetime
#
# Adjust the path to the parent directory of App_Function_Libraries
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(parent_dir)
#
# Local Imports
from App_Function_Libraries.DB.DB_Manager import search_notes_titles
#
####################################################################################################
#
# Test Functions

@pytest.fixture
def db_connection():
    """Create an in-memory SQLite database and populate it with test data."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    # Create the main notes table
    cursor.execute("""
    CREATE TABLE rag_qa_notes (
        id INTEGER PRIMARY KEY,
        title TEXT,
        content TEXT,
        timestamp TEXT,
        conversation_id TEXT
    )
    """)

    # Create the FTS (Full-Text Search) virtual table
    cursor.execute("""
    CREATE VIRTUAL TABLE rag_qa_notes_fts USING FTS5(title, content)
    """)

    # Sample test data
    notes = [
        (1, "First Note", "Content of the first note", datetime.now().isoformat(), "conv1"),
        (2, "Second Note", "Content of the second note", datetime.now().isoformat(), "conv2"),
        (3, "Another Note", "Content of another note", datetime.now().isoformat(), "conv3"),
        (4, "Note Four", "Fourth note content", datetime.now().isoformat(), "conv4"),
        (5, "Final Note", "This is the final note", datetime.now().isoformat(), "conv5")
    ]

    # Insert data into the main table
    cursor.executemany("""
    INSERT INTO rag_qa_notes (id, title, content, timestamp, conversation_id)
    VALUES (?, ?, ?, ?, ?)
    """, notes)

    # Insert data into the FTS table
    for note in notes:
        cursor.execute("""
        INSERT INTO rag_qa_notes_fts (rowid, title, content)
        VALUES (?, ?, ?)
        """, (note[0], note[1], note[2]))

    conn.commit()
    yield conn
    conn.close()


def test_search_notes_titles_with_search_term(db_connection):
    """Test searching with a non-empty search term."""
    search_term = "Note"
    results, total_pages, total_count = search_notes_titles(search_term, connection=db_connection)

    assert total_count == 5
    assert total_pages == 1
    assert len(results) == 5
    for result in results:
        assert "Note" in result[1]


def test_search_notes_titles_empty_search_term(db_connection):
    """Test searching with an empty search term, which should return all notes."""
    search_term = ""
    results, total_pages, total_count = search_notes_titles(search_term, connection=db_connection)

    assert total_count == 5
    assert total_pages == 1
    assert len(results) == 5


def test_search_notes_titles_pagination(db_connection):
    """Test pagination functionality."""
    search_term = ""
    results_per_page = 2

    # First page
    results, total_pages, total_count = search_notes_titles(
        search_term, page=1, results_per_page=results_per_page, connection=db_connection)
    assert total_count == 5
    assert total_pages == 3
    assert len(results) == 2

    # Second page
    results_page_2, _, _ = search_notes_titles(
        search_term, page=2, results_per_page=results_per_page, connection=db_connection)
    assert len(results_page_2) == 2

    # Third page
    results_page_3, _, _ = search_notes_titles(
        search_term, page=3, results_per_page=results_per_page, connection=db_connection)
    assert len(results_page_3) == 1


def test_search_notes_titles_invalid_page(db_connection):
    """Test that a ValueError is raised when an invalid page number is provided."""
    with pytest.raises(ValueError, match="Page number must be 1 or greater."):
        search_notes_titles("test", page=0, connection=db_connection)


def test_search_notes_titles_db_error():
    """Test that a database error is properly raised and handled."""
    # Create a mock connection
    mock_conn = MagicMock()
    mock_cursor = mock_conn.cursor.return_value
    # Set the side effect of the execute method to raise a database error
    mock_cursor.execute.side_effect = sqlite3.Error("Test database error")

    # Now call the function with the mock connection
    with pytest.raises(sqlite3.Error) as exc_info:
        search_notes_titles("test", connection=mock_conn)
    assert "Error searching notes: Test database error" in str(exc_info.value)
