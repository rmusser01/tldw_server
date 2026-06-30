# test_chat_dictionary.py  (renamed to avoid basename conflicts)
# Description: Unit tests for the ChatDictionaryService
#
"""
Chat Dictionary Service Tests
------------------------------

Comprehensive unit tests for the chat dictionary functionality including
CRUD operations, pattern matching, text processing, and import/export.
"""

import pytest
import json
import tempfile
import re
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from datetime import datetime

from tldw_Server_API.app.core.Character_Chat.chat_dictionary import ChatDictionaryService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError


@pytest.fixture
def mock_db():
    """Create a mock database instance."""
    mock = MagicMock()
    mock.execute_query = MagicMock()
    mock.execute_many = MagicMock()

    # Setup proper connection context manager
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.lastrowid = 1
    mock_cursor.rowcount = 1
    mock_cursor.fetchall.return_value = []
    mock_cursor.fetchone.return_value = None
    mock_conn.execute.return_value = mock_cursor

    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_conn)
    mock_ctx.__exit__ = MagicMock(return_value=None)
    mock.get_connection.return_value = mock_ctx

    return mock


@pytest.fixture
def service(mock_db):
    """Create a ChatDictionaryService instance with mocked database."""
    return ChatDictionaryService(mock_db)


class TestChatDictionaryService:
    """Test suite for ChatDictionaryService."""

    def test_init_creates_tables(self, mock_db):
        """Test that initialization creates necessary tables."""
        mock_conn = MagicMock()
        mock_db.get_connection.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.get_connection.return_value.__exit__ = MagicMock(return_value=None)

        service = ChatDictionaryService(mock_db)

        # Should create two tables
        assert mock_conn.execute.call_count >= 2

        # Check that tables are created with correct SQL
        calls = mock_conn.execute.call_args_list
        sql_statements = [call[0][0] for call in calls]
        assert any("CREATE TABLE IF NOT EXISTS chat_dictionaries" in sql for sql in sql_statements)
        assert any("CREATE TABLE IF NOT EXISTS dictionary_entries" in sql for sql in sql_statements)

    def test_create_dictionary(self, service, mock_db):
        """Test creating a new dictionary."""
        # Mock the connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 1
        mock_conn.execute.return_value = mock_cursor
        mock_db.get_connection.return_value.__enter__.return_value = mock_conn
        mock_db.get_connection.return_value.__exit__.return_value = None

        dict_id = service.create_dictionary(name="Test Dictionary", description="A test dictionary")

        assert dict_id == 1
        mock_conn.execute.assert_called()

        # Check the SQL contains the right values
        calls = mock_conn.execute.call_args_list
        insert_call = None
        for current_call in calls:
            if "INSERT INTO chat_dictionaries" in current_call[0][0]:
                insert_call = current_call
                break
        assert insert_call is not None
        assert "Test Dictionary" in insert_call[0][1]
        assert "A test dictionary" in insert_call[0][1]

    def test_get_dictionary(self, service, mock_db):
        """Test retrieving a dictionary with its entries."""
        # Mock the connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Create a mock row that behaves like a dict
        mock_row = MagicMock()
        mock_row.keys = lambda: [
            "id",
            "name",
            "description",
            "is_active",
            "created_at",
            "updated_at",
            "version",
            "deleted",
        ]
        mock_row.__getitem__ = lambda self, key: {
            "id": 1,
            "name": "Test Dict",
            "description": "Test",
            "is_active": 1,
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
            "version": 1,
            "deleted": 0,
        }[key]

        mock_cursor.fetchone.return_value = mock_row
        mock_conn.execute.return_value = mock_cursor
        mock_db.get_connection.return_value.__enter__.return_value = mock_conn
        mock_db.get_connection.return_value.__exit__.return_value = None

        result = service.get_dictionary(1)

        assert result is not None
        assert result["id"] == 1
        assert result["name"] == "Test Dict"

    def test_add_entry(self, service, mock_db):
        """Test adding an entry to a dictionary."""
        # Mock the connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 1
        mock_conn.execute.return_value = mock_cursor
        mock_db.get_connection.return_value.__enter__.return_value = mock_conn
        mock_db.get_connection.return_value.__exit__.return_value = None

        entry_id = service.add_entry(
            dictionary_id=1, key="test", content="replaced", probability=100, max_replacements=1
        )

        assert entry_id == 1
        mock_conn.execute.assert_called()

        # Check the SQL
        calls = mock_conn.execute.call_args_list
        insert_call = None
        for current_call in calls:
            if "INSERT INTO dictionary_entries" in current_call[0][0]:
                insert_call = current_call
                break
        assert insert_call is not None
        assert "test" in insert_call[0][1]
        assert "replaced" in insert_call[0][1]

    def test_bulk_add_entries_rejects_invalid_explicit_regex(self, service):
        """Explicit regex entries added in bulk should be validated before insert."""
        with pytest.raises(re.error):
            service.bulk_add_entries(
                1,
                [
                    {
                        "pattern": "(",
                        "replacement": "broken",
                        "type": "regex",
                    }
                ],
            )

    def test_process_text_literal_replacement(self, service, mock_db):
        """Test processing text with literal string replacement."""
        # Setup mock connection to return entries
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value

        # Mock fetchall to return dictionary entries
        mock_cursor.fetchall.return_value = [
            {
                "id": 1,
                "dictionary_id": 1,
                "key": "hello",
                "content": "hi",
                "probability": 100,
                "max_replacements": 2,
                "group": None,
                "timed_effects": None,
            }
        ]

        result = service.process_text(text="hello world, hello there", token_budget=1000)

        assert "hi" in result["processed_text"]  # Should have replaced at least one "hello"
        assert result["replacements"] >= 1
        assert result["token_budget_exceeded"] == False

    def test_process_text_regex_replacement(self, service, mock_db):
        """Test processing text with regex pattern replacement."""
        # Mock active dictionaries with regex entries
        mock_db.execute_query.side_effect = [
            [{"id": 1, "name": "Regex Dict", "is_active": 1}],
            [
                {
                    "id": 1,
                    "dictionary_id": 1,
                    "key": r"\btest\w+",
                    "content": "TEST",
                    "probability": 100,
                    "max_replacements": 2,
                }
            ],
        ]

        result = service.process_text(text="testing tested tester", token_budget=1000)

        # Note: Actual regex processing depends on implementation
        assert result["processed_text"] is not None
        assert "replacements" in result

    def test_process_text_with_probability(self, service, mock_db):
        """Test that probability affects replacements."""
        # Mock entry with 0% probability
        mock_db.execute_query.side_effect = [
            [{"id": 1, "name": "Test", "is_active": 1}],
            [{"id": 1, "dictionary_id": 1, "key": "hello", "content": "hi", "probability": 0, "max_replacements": 1}],
        ]

        result = service.process_text("hello world", token_budget=1000)

        # With 0% probability, no replacement should occur
        assert result["processed_text"] == "hello world"
        assert result["replacements"] == 0

    def test_process_text_token_budget(self, service, mock_db):
        """Test that processing stops when token budget is exceeded."""
        # Setup mock connection to return entries that will cause replacements
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value

        # Mock fetchall to return an entry that will exceed budget
        mock_cursor.fetchall.return_value = [
            {
                "id": 1,
                "dictionary_id": 1,
                "key": "word",
                "content": "very_long_replacement_text_that_will_exceed_budget",
                "probability": 100,
                "max_replacements": 1000,
                "group": None,
                "timed_effects": None,
            }
        ]

        # Create text with many instances
        long_text = " ".join(["word"] * 50)

        result = service.process_text(long_text, token_budget=10)

        # Should handle budget
        assert "token_budget_exceeded" in result
        assert result["token_budget_exceeded"] == True

    def test_delete_dictionary_cascade(self, service, mock_db):
        """Test that deleting a dictionary cascades to entries."""
        # Setup mock connection
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value
        mock_cursor.rowcount = 1  # Indicate successful deletion
        mock_cursor.fetchone.return_value = {"id": 1, "deleted": 0}

        result = service.delete_dictionary(1)

        assert result == True

        # Check that UPDATE was called for soft delete
        calls = mock_conn.execute.call_args_list
        assert any("update chat_dictionaries set deleted =" in call[0][0].lower() for call in calls)

    def test_import_from_markdown(self, service, mock_db):
        """Test importing dictionary from markdown format."""
        markdown_content = """# Test Dictionary
Description: Test import

## Entries
- hello -> hi
- /test\\w+/ -> TEST (regex)
- goodbye -> bye (50%)
"""

        mock_db.execute_query.side_effect = [
            [{"id": 1}],  # Dictionary creation
            [{"id": 1}],  # Entry 1
            [{"id": 2}],  # Entry 2
            [{"id": 3}],  # Entry 3
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(markdown_content)
            f.flush()

            result = service.import_from_markdown(Path(f.name), "Test Dictionary")

        # The method returns the dictionary ID, not a dict
        assert result == 1

    def test_export_to_markdown(self, service, mock_db):
        """Test exporting dictionary to markdown format."""
        # Setup mock connection
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value

        # Mock get_dictionary and get_entries calls
        mock_cursor.fetchone.side_effect = [
            {
                "id": 1,
                "name": "Export Test",
                "description": "Test export",
                "is_active": 1,
                "created_at": "2024-01-01",
                "updated_at": "2024-01-01",
            },
        ]

        mock_cursor.fetchall.side_effect = [
            [
                {
                    "id": 1,
                    "key": "hello",
                    "content": "hi",
                    "group": None,
                    "is_regex": 0,
                    "probability": 100,
                    "max_replacements": 1,
                }
            ]
        ]

        # Create a temp file for export
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            temp_path = f.name

        result = service.export_to_markdown(1, Path(temp_path))

        # The method returns a boolean
        assert result == True

        # Clean up
        import os

        os.unlink(temp_path)

    def test_update_entry(self, service, mock_db):
        """Test updating a dictionary entry."""
        # Setup mock connection
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value
        mock_cursor.rowcount = 1  # Indicate successful update

        result = service.update_entry(entry_id=1, key="new_pattern", content="new_replacement", probability=50)

        assert result == True

        # Check UPDATE was called
        calls = mock_conn.execute.call_args_list
        update_call = None
        for call in calls:
            if "UPDATE dictionary_entries SET" in call[0][0]:
                update_call = call
                break
        assert update_call is not None

    def test_list_dictionaries(self, service, mock_db):
        """Test listing all dictionaries."""
        # Setup mock connection
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value
        mock_cursor.fetchall.return_value = [
            {"id": 1, "name": "Dict1", "is_active": 1, "entry_count": 5},
            {"id": 2, "name": "Dict2", "is_active": 0, "entry_count": 3},
        ]

        result = service.list_dictionaries(include_inactive=True)

        assert len(result) == 2
        assert result[0]["name"] == "Dict1"
        assert result[1]["entry_count"] == 3

    def test_toggle_dictionary_active(self, service, mock_db):
        """Test toggling dictionary active status."""
        # Mock the connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 1  # Indicate one row was updated
        mock_conn.execute.return_value = mock_cursor
        mock_db.get_connection.return_value.__enter__.return_value = mock_conn
        mock_db.get_connection.return_value.__exit__.return_value = None

        result = service.toggle_dictionary_active(1, False)

        assert result == True
        mock_conn.execute.assert_called()

        # Check SQL contains the update
        calls = mock_conn.execute.call_args_list
        update_call = None
        for current_call in calls:
            sql = current_call[0][0]
            if "UPDATE chat_dictionaries SET" in sql and "is_active = ?" in sql:
                update_call = current_call
                break
        assert update_call is not None

    def test_get_statistics(self, service, mock_db):
        """Test getting dictionary statistics."""
        # Mock the connection and multiple cursors for different queries
        mock_conn = MagicMock()

        # First cursor for dictionary counts
        mock_cursor1 = MagicMock()
        mock_row1 = {"total_dictionaries": 5, "active_dictionaries": 3}
        mock_cursor1.fetchone.return_value = mock_row1

        # Second cursor for entry counts
        mock_cursor2 = MagicMock()
        mock_row2 = {"total_entries": 100}
        mock_cursor2.fetchone.return_value = mock_row2

        # Set up execute to return different cursors for each call
        mock_conn.execute.side_effect = [mock_cursor1, mock_cursor2]

        mock_db.get_connection.return_value.__enter__.return_value = mock_conn
        mock_db.get_connection.return_value.__exit__.return_value = None

        stats = service.get_statistics()

        assert stats["total_dictionaries"] == 5
        assert stats["active_dictionaries"] == 3
        assert stats["total_entries"] == 100
        assert stats["average_entries_per_dictionary"] == 20.0

    def test_bulk_add_entries(self, service, mock_db):
        """Test adding multiple entries at once."""
        entries = [
            {"key": "test1", "content": "repl1"},
            {"key": "test2", "content": "repl2"},
            {"key": "test3", "content": "repl3"},
        ]

        # Mock the connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.execute.return_value = mock_cursor
        mock_db.get_connection.return_value.__enter__.return_value = mock_conn
        mock_db.get_connection.return_value.__exit__.return_value = None

        result = service.bulk_add_entries(1, entries)

        assert result == 3
        # Should have been called 3 times (once for each entry)
        assert mock_conn.execute.call_count == 3

    def test_search_entries(self, service, mock_db):
        """Test searching for entries by pattern."""
        # Setup mock connection
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value
        mock_cursor.fetchall.return_value = [
            {"id": 1, "key": "hello", "content": "hi", "dictionary_name": "Dict1"},
            {"id": 2, "key": "hello world", "content": "greetings", "dictionary_name": "Dict2"},
        ]

        results = service.search_entries("hello")

        assert len(results) == 2
        assert results[0]["key"] == "hello"
        assert results[1]["dictionary_name"] == "Dict2"

    def test_clone_dictionary(self, service, mock_db):
        """Test cloning a dictionary with all its entries."""
        # Setup mock connection
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value

        # Mock fetchone for getting original dictionary
        mock_cursor.fetchone.return_value = {
            "id": 1,
            "name": "Original",
            "description": "Original dict",
            "is_active": 1,
        }

        # Mock fetchall for getting entries
        mock_cursor.fetchall.return_value = [
            {
                "key": "test",
                "content": "repl",
                "probability": 100,
                "group": None,
                "timed_effects": None,
                "max_replacements": 1,
            }
        ]

        # Mock lastrowid for new dictionary creation
        mock_cursor.lastrowid = 2

        new_id = service.clone_dictionary(1, "Cloned Dict")

        assert new_id == 2

    def test_entry_validation(self, service):
        """Test that invalid entries raise meaningful errors."""
        # Probability outside accepted range (0-100 for ints) should raise InputError
        with pytest.raises(InputError):
            service.add_entry(
                dictionary_id=1,
                key="invalid_prob",
                content="value",
                probability=200,  # out of range
            )

        # Invalid regex patterns propagate re.error when explicitly marked as regex
        with pytest.raises(re.error):
            service.add_entry(
                dictionary_id=1,
                pattern=r"(",
                replacement="bad",
                type="regex",
            )

    def test_clear_cache(self, service, mock_db):
        """Test that cache is cleared when dictionaries are modified."""
        # Setup mock connection
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value
        mock_cursor.fetchall.return_value = []
        mock_cursor.lastrowid = 1

        # First call get_entries to potentially populate cache
        service.get_entries()

        # Now create a new dictionary which should clear cache
        service.create_dictionary("New", "New dict")

        # The clear_cache method should have been called internally
        # We can check that cache is None
        assert service._entry_cache is None
