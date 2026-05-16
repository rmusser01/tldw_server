# test_world_book_manager.py  (renamed to avoid basename conflicts)
# Description: Unit tests for the WorldBookService
#
"""
World Book Manager Service Tests
---------------------------------

Comprehensive unit tests for the world book functionality including
CRUD operations, keyword matching, character attachments, and context processing.
"""

import pytest
import json
from unittest.mock import MagicMock, patch, call
from datetime import datetime

from tldw_Server_API.app.core.Character_Chat.world_book_manager import (
    WorldBookService,
    WorldBookEntry,
    BoundedDict,
)
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
    """Create a WorldBookService instance with mocked database."""
    return WorldBookService(mock_db)


class TestWorldBookService:
    """Test suite for WorldBookService."""

    def test_init_creates_tables(self, mock_db):
        """Test that initialization creates necessary tables."""
        mock_conn = mock_db.get_connection().__enter__()
        service = WorldBookService(mock_db)

        # Should create at least the table and index statements
        assert mock_conn.execute.call_count >= 7

        executed_sql = [args[0] for args, _ in mock_conn.execute.call_args_list]

        assert any("CREATE TABLE IF NOT EXISTS world_books" in sql for sql in executed_sql)
        assert any("CREATE TABLE IF NOT EXISTS world_book_entries" in sql for sql in executed_sql)
        assert any("CREATE TABLE IF NOT EXISTS character_world_books" in sql for sql in executed_sql)

        index_statements = [sql for sql in executed_sql if "CREATE INDEX" in sql]
        assert len(index_statements) >= 4

    def test_create_world_book(self, service, mock_db):
        """Test creating a new world book."""
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value
        mock_cursor.lastrowid = 1

        wb_id = service.create_world_book(
            name="Test World",
            description="A test world book",
            scan_depth=3,
            token_budget=500,
            recursive_scanning=True,
            enabled=True,
        )

        assert wb_id == 1
        mock_conn.execute.assert_called()

        # Check SQL parameters
        call_args = mock_conn.execute.call_args[0]
        assert "INSERT INTO world_books" in call_args[0]
        assert "Test World" in call_args[1]
        assert 500 in call_args[1]  # token_budget

    def test_get_world_book_with_entries(self, service, mock_db):
        """Test retrieving a world book and its entries."""
        # Mock world book data
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value
        mock_cursor.fetchone.return_value = {
            "id": 1,
            "name": "Fantasy World",
            "description": "Fantasy setting",
            "scan_depth": 3,
            "token_budget": 1000,
            "recursive_scanning": 1,
            "enabled": 1,
            "created_at": "2024-01-01",
            "updated_at": "2024-01-01",
        }

        # Get world book
        wb_result = service.get_world_book(1)

        # Mock entries data for get_entries call
        mock_cursor.fetchall.return_value = [
            {
                "id": 1,
                "world_book_id": 1,
                "keywords": '["dragon","castle"]',
                "content": "Dragons live in castles",
                "priority": 100,
                "enabled": 1,
                "metadata": "{}",
            }
        ]

        # Get entries separately
        entries = service.get_entries(world_book_id=1)

        assert wb_result["id"] == 1
        assert wb_result["name"] == "Fantasy World"
        assert len(entries) == 1
        assert entries[0].keywords == ["dragon", "castle"]

    def test_get_entries_cached_results_keep_hybrid_shape(self, service):
        """Cached entry reads should preserve both dict-style and attr-style access."""
        service._entry_cache = {
            1: [
                WorldBookEntry(
                    entry_id=1,
                    world_book_id=1,
                    keywords=["dragon"],
                    content="Dragon lore",
                    priority=10,
                    enabled=True,
                )
            ]
        }

        entries = service.get_entries(world_book_id=1)

        assert len(entries) == 1
        assert entries[0]["keywords"] == ["dragon"]
        assert entries[0].keywords == ["dragon"]
        assert entries[0].get("content") == "Dragon lore"

    def test_get_entries_applies_server_side_filters_from_cache(self, service):
        """Entry filters should be applied server-side for cached reads."""
        service._entry_cache = {
            1: [
                WorldBookEntry(
                    entry_id=1,
                    world_book_id=1,
                    keywords=["hero"],
                    content="Hero lore",
                    priority=100,
                    enabled=True,
                    metadata={"group": "Characters", "appendable": True, "recursive_scanning": True},
                ),
                WorldBookEntry(
                    entry_id=2,
                    world_book_id=1,
                    keywords=["city"],
                    content="City lore",
                    priority=90,
                    enabled=True,
                    metadata={"group": "Locations", "appendable": False, "recursive_scanning": False},
                ),
            ]
        }

        entries = service.get_entries(
            world_book_id=1,
            enabled_only=False,
            group="characters",
            appendable=True,
            recursive_scanning=True,
        )

        assert len(entries) == 1
        assert entries[0]["id"] == 1

    def test_get_entries_cache_filters_coerce_string_boolean_metadata(self, service):
        """Cache-path metadata filters should treat string booleans like native booleans."""
        service._entry_cache = {
            1: [
                WorldBookEntry(
                    entry_id=1,
                    world_book_id=1,
                    keywords=["hero"],
                    content="Hero lore",
                    priority=100,
                    enabled=True,
                    metadata={"group": "Characters", "appendable": "false", "recursive_scanning": "true"},
                ),
            ]
        }

        entries = service.get_entries(
            world_book_id=1,
            enabled_only=False,
            group="characters",
            appendable=False,
            recursive_scanning=True,
        )

        assert len(entries) == 1
        assert entries[0]["appendable"] is False
        assert entries[0]["recursive_scanning"] is True

    def test_get_entries_cache_respects_disabled_book_when_enabled_only(self, service):
        """Cache path should respect disabled world-book state for enabled_only reads."""
        service._entry_cache = {
            1: [
                WorldBookEntry(
                    entry_id=1,
                    world_book_id=1,
                    keywords=["hero"],
                    content="Hero lore",
                    priority=10,
                    enabled=True,
                )
            ]
        }
        service.get_world_book = MagicMock(return_value={"id": 1, "enabled": False})

        entries = service.get_entries(world_book_id=1, enabled_only=True)

        assert entries == []

    def test_get_entries_sql_includes_metadata_filters_for_sqlite(self, service, mock_db):
        """SQLite query path should push metadata filters into SQL."""
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value
        mock_cursor.fetchall.return_value = []

        _ = service.get_entries(
            world_book_id=1,
            enabled_only=False,
            group="Characters",
            appendable=True,
            recursive_scanning=False,
        )

        call_args = mock_conn.execute.call_args[0]
        sql_text = call_args[0]
        params = call_args[1]
        assert "json_extract" in sql_text
        assert "$.group" in sql_text
        assert "$.appendable" in sql_text
        assert "$.recursive_scanning" in sql_text
        assert "characters" in params

    def test_get_entries_does_not_cache_partial_reads(self, service, mock_db):
        """Enabled-only and metadata-filtered reads should not populate per-book cache."""
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value
        mock_cursor.fetchall.return_value = []

        _ = service.get_entries(world_book_id=1, enabled_only=True)
        assert 1 not in service._entry_cache

        _ = service.get_entries(world_book_id=1, group="Characters")
        assert 1 not in service._entry_cache

    def test_get_entry_fetches_single_entry_by_id(self, service, mock_db):
        """Single-entry lookup should query by id and return hybrid entry view."""
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value
        mock_cursor.fetchone.return_value = {
            "id": 55,
            "world_book_id": 9,
            "keywords": '["dragon"]',
            "content": "Dragon lore",
            "priority": 5,
            "enabled": 1,
            "case_sensitive": 0,
            "regex_match": 0,
            "whole_word_match": 1,
            "metadata": '{"recursive_scanning": true}',
            "created_at": "2026-02-27 00:00:00",
            "last_modified": "2026-02-27 00:00:01",
            "book_enabled": 1,
        }

        entry = service.get_entry(55)

        assert entry is not None
        assert entry.id == 55
        assert entry["keywords"] == ["dragon"]
        assert entry["recursive_scanning"] is True
        assert entry["created_at"] == "2026-02-27 00:00:00"
        assert entry["last_modified"] == "2026-02-27 00:00:01"

        call_args = mock_conn.execute.call_args[0]
        assert "WHERE e.id = ? AND wb.deleted = ?" in call_args[0]
        assert call_args[1] == (55, False)

    def test_invalidate_cache_reinitializes_bounded_dicts(self, service):
        """Cache invalidation should preserve bounded-cache semantics."""
        service._entry_cache = {}
        service._book_cache = {}

        service._invalidate_cache()

        assert isinstance(service._entry_cache, BoundedDict)
        assert isinstance(service._book_cache, BoundedDict)

    def test_add_entry(self, service, mock_db):
        """Test adding an entry to a world book."""
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value
        mock_cursor.lastrowid = 1

        entry_id = service.add_entry(
            world_book_id=1, keywords=["magic", "wizard"], content="Wizards use magic", priority=100, enabled=True
        )

        assert entry_id == 1

        # Check that keywords are stored as JSON
        call_args = mock_conn.execute.call_args[0]
        assert "INSERT INTO world_book_entries" in call_args[0]
        assert '["magic", "wizard"]' in call_args[1]

    def test_attach_to_character(self, service, mock_db):
        """Test attaching a world book to a character."""
        mock_conn = mock_db.get_connection().__enter__()

        # Clear previous calls from init
        mock_conn.execute.reset_mock()

        result = service.attach_to_character(character_id=1, world_book_id=1, enabled=True, priority=100)

        assert result == True

        # Check insert SQL
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        sql_text = call_args[0].lower()
        assert "insert into character_world_books" in sql_text
        assert "on conflict" in sql_text
        assert call_args[1] == (1, 1, True, 100)

    def test_detach_from_character(self, service, mock_db):
        """Test detaching a world book from a character."""
        mock_conn = mock_db.get_connection().__enter__()

        result = service.detach_from_character(character_id=1, world_book_id=1)

        assert result == True

        # Check delete SQL
        call_args = mock_conn.execute.call_args[0]
        assert "DELETE FROM character_world_books" in call_args[0]
        assert call_args[1] == (1, 1)

    def test_get_character_world_books(self, service, mock_db):
        """Test getting all world books for a character."""
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value
        mock_cursor.fetchall.return_value = [
            {"id": 1, "name": "Main World", "is_primary": 1, "enabled": 1},
            {"id": 2, "name": "Secondary World", "is_primary": 0, "enabled": 1},
        ]

        result = service.get_character_world_books(1)

        assert len(result) == 2
        assert result[0]["name"] == "Main World"
        assert result[0]["is_primary"] == True
        assert result[1]["is_primary"] == False

    def test_process_context_keyword_matching(self, service, mock_db):
        """Test context processing with keyword matching."""
        # Mock active world books and entries
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value

        # Mock get_world_book call
        mock_cursor.fetchone.return_value = {"id": 1, "name": "Test World", "token_budget": 500, "enabled": 1}

        # Mock get_entries call
        mock_cursor.fetchall.return_value = [
            {
                "id": 1,
                "world_book_id": 1,
                "keywords": '["sword","blade"]',
                "content": "A legendary sword",
                "priority": 100,
                "enabled": 1,
                "metadata": "{}",
            },
            {
                "id": 2,
                "world_book_id": 1,
                "keywords": '["magic"]',
                "content": "Magic is powerful",
                "priority": 50,
                "enabled": 1,
                "metadata": "{}",
            },
        ]

        result = service.process_context(
            text="The hero found a magic sword",  # Use "magic" not "magical" for exact match
            world_book_ids=[1],  # Specify which world book to use
            token_budget=1000,
        )

        assert result["processed_context"]
        assert "entries_matched" in result
        assert result["entries_matched"] > 0
        # Should match both "sword" and "magic"
        assert "legendary sword" in result["processed_context"].lower()
        assert "magic is powerful" in result["processed_context"].lower()

    def test_process_context_priority_ordering(self, service, mock_db):
        """Test that entries are processed by priority."""
        # Create entries with different priorities
        entries = [
            WorldBookEntry(
                entry_id=1, world_book_id=1, keywords=["test"], content="Low priority", priority=10, enabled=True
            ),
            WorldBookEntry(
                entry_id=2, world_book_id=1, keywords=["test"], content="High priority", priority=100, enabled=True
            ),
        ]

        # Mock to return entries
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value
        mock_cursor.fetchone.return_value = {"id": 1, "name": "Test", "token_budget": 500, "enabled": 1}
        mock_cursor.fetchall.return_value = []

        # Set entries directly for testing
        service._entry_cache = {1: entries}

        result = service.process_context(text="This is a test", world_book_ids=[1], token_budget=1000)

        # High priority should be processed first
        assert "High priority" in result["processed_context"]

    def test_process_context_diagnostics_include_static_or_pinned_hint(self, service):
        """Diagnostics should expose cache-safe static/pinned classification without raw metadata."""
        service.get_world_book = MagicMock(
            return_value={"id": 1, "name": "Test", "token_budget": 500, "enabled": 1}
        )
        service._entry_cache = {
            1: [
                WorldBookEntry(
                    entry_id=1,
                    world_book_id=1,
                    keywords=["clock"],
                    content="Clock tower lore",
                    priority=100,
                    enabled=True,
                    metadata={"pinned": True},
                )
            ]
        }

        result = service.process_context(
            text="clock",
            world_book_ids=[1],
            token_budget=1000,
            include_diagnostics=True,
        )

        assert result["diagnostics"][0]["static_or_pinned"] is True

    def test_process_context_keyword_lookup_skips_non_candidate_literal_entries(self, service):
        """Normalized keyword lookup should avoid scanning unrelated literal entries."""
        service.get_world_book = MagicMock(
            return_value={"id": 1, "name": "Test", "token_budget": 500, "enabled": 1}
        )

        non_candidate_entry = WorldBookEntry(
            entry_id=1,
            world_book_id=1,
            keywords=["dragon"],
            content="Dragon lore",
            priority=90,
            enabled=True,
            case_sensitive=False,
            regex_match=False,
            whole_word_match=True,
        )
        non_candidate_entry.get_first_match_info = MagicMock(
            side_effect=AssertionError("Unrelated literal entry should not be scanned")
        )

        matching_entry = WorldBookEntry(
            entry_id=2,
            world_book_id=1,
            keywords=["hero"],
            content="Hero lore",
            priority=100,
            enabled=True,
            case_sensitive=False,
            regex_match=False,
            whole_word_match=True,
        )

        service._entry_cache = {1: [non_candidate_entry, matching_entry]}

        result = service.process_context(text="A hero arrives", world_book_ids=[1], token_budget=1000)

        assert "Hero lore" in result["processed_context"]

    def test_process_context_keyword_lookup_falls_back_for_regex_entries(self, service):
        """Regex entries should still be scanned when quick lookup cannot index them."""
        service.get_world_book = MagicMock(
            return_value={"id": 1, "name": "Test", "token_budget": 500, "enabled": 1}
        )

        regex_entry = WorldBookEntry(
            entry_id=1,
            world_book_id=1,
            keywords=[r"hero.*"],
            content="Regex hero lore",
            priority=100,
            enabled=True,
            regex_match=True,
            case_sensitive=False,
            whole_word_match=True,
        )
        service._entry_cache = {1: [regex_entry]}

        result = service.process_context(text="heroic legend", world_book_ids=[1], token_budget=1000)

        assert "Regex hero lore" in result["processed_context"]

    def test_process_context_token_budget(self, service, mock_db):
        """Test that token budget is respected."""
        # Create a very long entry
        long_content = " ".join(["word"] * 1000)

        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value
        mock_cursor.fetchall.side_effect = [
            [{"id": 1, "name": "Test", "token_budget": 10}],  # Very small budget
            [
                {
                    "id": 1,
                    "keywords": '["test"]',
                    "content": long_content,
                    "priority": 100,
                    "enabled": 1,
                    "metadata": "{}",
                }
            ],
        ]

        result = service.process_context(text="This is a test", character_id=None, token_budget=10)

        # Should truncate to fit budget
        assert result.get("token_budget_exceeded", False) or result["tokens_used"] <= 10
        assert len(result["processed_context"].split()) < 1000

    def test_recursive_scanning(self, service, mock_db):
        """Test recursive keyword scanning."""
        # First entry triggers second entry
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value

        # Mock get_world_book
        mock_cursor.fetchone.return_value = {
            "id": 1,
            "name": "Test",
            "token_budget": 500,
            "recursive_scanning": 1,
            "enabled": 1,
        }

        # Mock get_entries
        mock_cursor.fetchall.return_value = [
            {
                "id": 1,
                "keywords": '["hero"]',
                "content": "The hero has a sword",
                "priority": 100,
                "enabled": 1,
                "metadata": "{}",
            },
            {
                "id": 2,
                "keywords": '["sword"]',
                "content": "The sword is magical",
                "priority": 50,
                "enabled": 1,
                "metadata": "{}",
            },
        ]

        result = service.process_context(
            text="Story about a hero", world_book_ids=[1], token_budget=1000, recursive_scanning=True
        )

        # Should match "hero" first, then "sword" from the hero entry
        assert result["entries_matched"] >= 1

    def test_recursive_scanning_inherits_book_setting_when_not_overridden(self, service):
        """Book-level recursive_scanning should apply when call-level override is omitted."""
        service.get_world_book = MagicMock(
            return_value={
                "id": 1,
                "name": "Test",
                "token_budget": 500,
                "recursive_scanning": 1,
                "enabled": 1,
            }
        )
        service._entry_cache = {
            1: [
                WorldBookEntry(
                    entry_id=1,
                    world_book_id=1,
                    keywords=["hero"],
                    content="The hero has a sword",
                    priority=100,
                    enabled=True,
                ),
                WorldBookEntry(
                    entry_id=2,
                    world_book_id=1,
                    keywords=["sword"],
                    content="The sword is magical",
                    priority=50,
                    enabled=True,
                ),
            ]
        }

        result = service.process_context(text="Story about a hero", world_book_ids=[1], token_budget=1000)

        assert result["entries_matched"] >= 2
        assert "The sword is magical" in result["processed_context"]

    def test_recursive_scanning_explicit_false_overrides_book_setting(self, service):
        """Call-level recursive_scanning=False should disable recursive expansion."""
        service.get_world_book = MagicMock(
            return_value={
                "id": 1,
                "name": "Test",
                "token_budget": 500,
                "recursive_scanning": 1,
                "enabled": 1,
            }
        )
        service._entry_cache = {
            1: [
                WorldBookEntry(
                    entry_id=1,
                    world_book_id=1,
                    keywords=["hero"],
                    content="The hero has a sword",
                    priority=100,
                    enabled=True,
                ),
                WorldBookEntry(
                    entry_id=2,
                    world_book_id=1,
                    keywords=["sword"],
                    content="The sword is magical",
                    priority=50,
                    enabled=True,
                ),
            ]
        }

        result = service.process_context(
            text="Story about a hero",
            world_book_ids=[1],
            token_budget=1000,
            recursive_scanning=False,
        )

        assert result["entries_matched"] == 1
        assert "The sword is magical" not in result["processed_context"]

    def test_recursive_scanning_respects_entry_recursive_source_flags(self, service):
        """Only entries flagged for recursive scanning should chain additional depth."""
        service.get_world_book = MagicMock(
            return_value={
                "id": 1,
                "name": "Test",
                "token_budget": 500,
                "recursive_scanning": 1,
                "enabled": 1,
            }
        )
        service._entry_cache = {
            1: [
                WorldBookEntry(
                    entry_id=1,
                    world_book_id=1,
                    keywords=["hero"],
                    content="The hero has a sword",
                    priority=100,
                    enabled=True,
                    metadata={"recursive_scanning": True},
                ),
                WorldBookEntry(
                    entry_id=2,
                    world_book_id=1,
                    keywords=["sword"],
                    content="The sword was forged by ancient smiths",
                    priority=50,
                    enabled=True,
                ),
                WorldBookEntry(
                    entry_id=3,
                    world_book_id=1,
                    keywords=["smiths"],
                    content="Ancient smiths lived in the mountain",
                    priority=25,
                    enabled=True,
                ),
            ]
        }

        result = service.process_context(
            text="Story about a hero",
            world_book_ids=[1],
            token_budget=1000,
            recursive_depth=3,
        )

        assert "The sword was forged by ancient smiths" in result["processed_context"]
        assert "Ancient smiths lived in the mountain" not in result["processed_context"]

    def test_recursive_scanning_falls_back_to_legacy_behavior_without_recursive_flags(self, service):
        """When no entry has recursive_scanning metadata, legacy chaining remains enabled."""
        service.get_world_book = MagicMock(
            return_value={
                "id": 1,
                "name": "Test",
                "token_budget": 500,
                "recursive_scanning": 1,
                "enabled": 1,
            }
        )
        service._entry_cache = {
            1: [
                WorldBookEntry(
                    entry_id=1,
                    world_book_id=1,
                    keywords=["hero"],
                    content="The hero has a sword",
                    priority=100,
                    enabled=True,
                ),
                WorldBookEntry(
                    entry_id=2,
                    world_book_id=1,
                    keywords=["sword"],
                    content="The sword is magical",
                    priority=50,
                    enabled=True,
                ),
            ]
        }

        result = service.process_context(
            text="Story about a hero",
            world_book_ids=[1],
            token_budget=1000,
            recursive_depth=1,
        )

        assert result["entries_matched"] >= 2
        assert "The sword is magical" in result["processed_context"]

    def test_import_world_book(self, service, mock_db):
        """Test importing a world book from JSON."""
        world_book_data = {
            "world_book": {
                "name": "Imported World",
                "description": "Imported from JSON",
                "scan_depth": 5,
                "token_budget": 800,
            },
            "entries": [
                {"keywords": ["test1"], "content": "Content 1", "priority": 100},
                {"keywords": ["test2"], "content": "Content 2", "priority": 50},
            ],
        }

        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value
        mock_cursor.lastrowid = 1

        result = service.import_world_book(world_book_data)

        assert result == True
        # Should have created world book and entries
        assert mock_conn.execute.call_count >= 3

    def test_export_world_book(self, service, mock_db):
        """Test exporting a world book to JSON."""
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value
        mock_cursor.fetchone.return_value = {
            "id": 1,
            "name": "Export Test",
            "description": "Test export",
            "scan_depth": 3,
            "token_budget": 500,
            "recursive_scanning": 0,
            "enabled": 1,
        }
        mock_cursor.fetchall.return_value = [
            {"keywords": '["test"]', "content": "Test content", "priority": 100, "enabled": 1, "metadata": "{}"}
        ]

        result = service.export_world_book(1)

        assert result["world_book"]["name"] == "Export Test"
        assert len(result["entries"]) == 1
        # entries are WorldBookEntry objects with to_dict method

    def test_delete_world_book_cascade(self, service, mock_db):
        """Test that deleting a world book cascades properly."""
        mock_conn = mock_db.get_connection().__enter__()

        # Clear calls from __init__
        mock_conn.execute.reset_mock()

        result = service.delete_world_book(1)

        assert result == True

        # Should soft delete (UPDATE to set deleted flag)
        calls = mock_conn.execute.call_args_list
        assert len(calls) == 1
        assert "update world_books set deleted =" in calls[0][0][0].lower()

    def test_update_entry(self, service, mock_db):
        """Test updating a world book entry."""
        mock_conn = mock_db.get_connection().__enter__()

        result = service.update_entry(entry_id=1, keywords=["new", "keywords"], content="New content", priority=75)

        assert result == True

        # Check SQL
        call_args = mock_conn.execute.call_args[0]
        assert "UPDATE world_book_entries SET" in call_args[0]
        assert '["new", "keywords"]' in call_args[1]

    def test_update_entry_validates_keywords_when_existing_entry_is_regex(
        self, service, mock_db
    ):
        """Updating keywords should still validate regex patterns when regex mode is already enabled."""
        mock_conn = mock_db.get_connection().__enter__()
        update_cursor = mock_conn.execute.return_value
        update_cursor.rowcount = 1

        select_cursor = MagicMock()
        select_cursor.fetchone.return_value = {"regex_match": 1}

        def execute_side_effect(sql, params=()):
            if "SELECT regex_match FROM world_book_entries" in sql:
                return select_cursor
            return update_cursor

        mock_conn.execute.side_effect = execute_side_effect

        with pytest.raises(InputError):
            service.update_entry(entry_id=1, keywords=["[invalid"])

    def test_get_statistics(self, service, mock_db):
        """Test getting world book statistics."""
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value
        mock_cursor.fetchone.side_effect = [
            {"total_world_books": 10, "enabled_world_books": 8},
            {"total_entries": 150},
            {"total_attachments": 25},
            {"avg_entries": 15.0},
        ]

        stats = service.get_statistics()

        assert stats["total_world_books"] == 10
        assert stats["enabled_world_books"] == 8
        assert stats["total_entries"] == 150
        assert stats["total_character_attachments"] == 25
        assert stats["average_entries_per_world_book"] == 15.0

    def test_get_entry_counts_for_world_books(self, service, mock_db):
        """Entry counts should be returned in a single mapping with missing IDs defaulted to 0."""
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value
        mock_cursor.fetchall.return_value = [
            {"world_book_id": 1, "entry_count": 4},
            {"world_book_id": 3, "entry_count": 1},
        ]

        counts = service.get_entry_counts_for_world_books([1, 2, 3])

        assert counts[1] == 4
        assert counts[2] == 0
        assert counts[3] == 1

    def test_search_entries(self, service, mock_db):
        """Test searching for entries by keyword or content."""
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value
        mock_cursor.fetchall.return_value = [
            {"id": 1, "keywords": "dragon,fire", "content": "Dragons breathe fire", "world_book_name": "Fantasy"},
            {"id": 2, "keywords": "dragon", "content": "Ancient dragon lore", "world_book_name": "Mythology"},
        ]

        results = service.search_entries("dragon")

        assert len(results) == 2
        assert results[0]["world_book_name"] == "Fantasy"
        assert "fire" in results[0]["keywords"]

    def test_bulk_operations(self, service, mock_db):
        """Test bulk enable/disable of entries."""
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value
        mock_cursor.rowcount = 3

        # Bulk disable entries
        result = service.bulk_update_entries(world_book_id=1, entry_ids=[1, 2, 3], enabled=False)

        assert result == 3

        # Check SQL - the actual query includes updated_at
        call_args = mock_conn.execute.call_args[0]
        assert "UPDATE world_book_entries" in call_args[0]
        assert "SET enabled = ?" in call_args[0]

    def test_clone_world_book(self, service, mock_db):
        """Test cloning a world book with all entries."""
        mock_conn = mock_db.get_connection().__enter__()
        mock_cursor = mock_conn.execute.return_value
        mock_cursor.fetchone.side_effect = [
            {"id": 1, "name": "Original", "description": "Original world", "scan_depth": 3, "token_budget": 500}
        ]
        mock_cursor.fetchall.return_value = [
            {"keywords": '["test"]', "content": "content", "priority": 100, "metadata": "{}"}
        ]
        mock_cursor.lastrowid = 2  # New world book ID

        new_id = service.clone_world_book(1, "Cloned World")

        assert new_id == 2
        # Should have executed many for bulk insert
        assert mock_conn.executemany.called or mock_conn.execute.call_count > 2

    def test_keyword_normalization(self, service):
        """Test that keywords are matched case-insensitively by default."""
        entry = WorldBookEntry(
            entry_id=1,
            world_book_id=1,
            keywords=["Dragon", "CASTLE", "magic"],
            content="Test",
            priority=100,
            enabled=True,
            case_sensitive=False,  # Case insensitive by default
        )

        # Keywords should match case-insensitively
        assert entry.matches("the dragon lives in a castle")
        assert entry.matches("MAGIC spell")
        assert entry.matches("Dragon's lair")
