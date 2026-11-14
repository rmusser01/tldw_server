"""
Unit tests for WorldBookService.

Tests the world book functionality for context management.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import json

from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError

# ========================================================================
# World Book Management Tests
# ========================================================================


class TestWorldBookManagement:
    """Test world book CRUD operations."""

    @pytest.mark.unit
    def test_create_world_book(self, world_book_service, sample_world_book):
        """Test creating a world book."""
        service = world_book_service

        wb_id = service.create_world_book(
            name=sample_world_book['name'],
            description=sample_world_book['description']
        )

        assert wb_id is not None
        assert wb_id > 0

    @pytest.mark.unit
    def test_get_world_book(self, world_book_service, sample_world_book):
        """Test getting a world book."""
        service = world_book_service

        wb_id = service.create_world_book(
            name=sample_world_book['name'],
            description=sample_world_book['description']
        )

        world_book = service.get_world_book(wb_id)

        assert world_book is not None
        assert world_book['name'] == sample_world_book['name']
        assert world_book['description'] == sample_world_book['description']

    @pytest.mark.unit
    def test_list_world_books(self, world_book_service):
        """Test listing all world books."""
        service = world_book_service

        # Create multiple world books
        for i in range(3):
            service.create_world_book(
                name=f"World Book {i}",
                description=f"Description {i}"
            )

        world_books = service.list_world_books()

        assert len(world_books) >= 3
        assert all('name' in wb for wb in world_books)

    @pytest.mark.unit
    def test_update_world_book(self, world_book_service):
        """Test updating a world book."""
        service = world_book_service

        wb_id = service.create_world_book(
            name="Original",
            description="Original description"
        )

        service.update_world_book(
            world_book_id=wb_id,
            name="Updated",
            description="Updated description"
        )

        updated = service.get_world_book(wb_id)
        assert updated['name'] == "Updated"
        assert updated['description'] == "Updated description"

    @pytest.mark.unit
    def test_update_world_book_with_expected_version(self, world_book_service):
        """Test optimistic-lock update with expected version."""
        service = world_book_service
        wb_id = service.create_world_book(name="Versioned", description="v1")
        created = service.get_world_book(wb_id)
        expected_version = created["version"]

        service.update_world_book(
            world_book_id=wb_id,
            description="v2",
            expected_version=expected_version,
        )

        updated = service.get_world_book(wb_id)
        assert updated["description"] == "v2"
        assert updated["version"] == expected_version + 1

    @pytest.mark.unit
    def test_update_world_book_rejects_stale_expected_version(self, world_book_service):
        """Test stale expected-version update is rejected with conflict."""
        service = world_book_service
        wb_id = service.create_world_book(name="Version conflict", description="v1")
        created = service.get_world_book(wb_id)
        expected_version = created["version"]

        service.update_world_book(
            world_book_id=wb_id,
            description="v2",
            expected_version=expected_version,
        )

        with pytest.raises(ConflictError, match="Version mismatch"):
            service.update_world_book(
                world_book_id=wb_id,
                description="v3",
                expected_version=expected_version,
            )

    @pytest.mark.unit
    def test_delete_world_book(self, world_book_service):
        """Test deleting a world book."""
        service = world_book_service

        wb_id = service.create_world_book(name="To Delete")

        service.delete_world_book(wb_id)

        # Should not find deleted world book
        deleted = service.get_world_book(wb_id)
        assert deleted is None or deleted.get('is_deleted', False)

    @pytest.mark.unit
    def test_delete_cascade(self, world_book_service, sample_world_book):
        """Test cascade deletion of entries."""
        service = world_book_service

        wb_id = service.create_world_book(name="Cascade Test")

        # Add entries
        for entry in sample_world_book['entries']:
            service.add_entry(wb_id, **entry)

        # Delete world book
        service.delete_world_book(wb_id, cascade=True)

        # Entries should also be deleted
        entries = service.get_entries(wb_id)
        assert len(entries) == 0

    @pytest.mark.unit
    def test_invalidate_cache_reinitializes_bounded_caches(self, world_book_service):
        """Cache invalidation keeps bounded cache structures."""
        service = world_book_service
        service._entry_cache = {}
        service._book_cache = {}

        service._invalidate_cache()

        assert hasattr(service._entry_cache, "_max_size")
        assert hasattr(service._book_cache, "_max_size")


# ========================================================================
# Entry Management Tests
# ========================================================================


class TestEntryManagement:
    """Test world book entry operations."""

    @pytest.mark.unit
    def test_add_entry(self, world_book_service):
        """Test adding an entry."""
        service = world_book_service

        wb_id = service.create_world_book(name="Entry Test")

        entry_id = service.add_entry(
            world_book_id=wb_id,
            keywords=['dragon', 'dragons'],
            content='Dragons are magical creatures.',
            priority=100
        )

        assert entry_id is not None
        entries = service.get_entries(wb_id)
        assert len(entries) == 1
        assert 'dragon' in entries[0]['keywords']

    @pytest.mark.unit
    def test_get_entry_returns_single_entry_by_id(self, world_book_service):
        """Single-entry lookup should return the expected entry without scanning callers."""
        service = world_book_service

        wb_id = service.create_world_book(name="Single Entry Lookup")
        entry_id = service.add_entry(
            world_book_id=wb_id,
            keywords=["hero"],
            content="Hero lore",
            priority=25,
        )

        entry = service.get_entry(entry_id)

        assert entry is not None
        assert int(entry["id"]) == int(entry_id)
        assert entry["keywords"] == ["hero"]
        assert entry["content"] == "Hero lore"

    @pytest.mark.unit
    def test_entry_group_roundtrip_and_backfill_behavior(self, world_book_service):
        """Entries expose nullable group while preserving metadata group values."""
        service = world_book_service
        wb_id = service.create_world_book(name="Entry Group Test")

        grouped_entry_id = service.add_entry(
            world_book_id=wb_id,
            keywords=["city"],
            content="City lore",
            metadata={"group": "Locations"}
        )
        service.add_entry(
            world_book_id=wb_id,
            keywords=["general"],
            content="Ungrouped lore"
        )

        entries_by_id = {int(entry["id"]): entry for entry in service.get_entries(wb_id)}

        grouped_entry = entries_by_id[grouped_entry_id]
        assert grouped_entry["group"] == "Locations"
        assert grouped_entry["metadata"]["group"] == "Locations"

        ungrouped_entries = [
            entry for entry_id, entry in entries_by_id.items() if entry_id != grouped_entry_id
        ]
        assert len(ungrouped_entries) == 1
        assert ungrouped_entries[0]["group"] is None

    @pytest.mark.unit
    def test_get_entries_supports_group_appendable_recursive_filters(self, world_book_service):
        """Server-side entry filters should narrow results by group/appendable/recursive flags."""
        service = world_book_service
        wb_id = service.create_world_book(name="Entry Filter Test")

        service.add_entry(
            world_book_id=wb_id,
            keywords=["hero"],
            content="Hero lore",
            metadata={"group": "Characters", "appendable": True, "recursive_scanning": True},
        )
        service.add_entry(
            world_book_id=wb_id,
            keywords=["city"],
            content="City lore",
            metadata={"group": "Locations", "appendable": False, "recursive_scanning": False},
        )

        filtered = service.get_entries(
            world_book_id=wb_id,
            group="characters",
            appendable=True,
            recursive_scanning=True,
        )

        assert len(filtered) == 1
        assert filtered[0]["group"] == "Characters"
        assert filtered[0]["appendable"] is True
        assert filtered[0]["recursive_scanning"] is True

    @pytest.mark.unit
    def test_get_entries_coerces_string_filter_booleans(self, world_book_service):
        """Entry filters should accept string boolean compatibility values."""
        service = world_book_service
        wb_id = service.create_world_book(name="Entry String Filter Args")

        service.add_entry(
            world_book_id=wb_id,
            keywords=["hero"],
            content="Hero lore",
            metadata={"group": "Characters", "appendable": False, "recursive_scanning": True},
        )

        filtered = service.get_entries(
            world_book_id=wb_id,
            group="characters",
            appendable="false",
            recursive_scanning="true",
        )

        assert len(filtered) == 1
        assert filtered[0]["appendable"] is False
        assert filtered[0]["recursive_scanning"] is True

    @pytest.mark.unit
    def test_get_entries_cache_filters_coerce_string_boolean_metadata(self, world_book_service):
        """Cache-path filtering should coerce string booleans in metadata."""
        service = world_book_service
        wb_id = service.create_world_book(name="Entry String Bool Filter Test")

        service.add_entry(
            world_book_id=wb_id,
            keywords=["hero"],
            content="Hero lore",
            metadata={"group": "Characters", "appendable": "false", "recursive_scanning": "true"},
        )

        # Warm the full per-book cache first so filter read uses cache path.
        _ = service.get_entries(world_book_id=wb_id, enabled_only=False)

        filtered = service.get_entries(
            world_book_id=wb_id,
            group="characters",
            appendable=False,
            recursive_scanning=True,
        )

        assert len(filtered) == 1
        assert filtered[0]["appendable"] is False
        assert filtered[0]["recursive_scanning"] is True

    @pytest.mark.unit
    def test_get_entries_db_filters_coerce_string_boolean_metadata(self, world_book_service):
        """Cold-cache DB filtering should treat string booleans as native booleans."""
        service = world_book_service
        wb_id = service.create_world_book(name="Entry String Bool DB Filter Test")

        service.add_entry(
            world_book_id=wb_id,
            keywords=["hero"],
            content="Hero lore",
            metadata={"group": "Characters", "appendable": "false", "recursive_scanning": "true"},
        )

        # No cache priming: force DB metadata filter path.
        filtered = service.get_entries(
            world_book_id=wb_id,
            group="characters",
            appendable=False,
            recursive_scanning=True,
        )

        assert len(filtered) == 1
        assert filtered[0]["appendable"] is False
        assert filtered[0]["recursive_scanning"] is True

    @pytest.mark.unit
    def test_get_entries_enabled_only_respects_disabled_world_book_with_cache(self, world_book_service):
        """Enabled-only cache reads should return no entries when the world book is disabled."""
        service = world_book_service
        wb_id = service.create_world_book(name="Cache Disabled Book", enabled=False)
        service.add_entry(
            world_book_id=wb_id,
            keywords=["hero"],
            content="Hero lore",
            enabled=True,
        )

        # Prime cache using non-enabled-only read.
        all_entries = service.get_entries(world_book_id=wb_id, enabled_only=False)
        assert len(all_entries) == 1

        enabled_only_entries = service.get_entries(world_book_id=wb_id, enabled_only=True)
        assert enabled_only_entries == []

    @pytest.mark.unit
    def test_get_entries_does_not_cache_enabled_only_or_filtered_reads(self, world_book_service):
        """Partial reads should not populate the canonical per-book cache."""
        service = world_book_service
        wb_id = service.create_world_book(name="Partial Read Cache Control", enabled=True)
        service.add_entry(
            world_book_id=wb_id,
            keywords=["hero"],
            content="Hero lore",
            enabled=True,
            metadata={"group": "Characters"},
        )

        # Partial read: enabled_only
        enabled_entries = service.get_entries(world_book_id=wb_id, enabled_only=True)
        assert len(enabled_entries) == 1
        assert wb_id not in service._entry_cache

        # Partial read: filtered
        filtered_entries = service.get_entries(world_book_id=wb_id, group="Characters")
        assert len(filtered_entries) == 1
        assert wb_id not in service._entry_cache

        # Full read should populate cache.
        full_entries = service.get_entries(world_book_id=wb_id, enabled_only=False)
        assert len(full_entries) == 1
        assert wb_id in service._entry_cache

    @pytest.mark.unit
    def test_add_recursive_entry(self, world_book_service):
        """Test adding recursive scanning entry."""
        service = world_book_service

        wb_id = service.create_world_book(name="Recursive Test")

        entry_id = service.add_entry(
            world_book_id=wb_id,
            keywords=['hero'],
            content='The hero wields a sword.',
            priority=100,
            recursive_scanning=True
        )

        entries = service.get_entries(wb_id)
        assert entries[0]['recursive_scanning'] is True

    @pytest.mark.unit
    def test_update_entry(self, world_book_service):
        """Test updating an entry."""
        service = world_book_service

        wb_id = service.create_world_book(name="Update Test")
        entry_id = service.add_entry(
            world_book_id=wb_id,
            keywords=['old'],
            content='Old content'
        )

        service.update_entry(
            entry_id=entry_id,
            keywords=['new', 'updated'],
            content='New content',
            priority=90
        )

        entries = service.get_entries(wb_id)
        assert 'new' in entries[0]['keywords']
        assert entries[0]['content'] == 'New content'
        assert entries[0]['priority'] == 90

    @pytest.mark.unit
    def test_delete_entry(self, world_book_service):
        """Test deleting an entry."""
        service = world_book_service

        wb_id = service.create_world_book(name="Delete Test")
        entry_id = service.add_entry(
            world_book_id=wb_id,
            keywords=['test'],
            content='Test content'
        )

        service.delete_entry(entry_id)

        entries = service.get_entries(wb_id)
        assert len(entries) == 0

    @pytest.mark.unit
    def test_toggle_entry_enabled(self, world_book_service):
        """Test toggling entry enabled status."""
        service = world_book_service

        wb_id = service.create_world_book(name="Toggle Test")
        entry_id = service.add_entry(
            world_book_id=wb_id,
            keywords=['test'],
            content='Test'
        )

        # Initially enabled
        entries = service.get_entries(wb_id)
        initial_status = entries[0].get('enabled', True)

        # Toggle
        service.toggle_entry_enabled(entry_id)

        # Check toggled
        entries = service.get_entries(wb_id)
        assert entries[0].get('enabled', True) != initial_status

    @pytest.mark.unit
    def test_bulk_operations(self, world_book_service, sample_world_book):
        """Test bulk entry operations."""
        service = world_book_service

        wb_id = service.create_world_book(name="Bulk Test")

        # Bulk add
        results = service.bulk_add_entries(
            world_book_id=wb_id,
            entries=sample_world_book['entries']
        )

        assert results['added'] == len(sample_world_book['entries'])

        entries = service.get_entries(wb_id)
        assert len(entries) == len(sample_world_book['entries'])


# ========================================================================
# Character Association Tests
# ========================================================================


class TestCharacterAssociation:
    """Test world book to character associations."""

    @pytest.mark.unit
    def test_attach_to_character(self, world_book_service):
        """Test attaching world book to character."""
        service = world_book_service

        wb_id = service.create_world_book(name="Character World")
        character_id = 1  # Mock character ID

        result = service.attach_to_character(wb_id, character_id)

        assert result['success'] is True

        # Check association
        character_books = service.get_character_world_books(character_id)
        assert any(wb['id'] == wb_id for wb in character_books)

    @pytest.mark.unit
    def test_detach_from_character(self, world_book_service):
        """Test detaching world book from character."""
        service = world_book_service

        wb_id = service.create_world_book(name="Detach Test")
        character_id = 1

        # Attach first
        service.attach_to_character(wb_id, character_id)

        # Then detach
        result = service.detach_from_character(wb_id, character_id)

        assert result['success'] is True

        # Check no longer associated
        character_books = service.get_character_world_books(character_id)
        assert not any(wb['id'] == wb_id for wb in character_books)

    @pytest.mark.unit
    def test_get_character_world_books(self, world_book_service):
        """Test getting all world books for a character."""
        service = world_book_service

        character_id = 1

        # Create and attach multiple world books
        wb_ids = []
        for i in range(3):
            wb_id = service.create_world_book(name=f"Character Book {i}")
            service.attach_to_character(wb_id, character_id)
            wb_ids.append(wb_id)

        character_books = service.get_character_world_books(character_id)

        assert len(character_books) >= 3
        for wb_id in wb_ids:
            assert any(wb['id'] == wb_id for wb in character_books)


# ========================================================================
# Context Processing Tests
# ========================================================================


class TestContextProcessing:
    """Test context processing with world book entries."""

    @pytest.mark.unit
    def test_process_context_keyword_matching(self, world_book_service, sample_world_book):
        """Test keyword matching in context."""
        service = world_book_service

        wb_id = service.create_world_book(name="Context Test")
        for entry in sample_world_book['entries']:
            service.add_entry(wb_id, **entry)

        context = "Tell me about dragons and magic in the kingdom."

        activated = service.process_context(context, wb_id)

        # Should activate all three entries based on keywords
        assert len(activated) == 3
        assert any('Dragons are ancient' in e['content'] for e in activated)
        assert any('Magic flows' in e['content'] for e in activated)
        assert any('kingdom spans' in e['content'] for e in activated)

    @pytest.mark.unit
    def test_process_context_priority_ordering(self, world_book_service):
        """Test entries ordered by priority."""
        service = world_book_service

        wb_id = service.create_world_book(name="Priority Test")

        # Add entries with different priorities
        service.add_entry(wb_id, ['test'], 'Low priority', priority=50)
        service.add_entry(wb_id, ['test'], 'High priority', priority=100)
        service.add_entry(wb_id, ['test'], 'Medium priority', priority=75)

        context = "This is a test."
        activated = service.process_context(context, wb_id)

        # Should be ordered by priority (highest first)
        assert activated[0]['content'] == 'High priority'
        assert activated[1]['content'] == 'Medium priority'
        assert activated[2]['content'] == 'Low priority'

    @pytest.mark.unit
    def test_process_context_disabled_entries(self, world_book_service):
        """Test disabled entries are not activated."""
        service = world_book_service

        wb_id = service.create_world_book(name="Disabled Test")

        entry1 = service.add_entry(wb_id, ['active'], 'Active entry')
        entry2 = service.add_entry(wb_id, ['disabled'], 'Disabled entry')

        # Disable second entry
        service.toggle_entry_enabled(entry2)

        context = "Test active and disabled keywords."
        activated = service.process_context(context, wb_id)

        # Only active entry should be included
        assert len(activated) == 1
        assert activated[0]['content'] == 'Active entry'

    @pytest.mark.unit
    def test_recursive_scanning(self, world_book_service, complex_world_book):
        """Test recursive scanning of entries."""
        service = world_book_service

        wb_id = service.create_world_book(name="Recursive Test")
        for entry in complex_world_book['entries']:
            service.add_entry(wb_id, **entry)

        # Initial context only mentions 'hero'
        context = "The hero arrives."

        activated = service.process_context(
            context,
            wb_id,
            recursive_depth=2
        )

        # Should recursively activate: hero -> sword -> smiths
        assert len(activated) >= 2
        assert any('hero' in str(e['keywords']) for e in activated)
        assert any('sword' in e['content'] for e in activated)
        # With depth=2, might also get smiths entry

    @pytest.mark.unit
    def test_recursive_scanning_respects_per_entry_recursive_flags(self, world_book_service):
        """Recursive expansion should only chain through entries flagged as recursive sources."""
        service = world_book_service

        wb_id = service.create_world_book(name="Recursive Source Control", recursive_scanning=True)
        service.add_entry(
            wb_id,
            ["hero"],
            "The hero has a sword.",
            priority=100,
            recursive_scanning=True,
        )
        service.add_entry(
            wb_id,
            ["sword"],
            "The sword was forged by ancient smiths.",
            priority=90,
        )
        service.add_entry(
            wb_id,
            ["smiths"],
            "Ancient smiths lived in mountain forges.",
            priority=80,
        )

        activated = service.process_context("The hero arrives.", wb_id, recursive_depth=3)
        contents = [entry["content"] for entry in activated]

        assert any("sword was forged" in content for content in contents)
        assert not any("Ancient smiths lived" in content for content in contents)

    @pytest.mark.unit
    def test_recursive_scanning_preserves_legacy_behavior_without_entry_flags(self, world_book_service):
        """When no entry has recursive metadata, recursive chaining should remain enabled."""
        service = world_book_service

        wb_id = service.create_world_book(name="Recursive Legacy", recursive_scanning=True)
        service.add_entry(wb_id, ["hero"], "The hero has a sword.", priority=100)
        service.add_entry(wb_id, ["sword"], "The sword is magical.", priority=90)

        activated = service.process_context("The hero arrives.", wb_id, recursive_depth=1)
        contents = [entry["content"] for entry in activated]

        assert any("The sword is magical." == content for content in contents)

    @pytest.mark.unit
    def test_keyword_lookup_fallback_keeps_regex_matching(self, world_book_service):
        """Regex entries should still match when quick keyword lookup cannot index them."""
        service = world_book_service

        wb_id = service.create_world_book(name="Regex Lookup Fallback", recursive_scanning=False)
        service.add_entry(
            world_book_id=wb_id,
            keywords=[r"hero.*"],
            content="Regex hero lore",
            regex_match=True,
            case_sensitive=False,
            whole_word_match=True,
            priority=100,
        )

        activated = service.process_context("heroic tale", wb_id)
        contents = [entry["content"] for entry in activated]
        assert "Regex hero lore" in contents

    @pytest.mark.unit
    def test_token_budget_limit(self, world_book_service, mock_tokenizer):
        """Test respecting token budget."""
        service = world_book_service

        wb_id = service.create_world_book(name="Budget Test")

        # Add many entries
        for i in range(10):
            service.add_entry(
                wb_id,
                [f'keyword{i}'],
                f'This is a long content for entry {i} ' * 10,
                priority=100 - i
            )

        context = ' '.join([f'keyword{i}' for i in range(10)])

        with patch.object(service, 'count_tokens', side_effect=lambda x: len(x.split())):
            activated = service.process_context(
                context,
                wb_id,
                max_tokens=100
            )

        # Should limit entries to stay within budget
        total_tokens = sum(len(e['content'].split()) for e in activated)
        assert total_tokens <= 120  # Allow small overhead


# ========================================================================
# Search and Filter Tests
# ========================================================================


class TestSearchAndFilter:
    """Test search and filter functionality."""

    @pytest.mark.unit
    def test_search_entries(self, world_book_service):
        """Test searching world book entries."""
        service = world_book_service

        wb_id = service.create_world_book(name="Search Test")
        service.add_entry(wb_id, ['dragon'], 'Dragon content')
        service.add_entry(wb_id, ['wizard'], 'Wizard content')
        service.add_entry(wb_id, ['dragon', 'wizard'], 'Both content')

        results = service.search_entries(wb_id, query="dragon")

        assert len(results) == 2
        assert all('dragon' in str(r['keywords']) for r in results)

    @pytest.mark.unit
    def test_filter_by_priority(self, world_book_service):
        """Test filtering entries by priority."""
        service = world_book_service

        wb_id = service.create_world_book(name="Priority Filter")
        service.add_entry(wb_id, ['low'], 'Low', priority=25)
        service.add_entry(wb_id, ['medium'], 'Medium', priority=50)
        service.add_entry(wb_id, ['high'], 'High', priority=100)

        high_priority = service.filter_entries(wb_id, min_priority=75)

        assert len(high_priority) == 1
        assert high_priority[0]['content'] == 'High'

    @pytest.mark.unit
    def test_filter_recursive_entries(self, world_book_service):
        """Test filtering recursive scanning entries."""
        service = world_book_service

        wb_id = service.create_world_book(name="Recursive Filter")
        service.add_entry(wb_id, ['normal'], 'Normal', recursive_scanning=False)
        service.add_entry(wb_id, ['recursive'], 'Recursive', recursive_scanning=True)

        recursive = service.filter_entries(wb_id, recursive_only=True)

        assert len(recursive) == 1
        assert recursive[0]['content'] == 'Recursive'


# ========================================================================
# Import/Export Tests
# ========================================================================


class TestImportExport:
    """Test import/export functionality."""

    @pytest.mark.unit
    def test_export_world_book(self, world_book_service, sample_world_book):
        """Test exporting world book."""
        service = world_book_service

        wb_id = service.create_world_book(
            name=sample_world_book['name'],
            description=sample_world_book['description']
        )
        for entry in sample_world_book['entries']:
            service.add_entry(wb_id, **entry)

        exported = service.export_world_book(wb_id)

        assert exported['name'] == sample_world_book['name']
        assert 'entries' in exported
        assert len(exported['entries']) == len(sample_world_book['entries'])

    @pytest.mark.unit
    def test_import_world_book(self, world_book_service, sample_world_book):
        """Test importing world book."""
        service = world_book_service

        wb_id = service.import_world_book(sample_world_book)

        assert wb_id is not None

        world_book = service.get_world_book(wb_id)
        assert world_book['name'] == sample_world_book['name']

        entries = service.get_entries(wb_id)
        assert len(entries) == len(sample_world_book['entries'])

    @pytest.mark.unit
    def test_import_world_book_coerces_string_booleans(self, world_book_service):
        """Import should coerce string boolean flags for compatibility payloads."""
        service = world_book_service

        payload = {
            "world_book": {
                "name": "Imported String Bool Flags",
                "description": "compat test",
                "recursive_scanning": "false",
                "enabled": "false",
            },
            "entries": [
                {
                    "keywords": ["hero"],
                    "content": "Hero lore",
                    "enabled": "false",
                    "recursive_scanning": "false",
                    "metadata": {"appendable": "false"},
                }
            ],
        }

        wb_id = service.import_world_book(payload)

        world_book = service.get_world_book(wb_id)
        assert world_book is not None
        assert world_book["enabled"] in (0, False)
        assert world_book["recursive_scanning"] in (0, False)

        entries = service.get_entries(wb_id, enabled_only=False)
        assert len(entries) == 1
        assert entries[0]["enabled"] is False
        assert entries[0]["recursive_scanning"] is False
        assert entries[0]["appendable"] is False

        enabled_only_entries = service.get_entries(wb_id, enabled_only=True)
        assert enabled_only_entries == []

    @pytest.mark.unit
    def test_export_to_lorebook_format(self, world_book_service, sample_world_book):
        """Test exporting to lorebook format."""
        service = world_book_service

        wb_id = service.create_world_book(name=sample_world_book['name'])
        for entry in sample_world_book['entries']:
            service.add_entry(wb_id, **entry)

        lorebook = service.export_to_lorebook_format(wb_id)

        assert 'entries' in lorebook
        assert all('key' in e for e in lorebook['entries'])
        assert all('content' in e for e in lorebook['entries'])


# ========================================================================
# Statistics Tests
# ========================================================================


class TestStatistics:
    """Test statistics and analytics."""

    @pytest.mark.unit
    def test_get_statistics(self, world_book_service, sample_world_book):
        """Test getting world book statistics."""
        service = world_book_service

        wb_id = service.create_world_book(name="Stats Test")
        for entry in sample_world_book['entries']:
            service.add_entry(wb_id, **entry)

        stats = service.get_statistics(wb_id)

        assert stats['total_entries'] == len(sample_world_book['entries'])
        assert stats['total_keywords'] > 0
        assert 'avg_priority' in stats
        assert 'recursive_entries' in stats

    @pytest.mark.unit
    def test_get_activation_statistics(self, world_book_service):
        """Test getting activation statistics."""
        service = world_book_service

        wb_id = service.create_world_book(name="Activation Stats")
        service.add_entry(wb_id, ['test'], 'Test content')

        # Process context multiple times
        for _ in range(5):
            service.process_context("test context", wb_id)

        stats = service.get_activation_statistics(wb_id)

        assert stats.get('total_activations', 0) >= 5

    @pytest.mark.unit
    def test_activation_statistics_counts_per_book(self, world_book_service):
        """Ensure activations are counted per world book."""
        service = world_book_service

        wb1 = service.create_world_book(name="WB One")
        wb2 = service.create_world_book(name="WB Two")
        service.add_entry(wb1, ['alpha'], 'Alpha lore')
        service.add_entry(wb2, ['beta'], 'Beta lore')

        service.process_context("alpha beta", [wb1, wb2])

        stats1 = service.get_activation_statistics(wb1)
        stats2 = service.get_activation_statistics(wb2)
        assert stats1.get('total_activations', 0) == 1
        assert stats2.get('total_activations', 0) == 1

    @pytest.mark.unit
    def test_process_context_skips_large_entries_when_over_budget(self, world_book_service):
        """Large entries over the token budget should not block smaller matches."""
        service = world_book_service

        wb_id = service.create_world_book(name="Budget Book")
        big_entry_id = service.add_entry(
            wb_id,
            ['topic'],
            ' '.join(['large'] * 20),
            priority=200,
        )
        small_entry_id = service.add_entry(
            wb_id,
            ['topic'],
            'small info',
            priority=50,
        )

        result = service.process_context("topic", [wb_id], token_budget=5)

        assert isinstance(result, dict)
        assert result.get('entries_matched') == 1
        assert result.get('tokens_used', 0) <= 5
        assert small_entry_id in result.get('entry_ids', [])
        assert big_entry_id not in result.get('entry_ids', [])
        assert 'small info' in result.get('processed_context', "")

    @pytest.mark.unit
    def test_process_context_deduplicates_world_books(self, world_book_service):
        """Duplicate world book IDs should not duplicate matched entries."""
        service = world_book_service

        wb_id = service.create_world_book(name="Dedup Book")
        service.add_entry(wb_id, ['signal'], 'One signal entry')

        result = service.process_context("signal", [wb_id, wb_id])

        assert isinstance(result, dict)
        assert result.get('entries_matched') == 1
        assert result.get('books_used') == 1

    @pytest.mark.unit
    def test_process_context_respects_scan_depth_limits(self, world_book_service):
        """Respect both book and request scan depth caps."""
        service = world_book_service

        wb_id = service.create_world_book(name="Depth Book", scan_depth=2)
        service.add_entry(wb_id, ['topic'], 'First entry', priority=100)
        service.add_entry(wb_id, ['topic'], 'Second entry', priority=90)
        service.add_entry(wb_id, ['topic'], 'Third entry', priority=80)

        result = service.process_context("topic", [wb_id], token_budget=100)
        assert result.get('entries_matched') == 2
        assert 'Third entry' not in result.get('processed_context', "")

        # Request-level override should further reduce depth
        result_override = service.process_context("topic", [wb_id], token_budget=100, scan_depth=1)
        assert result_override.get('entries_matched') == 1

    @pytest.mark.unit
    def test_process_context_returns_diagnostics_with_reason_and_token_cost(self, world_book_service):
        """Diagnostics include activation reason and per-entry token accounting."""
        service = world_book_service

        wb_id = service.create_world_book(name="Diagnostics Book")
        keyword_entry_id = service.add_entry(
            wb_id,
            ["seed"],
            "dragon marker",
            priority=120,
        )
        regex_entry_id = service.add_entry(
            wb_id,
            [r"se.d"],
            "regex marker",
            priority=110,
            regex_match=True,
        )
        depth_entry_id = service.add_entry(
            wb_id,
            ["dragon"],
            "depth marker",
            priority=90,
        )

        result = service.process_context(
            "seed",
            [wb_id],
            token_budget=200,
            recursive_scanning=True,
            recursive_depth=1,
            include_diagnostics=True,
        )

        assert isinstance(result, dict)
        diagnostics = result.get("diagnostics") or []
        assert diagnostics

        diagnostics_by_id = {
            item.get("entry_id"): item
            for item in diagnostics
            if isinstance(item, dict)
        }

        keyword_diag = diagnostics_by_id.get(keyword_entry_id)
        regex_diag = diagnostics_by_id.get(regex_entry_id)
        depth_diag = diagnostics_by_id.get(depth_entry_id)

        assert keyword_diag is not None
        assert keyword_diag.get("activation_reason") == "keyword_match"
        assert keyword_diag.get("token_cost") == service.count_tokens("dragon marker")

        assert regex_diag is not None
        assert regex_diag.get("activation_reason") == "regex_match"
        assert regex_diag.get("token_cost") == service.count_tokens("regex marker")

        assert depth_diag is not None
        assert depth_diag.get("activation_reason") == "depth"
        assert depth_diag.get("token_cost") == service.count_tokens("depth marker")

    @pytest.mark.unit
    def test_process_context_reports_budget_exhaustion_metadata(self, world_book_service):
        """Budget metadata reports skipped entries when token budget is exhausted."""
        service = world_book_service

        wb_id = service.create_world_book(name="Budget Diagnostics")
        service.add_entry(wb_id, ["topic"], "small one", priority=100)
        service.add_entry(wb_id, ["topic"], "small two", priority=90)
        service.add_entry(wb_id, ["topic"], "small three", priority=80)

        result = service.process_context(
            "topic",
            [wb_id],
            token_budget=2,
            include_diagnostics=True,
        )

        assert isinstance(result, dict)
        assert result.get("token_budget") == 2
        assert result.get("budget_exhausted") is True
        assert int(result.get("skipped_entries_due_to_budget", 0)) >= 1


# ========================================================================
# Cloning Tests
# ========================================================================


class TestCloning:
    """Test world book cloning."""

    @pytest.mark.unit
    def test_clone_world_book(self, world_book_service, sample_world_book):
        """Test cloning a world book."""
        service = world_book_service

        # Create original
        original_id = service.create_world_book(name="Original")
        for entry in sample_world_book['entries']:
            service.add_entry(original_id, **entry)

        # Clone
        cloned_id = service.clone_world_book(original_id, new_name="Clone")

        assert cloned_id != original_id

        # Check clone has same entries
        original_entries = service.get_entries(original_id)
        cloned_entries = service.get_entries(cloned_id)

        assert len(cloned_entries) == len(original_entries)

        # Check entries match (excluding IDs)
        for orig, clone in zip(original_entries, cloned_entries):
            assert orig['keywords'] == clone['keywords']
            assert orig['content'] == clone['content']
            assert orig['priority'] == clone['priority']


# ========================================================================
# Keyword Processing Tests
# ========================================================================


class TestKeywordProcessing:
    """Test keyword normalization and processing."""

    @pytest.mark.unit
    def test_keyword_normalization(self, world_book_service):
        """Test keyword normalization."""
        service = world_book_service

        # Test various normalizations
        assert service.normalize_keyword("  Test  ") == "test"
        assert service.normalize_keyword("TEST") == "test"
        assert service.normalize_keyword("test-keyword") == "test-keyword"
        assert service.normalize_keyword("test_keyword") == "test_keyword"

    @pytest.mark.unit
    def test_keyword_matching_case_insensitive(self, world_book_service):
        """Test case-insensitive keyword matching."""
        service = world_book_service

        wb_id = service.create_world_book(name="Case Test")
        service.add_entry(wb_id, ['Dragon'], 'Dragon entry')

        # Should match regardless of case
        context_lower = "dragon appears"
        context_upper = "DRAGON appears"
        context_mixed = "DrAgOn appears"

        for context in [context_lower, context_upper, context_mixed]:
            activated = service.process_context(context, wb_id)
            assert len(activated) == 1
            assert activated[0]['content'] == 'Dragon entry'


# ========================================================================
# Error Handling Tests
# ========================================================================


class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.unit
    def test_world_book_not_found(self, world_book_service):
        """Test handling world book not found."""
        service = world_book_service

        result = service.get_world_book(999999)
        assert result is None

    @pytest.mark.unit
    def test_entry_validation(self, world_book_service):
        """Test entry validation."""
        service = world_book_service

        wb_id = service.create_world_book(name="Validation Test")

        # Empty keywords should fail
        with pytest.raises(ValueError):
            service.add_entry(wb_id, [], "Content")

        # Empty content is allowed but might warn
        entry_id = service.add_entry(wb_id, ['test'], "")
        assert entry_id is not None

    @pytest.mark.unit
    def test_invalid_priority(self, world_book_service):
        """Test invalid priority handling."""
        service = world_book_service

        wb_id = service.create_world_book(name="Priority Test")

        # Negative priority should be clamped to 0
        entry_id = service.add_entry(
            wb_id,
            ['test'],
            'Content',
            priority=-10
        )

        entries = service.get_entries(wb_id)
        assert entries[0]['priority'] >= 0

        # Priority over 100 should be clamped to 100
        entry_id2 = service.add_entry(
            wb_id,
            ['test2'],
            'Content2',
            priority=200
        )

        entries = service.get_entries(wb_id)
        entry = next(e for e in entries if 'test2' in e['keywords'])
        assert entry['priority'] <= 100


def test_priority_is_preserved_between_literal_and_regex_candidates(world_book_service):
    service = world_book_service
    book = service.create_world_book("Competing match priority")
    service.add_entry(book, ["harbor"], "Low literal", priority=1)
    high = service.add_entry(book, [r"harb.r"], "High regex", priority=9, regex_match=True)
    result = service.process_context("harbor", world_book_ids=[book], scan_depth=1, token_budget=20)
    assert result["entry_ids"] == [high]


def test_strict_regex_timeout_reports_failure_and_preserves_legacy_nonmatch():
    from tldw_Server_API.app.core.Character_Chat.world_book_manager import (
        WorldBookEntry,
    )

    class TimedOutPattern:
        def search(self, text, **kwargs):
            raise TimeoutError("Owned regex evaluation timed out")

    entry = WorldBookEntry(entry_id=1, keywords=["harbor"], regex_match=True)
    entry._patterns = [(TimedOutPattern(), "harbor")]
    assert entry.get_first_match_info("harbor") is None
    with pytest.raises(TimeoutError):
        entry.get_first_match_info("harbor", strict=True)
