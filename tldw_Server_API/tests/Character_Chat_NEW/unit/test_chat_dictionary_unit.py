"""
Unit tests for ChatDictionaryService.

Tests the chat dictionary functionality for text replacements.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import re
import random
from datetime import timedelta, timezone

from tldw_Server_API.app.core.Character_Chat.chat_dictionary import ChatDictionaryService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

# ========================================================================
# Dictionary Management Tests
# ========================================================================

class TestDictionaryManagement:
    """Test dictionary CRUD operations."""

    @pytest.mark.unit
    def test_create_dictionary(self, chat_dictionary_service, sample_dictionary):
        """Test creating a new dictionary."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(
            name=sample_dictionary['name'],
            description=sample_dictionary['description']
        )

        assert dict_id is not None
        assert dict_id > 0

    @pytest.mark.unit
    def test_get_dictionary(self, chat_dictionary_service, sample_dictionary):
        """Test getting a dictionary."""
        service = chat_dictionary_service

        # Create dictionary
        dict_id = service.create_dictionary(
            name=sample_dictionary['name'],
            description=sample_dictionary['description']
        )

        # Get dictionary
        dictionary = service.get_dictionary(dict_id)

        assert dictionary is not None
        assert dictionary['name'] == sample_dictionary['name']
        assert dictionary['description'] == sample_dictionary['description']

    @pytest.mark.unit
    def test_list_dictionaries(self, chat_dictionary_service):
        """Test listing all dictionaries."""
        service = chat_dictionary_service

        # Create multiple dictionaries
        for i in range(3):
            service.create_dictionary(
                name=f"Dictionary {i}",
                description=f"Description {i}"
            )

        dictionaries = service.list_dictionaries()

        assert len(dictionaries) >= 3
        assert all('name' in d for d in dictionaries)

    @pytest.mark.unit
    def test_update_dictionary(self, chat_dictionary_service):
        """Test updating a dictionary."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(
            name="Original",
            description="Original description"
        )

        service.update_dictionary(
            dict_id=dict_id,
            name="Updated",
            description="Updated description"
        )

        updated = service.get_dictionary(dict_id)
        assert updated['name'] == "Updated"
        assert updated['description'] == "Updated description"

    @pytest.mark.unit
    def test_delete_dictionary(self, chat_dictionary_service):
        """Test deleting a dictionary."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="To Delete")

        service.delete_dictionary(dict_id)

        # Should not find deleted dictionary
        deleted = service.get_dictionary(dict_id)
        assert deleted is None or deleted.get('is_deleted', False)

    @pytest.mark.unit
    def test_toggle_dictionary_active(self, chat_dictionary_service):
        """Test toggling dictionary active status."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Toggle Test")

        # Initially active
        dictionary = service.get_dictionary(dict_id)
        initial_status = dictionary.get('is_active', True)

        # Toggle
        service.toggle_dictionary_active(dict_id)

        # Check toggled
        dictionary = service.get_dictionary(dict_id)
        assert dictionary.get('is_active', True) != initial_status

# ========================================================================
# Entry Management Tests
# ========================================================================

class TestEntryManagement:
    """Test dictionary entry operations."""

    @pytest.mark.unit
    def test_add_literal_entry(self, chat_dictionary_service):
        """Test adding a literal replacement entry."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Test Dict")

        entry_id = service.add_entry(
            dictionary_id=dict_id,
            pattern="AI",
            replacement="Artificial Intelligence",
            type="literal"
        )

        assert entry_id is not None
        entries = service.get_entries(dict_id)
        assert len(entries) == 1
        assert entries[0]['pattern'] == "AI"

    @pytest.mark.unit
    def test_add_regex_entry(self, chat_dictionary_service):
        """Test adding a regex replacement entry."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Regex Dict")

        entry_id = service.add_entry(
            dictionary_id=dict_id,
            pattern=r'\b(\d+)\s*F\b',
            replacement=r'\1 Fahrenheit',
            type="regex"
        )

        assert entry_id is not None
        entries = service.get_entries(dict_id)
        assert entries[0]['type'] == "regex"

    @pytest.mark.unit
    def test_add_entry_with_probability(self, chat_dictionary_service):
        """Test adding entry with probability."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Prob Dict")

        entry_id = service.add_entry(
            dictionary_id=dict_id,
            pattern="lol",
            replacement="laugh out loud",
            type="literal",
            probability=0.5
        )

        entries = service.get_entries(dict_id)
        assert entries[0]['probability'] == 0.5

    @pytest.mark.unit
    def test_update_entry(self, chat_dictionary_service):
        """Test updating an entry."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Update Test")
        entry_id = service.add_entry(
            dictionary_id=dict_id,
            pattern="old",
            replacement="new"
        )

        service.update_entry(
            entry_id=entry_id,
            pattern="updated",
            replacement="changed"
        )

        entries = service.get_entries(dict_id)
        assert entries[0]['pattern'] == "updated"
        assert entries[0]['replacement'] == "changed"

    @pytest.mark.unit
    def test_delete_entry(self, chat_dictionary_service):
        """Test deleting an entry."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Delete Test")
        entry_id = service.add_entry(
            dictionary_id=dict_id,
            pattern="test",
            replacement="replacement"
        )

        service.delete_entry(entry_id)

        entries = service.get_entries(dict_id)
        assert len(entries) == 0

    @pytest.mark.unit
    def test_bulk_add_entries(self, chat_dictionary_service, sample_dictionary):
        """Test bulk adding entries."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Bulk Test")

        results = service.bulk_add_entries(
            dictionary_id=dict_id,
            entries=sample_dictionary['entries']
        )

        assert results['added'] == len(sample_dictionary['entries'])
        entries = service.get_entries(dict_id)
        assert len(entries) == len(sample_dictionary['entries'])

# ========================================================================
# Text Processing Tests
# ========================================================================

class TestTextProcessing:
    """Test text replacement functionality."""

    @pytest.mark.unit
    def test_literal_replacement(self, chat_dictionary_service):
        """Test literal text replacement."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Literal Test")
        service.add_entry(
            dictionary_id=dict_id,
            pattern="AI",
            replacement="Artificial Intelligence",
            type="literal"
        )

        text = "AI is amazing. I love AI!"
        processed = service.process_text(text, dict_id)

        assert "Artificial Intelligence" in processed
        assert "AI" not in processed

    @pytest.mark.unit
    def test_regex_replacement(self, chat_dictionary_service):
        """Test regex text replacement."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Regex Test")
        service.add_entry(
            dictionary_id=dict_id,
            pattern=r'\b(\d+)\s*F\b',
            replacement=r'\1 Fahrenheit',
            type="regex"
        )

        text = "It's 72F today, but tomorrow will be 65F."
        processed = service.process_text(text, dict_id)

        assert "72 Fahrenheit" in processed
        assert "65 Fahrenheit" in processed
        assert "F" not in processed.replace("Fahrenheit", "")

    @pytest.mark.unit
    def test_regex_case_sensitive_ignores_entry_flag(self, chat_dictionary_service):
        """Regex case sensitivity should be controlled by /i flag, not entry flag."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Regex Case Flag")
        service.add_entry(
            dictionary_id=dict_id,
            pattern=r"hello",
            replacement="hi",
            type="regex",
            case_sensitive=False,
        )

        text = "Hello there"
        processed = service.process_text(text, dict_id)

        # Without /i, regex stays case-sensitive even if case_sensitive=False
        assert processed == text

    @pytest.mark.unit
    def test_regex_case_insensitive_requires_flag(self, chat_dictionary_service):
        """Regex /i flag should make matches case-insensitive regardless of entry flag."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Regex Case Flag i")
        service.add_entry(
            dictionary_id=dict_id,
            pattern=r"/hello/i",
            replacement="hi",
            type="regex",
            case_sensitive=True,
        )

        text = "Hello there"
        processed = service.process_text(text, dict_id)

        assert "hi" in processed

    @pytest.mark.unit
    def test_process_text_rejects_oversized_input(self, chat_dictionary_service, monkeypatch):
        """Ensure oversized input is rejected before regex processing."""
        from tldw_Server_API.app.core.Character_Chat import chat_dictionary as cd

        monkeypatch.setattr(cd, "MAX_CHAT_DICTIONARY_TEXT_LENGTH", 10)

        service = chat_dictionary_service
        dict_id = service.create_dictionary(name="Limit Test")
        service.add_entry(
            dictionary_id=dict_id,
            pattern="AI",
            replacement="Artificial Intelligence",
            type="literal",
        )

        with pytest.raises(InputError):
            service.process_text("x" * 11, dict_id)

    @pytest.mark.unit
    def test_case_sensitive_replacement(self, chat_dictionary_service):
        """Test case-sensitive replacement."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Case Test")
        service.add_entry(
            dictionary_id=dict_id,
            pattern="AI",
            replacement="Artificial Intelligence",
            type="literal",
            case_sensitive=True
        )

        text = "AI is different from ai or Ai."
        processed = service.process_text(text, dict_id)

        assert "Artificial Intelligence" in processed
        assert "ai" in processed  # Lowercase not replaced
        assert "Ai" in processed  # Different case not replaced

    @pytest.mark.unit
    def test_case_insensitive_literal_replacement(self, chat_dictionary_service):
        """Ensure case-insensitive literals match before replacement."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Case Insensitive Test")
        service.add_entry(
            dictionary_id=dict_id,
            pattern="AI",
            replacement="ZED",
            type="literal",
            case_sensitive=False,
        )

        text = "AI and ai and Ai should all change."
        processed = service.process_text(text, dict_id)

        assert processed.count("ZED") == 3
        assert "AI" not in processed
        assert "ai" not in processed
        assert "Ai" not in processed

    @pytest.mark.unit
    def test_probability_replacement(self, chat_dictionary_service):
        """Test probabilistic replacement."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Prob Test")
        service.add_entry(
            dictionary_id=dict_id,
            pattern="maybe",
            replacement="perhaps",
            probability=0.5
        )

        iterations = 20
        deterministic_rng = random.Random(0)  # nosec B311

        replacements = 0
        with patch(
            "tldw_Server_API.app.core.Character_Chat.chat_dictionary.random.random",
            side_effect=deterministic_rng.random,
        ):
            for _ in range(iterations):
                text = "maybe this will work"
                processed = service.process_text(text, dict_id)
                if "perhaps" in processed:
                    replacements += 1

        assert 0 < replacements < iterations

    @pytest.mark.unit
    def test_delay_prevents_initial_trigger(self, chat_dictionary_service, monkeypatch):
        """Ensure configured delay blocks the first replacement until the window has elapsed."""
        from tldw_Server_API.app.core.Character_Chat import chat_dictionary as cd

        original_datetime = cd.datetime
        anchor_time = original_datetime.utcnow()

        class _FakeDatetime(original_datetime):  # type: ignore[misc]
            _delta = timedelta(0)

            @classmethod
            def now(cls, tz=None):  # type: ignore[override]
                current = anchor_time + cls._delta
                if tz is not None:
                    return tz.fromutc(current.replace(tzinfo=tz))
                return current

            @classmethod
            def utcnow(cls):
                return anchor_time + cls._delta

        monkeypatch.setattr(cd, "datetime", _FakeDatetime)

        service = chat_dictionary_service
        dict_id = service.create_dictionary(name="Delay Test")
        service.add_entry(
            dict_id,
            pattern="trigger",
            replacement="applied",
            timed_effects={"delay": 60, "cooldown": 0, "sticky": 0},
        )

        source_text = "trigger phrase"
        first_pass = service.process_text(source_text, dict_id)
        assert "applied" not in str(first_pass)

        _FakeDatetime._delta = timedelta(seconds=61)
        entry_snapshot = service.get_entry_objects(dict_id, active_only=True)[0]
        assert (
            cd.datetime.now(timezone.utc) - entry_snapshot._loaded_at
        ).total_seconds() >= 60
        second_pass = service.process_text(source_text, dict_id)
        assert "applied" in str(second_pass)

    @pytest.mark.unit
    def test_multiple_replacements(self, chat_dictionary_service):
        """Test multiple replacements in same text."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Multi Test")
        service.add_entry(dict_id, "AI", "Artificial Intelligence")
        service.add_entry(dict_id, "ML", "Machine Learning")
        service.add_entry(dict_id, "DL", "Deep Learning")

        text = "AI includes ML and DL techniques."
        processed = service.process_text(text, dict_id)

        assert "Artificial Intelligence" in processed
        assert "Machine Learning" in processed
        assert "Deep Learning" in processed

    @pytest.mark.unit
    def test_token_budget_limit(self, chat_dictionary_service, mock_tokenizer):
        """Test respecting token budget during replacement."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Budget Test")
        service.add_entry(
            dict_id,
            "short",
            "this is a very long replacement text that uses many tokens"
        )

        text = "short short short short short"

        with patch.object(service, 'count_tokens', side_effect=lambda x: len(x.split())):
            processed = service.process_text(text, dict_id, max_tokens=20)

        # Should stop replacing when budget exceeded
        token_count = len(processed.split())
        assert token_count <= 25  # Allow small overhead

# ========================================================================
# Search and Filter Tests
# ========================================================================

class TestSearchAndFilter:
    """Test search and filter functionality."""

    @pytest.mark.unit
    def test_search_entries(self, chat_dictionary_service):
        """Test searching dictionary entries."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Search Test")
        service.add_entry(dict_id, "test1", "replacement1")
        service.add_entry(dict_id, "test2", "replacement2")
        service.add_entry(dict_id, "other", "replacement3")

        results = service.search_entries(dict_id, query="test")

        assert len(results) == 2
        assert all("test" in r['pattern'] for r in results)

    @pytest.mark.unit
    def test_filter_by_type(self, chat_dictionary_service):
        """Test filtering entries by type."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Filter Test")
        service.add_entry(dict_id, "literal1", "rep1", type="literal")
        service.add_entry(dict_id, r"\d+", "numbers", type="regex")
        service.add_entry(dict_id, "literal2", "rep2", type="literal")

        literals = service.filter_entries(dict_id, type="literal")
        regexes = service.filter_entries(dict_id, type="regex")

        assert len(literals) == 2
        assert len(regexes) == 1

    @pytest.mark.unit
    def test_filter_active_entries(self, chat_dictionary_service):
        """Test filtering active/inactive entries."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Active Test")
        entry1 = service.add_entry(dict_id, "active", "rep")
        entry2 = service.add_entry(dict_id, "inactive", "rep")

        # Disable one entry
        service.toggle_entry_active(entry2)

        active = service.filter_entries(dict_id, active_only=True)

        assert len(active) == 1
        assert active[0]['pattern'] == "active"

# ========================================================================
# Import/Export Tests
# ========================================================================

class TestImportExport:
    """Test import/export functionality."""

    @pytest.mark.unit
    def test_export_to_markdown(self, chat_dictionary_service, sample_dictionary):
        """Test exporting dictionary to markdown."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name=sample_dictionary['name'])
        for entry in sample_dictionary['entries']:
            service.add_entry(dict_id, **entry)

        markdown = service.export_to_markdown(dict_id)

        assert f"# {sample_dictionary['name']}" in markdown
        assert "## Entry:" in markdown
        assert "AI" in markdown
        assert "Artificial Intelligence" in markdown

    @pytest.mark.unit
    def test_import_from_markdown(self, chat_dictionary_service, markdown_dictionary):
        """Test importing dictionary from markdown."""
        service = chat_dictionary_service

        dict_id = service.import_from_markdown(markdown_dictionary)

        assert dict_id is not None

        dictionary = service.get_dictionary(dict_id)
        assert dictionary['name'] == "Test Dictionary"

        entries = service.get_entries(dict_id)
        assert len(entries) == 3
        assert any(e['pattern'] == 'AI' for e in entries)
        slang_entry = next(e for e in entries if e['pattern'] == 'lol')
        assert slang_entry['probability'] == pytest.approx(0.5, rel=1e-6)

    @pytest.mark.unit
    def test_import_from_markdown_string_does_not_open_file(self, chat_dictionary_service, monkeypatch):
        """Ensure string inputs are treated as content, not file paths."""
        from tldw_Server_API.app.core.Character_Chat import chat_dictionary as cd

        def _fail_open(*_args, **_kwargs):
            raise AssertionError("open should not be called for string content")

        monkeypatch.setattr(cd, "open", _fail_open, raising=False)

        service = chat_dictionary_service
        dict_id = service.import_from_markdown("not_a_path", "String Content")
        assert dict_id is not None

    @pytest.mark.unit
    def test_export_to_json(self, chat_dictionary_service, sample_dictionary):
        """Test exporting dictionary to JSON."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name=sample_dictionary['name'])
        for entry in sample_dictionary['entries']:
            service.add_entry(dict_id, **entry)

        json_data = service.export_to_json(dict_id)

        assert 'name' in json_data
        assert 'entries' in json_data
        assert len(json_data['entries']) == len(sample_dictionary['entries'])

    @pytest.mark.unit
    def test_import_from_json(self, chat_dictionary_service, sample_dictionary):
        """Test importing dictionary from JSON."""
        service = chat_dictionary_service

        dict_id = service.import_from_json(sample_dictionary)

        assert dict_id is not None

        dictionary = service.get_dictionary(dict_id)
        assert dictionary['name'] == sample_dictionary['name']

        entries = service.get_entries(dict_id)
        assert len(entries) == len(sample_dictionary['entries'])

# ========================================================================
# Statistics Tests
# ========================================================================

class TestStatistics:
    """Test statistics and analytics."""

    @pytest.mark.unit
    def test_get_statistics(self, chat_dictionary_service):
        """Test getting dictionary statistics."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Stats Test")
        service.add_entry(dict_id, "entry1", "rep1")
        service.add_entry(dict_id, "entry2", "rep2", type="regex")
        service.add_entry(dict_id, "entry3", "rep3", probability=0.5)

        stats = service.get_statistics(dict_id)

        assert stats['total_entries'] == 3
        assert stats['literal_entries'] == 2
        assert stats['regex_entries'] == 1
        assert stats['probabilistic_entries'] == 1

    @pytest.mark.unit
    def test_get_usage_statistics(self, chat_dictionary_service):
        """Test getting usage statistics."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Usage Test")
        service.add_entry(dict_id, "test", "replacement")

        # Process text multiple times
        for _ in range(5):
            service.process_text("test text", dict_id)

        usage = service.get_usage_statistics(dict_id)

        assert usage.get('times_used', 0) >= 5

# ========================================================================
# Cloning and Duplication Tests
# ========================================================================

class TestCloning:
    """Test dictionary cloning."""

    @pytest.mark.unit
    def test_clone_dictionary(self, chat_dictionary_service, sample_dictionary):
        """Test cloning a dictionary."""
        service = chat_dictionary_service

        # Create original
        original_id = service.create_dictionary(name="Original")
        for entry in sample_dictionary['entries']:
            service.add_entry(original_id, **entry)

        # Clone
        cloned_id = service.clone_dictionary(original_id, new_name="Clone")

        assert cloned_id != original_id

        # Check clone has same entries
        original_entries = service.get_entries(original_id)
        cloned_entries = service.get_entries(cloned_id)

        assert len(cloned_entries) == len(original_entries)

        # Check entries match (excluding IDs)
        for orig, clone in zip(original_entries, cloned_entries):
            assert orig['pattern'] == clone['pattern']
            assert orig['replacement'] == clone['replacement']

# ========================================================================
# Error Handling Tests
# ========================================================================

class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.unit
    def test_invalid_regex_pattern(self, chat_dictionary_service):
        """Test handling invalid regex pattern."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Invalid Regex")

        with pytest.raises(re.error):
            service.add_entry(
                dictionary_id=dict_id,
                pattern="[invalid(regex",  # Unclosed bracket
                replacement="test",
                type="regex"
            )

    @pytest.mark.unit
    def test_dictionary_not_found(self, chat_dictionary_service):
        """Test handling dictionary not found."""
        service = chat_dictionary_service

        result = service.get_dictionary(999999)
        assert result is None

    @pytest.mark.unit
    def test_entry_validation(self, chat_dictionary_service):
        """Test entry validation."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Validation Test")

        # Empty pattern should fail
        with pytest.raises(ValueError):
            service.add_entry(dict_id, "", "replacement")

        # Empty replacement is allowed but should warn
        entry_id = service.add_entry(dict_id, "pattern", "")
        assert entry_id is not None  # Allow but might log warning

# ========================================================================
# Cache Management Tests
# ========================================================================

class TestCacheManagement:
    """Test cache functionality."""

    @pytest.mark.unit
    def test_clear_cache(self, chat_dictionary_service):
        """Test clearing dictionary cache."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Cache Test")
        service.add_entry(dict_id, "test", "replacement")

        # Process text (should cache)
        service.process_text("test text", dict_id)

        # Clear cache
        service.clear_cache()

        # Process again (should work without cache)
        result = service.process_text("test text", dict_id)
        assert "replacement" in result

    @pytest.mark.unit
    def test_toggle_entry_invalidates_cache(self, chat_dictionary_service):
        """Disabling an entry should stop replacements even with cached entries."""
        service = chat_dictionary_service

        dict_id = service.create_dictionary(name="Toggle Cache Test")
        entry_id = service.add_entry(dict_id, "world", "universe")

        warm_result = service.process_text("hello world")
        assert "universe" in str(warm_result)

        service.toggle_entry_active(entry_id, is_active=False)

        post_toggle_result = service.process_text("hello world")
        assert "universe" not in str(post_toggle_result)
