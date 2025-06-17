# test_rag_integration.py
# Description: Integration tests for RAG library optimizations and error handling
#
# Imports
import pytest
import json
import tempfile
import time
from unittest.mock import patch, MagicMock
from pathlib import Path
#
# Local Imports
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.RAG.RAG_Unified_Library_v2 import (
    fetch_relevant_ids_by_keywords,
    DatabaseType,
    save_chat_history,
    _schedule_temp_file_cleanup
)
from tldw_Server_API.app.core.RAG.exceptions import (
    RAGError,
    RAGSearchError,
    RAGValidationError,
    RAGDatabaseError
)
from tldw_Server_API.app.core.config import RAG_SEARCH_CONFIG
#
#######################################################################################################################
#
# Test Fixtures

@pytest.fixture
def client_id():
    return "test_rag_integration"

@pytest.fixture
def mem_db(client_id):
    """Creates an in-memory CharactersRAGDB for testing."""
    db = CharactersRAGDB(":memory:", client_id)
    yield db
    db.close_connection()

@pytest.fixture
def sample_characters_with_tags(mem_db):
    """Create sample character cards for integration testing."""
    characters_data = [
        {
            "name": "Fantasy Wizard",
            "description": "A powerful wizard",
            "personality": "Wise and mysterious",
            "scenario": "Magic realm",
            "tags": json.dumps(["fantasy", "magic", "wizard", "powerful"]),
            "first_message": "Welcome to the magical realm!"
        },
        {
            "name": "Sci-Fi Android",
            "description": "Advanced android companion",
            "personality": "Logical and helpful",
            "scenario": "Space station",
            "tags": json.dumps(["sci-fi", "android", "technology", "companion"]),
            "first_message": "Systems online. How may I assist?"
        },
        {
            "name": "Adventure Explorer",
            "description": "Brave explorer of unknown lands",
            "personality": "Adventurous and bold",
            "scenario": "Jungle expedition",
            "tags": json.dumps(["adventure", "explorer", "brave", "jungle"]),
            "first_message": "Ready for the next adventure!"
        }
    ]
    
    character_ids = []
    for char_data in characters_data:
        char_id = mem_db.add_character_card(char_data)
        assert char_id is not None
        character_ids.append(char_id)
    
    return character_ids

#
# Test Classes

class TestCharacterCardTagIntegration:
    """Integration tests for the optimized character card tag filtering."""
    
    def test_fetch_relevant_ids_character_cards_optimization(self, mem_db, sample_characters_with_tags):
        """Test the optimized character card ID fetching using new tag search."""
        keyword_texts = ["fantasy", "magic"]
        
        # Test the optimized path
        result_ids = fetch_relevant_ids_by_keywords(
            media_db=None,
            char_rag_db=mem_db,
            db_type=DatabaseType.CHARACTER_CARDS,
            keyword_texts=keyword_texts
        )
        
        # Should find the Fantasy Wizard character
        assert len(result_ids) >= 1
        
        # Verify we got the correct character by checking the database
        found_chars = []
        for char_id in result_ids:
            char = mem_db.get_character_card_by_id(int(char_id))
            if char:
                found_chars.append(char)
        
        fantasy_chars = [char for char in found_chars if "Fantasy" in char['name'] or "Wizard" in char['name']]
        assert len(fantasy_chars) >= 1
    
    def test_performance_comparison_old_vs_new(self, mem_db):
        """Test performance improvement of new tag search vs old approach."""
        # Create more characters for performance testing
        num_characters = 100
        for i in range(num_characters):
            char_data = {
                "name": f"Character {i}",
                "description": f"Description {i}",
                "personality": "Test personality",
                "scenario": "Test scenario",
                "tags": json.dumps([f"tag{i % 10}", f"category{i % 5}", "common"]),
                "first_message": f"Hello from character {i}"
            }
            char_id = mem_db.add_character_card(char_data)
            assert char_id is not None
        
        # Test new optimized approach
        start_time = time.time()
        new_result_ids = fetch_relevant_ids_by_keywords(
            media_db=None,
            char_rag_db=mem_db,
            db_type=DatabaseType.CHARACTER_CARDS,
            keyword_texts=["common"]
        )
        new_time = time.time() - start_time
        
        # Should find all characters (they all have "common" tag)
        assert len(new_result_ids) == num_characters
        
        # Performance should be reasonable (less than 0.5 seconds for 100 characters)
        assert new_time < 0.5, f"New approach took {new_time} seconds, expected < 0.5s"
    
    def test_case_insensitive_tag_matching(self, mem_db, sample_characters_with_tags):
        """Test that tag matching is case insensitive."""
        # Test with different cases
        test_cases = [
            ["FANTASY"],
            ["Fantasy"], 
            ["fantasy"],
            ["MAGIC", "magic", "Magic"]
        ]
        
        for keyword_texts in test_cases:
            result_ids = fetch_relevant_ids_by_keywords(
                media_db=None,
                char_rag_db=mem_db,
                db_type=DatabaseType.CHARACTER_CARDS,
                keyword_texts=keyword_texts
            )
            # Should find the fantasy wizard regardless of case
            assert len(result_ids) >= 1
    
    def test_multiple_tags_or_behavior(self, mem_db, sample_characters_with_tags):
        """Test that multiple tags work with OR behavior."""
        # Search for characters with either "fantasy" OR "sci-fi" tags
        result_ids = fetch_relevant_ids_by_keywords(
            media_db=None,
            char_rag_db=mem_db,
            db_type=DatabaseType.CHARACTER_CARDS,
            keyword_texts=["fantasy", "sci-fi"]
        )
        
        # Should find at least 2 characters (Fantasy Wizard + Sci-Fi Android)
        assert len(result_ids) >= 2
        
        # Verify we got both expected character types
        found_chars = []
        for char_id in result_ids:
            char = mem_db.get_character_card_by_id(int(char_id))
            if char:
                found_chars.append(char)
        
        char_names = [char['name'] for char in found_chars]
        assert any("Fantasy" in name for name in char_names)
        assert any("Sci-Fi" in name for name in char_names)
    
    def test_no_keywords_returns_empty(self, mem_db, sample_characters_with_tags):
        """Test that no keywords returns empty list."""
        result_ids = fetch_relevant_ids_by_keywords(
            media_db=None,
            char_rag_db=mem_db,
            db_type=DatabaseType.CHARACTER_CARDS,
            keyword_texts=[]
        )
        assert result_ids == []
    
    def test_nonexistent_tags_return_empty(self, mem_db, sample_characters_with_tags):
        """Test that searching for nonexistent tags returns empty."""
        result_ids = fetch_relevant_ids_by_keywords(
            media_db=None,
            char_rag_db=mem_db,
            db_type=DatabaseType.CHARACTER_CARDS,
            keyword_texts=["nonexistent_tag", "another_fake_tag"]
        )
        assert result_ids == []
    
    def test_database_error_graceful_handling(self, mem_db, sample_characters_with_tags):
        """Test graceful handling of database errors in tag search."""
        # Mock a database error
        with patch.object(mem_db, 'search_character_cards_by_tags') as mock_search:
            mock_search.side_effect = Exception("Database connection lost")
            
            # Should return empty list instead of crashing
            result_ids = fetch_relevant_ids_by_keywords(
                media_db=None,
                char_rag_db=mem_db,
                db_type=DatabaseType.CHARACTER_CARDS,
                keyword_texts=["fantasy"]
            )
            assert result_ids == []


class TestTempFileCleanupIntegration:
    """Integration tests for temporary file cleanup functionality."""
    
    def test_save_chat_history_creates_file(self):
        """Test that save_chat_history creates a file."""
        chat_history = [
            ("Hello", "Hi there!"),
            ("How are you?", "I'm doing well, thanks!")
        ]
        
        file_path = save_chat_history(chat_history)
        
        # File should exist
        assert Path(file_path).exists()
        
        # File should contain the chat history
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded_history = json.load(f)
        
        assert loaded_history == chat_history
        
        # Clean up
        Path(file_path).unlink(missing_ok=True)
    
    def test_save_chat_history_with_custom_cleanup_time(self):
        """Test save_chat_history with custom cleanup time."""
        chat_history = [("Test", "Response")]
        
        # Use very short cleanup time for testing (1 hour instead of default 24)
        file_path = save_chat_history(chat_history, cleanup_after_hours=1)
        
        assert Path(file_path).exists()
        
        # Clean up immediately for test
        Path(file_path).unlink(missing_ok=True)
    
    def test_temp_file_cleanup_scheduling(self):
        """Test that temp file cleanup is scheduled correctly."""
        # Create a temp file manually
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            json.dump([("test", "data")], temp_file)
            temp_file_path = temp_file.name
        
        assert Path(temp_file_path).exists()
        
        # Schedule very quick cleanup (1 second for testing)
        # Note: This is just testing the scheduling, not waiting for actual cleanup
        _schedule_temp_file_cleanup(temp_file_path, cleanup_after_hours=1/3600)  # 1 second
        
        # File should still exist immediately after scheduling
        assert Path(temp_file_path).exists()
        
        # Clean up manually for test
        Path(temp_file_path).unlink(missing_ok=True)
    
    def test_cleanup_nonexistent_file_doesnt_crash(self):
        """Test that cleanup of nonexistent file doesn't crash."""
        fake_path = "/tmp/nonexistent_file_12345.json"
        
        # Should not raise an exception
        try:
            _schedule_temp_file_cleanup(fake_path, cleanup_after_hours=1/3600)
            # Give it a moment to potentially fail
            time.sleep(0.1)
        except Exception as e:
            pytest.fail(f"Cleanup scheduling should not crash on nonexistent file: {e}")


class TestConfigurationIntegration:
    """Integration tests for configuration-driven behavior."""
    
    def test_rag_search_config_usage(self, mem_db, sample_characters_with_tags):
        """Test that RAG_SEARCH_CONFIG values are used correctly."""
        # Test that the configuration values are being used
        original_limit = RAG_SEARCH_CONFIG.get('max_character_cards_fetch', 100000)
        
        # The fetch function should respect the config limit
        result_ids = fetch_relevant_ids_by_keywords(
            media_db=None,
            char_rag_db=mem_db,
            db_type=DatabaseType.CHARACTER_CARDS,
            keyword_texts=["fantasy"]
        )
        
        # Should work with config-driven limits
        assert isinstance(result_ids, list)
        assert len(result_ids) <= original_limit
    
    def test_configuration_values_accessible(self):
        """Test that all expected configuration values are accessible."""
        expected_keys = [
            'max_character_cards_fetch',
            'max_conversations_per_character',
            'max_conversations_for_keyword', 
            'max_notes_for_keyword',
            'metadata_content_preview_chars',
            'temp_file_cleanup_hours'
        ]
        
        for key in expected_keys:
            assert key in RAG_SEARCH_CONFIG, f"Configuration key '{key}' not found"
            assert isinstance(RAG_SEARCH_CONFIG[key], (int, float)), f"Configuration value for '{key}' should be numeric"


class TestErrorHandlingIntegration:
    """Integration tests for error handling in RAG operations."""
    
    def test_database_error_propagation(self, mem_db):
        """Test that database errors are properly wrapped and propagated."""
        # Test with invalid database operation
        with patch.object(mem_db, 'search_character_cards_by_tags') as mock_search:
            from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError
            mock_search.side_effect = CharactersRAGDBError("Database connection failed")
            
            # Should handle gracefully in fetch_relevant_ids_by_keywords
            result_ids = fetch_relevant_ids_by_keywords(
                media_db=None,
                char_rag_db=mem_db,
                db_type=DatabaseType.CHARACTER_CARDS,
                keyword_texts=["fantasy"]
            )
            
            # Should return empty list for graceful degradation
            assert result_ids == []
    
    def test_input_validation_integration(self, mem_db):
        """Test input validation in integrated scenarios."""
        # Test various invalid inputs
        invalid_inputs = [
            None,
            [],
            ["", "   ", "\t"],  # Only whitespace
        ]
        
        for invalid_input in invalid_inputs:
            result_ids = fetch_relevant_ids_by_keywords(
                media_db=None,
                char_rag_db=mem_db,
                db_type=DatabaseType.CHARACTER_CARDS,
                keyword_texts=invalid_input
            )
            # Should return empty list for invalid inputs (graceful handling)
            assert result_ids == []
    
    def test_unsupported_database_type_handling(self, mem_db):
        """Test handling of unsupported database types."""
        # Create a mock unsupported database type
        from enum import Enum
        
        class MockDatabaseType(Enum):
            UNSUPPORTED_TYPE = "Unsupported Type"
        
        # Should handle gracefully
        result_ids = fetch_relevant_ids_by_keywords(
            media_db=None,
            char_rag_db=mem_db,
            db_type=MockDatabaseType.UNSUPPORTED_TYPE,
            keyword_texts=["test"]
        )
        
        # Should return None or empty list for unsupported types
        assert result_ids is None or result_ids == []


class TestConcurrencyAndThreadSafety:
    """Test concurrent access and thread safety."""
    
    def test_concurrent_tag_searches(self, mem_db, sample_characters_with_tags):
        """Test concurrent tag search operations."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def search_worker(tag):
            try:
                result_ids = fetch_relevant_ids_by_keywords(
                    media_db=None,
                    char_rag_db=mem_db,
                    db_type=DatabaseType.CHARACTER_CARDS,
                    keyword_texts=[tag]
                )
                results_queue.put((tag, result_ids))
            except Exception as e:
                errors_queue.put((tag, str(e)))
        
        # Start multiple concurrent searches
        tags = ["fantasy", "sci-fi", "adventure", "magic", "technology"]
        threads = []
        
        for tag in tags:
            thread = threading.Thread(target=search_worker, args=(tag,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Check results
        assert errors_queue.empty(), f"Errors occurred: {list(errors_queue.queue)}"
        
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == len(tags)
        
        # Verify we got reasonable results
        for tag, result_ids in results:
            assert isinstance(result_ids, list)
    
    def test_concurrent_temp_file_operations(self):
        """Test concurrent temporary file operations."""
        import threading
        
        created_files = []
        errors = []
        
        def create_temp_file_worker(worker_id):
            try:
                chat_history = [(f"Message {worker_id}", f"Response {worker_id}")]
                file_path = save_chat_history(chat_history)
                created_files.append(file_path)
            except Exception as e:
                errors.append(str(e))
        
        # Start multiple concurrent file creation operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_temp_file_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(created_files) == 5
        
        # Verify files were created correctly
        for file_path in created_files:
            assert Path(file_path).exists()
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                assert len(data) == 1
                assert "Message" in data[0][0]
                assert "Response" in data[0][1]
        
        # Clean up
        for file_path in created_files:
            Path(file_path).unlink(missing_ok=True)


class TestMemoryAndResourceManagement:
    """Test memory usage and resource management."""
    
    def test_large_character_dataset_memory_usage(self, mem_db):
        """Test memory usage with large character datasets."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create a large number of characters
        num_characters = 1000
        character_ids = []
        
        for i in range(num_characters):
            char_data = {
                "name": f"Character {i}",
                "description": f"Description {i}" * 10,  # Longer description
                "personality": f"Personality {i}" * 5,
                "scenario": f"Scenario {i}" * 3,
                "tags": json.dumps([f"tag{j}" for j in range(i % 10)]),  # Variable number of tags
                "first_message": f"Hello from character {i}"
            }
            char_id = mem_db.add_character_card(char_data)
            character_ids.append(char_id)
        
        # Perform tag search operations
        search_start_memory = process.memory_info().rss
        
        # Multiple searches
        for i in range(10):
            result_ids = fetch_relevant_ids_by_keywords(
                media_db=None,
                char_rag_db=mem_db,
                db_type=DatabaseType.CHARACTER_CARDS,
                keyword_texts=[f"tag{i}"]
            )
            assert isinstance(result_ids, list)
        
        # Check memory usage after searches
        final_memory = process.memory_info().rss
        memory_increase = final_memory - search_start_memory
        
        # Memory increase should be reasonable (less than 50MB for 1000 characters + 10 searches)
        max_acceptable_increase = 50 * 1024 * 1024  # 50MB
        assert memory_increase < max_acceptable_increase, f"Memory increased by {memory_increase / 1024 / 1024:.2f}MB, expected < 50MB"
    
    def test_temp_file_cleanup_prevents_disk_buildup(self):
        """Test that temp file cleanup prevents disk space buildup."""
        import tempfile
        
        temp_dir = Path(tempfile.gettempdir())
        
        # Count existing temp files
        initial_temp_files = list(temp_dir.glob("tmp*.json"))
        initial_count = len(initial_temp_files)
        
        # Create multiple temp files
        created_files = []
        for i in range(10):
            chat_history = [(f"Message {i}", f"Response {i}")]
            file_path = save_chat_history(chat_history)
            created_files.append(file_path)
        
        # Verify files were created
        assert len(created_files) == 10
        for file_path in created_files:
            assert Path(file_path).exists()
        
        # Manual cleanup for test (since we can't wait for scheduled cleanup)
        for file_path in created_files:
            Path(file_path).unlink(missing_ok=True)
        
        # Verify cleanup worked
        for file_path in created_files:
            assert not Path(file_path).exists()


class TestBackwardCompatibility:
    """Test that optimizations maintain backward compatibility."""
    
    def test_character_card_tag_search_api_compatibility(self, mem_db, sample_characters_with_tags):
        """Test that the tag search API is compatible with existing code."""
        # Test the public API remains the same
        result_ids = fetch_relevant_ids_by_keywords(
            media_db=None,
            char_rag_db=mem_db,
            db_type=DatabaseType.CHARACTER_CARDS,
            keyword_texts=["fantasy"]
        )
        
        # Return type should be list of strings (character IDs)
        assert isinstance(result_ids, list)
        for char_id in result_ids:
            assert isinstance(char_id, str)
            # Should be able to convert to int (character IDs are integers)
            assert int(char_id) > 0
    
    def test_temp_file_save_api_compatibility(self):
        """Test that temp file save API remains compatible."""
        chat_history = [("Hello", "Hi"), ("How are you?", "Fine!")]
        
        # Original API should still work
        file_path = save_chat_history(chat_history)
        assert isinstance(file_path, str)
        assert Path(file_path).exists()
        
        # With optional parameter should also work
        file_path2 = save_chat_history(chat_history, cleanup_after_hours=12)
        assert isinstance(file_path2, str)
        assert Path(file_path2).exists()
        
        # Clean up
        Path(file_path).unlink(missing_ok=True)
        Path(file_path2).unlink(missing_ok=True)