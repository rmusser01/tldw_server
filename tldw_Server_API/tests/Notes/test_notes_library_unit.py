# Tests/test_notes_library_unit.py
#
#
# Imports
import unittest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import tempfile
import shutil
import logging
import sqlite3  # For mocking sqlite3.Error
#
# Third-party imports
#
# local imports
from tldw_Server_API.app.core.Notes.Notes_Library import NotesInteropService
#
#
########################################################################################################################
# Import exceptions that NotesInteropService might raise or handle,
# assuming they are defined in ChaChaNotes_DB.py
# If using dummy exceptions from conftest.py, adjust imports or rely on mock's side_effect.
# For robust mocking, it's best to mock the actual path of these exceptions.
MODULE_PATH_PREFIX = "tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB"
CHARACHERS_RAGDB_PATH = f"{MODULE_PATH_PREFIX}.CharactersRAGDB"
CHARACHERS_RAGDB_ERROR_PATH = f"{MODULE_PATH_PREFIX}.CharactersRAGDBError"
SCHEMA_ERROR_PATH = f"{MODULE_PATH_PREFIX}.SchemaError"
INPUT_ERROR_PATH = f"{MODULE_PATH_PREFIX}.InputError"
CONFLICT_ERROR_PATH = f"{MODULE_PATH_PREFIX}.ConflictError"

# Mock the actual exception classes from their source module
# This assumes these exceptions are indeed defined in the specified module.
# If not, the `create=True` argument in `@patch` would create them as MagicMocks.
MockCharactersRAGDBError = type('CharactersRAGDBError', (Exception,), {})
MockSchemaError = type('SchemaError', (MockCharactersRAGDBError,), {})
MockInputError = type('InputError', (MockCharactersRAGDBError,), {})
MockConflictError = type('ConflictError', (MockCharactersRAGDBError,), {'entity': None, 'entity_id': None})


class TestNotesInteropService(unittest.TestCase):

    def setUp(self):
        self.temp_dir_obj = tempfile.TemporaryDirectory(prefix="notes_service_test_")
        self.base_db_dir = Path(self.temp_dir_obj.name)
        self.api_client_id = "test_api_client_v1"

        # Patch CharactersRAGDB and logging for all tests in this class
        self.mock_ragdb_class_patcher = patch(CHARACHERS_RAGDB_PATH, spec=True)
        self.MockCharactersRAGDB = self.mock_ragdb_class_patcher.start()

        self.mock_logger_patcher = patch('tldw_Server_API.Notes_Library.logger', spec=True)
        self.mock_logger = self.mock_logger_patcher.start()

        # Mocks for specific exception types from the target module
        # This ensures that when the code does `except OriginalInputError`, it can be caught.
        self.patchers = {
            'InputError': patch(INPUT_ERROR_PATH, MockInputError, create=True),
            'ConflictError': patch(CONFLICT_ERROR_PATH, MockConflictError, create=True),
            'CharactersRAGDBError': patch(CHARACHERS_RAGDB_ERROR_PATH, MockCharactersRAGDBError, create=True),
            'SchemaError': patch(SCHEMA_ERROR_PATH, MockSchemaError, create=True),
        }
        for p in self.patchers.values():
            p.start()

        self.service = NotesInteropService(base_db_directory=str(self.base_db_dir),
                                           api_client_id=self.api_client_id)
        self.mock_db_instance = MagicMock()
        self.MockCharactersRAGDB.return_value = self.mock_db_instance

    def tearDown(self):
        self.service.close_all_user_connections()  # Test this method too
        self.temp_dir_obj.cleanup()
        self.mock_ragdb_class_patcher.stop()
        self.mock_logger_patcher.stop()
        for p in self.patchers.values():
            p.stop()

    def test_initialization(self):
        self.assertTrue(self.base_db_dir.exists())
        self.assertEqual(self.service.api_client_id, self.api_client_id)
        self.mock_logger.info.assert_any_call(
            f"NotesInteropService initialized. Base DB directory: {self.base_db_dir.resolve()}")

    @patch('tldw_Server_API.Notes_Library.Path.mkdir')
    def test_initialization_failure_os_error(self, mock_mkdir):
        mock_mkdir.side_effect = OSError("Permission denied")
        # Since self.service is created in setUp, we need to test this scenario differently
        # or re-initialize here. Let's re-initialize.
        with self.assertRaises(MockCharactersRAGDBError) as cm:  # Use the patched error type
            NotesInteropService(base_db_directory=str(self.base_db_dir), api_client_id="fail_client")
        self.assertIn("Failed to create base DB directory", str(cm.exception))
        self.mock_logger.error.assert_called_with(
            f"Failed to create base DB directory {Path(self.base_db_dir).resolve()}: Permission denied"
        )

    def test_get_db_new_instance(self):
        user_id = "user1"
        db_instance = self.service._get_db(user_id)
        self.MockCharactersRAGDB.assert_called_once_with(
            db_path=self.base_db_dir / f"user_{user_id}.sqlite",
            client_id=self.api_client_id
        )
        self.assertIs(db_instance, self.mock_db_instance)
        self.assertIn(user_id, self.service._db_instances)

    def test_get_db_cached_instance(self):
        user_id = "user1"
        self.service._get_db(user_id)  # First call
        self.MockCharactersRAGDB.reset_mock()  # Reset for the second call check

        db_instance_cached = self.service._get_db(user_id)  # Second call
        self.MockCharactersRAGDB.assert_not_called()  # Should not create new
        self.assertIs(db_instance_cached, self.mock_db_instance)

    def test_get_db_invalid_user_id(self):
        with self.assertRaises(ValueError):
            self.service._get_db("")
        with self.assertRaises(ValueError):
            self.service._get_db("   ")
        with self.assertRaises(ValueError):
            self.service._get_db(None)  # type: ignore

    def test_get_db_init_failure_ragdb_error(self):
        self.MockCharactersRAGDB.side_effect = MockCharactersRAGDBError("DB init failed")
        user_id = "user_fail"
        with self.assertRaises(MockCharactersRAGDBError):
            self.service._get_db(user_id)
        self.mock_logger.error.assert_called_once()

    def test_get_db_init_failure_sqlite_error(self):
        self.MockCharactersRAGDB.side_effect = sqlite3.Error("SQLite connection failed")
        user_id = "user_sqlite_fail"
        with self.assertRaises(sqlite3.Error):  # It should re-raise sqlite3.Error
            self.service._get_db(user_id)
        self.mock_logger.error.assert_called_once()

    def test_get_db_init_failure_unexpected_error(self):
        self.MockCharactersRAGDB.side_effect = Exception("Unexpected boom")
        user_id = "user_generic_fail"
        with self.assertRaises(MockCharactersRAGDBError) as cm:  # Wraps in CharactersRAGDBError
            self.service._get_db(user_id)
        self.assertIn("Unexpected error initializing DB", str(cm.exception))
        self.mock_logger.error.assert_called_once()

    # --- Note Methods ---
    def test_add_note(self):
        user_id = "user1"
        title, content = "Test Note", "Test Content"
        expected_note_id = "note_uuid_1"
        self.mock_db_instance.add_note.return_value = expected_note_id

        note_id = self.service.add_note(user_id, title, content)

        self.service._get_db(user_id)  # Ensure _get_db was called
        self.mock_db_instance.add_note.assert_called_once_with(title=title, content=content, note_id=None)
        self.assertEqual(note_id, expected_note_id)

    def test_add_note_with_provided_id(self):
        user_id = "user1"
        title, content, provided_note_id = "Test Note", "Test Content", "client_note_id"
        self.mock_db_instance.add_note.return_value = provided_note_id

        note_id = self.service.add_note(user_id, title, content, note_id=provided_note_id)

        self.mock_db_instance.add_note.assert_called_once_with(title=title, content=content, note_id=provided_note_id)
        self.assertEqual(note_id, provided_note_id)

    def test_add_note_returns_none_unexpectedly(self):
        user_id = "user1"
        title, content = "Test Note", "Test Content"
        self.mock_db_instance.add_note.return_value = None  # Simulate unexpected None

        with self.assertRaises(MockCharactersRAGDBError) as cm:
            self.service.add_note(user_id, title, content)
        self.assertIn("Failed to create note, received None ID unexpectedly", str(cm.exception))
        self.mock_logger.error.assert_called_once()

    def test_get_note_by_id(self):
        user_id, note_id = "user1", "note_uuid_1"
        expected_data = {"id": note_id, "title": "Test"}
        self.mock_db_instance.get_note_by_id.return_value = expected_data

        note = self.service.get_note_by_id(user_id, note_id)
        self.mock_db_instance.get_note_by_id.assert_called_once_with(note_id=note_id)
        self.assertEqual(note, expected_data)

    def test_list_notes(self):
        user_id = "user1"
        expected_notes = [{"id": "1"}, {"id": "2"}]
        self.mock_db_instance.list_notes.return_value = expected_notes

        notes = self.service.list_notes(user_id, limit=10, offset=0)
        self.mock_db_instance.list_notes.assert_called_once_with(limit=10, offset=0)
        self.assertEqual(notes, expected_notes)

    def test_update_note(self):
        user_id, note_id = "user1", "note_uuid_1"
        update_data = {"title": "New Title"}
        expected_version = 1
        self.mock_db_instance.update_note.return_value = True

        success = self.service.update_note(user_id, note_id, update_data, expected_version)
        self.mock_db_instance.update_note.assert_called_once_with(
            note_id=note_id, update_data=update_data, expected_version=expected_version
        )
        self.assertTrue(success)

    def test_soft_delete_note(self):
        user_id, note_id = "user1", "note_uuid_1"
        expected_version = 2
        self.mock_db_instance.soft_delete_note.return_value = True

        success = self.service.soft_delete_note(user_id, note_id, expected_version)
        self.mock_db_instance.soft_delete_note.assert_called_once_with(
            note_id=note_id, expected_version=expected_version
        )
        self.assertTrue(success)

    def test_search_notes(self):
        user_id, term = "user1", "search term"
        expected_results = [{"id": "1", "content": "Contains search term"}]
        self.mock_db_instance.search_notes.return_value = expected_results

        results = self.service.search_notes(user_id, term, limit=5)
        self.mock_db_instance.search_notes.assert_called_once_with(search_term=term, limit=5)
        self.assertEqual(results, expected_results)

    # --- Keyword and Linking Methods (similar structure to note methods) ---
    def test_add_keyword(self):
        user_id, keyword_text = "user1", "test_keyword"
        expected_keyword_id = 1
        self.mock_db_instance.add_keyword.return_value = expected_keyword_id

        keyword_id = self.service.add_keyword(user_id, keyword_text)
        self.mock_db_instance.add_keyword.assert_called_once_with(keyword_text=keyword_text)
        self.assertEqual(keyword_id, expected_keyword_id)

    def test_link_note_to_keyword(self):
        user_id, note_id, keyword_id = "user1", "note_uuid_1", 1
        self.mock_db_instance.link_note_to_keyword.return_value = True

        success = self.service.link_note_to_keyword(user_id, note_id, keyword_id)
        self.mock_db_instance.link_note_to_keyword.assert_called_once_with(note_id=note_id, keyword_id=keyword_id)
        self.assertTrue(success)

    # ... Add more tests for other keyword and linking methods:
    # get_keyword_by_id, get_keyword_by_text, list_keywords, soft_delete_keyword, search_keywords
    # unlink_note_from_keyword, get_keywords_for_note, get_notes_for_keyword
    # These will follow the same pattern as the note methods above.

    # --- Resource Management ---
    def test_close_user_connection(self):
        user_id = "user1"
        # Ensure DB instance exists for this user
        db_instance = self.service._get_db(user_id)
        self.assertIn(user_id, self.service._db_instances)

        self.service.close_user_connection(user_id)
        db_instance.close_connection.assert_called_once()
        self.assertNotIn(user_id, self.service._db_instances)
        self.mock_logger.info.assert_any_call(f"Closed and removed DB connection for user_id '{user_id}'.")

    def test_close_user_connection_not_exist(self):
        user_id = "non_existent_user"
        self.service.close_user_connection(user_id)
        # No error should be raised, mock_db_instance.close_connection should not be called
        self.mock_db_instance.close_connection.assert_not_called()
        self.mock_logger.debug.assert_any_call(
            f"No active DB connection found in cache for user_id '{user_id}' to close.")

    def test_close_all_user_connections(self):
        user1_id, user2_id = "user1", "user2"
        db_instance1 = self.service._get_db(user1_id)
        db_instance2 = self.service._get_db(user2_id)  # This will re-use mock_db_instance if not careful

        # To test multiple instances, we need MockCharactersRAGDB to return different mocks
        mock_db_1 = MagicMock()
        mock_db_2 = MagicMock()
        self.MockCharactersRAGDB.side_effect = [mock_db_1, mock_db_2]

        # Re-setup service with this new side_effect for this specific test
        self.service = NotesInteropService(base_db_directory=str(self.base_db_dir),
                                           api_client_id=self.api_client_id)
        db_instance1_new = self.service._get_db(user1_id)
        db_instance2_new = self.service._get_db(user2_id)
        self.assertIs(db_instance1_new, mock_db_1)
        self.assertIs(db_instance2_new, mock_db_2)

        self.service.close_all_user_connections()
        mock_db_1.close_connection.assert_called_once()
        mock_db_2.close_connection.assert_called_once()
        self.assertEqual(len(self.service._db_instances), 0)
        self.mock_logger.info.assert_any_call("All cached user DB connections have been processed for closure.")

    def test_close_connection_exception(self):
        user_id = "user_close_fail"
        db_instance = self.service._get_db(user_id)
        db_instance.close_connection.side_effect = Exception("Failed to close")

        self.service.close_user_connection(user_id)  # Should not raise, but log error
        self.assertNotIn(user_id, self.service._db_instances)  # Still removed from cache
        self.mock_logger.error.assert_called_with(
            f"Error closing DB connection for user_id '{user_id}': Failed to close", exc_info=True
        )


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

#
# End of test_notes_library_unit.py
########################################################################################################################