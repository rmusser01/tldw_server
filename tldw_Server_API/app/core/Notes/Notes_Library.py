# Notes_Library.py
# Description: This module provides a service layer for managing notes and note keywords
#
# Imports
import logging
import threading
import sqlite3  # For exception handling in _get_db
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
#
# Third-Party Imports
#
# Local Imports
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    SchemaError,
    InputError,
    ConflictError
)
#
#######################################################################################################################
#
# Functions:

logger = logging.getLogger(__name__)


class NotesInteropService:
    """
    A service layer for interacting with the notes-specific functionalities
    of CharactersRAGDB, designed for per-user database instances.
    This service manages CharactersRAGDB instances for each user and exposes
    methods for note and related keyword operations.
    """

    def __init__(self, base_db_directory: Union[str, Path], api_client_id: str):
        """
        Initializes the NotesInteropService.

        Args:
            base_db_directory: The base directory where user-specific SQLite database
                               files will be stored (e.g., "data/user_dbs").
                               Each user's DB will be in a sub-file like "user_{user_id}.sqlite".
            api_client_id: A client ID string representing this API application.
                           This ID is passed to CharactersRAGDB instances.
        """
        self.base_db_directory = Path(base_db_directory).resolve()
        self.api_client_id = api_client_id
        self._db_instances: Dict[str, CharactersRAGDB] = {}
        self._db_lock = threading.Lock()  # To protect access to _db_instances

        try:
            self.base_db_directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"NotesInteropService initialized. Base DB directory: {self.base_db_directory}")
        except OSError as e:
            logger.error(f"Failed to create base DB directory {self.base_db_directory}: {e}")
            # This is a critical failure for the service's operation.
            raise CharactersRAGDBError(f"Failed to create base DB directory {self.base_db_directory}: {e}") from e

    def _get_db(self, user_id: str) -> CharactersRAGDB:
        """
        Retrieves or creates a CharactersRAGDB instance for a given user_id.
        Instances are cached for efficiency. This method is thread-safe.

        Args:
            user_id: The unique identifier for the user.

        Returns:
            A CharactersRAGDB instance for the specified user.

        Raises:
            ValueError: If user_id is empty.
            CharactersRAGDBError: If the database initialization fails.
        """
        if not user_id:
            # Ensure user_id is a non-empty string, basic validation
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError("user_id must be a non-empty string.")
            user_id = user_id.strip()

        # Fast path: check if instance already exists without lock
        if user_id in self._db_instances:
            return self._db_instances[user_id]

        # Slow path: acquire lock and double-check
        with self._db_lock:
            if user_id not in self._db_instances:
                db_path = self.base_db_directory / f"user_{user_id}.sqlite"
                logger.info(f"Creating or loading DB for user_id '{user_id}' at path: {db_path}")
                try:
                    db_instance = CharactersRAGDB(db_path=db_path, client_id=self.api_client_id)
                    self._db_instances[user_id] = db_instance
                    logger.info(f"Successfully initialized DB for user_id '{user_id}'.")
                except (CharactersRAGDBError, SchemaError, sqlite3.Error) as e:
                    logger.error(f"Failed to initialize DB for user_id '{user_id}' at {db_path}: {e}", exc_info=True)
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error initializing DB for user_id '{user_id}' at {db_path}: {e}",
                                 exc_info=True)
                    raise CharactersRAGDBError(f"Unexpected error initializing DB for user {user_id}: {e}") from e

            return self._db_instances[user_id]

    # --- Note Methods ---

    def add_note(self, user_id: str, title: str, content: str, note_id: Optional[str] = None) -> str:
        """
        Adds a new note for the specified user.

        Args:
            user_id: The ID of the user.
            title: The title of the note.
            content: The content of the note.
            note_id: Optional UUID for the note. If None, one will be generated.

        Returns:
            The ID of the newly created note.

        Raises:
            InputError, ConflictError, CharactersRAGDBError: From ChaChaDB.
            ValueError: If user_id is invalid.
        """
        db = self._get_db(user_id)
        # ChaChaDB's add_note returns `str | None` in its type hint.
        # Current implementation of ChaChaDB.add_note returns `str` on success or raises.
        created_note_id = db.add_note(title=title, content=content, note_id=note_id)
        if created_note_id is None:
            # This path suggests an issue if add_note could return None without raising.
            logger.error(f"add_note for user {user_id} returned None unexpectedly for title '{title}'.")
            raise CharactersRAGDBError("Failed to create note, received None ID unexpectedly.")
        return created_note_id

    def get_note_by_id(self, user_id: str, note_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific note by its ID for the given user."""
        db = self._get_db(user_id)
        return db.get_note_by_id(note_id=note_id)

    def list_notes(self, user_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Lists notes for the given user."""
        db = self._get_db(user_id)
        return db.list_notes(limit=limit, offset=offset)

    def update_note(self, user_id: str, note_id: str, update_data: Dict[str, Any], expected_version: int) -> bool:
        """Updates a note for the given user with optimistic locking."""
        db = self._get_db(user_id)
        return db.update_note(note_id=note_id, update_data=update_data, expected_version=expected_version)

    def soft_delete_note(self, user_id: str, note_id: str, expected_version: int) -> bool:
        """Soft-deletes a note for the given user with optimistic locking."""
        db = self._get_db(user_id)
        return db.soft_delete_note(note_id=note_id, expected_version=expected_version)

    def search_notes(self, user_id: str, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Searches notes for the given user."""
        db = self._get_db(user_id)
        return db.search_notes(search_term=search_term, limit=limit)

    # --- Note-Keyword Linking Methods ---

    def link_note_to_keyword(self, user_id: str, note_id: str, keyword_id: int) -> bool:
        """Links a note to a keyword for the given user."""
        db = self._get_db(user_id)
        return db.link_note_to_keyword(note_id=note_id, keyword_id=keyword_id)

    def unlink_note_from_keyword(self, user_id: str, note_id: str, keyword_id: int) -> bool:
        """Unlinks a note from a keyword for the given user."""
        db = self._get_db(user_id)
        return db.unlink_note_from_keyword(note_id=note_id, keyword_id=keyword_id)

    def get_keywords_for_note(self, user_id: str, note_id: str) -> List[Dict[str, Any]]:
        """Retrieves all keywords linked to a specific note for the given user."""
        db = self._get_db(user_id)
        return db.get_keywords_for_note(note_id=note_id)

    def get_notes_for_keyword(self, user_id: str, keyword_id: int, limit: int = 50, offset: int = 0) -> List[
        Dict[str, Any]]:
        """Retrieves all notes linked to a specific keyword for the given user."""
        db = self._get_db(user_id)
        return db.get_notes_for_keyword(keyword_id=keyword_id, limit=limit, offset=offset)

    # --- Keyword Methods (as they relate to notes functionality) ---

    def add_keyword(self, user_id: str, keyword_text: str) -> Optional[int]:
        """Adds a new keyword for the specified user. Returns keyword ID."""
        db = self._get_db(user_id)
        return db.add_keyword(keyword_text=keyword_text)

    def get_keyword_by_id(self, user_id: str, keyword_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a keyword by its ID for the given user."""
        db = self._get_db(user_id)
        return db.get_keyword_by_id(keyword_id=keyword_id)

    def get_keyword_by_text(self, user_id: str, keyword_text: str) -> Optional[Dict[str, Any]]:
        """Retrieves a keyword by its text for the given user."""
        db = self._get_db(user_id)
        return db.get_keyword_by_text(keyword_text=keyword_text)

    def list_keywords(self, user_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Lists keywords for the given user."""
        db = self._get_db(user_id)
        return db.list_keywords(limit=limit, offset=offset)

    def soft_delete_keyword(self, user_id: str, keyword_id: int, expected_version: int) -> bool:
        """Soft-deletes a keyword for the given user with optimistic locking."""
        db = self._get_db(user_id)
        return db.soft_delete_keyword(keyword_id=keyword_id, expected_version=expected_version)

    def search_keywords(self, user_id: str, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Searches keywords for the given user."""
        db = self._get_db(user_id)
        return db.search_keywords(search_term=search_term, limit=limit)

    # --- Resource Management ---

    def close_all_user_connections(self):
        """
        Closes all cached database connections.
        Useful for application shutdown.
        """
        with self._db_lock:
            logger.info(f"Closing all {len(self._db_instances)} cached user DB connections.")
            for user_id, db_instance in self._db_instances.items():
                try:
                    db_instance.close_connection()
                    logger.debug(f"Closed DB connection for user_id '{user_id}'.")
                except Exception as e:
                    logger.error(f"Error closing DB connection for user_id '{user_id}': {e}", exc_info=True)
            self._db_instances.clear()
        logger.info("All cached user DB connections have been processed for closure.")

    def close_user_connection(self, user_id: str):
        """
        Closes and removes a specific user's database connection from the cache.
        Useful if a user session ends and resources should be freed.
        """
        with self._db_lock:
            if user_id in self._db_instances:
                db_instance = self._db_instances.pop(user_id)  # Remove from cache
                try:
                    db_instance.close_connection()
                    logger.info(f"Closed and removed DB connection for user_id '{user_id}'.")
                except Exception as e:
                    logger.error(f"Error closing DB connection for user_id '{user_id}': {e}", exc_info=True)
                    # Even if close fails, it's removed from cache.
            else:
                logger.debug(f"No active DB connection found in cache for user_id '{user_id}' to close.")

#
# # Example Usage (Conceptual - typically this would be in your API layer)
# if __name__ == "__main__":
#     # Basic logging setup for the example
#     logging.basicConfig(
#         level=logging.DEBUG,  # Set to INFO for less verbosity from ChaChaDB
#         format='%(asctime)s - %(levelname)s - %(name)s - %(threadName)s - %(message)s'
#     )
#
#     # Create a temporary directory for DBs for this example run
#     import tempfile
#     import shutil
#
#     temp_db_dir_obj = tempfile.TemporaryDirectory(prefix="chachadb_interop_test_")
#     temp_db_dir = temp_db_dir_obj.name
#     print(f"Temporary DB directory: {temp_db_dir}")
#
#     notes_service_instance = None  # Define for finally block
#
#     try:
#         # Initialize the service
#         notes_service_instance = NotesInteropService(
#             base_db_directory=temp_db_dir,
#             api_client_id="example_api_v1.0"
#         )
#
#         user1_id = "user123"
#         user2_id = "user456"
#
#         # --- User 1 operations ---
#         print(f"\n--- Operations for {user1_id} ---")
#         try:
#             note1_title = "User1's First Note"
#             note1_content = "This is the first note for user1."
#             u1_note1_id = notes_service_instance.add_note(user_id=user1_id, title=note1_title, content=note1_content)
#             print(f"Added note for {user1_id}: ID={u1_note1_id}, Title='{note1_title}'")
#
#             retrieved_note = notes_service_instance.get_note_by_id(user_id=user1_id, note_id=u1_note1_id)
#             print(f"Retrieved note for {user1_id}: {retrieved_note}")
#             assert retrieved_note is not None and retrieved_note['title'] == note1_title
#
#             kw1_text = "important"
#             u1_kw1_id = notes_service_instance.add_keyword(user_id=user1_id, keyword_text=kw1_text)
#             print(f"Added keyword for {user1_id}: ID={u1_kw1_id}, Text='{kw1_text}'")
#             assert u1_kw1_id is not None
#
#             if u1_note1_id and u1_kw1_id:
#                 link_success = notes_service_instance.link_note_to_keyword(user_id=user1_id, note_id=u1_note1_id,
#                                                                            keyword_id=u1_kw1_id)
#                 print(f"Link note to keyword success: {link_success}")
#                 assert link_success
#                 keywords_for_note = notes_service_instance.get_keywords_for_note(user_id=user1_id, note_id=u1_note1_id)
#                 print(f"Keywords for note {u1_note1_id}: {keywords_for_note}")
#                 assert len(keywords_for_note) == 1 and keywords_for_note[0]['id'] == u1_kw1_id
#
#             user1_notes = notes_service_instance.list_notes(user_id=user1_id)
#             print(f"Notes for {user1_id}: {user1_notes}")
#             assert len(user1_notes) == 1
#
#             update_success = notes_service_instance.update_note(
#                 user_id=user1_id, note_id=u1_note1_id,
#                 update_data={"content": "Updated content for user1's first note."}, expected_version=1
#             )
#             print(f"Update note success: {update_success}")
#             assert update_success
#             updated_note = notes_service_instance.get_note_by_id(user_id=user1_id, note_id=u1_note1_id)
#             print(f"Updated note: {updated_note}")
#             assert updated_note['content'] == "Updated content for user1's first note." and updated_note['version'] == 2
#
#             try:
#                 notes_service_instance.update_note(
#                     user_id=user1_id, note_id=u1_note1_id,
#                     update_data={"content": "Another update attempt."}, expected_version=1  # Wrong version
#                 )
#             except ConflictError as ce:
#                 print(f"Caught expected ConflictError for optimistic lock: {ce}")
#
#         except (InputError, ConflictError, CharactersRAGDBError, ValueError) as e:
#             print(f"Error during {user1_id} operations: {e}")
#
#         # --- User 2 operations (to show DB separation) ---
#         print(f"\n--- Operations for {user2_id} ---")
#         try:
#             note2_title = "User2's Special Note"
#             u2_note1_id = notes_service_instance.add_note(user_id=user2_id, title=note2_title,
#                                                           content="Content for user2.")
#             print(f"Added note for {user2_id}: ID={u2_note1_id}, Title='{note2_title}'")
#
#             user2_notes = notes_service_instance.list_notes(user_id=user2_id)
#             print(f"Notes for {user2_id}: {user2_notes}")
#             assert len(user2_notes) == 1 and user2_notes[0]['id'] == u2_note1_id
#
#             user1_notes_recheck = notes_service_instance.list_notes(user_id=user1_id)
#             print(f"Notes for {user1_id} (recheck): {user1_notes_recheck}")
#             assert len(user1_notes_recheck) == 1 and user1_notes_recheck[0]['id'] == u1_note1_id
#
#         except (InputError, ConflictError, CharactersRAGDBError, ValueError) as e:
#             print(f"Error during {user2_id} operations: {e}")
#
#         # --- Search example ---
#         print(f"\n--- Search for {user1_id} ---")
#         search_results_u1 = notes_service_instance.search_notes(user_id=user1_id, search_term="User1")
#         print(f"Search for 'User1' in {user1_id}'s notes: {search_results_u1}")
#         assert len(search_results_u1) > 0 and search_results_u1[0]['id'] == u1_note1_id
#
#         # --- Soft delete example ---
#         print(f"\n--- Soft delete for {user1_id} ---")
#         note_to_delete = notes_service_instance.get_note_by_id(user_id=user1_id, note_id=u1_note1_id)
#         if note_to_delete:
#             delete_success = notes_service_instance.soft_delete_note(user_id=user1_id, note_id=u1_note1_id,
#                                                                      expected_version=note_to_delete['version'])
#             print(f"Soft delete success: {delete_success}")
#             assert delete_success
#             deleted_note_check = notes_service_instance.get_note_by_id(user_id=user1_id, note_id=u1_note1_id)
#             print(f"Check after delete (should be None): {deleted_note_check}")
#             assert deleted_note_check is None
#             user1_notes_after_delete = notes_service_instance.list_notes(user_id=user1_id)
#             print(f"Notes for {user1_id} after delete: {user1_notes_after_delete}")
#             assert len(user1_notes_after_delete) == 0
#         else:
#             print(f"Note {u1_note1_id} not found for deletion, skipping delete test.")
#
#
#     finally:
#         if notes_service_instance:
#             notes_service_instance.close_all_user_connections()
#
#         temp_db_dir_obj.cleanup()  # Safely removes the temporary directory
#         print(f"\nCleaned up temporary DB directory: {temp_db_dir}")
#
#     print("\nExample script finished.")


#
# End of Notes_Library.py
#######################################################################################################################
