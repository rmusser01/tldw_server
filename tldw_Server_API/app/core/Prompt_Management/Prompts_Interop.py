# prompts_interop.py
#
"""
Prompts Interop Library
-----------------------

This library serves as an intermediary layer between an API endpoint (or other
application logic) and the Prompts_DB_v2 library. It manages a single instance
of the PromptsDatabase and exposes its functionality, promoting decoupling and
centralized database configuration.

Usage:
1. Initialize at application startup:
   `initialize_interop(db_path="your_prompts.db", client_id="your_api_client")`

2. Call interop functions in your application code:
   `prompts = list_prompts(page=1, per_page=10)`

3. (Optional) Clean up at application shutdown:
   `shutdown_interop()`
"""
#
# Imports
import logging
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path

from loguru import logger

#
# Local Imports
from tldw_Server_API.app.core.DB_Management.Prompts_DB import (
    PromptsDatabase,
    DatabaseError,
    SchemaError,
    InputError,
    ConflictError
)
from tldw_Server_API.app.core.DB_Management.Prompts_DB import (
    add_or_update_prompt as db_add_or_update_prompt,
    load_prompt_details_for_ui as db_load_prompt_details_for_ui,
    export_prompt_keywords_to_csv as db_export_prompt_keywords_to_csv,
    view_prompt_keywords_markdown as db_view_prompt_keywords_markdown,
    export_prompts_formatted as db_export_prompts_formatted
)
#
#######################################################################################################################
#
# Functions:

_db_instance: Optional[PromptsDatabase] = None
_db_path_global: Optional[Union[str, Path]] = None
_client_id_global: Optional[str] = None

# --- Initialization and Management ---

def initialize_interop(db_path: Union[str, Path], client_id: str) -> None:
    """
    Initializes the interop library with a single PromptsDatabase instance.
    This should be called once, e.g., at application startup.

    Args:
        db_path: Path to the SQLite database file or ':memory:'.
        client_id: A unique identifier for the client using this database instance.

    Raises:
        ValueError: If db_path or client_id are invalid.
        DatabaseError, SchemaError: If PromptsDatabase initialization fails.
    """
    global _db_instance, _db_path_global, _client_id_global
    if _db_instance is not None:
        logger.warning(
            f"Prompts interop library already initialized with DB: '{_db_path_global}'. "
            f"Re-initializing with DB: '{db_path}', Client ID: '{client_id}'."
        )
        # PromptsDatabase manages thread-local connections; explicit close of old global instance
        # might be complex if other threads are still using it.
        # The PromptsDatabase instance itself doesn't hold a global connection to close here.

    if not db_path: # Pathlib('') is False-y, so this covers empty strings too
        raise ValueError("db_path is required for initialization.")
    if not client_id:
        raise ValueError("client_id is required for initialization.")

    logger.info(f"Initializing Prompts Interop Library. DB Path: {db_path}, Client ID: {client_id}")
    try:
        _db_instance = PromptsDatabase(db_path=db_path, client_id=client_id)
        _db_path_global = db_path
        _client_id_global = client_id
        logger.info("Prompts Interop Library initialized successfully.")
    except (DatabaseError, SchemaError, ValueError) as e:
        logger.critical(f"Failed to initialize PromptsDatabase for interop: {e}", exc_info=True)
        _db_instance = None # Ensure it's None if init fails
        raise

def get_db_instance() -> PromptsDatabase:
    """
    Returns the initialized PromptsDatabase instance.

    Returns:
        The active PromptsDatabase instance.

    Raises:
        RuntimeError: If the library has not been initialized.
    """
    if _db_instance is None:
        msg = "Prompts Interop Library not initialized. Call initialize_interop() first."
        logger.error(msg)
        raise RuntimeError(msg)
    return _db_instance

def is_initialized() -> bool:
    """Checks if the interop library (and thus the DB instance) is initialized."""
    return _db_instance is not None

def shutdown_interop() -> None:
    """
    Cleans up resources. For PromptsDatabase, this primarily means attempting
    to close the database connection for the current thread if one is active.
    Other thread-local connections are managed by their respective threads.
    Resets the global _db_instance to None.
    """
    global _db_instance, _db_path_global, _client_id_global
    if _db_instance:
        logger.info(f"Shutting down Prompts Interop Library for DB: {_db_path_global}.")
        try:
            # This will close the connection for the current thread
            _db_instance.close_connection()
        except Exception as e:
            logger.error(f"Error during Prompts Interop Library shutdown (closing current thread's connection): {e}", exc_info=True)
        _db_instance = None
        _db_path_global = None
        _client_id_global = None
        logger.info("Prompts Interop Library shut down.")
    else:
        logger.info("Prompts Interop Library was not initialized or already shut down.")

# --- Wrapper Functions for PromptsDatabase methods ---

# --- Mutating Methods ---
def add_keyword(keyword_text: str) -> Tuple[Optional[int], Optional[str]]:
    """Adds a keyword. See PromptsDatabase.add_keyword for details."""
    db = get_db_instance()
    return db.add_keyword(keyword_text)

def add_prompt(name: str, author: Optional[str], details: Optional[str],
               system_prompt: Optional[str] = None, user_prompt: Optional[str] = None,
               keywords: Optional[List[str]] = None, overwrite: bool = False
               ) -> Tuple[Optional[int], Optional[str], str]:
    """Adds or updates a prompt. See PromptsDatabase.add_prompt for details."""
    db = get_db_instance()
    return db.add_prompt(name, author, details, system_prompt, user_prompt, keywords, overwrite)

def update_keywords_for_prompt(prompt_id: int, keywords_list: List[str]) -> None:
    """Updates keywords for a specific prompt. See PromptsDatabase.update_keywords_for_prompt for details."""
    db = get_db_instance()
    db.update_keywords_for_prompt(prompt_id, keywords_list)

def soft_delete_prompt(prompt_id_or_name_or_uuid: Union[int, str]) -> bool:
    """Soft deletes a prompt. See PromptsDatabase.soft_delete_prompt for details."""
    db = get_db_instance()
    return db.soft_delete_prompt(prompt_id_or_name_or_uuid)

def soft_delete_keyword(keyword_text: str) -> bool:
    """Soft deletes a keyword. See PromptsDatabase.soft_delete_keyword for details."""
    db = get_db_instance()
    return db.soft_delete_keyword(keyword_text)

# --- Read Methods ---
def get_prompt_by_id(prompt_id: int, include_deleted: bool = False) -> Optional[Dict]:
    """Fetches a prompt by its ID. See PromptsDatabase.get_prompt_by_id for details."""
    db = get_db_instance()
    return db.get_prompt_by_id(prompt_id, include_deleted)

def get_prompt_by_uuid(prompt_uuid: str, include_deleted: bool = False) -> Optional[Dict]:
    """Fetches a prompt by its UUID. See PromptsDatabase.get_prompt_by_uuid for details."""
    db = get_db_instance()
    return db.get_prompt_by_uuid(prompt_uuid, include_deleted)

def get_prompt_by_name(name: str, include_deleted: bool = False) -> Optional[Dict]:
    """Fetches a prompt by its name. See PromptsDatabase.get_prompt_by_name for details."""
    db = get_db_instance()
    return db.get_prompt_by_name(name, include_deleted)

def list_prompts(page: int = 1, per_page: int = 10, include_deleted: bool = False
                 ) -> Tuple[List[Dict], int, int, int]:
    """Lists prompts with pagination. See PromptsDatabase.list_prompts for details."""
    db = get_db_instance()
    return db.list_prompts(page, per_page, include_deleted)

def fetch_prompt_details(prompt_id_or_name_or_uuid: Union[int, str], include_deleted: bool = False
                         ) -> Optional[Dict]:
    """Fetches detailed information for a prompt. See PromptsDatabase.fetch_prompt_details for details."""
    db = get_db_instance()
    return db.fetch_prompt_details(prompt_id_or_name_or_uuid, include_deleted)

def fetch_all_keywords(include_deleted: bool = False) -> List[str]:
    """Fetches all keywords. See PromptsDatabase.fetch_all_keywords for details."""
    db = get_db_instance()
    return db.fetch_all_keywords(include_deleted)

def fetch_keywords_for_prompt(prompt_id: int, include_deleted: bool = False) -> List[str]:
    """Fetches keywords associated with a specific prompt. See PromptsDatabase.fetch_keywords_for_prompt for details."""
    db = get_db_instance()
    return db.fetch_keywords_for_prompt(prompt_id, include_deleted)

def search_prompts(search_query: Optional[str],
                   search_fields: Optional[List[str]] = None,
                   page: int = 1,
                   results_per_page: int = 20,
                   include_deleted: bool = False
                   ) -> Tuple[List[Dict[str, Any]], int]:
    """Searches prompts using FTS. See PromptsDatabase.search_prompts for details."""
    db = get_db_instance()
    return db.search_prompts(search_query, search_fields, page, results_per_page, include_deleted)

# --- Sync Log Access Methods ---
def get_sync_log_entries(since_change_id: int = 0, limit: Optional[int] = None) -> List[Dict]:
    """Retrieves entries from the sync log. See PromptsDatabase.get_sync_log_entries for details."""
    db = get_db_instance()
    return db.get_sync_log_entries(since_change_id, limit)

def delete_sync_log_entries(change_ids: List[int]) -> int:
    """Deletes entries from the sync log. See PromptsDatabase.delete_sync_log_entries for details."""
    db = get_db_instance()
    return db.delete_sync_log_entries(change_ids)


# --- Wrappers for Standalone Functions from Prompts_DB_v2 ---
# These functions from Prompts_DB_v2.py originally took a db_instance.
# Here, they use the globally managed _db_instance.

def add_or_update_prompt_interop(name: str, author: Optional[str], details: Optional[str],
                                 system_prompt: Optional[str] = None, user_prompt: Optional[str] = None,
                                 keywords: Optional[List[str]] = None
                                 ) -> Tuple[Optional[int], Optional[str], str]:
    """
    Adds a new prompt or updates an existing one (identified by name).
    If the prompt exists (even if soft-deleted), it will be updated/undeleted.
    This wraps the standalone add_or_update_prompt function from Prompts_DB_v2.
    """
    db = get_db_instance()
    return db_add_or_update_prompt(db, name, author, details, system_prompt, user_prompt, keywords)

def load_prompt_details_for_ui_interop(prompt_name: str) -> Tuple[str, str, str, str, str, str]:
    """
    Loads prompt details formatted for UI display.
    This wraps the standalone load_prompt_details_for_ui function from Prompts_DB_v2.
    """
    db = get_db_instance()
    return db_load_prompt_details_for_ui(db, prompt_name)

def export_prompt_keywords_to_csv_interop() -> Tuple[str, str]:
    """
    Exports prompt keywords to a CSV file.
    This wraps the standalone export_prompt_keywords_to_csv function from Prompts_DB_v2.
    """
    db = get_db_instance()
    return db_export_prompt_keywords_to_csv(db)

def view_prompt_keywords_markdown_interop() -> str:
    """
    Generates a Markdown representation of prompt keywords.
    This wraps the standalone view_prompt_keywords_markdown function from Prompts_DB_v2.
    """
    db = get_db_instance()
    return db_view_prompt_keywords_markdown(db)

def export_prompts_formatted_interop(export_format: str = 'csv',
                                     filter_keywords: Optional[List[str]] = None,
                                     include_system: bool = True,
                                     include_user: bool = True,
                                     include_details: bool = True,
                                     include_author: bool = True,
                                     include_associated_keywords: bool = True,
                                     markdown_template_name: Optional[str] = "Basic Template"
                                     ) -> Tuple[str, str]:
    """
    Exports prompts to a specified format (CSV or Markdown).
    This wraps the standalone export_prompts_formatted function from Prompts_DB_v2.
    """
    db = get_db_instance()
    return db_export_prompts_formatted(db, export_format, filter_keywords,
                                       include_system, include_user, include_details,
                                       include_author, include_associated_keywords,
                                       markdown_template_name)

# --- Expose Exceptions for API layer ---
# These are already imported at the top: DatabaseError, SchemaError, InputError, ConflictError
# They can be caught by the calling code (e.g., your API endpoint) like:
#   try:
#       prompts_interop.add_prompt(...)
#   except prompts_interop.InputError as e:
#       # handle bad input
#   except prompts_interop.ConflictError as e:
#       # handle conflict
#   except prompts_interop.DatabaseError as e:
#       # handle general DB error


if __name__ == '__main__':
    # Example Usage (primarily for testing the interop layer)
    # Ensure Prompts_DB_v2.py is in the same directory or Python path

    # Setup basic logging for the example
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running prompts_interop.py example usage...")

    # --- Configuration ---
    # Use an in-memory database for this example for easy cleanup
    # For a real application, this would be a file path.
    EXAMPLE_DB_PATH = ":memory:"
    # EXAMPLE_DB_PATH = "test_interop_prompts.db" # For file-based testing
    EXAMPLE_CLIENT_ID = "interop_example_client"

    try:
        # 1. Initialize
        print("\n--- Initializing Interop Library ---")
        initialize_interop(db_path=EXAMPLE_DB_PATH, client_id=EXAMPLE_CLIENT_ID)
        print(f"Interop initialized: {is_initialized()}")
        print(f"DB instance client ID: {get_db_instance().client_id}")

        # 2. Add some data using interop functions
        print("\n--- Adding Data ---")
        kw_id1, kw_uuid1 = add_keyword("test")
        print(f"Added keyword 'test': ID {kw_id1}, UUID {kw_uuid1}")
        kw_id2, kw_uuid2 = add_keyword("example")
        print(f"Added keyword 'example': ID {kw_id2}, UUID {kw_uuid2}")

        p_id1, p_uuid1, msg1 = add_prompt(
            name="My First Prompt",
            author="Interop Test",
            details="This is a test prompt added via interop.",
            system_prompt="You are a helpful assistant.",
            user_prompt="Tell me a joke.",
            keywords=["test", "funny"] # "funny" will be newly created
        )
        print(f"Added prompt: {msg1} (ID: {p_id1}, UUID: {p_uuid1})")

        p_id2, p_uuid2, msg2 = add_or_update_prompt_interop(
            name="My Second Prompt",
            author="Interop Test",
            details="Another test prompt.",
            system_prompt="You are a creative writer.",
            user_prompt="Write a short story.",
            keywords=["example", "story"]
        )
        print(f"Added/Updated prompt via interop wrapper: {msg2} (ID: {p_id2}, UUID: {p_uuid2})")


        # 3. Read data
        print("\n--- Reading Data ---")
        prompt1_details = fetch_prompt_details(p_id1)
        if prompt1_details:
            print(f"Details for Prompt ID {p_id1} ('{prompt1_details.get('name')}'):")
            print(f"  Author: {prompt1_details.get('author')}")
            print(f"  Keywords: {prompt1_details.get('keywords')}")
        else:
            print(f"Could not fetch details for Prompt ID {p_id1}")

        all_prompts, total_pages, _, total_items = list_prompts()
        print(f"List Prompts (Page 1): {len(all_prompts)} items. Total items: {total_items}, Total pages: {total_pages}")
        for p in all_prompts:
            print(f"  - {p.get('name')} (Author: {p.get('author')})")

        all_kws = fetch_all_keywords()
        print(f"All active keywords: {all_kws}")


        # 4. Search
        print("\n--- Searching Data ---")
        search_results, total_matches = search_prompts(search_query="test", search_fields=["details", "keywords"])
        print(f"Search results for 'test': {len(search_results)} matches (Total found: {total_matches})")
        for res in search_results:
            print(f"  - Found: {res.get('name')} (Keywords: {res.get('keywords')})")


        # 5. Using other interop-wrapped standalone functions
        print("\n--- Using Other Interop Functions ---")
        markdown_keywords = view_prompt_keywords_markdown_interop()
        print("Keywords in Markdown:")
        print(markdown_keywords)

        csv_export_status, csv_file_path = export_prompts_formatted_interop(export_format='csv')
        print(f"CSV Export: {csv_export_status} -> {csv_file_path}")
        if csv_file_path != "None" and EXAMPLE_DB_PATH == ":memory:":
             print(f" (Note: CSV file '{csv_file_path}' would exist if not using in-memory DB)")


        # 6. Soft delete
        print("\n--- Soft Deleting ---")
        if p_id1:
            deleted = soft_delete_prompt(p_id1)
            print(f"Soft deleted prompt ID {p_id1}: {deleted}")
            prompt1_after_delete = get_prompt_by_id(p_id1)
            print(f"Prompt ID {p_id1} after delete (should be None): {prompt1_after_delete}")
            prompt1_deleted_rec = get_prompt_by_id(p_id1, include_deleted=True)
            print(f"Prompt ID {p_id1} after delete (fetching deleted, should exist): {prompt1_deleted_rec is not None}")


        # 7. Sync Log (Example)
        print("\n--- Sync Log ---")
        sync_entries = get_sync_log_entries(limit=5)
        print(f"First 5 sync log entries:")
        for entry in sync_entries:
            print(f"  ID: {entry['change_id']}, Entity: {entry['entity']}, Op: {entry['operation']}, UUID: {entry['entity_uuid']}")
        if sync_entries:
            deleted_count = delete_sync_log_entries([e['change_id'] for e in sync_entries])
            print(f"Deleted {deleted_count} sync log entries.")


    except (DatabaseError, SchemaError, InputError, ConflictError, RuntimeError, ValueError) as e:
        logger.error(f"An error occurred during interop example: {type(e).__name__} - {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {type(e).__name__} - {e}", exc_info=True)
    finally:
        # 8. Shutdown
        print("\n--- Shutting Down Interop Library ---")
        shutdown_interop()
        print(f"Interop initialized after shutdown: {is_initialized()}")

        # If using a file-based DB for testing, you might want to clean it up
        # if EXAMPLE_DB_PATH != ":memory:":
        #     import os
        #     if os.path.exists(EXAMPLE_DB_PATH):
        #         try:
        #             os.remove(EXAMPLE_DB_PATH)
        #             print(f"Cleaned up test database file: {EXAMPLE_DB_PATH}")
        #         except OSError as e:
        #             print(f"Error removing test database file {EXAMPLE_DB_PATH}: {e}")

    logger.info("Prompts_interop.py example usage finished.")

#
# End of prompts_interop.py
#######################################################################################################################
