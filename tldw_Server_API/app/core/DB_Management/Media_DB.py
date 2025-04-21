# SQLite_DB.py
#########################################
# SQLite_DB Library
# This library is used to perform any/all DB operations related to SQLite.
#
####
import configparser
####################
# Function List
# FIXME - UPDATE Function Arguments
# 1. get_connection(self)
# 2. execute_query(self, query: str, params: Tuple = ())
# 3. create_tables()
# 4. add_keyword(keyword: str)
# 5. delete_keyword(keyword: str)
# 6. add_media_with_keywords(url, title, media_type, content, keywords, prompt, summary, transcription_model, author, ingestion_date)
# 7. fetch_all_keywords()
# 8. keywords_browser_interface()
# 9. display_keywords()
# 10. export_keywords_to_csv()
# 11. browse_items(search_query, search_type)
# 12. fetch_item_details(media_id: int)
# 13. add_media_version(media_id: int, prompt: str, summary: str)
# 14. search_media_db(search_query: str, search_fields: List[str], keywords: str, page: int = 1, results_per_page: int = 10)
# 15. search_and_display(search_query, search_fields, keywords, page)
# 16. display_details(index, results)
# 17. get_details(index, dataframe)
# 18. format_results(results)
# 19. export_to_csv(search_query: str, search_fields: List[str], keyword: str, page: int = 1, results_per_file: int = 1000)
# 20. is_valid_url(url: str) -> bool
# 21. is_valid_date(date_string: str) -> bool
# 22. add_media_to_database(url, info_dict, segments, summary, keywords, custom_prompt_input, whisper_model)
# 23. create_prompts_db()
# 24. add_prompt(name, details, system, user=None)
# 25. fetch_prompt_details(name)
# 26. list_prompts()
# 27. insert_prompt_to_db(title, description, system_prompt, user_prompt)
# 28. update_media_content(media_id: int, content: str, prompt: str, summary: str)
# 29. search_media_database(query: str) -> List[Tuple[int, str, str]]
# 30. load_media_content(media_id: int)
# 31. create_document_version`
# 32. get_document_version`
# 33. get_all_document_versions`
# 34. delete_document_version`
# 35. rollback_to_version`
# 36.
#
#
#####################
#
# Import necessary libraries
import csv
import hashlib
import html
import os
import queue
import re
import shutil
import sqlite3
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional
#
# Local Libraries
from tldw_Server_API.app.core.Utils.Utils import get_project_relative_path, get_database_path, \
    get_database_dir, logger, logging
from tldw_Server_API.app.core.Utils.Chunk_Lib import chunk_text
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
#
# Third-Party Libraries
import gradio as gr
import pandas as pd
import yaml
#
#######################################################################################################################
# Function Definitions
#

def ensure_database_directory():
    os.makedirs(get_database_dir(), exist_ok=True)

ensure_database_directory()

# FIXME - Setup properly and test/add documentation for its existence...
# Construct the path to the config file
config_path = get_project_relative_path('Config_Files/config.txt')

# Read the config file
config = configparser.ConfigParser()
config.read(config_path)

# Get the SQLite path from the config, or use the default if not specified
sqlite_path = config.get('Database', 'sqlite_path', fallback=get_database_path('tldw_Server_API\\Databases\\server_media_summary.db'))

# Get the backup path from the config, or use the default if not specified
backup_path = config.get('Database', 'backup_path', fallback='server_database_backups')
backup_path = get_project_relative_path(backup_path)

# Set the final paths
db_path = sqlite_path
backup_dir = backup_path

logging.info(f"Media Database path: {db_path}")
logging.info(f"Media Backup directory: {backup_dir}")
#create_automated_backup(db_path, backup_dir)

# FIXME - Setup properly and test/add documentation for its existence...
#backup_file = create_automated_backup(db_path, backup_dir)
#upload_to_s3(backup_file, 'your-s3-bucket-name', f"database_backups/{os.path.basename(backup_file)}")

# FIXME - Setup properly and test/add documentation for its existence...
#create_incremental_backup(db_path, backup_dir)

# FIXME - Setup properly and test/add documentation for its existence...
#rotate_backups(backup_dir)

#
#
#######################################################################################################################


#######################################################################################################################
#
# Backup-related functions

def create_incremental_backup(db_path, backup_dir):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get the page count of the database
    cursor.execute("PRAGMA page_count")
    page_count = cursor.fetchone()[0]

    # Create a new backup file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(backup_dir, f"incremental_backup_{timestamp}.sqlib")

    # Perform the incremental backup
    conn.execute(f"VACUUM INTO '{backup_file}'")

    conn.close()
    print(f"Incremental backup created: {backup_file}")
    return backup_file


def create_automated_backup(db_path, backup_dir):
    # Ensure backup directory exists
    os.makedirs(backup_dir, exist_ok=True)

    # Create a timestamped backup file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(backup_dir, f"media_db_backup_{timestamp}.db")

    # Copy the database file
    shutil.copy2(db_path, backup_file)

    print(f"Backup created: {backup_file}")
    return backup_file

# FIXME - boto3 aint getting installed by default....
# def upload_to_s3(file_path, bucket_name, s3_key):
#     import boto3
#     s3 = boto3.client('s3')
#     try:
#         s3.upload_file(file_path, bucket_name, s3_key)
#         print(f"File uploaded to S3: {s3_key}")
#     except Exception as e:
#         print(f"Error uploading to S3: {str(e)}")


def rotate_backups(backup_dir, max_backups=10):
    backups = sorted(
        [f for f in os.listdir(backup_dir) if f.endswith('.db')],
        key=lambda x: os.path.getmtime(os.path.join(backup_dir, x)),
        reverse=True
    )

    while len(backups) > max_backups:
        old_backup = backups.pop()
        os.remove(os.path.join(backup_dir, old_backup))
        print(f"Removed old backup: {old_backup}")

#
#
#######################################################################################################################


#######################################################################################################################
#
# DB-Integrity Check Functions

def check_database_integrity(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("PRAGMA integrity_check")
    result = cursor.fetchone()

    conn.close()

    if result[0] == "ok":
        print("Database integrity check passed.")
        return True
    else:
        print("Database integrity check failed:", result[0])
        return False

#check_database_integrity(db_path)

#
# End of DB-Integrity Check functions
#######################################################################################################################


#######################################################################################################################
#
# DB Setup Functions

# Version 2
# Define a custom exception for clarity
class DatabaseError(Exception):
    """Custom exception for database related errors."""
    pass

class InputError(Exception):
    pass

class Database:
    """
    Manages a connection and operations for a specific SQLite database file.
    Ensures the necessary schema (tables, indices) exists upon initialization.
    """

    # Store table/index creation queries as class attributes for organization
    _TABLE_QUERIES = [
        '''
        CREATE TABLE IF NOT EXISTS Media (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE, -- Added UNIQUE constraint if URL should be unique identifier
            title TEXT NOT NULL,
            type TEXT NOT NULL,
            content TEXT, -- Holds primary content (latest transcript for A/V, text for docs, etc.)
            author TEXT,
            ingestion_date TEXT,
            -- Removed prompt
            -- Removed analysis_content
            transcription_model TEXT, -- Model used for the content currently in Media.content (if applicable)
            is_trash BOOLEAN DEFAULT 0 NOT NULL,
            trash_date DATETIME,
            vector_embedding BLOB,
            chunking_status TEXT DEFAULT 'pending' NOT NULL,
            vector_processing INTEGER DEFAULT 0 NOT NULL,
            content_hash TEXT UNIQUE NOT NULL -- Make hash non-nullable if always generated
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS Keywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT NOT NULL UNIQUE COLLATE NOCASE
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS MediaKeywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id INTEGER NOT NULL,
            keyword_id INTEGER NOT NULL,
            UNIQUE (media_id, keyword_id),
            FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,
            FOREIGN KEY (keyword_id) REFERENCES Keywords(id) ON DELETE CASCADE
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS Transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id INTEGER NOT NULL, -- Added NOT NULL constraint
            whisper_model TEXT,
            transcription TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (media_id, whisper_model), -- Allow multiple only if model differs
            FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS MediaChunks ( -- Keep as requested, but consider if UnvectorizedMediaChunks replaces it
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id INTEGER NOT NULL, -- Added NOT NULL
            chunk_text TEXT,
            start_index INTEGER,
            end_index INTEGER,
            chunk_id TEXT UNIQUE, -- chunk_id should probably be unique
            FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE
        )''',
        '''
        CREATE TABLE IF NOT EXISTS UnvectorizedMediaChunks ( -- Keep as requested
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            start_char INTEGER,
            end_char INTEGER,
            chunk_type TEXT,
            creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_processed BOOLEAN DEFAULT FALSE NOT NULL, -- Added NOT NULL
            metadata TEXT,
            UNIQUE (media_id, chunk_index, chunk_type),
            FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE
        )
        ''',
        '''
        CREATE TABLE IF NOT EXISTS DocumentVersions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id INTEGER NOT NULL,
            version_number INTEGER NOT NULL,
            prompt TEXT, -- Prompt associated with this version's analysis_content
            analysis_content TEXT, -- Analysis associated with this version's content
            content TEXT NOT NULL, -- The actual content snapshot for this version
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,
            UNIQUE (media_id, version_number)
        )
        ''',
    ]

    _INDEX_QUERIES = [
        # Indices for Media table (URL index covered by UNIQUE constraint if added)
        'CREATE INDEX IF NOT EXISTS idx_media_title ON Media(title)',
        'CREATE INDEX IF NOT EXISTS idx_media_type ON Media(type)',
        'CREATE INDEX IF NOT EXISTS idx_media_author ON Media(author)',
        'CREATE INDEX IF NOT EXISTS idx_media_ingestion_date ON Media(ingestion_date)',
        'CREATE INDEX IF NOT EXISTS idx_media_chunking_status ON Media(chunking_status)',
        'CREATE INDEX IF NOT EXISTS idx_media_vector_processing ON Media(vector_processing)',
        'CREATE INDEX IF NOT EXISTS idx_media_is_trash ON Media(is_trash)',
        # Content Hash index covered by UNIQUE constraint

        # Indices for Keywords/MediaKeywords (covered by UNIQUE constraints)
        # 'CREATE INDEX IF NOT EXISTS idx_keywords_keyword ON Keywords(keyword)',
        'CREATE INDEX IF NOT EXISTS idx_mediakeywords_media_id ON MediaKeywords(media_id)',
        'CREATE INDEX IF NOT EXISTS idx_mediakeywords_keyword_id ON MediaKeywords(keyword_id)',

        # Indices for Transcripts
        'CREATE INDEX IF NOT EXISTS idx_transcripts_media_id ON Transcripts(media_id)',

        # Indices for MediaChunks
        'CREATE INDEX IF NOT EXISTS idx_mediachunks_media_id ON MediaChunks(media_id)',
        # 'CREATE INDEX IF NOT EXISTS idx_mediachunks_chunk_id ON MediaChunks(chunk_id)', # Covered by UNIQUE

        # Indices for UnvectorizedMediaChunks
        'CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_media_id ON UnvectorizedMediaChunks(media_id)',
        'CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_is_processed ON UnvectorizedMediaChunks(is_processed)',
        'CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_chunk_type ON UnvectorizedMediaChunks(chunk_type)',

        # Indices for DocumentVersions
        'CREATE INDEX IF NOT EXISTS idx_document_versions_media_id ON DocumentVersions(media_id)',
        'CREATE INDEX IF NOT EXISTS idx_document_versions_version_number ON DocumentVersions(version_number)',
    ]

    _VIRTUAL_TABLE_QUERIES = [
        # Use content='Media' to link virtual table rows to the Media table's rowid
        'CREATE VIRTUAL TABLE IF NOT EXISTS media_fts USING fts5(title, content, content=\'Media\', content_rowid=\'id\')',
        # Optional: Add trigger to keep FTS table synchronized with Media table
        '''
        CREATE TRIGGER IF NOT EXISTS media_ai AFTER INSERT ON Media BEGIN
            INSERT INTO media_fts (rowid, title, content) VALUES (new.id, new.title, new.content);
        END;
        ''',
        '''
        CREATE TRIGGER IF NOT EXISTS media_ad AFTER DELETE ON Media BEGIN
            DELETE FROM media_fts WHERE rowid = old.id;
        END;
        ''',
        '''
        CREATE TRIGGER IF NOT EXISTS media_au AFTER UPDATE ON Media BEGIN
            UPDATE media_fts SET title = new.title, content = new.content WHERE rowid = old.id;
        END;
        ''',
        # Similar for keywords if needed, though less common
        # 'CREATE VIRTUAL TABLE IF NOT EXISTS keyword_fts USING fts5(keyword, content=\'Keywords\', content_rowid=\'id\')'
    ]

    def __init__(self, db_path: str):
        """
        Initializes the Database object for a specific file path.

        Args:
            db_path (str): The full path to the SQLite database file.
        """
        from pathlib import Path
        self.db_path = Path(db_path).resolve() # Store the absolute path
        # Ensure the parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Use thread-local storage for connections
        self._local = threading.local()
        logging.info(f"Initializing Database object for path: {self.db_path}")
        self._ensure_schema() # IMPORTANT: Create/verify schema on initialization

    def _get_thread_connection(self) -> sqlite3.Connection:
        """Gets or creates a database connection for the current thread."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            try:
                # Connect to the specific database file for this instance
                # check_same_thread=False is generally needed for web frameworks
                # timeout can be increased if experiencing database locking issues
                self._local.conn = sqlite3.connect(
                    str(self.db_path),
                    check_same_thread=False,
                    timeout=10 # Increased timeout (default is 5 seconds)
                )
                # Use Row factory for dict-like access to columns
                self._local.conn.row_factory = sqlite3.Row
                # Enable Write-Ahead Logging for better concurrency
                self._local.conn.execute("PRAGMA journal_mode=WAL;")
                # Enable foreign key constraints
                self._local.conn.execute("PRAGMA foreign_keys = ON;")
                logging.debug(f"Opened SQLite connection to {self.db_path} [thread: {threading.current_thread().name}]")
            except sqlite3.Error as e:
                logging.error(f"Failed to connect to database at {self.db_path}: {e}", exc_info=True)
                # Reset to prevent retrying with a failed connection object
                self._local.conn = None
                raise DatabaseError(f"Failed to connect to database '{self.db_path}': {e}") from e
        return self._local.conn

    def get_connection(self) -> sqlite3.Connection:
        """Provides access to the thread-local database connection."""
        return self._get_thread_connection()

    def close_connection(self):
        """Closes the connection for the current thread, if open."""
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
            logging.debug(f"Closed SQLite connection to {self.db_path} [thread: {threading.current_thread().name}]")

    def execute_query(self, query: str, params: tuple = None, *, commit: bool = False) -> sqlite3.Cursor:
        """
        Executes a given SQL query.

        Args:
            query (str): The SQL query to execute.
            params (tuple, optional): Parameters to bind to the query. Defaults to None.
            commit (bool): Whether to commit after this query (use False within transactions). Defaults to False.

        Returns:
            sqlite3.Cursor: The cursor object after execution.

        Raises:
            DatabaseError: If query execution fails.
        """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            logging.debug(f"Executing Query on {self.db_path}: {query[:150]}... Params: {params}")
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            if commit:
                conn.commit()
                logging.debug(f"Query committed on {self.db_path}")
            return cursor
        except sqlite3.Error as e:
            logging.error(f"Query failed on {self.db_path}: {query[:150]}... Error: {e}", exc_info=True)
            # Do not rollback here; let the transaction context handle it
            raise DatabaseError(f"Query execution failed: {e}") from e

    @contextmanager
    def transaction(self):
        """Provides a transactional context manager."""
        conn = self.get_connection()
        in_transaction = conn.in_transaction # Check if already in transaction
        try:
            if not in_transaction:
                logging.debug(f"Beginning transaction for {self.db_path}")
                conn.execute("BEGIN")
            yield conn # Yield the connection for use within the 'with' block
            if not in_transaction:
                conn.commit()
                logging.debug(f"Transaction committed for {self.db_path}")
        except Exception as e:
            if not in_transaction:
                logging.error(f"Transaction failed for {self.db_path}, rolling back: {e}", exc_info=True)
                conn.rollback()
            raise # Re-raise the exception after rollback/logging

    def table_exists(self, table_name: str) -> bool:
        """Checks if a table exists in the database."""
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        try:
            cursor = self.execute_query(query, (table_name,))
            return cursor.fetchone() is not None
        except DatabaseError:
            # If the query fails for some reason, assume table doesn't exist or DB is broken
            return False

    def process_chunks(db_instance, chunks: List[Dict], media_id: int, batch_size: int = 100):
        """
        Process chunks in batches and insert them into the database within a single transaction.

        Args:
            db_instance: Database instance to use for inserting chunks.
            chunks: List of chunk dictionaries (e.g., {'text': str, 'start_index': int, 'end_index': int}).
            media_id: ID of the media these chunks belong to.
            batch_size: Number of chunks to process in each batch.
        """
        # database=db_instance # No need to rename
        # log_counter("process_chunks_attempt", labels={"media_id": media_id}) # Placeholder
        start_time = time.time()
        total_chunks = len(chunks)
        processed_chunks = 0
        errors_in_transaction = False
        db = db_instance
        if not chunks:
            logging.warning(f"process_chunks called with empty chunk list for media_id {media_id}.")
            return

        logging.info(
            f"Starting chunk processing for media_id {media_id}. Total chunks: {total_chunks}, Batch size: {batch_size}")

        try:
            # Use a single transaction for the entire operation
            with db.transaction() as conn:  # conn is managed by the context manager
                for i in range(0, total_chunks, batch_size):
                    batch = chunks[i:i + batch_size]

                    # --- IMPORTANT: Adapt this to match your chunk dict structure AND target table ---
                    # Example for inserting into UnvectorizedMediaChunks:
                    chunk_data = []
                    for chunk in batch:
                        text = chunk.get('text', chunk.get('chunk_text'))
                        if text is not None:  # Ensure text exists
                            chunk_data.append((
                                media_id,
                                text,
                                chunk.get('chunk_index'),  # Assuming this exists in your chunk dict
                                chunk.get('start_index', chunk.get('start_char')),
                                chunk.get('end_index', chunk.get('end_char')),
                                chunk.get('chunk_type'),
                                # Add other fields like metadata if needed
                            ))
                        else:
                            logging.warning(
                                f"Skipping chunk with None text at index {i + batch.index(chunk)} for media_id {media_id}")

                    # Skip empty batches (e.g., if all chunks had None text)
                    if not chunk_data:
                        logging.debug(f"Skipping empty processed batch at index {i} for media_id {media_id}")
                        continue

                    try:
                        # --- IMPORTANT: Use the correct SQL for your target table ---
                        # Example SQL for UnvectorizedMediaChunks:
                        sql = """
                              INSERT INTO UnvectorizedMediaChunks
                                  (media_id, chunk_text, chunk_index, start_char, end_char, chunk_type)
                              VALUES (?, ?, ?, ?, ?, ?) \
                              """
                        # Call the new execute_many method.
                        # commit=False because we are inside a transaction managed by 'with db_instance.transaction()'
                        db_instance.execute_many(sql, chunk_data, commit=False)

                        processed_chunks += len(chunk_data)
                        logging.debug(
                            f"Inserted batch {i // batch_size + 1}, total processed: {processed_chunks}/{total_chunks} chunks for media_id {media_id}")
                        # log_counter("process_chunks_batch_success", labels={"media_id": media_id}) # Placeholder

                    except (DatabaseError, sqlite3.Error, TypeError) as batch_err:
                        # Catch specific errors from execute_many or sqlite3
                        logging.error(
                            f"Error inserting chunk batch starting at index {i} for media_id {media_id}: {batch_err}",
                            exc_info=True)
                        # log_counter("process_chunks_batch_error", labels={"media_id": media_id, "error_type": type(batch_err).__name__}) # Placeholder
                        errors_in_transaction = True
                        # Rollback will happen automatically when the exception leaves the 'with' block
                        raise DatabaseError(
                            f"Failed to insert chunk batch: {batch_err}") from batch_err  # Re-raise to trigger rollback

            # If the loop completes without error, the transaction commits automatically here.
            logging.info(f"Successfully finished processing {processed_chunks} chunks for media_id {media_id}")
            # log_counter("process_chunks_success", labels={"media_id": media_id}) # Placeholder

        except DatabaseError as e:
            # Catch errors from transaction management or re-raised batch errors
            logging.error(f"Database transaction error during chunk processing for media_id {media_id}: {e}",
                          exc_info=True)
            # log_counter("process_chunks_error", labels={"media_id": media_id, "error_type": "DatabaseError"}) # Placeholder
            # Transaction already rolled back
        except Exception as e:
            logging.error(f"Unexpected error during chunk processing for media_id {media_id}: {e}", exc_info=True)
            # log_counter("process_chunks_error", labels={"media_id": media_id, "error_type": type(e).__name__}) # Placeholder
            # Rollback should happen if the error occurred within the 'with' block

        finally:
            duration = time.time() - start_time
            logging.info(
                f"Chunk processing finished for media_id {media_id}. Duration: {duration:.4f}s. Processed: {processed_chunks}/{total_chunks}.")
            log_histogram("process_chunks_duration", duration, labels={"media_id": media_id}) # Placeholder


    def execute_many(self, query: str, params_list: List[tuple], *, commit: bool = False) -> sqlite3.Cursor:
        """
        Executes a given SQL query for multiple sets of parameters using cursor.executemany().

        Args:
            query (str): The SQL query to execute (e.g., "INSERT INTO table (...) VALUES (?, ?)")
            params_list (List[tuple]): A list of tuples, where each tuple contains parameters
                                        for one execution of the query.
            commit (bool): Whether to commit after this operation. Defaults to False.
                           It's often more efficient to wrap multiple execute_many calls
                           within a single transaction managed by the caller.

        Returns:
            sqlite3.Cursor: The cursor object after execution.

        Raises:
            DatabaseError: If query execution fails.
            TypeError: If params_list is not suitable for executemany.
        """
        conn = self.get_connection()
        # Basic type checking
        if not isinstance(params_list, list):
            # executemany expects an iterable sequence of sequences
            raise TypeError("params_list must be a list of sequences (e.g., list of tuples).")
        # Add check for empty list to avoid unnecessary DB call?
        if not params_list:
            logging.debug(f"execute_many called with empty params_list for query: {query[:100]}...")
            # Return a dummy cursor or handle as needed, maybe just return None or raise?
            # For now, proceed, sqlite3 might handle empty list gracefully or raise its own error.
            pass

        try:
            cursor = conn.cursor()
            logging.debug(f"Executing Many on {self.db_path}: {query[:150]}... with {len(params_list)} parameter sets.")
            # Use the standard sqlite3 cursor.executemany
            cursor.executemany(query, params_list)

            if commit:
                conn.commit()
                logging.debug(f"Execute Many committed on {self.db_path}")
            return cursor
        except sqlite3.Error as e:
            logging.error(f"Execute Many failed on {self.db_path}: {query[:150]}... Error: {e}", exc_info=True)
            # Do not rollback here; let the transaction context handle it if called within one
            raise DatabaseError(f"Execute Many failed: {e}") from e
        except TypeError as te:
            # Catch potential TypeError if params_list contents are wrong format for executemany
            logging.error(f"TypeError during Execute Many on {self.db_path}: {te}. Check format of params_list.",
                          exc_info=True)
            raise TypeError(f"Parameter list format error for executemany: {te}") from te
        except Exception as e:
            # Catch other unexpected errors
            logging.error(f"Unexpected error during Execute Many on {self.db_path}: {e}", exc_info=True)
            raise DatabaseError(f"An unexpected error occurred during Execute Many: {e}") from e

    def _ensure_schema(self):
        """
        Ensures the necessary tables and indices exist in the database file.
        This method is called automatically during __init__.
        Uses a separate connection for schema setup to avoid transaction conflicts.
        """
        conn = None # Ensure conn is defined for finally block
        try:
            # Use a dedicated connection for schema modifications
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("PRAGMA foreign_keys = ON;") # Enable FKs for schema changes too
            cursor = conn.cursor()

            logging.info(f"Ensuring schema exists for database: {self.db_path}")

            # Begin transaction for schema modifications
            cursor.execute("BEGIN")

            # 1. Create Tables
            logging.debug(f"Creating tables if not exist for {self.db_path}...")
            for query in self._TABLE_QUERIES:
                cursor.execute(query)

            # 2. Create Indices (handle potential pre-existence gracefully)
            logging.debug(f"Creating indices if not exist for {self.db_path}...")
            for query in self._INDEX_QUERIES:
                 try: cursor.execute(query)
                 except sqlite3.Error as idx_err: logging.warning(f"Non-fatal: Could not execute index query '{query[:70]}...' for {self.db_path} (may already exist or conflict): {idx_err}")

            # 3. Create Virtual Tables & Triggers (handle potential pre-existence)
            logging.debug(f"Creating virtual tables/triggers if not exist for {self.db_path}...")
            for query in self._VIRTUAL_TABLE_QUERIES:
                 try: cursor.execute(query)
                 except sqlite3.Error as vt_err: logging.warning(f"Non-fatal: Could not execute virtual table/trigger query '{query[:70]}...' for {self.db_path} (may already exist or conflict): {vt_err}")

            # 4. Schema Updates (ALTER TABLE, etc.)
            logging.debug(f"Applying schema updates if needed for {self.db_path}...")
            # Check/Add content_hash column and index (keep this logic)
            cursor.execute("SELECT COUNT(*) FROM pragma_table_info('Media') WHERE name = 'content_hash'")
            if cursor.fetchone()[0] == 0:
                logging.info(f"Adding 'content_hash' column to Media table in {self.db_path}")
                cursor.execute('ALTER TABLE Media ADD COLUMN content_hash TEXT')
            # Create index separately
            try:
                cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_media_content_hash ON Media(content_hash)')
            except sqlite3.Error as idx_err:
                logging.warning(...)

            # 5. Data Integrity Checks / Migrations (like ensuring initial versions)
            logging.debug(f"Performing data integrity checks for {self.db_path}...")
            # Modify this query to fetch necessary data *if* prompt/analysis were removed from Media
            # You might need to fetch the *last* modification's prompt/analysis if available
            # or just use NULLs for the initial version's prompt/analysis fields.

            # Simpler initial version creation if prompt/analysis removed from Media:
            cursor.execute("""
                           SELECT id, content
                           FROM Media
                           WHERE NOT EXISTS (SELECT 1 FROM DocumentVersions WHERE media_id = Media.id)
                           """)
            items_needing_version = cursor.fetchall()
            if items_needing_version:
                logging.info(
                    f"Creating initial versions for {len(items_needing_version)} media items in {self.db_path}")
                for media_id, content in items_needing_version:
                    cursor.execute("""
                                   INSERT INTO DocumentVersions (media_id, version_number, content, prompt,
                                                                 analysis_content, created_at)
                                   VALUES (?, 1, ?, NULL, NULL, CURRENT_TIMESTAMP)
                                   """, (media_id, content or ''))  # Insert NULL for prompt/analysis

            # Commit transaction
            conn.commit()
            logging.info(f"Schema verification and update successfully completed for {self.db_path}")

        except sqlite3.Error as e:
            if conn: conn.rollback() # Rollback on any schema error
            logging.error(f"Failed to ensure schema for {self.db_path}: {e}", exc_info=True)
            raise DatabaseError(f"Database schema initialization failed for '{self.db_path}': {e}") from e
        finally:
             if conn: conn.close() # Close the dedicated schema connection

    # FIXME - Update to reflect new schema
    def diagnose_schema(self):
        """Logs the schema of key tables for diagnostic purposes."""
        logging.info(f"--- Diagnosing Schema for {self.db_path} ---")
        try:
            conn = self.get_connection() # Use thread connection for reading schema
            cursor = conn.cursor()
            tables_to_check = ['Media', 'Keywords', 'MediaKeywords', 'MediaModifications', 'Transcripts', 'MediaChunks', 'UnvectorizedMediaChunks', 'DocumentVersions']
            for table_name in tables_to_check:
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                if columns:
                     logging.info(f"Table '{table_name}' columns: {[col['name'] for col in columns]}")
                else:
                     logging.warning(f"Table '{table_name}' not found during diagnosis.")

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            logging.info(f"All Tables found: {[table['name'] for table in tables]}")
            cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='index'")
            indices = cursor.fetchall()
            logging.info(f"All Indices found: {[idx['name'] for idx in indices]}")
            logging.info(f"--- End Schema Diagnosis for {self.db_path} ---")
        except Exception as e:
            logging.error(f"Schema diagnosis failed for {self.db_path}: {e}", exc_info=True)

    # --- Add other specific data access methods as needed ---
    # Example:
    # def get_media_by_id(self, media_id: int):
    #     cursor = self.execute_query("SELECT * FROM Media WHERE id = ?", (media_id,))
    #     return cursor.fetchone()

    # def add_keyword(self, keyword: str) -> int:
    #     # ... implementation using self.execute_query and transaction ...



# End of DB Setup Functions
#######################################################################################################################


#######################################################################################################################
#
# Media-related Functions

def check_media_exists(title: str, url: str, db_instance: Database) -> Optional[int]:
    """
    Checks if media exists by title or URL using the provided Database instance.

    Args:
        title: The title of the media to check.
        url: The URL of the media to check.
        db_instance: The specific Database instance (connected to the correct user's DB) to query.

    Returns:
        The media ID if found, otherwise None.
    """
    # Optional: Add a type check for robustness, though Python is duck-typed
    if not isinstance(db_instance, Database):
        logging.error("check_media_exists received an invalid db_instance type.")
        # Decide on error handling: raise TypeError, return None, etc.
        # Raising an error might be clearer.
        raise TypeError("A valid Database instance must be provided.")

    # Use the passed db_instance instead of the global 'db'
    try:
        # Use the get_connection method from the passed instance
        with db_instance.get_connection() as conn:
            cursor = conn.cursor()
            # Consider if title OR url is the right check, or if they should be separate
            # If URL is unique, checking only URL might be better.
            query = 'SELECT id FROM Media WHERE title = ? OR url = ? LIMIT 1'
            cursor.execute(query, (title, url))
            # Assuming row_factory is set to sqlite3.Row in your Database class
            result = cursor.fetchone()
            logging.debug(f"check_media_exists query on '{db_instance.db_path}': {query}")
            logging.debug(f"check_media_exists params: title={title}, url={url}")
            logging.debug(f"check_media_exists result: {result}")
            # Access by column name if using row_factory, otherwise by index [0]
            return result['id'] if result else None
    except sqlite3.Error as db_err: # Catch specific SQLite errors
        logging.error(f"SQLite error checking media existence on '{db_instance.db_path}': {db_err}", exc_info=True)
        return None
    except Exception as e: # Catch other unexpected errors
        logging.error(f"Unexpected error checking media existence on '{db_instance.db_path}': {e}")
        logging.error(f"Exception details: {traceback.format_exc()}")
        return None


def check_should_process_by_url(
    url: str,
    current_transcription_model: str,
    db_instance: Database
) -> Tuple[bool, Optional[int], str]:
    """
    Checks if media exists by URL and if processing should proceed based on
    the transcription model stored in the Media table.

    Args:
        url: The URL identifier of the media.
        current_transcription_model: The model intended for the current processing request.
        db_instance: Database connection/manager instance.

    Returns:
        Tuple (should_process, existing_media_id, reason)
        - should_process (bool): True if processing should proceed, False otherwise.
        - existing_media_id (Optional[int]): The ID if media exists, None otherwise.
        - reason (str): Explanation for the decision.
    """
    db=db_instance
    with db.transaction() as conn:
        cursor = conn.cursor()

    try:
        cursor = conn.cursor()
        logging.debug(f"Pre-checking media by URL: {url}")
        cursor.execute(
            'SELECT id, transcription_model FROM Media WHERE url = ?',
            (url,)
        )
        existing_media = cursor.fetchone()

        if not existing_media:
            logging.debug(f"Media not found for URL: {url}. Proceeding.")
            return True, None, "Media not found in database"

        existing_media_id, db_transcription_model = existing_media
        logging.debug(f"Found existing media ID: {existing_media_id}, DB model: '{db_transcription_model}'")

        # Handle cases where the stored model might be None or empty
        if not db_transcription_model:
             logging.warning(f"Existing media (ID: {existing_media_id}) has no transcription model recorded. Allowing processing.")
             return True, existing_media_id, "Existing media lacks transcription model info"

        if not current_transcription_model:
             # This case is less likely if the request mandates a model, but handle defensively
             logging.warning(f"No current transcription model provided for comparison. Skipping processing for existing media (ID: {existing_media_id}).")
             return False, existing_media_id, "No current transcription model provided for comparison"

        if db_transcription_model == current_transcription_model:
            logging.info(f"Media {existing_media_id} found with same transcription model ('{current_transcription_model}'). Skipping advised.")
            return False, existing_media_id, f"Media found with same transcription model (ID: {existing_media_id})"
        else:
            logging.info(f"Media {existing_media_id} found with different transcription model (DB: '{db_transcription_model}', Current: '{current_transcription_model}'). Processing allowed.")
            return True, existing_media_id, f"Different transcription model (DB: {db_transcription_model}, Current: {current_transcription_model})"

    except sqlite3.Error as e:
        logging.error(f"SQLite error during pre-check for URL {url}: {e}", exc_info=True)
        # Fail safe - allow processing if check fails, but log error
        return True, None, f"Database error during pre-check: {e}"


def check_media_and_whisper_model(title=None, url=None, current_whisper_model=None, db_instance: Database=None):
    """
    Check if media exists in the database and compare the whisper model used.

    :param title: Title of the media (optional)
    :param url: URL of the media (optional)
    :param current_whisper_model: The whisper model currently selected for use
    :param db_instance: Database connection/manager instance.
    :return: Tuple (bool, str) - (should_download, reason)
    """
    db=db_instance
    if not title and not url:
        return True, "No title or URL provided"

    with db_instance.get_connection() as conn:
        cursor = conn.cursor()

        # First, find the media_id
        query = "SELECT id FROM Media WHERE "
        params = []

        if title:
            query += "title = ?"
            params.append(title)

        if url:
            if params:
                query += " OR "
            query += "url = ?"
            params.append(url)

        cursor.execute(query, tuple(params))
        result = cursor.fetchone()

        if not result:
            return True, "Media not found in database"

        media_id = result[0]

        # Now, get the latest transcript for this media
        cursor.execute("""
            SELECT transcription 
            FROM Transcripts 
            WHERE media_id = ? 
            ORDER BY created_at DESC 
            LIMIT 1
        """, (media_id,))

        transcript_result = cursor.fetchone()

        if not transcript_result:
            return True, f"No transcript found for media (ID: {media_id})"

        transcription = transcript_result[0]

        # Extract the whisper model from the transcription
        match = re.search(r"This text was transcribed using whisper model: (.+)$", transcription, re.MULTILINE)
        if not match:
            return True, f"Whisper model information not found in transcript (Media ID: {media_id})"

        db_whisper_model = match.group(1).strip()

        if not current_whisper_model:
            return False, f"Media found in database (ID: {media_id})"

        if db_whisper_model != current_whisper_model:
            return True, f"Different whisper model (DB: {db_whisper_model}, Current: {current_whisper_model})"

        return False, f"Media found with same whisper model (ID: {media_id})"


def add_media_chunk(media_id: int, chunk_text: str, start_index: int, end_index: int, chunk_id: str, db_instance: Database):
    db=db_instance
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO MediaChunks (media_id, chunk_text, start_index, end_index, chunk_id) VALUES (?, ?, ?, ?, ?)",
            (media_id, chunk_text, start_index, end_index, chunk_id)
        )
        conn.commit()

def sqlite_update_fts_for_media(db_instance: Database, media_id: int):
    db=db_instance
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO media_fts (rowid, title, content) SELECT id, title, content FROM Media WHERE id = ?", (media_id,))
        conn.commit()


def get_unprocessed_media(db_instance):
    query = """
    SELECT id, content, type, COALESCE(title, '') as file_name
    FROM Media 
    WHERE vector_processing = 0
    ORDER BY id
    """
    return db_instance.execute_query(query)

# FIXME - rewrite
def get_next_media_id(db_instance: Database):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(media_id) FROM media")
        max_id = cursor.fetchone()[0]
        return (max_id or 0) + 1
    finally:
        conn.close()


def mark_media_as_processed(db_instance: Database, media_id):
    database = db_instance
    try:
        query = "UPDATE Media SET vector_processing = 1 WHERE id = ?"
        database.execute_query(query, (media_id,))
        logger.info(f"Marked media_id {media_id} as processed")
    except Exception as e:
        logger.error(f"Error marking media_id {media_id} as processed: {str(e)}")
        raise

#
# End of Vector-chunk-related Functions
#######################################################################################################################


#######################################################################################################################
# Keyword-related Functions
#

# Wrapper function for legacy support
def add_media_to_database(url, info_dict, segments, analysis_content, keywords, custom_prompt_input, whisper_model,
                          media_type='video', overwrite=False, db_instance: Database=None):
    """Legacy wrapper for add_media_with_keywords"""
    db=db_instance
    # Extract content from segments
    if isinstance(segments, list):
        content = ' '.join([segment.get('Text', '') for segment in segments if 'Text' in segment])
    elif isinstance(segments, dict):
        content = segments.get('text', '') or segments.get('content', '')
    else:
        content = str(segments)

    # Call the new function
    media_id, message = add_media_with_keywords(
        url=url,
        title=info_dict.get('title', 'Untitled'),
        media_type=media_type,
        content=content,
        keywords=keywords,
        prompt=custom_prompt_input,
        analysis_content=analysis_content,
        transcription_model=whisper_model,
        author=info_dict.get('uploader', 'Unknown'),
        ingestion_date=datetime.now().strftime('%Y-%m-%d'),
        overwrite=overwrite,
        db_instance=db
    )

    return message  # Return just the message to maintain backward compatibility


# Function to add media with keywords
def add_media_with_keywords(
    url: Optional[str], # Use Optional type hint
    title: Optional[str],
    media_type: Optional[str],
    content: Optional[str],
    keywords: Optional[List[str]], # Expect List now based on endpoint model parsing
    prompt: Optional[str],
    analysis_content: Optional[str],
    transcription_model: Optional[str],
    author: Optional[str],
    ingestion_date: Optional[str], # Expect string YYYY-MM-DD
    overwrite: bool = False,
    db_instance: Database = None,
    chunk_options: Optional[Dict] = None, # Pass through if needed for scheduling
    segments: Optional[Any] = None # Pass through if needed elsewhere
) -> Tuple[Optional[int], str]: # Return media_id (or None on error) and message
    """
    Adds or updates a media item in the database, including its keywords and
    initial document version.

    Args:
        url: URL or identifier of the media.
        title: Title of the media.
        media_type: Type of media (e.g., 'audio', 'pdf').
        content: The main content (transcript, text).
        keywords: A list of keyword strings.
        prompt: Initial prompt associated with the content/analysis.
        analysis_content: Initial analysis content.
        transcription_model: Model used for transcription (if applicable).
        author: Author of the media.
        ingestion_date: Date of ingestion (YYYY-MM-DD string).
        overwrite: If True, update existing media found by URL or content hash.
        db_instance: The Database instance.
        chunk_options: Options for chunking (passed to scheduler).
        segments: Original segments data (passed through).

    Returns:
        Tuple containing the media_id (int or None) and a status message (str).

    Raises:
        DatabaseError: If database operations fail.
        InputError: If input validation fails.
        TypeError: If db_instance is not valid.
    """
    # log_counter("add_media_with_keywords_attempt") # Using placeholders
    start_time = time.time()
    logging.debug(f"Entering add_media_with_keywords: URL={url}, Title={title}, Type={media_type}")

    # --- Validation ---
    if not isinstance(db_instance, Database):
        raise TypeError("A valid Database instance must be provided.")

    # Set default values if None provided (but content is crucial)
    # url = url # Keep URL as potentially None initially
    title = title or 'Untitled'
    media_type = media_type or 'unknown'
    if content is None:
        # Content is essential for hash and versioning
        logging.error("add_media_with_keywords called with None content.")
        raise InputError("Content cannot be None when adding media.")
    keywords_list = [k.strip().lower() for k in keywords if k and k.strip()] if keywords else ['default'] # Default if empty/None
    # prompt, analysis_content, transcription_model, author can be None (will be stored as NULL)
    ingestion_date_str = ingestion_date or datetime.now().strftime('%Y-%m-%d')

    # Simplified media type validation (adjust allowed types as needed)
    allowed_types = {'article', 'audio', 'book', 'document', 'ebook', 'mediawiki_article', 'mediawiki_dump',
                     'obsidian_note', 'pdf', 'podcast', 'text', 'video', 'unknown'}
    log_counter("add_media_with_keywords_error", labels={"error_type": "InvalidMediaType"})
    duration = time.time() - start_time
    log_histogram("add_media_with_keywords_duration", duration)
    if media_type not in allowed_types:
        # log_counter("add_media_with_keywords_error", labels={"error_type": "InvalidMediaType"})
        raise InputError(f"Invalid media type '{media_type}'. Allowed: {', '.join(allowed_types)}.")

    if ingestion_date and not is_valid_date(ingestion_date_str):
        log_counter("add_media_with_keywords_error", labels={"error_type": "InvalidDateFormat"})
        duration = time.time() - start_time
        log_histogram("add_media_with_keywords_duration", duration)
        raise InputError(f"Invalid ingestion date format '{ingestion_date}'. Use YYYY-MM-DD.")

    # --- Preparation ---
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    logging.debug(f"Generated content hash: {content_hash}")

    # Generate a placeholder URL if none provided (using hash ensures uniqueness if content is unique)
    if not url:
        url = f"local://{media_type}/{content_hash}"
        logging.info(f"No URL provided, generated placeholder: {url}")

    logging.info(f"Processing add/update for: URL='{url}', Title='{title}', Type='{media_type}'")
    logging.debug(f"Keywords: {keywords_list}, Overwrite={overwrite}")
    # Avoid logging full content/prompt/analysis unless necessary and sensitive data is masked
    # logging.debug(f"Content Hash: {content_hash}")
    # logging.debug(f"Prompt: {prompt[:100] if prompt else 'None'}...")
    # logging.debug(f"Analysis: {analysis_content[:100] if analysis_content else 'None'}...")

    # --- Database Interaction ---
    try:
        # Use a single transaction for all related inserts/updates
        with db_instance.transaction() as conn:
            cursor = conn.cursor()

            # Check if media already exists by URL (preferred identifier) or content_hash
            cursor.execute('SELECT id FROM Media WHERE url = ? OR content_hash = ?', (url, content_hash))
            existing_media = cursor.fetchone()
            logging.debug(f"Existing media check result for URL/Hash: {existing_media}")

            media_id = None
            action = "skipped" # Default action

            if existing_media:
                media_id = existing_media[0]
                logging.debug(f"Found existing media with ID: {media_id}")

                if overwrite:
                    logging.info(f"Overwrite requested. Updating existing media ID: {media_id}")
                    # --- Update Media Table ---
                    cursor.execute('''
                    UPDATE Media
                    SET url = ?, title = ?, type = ?, content = ?, author = ?,
                        ingestion_date = ?, transcription_model = ?,
                        chunking_status = ?, -- Reset chunking status on update
                        content_hash = ?,
                        is_trash = 0, trash_date = NULL -- Un-trash if overwriting
                    WHERE id = ?
                    ''', (url, title, media_type, content, author, ingestion_date_str,
                          transcription_model, 'pending', content_hash, media_id))
                    action = "updated"
                    # log_counter("add_media_with_keywords_update")

                    # --- Clear Old Keywords ---
                    # Important: Clear before adding new ones
                    cursor.execute('DELETE FROM MediaKeywords WHERE media_id = ?', (media_id,))
                    logging.debug(f"Cleared old keywords for updated media ID: {media_id}")

                    # Optional: Delete old versions? Decide based on requirements.
                    # For now, keep history and just add a new version later.

                else:
                    action = "already_exists_skipped"
                    logging.info(f"Media already exists (ID: {media_id}) and overwrite is False. Skipping update.")
                    # log_counter("add_media_with_keywords_skipped")
            else:
                logging.info("No existing media found by URL/Hash. Inserting new media.")
                # --- Insert New Media Record ---
                cursor.execute('''
                INSERT INTO Media (url, title, type, content, author, ingestion_date,
                                   transcription_model, chunking_status, content_hash, is_trash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0) -- Ensure is_trash is initially 0
                ''', (url, title, media_type, content, author, ingestion_date_str,
                      transcription_model, 'pending', content_hash))
                media_id = cursor.lastrowid # Get the ID of the newly inserted row
                action = "added"
                logging.info(f"New media inserted with ID: {media_id}")
                log_counter("add_media_with_keywords_insert")

            # --- Process Keywords & Versioning (Only if Added or Updated) ---
            if action in ["updated", "added"]:
                # --- Handle Keywords ---
                if keywords_list:
                    # Ensure keywords exist in Keywords table
                    unique_keywords = list(set(keywords_list)) # Process unique keywords
                    keyword_params = [(kw,) for kw in unique_keywords]
                    cursor.executemany('INSERT OR IGNORE INTO Keywords (keyword) VALUES (?)', keyword_params)

                    # Get keyword IDs for linking
                    placeholders = ','.join(['?'] * len(unique_keywords))
                    cursor.execute(f'SELECT id, keyword FROM Keywords WHERE keyword IN ({placeholders})', unique_keywords)
                    keyword_id_map = {kw: kid for kid, kw in cursor.fetchall()}

                    # Insert media-keyword links
                    media_keyword_params = [
                        (media_id, keyword_id_map[kw])
                        for kw in keywords_list # Iterate original list to preserve potential duplicates if needed by logic, though link table is unique
                        if kw in keyword_id_map # Ensure keyword was found/inserted
                    ]
                    if media_keyword_params:
                         # Use INSERT OR IGNORE to handle potential race conditions or duplicate calls
                         cursor.executemany('INSERT OR IGNORE INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)', media_keyword_params)
                         logging.debug(f"Linked {len(media_keyword_params)} keywords to media ID: {media_id}")

                # --- Update FTS Index ---
                # Use INSERT OR REPLACE to handle both inserts and updates cleanly
                # FTS trigger should ideally handle this, but explicit update is safer
                cursor.execute('INSERT OR REPLACE INTO media_fts (rowid, title, content) VALUES (?, ?, ?)',
                            (media_id, title, content))
                logging.debug(f"Updated FTS index for media ID: {media_id}")

                # --- Create Initial/Updated Document Version ---
                # Call the *updated* create_document_version, passing the connection
                logging.debug(f"Creating document version for media ID: {media_id} (Action: {action})")
                version_info = create_document_version(
                    media_id=media_id,
                    content=content,
                    prompt=prompt, # Pass the initial/updated prompt (can be None)
                    analysis_content=analysis_content, # Pass the initial/updated analysis (can be None)
                    db_instance=db_instance, # Pass instance for context
                    conn=conn # IMPORTANT: Pass the active connection
                )
                logging.info(f"Created version {version_info.get('version_number')} for media ID: {media_id}")

            # Transaction commits automatically if no exceptions raised

        # --- Post-Transaction Actions (e.g., Scheduling) ---
        if action in ["updated", "added"] and media_id is not None:
             if chunk_options:
                  logging.info(f"Scheduling chunking for media ID: {media_id} with options: {chunk_options}")
                  # Replace with actual scheduling call:
                  # schedule_chunking_task(media_id, content, title, chunk_options, db_instance.db_path)
             else:
                  logging.debug(f"No chunking options provided for media ID: {media_id}, skipping scheduling.")
             # Set chunking_status more accurately based on scheduling success?
             # db_instance.execute_query("UPDATE Media SET chunking_status = ? WHERE id = ?", ('scheduled', media_id), commit=True)


        # --- Logging and Return ---
        duration = time.time() - start_time
        log_histogram("add_media_with_keywords_duration", duration)

        if action == "updated":
            message = f"Media '{title}' (ID: {media_id}) updated successfully."
            log_counter("add_media_with_keywords_success")
        elif action == "added":
            message = f"Media '{title}' (ID: {media_id}) added successfully."
            log_counter("add_media_with_keywords_success")
        else: # Skipped
            message = f"Media '{title}' (ID: {media_id}) already exists and was not overwritten."

        logging.info(f"add_media_with_keywords completed for URL '{url}'. Action: {action}. Duration: {duration:.4f}s. Message: {message}")
        # Ensure media_id is returned even if skipped, allows endpoint to report correctly
        return media_id, message

    # --- Error Handling ---
    except InputError as ie:
         logging.warning(f"Input error in add_media_with_keywords for URL {url}: {ie}")
         raise # Re-raise InputError to be handled by the endpoint
    except sqlite3.IntegrityError as ie:
         # Often due to UNIQUE constraint violations (e.g., URL already exists by different hash?)
         logging.error(f"Database Integrity Error for URL {url}: {ie}", exc_info=True)
         # log_counter("add_media_with_keywords_error", labels={"error_type": "IntegrityError"})
         raise DatabaseError(f"Failed adding media due to data integrity issue: {ie}") from ie
    except sqlite3.Error as e:
        logging.error(f"SQLite Error for URL {url}: {e}", exc_info=True)
        # log_counter("add_media_with_keywords_error", labels={"error_type": "SQLiteError"})
        raise DatabaseError(f"Database error adding media: {e}") from e
    except Exception as e:
        logging.error(f"Unexpected Error for URL {url}: {e}", exc_info=True)
        # log_counter("add_media_with_keywords_error", labels={"error_type": type(e).__name__})
        raise DatabaseError(f"An unexpected error occurred while adding media: {e}") from e
    # finally: # Duration logging can be done here if needed even on error
    #      duration = time.time() - start_time
    #      log_histogram("add_media_with_keywords_duration", duration)


# Function to add a keyword
def add_keyword(keyword: str, db_instance: Database) -> int:
    db=db_instance
    log_counter("add_keyword_attempt")
    start_time = time.time()

    if not keyword.strip():
        log_counter("add_keyword_error", labels={"error_type": "EmptyKeyword"})
        duration = time.time() - start_time
        log_histogram("add_keyword_duration", duration)
        raise DatabaseError("Keyword cannot be empty")

    keyword = keyword.strip().lower()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        try:
            # Insert into Keywords table
            cursor.execute('INSERT OR IGNORE INTO Keywords (keyword) VALUES (?)', (keyword,))

            # Get the keyword_id (whether it was just inserted or already existed)
            cursor.execute('SELECT id FROM Keywords WHERE keyword = ?', (keyword,))
            keyword_id = cursor.fetchone()[0]

            # Check if the keyword exists in keyword_fts
            cursor.execute('SELECT rowid FROM keyword_fts WHERE rowid = ?', (keyword_id,))
            if not cursor.fetchone():
                # If it doesn't exist in keyword_fts, insert it
                cursor.execute('INSERT OR IGNORE INTO keyword_fts (rowid, keyword) VALUES (?, ?)', (keyword_id, keyword))

            logging.info(f"Keyword '{keyword}' added or updated with ID: {keyword_id}")
            conn.commit()

            duration = time.time() - start_time
            log_histogram("add_keyword_duration", duration)
            log_counter("add_keyword_success")

            return keyword_id
        except sqlite3.IntegrityError as e:
            logging.error(f"Integrity error adding keyword: {e}")
            duration = time.time() - start_time
            log_histogram("add_keyword_duration", duration)
            log_counter("add_keyword_error", labels={"error_type": "IntegrityError"})
            raise DatabaseError(f"Integrity error adding keyword: {e}")
        except sqlite3.Error as e:
            logging.error(f"Error adding keyword: {e}")
            duration = time.time() - start_time
            log_histogram("add_keyword_duration", duration)
            log_counter("add_keyword_error", labels={"error_type": "SQLiteError"})
            raise DatabaseError(f"Error adding keyword: {e}")



# Function to delete a keyword
def delete_keyword(keyword: str, db_instance: Database) -> str:
    db=db_instance
    log_counter("delete_keyword_attempt")
    start_time = time.time()

    keyword = keyword.strip().lower()
    with db.get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT id FROM Keywords WHERE keyword = ?', (keyword,))
            keyword_id = cursor.fetchone()
            if keyword_id:
                cursor.execute('DELETE FROM Keywords WHERE keyword = ?', (keyword,))
                cursor.execute('DELETE FROM keyword_fts WHERE rowid = ?', (keyword_id[0],))
                conn.commit()

                duration = time.time() - start_time
                log_histogram("delete_keyword_duration", duration)
                log_counter("delete_keyword_success")

                return f"Keyword '{keyword}' deleted successfully."
            else:
                duration = time.time() - start_time
                log_histogram("delete_keyword_duration", duration)
                log_counter("delete_keyword_not_found")

                return f"Keyword '{keyword}' not found."
        except sqlite3.Error as e:
            duration = time.time() - start_time
            log_histogram("delete_keyword_duration", duration)
            log_counter("delete_keyword_error", labels={"error_type": type(e).__name__})
            logging.error(f"Error deleting keyword: {e}")
            raise DatabaseError(f"Error deleting keyword: {e}")


def fetch_all_keywords(db_instance: Database) -> List[str]:
    db=db_instance
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT keyword FROM Keywords')
            keywords = [row[0] for row in cursor.fetchall()]
            return keywords
    except sqlite3.Error as e:
        raise DatabaseError(f"Error fetching keywords: {e}")

def keywords_browser_interface(db_instance: Database):
    keywords = fetch_all_keywords(db_instance)
    return gr.Markdown("\n".join(f"- {keyword}" for keyword in keywords))

def display_keywords(db_instance: Database):
    try:
        keywords = fetch_all_keywords(db_instance)
        return "\n".join(keywords) if keywords else "No keywords found."
    except DatabaseError as e:
        return str(e)


def export_keywords_to_csv(db_instance: Database):
    try:
        keywords = fetch_all_keywords(db_instance)
        if not keywords:
            return None, "No keywords found in the database."

        filename = "keywords.csv"
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Keyword"])
            for keyword in keywords:
                writer.writerow([keyword])

        return filename, f"Keywords exported to {filename}"
    except Exception as e:
        logger.error(f"Error exporting keywords to CSV: {e}")
        return None, f"Error exporting keywords: {e}"

# FIXME - REWRITE TO NOT USE MEDIAMODFIICATIONS
def fetch_keywords_for_media(media_id, db_instance: Database):
    try:
        # First check if the keywords column exists in MediaModifications
        with db_instance.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) 
                FROM pragma_table_info('MediaModifications') 
                WHERE name = 'keywords'
            """)
            keywords_column_exists = cursor.fetchone()[0]

            if keywords_column_exists:
                # Try to get keywords from MediaModifications first
                cursor.execute("""
                    SELECT keywords
                    FROM MediaModifications
                    WHERE media_id = ?
                    ORDER BY modification_date DESC
                    LIMIT 1
                """, (media_id,))
                result = cursor.fetchone()

                if result and result[0]:
                    # If we found keywords in MediaModifications, return them
                    return [k.strip() for k in result[0].split(',') if k.strip()]

            # Fallback: Get keywords from Keywords table via MediaKeywords
            cursor.execute('''
                SELECT k.keyword
                FROM Keywords k
                JOIN MediaKeywords mk ON k.id = mk.keyword_id
                WHERE mk.media_id = ?
            ''', (media_id,))
            keywords = [row[0] for row in cursor.fetchall()]

            return keywords or ["default"]
    except sqlite3.Error as e:
        logging.error(f"Error fetching keywords: {e}")
        return ["default"]  # Return a default keyword on error

def update_keywords_for_media(media_id: int, keywords: List[str], db_instance: Database):
    """Update keywords with validation and error handling"""
    try:
        valid_keywords = [k.strip().lower() for k in keywords if k.strip()]
        if not valid_keywords:
            return

        with db_instance.transaction() as conn:
            cursor = conn.cursor()

            # Clear existing keywords
            cursor.execute('DELETE FROM MediaKeywords WHERE media_id = ?', (media_id,))

            # Insert new keywords
            keyword_ids = []
            for keyword in set(valid_keywords):
                # Insert or ignore existing keywords
                cursor.execute('''
                    INSERT OR IGNORE INTO Keywords (keyword) VALUES (?)
                ''', (keyword,))

                # Get keyword ID
                cursor.execute('''
                    SELECT id FROM Keywords WHERE keyword = ?
                ''', (keyword,))
                result = cursor.fetchone()
                if not result:
                    raise ValueError(f"Keyword '{keyword}' not found after insertion")
                keyword_ids.append(result[0])

            # Insert relationships
            cursor.executemany('''
                INSERT INTO MediaKeywords (media_id, keyword_id)
                VALUES (?, ?)
            ''', [(media_id, kid) for kid in keyword_ids])

    except sqlite3.Error as e:
        logging.error(f"Database error updating keywords: {e}")
        raise DatabaseError(f"Keyword update failed: {e}")
    except ValueError as e:
        logging.error(f"Keyword validation error: {e}")
        raise DatabaseError(str(e))

#
# End of Keyword-related functions
#######################################################################################################################


#######################################################################################################################
#
# Media-related Functions


###################################################
# Function to fetch items based on search query and type
###################################################
def browse_items(search_query, search_type, db_instance: Database):
    db=db_instance
    try:
        with db_instance.get_connection() as conn:
            cursor = conn.cursor()
            if search_type == 'Title':
                cursor.execute("SELECT id, title, url FROM Media WHERE title LIKE ?", (f'%{search_query}%',))
            elif search_type == 'URL':
                cursor.execute("SELECT id, title, url FROM Media WHERE url LIKE ?", (f'%{search_query}%',))
            elif search_type == 'Keyword':
                return fetch_items_by_keyword(search_query, db_instance)
            elif search_type == 'Content':
                cursor.execute("SELECT id, title, url FROM Media WHERE content LIKE ?", (f'%{search_query}%',))
            else:
                raise ValueError(f"Invalid search type: {search_type}")

            results = cursor.fetchall()
            return results
    except sqlite3.Error as e:
        logger.error(f"Error fetching items by {search_type}: {e}")
        raise DatabaseError(f"Error fetching items by {search_type}: {e}")

###################################################
# Function to fetch item details
###################################################
def fetch_item_details(media_id: int, db_instance: Database) -> Tuple[str, str, str]:
    """
    Fetches the prompt, analysis_content, and content from the LATEST document version
    associated with the media_id.

    Args:
        media_id: The ID of the media item.
        db_instance: The Database instance to use.

    Returns:
        A tuple containing (prompt, analysis_content, content).
        Returns default messages/None if no version or data is found.

    Raises:
        DatabaseError: If a database query fails.
        TypeError: If db_instance is not a valid Database object.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("A valid Database instance must be provided.")

    logging.debug(f"Fetching latest item details (prompt, analysis, content) for media_id={media_id} from DB: {db_instance.db_path}")

    try:
        # Call the (soon to be updated) get_document_version to fetch the LATEST version's details
        # Passing None for version_number fetches the latest.
        # Ensure include_content=True to get the 'content' field.
        latest_version = get_document_version(
            media_id=media_id,
            version_number=None,
            include_content=True,
            db_instance=db_instance
        )

        if latest_version and 'error' not in latest_version:
            # Extract data using .get() with default values (though NULL is stored in DB)
            prompt = latest_version.get('prompt')  # Will be None if NULL in DB
            analysis_content = latest_version.get('analysis_content') # Will be None if NULL in DB
            content = latest_version.get('content') # Should ideally not be NULL

            logging.debug(f"Found latest version {latest_version.get('version_number')} for media_id={media_id}")
            # Return the actual values (could be None)
            return prompt, analysis_content, content
        else:
            # This case means the media item exists but has NO versions, which shouldn't happen
            # with the updated add_media_with_keywords logic. Log a warning.
            error_msg = latest_version.get('error', 'Unknown error') if latest_version else 'No version found'
            logging.warning(f"No document version found for media_id {media_id}: {error_msg}. This might indicate an issue with initial version creation.")
            # Return None for all fields to indicate data wasn't found in versions
            return None, None, None

    # Catch specific DB errors first if possible
    except sqlite3.Error as e:
        logging.error(f"SQLite error fetching latest version details for media_id {media_id} from {db_instance.db_path}: {e}", exc_info=True)
        raise DatabaseError(f"Database error fetching item details: {e}") from e
    except Exception as e: # Catch other potential errors (like DatabaseError from get_document_version)
        logging.error(f"Unexpected error fetching item details for media_id {media_id} from {db_instance.db_path}: {e}", exc_info=True)
        raise DatabaseError(f"Unexpected error fetching item details: {e}") from e

#
#  End of Media-related Functions
#######################################################################################################################


#######################################################################################################################
#
# Media-related Functions

def search_media_db(
        search_query: Optional[str], # Allow None
        search_fields: List[str],
        keywords: Optional[List[str]], # Allow None or empty list
        page: int = 1,
        results_per_page: int = 20,
        db_instance: Database = None # Make mandatory later if desired, for now check explicitly
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Search for media items using the provided Database instance, FTS, and keyword filters.

    Args:
        search_query: The term to search for in specified fields (can be None).
        search_fields: A list of fields ('title', 'content', 'author', 'type') to search within.
                       'title' and 'content' will use FTS if search_query is provided.
        keywords: A list of keywords to filter by (can be None or empty).
        page: The page number for pagination (starts at 1).
        results_per_page: The number of results to return per page.
        db_instance: The Database instance connected to the specific user's DB.

    Returns:
        A tuple containing:
            - A list of dictionaries representing the matching media items.
            - The total number of matches found (for pagination calculation).

    Raises:
        ValueError: If input parameters (page, results_per_page, search_fields) are invalid.
        DatabaseError: If a database query fails.
        TypeError: If db_instance is not a valid Database object.
    """
    # --- Input Validation ---
    if not isinstance(db_instance, Database):
        # Make this mandatory by removing the default None in signature if needed
        raise TypeError("A valid Database instance must be provided to search_media_db.")

    if page < 1:
        raise ValueError("Page number must be 1 or greater")
    if results_per_page < 1: # Removed upper limit check, handle large requests carefully
        raise ValueError("Results per page must be 1 or greater")

    valid_fields = {"title", "content", "author", "type"}
    # Use provided fields, default to title/content if none are given but query exists?
    sanitized_fields = [field for field in search_fields if field in valid_fields] if search_fields else []
    if not sanitized_fields and search_query:
        # Default search fields if query is present but fields are not
        sanitized_fields = ["title", "content"]
        # raise ValueError("No valid search fields provided when search_query is present.")

    # Ensure keywords is a list if provided
    keyword_list = [k.strip().lower() for k in keywords if k and k.strip()] if keywords else []

    if not search_query and not keyword_list:
         # If you want to prevent searches without criteria, raise error here:
         # raise ValueError("Search query or keywords must be provided.")
         # Otherwise, it will return all non-trashed items paginated.
         logging.debug("Executing search with no query or keywords - will return all non-trashed items.")

    # --- Query Building ---
    offset = (page - 1) * results_per_page
    params = []
    conditions = ["Media.is_trash = 0"] # Always exclude trashed items by default

    # Build search query conditions
    fts_fields = {"title", "content"}
    like_fields = {"author", "type"} # Fields suitable for LIKE

    # Fields requested for searching that can use FTS
    fts_search_requested = bool(set(sanitized_fields) & fts_fields)
    # Fields requested for searching that should use LIKE
    like_search_requested = list(set(sanitized_fields) & like_fields)

    if search_query:
        # Use FTS if title or content are among the search fields
        if fts_search_requested:
            # Assumes media_fts table exists and has columns mirroring Media.id, title, content
            # Basic MATCH query. Consider adding field specifiers e.g., "title:term content:term"
            # Or use more complex FTS queries if needed.
            conditions.append("Media.id IN (SELECT rowid FROM media_fts WHERE media_fts MATCH ?)")
            params.append(search_query) # Simple phrase search
            logging.debug(f"Using FTS search for query: '{search_query}'")

        # Use LIKE for other specified fields
        if like_search_requested:
            like_conditions = []
            for field in like_search_requested:
                # Ensure field name is safe (already validated against valid_fields)
                like_conditions.append(f"Media.{field} LIKE ? COLLATE NOCASE") # Case-insensitive LIKE
                params.append(f"%{search_query}%")
            if like_conditions:
                conditions.append(f"({' OR '.join(like_conditions)})")
                logging.debug(f"Using LIKE search on fields: {like_search_requested} for query: '{search_query}'")

    # Build keyword conditions
    if keyword_list:
        placeholders = ", ".join(["?"] * len(keyword_list))
        # Use a subquery checking MediaKeywords and Keywords tables
        conditions.append(f"""
            Media.id IN (
                SELECT mk.media_id
                FROM MediaKeywords mk
                JOIN Keywords k ON mk.keyword_id = k.id
                WHERE k.keyword IN ({placeholders})
                GROUP BY mk.media_id
                HAVING COUNT(DISTINCT k.id) = ?
            )
        """) # This ensures ALL provided keywords match for a media item
        params.extend(keyword_list)
        params.append(len(keyword_list)) # Add count for HAVING clause
        logging.debug(f"Filtering by ALL keywords: {keyword_list}")
        # If you want ANY keyword match, remove GROUP BY/HAVING and use EXISTS or simple IN

    # Construct WHERE clause
    where_clause = " AND ".join(conditions) if conditions else "1=1" # Should always have is_trash = 0

    # --- Database Interaction ---
    try:
        # Use transaction context for connection management
        with db_instance.transaction() as conn:
            cursor = conn.cursor()

            # 1. Get the total count matching the criteria
            count_query = f"SELECT COUNT(Media.id) FROM Media WHERE {where_clause}"
            logging.debug(f"Executing Count Query on {db_instance.db_path}: {count_query} | Params: {params}")
            cursor.execute(count_query, tuple(params)) # Use tuple for params
            count_result = cursor.fetchone()
            total_matches = count_result[0] if count_result else 0
            logging.debug(f"Total matches found: {total_matches}")

            # 2. Get the paginated results
            results_list = []
            if total_matches > 0:
                # Select desired columns from Media table
                # Note: Does not fetch keywords here, assumes caller will fetch if needed
                results_query = f"""
                    SELECT
                        Media.id, Media.url, Media.title, Media.type,
                        Media.content, Media.author, Media.ingestion_date,
                        Media.transcription_model, Media.chunking_status, Media.content_hash
                    FROM Media
                    WHERE {where_clause}
                    ORDER BY Media.ingestion_date DESC, Media.id DESC -- Sort by date then ID
                    LIMIT ? OFFSET ?
                """
                paginated_params = tuple(params + [results_per_page, offset]) # Add limit/offset
                logging.debug(f"Executing Results Query on {db_instance.db_path} | Limit={results_per_page}, Offset={offset} | Params: {paginated_params}")
                cursor.execute(results_query, paginated_params)
                results_raw = cursor.fetchall() # Returns list of Row objects

                # Convert Row objects to standard dictionaries
                results_list = [dict(row) for row in results_raw]
                logging.debug(f"Fetched {len(results_list)} results for page {page}")

            # Fetch keywords separately if needed by caller (example commented out)
            # if results_list:
            #     media_ids = [r['id'] for r in results_list]
            #     keywords_map = fetch_keywords_for_media_batch(media_ids, db_instance) # Assumes this helper exists
            #     for res in results_list:
            #         res['keywords'] = keywords_map.get(res['id'], [])

            return results_list, total_matches

    except sqlite3.Error as e:
        logging.error(f"SQLite error in search_media_db on {db_instance.db_path}: {e}", exc_info=True)
        raise DatabaseError(f"Failed to search media database: {e}") from e
    except Exception as e:
        logging.error(f"Unexpected error in search_media_db on {db_instance.db_path}: {e}", exc_info=True)
        raise DatabaseError(f"An unexpected error occurred during media search: {e}") from e


# Gradio function to handle user input and display results with pagination, with better feedback
def search_and_display(search_query, search_fields, keywords, page, db_instance: Database):
    results = search_media_db(search_query, search_fields, keywords, page, results_per_page=20, db_instance=db_instance)

    if isinstance(results, pd.DataFrame):
        # Convert DataFrame to a list of tuples or lists
        processed_results = results.values.tolist()  # This converts DataFrame rows to lists
    elif isinstance(results, list):
        # Ensure that each element in the list is itself a list or tuple (not a dictionary)
        processed_results = [list(item.values()) if isinstance(item, dict) else item for item in results]
    else:
        raise TypeError("Unsupported data type for results")

    return processed_results


def display_details(index, results):
    if index is None or results is None:
        return "Please select a result to view details."

    try:
        # Ensure the index is an integer and access the row properly
        index = int(index)
        if isinstance(results, pd.DataFrame):
            if index >= len(results):
                return "Index out of range. Please select a valid index."
            selected_row = results.iloc[index]
        else:
            # If results is not a DataFrame, but a list (assuming list of dicts)
            selected_row = results[index]
    except ValueError:
        return "Index must be an integer."
    except IndexError:
        return "Index out of range. Please select a valid index."

    # Build HTML output safely
    details_html = f"""
    <h3>{selected_row.get('Title', 'No Title')}</h3>
    <p><strong>URL:</strong> {selected_row.get('URL', 'No URL')}</p>
    <p><strong>Type:</strong> {selected_row.get('Type', 'No Type')}</p>
    <p><strong>Author:</strong> {selected_row.get('Author', 'No Author')}</p>
    <p><strong>Ingestion Date:</strong> {selected_row.get('Ingestion Date', 'No Date')}</p>
    <p><strong>Prompt:</strong> {selected_row.get('Prompt', 'No Prompt')}</p>
    <p><strong>analysis_content:</strong> {selected_row.get('analysis_content', 'No analysis_content')}</p>
    <p><strong>Content:</strong> {selected_row.get('Content', 'No Content')}</p>
    """
    return details_html


def get_details(index, dataframe):
    if index is None or dataframe is None or index >= len(dataframe):
        return "Please select a result to view details."
    row = dataframe.iloc[index]
    details = f"""
    <h3>{row['Title']}</h3>
    <p><strong>URL:</strong> {row['URL']}</p>
    <p><strong>Type:</strong> {row['Type']}</p>
    <p><strong>Author:</strong> {row['Author']}</p>
    <p><strong>Ingestion Date:</strong> {row['Ingestion Date']}</p>
    <p><strong>Prompt:</strong> {row['Prompt']}</p>
    <p><strong>analysis_content:</strong> {row['analysis_content']}</p>
    <p><strong>Content:</strong></p>
    <pre>{row['Content']}</pre>
    """
    return details


def format_results(results):
    if not results:
        return pd.DataFrame(columns=['URL', 'Title', 'Type', 'Content', 'Author', 'Ingestion Date', 'Prompt', 'analysis_content'])

    df = pd.DataFrame(results, columns=['URL', 'Title', 'Type', 'Content', 'Author', 'Ingestion Date', 'Prompt', 'analysis_content'])
    logging.debug(f"Formatted DataFrame: {df}")

    return df


def fetch_keywords_for_media_batch(media_ids: List[int], db_instance: Database) -> Dict[int, List[str]]:
    """
    Placeholder: Fetches keywords for multiple media IDs efficiently.

    Args:
        media_ids: A list of media IDs.
        db_instance: The Database instance.

    Returns:
        A dictionary mapping media_id to a list of its keywords.
    """
    if not media_ids:
        return {}

    keywords_map = {media_id: [] for media_id in media_ids}
    placeholders = ','.join('?' * len(media_ids))
    query = f"""
        SELECT mk.media_id, k.keyword
        FROM MediaKeywords mk
        JOIN Keywords k ON mk.keyword_id = k.id
        WHERE mk.media_id IN ({placeholders})
        ORDER BY mk.media_id, k.keyword COLLATE NOCASE
    """
    try:
        cursor = db_instance.execute_query(query, tuple(media_ids))
        for row in cursor.fetchall():
            keywords_map[row['media_id']].append(row['keyword'])
        return keywords_map
    except (DatabaseError, sqlite3.Error) as e:
         logging.error(f"Failed to fetch keywords batch: {e}", exc_info=True)
         return keywords_map # Return empty lists on error


# Function to export search results to CSV or markdown with pagination
def export_to_file(
    search_query: str,
    search_fields: List[str],
    keywords: List[str], # Renamed parameter
    db_instance: Database, # Added db_instance parameter
    page: int = 1,
    results_per_page: int = 1000, # Renamed parameter for consistency
    export_format: str = 'csv'
) -> str: # Return status message string
    """
    Exports search results to a CSV or Markdown file.

    Args:
        search_query: The term to search for.
        search_fields: List of fields ('title', 'content', 'author', 'type') to search.
        keywords: A list of keywords to filter by.
        db_instance: The Database instance to use for searching and fetching keywords.
        page: The page number for pagination.
        results_per_page: Max number of results to fetch and write to the file.
        export_format: The desired output format ('csv' or 'markdown').

    Returns:
        A string indicating the success or failure message.
    """
    try:
        logging.info(f"Exporting search results: page={page}, format={export_format}, query='{search_query}', keywords={keywords}")

        # 1. Perform the search using the updated search_media_db
        results_list, total_matches = search_media_db(
            search_query=search_query,
            search_fields=search_fields,
            keywords=keywords,
            page=page,
            results_per_page=results_per_page,
            db_instance=db_instance # Pass the db instance
        )

        if not results_list:
            logging.warning("No results found for the given search criteria.")
            return "No results found to export."

        logging.info(f"Found {len(results_list)} results (out of {total_matches} total) for page {page} to export.")

        # 2. Fetch keywords for the found results (efficiently)
        media_ids = [item['id'] for item in results_list]
        keywords_map = fetch_keywords_for_media_batch(media_ids, db_instance)

        # 3. Create 'exports' directory if needed
        export_dir = 'exports'
        try:
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
                logging.info(f"Created export directory: {export_dir}")
        except OSError as e:
             logging.error(f"Failed to create export directory '{export_dir}': {e}", exc_info=True)
             # Decide whether to raise or return error message
             return f"Error: Could not create export directory '{export_dir}'."


        # Define base filename
        # Consider adding query/keywords to filename for clarity if desired
        sanitized_query = "".join(c if c.isalnum() else "_" for c in search_query[:20]) if search_query else "all"
        filename_base = f"search_results_{sanitized_query}_page_{page}"

        # 4. Handle Export based on format
        if export_format.lower() == 'csv':
            filename = os.path.join(export_dir, f'{filename_base}.csv')
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as file:
                    # Define header based on available data + fetched keywords
                    header = ['ID', 'URL', 'Title', 'Type', 'Author', 'Ingestion Date', 'Keywords', 'Content']
                    writer = csv.writer(file)
                    writer.writerow(header)

                    for item in results_list:
                        item_keywords = keywords_map.get(item['id'], [])
                        # Write row using dictionary keys, join keywords for CSV cell
                        writer.writerow([
                            item.get('id'),
                            item.get('url'),
                            item.get('title'),
                            item.get('type'),
                            item.get('author'),
                            item.get('ingestion_date'),
                            ', '.join(item_keywords), # Join keywords list into string
                            item.get('content') # Content last as it can be long
                        ])
                logging.info(f"Successfully exported {len(results_list)} results to CSV: {filename}")
                return f"Results exported successfully to {filename}"
            except IOError as e:
                 logging.error(f"Failed to write CSV file '{filename}': {e}", exc_info=True)
                 return f"Error: Failed to write CSV file '{filename}'."


        elif export_format.lower() == 'markdown':
            filename = os.path.join(export_dir, f'{filename_base}.md')
            try:
                with open(filename, 'w', encoding='utf-8') as file:
                    for i, item in enumerate(results_list):
                        # Prepare data dict for the conversion helper
                        item_data_for_md = {
                            'id': item.get('id'),
                            'title': item.get('title'),
                            'url': item.get('url'),
                            'type': item.get('type'),
                            'content': item.get('content'),
                            'author': item.get('author'),
                            'ingestion_date': item.get('ingestion_date'),
                            'keywords': keywords_map.get(item['id'], []) # Pass keywords as list
                        }
                        markdown_content = convert_to_markdown(item_data_for_md)
                        file.write(markdown_content)
                        # Add separator only if it's not the last item
                        if i < len(results_list) - 1:
                            file.write("\n\n---\n\n") # Separator between items

                logging.info(f"Successfully exported {len(results_list)} results to Markdown: {filename}")
                return f"Results exported successfully to {filename}"
            except IOError as e:
                 logging.error(f"Failed to write Markdown file '{filename}': {e}", exc_info=True)
                 return f"Error: Failed to write Markdown file '{filename}'."

        else:
            logging.warning(f"Unsupported export format requested: {export_format}")
            return f"Error: Unsupported export format '{export_format}'. Use 'csv' or 'markdown'."

    # Catch specific DB errors and general exceptions
    except (DatabaseError, InputError) as db_err:
        logging.error(f"Database or Input Error during export: {db_err}", exc_info=True)
        return f"Export failed due to database/input error: {db_err}"
    except Exception as e:
        logging.error(f"An unexpected error occurred during export: {e}", exc_info=True)
        return f"An unexpected error occurred during export: {e}"


# Helper function to validate date format
def is_valid_date(date_string: str) -> bool:
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def check_existing_media(url, db_instance: Database):
    db = db_instance
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM Media WHERE url = ?', (url,))
            result = cursor.fetchone()
            return {'id': result[0]} if result else None
    except Exception as e:
        logging.error(f"Error checking existing media: {e}")
        return None


# FIXME: This function is not complete and needs to be implemented
def schedule_chunking(db_instance: Database, media_id: int, content: str, media_name: str, media_type: str = None, chunk_options: dict = None):
    db = db_instance
    try:
        # Ensure chunk_options is provided; if not, use defaults.
        if chunk_options is None:
            chunk_options = {'method': 'words', 'max_size': 300, 'overlap': 0}

        # Retrieve the values from chunk_options as provided.
        method = chunk_options.get('method', 'words')
        max_size = chunk_options.get('max_size', 300)  # preserve original type (could be str or int)
        overlap = chunk_options.get('overlap', 0)        # preserve original type (could be str or int)

        # Convert max_size and overlap to integers for arithmetic without modifying the original chunk_options
        try:
            max_size_int = int(max_size)
            overlap_int = int(overlap)
        except ValueError as e:
            logging.error(f"Error converting chunk_options values to int: {e}")
            raise

        # Use the converted integers when calling the chunking function.
        chunks = chunk_text(content, method, max_size_int, overlap_int)

        with db.get_connection() as conn:
            cursor = conn.cursor()
            for i, chunk in enumerate(chunks):
                # Calculate start and end indices for the chunk using the integer values
                start_index = i * max_size_int
                end_index = min((i + 1) * max_size_int, len(content))
                cursor.execute('''
                    INSERT INTO MediaChunks (media_id, chunk_text, start_index, end_index, chunk_id)
                    VALUES (?, ?, ?, ?, ?)
                    ''', (media_id, chunk, start_index, end_index, f"{media_id}_chunk_{i}"))
            conn.commit()

        # Update chunking status in the Media table.
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE Media SET chunking_status = 'completed' WHERE id = ?", (media_id,))
            conn.commit()

    except Exception as e:
        logging.error(f"Error scheduling chunking for media_id {media_id}: {str(e)}")
        # You might want to update the chunking_status to 'failed' here

#
# End of ....
#######################################################################################################################


#######################################################################################################################
#
# Function to fetch/update media content

# FIXME - REWRITE TO NOT USE MEDIAMODIFICATIONS
def update_media_content(selected_item, item_mapping, content_input, prompt_input, analysis_content, db_instance: Database):
    db=db_instance
    try:
        if selected_item and item_mapping and selected_item in item_mapping:
            media_id = item_mapping[selected_item]

            with db.get_connection() as conn:
                cursor = conn.cursor()

                # Update the main content in the Media table
                cursor.execute("UPDATE Media SET content = ? WHERE id = ?", (content_input, media_id))

                # Check if a row already exists in MediaModifications for this media_id
                cursor.execute("SELECT COUNT(*) FROM MediaModifications WHERE media_id = ?", (media_id,))
                exists = cursor.fetchone()[0] > 0

                if exists:
                    # Update existing row
                    cursor.execute("""
                        UPDATE MediaModifications
                        SET prompt = ?, analysis_content = ?, modification_date = CURRENT_TIMESTAMP
                        WHERE media_id = ?
                    """, (prompt_input, analysis_content, media_id))
                else:
                    # Insert new row
                    cursor.execute("""
                        INSERT INTO MediaModifications (media_id, prompt, analysis_content, modification_date)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    """, (media_id, prompt_input, analysis_content))

                # Create new document version
                new_version = create_document_version(media_id, content_input)

                conn.commit()

            return f"Content updated successfully for media ID: {media_id}. New version: {new_version}"
        else:
            return "No item selected or invalid selection"
    except Exception as e:
        logging.error(f"Error updating media content: {e}")
        return f"Error updating content: {str(e)}"


def search_media_database(db_instance: Database, query: str, connection=None) -> List[Tuple[int, str, str]]:
    def execute_query(conn):
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title, url FROM Media WHERE title LIKE ?", (f'%{query}%',))
            return cursor.fetchall()
        except sqlite3.Error as e:
            raise Exception(f"Error searching media database: {e}")

    if connection:
        return execute_query(connection)
    else:
        db=db_instance
        with db.get_connection() as conn:
            return execute_query(conn)


def load_media_content(media_id: int, db_instance: Database) -> dict:
    db=db_instance
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content, prompt, analysis_content FROM Media WHERE id = ?", (media_id,))
            result = cursor.fetchone()
            if result:
                return {
                    "content": result[0],
                    "prompt": result[1],
                    "analysis_content": result[2]
                }
            return {"content": "", "prompt": "", "analysis_content": ""}
    except sqlite3.Error as e:
        raise Exception(f"Error loading media content: {e}")


def fetch_items_by_title_or_url(search_query: str, search_type: str, db_instance: Database):
    db=db_instance
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            if search_type == 'Title':
                cursor.execute("SELECT id, title, url FROM Media WHERE title LIKE ?", (f'%{search_query}%',))
            elif search_type == 'URL':
                cursor.execute("SELECT id, title, url FROM Media WHERE url LIKE ?", (f'%{search_query}%',))
            results = cursor.fetchall()
            return results
    except sqlite3.Error as e:
        raise DatabaseError(f"Error fetching items by {search_type}: {e}")


def fetch_items_by_keyword(search_query: str, db_instance: Database):
    db=db_instance
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT m.id, m.title, m.url
                FROM Media m
                JOIN MediaKeywords mk ON m.id = mk.media_id
                JOIN Keywords k ON mk.keyword_id = k.id
                WHERE k.keyword LIKE ?
            """, (f'%{search_query}%',))
            results = cursor.fetchall()
            return results
    except sqlite3.Error as e:
        raise DatabaseError(f"Error fetching items by keyword: {e}")


def fetch_items_by_content(search_query: str, db_instance: Database):
    db=db_instance
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title, url FROM Media WHERE content LIKE ?", (f'%{search_query}%',))
            results = cursor.fetchall()
            return results
    except sqlite3.Error as e:
        raise DatabaseError(f"Error fetching items by content: {e}")


# FIXME REWRITE THIS FUNCTION
def fetch_item_details_single(media_id: int, db_instance: Database):
    db=db_instance
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT prompt, analysis_content 
                FROM MediaModifications 
                WHERE media_id = ? 
                ORDER BY modification_date DESC 
                LIMIT 1
            """, (media_id,))
            prompt_analysis_result = cursor.fetchone()
            cursor.execute("SELECT content FROM Media WHERE id = ?", (media_id,))
            content_result = cursor.fetchone()

            prompt = prompt_analysis_result[0] if prompt_analysis_result else "No prompt available."
            analysis_content = prompt_analysis_result[1] if prompt_analysis_result else "No analysis_content available."
            content = content_result[0] if content_result else "No content available."

            return prompt, analysis_content, content
    except sqlite3.Error as e:
        logging.error(f"Error fetching item details: {e}")
        return "Error fetching prompt.", "Error fetching analysis_content.", "Error fetching content."


def convert_to_markdown(item_data: Dict[str, Any]) -> str:
    """
    Placeholder function to convert a media item dictionary to a Markdown string.
    Replace with your actual implementation.
    """
    lines = []
    lines.append(f"# {item_data.get('title', 'No Title')}")
    if item_data.get('url'):
        lines.append(f"**URL:** {item_data['url']}")
    if item_data.get('author'):
        lines.append(f"**Author:** {item_data['author']}")
    if item_data.get('ingestion_date'):
        lines.append(f"**Date:** {item_data['ingestion_date']}")
    if item_data.get('type'):
        lines.append(f"**Type:** {item_data['type']}")
    if item_data.get('keywords'):
        lines.append(f"**Keywords:** {', '.join(item_data['keywords'])}")
    lines.append("\n## Content")
    lines.append(f"```\n{item_data.get('content', 'No Content')}\n```") # Basic content block
    return "\n".join(lines)


# Gradio function to handle user input and display results with pagination for displaying entries in the DB
def fetch_paginated_data(page: int, results_per_page: int, db_instance: Database) -> Tuple[List[Tuple], int]:
    db=db_instance
    try:
        offset = (page - 1) * results_per_page
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM Media")
            total_entries = cursor.fetchone()[0]

            cursor.execute("SELECT id, title, url FROM Media LIMIT ? OFFSET ?", (results_per_page, offset))
            results = cursor.fetchall()

        return results, total_entries
    except sqlite3.Error as e:
        raise Exception(f"Error fetching paginated data: {e}")

def format_results_as_html(results: List[Tuple]) -> str:
    html = "<table class='table table-striped'>"
    html += "<tr><th>ID</th><th>Title</th><th>URL</th></tr>"
    for row in results:
        html += f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td></tr>"
    html += "</table>"
    return html

def view_database(page: int, results_per_page: int, db_instance) -> Tuple[str, str, int]:
    results, total_entries = fetch_paginated_data(page, results_per_page, db_instance)
    formatted_results = format_results_as_html(results)
    # Calculate total pages
    total_pages = (total_entries + results_per_page - 1) // results_per_page
    return formatted_results, f"Page {page} of {total_pages}", total_pages


# FIXME - Proper DB usage
def search_and_display_items(query, search_type, page, entries_per_page,char_count, db_instance: Database):
    offset = (page - 1) * entries_per_page
    db=db_instance
    try:
        with sqlite3.connect('./Databases/server_media_summary.db') as conn:
            cursor = conn.cursor()

            # Adjust the SQL query based on the search type
            if search_type == "Title":
                where_clause = "WHERE m.title LIKE ?"
            elif search_type == "URL":
                where_clause = "WHERE m.url LIKE ?"
            elif search_type == "Keyword":
                where_clause = "WHERE k.keyword LIKE ?"
            elif search_type == "Content":
                where_clause = "WHERE m.content LIKE ?"
            else:
                raise ValueError("Invalid search type")

            cursor.execute(f'''
                SELECT m.id, m.title, m.url, m.content, mm.analysis_content, GROUP_CONCAT(k.keyword, ', ') as keywords
                FROM Media m
                LEFT JOIN MediaModifications mm ON m.id = mm.media_id
                LEFT JOIN MediaKeywords mk ON m.id = mk.media_id
                LEFT JOIN Keywords k ON mk.keyword_id = k.id
                {where_clause}
                GROUP BY m.id
                ORDER BY m.ingestion_date DESC
                LIMIT ? OFFSET ?
            ''', (f'%{query}%', entries_per_page, offset))
            items = cursor.fetchall()

            cursor.execute(f'''
                SELECT COUNT(DISTINCT m.id)
                FROM Media m
                LEFT JOIN MediaKeywords mk ON m.id = mk.media_id
                LEFT JOIN Keywords k ON mk.keyword_id = k.id
                {where_clause}
            ''', (f'%{query}%',))
            total_items = cursor.fetchone()[0]

        results = ""
        for item in items:
            title = html.escape(item[1]).replace('\n', '<br>')
            url = html.escape(item[2]).replace('\n', '<br>')
            # First X amount of characters of the content
            content = html.escape(item[3] or '')[:char_count] + '...'
            analysis_content = html.escape(item[4] or '').replace('\n', '<br>')
            keywords = html.escape(item[5] or '').replace('\n', '<br>')

            results += f"""
            <div style="border: 1px solid #ddd; padding: 10px; margin-bottom: 20px;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div><strong>Title:</strong> {title}</div>
                    <div><strong>URL:</strong> {url}</div>
                </div>
                <div style="margin-top: 10px;">
                    <strong>Content (first {char_count} characters):</strong>
                    <pre style="white-space: pre-wrap; word-wrap: break-word;">{content}</pre>
                </div>
                <div style="margin-top: 10px;">
                    <strong>analysis_content:</strong>
                    <pre style="white-space: pre-wrap; word-wrap: break-word;">{analysis_content}</pre>
                </div>
                <div style="margin-top: 10px;">
                    <strong>Keywords:</strong> {keywords}
                </div>
            </div>
            """

        total_pages = (total_items + entries_per_page - 1) // entries_per_page
        pagination = f"Page {page} of {total_pages} (Total items: {total_items})"

        return results, pagination, total_pages
    except sqlite3.Error as e:
        return f"<p>Error searching items: {e}</p>", "Error", 0

#
# End of Functions to manage prompts DB / Fetch and update media content
#######################################################################################################################


#######################################################################################################################
#
# Obsidian-related Functions

def import_obsidian_note_to_db(note_data, db_instance: Database):
    """
    Imports or updates an Obsidian note into the database, including tags and
    creating an initial document version containing the frontmatter.

    Args:
        note_data (dict): Dictionary containing note info ('title', 'content', 'tags', 'frontmatter', 'file_path').
        db_instance (Database, optional): The Database instance. Defaults to None (uses global).

    Returns:
        Tuple[bool, Optional[str]]: (success_status, error_message_or_none)
    """
    db=db_instance
    if db is None:
        logging.warning("import_obsidian_note_to_db called without explicit db instance.")
        raise ValueError("Global db instance not found and no instance passed.")

    existing_note = None # Define outside try block for use in logging/return
    try:
        with db.transaction() as conn: # Use transaction context
            cursor = conn.cursor()

            # Check if note exists by title and type
            cursor.execute("SELECT id FROM Media WHERE title = ? AND type = 'obsidian_note'", (note_data['title'],))
            existing_note = cursor.fetchone()

            # Use relative path or title as URL/identifier if file_path might change
            # Consider using a more stable ID if possible. Using title for now as fallback.
            url_identifier = note_data['title'] # Or generate a unique ID based on vault/path

            media_id = None
            action = "Imported"

            if existing_note:
                media_id = existing_note[0]
                action = "Updated"
                # Update Media record
                cursor.execute("""
                    UPDATE Media
                    SET content = ?, author = ?, ingestion_date = CURRENT_TIMESTAMP, url = ?
                       -- Removed prompt = ?, analysis_content = ?
                    WHERE id = ?
                """, (note_data['content'], note_data['frontmatter'].get('author', 'Unknown'),
                      url_identifier, media_id))
                # Clear old keywords before adding new ones
                cursor.execute("DELETE FROM MediaKeywords WHERE media_id = ?", (media_id,))
            else:
                 # Insert new Media record
                 cursor.execute("""
                    INSERT INTO Media (title, content, type, author, ingestion_date, url, chunking_status)
                    VALUES (?, ?, 'obsidian_note', ?, CURRENT_TIMESTAMP, ?, ?)
                 """, (note_data['title'], note_data['content'], note_data['frontmatter'].get('author', 'Unknown'),
                       url_identifier, 'pending'))
                 media_id = cursor.lastrowid # Get the new ID

            if media_id is None:
                 raise DatabaseError("Failed to get media_id after insert/update.")

            # --- Handle Keywords (Tags) ---
            tags = note_data.get('tags', [])
            if tags:
                keyword_params = [(tag.strip().lower(),) for tag in tags if tag.strip()]
                if keyword_params:
                    cursor.executemany('INSERT OR IGNORE INTO Keywords (keyword) VALUES (?)', keyword_params)

                    # Get keyword IDs
                    placeholders = ','.join(['?'] * len(keyword_params))
                    cursor.execute(f'SELECT id, keyword FROM Keywords WHERE keyword IN ({placeholders})', [p[0] for p in keyword_params])
                    keyword_id_map = {kw: kid for kid, kw in cursor.fetchall()}

                    # Insert media-keyword associations
                    media_keyword_params = [(media_id, keyword_id_map[kw]) for kw, kid in keyword_id_map.items()]
                    if media_keyword_params:
                        cursor.executemany('INSERT OR IGNORE INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)', media_keyword_params)

            # --- Create Initial Document Version (Store frontmatter here) ---
            frontmatter_str = yaml.dump(note_data.get('frontmatter', {}))
            # Store frontmatter YAML string in the 'analysis_content' field of the version
            version_result = create_document_version(
                media_id=media_id,
                content=note_data['content'],
                prompt="Obsidian Frontmatter", # Use a standard prompt text
                analysis_content=frontmatter_str,
                db_instance=db # Pass connection
            )
            if 'error' in version_result:
                 raise DatabaseError(f"Failed to create document version for Obsidian note: {version_result['error']}")

            # --- Update FTS ---
            cursor.execute('INSERT OR REPLACE INTO media_fts (rowid, title, content) VALUES (?, ?, ?)',
                           (media_id, note_data['title'], note_data['content']))

            # Transaction commits automatically

        logger.info(f"{action} Obsidian note: {note_data['title']} (ID: {media_id})")
        return True, None # Success

    except sqlite3.Error as e:
        status_verb = 'updating' if existing_note else 'importing'
        error_msg = f"Database error {status_verb} note '{note_data.get('title', 'N/A')}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg
    except Exception as e:
        status_verb = 'updating' if existing_note else 'importing'
        error_msg = f"Unexpected error {status_verb} note '{note_data.get('title', 'N/A')}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg


#
# End of Obsidian-related Functions
#######################################################################################################################


#######################################################################################################################
#
# Functions to Compare Transcripts

# Fetch Transcripts
def get_transcripts(media_id, db_instance: Database):
    db=db_instance
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT id, whisper_model, transcription, created_at
            FROM Transcripts
            WHERE media_id = ?
            ORDER BY created_at DESC
            ''', (media_id,))
            return cursor.fetchall()
    except Exception as e:
        logging.error(f"Error in get_transcripts: {str(e)}")
        return []

def get_latest_transcription(media_id: int, db_instance: Database):
    db=db_instance
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT transcription
                FROM Transcripts
                WHERE media_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (media_id,))
            result = cursor.fetchone()
            return result[0] if result else "No transcription available."
    except sqlite3.Error as e:
        logging.error(f"Error fetching latest transcription: {e}")
        return "Error fetching transcription."

#
# End of Functions to Compare Transcripts
#######################################################################################################################


#######################################################################################################################
#
# Functions to handle deletion of media items


def mark_as_trash(media_id: int, db_instance: Database) -> None:
    db=db_instance
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE Media 
            SET is_trash = 1, trash_date = ?
            WHERE id = ?
        """, (datetime.now(), media_id))
        conn.commit()


def restore_from_trash(media_id: int, db_instance: Database) -> None:
    db=db_instance
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE Media 
            SET is_trash = 0, trash_date = NULL
            WHERE id = ?
        """, (media_id,))
        conn.commit()


def get_trashed_items(db_instance: Database) -> List[Dict]:
    db=db_instance
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, trash_date 
            FROM Media 
            WHERE is_trash = 1
            ORDER BY trash_date DESC
        """)
        return [{'id': row[0], 'title': row[1], 'trash_date': row[2]} for row in cursor.fetchall()]


# FIXME - REWRITE TO NOT USE MEDIAMODIFICATIOSN
def permanently_delete_item(media_id: int, db_instance: Database) -> None:
    db=db_instance
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM Media WHERE id = ?", (media_id,))
        cursor.execute("DELETE FROM MediaKeywords WHERE media_id = ?", (media_id,))
        cursor.execute("DELETE FROM MediaVersion WHERE media_id = ?", (media_id,))
        cursor.execute("DELETE FROM MediaModifications WHERE media_id = ?", (media_id,))
        cursor.execute("DELETE FROM media_fts WHERE rowid = ?", (media_id,))
        conn.commit()


def empty_trash(days_threshold: int, db_instance: Database) -> Tuple[int, int]:
    db=db_instance
    threshold_date = datetime.now() - timedelta(days=days_threshold)
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id FROM Media 
            WHERE is_trash = 1 AND trash_date <= ?
        """, (threshold_date,))
        old_items = cursor.fetchall()

        for item in old_items:
            permanently_delete_item(item[0], db_instance)

        cursor.execute("""
            SELECT COUNT(*) FROM Media 
            WHERE is_trash = 1 AND trash_date > ?
        """, (threshold_date,))
        remaining_items = cursor.fetchone()[0]

    return len(old_items), remaining_items


def user_delete_item(db_instance: Database, media_id: int, force: bool = False) -> str:
    db=db_instance
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT is_trash, trash_date FROM Media WHERE id = ?", (media_id,))
        result = cursor.fetchone()

        if not result:
            return "Item not found."

        is_trash, trash_date = result

        if not is_trash:
            mark_as_trash(media_id, db_instance)
            return "Item moved to trash."

        if force or (trash_date and (datetime.now() - trash_date).days >= 30):
            permanently_delete_item(media_id, db_instance)
            return "Item permanently deleted."
        else:
            return "Item is already in trash. Use force=True to delete permanently before 30 days."


def get_chunk_text(media_id: int, chunk_index: int, db_instance: Database) -> str:
    db=db_instance
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM MediaChunks WHERE media_id = ? AND chunk_index = ?",
                       (media_id, chunk_index))
        result = cursor.fetchone()
    return result[0] if result else None

def get_full_document(media_id: int, db_instance: Database) -> str:
    db=db_instance
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM Media WHERE id = ?", (media_id,))
        result = cursor.fetchone()
    return result[0] if result else None

def get_all_content_from_database(db_instance: Database) -> List[Dict[str, Any]]:
    """
    Retrieve all media content from the database that requires embedding.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing the media ID, content, title, and other relevant fields.
    """
    db=db_instance
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, content, title, author, type
                FROM Media
                WHERE is_trash = 0  -- Exclude items marked as trash
            """)
            media_items = cursor.fetchall()

            all_content = [
                {
                    'id': item[0],
                    'content': item[1],
                    'title': item[2],
                    'author': item[3],
                    'type': item[4]
                }
                for item in media_items
            ]

        return all_content

    except sqlite3.Error as e:
        logger.error(f"Error retrieving all content from database: {e}")
        raise DatabaseError(f"Error retrieving all content from database: {e}")


def get_media_content(media_id: int, db_instance: Database) -> str:
    db=db_instance
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content FROM Media WHERE id = ?", (media_id,))
            result = cursor.fetchone()
            if result is None:
                raise ValueError(f"No media found with id {media_id}")
            return result[0]
    except sqlite3.Error as e:
        logging.error(f"Database error in get_media_content: {e}")
        raise DatabaseError(f"Failed to retrieve media content: {e}")
    except Exception as e:
        logging.error(f"Unexpected error in get_media_content: {e}")
        raise

def get_media_title(media_id: int, db_instance: Database) -> str:
    db=db_instance
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT title FROM Media WHERE id = ?", (media_id,))
            result = cursor.fetchone()
            return result[0] if result else f"Unknown Source (ID: {media_id})"
    except sqlite3.Error as e:
        logging.error(f"Database error in get_media_title: {e}")
        return f"Unknown Source (ID: {media_id})"

def get_media_transcripts(media_id, db_instance: Database):
    db=db_instance
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT id, whisper_model, transcription, created_at
            FROM Transcripts
            WHERE media_id = ?
            ORDER BY created_at DESC
            ''', (media_id,))
            results = cursor.fetchall()
            return [
                {
                    'id': row[0],
                    'whisper_model': row[1],
                    'content': row[2],
                    'created_at': row[3]
                }
                for row in results
            ]
    except Exception as e:
        logging.error(f"Error in get_media_transcripts: {str(e)}")
        return []

def get_specific_transcript(transcript_id: int, db_instance: Database) -> Dict:
    db=db_instance
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            SELECT id, whisper_model, transcription, created_at
            FROM Transcripts
            WHERE id = ?
            ''', (transcript_id,))
            result = cursor.fetchone()
            if result:
                return {
                    'id': result[0],
                    'whisper_model': result[1],
                    'content': result[2],
                    'created_at': result[3]
                }
            return {'error': f"No transcript found with ID {transcript_id}"}
    except Exception as e:
        logging.error(f"Error in get_specific_transcript: {str(e)}")
        return {'error': f"Error retrieving transcript: {str(e)}"}


def get_media_summaries(media_id: int, db_instance: Database) -> List[Dict]:
    """
    Retrieves all non-empty analysis content entries (summaries) from the
    DocumentVersions table for a given media ID, ordered by version number descending.

    Args:
        media_id: The ID of the media item.
        db_instance: The Database instance.

    Returns:
        A list of dictionaries, each containing the version ID ('id'),
        the analysis content ('content'), and the creation timestamp ('created_at').
        Returns an empty list on error or if none found.
    """
    db=db_instance
    if db is None:
        logging.warning("get_media_summaries called without explicit db instance.")
        raise ValueError("Global db instance not found and no instance passed.")

    try:
        with db.get_connection() as conn:
            # Use the row factory for dict-like access
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Fetch id, analysis_content, and created_at from DocumentVersions
            cursor.execute('''
                           SELECT id, analysis_content, created_at
                           FROM DocumentVersions
                           WHERE media_id = ?
                             AND analysis_content IS NOT NULL
                             AND analysis_content != ''
                           ORDER BY version_number DESC
                           ''', (media_id,))
            results = cursor.fetchall()
            # Format results into the expected dictionary structure
            return [
                {
                    'id': row['id'],  # Use DocumentVersions.id
                    'content': row['analysis_content'],  # The actual summary content
                    'created_at': row['created_at']  # Timestamp of the version creation
                }
                for row in results
            ]
    except Exception as e:
        logging.error(f"Error in get_media_summaries for media_id {media_id}: {str(e)}", exc_info=True)
        return []  # Return empty list on error


def get_specific_analysis(version_id: int, db_instance: Database) -> Dict:
    """
    Retrieves the analysis content for a specific document version ID.

    Args:
        version_id: The ID of the specific DocumentVersions record.
        db_instance: The Database instance.

    Returns:
        A dictionary containing the version ID ('id'), the analysis content ('content'),
        and the creation timestamp ('created_at'), or an error dictionary.
    """
    db=db_instance
    if db is None:
        logging.warning("get_specific_analysis called without explicit db instance.")
        raise ValueError("Global db instance not found and no instance passed.")

    try:
        with db.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Fetch analysis_content and created_at using the DocumentVersions primary key (id)
            cursor.execute('''
            SELECT id, analysis_content, created_at
            FROM DocumentVersions
            WHERE id = ?
            ''', (version_id,))
            result = cursor.fetchone()
            if result and result['analysis_content'] is not None:
                return {
                    'id': result['id'],
                    'content': result['analysis_content'],
                    'created_at': result['created_at']
                }
            # Handle cases where the version exists but has no analysis content
            elif result:
                 return {'error': f"No analysis content found for version ID {version_id}."}
            else:
                return {'error': f"No document version found with ID {version_id}."}
    except Exception as e:
        logging.error(f"Error in get_specific_analysis for version_id {version_id}: {str(e)}", exc_info=True)
        return {'error': f"Error retrieving analysis content: {str(e)}"}


def get_media_prompts(media_id: int, db_instance: Database) -> List[Dict]:
    """
    Retrieves all non-empty prompt entries from the DocumentVersions table
    for a given media ID, ordered by version number descending.

    Args:
        media_id: The ID of the media item.
        db_instance: The Database instance.

    Returns:
        A list of dictionaries, each containing the version ID ('id'),
        the prompt content ('content'), and the creation timestamp ('created_at').
        Returns an empty list on error or if none found.
    """
    db=db_instance
    if db is None:
        logging.warning("get_media_prompts called without explicit db instance.")
        raise ValueError("Global db instance not found and no instance passed.")

    try:
        with db.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Fetch id, prompt, and created_at from DocumentVersions
            cursor.execute('''
                           SELECT id, prompt, created_at
                           FROM DocumentVersions
                           WHERE media_id = ?
                             AND prompt IS NOT NULL
                             AND prompt != ''
                           ORDER BY version_number DESC
                           ''', (media_id,))
            results = cursor.fetchall()
            # Format results
            return [
                {
                    'id': row['id'],  # Use DocumentVersions.id
                    'content': row['prompt'],  # The actual prompt content
                    'created_at': row['created_at']  # Timestamp of the version creation
                }
                for row in results
            ]
    except Exception as e:
        logging.error(f"Error in get_media_prompts for media_id {media_id}: {str(e)}", exc_info=True)
        return []  # Return empty list on error


def get_specific_prompt(version_id: int, db_instance: Database) -> Dict:
    """
    Retrieves the prompt for a specific document version ID.

    Args:
        version_id: The ID of the specific DocumentVersions record.
        db_instance: The Database instance.

    Returns:
        A dictionary containing the version ID ('id'), the prompt content ('content'),
        and the creation timestamp ('created_at'), or an error dictionary.
    """
    db=db_instance
    if db is None:
        logging.warning("get_specific_prompt called without explicit db instance.")
        raise ValueError("Global db instance not found and no instance passed.")

    try:
        with db.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Fetch prompt and created_at using the DocumentVersions primary key (id)
            cursor.execute('''
            SELECT id, prompt, created_at
            FROM DocumentVersions
            WHERE id = ?
            ''', (version_id,))
            result = cursor.fetchone()
            if result and result['prompt'] is not None:
                return {
                    'id': result['id'],
                    'content': result['prompt'],
                    'created_at': result['created_at']
                }
            elif result:
                 return {'error': f"No prompt found for version ID {version_id}."}
            else:
                 return {'error': f"No document version found with ID {version_id}."}
    except Exception as e:
        logging.error(f"Error in get_specific_prompt for version_id {version_id}: {str(e)}", exc_info=True)
        return {'error': f"Error retrieving prompt: {str(e)}"}


def delete_specific_transcript(transcript_id: int, db_instance: Database) -> str:
    db=db_instance
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM Transcripts WHERE id = ?', (transcript_id,))
            conn.commit()
            if cursor.rowcount > 0:
                return f"Transcript with ID {transcript_id} has been deleted successfully."
            else:
                return f"No transcript found with ID {transcript_id}."
    except Exception as e:
        logging.error(f"Error in delete_specific_transcript: {str(e)}")
        return f"Error deleting transcript: {str(e)}"


def delete_specific_analysis(version_id: int, db_instance: Database) -> str:
    """
    Deletes (sets to NULL) the analysis content for a specific document version ID.
    Warning: This modifies historical version data.

    Args:
        version_id: The ID of the DocumentVersions record to modify.
        db_instance: The Database instance.

    Returns:
        A status message string.
    """
    db=db_instance
    if db is None:
        logging.warning("delete_specific_analysis called without explicit db instance.")
        raise ValueError("Global db instance not found and no instance passed.")

    try:
        # Use transaction for the update
        with db.transaction() as conn:
            cursor = conn.cursor()
            # Update the analysis_content field in the DocumentVersions table
            cursor.execute('UPDATE DocumentVersions SET analysis_content = NULL WHERE id = ?', (version_id,))
            rows_affected = cursor.rowcount

        if rows_affected > 0:
            logging.info(f"Cleared analysis_content for document version ID {version_id}.")
            return f"Analysis content for version ID {version_id} has been cleared successfully."
        else:
            logging.warning(f"Attempted to clear analysis_content, but no document version found with ID {version_id}.")
            return f"No document version found with ID {version_id}."
    except Exception as e:
        logging.error(f"Error in delete_specific_analysis for version_id {version_id}: {str(e)}", exc_info=True)
        return f"Error deleting analysis content: {str(e)}"


def delete_specific_prompt(version_id: int, db_instance: Database) -> str:
    """
    Deletes (sets to NULL) the prompt for a specific document version ID.
    Warning: This modifies historical version data.

    Args:
        version_id: The ID of the DocumentVersions record to modify.
        db_instance: The Database instance.

    Returns:
        A status message string.
    """
    db=db_instance
    if db is None:
        logging.warning("delete_specific_prompt called without explicit db instance.")
        raise ValueError("Global db instance not found and no instance passed.")

    try:
        with db.transaction() as conn:
            cursor = conn.cursor()
            # Update the prompt field in the DocumentVersions table
            cursor.execute('UPDATE DocumentVersions SET prompt = NULL WHERE id = ?', (version_id,))
            rows_affected = cursor.rowcount

        if rows_affected > 0:
            logging.info(f"Cleared prompt for document version ID {version_id}.")
            return f"Prompt for version ID {version_id} has been cleared successfully."
        else:
            logging.warning(f"Attempted to clear prompt, but no document version found with ID {version_id}.")
            return f"No document version found with ID {version_id}."
    except Exception as e:
        logging.error(f"Error in delete_specific_prompt for version_id {version_id}: {str(e)}", exc_info=True)
        return f"Error deleting prompt: {str(e)}"


def get_paginated_files(db_instance: Database, page: int = 1, results_per_page: int = 50) -> Tuple[List[Tuple[int, str]], int, int]:
    """
    Fetches a paginated list of media items (id, title) from the database.

    Args:
        db_instance: The Database instance to use.
        page: The page number (1-based).
        results_per_page: The number of items per page.

    Returns:
        A tuple containing:
            - results: List of tuples (id, title).
            - total_pages: Total number of pages.
            - current_page: The requested page number.

    Raises:
        DatabaseError: If a database query fails.
        TypeError: If db_instance is not a valid Database object.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("A valid Database instance must be provided.")
    if page < 1:
        raise ValueError("Page number must be 1 or greater.")
    if results_per_page < 1:
        raise ValueError("Results per page must be 1 or greater.")

    logging.debug(f"Fetching paginated files: page={page}, results_per_page={results_per_page} from DB: {db_instance.db_path}")

    try:
        offset = (page - 1) * results_per_page
        # Use the transaction context manager even for reads for consistent connection handling
        with db_instance.transaction() as conn: # Ensures connection is managed correctly
            cursor = conn.cursor()

            # Get total count of non-trashed media items (assuming you want to exclude trashed items)
            # If you want all items, remove the "WHERE is_trash = 0" clause
            cursor.execute("SELECT COUNT(*) FROM Media WHERE is_trash = 0")
            count_result = cursor.fetchone()
            total_entries = count_result[0] if count_result else 0
            logging.debug(f"Total non-trashed media entries found: {total_entries}")


            # Fetch paginated results, excluding trashed items
            cursor.execute("""
                SELECT id, title
                FROM Media
                WHERE is_trash = 0
                ORDER BY title COLLATE NOCASE -- Case-insensitive title sorting
                LIMIT ? OFFSET ?
            """, (results_per_page, offset))
            # fetchall() will use the row_factory set in the Database class
            results_raw = cursor.fetchall()
            # Convert Row objects to simple tuples if required by the endpoint caller
            results = [(row['id'], row['title']) for row in results_raw]

        # Calculate total pages
        total_pages = 0
        if total_entries > 0 and results_per_page > 0:
             total_pages = (total_entries + results_per_page - 1) // results_per_page
        # Alternative: Use math.ceil
        # import math
        # total_pages = math.ceil(total_entries / results_per_page) if results_per_page > 0 else 0


        logging.debug(f"Returning {len(results)} results. Total pages: {total_pages}")
        return results, total_pages, page

    except sqlite3.Error as e:
        logging.error(f"SQLite error fetching paginated files from {db_instance.db_path}: {e}", exc_info=True)
        raise DatabaseError(f"Error fetching paginated files: {e}") from e
    except Exception as e:
        logging.error(f"Unexpected error fetching paginated files from {db_instance.db_path}: {e}", exc_info=True)
        raise DatabaseError(f"Unexpected error fetching paginated files: {e}") from e


#
# End of Functions to handle deletion of media items
#######################################################################################################################


#######################################################################################################################
#
# Functions to manage document versions

def get_full_media_details2(media_id: int, db_instance: Database = None): # Use TypedDict in return hint
    """
    Get complete media details including keywords and all versions.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("A valid Database instance must be provided.")

    logger.debug(f"Attempting to get full details for ID: {media_id} on DB: {db_instance.db_path}")
    try:
        with db_instance.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 1. Get basic media info
            cursor.execute('''
                SELECT
                    id, url, title, type, content, author, ingestion_date,
                    transcription_model, is_trash, trash_date,
                    vector_embedding, chunking_status, vector_processing, content_hash
                FROM Media WHERE id = ?
            ''', (media_id,))
            media_row = cursor.fetchone()

            if not media_row:
                logger.warning(f"No media found for ID {media_id} in DB {db_instance.db_path}.")
                return None

            # 2. Populate the dictionary, ensuring types
            media_dict = {
                "id": media_row['id'],
                "url": media_row['url'],
                "title": media_row['title'],
                "type": media_row['type'],
                "content": media_row['content'],
                "author": media_row['author'],
                "ingestion_date": media_row['ingestion_date'],
                "transcription_model": media_row['transcription_model'],
                "is_trash": bool(media_row['is_trash']), # Ensure bool
                "trash_date": media_row['trash_date'],
                "vector_embedding": media_row['vector_embedding'],
                "chunking_status": media_row['chunking_status'],
                "vector_processing": media_row['vector_processing'],
                "content_hash": media_row['content_hash'],
                "keywords": [], # Initialize as empty list
                "versions": []  # Initialize as empty list
            }

            # 3. Get keywords
            cursor.execute('''
                SELECT k.keyword FROM Keywords k JOIN MediaKeywords mk ON k.id = mk.keyword_id
                WHERE mk.media_id = ? ORDER BY k.keyword COLLATE NOCASE
            ''', (media_id,))
            # Assign directly to the key
            media_dict["keywords"] = [row['keyword'] for row in cursor.fetchall()]
            logger.debug(f"Keywords fetched: {media_dict['keywords']}")

        # 4. Get versions (outside the 'with' block for the connection)
        # Assign directly to the key
        media_dict["versions"] = get_all_document_versions(
            media_id=media_id,
            include_content=False,
            db_instance=db_instance
        )
        logger.debug(f"Versions fetched: {len(media_dict['versions'])} versions found.")

        # Cast the final dictionary to the TypedDict type before returning
        # This helps the type checker verify the structure.
        return media_dict # Pylance should understand this structure now

    except sqlite3.Error as e:
        logger.error(f"Database error getting full media details for ID {media_id} on {db_instance.db_path}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting full media details for ID {media_id} on {db_instance.db_path}: {e}", exc_info=True)
        return None


def create_document_version(
        media_id: int,
        content: str,
        prompt: Optional[str] = None,
        analysis_content: Optional[str] = None,
        db_instance: Database = None,
        # Add conn parameter for use within existing transactions
        conn: Optional[sqlite3.Connection] = None
) -> Dict[str, Any]:
    """
    Creates a new document version record in the DocumentVersions table.

    Args:
        media_id: The ID of the associated media item.
        content: The content snapshot for this version.
        prompt: The prompt associated with this version's analysis (optional).
        analysis_content: The analysis associated with this version's content (optional).
        db_instance: The Database instance (required if conn is not provided).
        conn: An existing sqlite3.Connection (optional, used for transactions).

    Returns:
        A dictionary containing the new version_number and media_id.

    Raises:
        DatabaseError: If the database operation fails.
        ValueError: If neither db_instance nor conn is provided.
    """
    if conn is None and db_instance is None:
        raise ValueError("Either db_instance or conn must be provided.")
    if conn and not isinstance(conn, sqlite3.Connection):
         raise TypeError("Provided conn must be a valid sqlite3.Connection.")
    if db_instance and not isinstance(db_instance, Database):
         raise TypeError("Provided db_instance must be a valid Database object.")

    # Prefer using the passed connection if available (for transactions)
    db_to_use = db_instance if db_instance else None # Primarily for logging path
    log_path = db_to_use.db_path if db_to_use else "existing connection"
    logging.debug(f"Creating document version for media_id={media_id} on DB: {log_path}")

    # Define the operation as a function to run within transaction or directly
    def _create_version_operation(connection: sqlite3.Connection) -> Dict[str, Any]:
        try:
            cursor = connection.cursor()

            # --- Get the next version number ---
            cursor.execute('''
                SELECT COALESCE(MAX(version_number), 0) + 1
                FROM DocumentVersions
                WHERE media_id = ?
            ''', (media_id,))
            version_number_result = cursor.fetchone()
            # Ensure we handle the case where fetchone might return None (though COALESCE should prevent it)
            version_number = version_number_result[0] if version_number_result else 1

            logging.debug(f"Determined next version number: {version_number} for media_id={media_id}")

            # --- Insert the new version record ---
            # Note: prompt and analysis_content can be NULL in the table
            cursor.execute('''
                INSERT INTO DocumentVersions
                (media_id, version_number, content, prompt, analysis_content, created_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (media_id, version_number, content, prompt, analysis_content))

            logging.info(f"Successfully created version {version_number} for media_id={media_id}")

            # Return essential info
            return {
                'media_id': media_id,
                'version_number': version_number,
                # Add content length if needed, but usually not required
                # 'content_length': len(content)
            }
        except sqlite3.IntegrityError as ie:
             # This might happen if (media_id, version_number) UNIQUE constraint is violated (race condition?)
             logging.error(f"Integrity error creating version for media_id={media_id}: {ie}", exc_info=True)
             raise DatabaseError(f"Failed to create document version due to integrity constraint: {ie}") from ie
        except sqlite3.Error as e:
            logging.error(f"SQLite error creating version {version_number} for media_id={media_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to create document version: {e}") from e

    # Execute the operation
    try:
        if conn:
            # Run directly using the provided connection (part of an outer transaction)
            return _create_version_operation(conn)
        else:
            # Run within its own transaction using the db_instance
            with db_to_use.transaction() as new_conn:
                return _create_version_operation(new_conn)
    except DatabaseError: # Re-raise DatabaseErrors caught inside
        raise
    except Exception as e:
        logging.error(f"Unexpected error wrapper creating version for media_id={media_id}: {e}", exc_info=True)
        raise DatabaseError(f"An unexpected error occurred: {e}") from e


def get_document_version(
        media_id: int,
        version_number: Optional[int] = None,
        include_content: bool = True,
        db_instance: Database = None
) -> Optional[Dict[str, Any]]: # Return Optional[Dict] or raise error
    """
    Get a specific document version or the latest version for a media item.

    Args:
        media_id: The ID of the media item.
        version_number: The specific version number to retrieve. If None, retrieves the latest.
        include_content: Whether to include the full 'content' field.
        db_instance: The Database instance to use.

    Returns:
        A dictionary representing the document version, or None if not found.
        The dictionary contains 'id', 'media_id', 'version_number', 'created_at',
        'prompt', 'analysis_content', and optionally 'content'.

    Raises:
        DatabaseError: If a database query fails.
        TypeError: If db_instance is not a valid Database object.
        ValueError: If version_number is provided but is not a positive integer.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("A valid Database instance must be provided.")
    if version_number is not None and (not isinstance(version_number, int) or version_number < 1):
        raise ValueError("Version number must be a positive integer.")

    log_msg = f"Getting {'latest' if version_number is None else f'version {version_number}'} for media_id={media_id}"
    logging.debug(f"{log_msg} from DB: {db_instance.db_path} (Include content: {include_content})")

    try:
        # Use transaction context for connection management
        with db_instance.transaction() as conn:
            cursor = conn.cursor()

            # Construct the SELECT clause dynamically
            select_cols = "id, version_number, created_at, prompt, analysis_content"
            if include_content:
                select_cols += ", content"

            params = [media_id]
            if version_number is None:
                # Get latest version
                query = f'''
                    SELECT {select_cols}
                    FROM DocumentVersions
                    WHERE media_id = ?
                    ORDER BY version_number DESC
                    LIMIT 1
                '''
            else:
                # Get specific version
                query = f'''
                    SELECT {select_cols}
                    FROM DocumentVersions
                    WHERE media_id = ? AND version_number = ?
                '''
                params.append(version_number)

            cursor.execute(query, tuple(params))
            result = cursor.fetchone() # Fetch using the row factory

            if not result:
                logging.warning(f"Version {'latest' if version_number is None else version_number} not found for media_id {media_id}")
                return None # Return None instead of {'error': ...}

            # Convert Row object to dictionary
            version_data = dict(result) # Convert Row to dict
            version_data['media_id'] = media_id # Ensure media_id is present

            return version_data

    except sqlite3.Error as e:
        logging.error(f"SQLite error retrieving {log_msg}: {e}", exc_info=True)
        raise DatabaseError(f"Database error retrieving version: {e}") from e
    except Exception as e:
         logging.error(f"Unexpected error retrieving {log_msg}: {e}", exc_info=True)
         raise DatabaseError(f"Unexpected error retrieving version: {e}") from e


def get_all_document_versions(
        media_id: int,
        include_content: bool = False,
        limit: Optional[int] = None,
        offset: Optional[int] = 0, # Default offset to 0
        db_instance: Database = None
) -> List[Dict[str, Any]]:
    """
    Get all versions for a media item with pagination, including prompt and analysis_content.

    Args:
        media_id: The ID of the media item.
        include_content: Whether to include the full content of each version.
        limit: Maximum number of versions to return. None for no limit.
        offset: Number of versions to skip (for pagination). Defaults to 0.
        db_instance: The Database instance.

    Returns:
        A list of dictionaries, each representing a document version. Returns empty list if none found or on error.

    Raises:
        DatabaseError: If a database query fails.
        TypeError: If db_instance is not a valid Database object.
        ValueError: If limit or offset are invalid.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("A valid Database instance must be provided.")
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        raise ValueError("Limit must be a positive integer.")
    if offset is not None and (not isinstance(offset, int) or offset < 0):
         raise ValueError("Offset must be a non-negative integer.")

    logging.debug(f"Getting all versions for media_id={media_id} (Limit={limit}, Offset={offset}, Content={include_content}) from DB: {db_instance.db_path}")

    try:
        # Use transaction context for connection management
        with db_instance.transaction() as conn:
            cursor = conn.cursor()

            # Include prompt and analysis_content in the selection
            select_clause = 'id, version_number, created_at, prompt, analysis_content'
            if include_content:
                select_clause += ', content'

            query = f'''
                    SELECT {select_clause}
                    FROM DocumentVersions
                    WHERE media_id = ?
                    ORDER BY version_number DESC
                '''

            params = [media_id]

            # Apply limit and offset if specified
            # Note: SQLite requires LIMIT before OFFSET if both are used.
            if limit is not None:
                query += ' LIMIT ?'
                params.append(limit)
                # OFFSET only makes sense if LIMIT is also applied
                if offset is not None and offset > 0:
                    query += ' OFFSET ?'
                    params.append(offset)

            logging.debug(f"Executing get_all_document_versions query | Params: {params}")
            cursor.execute(query, tuple(params))
            results_raw = cursor.fetchall()

            # Convert rows to dictionaries using the Row factory's dict conversion
            versions_list = [dict(row) for row in results_raw]
            # Add media_id for context if needed elsewhere (optional)
            for v in versions_list:
                v['media_id'] = media_id

            logging.debug(f"Found {len(versions_list)} versions for media_id={media_id}")
            return versions_list

    except sqlite3.Error as e:
        logging.error(f"SQLite error retrieving versions for media_id {media_id} from {db_instance.db_path}: {e}",
                     exc_info=True)
        # Return empty list on error as per original docstring
        return []
    except Exception as e:
        logging.error(f"Unexpected error retrieving versions for media_id {media_id} from {db_instance.db_path}: {e}",
                     exc_info=True)
        return []


def delete_document_version(media_id: int, version_number: int, db_instance: Database) -> Dict[str, Any]:
    """
    Delete a specific document version.

    Returns {'error': message} if:
      - The version doesn't exist.
      - It's the last existing version for the media item.
    Returns {'success': message} on successful deletion.

    Args:
        media_id: The ID of the media item.
        version_number: The specific version number to delete.
        db_instance: The Database instance.

    Raises:
        DatabaseError: If a database query fails unexpectedly.
        TypeError: If db_instance is not a valid Database object.
        ValueError: If version_number is invalid.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("A valid Database instance must be provided.")
    if not isinstance(version_number, int) or version_number < 1:
        raise ValueError("Version number must be a positive integer.")

    logging.debug(f"Attempting to delete version {version_number} for media_id={media_id} from DB: {db_instance.db_path}")

    try:
        # Use a transaction to ensure atomicity of checks and delete
        with db_instance.transaction() as conn:
            cursor = conn.cursor()

            # Check how many total versions exist for this media item
            cursor.execute('''
                SELECT COUNT(*) FROM DocumentVersions
                WHERE media_id = ?
            ''', (media_id,))
            count_result = cursor.fetchone()
            total_versions = count_result[0] if count_result else 0

            if total_versions <= 1:
                logging.warning(f"Attempted to delete the last version ({version_number}) for media_id={media_id}")
                return {'error': 'Cannot delete the last version'}

            # Check if the target version exists before attempting delete
            cursor.execute('''
                SELECT 1 FROM DocumentVersions
                WHERE media_id = ? AND version_number = ?
            ''', (media_id, version_number))
            exists = cursor.fetchone()

            if not exists:
                logging.warning(f"Version {version_number} not found for deletion for media_id={media_id}")
                return {'error': 'Version not found'}

            # Perform the delete operation
            cursor.execute('''
                DELETE FROM DocumentVersions
                WHERE media_id = ? AND version_number = ?
            ''', (media_id, version_number))
            rows_affected = cursor.rowcount

            if rows_affected > 0:
                 logging.info(f"Successfully deleted version {version_number} for media_id={media_id}")
                 return {'success': f'Version {version_number} deleted successfully'}
            else:
                 # Should not happen if exists check passed, but handle defensively
                 logging.error(f"Version {version_number} found but delete affected 0 rows for media_id={media_id}")
                 return {'error': 'Deletion failed unexpectedly after existence check'}

    except sqlite3.Error as e:
        logging.error(f"SQLite error deleting version {version_number} for media_id={media_id}: {e}", exc_info=True)
        # Don't return {'error': str(e)} here, raise a proper exception
        raise DatabaseError(f"Database error deleting version: {e}") from e
    except Exception as e:
        logging.error(f"Unexpected error deleting version {version_number} for media_id={media_id}: {e}", exc_info=True)
        raise DatabaseError(f"Unexpected error deleting version: {e}") from e


def rollback_to_version(
        media_id: int,
        version_number: int,
        db_instance: Database
) -> Dict[str, Any]:
    """
    Rolls back the main Media record to a previous version's state by:
    1. Fetching the content, prompt, and analysis from the target version.
    2. Creating a NEW version record in DocumentVersions with this fetched data.
    3. Updating the main Media table's 'content' and 'content_hash' fields
       to match the rolled-back content.

    Args:
        media_id: The ID of the media item.
        version_number: The version number to roll back to.
        db_instance: The Database instance.

    Returns:
        A dictionary indicating success and the new version number created,
        or an error dictionary. Example: {'success': msg, 'new_version_number': num} or {'error': msg}

    Raises:
        DatabaseError: If database operations fail.
        TypeError: If db_instance is not valid.
        ValueError: If version_number is invalid.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("A valid Database instance must be provided.")
    if not isinstance(version_number, int) or version_number < 1:
        raise ValueError("Version number must be a positive integer.")

    logging.debug(f"Attempting rollback to version {version_number} for media_id={media_id} on DB: {db_instance.db_path}")

    try:
        # Use a single transaction for all operations
        with db_instance.transaction() as conn:
            cursor = conn.cursor()

            # --- 1. Get the target version data ---
            # Use the updated get_document_version
            target_version_data = get_document_version(
                media_id=media_id,
                version_number=version_number,
                include_content=True,
                db_instance=db_instance # Pass instance, get_document_version will use the transaction's conn
            )

            if target_version_data is None:
                logging.warning(f"Rollback failed: Target version {version_number} not found for media_id={media_id}")
                return {'error': f'Version {version_number} not found'}

            target_content = target_version_data.get('content')
            target_prompt = target_version_data.get('prompt')
            target_analysis = target_version_data.get('analysis_content')

            # Ensure content exists before proceeding
            if target_content is None:
                 logging.error(f"Rollback failed: Target version {version_number} for media_id={media_id} has NULL content.")
                 return {'error': f'Version {version_number} has no content to roll back to.'}

            # --- 2. Create a *new* version reflecting the rollback state ---
            # Pass the connection 'conn' to run within the current transaction
            new_version_info = create_document_version(
                media_id=media_id,
                content=target_content,
                prompt=target_prompt,
                analysis_content=target_analysis,
                db_instance=db_instance, # Still pass instance for logging etc.
                conn=conn # Pass the active connection!
            )
            new_version_number = new_version_info.get('version_number')
            if not new_version_number:
                 # This shouldn't happen if create_document_version is correct
                 logging.error(f"Rollback failed: create_document_version did not return a version number for media_id={media_id}")
                 raise DatabaseError("Failed to get new version number during rollback.")

            logging.debug(f"Created new version {new_version_number} during rollback for media_id={media_id}")

            # --- 3. Update the main Media table ---
            # Calculate the hash of the rolled-back content
            new_content_hash = hashlib.sha256(target_content.encode()).hexdigest()

            # Update Media.content and Media.content_hash
            # Also update transcription_model if relevant? Maybe copy from original rolled-back version?
            # For now, just update content and hash.
            cursor.execute('''
                UPDATE Media
                SET content = ?,
                    content_hash = ?
                WHERE id = ?
            ''', (target_content, new_content_hash, media_id))
            rows_affected = cursor.rowcount

            if rows_affected == 0:
                 # This indicates the media_id doesn't exist in the Media table, a major inconsistency.
                 logging.error(f"Rollback warning: Media record for ID {media_id} not found during update.")
                 # Rollback might still be considered partially successful as version was created,
                 # but raise an error because the main record wasn't updated.
                 raise DatabaseError(f"Media record {media_id} not found for final rollback update.")

            logging.info(f"Successfully rolled back media_id={media_id} to state of version {version_number} (New version: {new_version_number})")

            # Commit happens automatically when 'with' block exits without error

            return {
                'success': f'Successfully rolled back to version {version_number}. State saved as new version {new_version_number}.',
                'new_version_number': new_version_number
            }

    except sqlite3.Error as e:
        logging.error(f"SQLite error during rollback for media_id={media_id} to version {version_number}: {e}", exc_info=True)
        # Re-raise as specific error
        raise DatabaseError(f"Database error during rollback: {e}") from e
    except DatabaseError as de: # Catch errors raised by helpers
        logging.error(f"DatabaseError during rollback for media_id={media_id} to version {version_number}: {de}", exc_info=True)
        raise # Re-raise
    except Exception as e:
        logging.error(f"Unexpected error during rollback for media_id={media_id} to version {version_number}: {e}", exc_info=True)
        raise DatabaseError(f"Unexpected error during rollback: {e}") from e

#
# End of Functions to manage document versions
#######################################################################################################################


#######################################################################################################################
#
# Functions to manage media chunks

def process_chunks(db_instance: Database, chunks: List[Dict], media_id: int, batch_size: int = 100):
    """
    Process chunks in batches and insert them into the database.

    :param db_instance: Database instance to use for inserting chunks
    :param chunks: List of chunk dictionaries
    :param media_id: ID of the media these chunks belong to
    :param batch_size: Number of chunks to process in each batch
    """
    database=db_instance
    log_counter("process_chunks_attempt", labels={"media_id": media_id})
    start_time = time.time()
    total_chunks = len(chunks)
    processed_chunks = 0

    try:
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            chunk_data = [
                (media_id, chunk['text'], chunk['start_index'], chunk['end_index'])
                for chunk in batch
            ]

            try:
                database.execute_many(
                    "INSERT INTO MediaChunks (media_id, chunk_text, start_index, end_index) VALUES (?, ?, ?, ?)",
                    chunk_data
                )
                processed_chunks += len(batch)
                logging.info(f"Processed {processed_chunks}/{total_chunks} chunks for media_id {media_id}")
                log_counter("process_chunks_batch_success", labels={"media_id": media_id})
            except Exception as e:
                logging.error(f"Error inserting chunk batch for media_id {media_id}: {e}")
                log_counter("process_chunks_batch_error", labels={"media_id": media_id, "error_type": type(e).__name__})
                # Optionally, you could raise an exception here to stop processing
                # raise

            logging.info(f"Finished processing all {total_chunks} chunks for media_id {media_id}")
            duration = time.time() - start_time
            log_histogram("process_chunks_duration", duration, labels={"media_id": media_id})
            log_counter("process_chunks_success", labels={"media_id": media_id})
    except Exception as e:
        duration = time.time() - start_time
        log_histogram("process_chunks_duration", duration, labels={"media_id": media_id})
        log_counter("process_chunks_error", labels={"media_id": media_id, "error_type": type(e).__name__})
        logging.error(f"Error processing chunks for media_id {media_id}: {e}")


# Usage example:
# chunks = [{'text': 'chunk1', 'start_index': 0, 'end_index': 10}, ...]
# process_chunks(db, chunks, media_id=1, batch_size=100)

def batch_insert_chunks(db_instance, chunks, media_id):
    cursor = db_instance.cursor()
    chunk_data = [(
        media_id,
        chunk['text'],
        chunk['metadata']['start_index'],
        chunk['metadata']['end_index'],
        f"{media_id}_chunk_{i}"
    ) for i, chunk in enumerate(chunks, 1)]

    cursor.executemany('''
    INSERT INTO MediaChunks (media_id, chunk_text, start_index, end_index, chunk_id)
    VALUES (?, ?, ?, ?, ?)
    ''', chunk_data)


chunk_queue = queue.Queue()

def chunk_processor(db_instance: Database = None):
    db=db_instance
    while True:
        chunk_batch = chunk_queue.get()
        if chunk_batch is None:
            break
        try:
            with db.get_connection() as conn:
                conn.execute("BEGIN TRANSACTION")
                try:
                    batch_insert_chunks(conn, chunk_batch['chunks'], chunk_batch['media_id'])
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    logging.error(f"Error in batch insert: {str(e)}")
        except Exception as e:
            logging.error(f"Error processing chunk batch: {str(e)}")
        finally:
            chunk_queue.task_done()

# Start the chunk processor thread
#chunk_processor_thread = threading.Thread(target=chunk_processor)
#chunk_processor_thread.start()

# Make sure to properly shut down the chunk processor when your application exits
# def shutdown_chunk_processor():
#     chunk_queue.put(None)
#     chunk_processor_thread.join()

#FIXME - add into main db creation code
def update_media_chunks_table(db_instance: Database):
    db=db_instance
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS MediaChunks_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id INTEGER,
            chunk_text TEXT,
            start_index INTEGER,
            end_index INTEGER,
            chunk_id TEXT,
            FOREIGN KEY (media_id) REFERENCES Media(id)
        )
        ''')
        cursor.execute('''
        INSERT INTO MediaChunks_new (media_id, chunk_text, start_index, end_index)
        SELECT media_id, chunk_text, start_index, end_index FROM MediaChunks
        ''')
        cursor.execute('DROP TABLE MediaChunks')
        cursor.execute('ALTER TABLE MediaChunks_new RENAME TO MediaChunks')

    logger.info("Updated MediaChunks table schema")

# Above function is a dirty hack that should be merged into the initial DB creation statement. This is a placeholder
# FIXME

#
# End of Functions to manage media chunks
#######################################################################################################################


#######################################################################################################################
#
# Workflow Functions

# Workflow Functions
def save_workflow_chat_to_db(chat_history, workflow_name, conversation_id=None):
    pass
#     try:
#         with db.get_connection() as conn:
#             cursor = conn.cursor()
#
#             if conversation_id is None:
#                 # Create a new conversation
#                 conversation_name = f"{workflow_name}_Workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
#                 cursor.execute('''
#                     INSERT INTO ChatConversations (media_id, media_name, conversation_name, created_at, updated_at)
#                     VALUES (NULL, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
#                 ''', (workflow_name, conversation_name))
#                 conversation_id = cursor.lastrowid
#             else:
#                 # Update existing conversation
#                 cursor.execute('''
#                     UPDATE ChatConversations
#                     SET updated_at = CURRENT_TIMESTAMP
#                     WHERE id = ?
#                 ''', (conversation_id,))
#
#             # Save messages
#             for user_msg, ai_msg in chat_history:
#                 if user_msg:
#                     cursor.execute('''
#                         INSERT INTO ChatMessages (conversation_id, sender, message, timestamp)
#                         VALUES (?, 'user', ?, CURRENT_TIMESTAMP)
#                     ''', (conversation_id, user_msg))
#                 if ai_msg:
#                     cursor.execute('''
#                         INSERT INTO ChatMessages (conversation_id, sender, message, timestamp)
#                         VALUES (?, 'ai', ?, CURRENT_TIMESTAMP)
#                     ''', (conversation_id, ai_msg))
#
#             conn.commit()
#
#         return conversation_id, f"Chat saved successfully! Conversation ID: {conversation_id}"
#     except Exception as e:
#         logging.error(f"Error saving workflow chat to database: {str(e)}")
#         return None, f"Error saving chat to database: {str(e)}"


def get_workflow_chat(conversation_id, db_instance: Database):
    """
    Retrieve a workflow chat from the database.

    Args:
    conversation_id: ID of the conversation to retrieve

    Returns:
    tuple: (chat_history, workflow_name, status_message)
    """
    db=db_instance
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()

            # Get conversation details
            cursor.execute('''
                SELECT media_name, conversation_name FROM ChatConversations
                WHERE id = ?
            ''', (conversation_id,))
            result = cursor.fetchone()
            if not result:
                return None, None, "Conversation not found"

            workflow_name, conversation_name = result

            # Get chat messages
            cursor.execute('''
                SELECT sender, message FROM ChatMessages
                WHERE conversation_id = ?
                ORDER BY timestamp
            ''', (conversation_id,))
            messages = cursor.fetchall()

            chat_history = []
            for sender, message in messages:
                if sender == 'user':
                    chat_history.append((message, None))
                else:
                    if chat_history and chat_history[-1][1] is None:
                        chat_history[-1] = (chat_history[-1][0], message)
                    else:
                        chat_history.append((None, message))

        return chat_history, workflow_name, f"Chat retrieved successfully"
    except Exception as e:
        logging.error(f"Error retrieving workflow chat from database: {str(e)}")
        return None, None, f"Error retrieving chat from database: {str(e)}"

#
# End of Workflow Functions
#######################################################################################################################
