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
import json
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

from fastapi import Path

# Local Libraries
from tldw_Server_API.app.core.Utils.Utils import get_project_relative_path, get_database_path, \
    get_database_dir, logger, logging
from tldw_Server_API.app.core.Utils.Chunk_Lib import chunk_options, chunk_text
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

            # --- REMOVE Check/Add keywords column in MediaModifications ---
            # --- REMOVE Check/Add prompt/analysis_content columns in Media if decided ---

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



# Version 1
# class DatabaseError(Exception):
#     pass
#
# class InputError(Exception):
#     pass
#
#
# class Database:
#     def __init__(self, db_name='server_media_summary.db'):
#         self.db_path = get_database_path(db_name)
#         self.timeout = 10.0
#         self._local = threading.local()
#
#     @contextmanager
#     def get_connection(self):
#         if not hasattr(self._local, 'connection') or self._local.connection is None:
#             self._local.connection = sqlite3.connect(self.db_path, timeout=self.timeout)
#             self._local.connection.isolation_level = None  # This enables autocommit mode
#         yield self._local.connection
#
#     def close_connection(self):
#         if hasattr(self._local, 'connection') and self._local.connection:
#             self._local.connection.close()
#             self._local.connection = None
#
#     @contextmanager
#     def transaction(self):
#         with self.get_connection() as conn:
#             try:
#                 # Write-Ahead-Logging enable/disable
#                 #conn.execute("PRAGMA journal_mode = WAL;")
#                 conn.execute("BEGIN")
#                 yield conn
#                 conn.execute("COMMIT")
#             except Exception:
#                 conn.execute("ROLLBACK")
#                 raise
#
#     def execute_query(self, query: str, params: Tuple = ()) -> Any:
#         with self.get_connection() as conn:
#             cursor = conn.cursor()
#             cursor.execute(query, params)
#             if query.strip().upper().startswith("SELECT"):
#                 return cursor.fetchall()
#             else:
#                 return cursor.rowcount
#
#     def execute_many(self, query: str, params_list: List[Tuple]) -> None:
#         with self.get_connection() as conn:
#             cursor = conn.cursor()
#             cursor.executemany(query, params_list)
#
#     def table_exists(self, table_name: str) -> bool:
#         query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
#         result = self.execute_query(query, (table_name,))
#         return bool(result)
#
# db = Database()
#
# # Usage example:
# if db.table_exists('DocumentVersions'):
#     logging.debug("DocumentVersions table exists")
# else:
#     logging.debug("DocumentVersions table does not exist")
#
#
# # Function to create tables with the new media schema
# def create_tables(db) -> None:
#     table_queries = [
#         # CREATE TABLE statements
#         '''
#         CREATE TABLE IF NOT EXISTS Media (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             url TEXT,
#             title TEXT NOT NULL,
#             type TEXT NOT NULL,
#             content TEXT,
#             author TEXT,
#             ingestion_date TEXT,
#             prompt TEXT,
#             analysis_content TEXT,
#             transcription_model TEXT,
#             is_trash BOOLEAN DEFAULT 0,
#             trash_date DATETIME,
#             vector_embedding BLOB,
#             chunking_status TEXT DEFAULT 'pending',
#             vector_processing INTEGER DEFAULT 0,
#             content_hash TEXT UNIQUE
#         )
#         ''',
#         '''
#         CREATE TABLE IF NOT EXISTS Keywords (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             keyword TEXT NOT NULL UNIQUE
#         )
#         ''',
#         '''
#         CREATE TABLE IF NOT EXISTS MediaKeywords (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             media_id INTEGER NOT NULL,
#             keyword_id INTEGER NOT NULL,
#             FOREIGN KEY (media_id) REFERENCES Media(id),
#             FOREIGN KEY (keyword_id) REFERENCES Keywords(id)
#         )
#         ''',
#         '''
#         CREATE TABLE IF NOT EXISTS MediaModifications (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             media_id INTEGER NOT NULL UNIQUE,
#             prompt TEXT,
#             analysis_content TEXT,
#             keywords TEXT,
#             modification_date TEXT,
#             FOREIGN KEY (media_id) REFERENCES Media(id)
#         )
#         ''',
#         '''
#         CREATE TABLE IF NOT EXISTS Transcripts (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             media_id INTEGER,
#             whisper_model TEXT,
#             transcription TEXT,
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#             FOREIGN KEY (media_id) REFERENCES Media(id)
#         )
#         ''',
#         '''
#         CREATE TABLE IF NOT EXISTS MediaChunks (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             media_id INTEGER,
#             chunk_text TEXT,
#             start_index INTEGER,
#             end_index INTEGER,
#             chunk_id TEXT,
#             FOREIGN KEY (media_id) REFERENCES Media(id)
#         )''',
#         '''
#         CREATE TABLE IF NOT EXISTS UnvectorizedMediaChunks (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             media_id INTEGER NOT NULL,
#             chunk_text TEXT NOT NULL,
#             chunk_index INTEGER NOT NULL,
#             start_char INTEGER NOT NULL,
#             end_char INTEGER NOT NULL,
#             chunk_type TEXT,
#             creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#             last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#             is_processed BOOLEAN DEFAULT FALSE,
#             metadata TEXT,
#             FOREIGN KEY (media_id) REFERENCES Media(id)
#         )
#         ''',
#
#         # Actual table for versioning:
#         '''
#         CREATE TABLE IF NOT EXISTS DocumentVersions (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             media_id INTEGER NOT NULL,
#             version_number INTEGER NOT NULL,
#             prompt TEXT,
#             analysis_content TEXT,
#             content TEXT,
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#             FOREIGN KEY (media_id) REFERENCES Media(id),
#             UNIQUE (media_id, version_number)
#         )
#         ''',
#     ]
#
#     basic_index_queries = [
#         # CREATE INDEX statements (excluding content_hash index)
#         'CREATE INDEX IF NOT EXISTS idx_media_title ON Media(title)',
#         'CREATE INDEX IF NOT EXISTS idx_media_type ON Media(type)',
#         'CREATE INDEX IF NOT EXISTS idx_media_author ON Media(author)',
#         'CREATE INDEX IF NOT EXISTS idx_media_ingestion_date ON Media(ingestion_date)',
#         'CREATE INDEX IF NOT EXISTS idx_keywords_keyword ON Keywords(keyword)',
#         'CREATE INDEX IF NOT EXISTS idx_mediakeywords_media_id ON MediaKeywords(media_id)',
#         'CREATE INDEX IF NOT EXISTS idx_mediakeywords_keyword_id ON MediaKeywords(keyword_id)',
#         'CREATE UNIQUE INDEX IF NOT EXISTS unique_media_modifications_media_id ON MediaModifications(media_id)',
#         'CREATE INDEX IF NOT EXISTS idx_media_is_trash ON Media(is_trash)',
#         'CREATE INDEX IF NOT EXISTS idx_mediachunks_media_id ON MediaChunks(media_id)',
#         'CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_media_id ON UnvectorizedMediaChunks(media_id)',
#         'CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_is_processed ON UnvectorizedMediaChunks(is_processed)',
#         'CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_chunk_type ON UnvectorizedMediaChunks(chunk_type)',
#         'CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_media_url ON Media(url)',
#         'CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_media_keyword ON MediaKeywords(media_id, keyword_id)',
#         'CREATE INDEX IF NOT EXISTS idx_document_versions_media_id ON DocumentVersions(media_id)',
#         'CREATE INDEX IF NOT EXISTS idx_document_versions_version_number ON DocumentVersions(version_number)'
#     ]
#
#     virtual_table_queries = [
#         # CREATE VIRTUAL TABLE statements
#         'CREATE VIRTUAL TABLE IF NOT EXISTS media_fts USING fts5(title, content)',
#         'CREATE VIRTUAL TABLE IF NOT EXISTS keyword_fts USING fts5(keyword)'
#     ]
#
#     # Execute all table creation queries first
#     for query in table_queries:
#         try:
#             db.execute_query(query)
#         except Exception as e:
#             logging.error(f"Error executing query: {query}")
#             logging.error(f"Error details: {str(e)}")
#             raise
#
#     # Execute basic index queries
#     for query in basic_index_queries:
#         try:
#             db.execute_query(query)
#         except Exception as e:
#             logging.error(f"Error executing query: {query}")
#             logging.error(f"Error details: {str(e)}")
#             raise
#
#     # Execute virtual table queries
#     for query in virtual_table_queries:
#         try:
#             db.execute_query(query)
#         except Exception as e:
#             logging.error(f"Error executing query: {query}")
#             logging.error(f"Error details: {str(e)}")
#             raise
#
#     try:
#         db.execute_query('CREATE UNIQUE INDEX IF NOT EXISTS idx_media_content_hash ON Media(content_hash)')
#     except Exception as e:
#         logging.error("Error creating content_hash index")
#         logging.error(f"Error details: {str(e)}")
#         # Don't raise here as this might fail on first creation
#
#     # Done
#     logging.info("All tables, indexes, and virtual tables created successfully.")
#
# # ------------------------------------------------------------------------------------------
# # Safe schema update for existing DBs
# def update_database_schema():
#     """
#     Check for and update schema elements that might be missing in older database versions:
#     - Checks for content_hash column in Media table
#     - Checks for keywords column in MediaModifications table
#     """
#     try:
#         logging.info("Checking database schema for updates...")
#         with db.get_connection() as conn:
#             cursor = conn.cursor()
#
#             # 1. Check for content_hash column in Media table
#             cursor.execute('''
#                 SELECT COUNT(*)
#                 FROM pragma_table_info('Media')
#                 WHERE name = 'content_hash'
#             ''')
#             column_exists = cursor.fetchone()[0]
#
#             if not column_exists:
#                 cursor.execute('ALTER TABLE Media ADD COLUMN content_hash TEXT')
#                 logging.info("Added content_hash column to 'Media' table.")
#
#                 cursor.execute('''
#                     CREATE UNIQUE INDEX IF NOT EXISTS idx_media_content_hash
#                     ON Media(content_hash)
#                 ''')
#                 logging.info("Created 'content_hash' unique index.")
#
#             # 2. Check for keywords column in MediaModifications table
#             cursor.execute('''
#                 SELECT COUNT(*)
#                 FROM pragma_table_info('MediaModifications')
#                 WHERE name = 'keywords'
#             ''')
#             keywords_column_exists = cursor.fetchone()[0]
#
#             if not keywords_column_exists:
#                 cursor.execute('ALTER TABLE MediaModifications ADD COLUMN keywords TEXT')
#                 logging.info("Added keywords column to 'MediaModifications' table.")
#
#             conn.commit()
#             logging.info("Database schema update completed successfully.")
#     except Exception as e:
#         logging.error(f"Schema update failed: {str(e)}")
#         logging.error(traceback.format_exc())
#         raise
#     finally:
#         db.close_connection()
#
#
# def diagnose_schema():
#     """Diagnostic function to check the actual schema of the database tables."""
#     try:
#         with db.get_connection() as conn:
#             cursor = conn.cursor()
#
#             # Check Media table schema
#             cursor.execute("PRAGMA table_info(Media)")
#             media_columns = cursor.fetchall()
#             logging.info(f"Media table columns: {[col[1] for col in media_columns]}")
#
#             # Check MediaModifications table schema
#             cursor.execute("PRAGMA table_info(MediaModifications)")
#             mod_columns = cursor.fetchall()
#             logging.info(f"MediaModifications table columns: {[col[1] for col in mod_columns]}")
#
#             # List all tables
#             cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
#             tables = cursor.fetchall()
#             logging.info(f"Tables in database: {[table[0] for table in tables]}")
#
#     except Exception as e:
#         logging.error(f"Schema diagnosis failed: {str(e)}")
#         logging.error(traceback.format_exc())
#
#
# def ensure_all_media_have_versions():
#     """Ensure all media items have at least one document version"""
#     try:
#         with db.transaction() as conn:
#             cursor = conn.cursor()
#
#             # Find media with no versions
#             cursor.execute("""
#                 SELECT id, content FROM Media
#                 WHERE NOT EXISTS (
#                     SELECT 1 FROM DocumentVersions WHERE media_id = Media.id
#                 )
#             """)
#
#             items = cursor.fetchall()
#             for media_id, content in items:
#                 logging.info(f"Creating initial version for media {media_id}")
#                 cursor.execute("""
#                     INSERT INTO DocumentVersions
#                     (media_id, version_number, content, created_at)
#                     VALUES (?, 1, ?, CURRENT_TIMESTAMP)
#                 """, (media_id, content))
#
#             if items:
#                 logging.info(f"Created initial versions for {len(items)} media items")
#
#     except Exception as e:
#         logging.error(f"Error ensuring versions: {str(e)}")
#
#
# # ------------------------------------------------------------------------------------------
# # Create tables (if they don't exist), then update schema (if needed)
# create_tables(db)
# update_database_schema()
# diagnose_schema()
# ensure_all_media_have_versions()
#

# End of DB Setup Functions
#######################################################################################################################


#######################################################################################################################
#
# Media-related Functions

def check_media_exists(title: str, url: str) -> Optional[int]:
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            query = 'SELECT id FROM Media WHERE title = ? OR url = ?'
            cursor.execute(query, (title, url))
            result = cursor.fetchone()
            logging.debug(f"check_media_exists query: {query}")
            logging.debug(f"check_media_exists params: title={title}, url={url}")
            logging.debug(f"check_media_exists result: {result}")
            return result[0] if result else None
    except Exception as e:
        logging.error(f"Error checking if media exists: {str(e)}")
        logging.error(f"Exception details: {traceback.format_exc()}")
        return None


def check_should_process_by_url(
    url: str,
    current_transcription_model: str,
    db=None # Pass db instance or use global
) -> Tuple[bool, Optional[int], str]:
    """
    Checks if media exists by URL and if processing should proceed based on
    the transcription model stored in the Media table.

    Args:
        url: The URL identifier of the media.
        current_transcription_model: The model intended for the current processing request.
        db: Database connection/manager instance.

    Returns:
        Tuple (should_process, existing_media_id, reason)
        - should_process (bool): True if processing should proceed, False otherwise.
        - existing_media_id (Optional[int]): The ID if media exists, None otherwise.
        - reason (str): Explanation for the decision.
    """
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


def check_media_and_whisper_model(title=None, url=None, current_whisper_model=None):
    """
    Check if media exists in the database and compare the whisper model used.

    :param title: Title of the media (optional)
    :param url: URL of the media (optional)
    :param current_whisper_model: The whisper model currently selected for use
    :return: Tuple (bool, str) - (should_download, reason)
    """
    if not title and not url:
        return True, "No title or URL provided"

    with db.get_connection() as conn:
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


def add_media_chunk(media_id: int, chunk_text: str, start_index: int, end_index: int, chunk_id: str):
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO MediaChunks (media_id, chunk_text, start_index, end_index, chunk_id) VALUES (?, ?, ?, ?, ?)",
            (media_id, chunk_text, start_index, end_index, chunk_id)
        )
        conn.commit()

def sqlite_update_fts_for_media(db, media_id: int):
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO media_fts (rowid, title, content) SELECT id, title, content FROM Media WHERE id = ?", (media_id,))
        conn.commit()


def get_unprocessed_media(db):
    query = """
    SELECT id, content, type, COALESCE(title, '') as file_name
    FROM Media 
    WHERE vector_processing = 0
    ORDER BY id
    """
    return db.execute_query(query)

def get_next_media_id():
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(media_id) FROM media")
        max_id = cursor.fetchone()[0]
        return (max_id or 0) + 1
    finally:
        conn.close()


def mark_media_as_processed(database, media_id):
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
                          media_type='video', overwrite=False, db=None):
    """Legacy wrapper for add_media_with_keywords"""
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
        db=db
    )

    return message  # Return just the message to maintain backward compatibility


# Function to add media with keywords
def add_media_with_keywords(url, title, media_type, content, keywords, prompt, analysis_content, transcription_model, author,
                           ingestion_date, overwrite=False, db=None, chunk_options=None, segments=None):
    log_counter("add_media_with_keywords_attempt")
    start_time = time.time()
    logging.debug(f"Entering add_media_with_keywords: URL={url}, Title={title}")

    if db is None:
        db = Database()

    # Set default values for missing fields
    if url is None:
        url = 'localhost'
    title = title or 'Untitled'
    media_type = media_type or 'Unknown'
    content = content or 'No content available'
    keywords = keywords or 'default'
    prompt = prompt or 'No prompt available'
    analysis_content = analysis_content or 'No analysis content available'
    transcription_model = transcription_model or 'Unknown'
    author = author or 'Unknown'
    ingestion_date = ingestion_date or datetime.now().strftime('%Y-%m-%d')

    if media_type not in ['article', 'audio', 'book', 'document', 'mediawiki_article', 'mediawiki_dump',
                        'obsidian_note', 'podcast', 'text', 'video', 'unknown']:
        log_counter("add_media_with_keywords_error", labels={"error_type": "InvalidMediaType"})
        duration = time.time() - start_time
        log_histogram("add_media_with_keywords_duration", duration)
        raise InputError("Invalid media type. Allowed types: article, audio file, document, obsidian_note, podcast, text, video, unknown.")

    if ingestion_date and not is_valid_date(ingestion_date):
        log_counter("add_media_with_keywords_error", labels={"error_type": "InvalidDateFormat"})
        duration = time.time() - start_time
        log_histogram("add_media_with_keywords_duration", duration)
        raise InputError("Invalid ingestion date format. Use YYYY-MM-DD.")

    # Handle keywords as either string or list
    if isinstance(keywords, str):
        keyword_list = [keyword.strip().lower() for keyword in keywords.split(',')]
    elif isinstance(keywords, list):
        keyword_list = [keyword.strip().lower() for keyword in keywords]
    else:
        keyword_list = ['default']

    # Generate content hash
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    # Generate URL if not provided
    if not url:
        url = f"https://No-URL-Submitted.com/{media_type}/{content_hash}"

    logging.info(f"Adding/updating media: URL={url}, Title={title}, Type={media_type}")
    logging.debug(f"Content (first 500 chars): {content[:500]}...")
    logging.debug(f"Keywords: {keyword_list}")
    logging.info(f"Prompt: {prompt}")
    logging.info(f"analysis_content: {analysis_content}")
    logging.info(f"Author: {author}")
    logging.info(f"Ingestion Date: {ingestion_date}")
    logging.info(f"Transcription Model: {transcription_model}")
    logging.info(f"Overwrite: {overwrite}")

    def extract_text_from_segments(segments):
        if isinstance(segments, list):
            return ' '.join([segment.get('Text', '') for segment in segments if 'Text' in segment])
        elif isinstance(segments, dict):
            return segments.get('text', '') or segments.get('content', '')
        else:
            return str(segments)

    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()

            # Check if media already exists using URL or content_hash
            cursor.execute('SELECT id FROM Media WHERE url = ? OR content_hash = ?', (url, content_hash))
            existing_media = cursor.fetchone()
            logging.debug(f"Existing media check result: {existing_media}")

            media_id = None
            action = "skipped"

            if existing_media:
                media_id = existing_media[0]
                logging.debug(f"Existing media ID: {media_id}")

                if overwrite:
                    # Update existing media
                    logging.debug("Updating existing media (overwrite=True)")
                    cursor.execute('''
                    UPDATE Media
                    SET url = ?, title = ?, type = ?, content = ?, author = ?,
                        ingestion_date = ?, transcription_model = ?, chunking_status = ?,
                        content_hash = ?, is_trash = 0, trash_date = NULL -- Reset trash status on overwrite
                    WHERE id = ?
                    ''', (url, title, media_type, content, author, ingestion_date,
                          transcription_model, 'pending', content_hash, media_id))
                    action = "updated"
                    # Clear old keywords for overwrite
                    cursor.execute('DELETE FROM MediaKeywords WHERE media_id = ?', (media_id,))
                    # Optional: Delete old versions? Or keep history? Keep for now.
                    log_counter("add_media_with_keywords_update")
                else:
                    action = "already exists (not updated)"
                    log_counter("add_media_with_keywords_skipped")
            else:
                logging.debug("Inserting new media")
                cursor.execute('''
                INSERT INTO Media (url, title, type, content, author, ingestion_date,
                                   transcription_model, chunking_status, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (url, title, media_type, content, author, ingestion_date,
                      transcription_model, 'pending', content_hash))
                media_id = cursor.lastrowid # Get the ID of the newly inserted row
                logging.debug(f"New media inserted with ID: {media_id}")
                action = "added"
                log_counter("add_media_with_keywords_insert")

            # Only proceed if added or updated
            if action in ["updated", "added"]:
                # --- Handle Keywords ---
                keyword_list = [k.strip().lower() for k in keywords if k.strip()] # Ensure keywords is a list here
                if keyword_list:
                    keyword_params = [(kw,) for kw in set(keyword_list)] # Use set for unique keywords
                    cursor.executemany('INSERT OR IGNORE INTO Keywords (keyword) VALUES (?)', keyword_params)

                    # Get keyword IDs
                    placeholders = ','.join(['?'] * len(keyword_params))
                    cursor.execute(f'SELECT id, keyword FROM Keywords WHERE keyword IN ({placeholders})', [p[0] for p in keyword_params])
                    keyword_id_map = {kw: kid for kid, kw in cursor.fetchall()}

                    # Insert media-keyword associations
                    media_keyword_params = [(media_id, keyword_id_map[kw]) for kw in keyword_list if kw in keyword_id_map]
                    if media_keyword_params:
                         cursor.executemany('INSERT OR IGNORE INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)', media_keyword_params)

                # --- Update FTS ---
                cursor.execute('INSERT OR REPLACE INTO media_fts (rowid, title, content) VALUES (?, ?, ?)',
                            (media_id, title, content))

                # --- Create Initial Document Version ---
                # This now stores the initial prompt and analysis
                create_document_version(
                    media_id=media_id,
                    content=content,
                    prompt=prompt, # Pass the initial prompt
                    analysis_content=analysis_content, # Pass the initial analysis
                    conn=conn # Pass the connection to run within the transaction
                )

        # Transaction committed automatically by context manager if no exceptions

        # Schedule chunking outside the transaction if it involves external processes or long operations
        if action in ["updated", "added"] and media_id is not None:
             # TODO: Refactor schedule_chunking if needed
             # schedule_chunking(media_id, content, title, chunk_options)
             pass # Placeholder for scheduling

        duration = time.time() - start_time
        log_histogram("add_media_with_keywords_duration", duration)

        if action in ["updated", "added"]:
            log_counter("add_media_with_keywords_success")
            message = f"Media '{title}' {action} with ID {media_id} and URL: {url}. Keywords: {', '.join(keyword_list)}." # Add keywords if desired
        else: # Skipped case
             message = f"Media '{title}' {action} (ID: {media_id}) with URL: {url}"


        logging.info(message)
        # Ensure media_id is returned even if skipped
        return media_id, message

    except sqlite3.Error as e:
        logging.error(f"SQL Error in add_media_with_keywords for URL {url}: {e}", exc_info=True)
        # Duration/log counter done in finally block if needed, or here
        raise DatabaseError(f"Error adding media with keywords: {e}") from e
    except Exception as e:
        logging.error(f"Unexpected Error in add_media_with_keywords: {e}")
        duration = time.time() - start_time
        log_histogram("add_media_with_keywords_duration", duration)
        log_counter("add_media_with_keywords_error", labels={"error_type": type(e).__name__})
        raise DatabaseError(f"Unexpected error: {e}")


def ingest_article_to_db(url, title, author, content, keywords, analysis_content, ingestion_date, custom_prompt):
    try:
        # Check if content is not empty or whitespace
        if not content.strip():
            raise ValueError("Content is empty.")

        keyword_list = keywords.split(',') if keywords else ["default"]
        keyword_str = ', '.join(keyword_list)

        # Set default values for missing fields
        url = url or 'Unknown'
        title = title or 'Unknown'
        author = author or 'Unknown'
        keywords = keywords or 'default'
        analysis_content = analysis_content or 'No analysis_content available'
        ingestion_date = ingestion_date or datetime.now().strftime('%Y-%m-%d')

        # Log the values of all fields before calling add_media_with_keywords
        logging.debug(f"URL: {url}")
        logging.debug(f"Title: {title}")
        logging.debug(f"Author: {author}")
        logging.debug(f"Content: {content[:50]}... (length: {len(content)})")  # Log first 50 characters of content
        logging.debug(f"Keywords: {keywords}")
        logging.debug(f"analysis_content: {analysis_content}")
        logging.debug(f"Ingestion Date: {ingestion_date}")
        logging.debug(f"Custom Prompt: {custom_prompt}")

        # Check if any required field is empty and log the specific missing field
        if not url:
            logging.error("URL is missing.")
            raise ValueError("URL is missing.")
        if not title:
            logging.error("Title is missing.")
            raise ValueError("Title is missing.")
        if not content:
            logging.error("Content is missing.")
            raise ValueError("Content is missing.")
        if not keywords:
            logging.error("Keywords are missing.")
            raise ValueError("Keywords are missing.")
        if not analysis_content:
            logging.error("analysis_content is missing.")
            raise ValueError("analysis_content is missing.")
        if not ingestion_date:
            logging.error("Ingestion date is missing.")
            raise ValueError("Ingestion date is missing.")
        if not custom_prompt:
            logging.error("Custom prompt is missing.")
            raise ValueError("Custom prompt is missing.")

        # Add media with keywords to the database
        result = add_media_with_keywords(
            url=url,
            title=title,
            media_type='article',
            content=content,
            keywords=keyword_str or "article_default",
            prompt=custom_prompt or None,
            analysis_content=analysis_content or "No analysis_content generated",
            transcription_model=None,  # or some default value if applicable
            author=author or 'Unknown',
            ingestion_date=ingestion_date
        )
        return result
    except Exception as e:
        logging.error(f"Failed to ingest article to the database: {e}")
        return str(e)


# Function to add a keyword
def add_keyword(keyword: str) -> int:
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
def delete_keyword(keyword: str) -> str:
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


def fetch_all_keywords() -> List[str]:
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT keyword FROM Keywords')
            keywords = [row[0] for row in cursor.fetchall()]
            return keywords
    except sqlite3.Error as e:
        raise DatabaseError(f"Error fetching keywords: {e}")

def keywords_browser_interface():
    keywords = fetch_all_keywords()
    return gr.Markdown("\n".join(f"- {keyword}" for keyword in keywords))

def display_keywords():
    try:
        keywords = fetch_all_keywords()
        return "\n".join(keywords) if keywords else "No keywords found."
    except DatabaseError as e:
        return str(e)


def export_keywords_to_csv():
    try:
        keywords = fetch_all_keywords()
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

def fetch_keywords_for_media(media_id):
    try:
        # First check if the keywords column exists in MediaModifications
        with db.get_connection() as conn:
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

def update_keywords_for_media(media_id: int, keywords: List[str]):
    """Update keywords with validation and error handling"""
    try:
        valid_keywords = [k.strip().lower() for k in keywords if k.strip()]
        if not valid_keywords:
            return

        with db.transaction() as conn:
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
def browse_items(search_query, search_type):
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            if search_type == 'Title':
                cursor.execute("SELECT id, title, url FROM Media WHERE title LIKE ?", (f'%{search_query}%',))
            elif search_type == 'URL':
                cursor.execute("SELECT id, title, url FROM Media WHERE url LIKE ?", (f'%{search_query}%',))
            elif search_type == 'Keyword':
                return fetch_items_by_keyword(search_query)
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
def fetch_item_details(media_id: int, db: Database = None) -> Tuple[str, str, str]:
    """
    Fetches the prompt, analysis_content, and content from the LATEST document version
    associated with the media_id.

    Args:
        media_id: The ID of the media item.
        db: The Database instance to use. If None, uses the global instance (avoid if possible).

    Returns:
        A tuple containing (prompt, analysis_content, content).
        Returns default messages if no version or data is found.
    """
    if db is None:
        # In a real application, you should ensure the db instance is passed
        # or handle the global instance creation carefully.
        # For now, we assume a global 'db' might exist for legacy compatibility,
        # but it's better practice to pass it explicitly.
        logging.warning("fetch_item_details called without explicit db instance.")
        raise ValueError("Global db instance not found and no instance passed.")

    try:
        # Use the get_document_version function which already fetches the latest version
        # We need the content, prompt, and analysis_content from that version
        latest_version = get_document_version(media_id=media_id, include_content=True, db=db) # Pass db instance

        if 'error' in latest_version:
            logging.warning(f"No document version found for media_id {media_id}: {latest_version['error']}")
            # Fallback: Try fetching content directly from Media table if absolutely needed,
            # but ideally, content should always exist in a version.
            try:
                with db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT content FROM Media WHERE id = ?", (media_id,))
                    content_result = cursor.fetchone()
                    content = content_result[0] if content_result else "No content available."
                return "No prompt available.", "No analysis content available.", content
            except Exception as fallback_e:
                 logging.error(f"Fallback fetch failed for media_id {media_id}: {fallback_e}")
                 return "Error fetching prompt.", "Error fetching analysis content.", "Error fetching media."

        # Extract data from the latest version dict (assuming get_document_version is updated too)
        # Need to ensure get_document_version actually fetches prompt/analysis from the version table
        prompt = latest_version.get('prompt', "No prompt available.")
        analysis_content = latest_version.get('analysis_content', "No analysis content available.")
        content = latest_version.get('content', "No content available.")

        # Handle None values gracefully
        prompt = prompt if prompt is not None else "No prompt available."
        analysis_content = analysis_content if analysis_content is not None else "No analysis content available."
        content = content if content is not None else "No content available."

        return prompt, analysis_content, content

    except sqlite3.Error as e:
        logging.error(f"Error fetching item details for media_id {media_id}: {e}", exc_info=True)
        return "Error fetching prompt.", "Error fetching analysis content.", "Error fetching media."
    except Exception as e: # Catch other potential errors like DatabaseError from get_document_version
        logging.error(f"Unexpected error fetching item details for media_id {media_id}: {e}", exc_info=True)
        return "Error fetching prompt.", "Error fetching analysis content.", "Error fetching media."

#
#  End of Media-related Functions
#######################################################################################################################


#######################################################################################################################
#
# Media-related Functions


# Function to add a version of a prompt and analysis_content
def add_media_version(media_id: int, prompt: str, analysis_content: str) -> Dict[str, Any]:
    """
    Legacy wrapper for create_document_version.
    Use create_document_version directly for new code.
    """
    logging.warning("add_media_version is deprecated, use create_document_version instead")

    try:
        # Get current content from Media table
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content FROM Media WHERE id = ?", (media_id,))
            result = cursor.fetchone()

            if not result:
                raise ValueError(f"No media found with ID {media_id}")
            content = result[0]

        # Use the new function
        return create_document_version(
            media_id=media_id,
            content=content,
            prompt=prompt,
            analysis_content=analysis_content
        )
    except Exception as e:
        logging.error(f"Error in add_media_version: {str(e)}")
        raise


def search_media_db(
        search_query: str,
        search_fields: List[str],
        keywords: List[str],
        page: int = 1,
        results_per_page: int = 20,
        connection=None
) -> Tuple[List[Tuple], int]:
    """
    Search for media items with advanced filtering options.
    """
    # Input validation (keep as is)
    if page < 1:
        raise ValueError("Page number must be 1 or greater")
    if results_per_page < 1 or results_per_page > 100:
        raise ValueError("Results per page must be between 1 and 100")

    # Sanitize and validate search fields (keep as is)
    valid_fields = {"title", "content", "author", "type"}
    sanitized_fields = [field for field in search_fields if field in valid_fields]
    if not sanitized_fields and search_query:
        raise ValueError("No valid search fields provided")

    def execute_query(conn):
        cursor = conn.cursor()
        try:
            # Build query components
            offset = (page - 1) * results_per_page
            params = []
            conditions = []

            # Build search query conditions
            if search_query and sanitized_fields:
                field_conditions = []
                for field in sanitized_fields:
                    field_conditions.append(f"Media.{field} LIKE ?")
                    params.append(f"%{search_query}%")
                conditions.append(f"({' OR '.join(field_conditions)})")

            # Build keyword conditions
            if keywords:
                placeholders = ", ".join(["?"] * len(keywords))
                if placeholders:  # Only add if we have keywords
                    conditions.append(f"""EXISTS (
                        SELECT 1 
                        FROM MediaKeywords mk 
                        JOIN Keywords k ON mk.keyword_id = k.id 
                        WHERE mk.media_id = Media.id 
                        AND k.keyword IN ({placeholders})
                    )""")
                    params.extend(keywords)

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            # First get the total count
            count_query = f"""
                SELECT COUNT(DISTINCT Media.id)
                FROM Media
                WHERE {where_clause}
            """
            cursor.execute(count_query, params.copy())
            total_matches = cursor.fetchone()[0]

            # Then get the paginated results
            results_query = f"""
                SELECT 
                    Media.id, Media.url, Media.title, Media.type, 
                    Media.content, Media.author, Media.ingestion_date
                FROM Media
                WHERE {where_clause}
                ORDER BY Media.ingestion_date DESC
                LIMIT ? OFFSET ?
            """
            params.extend([results_per_page, offset])
            cursor.execute(results_query, params)
            results = cursor.fetchall()

            return results, total_matches
        except Exception as e:
            logger.error(f"Database error in search_media_db: {str(e)}", exc_info=True)
            raise
        finally:
            cursor.close()

    # Execute with provided connection or get a new one
    if connection:
        return execute_query(connection)
    else:
        with db.get_connection() as conn:
            return execute_query(conn)


# Gradio function to handle user input and display results with pagination, with better feedback
def search_and_display(search_query, search_fields, keywords, page):
    results = search_media_db(search_query, search_fields, keywords, page)

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


# Function to export search results to CSV or markdown with pagination
def export_to_file(search_query: str, search_fields: List[str], keyword: str, page: int = 1, results_per_file: int = 1000, export_format: str = 'csv'):
    try:
        results = search_media_db(search_query, search_fields, keyword, page, results_per_file)
        if not results:
            return "No results found to export."

        # Create an 'exports' directory if it doesn't exist
        if not os.path.exists('exports'):
            os.makedirs('exports')

        if export_format == 'csv':
            filename = f'exports/search_results_page_{page}.csv'
            with open(filename, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['URL', 'Title', 'Type', 'Content', 'Author', 'Ingestion Date', 'Prompt', 'analysis_content'])
                for row in results:
                    writer.writerow(row)
        elif export_format == 'markdown':
            filename = f'exports/search_results_page_{page}.md'
            with open(filename, 'w', encoding='utf-8') as file:
                for item in results:
                    markdown_content = convert_to_markdown({
                        'title': item[1],
                        'url': item[0],
                        'type': item[2],
                        'content': item[3],
                        'author': item[4],
                        'ingestion_date': item[5],
                        'analysis_content': item[7],
                        'keywords': item[8].split(',') if item[8] else []
                    })
                    file.write(markdown_content)
                    file.write("\n---\n\n")  # Separator between items
        else:
            return f"Unsupported export format: {export_format}"

        return f"Results exported to {filename}"
    except (DatabaseError, InputError) as e:
        return str(e)


# Helper function to validate date format
def is_valid_date(date_string: str) -> bool:
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def check_existing_media(url):
    db = Database()
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
def schedule_chunking(media_id: int, content: str, media_name: str, media_type: str = None, chunk_options: dict = None):
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

        db = Database()
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

def update_media_content(selected_item, item_mapping, content_input, prompt_input, analysis_content):
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


def search_media_database(query: str, connection=None) -> List[Tuple[int, str, str]]:
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
        with db.get_connection() as conn:
            return execute_query(conn)


def load_media_content(media_id: int) -> dict:
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


def fetch_items_by_title_or_url(search_query: str, search_type: str):
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


def fetch_items_by_keyword(search_query: str):
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


def fetch_items_by_content(search_query: str):
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title, url FROM Media WHERE content LIKE ?", (f'%{search_query}%',))
            results = cursor.fetchall()
            return results
    except sqlite3.Error as e:
        raise DatabaseError(f"Error fetching items by content: {e}")


def fetch_item_details_single(media_id: int):
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


def convert_to_markdown(item):
    markdown = f"# {item['title']}\n\n"
    markdown += f"**URL:** {item['url']}\n\n"
    markdown += f"**Author:** {item['author']}\n\n"
    markdown += f"**Ingestion Date:** {item['ingestion_date']}\n\n"
    markdown += f"**Type:** {item['type']}\n\n"
    markdown += f"**Keywords:** {', '.join(item['keywords'])}\n\n"
    markdown += "## analysis_content\n\n"
    markdown += f"{item['analysis_content']}\n\n"
    markdown += "## Content\n\n"
    markdown += f"{item['content']}\n\n"
    return markdown


# Gradio function to handle user input and display results with pagination for displaying entries in the DB
def fetch_paginated_data(page: int, results_per_page: int) -> Tuple[List[Tuple], int]:
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

def view_database(page: int, results_per_page: int) -> Tuple[str, str, int]:
    results, total_entries = fetch_paginated_data(page, results_per_page)
    formatted_results = format_results_as_html(results)
    # Calculate total pages
    total_pages = (total_entries + results_per_page - 1) // results_per_page
    return formatted_results, f"Page {page} of {total_pages}", total_pages


def search_and_display_items(query, search_type, page, entries_per_page,char_count):
    offset = (page - 1) * entries_per_page
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

def import_obsidian_note_to_db(note_data, db: Database = None):
    """
    Imports or updates an Obsidian note into the database, including tags and
    creating an initial document version containing the frontmatter.

    Args:
        note_data (dict): Dictionary containing note info ('title', 'content', 'tags', 'frontmatter', 'file_path').
        db (Database, optional): The Database instance. Defaults to None (uses global).

    Returns:
        Tuple[bool, Optional[str]]: (success_status, error_message_or_none)
    """
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
                conn=conn # Pass connection
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
def get_transcripts(media_id):
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

def get_latest_transcription(media_id: int):
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


def mark_as_trash(media_id: int) -> None:
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE Media 
            SET is_trash = 1, trash_date = ?
            WHERE id = ?
        """, (datetime.now(), media_id))
        conn.commit()


def restore_from_trash(media_id: int) -> None:
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE Media 
            SET is_trash = 0, trash_date = NULL
            WHERE id = ?
        """, (media_id,))
        conn.commit()


def get_trashed_items() -> List[Dict]:
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, title, trash_date 
            FROM Media 
            WHERE is_trash = 1
            ORDER BY trash_date DESC
        """)
        return [{'id': row[0], 'title': row[1], 'trash_date': row[2]} for row in cursor.fetchall()]


def permanently_delete_item(media_id: int) -> None:
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM Media WHERE id = ?", (media_id,))
        cursor.execute("DELETE FROM MediaKeywords WHERE media_id = ?", (media_id,))
        cursor.execute("DELETE FROM MediaVersion WHERE media_id = ?", (media_id,))
        cursor.execute("DELETE FROM MediaModifications WHERE media_id = ?", (media_id,))
        cursor.execute("DELETE FROM media_fts WHERE rowid = ?", (media_id,))
        conn.commit()


def empty_trash(days_threshold: int) -> Tuple[int, int]:
    threshold_date = datetime.now() - timedelta(days=days_threshold)
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id FROM Media 
            WHERE is_trash = 1 AND trash_date <= ?
        """, (threshold_date,))
        old_items = cursor.fetchall()

        for item in old_items:
            permanently_delete_item(item[0])

        cursor.execute("""
            SELECT COUNT(*) FROM Media 
            WHERE is_trash = 1 AND trash_date > ?
        """, (threshold_date,))
        remaining_items = cursor.fetchone()[0]

    return len(old_items), remaining_items


def user_delete_item(media_id: int, force: bool = False) -> str:
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT is_trash, trash_date FROM Media WHERE id = ?", (media_id,))
        result = cursor.fetchone()

        if not result:
            return "Item not found."

        is_trash, trash_date = result

        if not is_trash:
            mark_as_trash(media_id)
            return "Item moved to trash."

        if force or (trash_date and (datetime.now() - trash_date).days >= 30):
            permanently_delete_item(media_id)
            return "Item permanently deleted."
        else:
            return "Item is already in trash. Use force=True to delete permanently before 30 days."


def get_chunk_text(media_id: int, chunk_index: int) -> str:
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM MediaChunks WHERE media_id = ? AND chunk_index = ?",
                       (media_id, chunk_index))
        result = cursor.fetchone()
    return result[0] if result else None

def get_full_document(media_id: int) -> str:
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM Media WHERE id = ?", (media_id,))
        result = cursor.fetchone()
    return result[0] if result else None

def get_all_content_from_database() -> List[Dict[str, Any]]:
    """
    Retrieve all media content from the database that requires embedding.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing the media ID, content, title, and other relevant fields.
    """
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


def get_media_content(media_id: int) -> str:
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

def get_media_title(media_id: int) -> str:
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT title FROM Media WHERE id = ?", (media_id,))
            result = cursor.fetchone()
            return result[0] if result else f"Unknown Source (ID: {media_id})"
    except sqlite3.Error as e:
        logging.error(f"Database error in get_media_title: {e}")
        return f"Unknown Source (ID: {media_id})"

def get_media_transcripts(media_id):
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

def get_specific_transcript(transcript_id: int) -> Dict:
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


def get_media_summaries(media_id: int, db: Database = None) -> List[Dict]:
    """
    Retrieves all non-empty analysis content entries (summaries) from the
    DocumentVersions table for a given media ID, ordered by version number descending.

    Args:
        media_id: The ID of the media item.
        db: The Database instance.

    Returns:
        A list of dictionaries, each containing the version ID ('id'),
        the analysis content ('content'), and the creation timestamp ('created_at').
        Returns an empty list on error or if none found.
    """
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


def get_specific_analysis(version_id: int, db: Database = None) -> Dict:
    """
    Retrieves the analysis content for a specific document version ID.

    Args:
        version_id: The ID of the specific DocumentVersions record.
        db: The Database instance.

    Returns:
        A dictionary containing the version ID ('id'), the analysis content ('content'),
        and the creation timestamp ('created_at'), or an error dictionary.
    """
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


def get_media_prompts(media_id: int, db: Database = None) -> List[Dict]:
    """
    Retrieves all non-empty prompt entries from the DocumentVersions table
    for a given media ID, ordered by version number descending.

    Args:
        media_id: The ID of the media item.
        db: The Database instance.

    Returns:
        A list of dictionaries, each containing the version ID ('id'),
        the prompt content ('content'), and the creation timestamp ('created_at').
        Returns an empty list on error or if none found.
    """
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


def get_specific_prompt(version_id: int, db: Database = None) -> Dict:
    """
    Retrieves the prompt for a specific document version ID.

    Args:
        version_id: The ID of the specific DocumentVersions record.
        db: The Database instance.

    Returns:
        A dictionary containing the version ID ('id'), the prompt content ('content'),
        and the creation timestamp ('created_at'), or an error dictionary.
    """
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


def delete_specific_transcript(transcript_id: int) -> str:
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


def delete_specific_analysis(version_id: int, db: Database = None) -> str:
    """
    Deletes (sets to NULL) the analysis content for a specific document version ID.
    Warning: This modifies historical version data.

    Args:
        version_id: The ID of the DocumentVersions record to modify.
        db: The Database instance.

    Returns:
        A status message string.
    """
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


def delete_specific_prompt(version_id: int, db: Database = None) -> str:
    """
    Deletes (sets to NULL) the prompt for a specific document version ID.
    Warning: This modifies historical version data.

    Args:
        version_id: The ID of the DocumentVersions record to modify.
        db: The Database instance.

    Returns:
        A status message string.
    """
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


def get_paginated_files(page: int = 1, results_per_page: int = 50) -> Tuple[List[Tuple[int, str]], int, int]:
    try:
        offset = (page - 1) * results_per_page
        with db.get_connection() as conn:
            cursor = conn.cursor()

            # Get total count of media items
            cursor.execute("SELECT COUNT(*) FROM Media")
            total_entries = cursor.fetchone()[0]

            # Fetch paginated results
            cursor.execute("""
                SELECT id, title 
                FROM Media 
                ORDER BY title
                LIMIT ? OFFSET ?
            """, (results_per_page, offset))
            results = cursor.fetchall()

        # Calculate total pages
        total_pages = (total_entries + results_per_page - 1) // results_per_page

        return results, total_pages, page
    except sqlite3.Error as e:
        logging.error(f"Error fetching paginated files: {e}")
        raise DatabaseError(f"Error fetching paginated files: {e}")


#
# End of Functions to handle deletion of media items
#######################################################################################################################


#######################################################################################################################
#
# Functions to manage document versions

def get_full_media_details2(media_id: int) -> Optional[Dict]:
    """Get complete media details with versions and keywords"""
    logger.debug(f"Attempting to get full details for ID: {media_id}")
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()

            # Get basic media info - matching your exact schema
            logger.debug("Executing main media query...")
            cursor.execute('''
                SELECT 
                    id, url, title, type, content,
                    author, ingestion_date, prompt, analysis_content,
                    transcription_model, is_trash, trash_date,
                    vector_embedding, chunking_status, vector_processing,
                    content_hash
                FROM Media
                WHERE id = ?
            ''', (media_id,))
            media = cursor.fetchone()
            logger.debug(f"fetchone() result type: {type(media)}") # Should be <class 'tuple'> or None

            if not media:
                logger.warning(f"No media found for ID {media_id} in DB.")
                return None

            # Convert to dict with all schema fields
            logger.debug("Attempting to create media_dict...")
            media_dict = {
                "id": media[0],
                "url": media[1],
                "title": media[2],
                "type": media[3],
                "content": media[4],
                "author": media[5],
                "ingestion_date": media[6],
                "prompt": media[7],
                "analysis_content": media[8],
                "transcription_model": media[9],
                "is_trash": bool(media[10]),
                "trash_date": media[11],
                "vector_embedding": media[12],
                "chunking_status": media[13],
                "vector_processing": media[14],
                "content_hash": media[15],
                "keywords": [],
                "versions": []
            }
            logger.debug(f"media_dict created. Type: {type(media_dict)}") # Should be <class 'dict'>

            # Get keywords
            logger.debug("Fetching keywords...")
            cursor.execute('''
                SELECT k.keyword
                FROM Keywords k
                JOIN MediaKeywords mk ON k.id = mk.keyword_id
                WHERE mk.media_id = ?
            ''', (media_id,))

            logger.debug("Fetching versions...")
            media_dict["keywords"] = [row[0] for row in cursor.fetchall()]
            logger.debug(f"Keywords fetched: {media_dict['keywords']}")

            # Get versions
            media_dict["versions"] = get_all_document_versions(media_id)

            return media_dict

    except Exception as e:
        logger.error(f"Error within get_full_media_details for ID {media_id}: {e}", exc_info=True)
        return None


def create_document_version(
        media_id: int,
        content: str,
        prompt: Optional[str] = None,
        analysis_content: Optional[str] = None,
        conn: Optional[sqlite3.Connection] = None
) -> Dict[str, Any]:
    """
    Creates a new document version with optional prompt and analysis_content.
    Returns the created version info.
    """
    try:
        # Combine content with prompt/analysis_content if provided
        full_content = content
        if prompt or analysis_content:
            full_content = f"Prompt: {prompt or ''}\n\nanalysis_content: {analysis_content or ''}\n\nContent:\n{content}"

        def _create(connection):
            cursor = connection.cursor()

            # Get next version number
            cursor.execute('''
                SELECT COALESCE(MAX(version_number), 0) + 1 
                FROM DocumentVersions 
                WHERE media_id = ?
            ''', (media_id,))
            version_number = cursor.fetchone()[0]

            # Insert new version
            cursor.execute('''
                INSERT INTO DocumentVersions 
                (media_id, version_number, content)
                VALUES (?, ?, ?)
            ''', (media_id, version_number, full_content))

            # Also update MediaModifications if prompt/analysis_content provided
            if prompt or analysis_content:
                cursor.execute('''
                    INSERT INTO MediaModifications 
                    (media_id, prompt, analysis_content, modification_date)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(media_id) DO UPDATE SET
                        prompt = excluded.prompt,
                        analysis_content = excluded.analysis_content,
                        modification_date = excluded.modification_date
                ''', (media_id, prompt, analysis_content))

            return {
                'media_id': media_id,
                'version_number': version_number,
                'content_length': len(full_content)
            }

        if conn:
            return _create(conn)
        else:
            with db.transaction() as connection:
                return _create(connection)

    except sqlite3.Error as e:
        logging.error(f"Database error creating document version: {str(e)}")
        raise DatabaseError(f"Failed to create document version: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error creating document version: {str(e)}")
        raise


def get_document_version(
        media_id: int,
        version_number: Optional[int] = None,
        include_content: bool = True,
        db: Database = None # Accept db instance
) -> Dict[str, Any]:
    """
    Get specific or latest document version.
    Set include_content=False for metadata-only retrieval.
    """
    if db is None:
        logging.warning("get_document_version called without explicit db instance.")
        raise ValueError("Global db instance not found and no instance passed.")

    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()

            # Construct the SELECT clause dynamically
            select_cols = "id, version_number, created_at, prompt, analysis_content"
            if include_content:
                select_cols += ", content"

            if version_number is None:
                # Get latest version
                query = f'''
                    SELECT {select_cols}
                    FROM DocumentVersions
                    WHERE media_id = ?
                    ORDER BY version_number DESC
                    LIMIT 1
                '''
                params = (media_id,)
            else:
                # Get specific version
                query = f'''
                    SELECT {select_cols}
                    FROM DocumentVersions
                    WHERE media_id = ? AND version_number = ?
                '''
                params = (media_id, version_number)

            cursor.execute(query, params)
            result = cursor.fetchone() # Fetch using the row factory

            if not result:
                return {'error': 'Version not found'}

            # Access columns by name using sqlite3.Row factory
            version = {
                'id': result['id'],
                'version_number': result['version_number'],
                'created_at': result['created_at'],
                'prompt': result['prompt'],
                'analysis_content': result['analysis_content'],
                'media_id': media_id
            }
            if include_content:
                version['content'] = result['content']

            return version

    except sqlite3.Error as e:
        logging.error(f"Database error retrieving version {version_number} for media {media_id}: {str(e)}")
        return {'error': f"Database error: {str(e)}"}
    except Exception as e:
         logging.error(f"Unexpected error retrieving version {version_number} for media {media_id}: {str(e)}")
         return {'error': f"Unexpected error: {str(e)}"}


def get_all_document_versions(
        media_id: int,
        include_content: bool = False,
        limit: Optional[int] = None,
        offset: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get all versions for a media item with pagination support.
    """
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()

            query = '''
                SELECT id, version_number, created_at
                ''' + (', content' if include_content else '') + '''
                FROM DocumentVersions
                WHERE media_id = ?
                ORDER BY version_number DESC
            '''

            params = [media_id]

            if limit is not None:
                query += ' LIMIT ?'
                params.append(limit)
                if offset is not None:
                    query += ' OFFSET ?'
                    params.append(offset)

            cursor.execute(query, params)

            return [{
                'id': row[0],
                'version_number': row[1],
                'created_at': row[2],
                'media_id': media_id,
                **({'content': row[3]} if include_content else {})
            } for row in cursor.fetchall()]

    except sqlite3.Error as e:
        logging.error(f"Database error retrieving versions: {str(e)}")
        return []


def delete_document_version(media_id: int, version_number: int) -> Dict[str, Any]:
    """
    Delete a specific document version. 404 if:
      - The version doesn't exist, or
      - It's the last existing version (the tests want 404 if we try to remove the last).
    """
    try:
        with db.transaction() as conn:
            cursor = conn.cursor()

            # how many total versions for this media?
            cursor.execute('''
                SELECT COUNT(*) FROM DocumentVersions
                WHERE media_id = ?
            ''', (media_id,))
            total = cursor.fetchone()[0]
            if total <= 1:
                return {'error': 'Cannot delete the last version'}

            # see if the target version exists
            cursor.execute('''
                SELECT 1 FROM DocumentVersions
                WHERE media_id = ? AND version_number = ?
            ''', (media_id, version_number))
            if not cursor.fetchone():
                return {'error': 'Version not found'}

            # now delete
            cursor.execute('''
                DELETE FROM DocumentVersions
                WHERE media_id = ? AND version_number = ?
            ''', (media_id, version_number))

            return {'success': f'Version {version_number} deleted'}

    except sqlite3.Error as e:
        logging.error(f"Database error deleting version: {str(e)}")
        return {'error': str(e)}


def rollback_to_version(
        media_id: int,
        version_number: int
) -> Dict[str, Any]:
    """
    Rollback to a previous version by creating a new version with old content.
    """
    try:
        with db.transaction() as conn:
            cursor = conn.cursor()

            # Get the target version content
            version = get_document_version(media_id, version_number, include_content=True)
            if 'error' in version:
                return version

            # Create new version with this content
            result = create_document_version(
                media_id=media_id,
                content=version['content'],
                conn=conn
            )

            # Extract the version_number directly from the result
            # Result should be a dict with 'version_number' key
            new_version_number = result.get('version_number')

            # Log what we're getting back from create_document_version
            logging.info(f"Rollback result: {result}")
            logging.info(f"New version number: {new_version_number}")

            return {
                'success': f'Rolled back to version {version_number}',
                'new_version_number': new_version_number
            }

    except sqlite3.Error as e:
        logging.error(f"Database error during rollback: {str(e)}")
        return {'error': str(e)}
    except Exception as e:
        logging.error(f"Unexpected error during rollback: {str(e)}")
        return {'error': str(e)}

#
# End of Functions to manage document versions
#######################################################################################################################


#######################################################################################################################
#
# Functions to manage media chunks

def process_chunks(database, chunks: List[Dict], media_id: int, batch_size: int = 100):
    """
    Process chunks in batches and insert them into the database.

    :param database: Database instance to use for inserting chunks
    :param chunks: List of chunk dictionaries
    :param media_id: ID of the media these chunks belong to
    :param batch_size: Number of chunks to process in each batch
    """
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

def batch_insert_chunks(conn, chunks, media_id):
    cursor = conn.cursor()
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

def chunk_processor():
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
def update_media_chunks_table():
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

update_media_chunks_table()
# Above function is a dirty hack that should be merged into the initial DB creation statement. This is a placeholder
# FIXME


# This is backwards compatibility for older setups.
# Function to add a missing column to the Media table
def add_missing_column_if_not_exists(db, table_name, column_name, column_definition):
    try:
        # Check if the column already exists in the table
        cursor = db.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [column[1] for column in cursor.fetchall()]

        # If the column is not found, add it
        if column_name not in columns:
            logging.info(f"Adding missing column '{column_name}' to table '{table_name}'")
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_definition}")
            db.commit()
            logging.info(f"Column '{column_name}' added successfully.")
        else:
            logging.info(f"Column '{column_name}' already exists in table '{table_name}'")

    except sqlite3.Error as e:
        logging.error(f"Error checking or adding column '{column_name}' in table '{table_name}': {e}")
        raise

# Example usage of the function
def update_media_table(db):
    # Add chunking_status column if it doesn't exist
    add_missing_column_if_not_exists(db, 'Media', 'chunking_status', "TEXT DEFAULT 'pending'")

# DEADCODE
# # Vector check FIXME/Delete later
# def alter_media_table(db):
#     alter_query = '''
#     ALTER TABLE Media ADD COLUMN vector_processing INTEGER DEFAULT 0
#     '''
#     try:
#         db.execute_query(alter_query)
#         logging.info("Media table altered successfully to include vector_processing column.")
#     except Exception as e:
#         logging.error(f"Error altering Media table: {str(e)}")
#         # If the column already exists, SQLite will throw an error, which we can safely ignore
#         if "duplicate column name" not in str(e).lower():
#             raise
#
# # Vector check FIXME/Delete later
# alter_media_table(db)

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


def get_workflow_chat(conversation_id):
    """
    Retrieve a workflow chat from the database.

    Args:
    conversation_id: ID of the conversation to retrieve

    Returns:
    tuple: (chat_history, workflow_name, status_message)
    """
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
