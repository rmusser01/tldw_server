# Media_DB_v2.py (Refactored for Multi-DB Instances & Internal Sync Meta)
#########################################
# Media_DB_v2 Library
# Manages Media_DB_v2 operations for specific instances, handling sync metadata internally.
# Requires a client_id during Database initialization.
# Standalone functions require a Database instance passed as an argument.
#
# Manages SQLite database interactions for media and related metadata.
#
# This library provides a `Database` class to encapsulate operations for a specific
# SQLite database file. It handles connection management (thread-locally),
# schema initialization and versioning, CRUD operations, Full-Text Search (FTS)
# updates, and internal logging of changes for synchronization purposes via a
# `sync_log` table.
#
# Key Features:
# - Instance-based: Each `Database` object connects to a specific DB file.
# - Client ID Tracking: Requires a `client_id` for attributing changes.
# - Internal Sync Logging: Automatically logs creates, updates, deletes, links,
#   and unlinks to the `sync_log` table for external sync processing.
# - Internal FTS Updates: Manages associated FTS5 tables (`media_fts`, `keyword_fts`)
#   within the Python code during relevant operations.
# - Schema Versioning: Checks and applies schema updates upon initialization.
# - Thread-Safety: Uses thread-local storage for database connections.
# - Soft Deletes: Implements soft deletes (`deleted=1`) for most entities,
#   allowing for recovery and synchronization of deletions.
# - Transaction Management: Provides a context manager for atomic operations.
# - Standalone Functions: Offers utility functions that operate on a `Database`
#   instance (e.g., searching, fetching related data, maintenance).
####
import hashlib
import json
import sqlite3
import threading
import time
import uuid  # For UUID generation
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta  # Use timezone-aware UTC
from math import ceil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
#
# Third-Party Libraries (Ensure these are installed if used)
# import gradio as gr # Removed if Gradio interfaces moved out
# import pandas as pd # Removed if Pandas formatting moved out
# import yaml # Keep if Obsidian import uses it
#
########################################################################################################################
#
# Functions:

# --- Logging Setup ---
# Assume logger is configured elsewhere or use basic config:
import logging

import yaml

from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Custom Exceptions ---
class DatabaseError(Exception):
    """Base exception for database related errors."""
    pass

class SchemaError(DatabaseError):
    """Exception for schema version mismatches or migration failures."""
    pass

class InputError(ValueError):
    """Custom exception for input validation errors."""
    pass

class ConflictError(DatabaseError):
    """Indicates a conflict due to concurrent modification (version mismatch)."""
    def __init__(self, message="Conflict detected: Record modified concurrently.", entity=None, identifier=None):
        super().__init__(message)
        self.entity = entity
        self.identifier = identifier  # Can be id or uuid

    def __str__(self):
        base = super().__str__()
        details = []
        if self.entity:
            details.append(f"Entity: {self.entity}")
        if self.identifier:
            details.append(f"ID: {self.identifier}")
        return f"{base} ({', '.join(details)})" if details else base

# --- Database Class ---
class Database:
    """
    Manages SQLite connection and operations for a specific database file,
    handling sync metadata and FTS updates internally via Python code.
    Requires client_id on initialization. Includes schema versioning.
    """
    _CURRENT_SCHEMA_VERSION = 1  # Define the version this code supports

    # <<< Schema Definition (Version 1) >>>

    _TABLES_SQL_V1 = """
    PRAGMA foreign_keys = ON;

    -- Schema Version Table --
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY NOT NULL
    );
    -- Initialize version if table is newly created
    INSERT OR IGNORE INTO schema_version (version) VALUES (0);

    -- Media Table --
    CREATE TABLE IF NOT EXISTS Media (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE,
        title TEXT NOT NULL,
        type TEXT NOT NULL,
        content TEXT,
        author TEXT,
        ingestion_date DATETIME,
        transcription_model TEXT,
        is_trash BOOLEAN DEFAULT 0 NOT NULL,
        trash_date DATETIME,
        vector_embedding BLOB,
        chunking_status TEXT DEFAULT 'pending' NOT NULL,
        vector_processing INTEGER DEFAULT 0 NOT NULL,
        content_hash TEXT UNIQUE NOT NULL,
        uuid TEXT UNIQUE NOT NULL,
        last_modified DATETIME NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT
    );

    -- Keywords Table --
    CREATE TABLE IF NOT EXISTS Keywords (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        keyword TEXT NOT NULL UNIQUE COLLATE NOCASE,
        uuid TEXT UNIQUE NOT NULL,
        last_modified DATETIME NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT
    );

    -- MediaKeywords Table (Junction Table) --
    CREATE TABLE IF NOT EXISTS MediaKeywords (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER NOT NULL,
        keyword_id INTEGER NOT NULL,
        UNIQUE (media_id, keyword_id),
        FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,
        FOREIGN KEY (keyword_id) REFERENCES Keywords(id) ON DELETE CASCADE
    );

    -- Transcripts Table --
    CREATE TABLE IF NOT EXISTS Transcripts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER NOT NULL,
        whisper_model TEXT,
        transcription TEXT,
        created_at DATETIME,
        uuid TEXT UNIQUE NOT NULL,
        last_modified DATETIME NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT,
        UNIQUE (media_id, whisper_model),
        FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE
    );

    -- MediaChunks Table --
    CREATE TABLE IF NOT EXISTS MediaChunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        start_index INTEGER,
        end_index INTEGER,
        chunk_id TEXT UNIQUE,
        uuid TEXT UNIQUE NOT NULL,
        last_modified DATETIME NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT,
        FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE
    );

    -- UnvectorizedMediaChunks Table --
    CREATE TABLE IF NOT EXISTS UnvectorizedMediaChunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        start_char INTEGER,
        end_char INTEGER,
        chunk_type TEXT,
        creation_date DATETIME,
        last_modified_orig DATETIME,
        is_processed BOOLEAN DEFAULT FALSE NOT NULL,
        metadata TEXT,
        uuid TEXT UNIQUE NOT NULL,
        last_modified DATETIME NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT,
        UNIQUE (media_id, chunk_index, chunk_type),
        FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE
    );

    -- DocumentVersions Table --
    CREATE TABLE IF NOT EXISTS DocumentVersions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        media_id INTEGER NOT NULL,
        version_number INTEGER NOT NULL,
        prompt TEXT,
        analysis_content TEXT,
        content TEXT NOT NULL,
        created_at DATETIME,
        uuid TEXT UNIQUE NOT NULL,
        last_modified DATETIME NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT,
        FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,
        UNIQUE (media_id, version_number)
    );

    -- Sync Log Table --
    CREATE TABLE IF NOT EXISTS sync_log (
        change_id INTEGER PRIMARY KEY AUTOINCREMENT,
        entity TEXT NOT NULL,
        entity_uuid TEXT NOT NULL,
        operation TEXT NOT NULL CHECK(operation IN ('create','update','delete', 'link', 'unlink')),
        timestamp DATETIME NOT NULL,
        client_id TEXT NOT NULL,
        version INTEGER NOT NULL,
        payload TEXT
    );
    """

    _INDICES_SQL_V1 = """
    -- Indices (Create after tables exist) --
    CREATE INDEX IF NOT EXISTS idx_media_title ON Media(title);
    CREATE INDEX IF NOT EXISTS idx_media_type ON Media(type);
    CREATE INDEX IF NOT EXISTS idx_media_author ON Media(author);
    CREATE INDEX IF NOT EXISTS idx_media_ingestion_date ON Media(ingestion_date);
    CREATE INDEX IF NOT EXISTS idx_media_chunking_status ON Media(chunking_status);
    CREATE INDEX IF NOT EXISTS idx_media_vector_processing ON Media(vector_processing);
    CREATE INDEX IF NOT EXISTS idx_media_is_trash ON Media(is_trash);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_media_content_hash ON Media(content_hash);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_media_uuid ON Media(uuid);
    CREATE INDEX IF NOT EXISTS idx_media_last_modified ON Media(last_modified);
    CREATE INDEX IF NOT EXISTS idx_media_deleted ON Media(deleted);
    CREATE INDEX IF NOT EXISTS idx_media_prev_version ON Media(prev_version);
    CREATE INDEX IF NOT EXISTS idx_media_merge_parent_uuid ON Media(merge_parent_uuid);

    CREATE UNIQUE INDEX IF NOT EXISTS idx_keywords_uuid ON Keywords(uuid);
    CREATE INDEX IF NOT EXISTS idx_keywords_last_modified ON Keywords(last_modified);
    CREATE INDEX IF NOT EXISTS idx_keywords_deleted ON Keywords(deleted);
    CREATE INDEX IF NOT EXISTS idx_keywords_prev_version ON Keywords(prev_version);
    CREATE INDEX IF NOT EXISTS idx_keywords_merge_parent_uuid ON Keywords(merge_parent_uuid);

    CREATE INDEX IF NOT EXISTS idx_mediakeywords_media_id ON MediaKeywords(media_id);
    CREATE INDEX IF NOT EXISTS idx_mediakeywords_keyword_id ON MediaKeywords(keyword_id);

    CREATE INDEX IF NOT EXISTS idx_transcripts_media_id ON Transcripts(media_id);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_transcripts_uuid ON Transcripts(uuid);
    CREATE INDEX IF NOT EXISTS idx_transcripts_last_modified ON Transcripts(last_modified);
    CREATE INDEX IF NOT EXISTS idx_transcripts_deleted ON Transcripts(deleted);
    CREATE INDEX IF NOT EXISTS idx_transcripts_prev_version ON Transcripts(prev_version);
    CREATE INDEX IF NOT EXISTS idx_transcripts_merge_parent_uuid ON Transcripts(merge_parent_uuid);

    CREATE INDEX IF NOT EXISTS idx_mediachunks_media_id ON MediaChunks(media_id);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_mediachunks_uuid ON MediaChunks(uuid);
    CREATE INDEX IF NOT EXISTS idx_mediachunks_last_modified ON MediaChunks(last_modified);
    CREATE INDEX IF NOT EXISTS idx_mediachunks_deleted ON MediaChunks(deleted);
    CREATE INDEX IF NOT EXISTS idx_mediachunks_prev_version ON MediaChunks(prev_version);
    CREATE INDEX IF NOT EXISTS idx_mediachunks_merge_parent_uuid ON MediaChunks(merge_parent_uuid);

    CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_media_id ON UnvectorizedMediaChunks(media_id);
    CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_is_processed ON UnvectorizedMediaChunks(is_processed);
    CREATE INDEX IF NOT EXISTS idx_unvectorized_media_chunks_chunk_type ON UnvectorizedMediaChunks(chunk_type);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_uuid ON UnvectorizedMediaChunks(uuid);
    CREATE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_last_modified ON UnvectorizedMediaChunks(last_modified);
    CREATE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_deleted ON UnvectorizedMediaChunks(deleted);
    CREATE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_prev_version ON UnvectorizedMediaChunks(prev_version);
    CREATE INDEX IF NOT EXISTS idx_unvectorizedmediachunks_merge_parent_uuid ON UnvectorizedMediaChunks(merge_parent_uuid);

    CREATE INDEX IF NOT EXISTS idx_document_versions_media_id ON DocumentVersions(media_id);
    CREATE INDEX IF NOT EXISTS idx_document_versions_version_number ON DocumentVersions(version_number);
    CREATE UNIQUE INDEX IF NOT EXISTS idx_documentversions_uuid ON DocumentVersions(uuid);
    CREATE INDEX IF NOT EXISTS idx_documentversions_last_modified ON DocumentVersions(last_modified);
    CREATE INDEX IF NOT EXISTS idx_documentversions_deleted ON DocumentVersions(deleted);
    CREATE INDEX IF NOT EXISTS idx_documentversions_prev_version ON DocumentVersions(prev_version);
    CREATE INDEX IF NOT EXISTS idx_documentversions_merge_parent_uuid ON DocumentVersions(merge_parent_uuid);

    CREATE INDEX IF NOT EXISTS idx_sync_log_ts ON sync_log(timestamp);
    CREATE INDEX IF NOT EXISTS idx_sync_log_entity_uuid ON sync_log(entity_uuid);
    CREATE INDEX IF NOT EXISTS idx_sync_log_client_id ON sync_log(client_id);
    """

    _TRIGGERS_SQL_V1 = """
    -- Validation Triggers (Create after tables and indices) --
    DROP TRIGGER IF EXISTS media_validate_sync_update;
    CREATE TRIGGER media_validate_sync_update BEFORE UPDATE ON Media
    BEGIN
        SELECT RAISE(ABORT, 'Sync Error (Media): Version must increment by exactly 1.')
        WHERE NEW.version IS NOT OLD.version + 1;
        SELECT RAISE(ABORT, 'Sync Error (Media): Client ID cannot be NULL or empty.')
        WHERE NEW.client_id IS NULL OR NEW.client_id = '';
        -- Add more checks if needed (e.g., UUID modification)
        SELECT RAISE(ABORT, 'Sync Error (Media): UUID cannot be changed.')
        WHERE NEW.uuid IS NOT OLD.uuid;
    END;

    DROP TRIGGER IF EXISTS keywords_validate_sync_update;
    CREATE TRIGGER keywords_validate_sync_update BEFORE UPDATE ON Keywords
    BEGIN
        SELECT RAISE(ABORT, 'Sync Error (Keywords): Version must increment by exactly 1.')
        WHERE NEW.version IS NOT OLD.version + 1;
        SELECT RAISE(ABORT, 'Sync Error (Keywords): Client ID cannot be NULL or empty.')
        WHERE NEW.client_id IS NULL OR NEW.client_id = '';
        SELECT RAISE(ABORT, 'Sync Error (Keywords): UUID cannot be changed.')
        WHERE NEW.uuid IS NOT OLD.uuid;
    END;

    DROP TRIGGER IF EXISTS transcripts_validate_sync_update;
    CREATE TRIGGER transcripts_validate_sync_update BEFORE UPDATE ON Transcripts
    BEGIN
        SELECT RAISE(ABORT, 'Sync Error (Transcripts): Version must increment by exactly 1.')
        WHERE NEW.version IS NOT OLD.version + 1;
        SELECT RAISE(ABORT, 'Sync Error (Transcripts): Client ID cannot be NULL or empty.')
        WHERE NEW.client_id IS NULL OR NEW.client_id = '';
        SELECT RAISE(ABORT, 'Sync Error (Transcripts): UUID cannot be changed.')
        WHERE NEW.uuid IS NOT OLD.uuid;
    END;

    DROP TRIGGER IF EXISTS mediachunks_validate_sync_update;
    CREATE TRIGGER mediachunks_validate_sync_update BEFORE UPDATE ON MediaChunks
    BEGIN
        SELECT RAISE(ABORT, 'Sync Error (MediaChunks): Version must increment by exactly 1.')
        WHERE NEW.version IS NOT OLD.version + 1;
        SELECT RAISE(ABORT, 'Sync Error (MediaChunks): Client ID cannot be NULL or empty.')
        WHERE NEW.client_id IS NULL OR NEW.client_id = '';
        SELECT RAISE(ABORT, 'Sync Error (MediaChunks): UUID cannot be changed.')
        WHERE NEW.uuid IS NOT OLD.uuid;
    END;

    DROP TRIGGER IF EXISTS unvectorizedmediachunks_validate_sync_update;
    CREATE TRIGGER unvectorizedmediachunks_validate_sync_update BEFORE UPDATE ON UnvectorizedMediaChunks
    BEGIN
        SELECT RAISE(ABORT, 'Sync Error (UnvectorizedMediaChunks): Version must increment by exactly 1.')
        WHERE NEW.version IS NOT OLD.version + 1;
        SELECT RAISE(ABORT, 'Sync Error (UnvectorizedMediaChunks): Client ID cannot be NULL or empty.')
        WHERE NEW.client_id IS NULL OR NEW.client_id = '';
        SELECT RAISE(ABORT, 'Sync Error (UnvectorizedMediaChunks): UUID cannot be changed.')
        WHERE NEW.uuid IS NOT OLD.uuid;
    END;

    DROP TRIGGER IF EXISTS documentversions_validate_sync_update;
    CREATE TRIGGER documentversions_validate_sync_update BEFORE UPDATE ON DocumentVersions
    BEGIN
        SELECT RAISE(ABORT, 'Sync Error (DocumentVersions): Version must increment by exactly 1.')
        WHERE NEW.version IS NOT OLD.version + 1;
        SELECT RAISE(ABORT, 'Sync Error (DocumentVersions): Client ID cannot be NULL or empty.')
        WHERE NEW.client_id IS NULL OR NEW.client_id = '';
        SELECT RAISE(ABORT, 'Sync Error (DocumentVersions): UUID cannot be changed.')
        WHERE NEW.uuid IS NOT OLD.uuid;
    END;
    """

    _FTS_TABLES_SQL = """
    -- FTS Tables (Executed Separately) --
    CREATE VIRTUAL TABLE IF NOT EXISTS media_fts USING fts5(
        title,
        content,
        content='Media',    -- Keep reference to source table
        content_rowid='id' -- Link to Media.id
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS keyword_fts USING fts5(
        keyword,
        content='Keywords',    -- Keep reference to source table
        content_rowid='id'  -- Link to Keywords.id
    );
    """

    def __init__(self, db_path: Union[str, Path], client_id: str):
        """
        Initializes the Database instance, sets up the connection pool (via threading.local),
        and ensures the database schema is correctly initialized or migrated.

        Args:
            db_path (Union[str, Path]): The path to the SQLite database file or ':memory:'.
            client_id (str): A unique identifier for the client using this database instance.

        Raises:
            ValueError: If client_id is empty or None.
            DatabaseError: If database initialization or schema setup fails.
        """
        # Determine if it's an in-memory DB and resolve the path
        if isinstance(db_path, Path):
            self.is_memory_db = False
            self.db_path = db_path.resolve()
        else:  # Treat as string
            self.is_memory_db = (db_path == ':memory:')
            if not self.is_memory_db:
                self.db_path = Path(db_path).resolve()
            else:
                # Even for memory, Path object can be useful internally, though str is ':memory:'
                self.db_path = Path(":memory:")  # Represent in-memory path consistently

        # Store the path as a string for convenience/logging
        self.db_path_str = str(self.db_path) if not self.is_memory_db else ':memory:'

        # Validate client_id
        if not client_id:
            raise ValueError("Client ID cannot be empty or None.")
        self.client_id = client_id

        # Ensure parent directory exists if it's a file-based DB
        if not self.is_memory_db:
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                # Catch potential errors creating the directory (e.g., permissions)
                raise DatabaseError(f"Failed to create database directory {self.db_path.parent}: {e}") from e

        logging.info(f"Initializing Database object for path: {self.db_path_str} [Client ID: {self.client_id}]")

        # Initialize thread-local storage for connections
        self._local = threading.local()

        # Flag to track successful initialization before logging completion
        initialization_successful = False
        try:
            # --- Core Initialization Logic ---
            # This establishes the first connection for the current thread
            # and applies/verifies the schema.
            self._initialize_schema()
            initialization_successful = True  # Mark as successful if no exception occurred
        except (DatabaseError, SchemaError, sqlite3.Error) as e:
            # Catch specific DB/Schema errors and general SQLite errors during init
            logging.critical(f"FATAL: DB Initialization failed for {self.db_path_str}: {e}", exc_info=True)
            # Attempt to clean up the connection before raising
            self.close_connection()
            # Re-raise as a DatabaseError to signal catastrophic failure
            raise DatabaseError(f"Database initialization failed: {e}") from e
        except Exception as e:
            # Catch any other unexpected errors during initialization
            logging.critical(f"FATAL: Unexpected error during DB Initialization for {self.db_path_str}: {e}", exc_info=True)
            # Attempt cleanup
            self.close_connection()
            # Re-raise as a DatabaseError
            raise DatabaseError(f"Unexpected database initialization error: {e}") from e
        finally:
            # Log completion status based on the flag
            if initialization_successful:
                logging.debug(f"Database initialization completed successfully for {self.db_path_str}")
            else:
                # This path indicates an exception was caught and raised above.
                # Logging here provides context that the __init__ block finished, albeit with failure.
                logging.error(f"Database initialization block finished for {self.db_path_str}, but failed.")

    # --- Connection Management (Unchanged) ---
    def _get_thread_connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, 'conn', None)
        is_closed = True
        if conn:
            try:
                conn.execute("SELECT 1")  # Simple check
                is_closed = False
            except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                logging.warning(f"Thread-local connection to {self.db_path_str} was closed. Reopening.")
                is_closed = True
                try:
                    conn.close()
                except Exception:
                    pass
                self._local.conn = None

        if is_closed:
            try:
                conn = sqlite3.connect(
                    self.db_path_str,
                    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                    check_same_thread=False,
                    timeout=10
                )
                conn.row_factory = sqlite3.Row
                if not self.is_memory_db:
                    conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA foreign_keys = ON;")
                self._local.conn = conn
                logging.debug(f"Opened/Reopened SQLite connection to {self.db_path_str} [Client: {self.client_id}, Thread: {threading.current_thread().name}]")
            except sqlite3.Error as e:
                logging.error(f"Failed to connect to database at {self.db_path_str}: {e}", exc_info=True)
                self._local.conn = None
                raise DatabaseError(f"Failed to connect to database '{self.db_path_str}': {e}") from e
        return self._local.conn

    def get_connection(self) -> sqlite3.Connection:
        """
        Provides the active database connection for the current thread.

        This is the public method to retrieve a connection managed by this instance.

        Returns:
            sqlite3.Connection: The thread-local database connection.
        """
        return self._get_thread_connection()

    def close_connection(self):
        """Closes the database connection for the current thread, if open."""
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            try:
                conn = self._local.conn
                self._local.conn = None  # Remove ref before closing
                conn.close()
                logging.debug(f"Closed connection for thread {threading.current_thread().name}.")
            except sqlite3.Error as e:
                logging.warning(f"Error closing connection: {e}")
            finally:
                if hasattr(self._local, 'conn'):  # Paranoid check
                    self._local.conn = None

    # --- Query Execution (Unchanged, catches IntegrityError from validation triggers) ---
    def execute_query(self, query: str, params: tuple = None, *, commit: bool = False) -> sqlite3.Cursor:
        """
         Executes a single SQL query.

         Args:
             query (str): The SQL query string.
             params (Optional[tuple]): Parameters to substitute into the query.
             commit (bool): If True, commit the transaction after execution.
                            Defaults to False. Usually managed by `transaction()`.

         Returns:
             sqlite3.Cursor: The cursor object after execution.

         Raises:
             DatabaseError: For general SQLite errors or integrity violations
                            not related to sync validation.
             sqlite3.IntegrityError: Specifically re-raised if a sync validation
                                     trigger (defined in schema) fails.
         """
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            logging.debug(f"Executing Query: {query[:200]}... Params: {str(params)[:100]}...")
            cursor.execute(query, params or ())
            if commit:
                conn.commit()
                logging.debug("Committed.")
            return cursor
        except sqlite3.IntegrityError as e:  # Catch validation errors specifically
            msg = str(e).lower()
            if "sync error" in msg:
                logging.error(f"Sync Validation Failed: {e}")
                raise e  # Re-raise the specific IntegrityError
            else:  # Other integrity errors
                logging.error(f"Integrity error: {query[:200]}... Error: {e}", exc_info=True)
                raise DatabaseError(f"Integrity constraint violation: {e}") from e
        except sqlite3.Error as e:  # Other SQLite errors
            logging.error(f"Query failed: {query[:200]}... Error: {e}", exc_info=True)
            raise DatabaseError(f"Query execution failed: {e}") from e

    def execute_many(self, query: str, params_list: List[tuple], *, commit: bool = False) -> Optional[sqlite3.Cursor]:
        """
        Executes a SQL query for multiple sets of parameters.

        Args:
            query (str): The SQL query string (e.g., INSERT INTO ... VALUES (?,?)).
            params_list (List[tuple]): A list of tuples, each tuple containing
                                       parameters for one execution.
            commit (bool): If True, commit the transaction after execution.
                           Defaults to False. Usually managed by `transaction()`.

        Returns:
            Optional[sqlite3.Cursor]: The cursor object after execution, or None if
                                     `params_list` was empty.

        Raises:
            TypeError: If `params_list` is not a list or contains invalid data types.
            DatabaseError: For general SQLite errors or integrity violations.
        """
        conn = self.get_connection()
        if not isinstance(params_list, list):
            raise TypeError("params_list must be a list.")
        if not params_list:
            return None
        try:
            cursor = conn.cursor()
            logging.debug(f"Executing Many: {query[:150]}... with {len(params_list)} sets.")
            cursor.executemany(query, params_list)
            if commit:
                conn.commit()
                logging.debug("Committed Many.")
            return cursor
        except sqlite3.IntegrityError as e:
            logging.error(f"Integrity error during Execute Many: {query[:150]}... Error: {e}", exc_info=True)
            raise DatabaseError(f"Integrity constraint violation during batch: {e}") from e
        except sqlite3.Error as e:
            logging.error(f"Execute Many failed: {query[:150]}... Error: {e}", exc_info=True)
            raise DatabaseError(f"Execute Many failed: {e}") from e
        except TypeError as te:
            logging.error(f"TypeError during Execute Many: {te}. Check params_list format.", exc_info=True)
            raise TypeError(f"Parameter list format error: {te}") from te

    # --- Transaction Context (Unchanged) ---
    @contextmanager
    def transaction(self):
        """
        Provides a context manager for database transactions.

        Ensures that a block of operations is executed atomically. Commits
        on successful exit, rolls back on any exception. Handles nested
        transactions gracefully (only outermost commit/rollback matters).

        Yields:
            sqlite3.Connection: The current thread's database connection.

        Raises:
            Exception: Re-raises any exception that occurs within the block
                       after attempting a rollback.
        """
        conn = self.get_connection()
        in_outer = conn.in_transaction
        try:
            if not in_outer:
                conn.execute("BEGIN")
                logging.debug("Started transaction.")
            # Yield the connection
            yield conn
            if not in_outer:
                conn.commit()
                logging.debug("Committed transaction.")
        except Exception as e:
            if not in_outer:
                logging.error(f"Transaction failed, rolling back: {type(e).__name__} - {e}", exc_info=False)
                try:
                    conn.rollback()
                    logging.debug("Rollback successful.")
                except sqlite3.Error as rb_err:
                    logging.error(f"Rollback FAILED: {rb_err}", exc_info=True)
            raise e

    # --- Schema Initialization and Migration ---
    def _get_db_version(self, conn: sqlite3.Connection) -> int:
        """Internal helper to get the current schema version."""
        try:
            cursor = conn.execute("SELECT version FROM schema_version LIMIT 1")
            result = cursor.fetchone()
            return result['version'] if result else 0
        except sqlite3.Error as e:
            if "no such table: schema_version" in str(e):
                return 0  # Table doesn't exist yet
            else:
                raise DatabaseError(f"Could not determine database schema version: {e}") from e

    _SCHEMA_UPDATE_VERSION_SQL_V1 = "UPDATE schema_version SET version = 1 WHERE version = 0;"

    def _apply_schema_v1(self, conn: sqlite3.Connection):
        """Applies the full Version 1 schema, ensuring version update is part of the main script."""
        logging.info(f"Applying initial schema (Version 1) to DB: {self.db_path_str}...")
        try:
            # --- Combine Core Schema + Version Update for one executescript call ---
            # This ensures the version update is part of the same atomic operation block
            # executed by executescript within the transaction.
            core_schema_script_with_version_update = f"""
                {self._TABLES_SQL_V1}
                {self._INDICES_SQL_V1}
                {self._TRIGGERS_SQL_V1}
                {self._SCHEMA_UPDATE_VERSION_SQL_V1}
            """  # Note the added UPDATE statement

            # --- Transaction for Core Schema + Version Update ---
            with self.transaction():  # Use the transaction context manager
                logging.debug("[Schema V1] Applying Core Schema + Version Update...")
                conn.executescript(core_schema_script_with_version_update)
                logging.debug("[Schema V1] Core Schema script (incl. version update) executed.")

                # --- Validation step (optional but good) - Check Media table ---
                try:
                    cursor = conn.execute("PRAGMA table_info(Media)")
                    columns = {row['name'] for row in cursor.fetchall()}
                    # Update this set to match ALL columns defined in _TABLES_SQL_V1.Media
                    expected_cols = {'id', 'url', 'title', 'type', 'content', 'author', 'ingestion_date', 'transcription_model', 'is_trash', 'trash_date', 'vector_embedding', 'chunking_status', 'vector_processing', 'content_hash', 'uuid', 'last_modified', 'version', 'client_id', 'deleted', 'prev_version', 'merge_parent_uuid'}
                    if not expected_cols.issubset(columns):
                        missing_cols = expected_cols - columns
                        raise SchemaError(f"Validation Error: Media table is missing columns after creation: {missing_cols}")
                    logging.debug("[Schema V1] Media table structure validated successfully.")
                except (sqlite3.Error, SchemaError) as val_err:
                    logging.error(f"[Schema V1] Validation failed after table creation: {val_err}", exc_info=True)
                    raise  # Re-raise to trigger rollback

                # --- Explicitly check version *inside* transaction AFTER script ---
                # This helps debug if the update itself isn't working
                cursor_check = conn.execute("SELECT version FROM schema_version LIMIT 1")
                version_in_tx = cursor_check.fetchone()
                if not version_in_tx or version_in_tx['version'] != 1:
                    logging.error(f"[Schema V1] Version check *inside* transaction failed. Found: {version_in_tx['version'] if version_in_tx else 'None'}")
                    raise SchemaError("Schema version update did not take effect within the transaction.")
                logging.debug(f"[Schema V1] Version check inside transaction confirmed version is 1.")

            # Transaction commits here if all steps above succeeded
            logging.info(f"[Schema V1] Core Schema V1 (incl. version update) applied and committed successfully for DB: {self.db_path_str}.")

            # --- Create FTS Tables Separately (Remains the same) ---
            try:
                logging.debug("[Schema V1] Applying FTS Tables...")
                conn.executescript(self._FTS_TABLES_SQL)
                conn.commit()
                logging.info("[Schema V1] FTS Tables created successfully.")
            except sqlite3.Error as fts_err:
                logging.error(f"[Schema V1] Failed to create FTS tables: {fts_err}", exc_info=True)

        except sqlite3.Error as e:
            logging.error(f"[Schema V1] Application failed during core transaction: {e}", exc_info=True)
            raise DatabaseError(f"DB schema V1 setup failed: {e}") from e
        except Exception as e:
            logging.error(f"[Schema V1] Unexpected error during schema V1 application: {e}", exc_info=True)
            raise DatabaseError(f"Unexpected error applying schema V1: {e}") from e

    def _initialize_schema(self):
        """Checks schema version and applies initial schema or migrations."""
        conn = self.get_connection()
        try:
            current_db_version = self._get_db_version(conn)
            target_version = self._CURRENT_SCHEMA_VERSION

            logging.info(f"Checking DB schema. Current version: {current_db_version}. Code supports: {target_version}")

            if current_db_version == target_version:
                logging.debug("Database schema is up to date.")
                # Optionally ensure FTS tables exist even if schema version matches
                try:
                    conn.executescript(self._FTS_TABLES_SQL)
                    conn.commit()
                    logging.debug("Verified FTS tables exist.")
                except sqlite3.Error as fts_err:
                    logging.warning(f"Could not verify/create FTS tables on already correct schema version: {fts_err}")
                return

            if current_db_version > target_version:
                raise SchemaError(f"Database schema version ({current_db_version}) is newer than supported by code ({target_version}).")

            # --- Apply Migrations ---
            if current_db_version == 0:
                self._apply_schema_v1(conn)  # Call the updated method
                # Verify version update AFTER _apply_schema_v1 has committed
                final_db_version = self._get_db_version(conn)
                if final_db_version != target_version:
                    # If this fails now, it means the commit didn't work or the read is stale
                    raise SchemaError(f"Schema migration applied, but final DB version is {final_db_version}, expected {target_version}. Manual check required.")
                logging.info(f"Database schema initialized/migrated to version {target_version}.")

            else:
                raise SchemaError(f"Migration needed from version {current_db_version} to {target_version}, but no migration path is defined in the code.")

        except (DatabaseError, SchemaError, sqlite3.Error) as e:
            logging.error(f"Schema initialization/migration failed: {e}", exc_info=True)
            raise DatabaseError(f"Schema initialization failed: {e}") from e
        except Exception as e:
            logging.error(f"Unexpected error during schema initialization: {e}", exc_info=True)
            raise DatabaseError(f"Unexpected error applying schema: {e}") from e

    # --- Internal Helpers (Unchanged) ---
    def _get_current_utc_timestamp_str(self) -> str:
        """
        Internal helper to generate a UTC timestamp string in ISO 8601 format.

        Returns:
            str: Timestamp string (e.g., '2023-10-27T10:30:00.123Z').
        """
        # Use ISO 8601 format with Z for UTC, more standard
        return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    def _generate_uuid(self) -> str:
        """
        Internal helper to generate a new UUID string.

        Returns:
            str: A unique UUID version 4 string.
        """
        return str(uuid.uuid4())

    def _get_next_version(self, conn: sqlite3.Connection, table: str, id_col: str, id_val: Any) -> Optional[Tuple[int, int]]:
        """
        Internal helper to get the current and next sync version for a record.

        Fetches the current 'version' column value for a given record and
        returns it along with the incremented next version number. Used for
        optimistic concurrency checks during updates.

        Args:
            conn (sqlite3.Connection): The database connection.
            table (str): The table name.
            id_col (str): The name of the identifier column (e.g., 'id', 'uuid').
            id_val (Any): The value of the identifier.

        Returns:
            Optional[Tuple[int, int]]: A tuple containing (current_version, next_version)
                                       if the record exists and has an integer version,
                                       otherwise None.

        Raises:
            DatabaseError: If the database query fails.
        """
        try:
            cursor = conn.execute(f"SELECT version FROM {table} WHERE {id_col} = ? AND deleted = 0", (id_val,))
            result = cursor.fetchone()
            if result:
                current_version = result['version']
                if isinstance(current_version, int):
                    return current_version, current_version + 1
                else:
                    logging.error(f"Invalid non-integer version '{current_version}' found for {table} {id_col}={id_val}")
                    return None
        except sqlite3.Error as e:
            logging.error(f"Database error fetching version for {table} {id_col}={id_val}: {e}")
            raise DatabaseError(f"Failed to fetch current version: {e}") from e
        return None

    # --- Internal Sync Logging Helper ---
    def _log_sync_event(self, conn: sqlite3.Connection, entity: str, entity_uuid: str, operation: str, version: int, payload: Optional[Dict] = None):
        """
        Internal helper to insert a record into the sync_log table.

        This should be called within an active transaction context after a
        successful data modification (insert, update, delete, link, unlink).

        Args:
            conn (sqlite3.Connection): The database connection (within transaction).
            entity (str): The name of the entity/table being changed (e.g., "Media").
            entity_uuid (str): The UUID of the entity affected. For links/unlinks,
                               this might be a composite identifier.
            operation (str): The type of operation ('create', 'update', 'delete',
                             'link', 'unlink').
            version (int): The new sync version number of the entity after the change.
            payload (Optional[Dict]): A dictionary containing relevant data about
                                      the change (e.g., the updated row). Sensitive
                                      or large fields like 'vector_embedding' are
                                      automatically excluded. Defaults to None.

        Raises:
            DatabaseError: If the sync log insertion fails.
        """
        if not entity or not entity_uuid or not operation:
            logging.error("Sync log attempt with missing entity, uuid, or operation.")
            return

        current_time = self._get_current_utc_timestamp_str()  # Generate timestamp here
        client_id = self.client_id

        # Exclude potentially large/binary fields from default payload logging
        if payload:
            payload = payload.copy()  # Avoid modifying the original dict
            if 'vector_embedding' in payload:
                del payload['vector_embedding']
            #  Add other fields to exclude if necessary

        payload_json = json.dumps(payload, separators=(',', ':')) if payload else None  # Compact JSON

        try:
            conn.execute("""
                INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (entity, entity_uuid, operation, current_time, client_id, version, payload_json))  # Pass current_time
            logging.debug(f"Logged sync event: {entity} {entity_uuid} {operation} v{version} at {current_time}")
        except sqlite3.Error as e:
            logging.error(f"Failed to insert sync log event for {entity} {entity_uuid}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to log sync event: {e}") from e

    # --- NEW: Internal FTS Helper Methods ---
    def _update_fts_media(self, conn: sqlite3.Connection, media_id: int, title: str, content: Optional[str]):
        """
        Internal helper to update or insert into the media_fts table.

        Uses INSERT OR REPLACE to handle both creating new FTS entries and
        updating existing ones based on the Media.id (rowid). Should be called
        within a transaction after Media insert/update.

        Args:
            conn (sqlite3.Connection): The database connection (within transaction).
            media_id (int): The ID (rowid) of the Media item.
            title (str): The title of the media.
            content (Optional[str]): The content of the media. Empty string if None.

        Raises:
            DatabaseError: If the FTS update fails.
        """
        content = content or ""
        try:
            # Use INSERT OR REPLACE
            conn.execute("INSERT OR REPLACE INTO media_fts (rowid, title, content) VALUES (?, ?, ?)",
                           (media_id, title, content))
            logging.debug(f"Updated FTS (insert or replace) for Media ID {media_id}")
        except sqlite3.Error as e:
            logging.error(f"Failed to update media_fts for Media ID {media_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to update FTS for Media ID {media_id}: {e}") from e

    def _delete_fts_media(self, conn: sqlite3.Connection, media_id: int):
        """
        Internal helper to delete from the media_fts table.

        Deletes the FTS entry corresponding to the given Media ID (rowid).
        Should be called within a transaction after Media soft delete or
        permanent delete. Ignores if the entry doesn't exist.

        Args:
            conn (sqlite3.Connection): The database connection (within transaction).
            media_id (int): The ID (rowid) of the Media item whose FTS entry to delete.

        Raises:
            DatabaseError: If the FTS deletion fails (excluding 'not found').
        """
        try:
            # Delete based on rowid, ignore if not found
            conn.execute("DELETE FROM media_fts WHERE rowid = ?", (media_id,))
            logging.debug(f"Deleted FTS entry for Media ID {media_id}")
        except sqlite3.Error as e:
            logging.error(f"Failed to delete from media_fts for Media ID {media_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to delete FTS for Media ID {media_id}: {e}") from e

    def _update_fts_keyword(self, conn: sqlite3.Connection, keyword_id: int, keyword: str):
        """
        Internal helper to update or insert into the keyword_fts table.

        Uses INSERT OR REPLACE based on the Keywords.id (rowid). Should be
        called within a transaction after Keywords insert/update/undelete.

        Args:
            conn (sqlite3.Connection): The database connection (within transaction).
            keyword_id (int): The ID (rowid) of the Keywords item.
            keyword (str): The keyword text.

        Raises:
            DatabaseError: If the FTS update fails.
        """
        try:
            # Use INSERT OR REPLACE
            conn.execute("INSERT OR REPLACE INTO keyword_fts (rowid, keyword) VALUES (?, ?)",
                           (keyword_id, keyword))
            logging.debug(f"Updated FTS (insert or replace) for Keyword ID {keyword_id}")
        except sqlite3.Error as e:
            logging.error(f"Failed to update keyword_fts for Keyword ID {keyword_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to update FTS for Keyword ID {keyword_id}: {e}") from e

    def _delete_fts_keyword(self, conn: sqlite3.Connection, keyword_id: int):
        """
        Internal helper to delete from the keyword_fts table.

        Deletes the FTS entry corresponding to the given Keyword ID (rowid).
        Should be called within a transaction after Keyword soft delete.
        Ignores if the entry doesn't exist.

        Args:
            conn (sqlite3.Connection): The database connection (within transaction).
            keyword_id (int): The ID (rowid) of the Keyword whose FTS entry to delete.

        Raises:
            DatabaseError: If the FTS deletion fails (excluding 'not found').
        """
        try:
            conn.execute("DELETE FROM keyword_fts WHERE rowid = ?", (keyword_id,))
            logging.debug(f"Deleted FTS entry for Keyword ID {keyword_id}")
        except sqlite3.Error as e:
            logging.error(f"Failed to delete from keyword_fts for Keyword ID {keyword_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to delete FTS for Keyword ID {keyword_id}: {e}") from e

        # In Media_DB_v2.py (within the Database class)

        # Add 'media_ids_filter' to the method signature
        # from typing import List, Tuple, Dict, Any, Optional, Union # Ensure Union is imported

    def search_media_db(self,
                        search_query: Optional[str],
                        search_fields: Optional[List[str]] = None,
                        keywords: Optional[List[str]] = None,
                        media_ids_filter: Optional[List[Union[int, str]]] = None,  # New parameter
                        page: int = 1,
                        results_per_page: int = 20,
                        include_trash: bool = False,
                        include_deleted: bool = False) -> Tuple[List[Dict[str, Any]], int]:
        """
        Searches media items based on query text, keywords, explicit media ID filter, and other filters.

        Supports FTS search on 'title' and 'content' via `media_fts` table.
        Supports basic LIKE search on 'author' and 'type'.
        Filters by a list of required keywords (all must match).
        Filters by a list of provided media IDs if `media_ids_filter` is given.
        Applies `is_trash` and `deleted` filters.
        Implements pagination.

        Args:
            self (Database): An initialized Database instance.
            search_query (Optional[str]): The text query string. Matched against
                selected `search_fields`. Can be None for keyword-only/ID-only search.
            search_fields (Optional[List[str]]): Fields to match `search_query` against.
                Valid options: 'title', 'content' (use FTS), 'author', 'type' (use LIKE).
                Defaults to ['title', 'content'] if `search_query` is provided.
                If `search_query` is None, this is ignored.
            keywords (Optional[List[str]]): A list of keywords. Media items must be
                associated with *all* provided keywords to match. Case-insensitive.
            media_ids_filter (Optional[List[Union[int, str]]]): A list of media IDs
                to restrict the search to. If provided, only media items with these
                IDs will be considered. Defaults to None.
            page (int): The page number for pagination (1-based). Defaults to 1.
            results_per_page (int): Number of results per page. Defaults to 20.
            include_trash (bool): If True, include items marked as trash. Defaults to False.
            include_deleted (bool): If True, include soft-deleted items. Defaults to False.

        Returns:
            Tuple[List[Dict[str, Any]], int]: A tuple containing:
                - results_list (List[Dict[str, Any]]): A list of dictionaries, each
                  representing a matching media item for the current page.
                - total_matches (int): The total number of items matching the criteria
                  across all pages.

        Raises:
            TypeError: If `db_instance` is not a Database object. (Not applicable here as it's a method)
            ValueError: If `page` or `results_per_page` are less than 1, or if
                        `media_ids_filter` contains invalid types.
            DatabaseError: If FTS table is missing or other database errors occur.
        """
        # Parameter validation (page, results_per_page remains the same)
        if page < 1:
            raise ValueError("Page number must be 1 or greater")
        if results_per_page < 1:
            raise ValueError("Results per page must be 1 or greater")

        if search_query and not search_fields:
            search_fields = ["title", "content"]
        elif not search_fields:
            search_fields = []  # Ensure search_fields is a list

        valid_fields = {"title", "content", "author", "type"}
        # Ensure sanitized_fields is based on the potentially defaulted search_fields
        sanitized_fields = [f for f in search_fields if f in valid_fields]

        keyword_list = [k.strip().lower() for k in keywords if k and k.strip()] if keywords else []
        search_query_stripped = search_query.strip() if search_query else None  # Use a new var

        # Log if no actual search criteria is provided (query, keywords, or ids)
        if not search_query_stripped and not keyword_list and not media_ids_filter:
            logging.debug(f"Executing browse query (no specific search criteria) on DB {self.db_path_str}.")

        offset = (page - 1) * results_per_page
        base_select = "m.id, m.uuid, m.url, m.title, m.type, m.author, m.ingestion_date, m.transcription_model, m.is_trash, m.trash_date, m.chunking_status, m.vector_processing, m.content_hash, m.last_modified, m.version, m.client_id, m.deleted"
        # Optionally add relevance score if FTS is used and available
        # if fts_search_requested: base_select += ", fts.rank AS relevance_score"

        count_select = "COUNT(DISTINCT m.id)"  # Use DISTINCT m.id for count if joins can create duplicates
        base_from = "FROM Media m"
        joins = []
        conditions = []
        params = []

        if not include_deleted:
            conditions.append("m.deleted = 0")
        if not include_trash:
            conditions.append("m.is_trash = 0")

        # New: Filter by specific media IDs
        if media_ids_filter:
            # Basic validation for media_ids_filter elements
            if not all(isinstance(mid, (int, str)) for mid in media_ids_filter):
                raise ValueError("media_ids_filter must be a list of ints or strings.")
            # Ensure the list is not empty before adding the clause
            if media_ids_filter:  # Pythonic check for non-empty list
                id_placeholders = ','.join('?' * len(media_ids_filter))
                conditions.append(f"m.id IN ({id_placeholders})")
                params.extend(media_ids_filter)

        if keyword_list:
            kw_placeholders = ','.join('?' * len(keyword_list))
            # Subquery to check if media_id is linked to ALL provided keywords
            conditions.append(f"""
                (SELECT COUNT(DISTINCT k.id)
                 FROM MediaKeywords mk
                 JOIN Keywords k ON mk.keyword_id = k.id
                 WHERE mk.media_id = m.id AND k.deleted = 0 AND k.keyword IN ({kw_placeholders})
                ) = ?
            """)
            params.extend(keyword_list)
            params.append(len(keyword_list))

        fts_search_requested = search_query_stripped and (
                    "title" in sanitized_fields or "content" in sanitized_fields)
        like_fields = {"author", "type"}
        # Determine fields for LIKE search based on sanitized_fields
        like_search_fields = [f for f in sanitized_fields if f in like_fields]

        if fts_search_requested:
            # Ensure FTS join is added only once
            if not any("media_fts fts" in j for j in joins):  # More robust check
                joins.append("JOIN media_fts fts ON fts.rowid = m.id")
            # Construct FTS match query for specified fields
            fts_match_parts = []
            if "title" in sanitized_fields:
                fts_match_parts.append("title:")
            if "content" in sanitized_fields:
                fts_match_parts.append("content:")
            # If both, SQLite FTS5 searches across all columns by default if no column is specified.
            # Or, build a more specific FTS query if needed, e.g. 'title:query OR content:query'
            # For simplicity here, if any FTS field is chosen, match against the whole fts table.
            # A more refined approach might be:
            # fts_query_string = ""
            # if "title" in sanitized_fields: fts_query_string += f"title:({search_query_stripped}) "
            # if "content" in sanitized_fields: fts_query_string += f"content:({search_query_stripped}) "
            # conditions.append(f"fts.media_fts MATCH ?"); params.append(fts_query_string.strip())
            # Simpler: FTS5 will search all its columns if not specified.
            conditions.append("fts.media_fts MATCH ?")
            params.append(search_query_stripped)

        if like_search_fields:  # If there are fields designated for LIKE search
            like_conditions_parts = []
            for field in like_search_fields:
                like_conditions_parts.append(f"m.{field} LIKE ? COLLATE NOCASE")
                params.append(f"%{search_query_stripped}%")
            if like_conditions_parts:
                conditions.append(f"({' OR '.join(like_conditions_parts)})")

        # Warning if query text is provided but no applicable search fields are selected
        if search_query_stripped and not fts_search_requested and not like_search_fields and sanitized_fields:
            logging.warning(f"Search query '{search_query_stripped}' provided but no FTS/LIKE searchable fields "
                            f"(title, content, author, type) were selected from: {search_fields}. Query text may not be effective.")

        join_clause = " ".join(list(dict.fromkeys(joins)))  # Ensure unique joins
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        try:
            # Count query
            count_sql_query = f"SELECT {count_select} {base_from} {join_clause} {where_clause}"
            logging.debug(f"Search Count SQL ({self.db_path_str}): {count_sql_query} | Params: {params}")
            # Use execute_query which uses the instance's connection handling
            count_cursor = self.execute_query(count_sql_query, tuple(params))
            total_matches = count_cursor.fetchone()[0] or 0

            results_list = []
            if total_matches > 0 and offset < total_matches:
                # Results query - order by last_modified then ID for stable pagination
                # If FTS used, could also order by fts.rank
                order_by_clause = "ORDER BY m.last_modified DESC, m.id DESC"
                # if fts_search_requested: order_by_clause = "ORDER BY fts.rank, m.last_modified DESC, m.id DESC"

                results_sql_query = f"SELECT DISTINCT {base_select} {base_from} {join_clause} {where_clause} {order_by_clause} LIMIT ? OFFSET ?"
                paginated_params = tuple(params + [results_per_page, offset])
                logging.debug(
                    f"Search Results SQL ({self.db_path_str}): {results_sql_query} | Params: {paginated_params}")
                results_cursor = self.execute_query(results_sql_query, paginated_params)
                results_list = [dict(row) for row in results_cursor.fetchall()]

            return results_list, total_matches

        except sqlite3.Error as e:
            if "no such table: media_fts" in str(e).lower():
                logging.error(f"FTS table 'media_fts' missing in database '{self.db_path_str}'. Search will fail.")
                raise DatabaseError(f"FTS table 'media_fts' not found in {self.db_path_str}.") from e
            logging.error(f"Database error during media search in '{self.db_path_str}': {e}", exc_info=True)
            raise DatabaseError(f"Failed to search media in {self.db_path_str}: {e}") from e
        except Exception as e:  # Catch any other unexpected error
            logging.error(f"Unexpected error during media search in '{self.db_path_str}': {e}", exc_info=True)
            raise DatabaseError(
                f"An unexpected error occurred during media search in {self.db_path_str}: {e}") from e

    # --- Public Mutating Methods (Modified for Python Sync/FTS Logging) ---
    def add_keyword(self, keyword: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Adds a new keyword or undeletes an existing soft-deleted one.

        Handles case-insensitivity (stores lowercase) and ensures uniqueness.
        Logs a 'create' or 'update' (for undelete) sync event.
        Updates the `keyword_fts` table accordingly.

        Args:
            keyword (str): The keyword text to add or activate.

        Returns:
            Tuple[Optional[int], Optional[str]]: A tuple containing the keyword's
                database ID and UUID. Returns (None, None) or raises error on failure.

        Raises:
            InputError: If the keyword is empty or whitespace only.
            ConflictError: If an update (undelete) fails due to version mismatch.
            DatabaseError: For other database errors during insert/update or sync logging.
        """
        if not keyword or not keyword.strip():
            raise InputError("Keyword cannot be empty.")
        keyword = keyword.strip().lower()
        current_time = self._get_current_utc_timestamp_str()  # Get current time once
        client_id = self.client_id

        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id, uuid, deleted, version FROM Keywords WHERE keyword = ?', (keyword,))
                existing = cursor.fetchone()

                if existing:
                    kw_id, kw_uuid, is_deleted, current_version = existing['id'], existing['uuid'], existing['deleted'], existing['version']
                    if is_deleted:
                        new_version = current_version + 1
                        logger.info(f"Undeleting keyword '{keyword}' (ID: {kw_id}). New ver: {new_version}")
                        # Pass current_time for last_modified
                        cursor.execute("UPDATE Keywords SET deleted=0, last_modified=?, version=?, client_id=? WHERE id=? AND version=?",
                                       (current_time, new_version, client_id, kw_id, current_version))
                        if cursor.rowcount == 0:
                            raise ConflictError("Keywords", kw_id)

                        # Fetch data for payload AFTER update to get correct last_modified
                        cursor.execute("SELECT * FROM Keywords WHERE id=?", (kw_id,))
                        payload_data = dict(cursor.fetchone())
                        self._log_sync_event(conn, 'Keywords', kw_uuid, 'update', new_version, payload_data)
                        self._update_fts_keyword(conn, kw_id, keyword)
                        return kw_id, kw_uuid
                    else:
                        logger.debug(f"Keyword '{keyword}' already active.")
                        return kw_id, kw_uuid
                else:
                    new_uuid = self._generate_uuid()
                    new_version = 1
                    logger.info(f"Adding new keyword '{keyword}' UUID {new_uuid}")
                    # Pass current_time for last_modified
                    cursor.execute("INSERT INTO Keywords (keyword, uuid, last_modified, version, client_id, deleted) VALUES (?, ?, ?, ?, ?, 0)",
                                   (keyword, new_uuid, current_time, new_version, client_id))
                    kw_id = cursor.lastrowid
                    if not kw_id:
                        raise DatabaseError("Failed to get last row ID for new keyword.")

                    # Fetch data for payload AFTER insert to get correct last_modified
                    cursor.execute("SELECT * FROM Keywords WHERE id=?", (kw_id,))
                    payload_data = dict(cursor.fetchone())
                    self._log_sync_event(conn, 'Keywords', new_uuid, 'create', new_version, payload_data)
                    self._update_fts_keyword(conn, kw_id, keyword)
                    return kw_id, new_uuid
        except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error in add_keyword for '{keyword}': {e}", exc_info=isinstance(e, (DatabaseError, sqlite3.Error)))
            if isinstance(e, (InputError, ConflictError, DatabaseError)):
                raise e
            else:
                raise DatabaseError(f"Failed to add/update keyword: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in add_keyword for '{keyword}': {e}", exc_info=True)
            raise DatabaseError(f"Unexpected error adding/updating keyword: {e}") from e

    def fetch_media_for_keywords(self, keywords: List[str], include_trash: bool = False) -> Dict[
        str, List[Dict[str, Any]]]:
        """
        Fetches all active, non-deleted media items associated with each of the
        provided active keywords.

        The media items themselves are filtered based on their `deleted` status (must be 0)
        and optionally their `is_trash` status. Keywords are always filtered for `deleted = 0`.

        Args:
            keywords (List[str]): A list of keyword strings to search for.
                                  The search is case-insensitive.
            include_trash (bool): If True, include media items marked as trash
                                  (Media.is_trash = 1). Defaults to False.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary where keys are the
                unique, cleaned (lowercase, stripped) input keywords that were found
                active in the database and have associated media matching the criteria.
                Values are lists of dictionaries, each representing a media item.
                Media items include: 'id', 'uuid', 'title', 'type', 'url',
                'content_hash', 'last_modified', 'ingestion_date', 'author'.
                Returns an empty dictionary if no keywords are provided or if
                no matching media is found for any of the provided keywords under
                the given criteria.

        Raises:
            TypeError: If `keywords` is not a list.
            DatabaseError: For database query errors.
        """
        if not isinstance(keywords, list):
            raise TypeError("Input 'keywords' must be a list of strings.")

        if not keywords:
            logger.debug("fetch_media_for_keywords called with an empty list of keywords.")
            return {}

        # Normalize keywords: lowercase, strip whitespace, filter out empty strings, and ensure uniqueness.
        # Sort for consistent query parameter order (good for logging/debugging, though IN order doesn't matter for SQL).
        potential_keywords = [k.strip().lower() for k in keywords if k and k.strip()]
        if not potential_keywords:
            logger.debug("fetch_media_for_keywords: no valid keywords after initial cleaning and stripping.")
            return {}

        unique_clean_keywords = sorted(list(set(potential_keywords)))

        if not unique_clean_keywords:  # Should be redundant due to above check, but defensive.
            logger.debug("fetch_media_for_keywords: no unique valid keywords remain.")
            return {}

        placeholders = ','.join('?' * len(unique_clean_keywords))

        media_conditions = ["m.deleted = 0"]  # Always exclude soft-deleted media
        if not include_trash:
            media_conditions.append("m.is_trash = 0")
        media_where_clause = " AND ".join(media_conditions)

        # Select desired fields from Media table
        media_fields = "m.id AS media_id, m.uuid AS media_uuid, m.title AS media_title, " \
                       "m.type AS media_type, m.url AS media_url, m.content_hash AS media_content_hash, " \
                       "m.last_modified AS media_last_modified, m.ingestion_date AS media_ingestion_date, " \
                       "m.author AS media_author"

        query = f"""
            SELECT
                k.keyword AS keyword_text,
                {media_fields}
            FROM Keywords k
            JOIN MediaKeywords mk ON k.id = mk.keyword_id
            JOIN Media m ON mk.media_id = m.id
            WHERE k.keyword IN ({placeholders})
              AND k.deleted = 0               -- Only active keywords
              AND {media_where_clause}        -- Media status filters
            ORDER BY k.keyword, m.last_modified DESC, m.id DESC
        """

        params = tuple(unique_clean_keywords)

        logger.debug(
            f"Executing fetch_media_for_keywords query for keywords: {unique_clean_keywords}, include_trash: {include_trash}")

        # Initialize results dictionary. Keys will be the cleaned, unique input keywords.
        # If a keyword has no matching media, its list will remain empty.
        results_by_keyword: Dict[str, List[Dict[str, Any]]] = {kw: [] for kw in unique_clean_keywords}

        try:
            cursor = self.execute_query(query, params)
            rows = cursor.fetchall()

            for row in rows:
                # keyword_text from DB is the canonical version (e.g. "recipe")
                # It will be one of the unique_clean_keywords due to the IN clause and
                # case-insensitive matching + storage of keywords as lowercase.
                db_keyword = row['keyword_text']

                media_item = {
                    'id': row['media_id'],
                    'uuid': row['media_uuid'],
                    'title': row['media_title'],
                    'type': row['media_type'],
                    'url': row['media_url'],
                    'content_hash': row['media_content_hash'],
                    'last_modified': row['media_last_modified'],
                    'ingestion_date': row['media_ingestion_date'],
                    'author': row['media_author']
                }

                # db_keyword should be a key in results_by_keyword because unique_clean_keywords
                # are already lowercase, and keywords in DB are stored lowercase.
                if db_keyword in results_by_keyword:
                    results_by_keyword[db_keyword].append(media_item)
                else:
                    # This case should not be reached if keyword handling (storage, cleaning, query) is consistent.
                    # Logging an error if it occurs.
                    logger.error(f"Data consistency alert in fetch_media_for_keywords: "
                                 f"Keyword '{db_keyword}' from DB results was not in the "
                                 f"expected set of unique_clean_keywords: {unique_clean_keywords}. "
                                 f"This may indicate a mismatch in case handling or normalization.")
                    # Fallback: add it as a new key to avoid losing data, though it signals an issue.
                    results_by_keyword[db_keyword] = [media_item]

            # Filter out keywords that ended up with no media, if preferred.
            # The current approach returns all queried (unique, clean) keywords as keys.
            # To only return keywords that *had* media:
            # final_results = {k: v for k, v in results_by_keyword.items() if v}

            num_keywords_with_media = len([k for k, v in results_by_keyword.items() if v])
            total_media_items_found = sum(len(v) for v in results_by_keyword.values())
            logger.info(f"Fetched media for keywords. Queried unique keywords: {len(unique_clean_keywords)}. "
                        f"Keywords with media found: {num_keywords_with_media}. "
                        f"Total media items grouped: {total_media_items_found}")

            return results_by_keyword

        except sqlite3.Error as e:
            logger.error(f"SQLite error fetching media for keywords from DB {self.db_path_str}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to fetch media for keywords due to SQLite error: {e}") from e
        except DatabaseError:  # Re-raise DatabaseError if execute_query raised it
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching media for keywords from DB {self.db_path_str}: {e}", exc_info=True)
            raise DatabaseError(f"An unexpected error occurred while fetching media for keywords: {e}") from e

    def get_sync_log_entries(self, since_change_id: int = 0, limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieves sync log entries newer than a given change_id.

        Useful for fetching changes to be processed by a synchronization mechanism.

        Args:
            since_change_id (int): The minimum change_id (exclusive) to fetch.
                                   Defaults to 0 to fetch all entries.
            limit (Optional[int]): The maximum number of entries to return.
                                   Defaults to None (no limit).

        Returns:
            List[Dict]: A list of sync log entries, each as a dictionary.
                        The 'payload' field is JSON-decoded if present.
                        Returns an empty list if no new entries are found.

        Raises:
            DatabaseError: If fetching log entries fails.
        """
        query = "SELECT change_id, entity, entity_uuid, operation, timestamp, client_id, version, payload FROM sync_log WHERE change_id > ? ORDER BY change_id ASC"
        params = [since_change_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        try:
            cursor = self.execute_query(query, tuple(params))
            results = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                if row_dict.get('payload'):
                    try:
                        row_dict['payload'] = json.loads(row_dict['payload'])
                    except json.JSONDecodeError:
                        logging.warning(f"Failed to decode JSON payload for sync log change_id {row_dict.get('change_id')}")
                        row_dict['payload'] = None
                results.append(row_dict)
            return results
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching sync log entries from DB '{self.db_path_str}': {e}")
            raise DatabaseError("Failed to fetch sync log entries") from e

    def delete_sync_log_entries(self, change_ids: List[int]) -> int:
        """
        Deletes specific sync log entries by their change_id.

        Typically used after successfully processing sync events.

        Args:
            change_ids (List[int]): A list of `change_id` values to delete.

        Returns:
            int: The number of sync log entries actually deleted.

        Raises:
            ValueError: If `change_ids` is not a list of integers.
            DatabaseError: If the deletion fails.
        """
        if not change_ids:
            return 0
        if not all(isinstance(cid, int) for cid in change_ids):
            raise ValueError("change_ids must be a list of integers.")
        placeholders = ','.join('?' * len(change_ids))
        query = f"DELETE FROM sync_log WHERE change_id IN ({placeholders})"
        try:
            with self.transaction():
                cursor = self.execute_query(query, tuple(change_ids), commit=False)
                deleted_count = cursor.rowcount
                logger.info(f"Deleted {deleted_count} sync log entries from DB '{self.db_path_str}'.")
                return deleted_count
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error deleting sync log entries from DB '{self.db_path_str}': {e}")
            raise DatabaseError("Failed to delete sync log entries") from e
        except Exception as e:
            logger.error(f"Unexpected error deleting sync log entries from DB '{self.db_path_str}': {e}")
            raise DatabaseError(f"Unexpected error deleting sync log entries: {e}") from e

    def delete_sync_log_entries_before(self, change_id_threshold: int) -> int:
        """
        Deletes sync log entries with change_id less than or equal to a threshold.

        Useful for purging old, processed sync history.

        Args:
            change_id_threshold (int): The maximum `change_id` (inclusive) to delete.
                                       Must be a non-negative integer.

        Returns:
            int: The number of sync log entries actually deleted.

        Raises:
            ValueError: If `change_id_threshold` is not a non-negative integer.
            DatabaseError: If the deletion fails.
        """
        if not isinstance(change_id_threshold, int) or change_id_threshold < 0:
            raise ValueError("change_id_threshold must be a non-negative integer.")
        query = "DELETE FROM sync_log WHERE change_id <= ?"
        try:
            with self.transaction():
                cursor = self.execute_query(query, (change_id_threshold,), commit=False)
                deleted_count = cursor.rowcount
                logger.info(f"Deleted {deleted_count} sync log entries before or at ID {change_id_threshold} from DB '{self.db_path_str}'.")
                return deleted_count
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error deleting sync log entries before {change_id_threshold} from DB '{self.db_path_str}': {e}")
            raise DatabaseError("Failed to delete sync log entries before threshold") from e
        except Exception as e:
            logger.error(f"Unexpected error deleting sync log entries before {change_id_threshold} from DB '{self.db_path_str}': {e}")
            raise DatabaseError(f"Unexpected error deleting sync log entries before threshold: {e}") from e

    def soft_delete_media(self, media_id: int, cascade: bool = True) -> bool:
        """
        Soft deletes a Media item by setting its 'deleted' flag to 1.

        Increments the version number, updates `last_modified`, logs a 'delete'
        sync event for the Media item, and removes its FTS entry.
        If `cascade` is True (default), it also performs the following within
        the same transaction:
        - Deletes corresponding MediaKeywords links and logs 'unlink' events.
        - Soft deletes associated child records (Transcripts, MediaChunks,
          UnvectorizedMediaChunks, DocumentVersions), logging 'delete' events
          for each child.

        Args:
            media_id (int): The ID of the Media item to soft delete.
            cascade (bool): Whether to also soft delete related child records
                            and unlink keywords. Defaults to True.

        Returns:
            bool: True if the media item was successfully soft-deleted,
                  False if the item was not found or already deleted.

        Raises:
            ConflictError: If the media item's version has changed since being read.
            DatabaseError: For other database errors during the operation or sync logging.
        """
        current_time = self._get_current_utc_timestamp_str()  # Get time
        client_id = self.client_id
        logger.info(f"Attempting soft delete for Media ID: {media_id} [Client: {client_id}, Cascade: {cascade}]")

        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT uuid, version FROM Media WHERE id = ? AND deleted = 0", (media_id,))
                media_info = cursor.fetchone()
                if not media_info:
                    logger.warning(f"Cannot soft delete: Media ID {media_id} not found or already deleted.")
                    return False
                media_uuid, current_media_version = media_info['uuid'], media_info['version']
                new_media_version = current_media_version + 1

                # Update Media: Pass current_time for last_modified
                cursor.execute("UPDATE Media SET deleted = 1, last_modified = ?, version = ?, client_id = ? WHERE id = ? AND version = ?",
                               (current_time, new_media_version, client_id, media_id, current_media_version))
                if cursor.rowcount == 0:
                    raise ConflictError(entity="Media", identifier=media_id)

                # Payload reflects the state *after* the update
                delete_payload = {'uuid': media_uuid, 'last_modified': current_time, 'version': new_media_version, 'client_id': client_id, 'deleted': 1}
                self._log_sync_event(conn, 'Media', media_uuid, 'delete', new_media_version, delete_payload)
                self._delete_fts_media(conn, media_id)

                if cascade:
                    logger.info(f"Performing explicit cascade delete for Media ID: {media_id}")
                    # Unlinking MediaKeywords - logic remains the same
                    cursor.execute("SELECT mk.id, k.uuid AS keyword_uuid FROM MediaKeywords mk JOIN Keywords k ON mk.keyword_id = k.id WHERE mk.media_id = ? AND k.deleted = 0", (media_id,))
                    keywords_to_unlink = cursor.fetchall()
                    if keywords_to_unlink:
                        keyword_ids = [k['id'] for k in keywords_to_unlink]
                        placeholders = ','.join('?' * len(keyword_ids))
                        cursor.execute(f"DELETE FROM MediaKeywords WHERE media_id = ? AND keyword_id IN ({placeholders})", (media_id, *keyword_ids))
                        unlink_version = 1
                        for kw_link in keywords_to_unlink:
                            link_uuid = f"{media_uuid}_{kw_link['keyword_uuid']}"
                            unlink_payload = {'media_uuid': media_uuid, 'keyword_uuid': kw_link['keyword_uuid']}
                            self._log_sync_event(conn, 'MediaKeywords', link_uuid, 'unlink', unlink_version, unlink_payload)

                    # Soft deleting child tables
                    child_tables = [("Transcripts", "media_id", "uuid"), ("MediaChunks", "media_id", "uuid"),
                                    ("UnvectorizedMediaChunks", "media_id", "uuid"), ("DocumentVersions", "media_id", "uuid")]
                    for table, fk_col, uuid_col in child_tables:
                        cursor.execute(f"SELECT id, {uuid_col} AS uuid, version FROM {table} WHERE {fk_col} = ? AND deleted = 0", (media_id,))
                        children = cursor.fetchall()
                        if not children:
                            continue
                        # Pass current_time for last_modified in child update
                        update_sql = f"UPDATE {table} SET deleted = 1, last_modified = ?, version = ?, client_id = ? WHERE id = ? AND version = ? AND deleted = 0"
                        processed_children_count = 0
                        for child in children:
                            child_id, child_uuid, child_current_version = child['id'], child['uuid'], child['version']
                            child_new_version = child_current_version + 1
                            # Pass current_time here
                            params = (current_time, child_new_version, client_id, child_id, child_current_version)
                            child_cursor = conn.cursor()
                            child_cursor.execute(update_sql, params)
                            if child_cursor.rowcount == 1:
                                processed_children_count += 1
                                # Ensure payload includes correct last_modified and deleted status
                                child_delete_payload = {'uuid': child_uuid, 'media_uuid': media_uuid, 'last_modified': current_time, 'version': child_new_version, 'client_id': client_id, 'deleted': 1}
                                self._log_sync_event(conn, table, child_uuid, 'delete', child_new_version, child_delete_payload)
                            else:
                                logger.warning(f"Conflict/error cascade deleting {table} ID {child_id}")
                        logger.debug(f"Cascade deleted {processed_children_count}/{len(children)} records in {table}.")

            logger.info(f"Soft delete successful for Media ID: {media_id}.")
            return True
        except (ConflictError, DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error soft deleting media ID {media_id}: {e}", exc_info=True)
            if isinstance(e, (ConflictError, DatabaseError)):
                raise e
            else:
                raise DatabaseError(f"Failed to soft delete media: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error soft deleting media ID {media_id}: {e}", exc_info=True)
            raise DatabaseError(f"Unexpected error during soft delete: {e}") from e

    def add_media_with_keywords(self,
                                *,
                                url: Optional[str] = None,
                                title: Optional[str],
                                media_type: Optional[str],
                                content: Optional[str],
                                keywords: Optional[List[str]] = None,
                                prompt: Optional[str] = None,
                                analysis_content: Optional[str] = None,
                                transcription_model: Optional[str] = None,
                                author: Optional[str] = None,
                                ingestion_date: Optional[str] = None,
                                overwrite: bool = False,
                                chunk_options: Optional[Dict] = None,
                                chunks: Optional[List[Dict[str, Any]]] = None
        ) -> Tuple[Optional[int], Optional[str], str]:
        """
        Adds a new media item or updates an existing one based on URL or content hash.

        Handles creation or update of the Media record, generates a content hash,
        associates keywords (adding them if necessary), creates an initial
        DocumentVersion, logs appropriate sync events ('create' or 'update' for
        Media, plus events from keyword and document version handling), and
        updates the `media_fts` table.

        If `chunks` are provided, they are saved as UnvectorizedMediaChunks.
        If updating an existing media item and new chunks are provided, old chunks
        for that media item are hard-deleted before inserting the new ones.

        If an existing item is found (by URL or content hash) and `overwrite` is False,
        the operation is skipped. If `overwrite` is True, the existing item is updated.

        Args:
            url (Optional[str]): The URL of the media (unique). Generated if not provided.
            title (Optional[str]): Title of the media. Defaults to 'Untitled'.
            media_type (Optional[str]): Type of media (e.g., 'article', 'video'). Defaults to 'unknown'.
            content (Optional[str]): The main text content. Required.
            keywords (Optional[List[str]]): List of keyword strings to associate.
            prompt (Optional[str]): Optional prompt associated with this version.
            analysis_content (Optional[str]): Optional analysis/summary content.
            transcription_model (Optional[str]): Model used for transcription, if applicable.
            author (Optional[str]): Author of the media.
            ingestion_date (Optional[str]): ISO 8601 formatted UTC timestamp for ingestion.
                                           Defaults to current time if None.
            overwrite (bool): If True, update the media item if it already exists.
                              Defaults to False (skip if exists).
            chunk_options (Optional[Dict]): Placeholder for chunking parameters.
            chunks (Optional[List[Dict[str, Any]]]): A list of dictionaries, where each dictionary
                                                     represents a chunk of the media content. Expected keys
                                                     per dictionary: 'text' (str, required), and optional
                                                     'start_char' (int), 'end_char' (int),
                                                     'chunk_type' (str), 'metadata' (dict).

        Returns:
            Tuple[Optional[int], Optional[str], str]: A tuple containing:
                - media_id (Optional[int]): The ID of the added/updated media item.
                - media_uuid (Optional[str]): The UUID of the added/updated media item.
                - message (str): A status message indicating the action taken
                                 ("added", "updated", "already_exists_skipped").

        Raises:
            InputError: If `content` is None or required chunk data is malformed.
            ConflictError: If an update fails due to a version mismatch.
            DatabaseError: For underlying database issues or errors during sync/FTS logging.
        """
        if content is None:
            raise InputError("Content cannot be None.")
        title = title or 'Untitled'
        media_type = media_type or 'unknown'
        keywords_list = [k.strip().lower() for k in keywords if k and k.strip()] if keywords else []

        # Get current time and client ID
        current_time = self._get_current_utc_timestamp_str()
        client_id = self.client_id

        # Handle ingestion_date: Use provided, else generate now. Use full timestamp.
        ingestion_date_str = ingestion_date or current_time

        content_hash = hashlib.sha256(content.encode()).hexdigest()
        if not url:
            url = f"local://{media_type}/{content_hash}"

        logging.info(f"Processing add/update for: URL='{url}', Title='{title}', Client='{client_id}'")

        media_id: Optional[int] = None
        media_uuid: Optional[str] = None
        action = "skipped"  # Default action

        # Determine initial chunking_status based on presence of 'chunks' argument
        # This will be used when preparing insert_data or update_data for the Media table
        initial_chunking_status = "pending"
        if chunks is not None:  # chunks is an empty list or a list with items
            initial_chunking_status = "processing"

        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id, uuid, version FROM Media WHERE (url = ? OR content_hash = ?) AND deleted = 0 LIMIT 1', (url, content_hash))
                existing_media = cursor.fetchone()
                media_id, media_uuid, action = None, None, "skipped"

                if existing_media:
                    media_id, media_uuid, current_version = existing_media['id'], existing_media['uuid'], existing_media['version']
                    if overwrite:
                        action = "updated"
                        new_version = current_version + 1
                        logger.info(f"Updating existing media ID {media_id} (UUID: {media_uuid}) to version {new_version}.")
                        update_data = {  # Prepare dict for easier payload generation
                            'url': url,
                            'title': title,
                            'type': media_type,
                            'content': content,
                            'author': author,
                            'ingestion_date': ingestion_date_str,
                            'transcription_model': transcription_model,
                            'content_hash': content_hash,
                            'is_trash': 0,
                            'trash_date': None,  # Ensure trash_date is None here
                            'chunking_status': "pending",
                            'vector_processing': 0,
                            'last_modified': current_time,  # Set last_modified
                            'version': new_version,
                            'client_id': client_id,
                            'deleted': 0,
                            'uuid': media_uuid
                        }
                        cursor.execute(
                            """UPDATE Media SET url=?, title=?, type=?, content=?, author=?, ingestion_date=?,
                               transcription_model=?, content_hash=?, is_trash=?, trash_date=?, chunking_status=?,
                               vector_processing=?, last_modified=?, version=?, client_id=?, deleted=?
                               WHERE id=? AND version=?""",
                            (update_data['url'], update_data['title'], update_data['type'], update_data['content'],
                             update_data['author'], update_data['ingestion_date'], update_data['transcription_model'],
                             update_data['content_hash'], update_data['is_trash'], update_data['trash_date'],  # Pass None for trash_date
                             update_data['chunking_status'], update_data['vector_processing'],
                             update_data['last_modified'],  # Pass current_time
                             update_data['version'], update_data['client_id'], update_data['deleted'],
                             media_id, current_version)
                        )
                        if cursor.rowcount == 0:
                            raise ConflictError("Media", media_id)

                        # Use the update_data dict directly for the payload
                        self._log_sync_event(conn, 'Media', media_uuid, 'update', new_version, update_data)
                        self._update_fts_media(conn, media_id, update_data['title'], update_data['content'])

                        # Consolidate keyword and version creation here for "updated"
                        self.update_keywords_for_media(media_id, keywords_list)  # Manages its own logs
                        # Create a new document version representing this update
                        self.create_document_version(
                            media_id=media_id,
                            content=content,  # Use the new content for the version
                            prompt=prompt,
                            analysis_content=analysis_content
                        )
                    else:
                        action = "already_exists_skipped"
                else:  # Not existing_media
                    action = "added"
                    media_uuid = self._generate_uuid()
                    new_version = 1
                    logger.info(f"Inserting new media '{title}' with UUID {media_uuid}.")
                    insert_data = {  # Prepare dict for easier payload generation
                         'url': url,
                        'title': title,
                        'type': media_type,
                        'content': content,
                        'author': author,
                        'ingestion_date': ingestion_date_str,  # Use generated/passed ingestion_date
                        'transcription_model': transcription_model,
                        'content_hash': content_hash,
                        'is_trash': 0,
                        'trash_date': None,  # trash_date is NULL on creation
                        'chunking_status': initial_chunking_status,
                        'vector_processing': 0,
                        'uuid': media_uuid,
                        'last_modified': current_time,  # Set last_modified
                        'version': new_version,
                        'client_id': client_id,
                        'deleted': 0
                    }
                    cursor.execute(
                        """INSERT INTO Media (url, title, type, content, author, ingestion_date, transcription_model,
                           content_hash, is_trash, trash_date, chunking_status, vector_processing, uuid,
                           last_modified, version, client_id, deleted)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (insert_data['url'], insert_data['title'], insert_data['type'], insert_data['content'],
                         insert_data['author'], insert_data['ingestion_date'],  # Pass ingestion_date_str
                         insert_data['transcription_model'], insert_data['content_hash'], insert_data['is_trash'],
                         insert_data['trash_date'],  # Pass None for trash_date
                         insert_data['chunking_status'], insert_data['vector_processing'], insert_data['uuid'],
                         insert_data['last_modified'],  # Pass current_time
                         insert_data['version'], insert_data['client_id'], insert_data['deleted'])
                    )
                    media_id = cursor.lastrowid
                    if not media_id:
                        raise DatabaseError("Failed to get last row ID for new media.")

                    # Use the insert_data dict directly for the payload
                    self._log_sync_event(conn, 'Media', media_uuid, 'create', new_version, insert_data)
                    self._update_fts_media(conn, media_id, insert_data['title'], insert_data['content'])

                    # Consolidate keyword and version creation here for "added"
                    self.update_keywords_for_media(media_id, keywords_list)
                    self.create_document_version(
                        media_id=media_id,
                        content=content,
                        prompt=prompt,
                        analysis_content=analysis_content
                    )

                    # --- Handle Unvectorized Chunks ---
                    if chunks is not None:  # chunks argument was provided (could be empty or list of dicts)
                        if action == "updated":
                            # Hard delete old chunks for this media_id if updating
                            logging.debug(f"Hard deleting existing UnvectorizedMediaChunks for updated media_id {media_id}.")
                            conn.execute("DELETE FROM UnvectorizedMediaChunks WHERE media_id = ?", (media_id,))

                        num_chunks_saved = 0
                        if chunks:  # If chunks list is not empty
                            chunk_creation_time = self._get_current_utc_timestamp_str()
                            for i, chunk_data in enumerate(chunks):
                                if not isinstance(chunk_data, dict) or 'text' not in chunk_data or chunk_data['text'] is None:
                                    logging.warning(
                                        f"Skipping invalid chunk data at index {i} for media_id {media_id} "
                                        f"(missing 'text' or not a dict): {str(chunk_data)[:100]}"
                                    )
                                    continue

                                chunk_item_uuid = self._generate_uuid()
                                metadata_payload = chunk_data.get('metadata')

                                chunk_record_tuple = (
                                    media_id,
                                    chunk_data['text'],
                                    i,  # chunk_index
                                    chunk_data.get('start_char'),
                                    chunk_data.get('end_char'),
                                    chunk_data.get('chunk_type'),
                                    chunk_creation_time,  # creation_date
                                    chunk_creation_time,  # last_modified_orig
                                    False,  # is_processed
                                    json.dumps(metadata_payload) if isinstance(metadata_payload, dict) else None,  # metadata
                                    chunk_item_uuid,
                                    chunk_creation_time,  # last_modified
                                    1,  # version
                                    self.client_id,
                                    0,  # deleted
                                    None,  # prev_version
                                    None  # merge_parent_uuid
                                )
                                try:
                                    conn.execute("""
                                        INSERT INTO UnvectorizedMediaChunks (
                                            media_id, chunk_text, chunk_index, start_char, end_char, chunk_type,
                                            creation_date, last_modified_orig, is_processed, metadata, uuid,
                                            last_modified, version, client_id, deleted, prev_version, merge_parent_uuid
                                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """, chunk_record_tuple)

                                    # Construct payload for sync log (SQLite row is not directly usable as dict here)
                                    sync_payload_chunk = {
                                        'media_id': media_id, 'chunk_text': chunk_data['text'], 'chunk_index': i,
                                        'start_char': chunk_data.get('start_char'), 'end_char': chunk_data.get('end_char'),
                                        'chunk_type': chunk_data.get('chunk_type'), 'creation_date': chunk_creation_time,
                                        'last_modified_orig': chunk_creation_time, 'is_processed': False,
                                        'metadata': metadata_payload,  # Store original dict, not JSON string, in sync log if possible
                                        'uuid': chunk_item_uuid, 'last_modified': chunk_creation_time,
                                        'version': 1, 'client_id': self.client_id, 'deleted': 0
                                    }
                                    self._log_sync_event(conn, 'UnvectorizedMediaChunks', chunk_item_uuid, 'create', 1, sync_payload_chunk)
                                    num_chunks_saved += 1
                                except sqlite3.IntegrityError as e:
                                    logging.error(f"Integrity error saving chunk {i} (type: {chunk_data.get('chunk_type')}) for media_id {media_id}: {e}. Data: {chunk_data['text'][:50]}...")
                                    raise DatabaseError(f"Failed to save chunk {i} due to integrity constraint: {e}") from e
                            logging.info(f"Saved {num_chunks_saved} unvectorized chunks for media_id {media_id}.")

                        # Update Media chunking_status to 'completed' as chunk processing is done (even if 0 chunks were provided)
                        conn.execute("UPDATE Media SET chunking_status = 'completed' WHERE id = ?", (media_id,))
                        logging.debug(f"Updated Media chunking_status to 'completed' for media_id {media_id} after chunk processing.")

                    # Original chunk_options placeholder log
                    if chunk_options:
                        logging.info(f"Chunking logic placeholder (using chunk_options) for media {media_id}")

            # Determine message based on action (outside transaction)
            if action == "updated":
                message = f"Media '{title}' updated."
            elif action == "added":
                message = f"Media '{title}' added."
            else:
                message = f"Media '{title}' exists, not overwritten."
            return media_id, media_uuid, message
        except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error processing media (URL: {url}): {e}", exc_info=isinstance(e, (DatabaseError, sqlite3.Error)))
            if isinstance(e, (InputError, ConflictError, DatabaseError)):
                raise e
            else:
                raise DatabaseError(f"Failed to process media: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error processing media (URL: {url}): {e}", exc_info=True)
            raise DatabaseError(f"Unexpected error processing media: {e}") from e

    def create_document_version(self, media_id: int, content: str, prompt: Optional[str] = None, analysis_content: Optional[str] = None) -> Dict[str, Any]:
        """
        Creates a new version entry in the DocumentVersions table.

        Assigns the next available `version_number` for the given `media_id`.
        Generates a UUID for the version, sets timestamps, and logs a 'create'
        sync event for the `DocumentVersions` entity.

        This method assumes it's called within an existing transaction context
        (e.g., initiated by `add_media_with_keywords` or `rollback_to_version`).

        Args:
            media_id (int): The ID of the parent Media item.
            content (str): The content for this document version. Required.
            prompt (Optional[str]): The prompt associated with this version, if any.
            analysis_content (Optional[str]): Analysis or summary for this version.

        Returns:
            Dict[str, Any]: A dictionary containing the new version's 'id', 'uuid',
                            'media_id', and 'version_number'.

        Raises:
            InputError: If `content` is None or the parent `media_id` does not exist
                        or is deleted.
            DatabaseError: For database errors during insert or sync logging.
        """
        if content is None:
            raise InputError("Content is required for a document version.")
        current_time = self._get_current_utc_timestamp_str()   # Get time
        client_id = self.client_id
        new_uuid = self._generate_uuid()
        new_version = 1  # Sync version for the DocumentVersion entity itself

        # Assumes called within an existing transaction (e.g., from add_media_with_keywords)
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT uuid FROM Media WHERE id = ? AND deleted = 0", (media_id,))  # Ensure parent active
            media_info = cursor.fetchone()
            if not media_info:
                raise InputError(f"Parent Media ID {media_id} not found or deleted.")
            media_uuid = media_info['uuid']

            cursor.execute('SELECT COALESCE(MAX(version_number), 0) + 1 FROM DocumentVersions WHERE media_id = ?', (media_id,))
            local_version_number = cursor.fetchone()[0]
            logger.debug(f"Creating document version {local_version_number} for media ID {media_id}, UUID {new_uuid}")

            insert_data = {  # Prepare dict for easier payload generation
                'media_id': media_id, 'version_number': local_version_number, 'content': content, 'prompt': prompt,
                'analysis_content': analysis_content,
                'created_at': current_time,  # Set created_at
                'uuid': new_uuid,
                'last_modified': current_time,  # Set last_modified
                'version': new_version, 'client_id': client_id, 'deleted': 0,
                'media_uuid': media_uuid  # Add parent uuid for context in payload
            }
            cursor.execute(
                """INSERT INTO DocumentVersions (media_id, version_number, content, prompt, analysis_content, created_at,
                   uuid, last_modified, version, client_id, deleted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (insert_data['media_id'], insert_data['version_number'], insert_data['content'], insert_data['prompt'],
                 insert_data['analysis_content'],
                 insert_data['created_at'],  # Pass created_at
                 insert_data['uuid'],
                 insert_data['last_modified'],  # Pass last_modified
                 insert_data['version'], insert_data['client_id'], insert_data['deleted'])
            )
            version_id = cursor.lastrowid
            if not version_id:
                raise DatabaseError("Failed to get last row ID for new document version.")

            self._log_sync_event(conn, 'DocumentVersions', new_uuid, 'create', new_version, insert_data)
            return {'id': version_id, 'uuid': new_uuid, 'media_id': media_id, 'version_number': local_version_number}
        except (InputError, DatabaseError, sqlite3.Error) as e:
            if "foreign key constraint failed" in str(e).lower():
                logger.error(f"Failed create document version: Media ID {media_id} not found.", exc_info=False)
                raise InputError(f"Cannot create document version: Media ID {media_id} not found.") from e
            logger.error(f"DB error creating document version media {media_id}: {e}", exc_info=True)
            if isinstance(e, (InputError, DatabaseError)):
                raise e
            else:
                raise DatabaseError(f"Failed create document version: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error creating document version media {media_id}: {e}", exc_info=True)
            raise DatabaseError(f"Unexpected error creating document version: {e}") from e

    def update_keywords_for_media(self, media_id: int, keywords: List[str]):
        """
        Synchronizes the keywords linked to a specific media item.

        Compares the provided list of keywords with the currently linked active
        keywords. Adds missing links (calling `add_keyword` if needed for the
        keyword itself) and removes outdated links. Logs 'link' and 'unlink'
        sync events for changes in the `MediaKeywords` junction table.

        Assumes it's called within an existing transaction context.

        Args:
            media_id (int): The ID of the Media item whose keywords to update.
            keywords (List[str]): The desired list of keyword strings for the media item.
                                  Empty list removes all keywords.

        Returns:
            bool: True if the operation completed (even if no changes were needed).

        Raises:
            InputError: If the parent `media_id` does not exist or is deleted.
            DatabaseError: For underlying database errors, issues adding keywords,
                           or sync logging failures.
            ConflictError: If `add_keyword` encounters a conflict during undelete.
        """
        valid_keywords = sorted(list(set([k.strip().lower() for k in keywords if k and k.strip()])))
        # Assumes called within an existing transaction
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT uuid FROM Media WHERE id = ? AND deleted = 0", (media_id,))  # Check parent active
            media_info = cursor.fetchone()
            if not media_info:
                raise InputError(f"Cannot update keywords: Media ID {media_id} not found or deleted.")
            media_uuid = media_info['uuid']

            cursor.execute("SELECT mk.keyword_id, k.uuid AS keyword_uuid FROM MediaKeywords mk JOIN Keywords k ON k.id = mk.keyword_id WHERE mk.media_id = ? AND k.deleted = 0", (media_id,))
            current_links = {row['keyword_id']: row['keyword_uuid'] for row in cursor.fetchall()}
            current_keyword_ids = set(current_links.keys())

            target_keyword_data = {}
            if valid_keywords:
                for kw_text in valid_keywords:
                    kw_id, kw_uuid = self.add_keyword(kw_text)  # Handles create/undelete/logging/FTS for Keywords
                    if kw_id and kw_uuid:
                        target_keyword_data[kw_id] = kw_uuid
                    else:
                        raise DatabaseError(f"Failed get/add keyword '{kw_text}'")

            target_keyword_ids = set(target_keyword_data.keys())
            ids_to_add = target_keyword_ids - current_keyword_ids
            ids_to_remove = current_keyword_ids - target_keyword_ids
            link_sync_version = 1

            if ids_to_remove:
                remove_placeholders = ','.join('?' * len(ids_to_remove))
                cursor.execute(f"DELETE FROM MediaKeywords WHERE media_id = ? AND keyword_id IN ({remove_placeholders})", (media_id, *list(ids_to_remove)))
                for removed_id in ids_to_remove:
                    keyword_uuid = current_links.get(removed_id)
                    if keyword_uuid:
                        link_uuid = f"{media_uuid}_{keyword_uuid}"
                        payload = {'media_uuid': media_uuid, 'keyword_uuid': keyword_uuid}
                        self._log_sync_event(conn, 'MediaKeywords', link_uuid, 'unlink', link_sync_version, payload)

            if ids_to_add:
                insert_params = [(media_id, kid) for kid in ids_to_add]
                cursor.executemany("INSERT OR IGNORE INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)", insert_params)
                # Log links - Note: IGNORE means we might log links that weren't actually inserted if race condition. Robust check is complex.
                for added_id in ids_to_add:
                    keyword_uuid = target_keyword_data.get(added_id)
                    if keyword_uuid:
                        link_uuid = f"{media_uuid}_{keyword_uuid}"
                        payload = {'media_uuid': media_uuid, 'keyword_uuid': keyword_uuid}
                        self._log_sync_event(conn, 'MediaKeywords', link_uuid, 'link', link_sync_version, payload)

            if ids_to_add or ids_to_remove:
                logger.debug(f"Keywords updated media {media_id}. Added: {len(ids_to_add)}, Removed: {len(ids_to_remove)}.")
            else:
                logger.debug(f"No keyword changes media {media_id}.")
            return True
        except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error updating keywords media {media_id}: {e}", exc_info=True)
            if isinstance(e, (InputError, ConflictError, DatabaseError)):
                raise e
            else:
                raise DatabaseError(f"Keyword update failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected keywords error media {media_id}: {e}", exc_info=True)
            raise DatabaseError(f"Unexpected keyword update error: {e}") from e

    def soft_delete_keyword(self, keyword: str) -> bool:
        """
        Soft deletes a keyword by setting its 'deleted' flag to 1.

        Handles case-insensitivity. Increments the version number, updates
        `last_modified`, logs a 'delete' sync event for the Keyword, and removes
        its FTS entry. It also removes all links between this keyword and any
        media items in the `MediaKeywords` table, logging 'unlink' events for each.

        Args:
            keyword (str): The keyword text to soft delete (case-insensitive).

        Returns:
            bool: True if the keyword was successfully soft-deleted,
                  False if the keyword was not found or already deleted.

        Raises:
            InputError: If the keyword string is empty or whitespace only.
            ConflictError: If the keyword's version has changed since being read.
            DatabaseError: For other database errors or sync logging failures.
        """
        if not keyword or not keyword.strip():
            raise InputError("Keyword cannot be empty.")
        keyword = keyword.strip().lower()
        current_time = self._get_current_utc_timestamp_str()  # Get time
        client_id = self.client_id

        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id, uuid, version FROM Keywords WHERE keyword = ? AND deleted = 0', (keyword,))
                keyword_info = cursor.fetchone()
                if not keyword_info:
                    logger.warning(f"Keyword '{keyword}' not found/deleted.")
                    return False
                keyword_id, keyword_uuid, current_version = keyword_info['id'], keyword_info['uuid'], keyword_info['version']
                new_version = current_version + 1

                logger.info(f"Soft deleting keyword '{keyword}' (ID: {keyword_id}). New ver: {new_version}")
                # Pass current_time for last_modified
                cursor.execute("UPDATE Keywords SET deleted=1, last_modified=?, version=?, client_id=? WHERE id=? AND version=?",
                               (current_time, new_version, client_id, keyword_id, current_version))
                if cursor.rowcount == 0:
                    raise ConflictError("Keywords", keyword_id)

                # Payload reflects the state *after* the update
                delete_payload = {'uuid': keyword_uuid, 'last_modified': current_time, 'version': new_version, 'client_id': client_id, 'deleted': 1}
                self._log_sync_event(conn, 'Keywords', keyword_uuid, 'delete', new_version, delete_payload)
                self._delete_fts_keyword(conn, keyword_id)

                # Unlinking logic remains the same
                cursor.execute("SELECT mk.media_id, m.uuid AS media_uuid FROM MediaKeywords mk JOIN Media m ON mk.media_id = m.id WHERE mk.keyword_id = ? AND m.deleted = 0", (keyword_id,))
                media_to_unlink = cursor.fetchall()
                if media_to_unlink:
                    media_ids = [m['media_id'] for m in media_to_unlink]
                    placeholders = ','.join('?' * len(media_ids))
                    cursor.execute(f"DELETE FROM MediaKeywords WHERE keyword_id = ? AND media_id IN ({placeholders})", (keyword_id, *media_ids))
                    unlink_version = 1
                    deleted_link_count = cursor.rowcount  # Get actual count of deleted links
                    for media_link in media_to_unlink:
                        link_uuid = f"{media_link['media_uuid']}_{keyword_uuid}"
                        unlink_payload = {'media_uuid': media_link['media_uuid'], 'keyword_uuid': keyword_uuid}
                        self._log_sync_event(conn, 'MediaKeywords', link_uuid, 'unlink', unlink_version, unlink_payload)
                    logger.info(f"Unlinked keyword '{keyword}' from {deleted_link_count} items.")
            return True
        except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error soft delete keyword '{keyword}': {e}", exc_info=True)
            if isinstance(e, (InputError, ConflictError, DatabaseError)):
                raise e
            else:
                raise DatabaseError(f"Failed soft delete keyword: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected soft delete keyword error '{keyword}': {e}", exc_info=True)
            raise DatabaseError(f"Unexpected soft delete keyword error: {e}") from e

    def soft_delete_document_version(self, version_uuid: str) -> bool:
        """
        Soft deletes a specific DocumentVersion by its UUID.

        Prevents deletion if it's the last remaining active version for the media item.
        Increments the sync version, updates `last_modified`, and logs a 'delete'
        sync event for the `DocumentVersions` entity.

        Args:
            version_uuid (str): The UUID of the DocumentVersion to soft delete.

        Returns:
            bool: True if successfully soft-deleted, False if not found, already
                  deleted, or if it's the last active version.

        Raises:
            InputError: If `version_uuid` is empty or None.
            ConflictError: If the version's sync version has changed concurrently.
            DatabaseError: For other database errors or sync logging failures.
        """
        if not version_uuid:
            raise InputError("Version UUID required.")
        current_time = self._get_current_utc_timestamp_str()  # Get time
        client_id = self.client_id
        logger.debug(f"Attempting soft delete DocVersion UUID: {version_uuid}")
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT dv.id, dv.media_id, dv.version, m.uuid as media_uuid FROM DocumentVersions dv JOIN Media m ON dv.media_id = m.id WHERE dv.uuid = ? AND dv.deleted = 0", (version_uuid,))
                version_info = cursor.fetchone()
                if not version_info:
                    logger.warning(f"DocVersion UUID {version_uuid} not found/deleted.")
                    return False
                version_id, media_id, current_sync_version, media_uuid = version_info['id'], version_info['media_id'], version_info['version'], version_info['media_uuid']
                new_sync_version = current_sync_version + 1

                cursor.execute("SELECT COUNT(*) FROM DocumentVersions WHERE media_id = ? AND deleted = 0", (media_id,))
                active_count = cursor.fetchone()[0]
                if active_count <= 1:
                    logger.warning(f"Cannot delete DocVersion UUID {version_uuid} - last active.")
                    return False

                # Pass current_time for last_modified
                cursor.execute("UPDATE DocumentVersions SET deleted=1, last_modified=?, version=?, client_id=? WHERE id=? AND version=?",
                               (current_time, new_sync_version, client_id, version_id, current_sync_version))
                if cursor.rowcount == 0:
                    raise ConflictError("DocumentVersions", version_id)

                # Payload reflects the state *after* the update
                delete_payload = {'uuid': version_uuid, 'media_uuid': media_uuid, 'last_modified': current_time, 'version': new_sync_version, 'client_id': client_id, 'deleted': 1}
                self._log_sync_event(conn, 'DocumentVersions', version_uuid, 'delete', new_sync_version, delete_payload)
                logger.info(f"Soft deleted DocVersion UUID {version_uuid}. New ver: {new_sync_version}")
                return True
        except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error soft delete DocVersion UUID {version_uuid}: {e}", exc_info=True)
            if isinstance(e, (InputError, ConflictError, DatabaseError)):
                raise e
            else:
                raise DatabaseError(f"Failed soft delete doc version: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected soft delete DocVersion error UUID {version_uuid}: {e}", exc_info=True)
            raise DatabaseError(f"Unexpected version soft delete error: {e}") from e

    def mark_as_trash(self, media_id: int) -> bool:
        """
        Marks a media item as 'trash' (is_trash=1) without soft deleting it.

        Sets the `trash_date`, updates `last_modified`, increments the sync version,
        and logs an 'update' sync event for the Media item. Does not affect FTS.

        Args:
            media_id (int): The ID of the Media item to move to trash.

        Returns:
            bool: True if successfully marked as trash, False if not found, deleted,
                  or already in trash.

        Raises:
            ConflictError: If the media item's version has changed concurrently.
            DatabaseError: For other database errors or sync logging failures.
        """
        current_time = self._get_current_utc_timestamp_str()  # Get time
        client_id = self.client_id
        logger.debug(f"Marking media {media_id} as trash.")
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT uuid, version, is_trash FROM Media WHERE id = ? AND deleted = 0", (media_id,))
                media_info = cursor.fetchone()
                if not media_info:
                    logger.warning(f"Cannot trash: Media {media_id} not found/deleted.")
                    return False
                if media_info['is_trash']:
                    logger.warning(f"Media {media_id} already in trash.")
                    return False  # No change needed
                media_uuid, current_version = media_info['uuid'], media_info['version']
                new_version = current_version + 1

                # Pass current_time for both trash_date and last_modified
                cursor.execute("UPDATE Media SET is_trash=1, trash_date=?, last_modified=?, version=?, client_id=? WHERE id=? AND version=?",
                               (current_time, current_time, new_version, client_id, media_id, current_version))
                if cursor.rowcount == 0:
                    raise ConflictError("Media", media_id)

                cursor.execute("SELECT * FROM Media WHERE id = ?", (media_id,))  # Fetch updated state for payload
                sync_payload = dict(cursor.fetchone())
                self._log_sync_event(conn, 'Media', media_uuid, 'update', new_version, sync_payload)
                # No FTS change needed for trash status itself
                logger.info(f"Media {media_id} marked as trash. New ver: {new_version}")
                return True
        except (ConflictError, DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error marking media {media_id} as trash: {e}", exc_info=True)
            if isinstance(e, (ConflictError, DatabaseError)):
                raise e
            else:
                raise DatabaseError(f"Failed mark as trash: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error marking media {media_id} trash: {e}", exc_info=True)
            raise DatabaseError(f"Unexpected mark trash error: {e}") from e

    def restore_from_trash(self, media_id: int) -> bool:
        """
        Restores a media item from 'trash' (sets is_trash=0, trash_date=NULL).

        Updates `last_modified`, increments the sync version, and logs an 'update'
        sync event for the Media item. Does not affect FTS.

        Args:
            media_id (int): The ID of the Media item to restore.

        Returns:
            bool: True if successfully restored, False if not found, deleted,
                  or not currently in trash.

        Raises:
            ConflictError: If the media item's version has changed concurrently.
            DatabaseError: For other database errors or sync logging failures.
        """
        current_time = self._get_current_utc_timestamp_str()  # Get time
        client_id = self.client_id
        logger.debug(f"Restoring media {media_id} from trash.")
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT uuid, version, is_trash FROM Media WHERE id = ? AND deleted = 0", (media_id,))
                media_info = cursor.fetchone()
                if not media_info:
                    logger.warning(f"Cannot restore: Media {media_id} not found/deleted.")
                    return False
                if not media_info['is_trash']:
                    logger.warning(f"Cannot restore: Media {media_id} not in trash.")
                    return False  # No change needed
                media_uuid, current_version = media_info['uuid'], media_info['version']
                new_version = current_version + 1

                # Pass current_time for last_modified, set trash_date to NULL
                cursor.execute("UPDATE Media SET is_trash=0, trash_date=NULL, last_modified=?, version=?, client_id=? WHERE id=? AND version=?",
                               (current_time, new_version, client_id, media_id, current_version))
                if cursor.rowcount == 0:
                    raise ConflictError("Media", media_id)

                cursor.execute("SELECT * FROM Media WHERE id = ?", (media_id,))  # Fetch updated state for payload
                sync_payload = dict(cursor.fetchone())
                self._log_sync_event(conn, 'Media', media_uuid, 'update', new_version, sync_payload)
                # No FTS change needed
                logger.info(f"Media {media_id} restored from trash. New ver: {new_version}")
                return True
        except (ConflictError, DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error restoring media {media_id} trash: {e}", exc_info=True)
            if isinstance(e, (ConflictError, DatabaseError)):
                raise e
            else:
                raise DatabaseError(f"Failed restore trash: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error restoring media {media_id} trash: {e}", exc_info=True)
            raise DatabaseError(f"Unexpected restore trash error: {e}") from e

    def rollback_to_version(self, media_id: int, target_version_number: int) -> Dict[str, Any]:
        """
        Rolls back the main Media content to a previous DocumentVersion state.

        This involves:
        1. Fetching the content from the specified target `DocumentVersion`.
        2. Creating a *new* `DocumentVersion` entry containing this rolled-back content.
        3. Updating the main `Media` record's content, content_hash, `last_modified`,
           and incrementing its sync version.
        4. Logging 'create' for the new DocumentVersion and 'update' for the Media item.
        5. Updating the `media_fts` table with the rolled-back content.

        Prevents rolling back to the absolute latest version number.

        Args:
            media_id (int): The ID of the Media item to roll back.
            target_version_number (int): The `version_number` of the DocumentVersion
                                         to roll back to. Must be a positive integer.

        Returns:
            Dict[str, Any]: A dictionary containing either:
                - {'success': message, 'new_document_version_number': int,
                   'new_document_version_uuid': str, 'new_media_version': int}
                - {'error': message} if the rollback failed (e.g., version not found,
                  media not found, target is latest version).

        Raises:
            ValueError: If `target_version_number` is invalid.
            InputError: If underlying `create_document_version` fails input checks.
            ConflictError: If the Media item's version changed concurrently during update.
            DatabaseError: For other database errors or sync/FTS logging issues.
        """
        if not isinstance(target_version_number, int) or target_version_number < 1:
            raise ValueError("Target version invalid.")
        client_id = self.client_id
        current_time = self._get_current_utc_timestamp_str()  # Get time
        logger.debug(f"Rolling back media {media_id} to doc version {target_version_number}.")
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                # Get current media info
                cursor.execute("SELECT uuid, version, title FROM Media WHERE id = ? AND deleted = 0", (media_id,))
                media_info = cursor.fetchone()
                if not media_info:
                    return {'error': f'Media {media_id} not found or deleted.'}
                media_uuid, current_media_version, current_title = media_info['uuid'], media_info['version'], media_info['title']
                new_media_version = current_media_version + 1

                # Get target document version data (using standalone function)
                target_version_data = get_document_version(self, media_id, target_version_number, True)
                if target_version_data is None:
                    return {'error': f'Rollback target version {target_version_number} not found or inactive.'}

                # Prevent rolling back to the absolute latest version number
                cursor.execute("SELECT MAX(version_number) FROM DocumentVersions WHERE media_id=? AND deleted=0", (media_id,))
                latest_vn_res = cursor.fetchone()
                if latest_vn_res and target_version_number == latest_vn_res[0]:
                    return {'error': 'Cannot rollback to the current latest version number.'}

                target_content = target_version_data.get('content')
                target_prompt = target_version_data.get('prompt')
                target_analysis = target_version_data.get('analysis_content')
                if target_content is None:
                    return {'error': f'Version {target_version_number} has no content.'}

                # 1. Create new doc version representing the rollback state (handles its own logging & timestamps)
                new_doc_version_info = self.create_document_version(media_id=media_id, content=target_content, prompt=target_prompt, analysis_content=target_analysis)
                new_doc_version_number = new_doc_version_info.get('version_number')
                new_doc_version_uuid = new_doc_version_info.get('uuid')

                # 2. Update the Media table with the rolled-back content and new hash/timestamp
                new_content_hash = hashlib.sha256(target_content.encode()).hexdigest()
                # Pass current_time for last_modified
                cursor.execute(
                    """UPDATE Media SET content=?, content_hash=?, last_modified=?, version=?, client_id=?,
                       chunking_status="pending", vector_processing=0 WHERE id=? AND version=?""",
                    (target_content, new_content_hash, current_time, new_media_version, client_id, media_id, current_media_version))
                if cursor.rowcount == 0:
                    raise ConflictError("Media", media_id)

                # 3. Log the Media update sync event
                cursor.execute("SELECT * FROM Media WHERE id = ?", (media_id,))  # Fetch updated state for payload
                updated_media_data = dict(cursor.fetchone())
                # Add context about the rollback to the payload (optional but helpful)
                updated_media_data['rolled_back_to_doc_ver_uuid'] = new_doc_version_uuid
                updated_media_data['rolled_back_to_doc_ver_num'] = new_doc_version_number
                self._log_sync_event(conn, 'Media', media_uuid, 'update', new_media_version, updated_media_data)

                # 4. Update FTS for the Media item
                self._update_fts_media(conn, media_id, current_title, target_content)  # Use original title, new content

            logger.info(f"Rolled back media {media_id} to state of doc ver {target_version_number}. New DocVer: {new_doc_version_number}, New MediaVer: {new_media_version}")
            return {'success': f'Rolled back to version {target_version_number}. State saved as new version {new_doc_version_number}.',
                    'new_document_version_number': new_doc_version_number,
                    'new_document_version_uuid': new_doc_version_uuid,
                    'new_media_version': new_media_version}
        except (InputError, ValueError, ConflictError, DatabaseError, sqlite3.Error, TypeError) as e:
            logger.error(f"Rollback error media {media_id}: {e}", exc_info=True)
            if isinstance(e, (InputError, ValueError, ConflictError, DatabaseError, TypeError)):
                raise e
            else:
                raise DatabaseError(f"DB error during rollback: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected rollback error media {media_id}: {e}", exc_info=True)
            raise DatabaseError(f"Unexpected rollback error: {e}") from e

    def get_all_document_versions(
        self,  # Add self as the first parameter
        media_id: int,
        include_content: bool = False,
        include_deleted: bool = False,
        limit: Optional[int] = None,
        offset: Optional[int] = 0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves all document versions for an active media item with pagination.

        Filters results to only include versions where the parent Media item is active
        (`Media.deleted = 0`). By default, only active document versions
        (`DocumentVersions.deleted = 0`) are returned, unless `include_deleted` is True.

        Includes standard V2 sync metadata columns (uuid, version, last_modified, client_id).

        Args:
            media_id (int): The ID of the parent Media item.
            include_content (bool): Whether to include the 'content' field. Defaults to False.
            include_deleted (bool): If True, include versions marked as soft-deleted
                                    (`deleted = 1`). Defaults to False.
            limit (Optional[int]): Maximum number of versions to return. None for no limit.
                                   Defaults to None.
            offset (Optional[int]): Number of versions to skip (for pagination).
                                    Defaults to 0.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a document
                                  version matching the criteria. Returns an empty list
                                  if none found.

        Raises:
            TypeError: If input arguments have wrong types.
            ValueError: If limit or offset are invalid.
            DatabaseError: For database query errors.
        """
        # --- Input Validation ---
        # No need to validate self
        if not isinstance(media_id, int):
            raise TypeError("media_id must be an integer.")
        if not isinstance(include_content, bool):
            raise TypeError("include_content must be a boolean.")
        if not isinstance(include_deleted, bool):
            raise TypeError("include_deleted must be a boolean.")
        if limit is not None and (not isinstance(limit, int) or limit < 1):
            raise ValueError("Limit must be a positive integer.")
        if offset is not None and (not isinstance(offset, int) or offset < 0):
            raise ValueError("Offset must be a non-negative integer.")

        # --- Logging ---
        # Use self.db_path_str for logging context
        log_msg = (f"Getting {'all' if include_deleted else 'active'} versions for media_id={media_id} "
                   f"(Limit={limit}, Offset={offset}, Content={include_content}) "
                   f"from DB: {self.db_path_str}")  # Use self.db_path_str
        logger.debug(log_msg)

        # --- Query Construction ---
        try:
            # Select all relevant columns from DocumentVersions
            select_cols_list = [
                "dv.id", "dv.uuid", "dv.media_id", "dv.version_number", "dv.created_at",
                "dv.prompt", "dv.analysis_content", "dv.last_modified", "dv.version",
                "dv.client_id", "dv.deleted"
            ]
            if include_content:
                select_cols_list.append("dv.content")
            select_clause = ", ".join(select_cols_list)

            params = [media_id]
            where_conditions = ["dv.media_id = ?", "m.deleted = 0"]  # Always filter by active parent

            if not include_deleted:
                where_conditions.append("dv.deleted = 0")

            where_clause = " AND ".join(where_conditions)

            limit_offset_clause = ""
            if limit is not None:
                limit_offset_clause += " LIMIT ?"
                params.append(limit)
                if offset is not None and offset > 0:
                    limit_offset_clause += " OFFSET ?"
                    params.append(offset)

            final_query = f"""
                SELECT {select_clause}
                FROM DocumentVersions dv
                JOIN Media m ON dv.media_id = m.id
                WHERE {where_clause}
                ORDER BY dv.version_number DESC
                {limit_offset_clause}
            """

            # --- Execution ---
            logging.debug(f"Executing get_all_document_versions query | Params: {params}")
            # Use self.execute_query
            cursor = self.execute_query(final_query, tuple(params))
            results_raw = cursor.fetchall()

            versions_list = [dict(row) for row in results_raw]

            logging.debug(f"Found {len(versions_list)} versions for media_id={media_id}")
            return versions_list

        # --- Error Handling ---
        except sqlite3.Error as e:
            # Use self.db_path_str
            logging.error(f"SQLite error retrieving versions for media_id {media_id} from {self.db_path_str}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to retrieve document versions: {e}") from e
        except Exception as e:
            # Use self.db_path_str
            logging.error(f"Unexpected error retrieving versions for media_id {media_id} from {self.db_path_str}: {e}", exc_info=True)
            raise DatabaseError(f"An unexpected error occurred: {e}") from e

    def process_unvectorized_chunks(self, media_id: int, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        Adds a batch of unvectorized chunk records to the database.

        Inserts records into the `UnvectorizedMediaChunks` table in batches.
        Generates a UUID, sets timestamps, and logs a 'create' sync event
        for each chunk added. Assumes parent media item exists and is active.

        Args:
            media_id (int): The ID of the parent Media item for these chunks.
            chunks (List[Dict[str, Any]]): A list of dictionaries, each representing
                a chunk. Expected keys include 'chunk_text' (or 'text'),
                'chunk_index'. Optional keys: 'start_char', 'end_char',
                'chunk_type', 'creation_date', 'last_modified_orig',
                'is_processed', 'metadata'.
            batch_size (int): Number of chunks to insert per database transaction batch.
                              Defaults to 100.

        Raises:
            InputError: If the parent `media_id` does not exist or is deleted, or if
                        essential chunk data ('chunk_text', 'chunk_index') is missing.
            DatabaseError: For database errors during insertion or sync logging.
            TypeError: If 'metadata' is provided but cannot be JSON serialized.
        """
        if not chunks:
            logger.warning(f"process_unvectorized_chunks empty list for media {media_id}.")
            return
        client_id = self.client_id
        start_time = time.time()
        total_chunks = len(chunks)
        processed_count = 0
        logger.info(f"Processing {total_chunks} unvectorized chunks for media {media_id}.")
        try:
            # Use standalone check function (assumed to exist and work)
            if not check_media_exists(self, media_id=media_id):
                raise InputError(f"Cannot add chunks: Parent Media {media_id} not found or deleted.")
            conn_check = self.get_connection()
            cursor_check = conn_check.execute("SELECT uuid FROM Media WHERE id = ?", (media_id,))
            media_info = cursor_check.fetchone()
            if not media_info:
                raise InputError(f"Cannot add chunks: Parent Media ID {media_id} UUID not found.")
            media_uuid = media_info['uuid']

            with self.transaction() as conn:
                for i in range(0, total_chunks, batch_size):
                    batch = chunks[i:i + batch_size]
                    chunk_params = []
                    log_events_data = []
                    current_time = self._get_current_utc_timestamp_str()  # Get time for the batch
                    for chunk_dict in batch:
                        chunk_uuid = self._generate_uuid()
                        chunk_text = chunk_dict.get('chunk_text', chunk_dict.get('text'))
                        chunk_index = chunk_dict.get('chunk_index')
                        if chunk_text is None or chunk_index is None:
                            logger.warning(f"Skipping chunk missing text/index media {media_id}")
                            continue

                        new_sync_version = 1
                        insert_data = {  # Match table schema
                            'media_id': media_id, 'chunk_text': chunk_text, 'chunk_index': chunk_index,
                            'start_char': chunk_dict.get('start_char'), 'end_char': chunk_dict.get('end_char'),
                            'chunk_type': chunk_dict.get('chunk_type'),
                            # Use current_time if not provided in chunk_dict
                            'creation_date': chunk_dict.get('creation_date') or current_time,
                            'last_modified_orig': chunk_dict.get('last_modified_orig') or current_time,
                            'is_processed': chunk_dict.get('is_processed', False),
                            # Ensure metadata is JSON string
                            'metadata': json.dumps(chunk_dict.get('metadata')) if chunk_dict.get('metadata') else None,
                            'uuid': chunk_uuid,
                            'last_modified': current_time,  # Set sync last_modified
                            'version': new_sync_version, 'client_id': client_id, 'deleted': 0,
                            'media_uuid': media_uuid  # for payload context
                        }
                        params = (  # Order must match SQL query
                            insert_data['media_id'], insert_data['chunk_text'], insert_data['chunk_index'],
                            insert_data['start_char'], insert_data['end_char'], insert_data['chunk_type'],
                            insert_data['creation_date'],  # Pass creation_date
                            insert_data['last_modified_orig'],  # Pass last_modified_orig
                            insert_data['is_processed'], insert_data['metadata'], insert_data['uuid'],
                            insert_data['last_modified'],  # Pass sync last_modified
                            insert_data['version'], insert_data['client_id'], insert_data['deleted']
                        )
                        chunk_params.append(params)
                        # Pass the full insert_data as payload
                        log_events_data.append((chunk_uuid, new_sync_version, insert_data))

                    if not chunk_params:
                        continue
                    # Ensure columns match params order
                    sql = """INSERT INTO UnvectorizedMediaChunks (media_id, chunk_text, chunk_index, start_char, end_char, chunk_type,
                               creation_date, last_modified_orig, is_processed, metadata, uuid,
                               last_modified, version, client_id, deleted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
                    cursor = conn.cursor()
                    cursor.executemany(sql, chunk_params)
                    actual_inserted = len(chunk_params)  # executemany doesn't give reliable rowcount

                    for chunk_uuid_log, version_log, payload_log in log_events_data:
                        self._log_sync_event(conn, 'UnvectorizedMediaChunks', chunk_uuid_log, 'create', version_log, payload_log)
                    processed_count += actual_inserted
                    logger.debug(f"Processed batch {i//batch_size+1}: Inserted {actual_inserted} chunks for media {media_id}.")
            duration = time.time() - start_time
            logger.info(f"Finished processing {processed_count} unvectorized chunks media {media_id}. Duration: {duration:.4f}s")
        except (InputError, DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error processing unvectorized chunks media {media_id}: {e}", exc_info=True)
            if isinstance(e, (InputError, DatabaseError)):
                raise e
            else:
                raise DatabaseError(f"Failed process chunks: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected chunk processing error media {media_id}: {e}", exc_info=True)
            raise DatabaseError(f"Unexpected chunk error: {e}") from e

    # --- Read Methods (Ensure they filter by deleted=0) ---
    def fetch_all_keywords(self) -> List[str]:
        """
        Fetches all *active* (non-deleted) keywords from the database.

        Returns:
            List[str]: A sorted list of active keyword strings (lowercase).
                       Returns an empty list if no active keywords are found.

        Raises:
            DatabaseError: If the database query fails.
        """
        try:
            cursor = self.execute_query('SELECT keyword FROM Keywords WHERE deleted = 0 ORDER BY keyword COLLATE NOCASE')
            return [row['keyword'] for row in cursor.fetchall()]
        except DatabaseError as e:
            logger.error(f"Error fetching keywords: {e}")
            raise

    def get_paginated_media_list(self, page: int = 1, results_per_page: int = 10) -> Tuple[
        List[Dict[str, Any]], int, int, int]:
        """
        Fetches a paginated list of active media items (id, title, type, uuid)
        for the media listing endpoint.

        Filters for items where deleted = 0 and is_trash = 0.
        Returns data suitable for constructing MediaListItem objects.

        Args:
            page (int): The page number (1-based).
            results_per_page (int): Number of items per page.

        Returns:
            Tuple[List[Dict[str, Any]], int, int, int]:
                - results (List[Dict]): List of dictionaries for the current page.
                                       Each dict contains 'id', 'title', 'type', 'uuid'.
                - total_pages (int): Total number of pages.
                - current_page (int): The requested page number.
                - total_items (int): Total number of active items.

        Raises:
            ValueError: If page or results_per_page are invalid.
            DatabaseError: If a database query fails.
        """
        if page < 1:
            raise ValueError("Page number must be 1 or greater.")
        if results_per_page < 1:
            raise ValueError("Results per page must be 1 or greater.")

        logging.debug(
            f"DB: Fetching paginated media list: page={page}, rpp={results_per_page} from {self.db_path_str}"
        )
        offset = (page - 1) * results_per_page

        try:
            with self.transaction() as conn:  # Use a transaction for consistency if doing multiple queries
                cursor = conn.cursor()
                # Query 1: Get total count
                cursor.execute("SELECT COUNT(*) FROM Media WHERE deleted = 0 AND is_trash = 0")
                count_row = cursor.fetchone()
                total_items = count_row[0] if count_row else 0

                results_data = []
                if total_items > 0:
                    # Query 2: Get paginated items
                    query = """
                            SELECT id, title, type, uuid
                            FROM Media
                            WHERE deleted = 0 \
                              AND is_trash = 0
                            ORDER BY last_modified DESC, id DESC LIMIT ? \
                            OFFSET ? \
                            """
                    cursor.execute(query, (results_per_page, offset))
                    results_data = [dict(row) for row in cursor.fetchall()]

            total_pages = ceil(total_items / results_per_page) if results_per_page > 0 and total_items > 0 else 0
            # Ensure page is not out of bounds for returned results if total_items becomes 0 after count
            if page > total_pages and total_pages == 0:
                results_data = []  # No items if page is invalid for 0 total pages

            return results_data, total_pages, page, total_items

        except sqlite3.Error as e:
            logging.error(f"SQLite error during DB pagination: {e}", exc_info=True)
            raise DatabaseError(f"Failed DB pagination query: {e}") from e
        except Exception as e:
            logging.error(f"Unexpected error during DB pagination: {e}", exc_info=True)
            raise DatabaseError(f"Unexpected error during DB pagination: {e}") from e

    def get_media_by_id(self, media_id: int, include_deleted=False, include_trash=False) -> Optional[Dict]:
        """
        Retrieves a single media item by its primary key (ID).

        By default, only returns active (non-deleted, non-trash) items.

        Args:
            media_id (int): The integer ID of the media item.
            include_deleted (bool): If True, include items marked as soft-deleted
                                    (`deleted = 1`). Defaults to False.
            include_trash (bool): If True, include items marked as trash
                                  (`is_trash = 1`), provided they are not also
                                  soft-deleted (unless `include_deleted` is True).
                                  Defaults to False.

        Returns:
            Optional[Dict[str, Any]]: A dictionary representing the media item if found
                                      matching the criteria, otherwise None.

        Raises:
            InputError: If `media_id` is not an integer.
            DatabaseError: If a database query error occurs.
        """
        if not isinstance(media_id, int):
            raise InputError("media_id must be an integer.")

        query = "SELECT * FROM Media WHERE id = ?"
        params = [media_id]

        if not include_deleted:
            query += " AND deleted = 0"
        if not include_trash:
            query += " AND is_trash = 0"

        try:
            cursor = self.execute_query(query, tuple(params))
            result = cursor.fetchone()
            return dict(result) if result else None
        except sqlite3.Error as e:
            logger.error(f"Error fetching media by ID {media_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to fetch media by ID: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error fetching media by ID {media_id}: {e}", exc_info=True)
            raise DatabaseError(f"Unexpected error fetching media by ID: {e}") from e

    # Add similar get_media_by_uuid, get_media_by_url, get_media_by_hash, get_media_by_title
    # Ensure they include the include_deleted and include_trash filters correctly.
    def get_media_by_uuid(self, media_uuid: str, include_deleted=False, include_trash=False) -> Optional[Dict]:
        """
        Retrieves a single media item by its UUID.

        By default, only returns active (non-deleted, non-trash) items. UUIDs are unique.

        Args:
            media_uuid (str): The UUID string of the media item.
            include_deleted (bool): If True, include soft-deleted items. Defaults to False.
            include_trash (bool): If True, include trashed items. Defaults to False.

        Returns:
            Optional[Dict[str, Any]]: A dictionary representing the media item if found,
                                      otherwise None.

        Raises:
            InputError: If `media_uuid` is empty or None.
            DatabaseError: If a database query error occurs.
        """
        if not media_uuid:
            raise InputError("media_uuid cannot be empty.")
        query = "SELECT * FROM Media WHERE uuid = ?"
        params = [media_uuid]
        if not include_deleted:
            query += " AND deleted = 0"
        if not include_trash:
            query += " AND is_trash = 0"
        try:
            cursor = self.execute_query(query, tuple(params))
            result = cursor.fetchone()
            return dict(result) if result else None
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching media by UUID {media_uuid}: {e}")
            raise DatabaseError(f"Failed fetch media by UUID: {e}") from e

    def get_media_by_url(self, url: str, include_deleted=False, include_trash=False) -> Optional[Dict]:
        """
        Retrieves a single media item by its URL.

        By default, only returns active (non-deleted, non-trash) items. URLs are unique.

        Args:
            url (str): The URL string of the media item.
            include_deleted (bool): If True, include soft-deleted items. Defaults to False.
            include_trash (bool): If True, include trashed items. Defaults to False.

        Returns:
            Optional[Dict[str, Any]]: A dictionary representing the media item if found,
                                      otherwise None.

        Raises:
            InputError: If `url` is empty or None.
            DatabaseError: If a database query error occurs.
        """
        if not url:
            raise InputError("url cannot be empty or None.")

        query = "SELECT * FROM Media WHERE url = ?"
        params = [url]

        if not include_deleted:
            query += " AND deleted = 0"
        if not include_trash:
            query += " AND is_trash = 0"

        # URLs are unique, so LIMIT 1 is implicit but doesn't hurt
        query += " LIMIT 1"

        try:
            cursor = self.execute_query(query, tuple(params))
            result = cursor.fetchone()
            return dict(result) if result else None
        except sqlite3.Error as e:
            logger.error(f"Error fetching media by URL '{url}': {e}", exc_info=True)
            raise DatabaseError(f"Failed to fetch media by URL: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error fetching media by URL '{url}': {e}", exc_info=True)
            raise DatabaseError(f"Unexpected error fetching media by URL: {e}") from e

    def get_media_by_hash(self, content_hash: str, include_deleted=False, include_trash=False) -> Optional[Dict]:
        """
        Retrieves a single media item by its content hash (SHA256).

        By default, only returns active (non-deleted, non-trash) items. Hashes are unique.

        Args:
            content_hash (str): The SHA256 hash string of the media content.
            include_deleted (bool): If True, include soft-deleted items. Defaults to False.
            include_trash (bool): If True, include trashed items. Defaults to False.

        Returns:
            Optional[Dict[str, Any]]: A dictionary representing the media item if found,
                                      otherwise None.

        Raises:
            InputError: If `content_hash` is empty or None.
            DatabaseError: If a database query error occurs.
        """
        if not content_hash:
            raise InputError("content_hash cannot be empty or None.")

        query = "SELECT * FROM Media WHERE content_hash = ?"
        params = [content_hash]

        if not include_deleted:
            query += " AND deleted = 0"
        if not include_trash:
            query += " AND is_trash = 0"

        # Hashes are unique, so LIMIT 1 is implicit
        query += " LIMIT 1"

        try:
            cursor = self.execute_query(query, tuple(params))
            result = cursor.fetchone()
            return dict(result) if result else None
        except sqlite3.Error as e:
            logger.error(f"Error fetching media by hash '{content_hash[:10]}...': {e}", exc_info=True)
            raise DatabaseError(f"Failed to fetch media by hash: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error fetching media by hash '{content_hash[:10]}...': {e}", exc_info=True)
            raise DatabaseError(f"Unexpected error fetching media by hash: {e}") from e

    def get_media_by_title(self, title: str, include_deleted=False, include_trash=False) -> Optional[Dict]:
        """
        Retrieves the *first* media item matching a given title (case-sensitive).

        Note: Titles are not guaranteed to be unique. This returns the most recently
        modified match if multiple exist. By default, only returns active items.

        Args:
            title (str): The title string of the media item.
            include_deleted (bool): If True, include soft-deleted items. Defaults to False.
            include_trash (bool): If True, include trashed items. Defaults to False.

        Returns:
            Optional[Dict[str, Any]]: A dictionary representing the first matching media
                                      item (ordered by last_modified DESC), or None.

        Raises:
            InputError: If `title` is empty or None.
            DatabaseError: If a database query error occurs.
        """
        if not title:
            raise InputError("title cannot be empty or None.")

        query = "SELECT * FROM Media WHERE title = ?"
        params = [title]

        if not include_deleted:
            query += " AND deleted = 0"
        if not include_trash:
            query += " AND is_trash = 0"

        # Order by last_modified to get potentially the most relevant if duplicates exist
        query += " ORDER BY last_modified DESC LIMIT 1"

        try:
            cursor = self.execute_query(query, tuple(params))
            result = cursor.fetchone()
            return dict(result) if result else None
        except sqlite3.Error as e:
            logger.error(f"Error fetching media by title '{title}': {e}", exc_info=True)
            raise DatabaseError(f"Failed to fetch media by title: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error fetching media by title '{title}': {e}", exc_info=True)
            raise DatabaseError(f"Unexpected error fetching media by title: {e}") from e

    def get_paginated_files(self, page: int = 1, results_per_page: int = 50) -> Tuple[List[sqlite3.Row], int, int, int]:
        """
        Fetches a paginated list of active media items (id, title, type) from this database instance.

        Filters for items where `deleted = 0` and `is_trash = 0`.

        Args:
            page (int): The page number (1-based). Defaults to 1.
            results_per_page (int): The number of items per page. Defaults to 50.

        Returns:
            A tuple containing:
                - results (List[sqlite3.Row]): List of Row objects for the current page.
                                               Each row contains 'id', 'title', 'type'.
                - total_pages (int): Total number of pages for active items.
                - current_page (int): The requested page number.
                - total_items (int): The total number of active items matching the criteria.

        Raises:
            ValueError: If page or results_per_page are invalid.
            DatabaseError: If a database query fails.
        """
        # No need to check self type, it's guaranteed by method call
        if page < 1:
            raise ValueError("Page number must be 1 or greater.")
        if results_per_page < 1:
            raise ValueError("Results per page must be 1 or greater.")

        # Use self.db_path_str for logging context
        logging.debug(
            f"Fetching paginated files: page={page}, results_per_page={results_per_page} from DB: {self.db_path_str} (Active Only)")

        offset = (page - 1) * results_per_page
        total_items = 0
        results: List[sqlite3.Row] = []  # Type hint for clarity

        try:
            # Query 1: Get total count of active items
            count_query = "SELECT COUNT(*) FROM Media WHERE deleted = 0 AND is_trash = 0"
            # Use self.execute_query
            count_cursor = self.execute_query(count_query)
            count_result = count_cursor.fetchone()
            total_items = count_result[0] if count_result else 0

            # Query 2: Get paginated items if count > 0
            if total_items > 0:
                # Order by most recently modified, then ID for stable pagination
                items_query = """
                              SELECT id, title, type
                              FROM Media
                              WHERE deleted = 0
                                AND is_trash = 0
                              ORDER BY last_modified DESC, id DESC LIMIT ?
                              OFFSET ?
                              """
                # Use self.execute_query
                items_cursor = self.execute_query(items_query, (results_per_page, offset))
                # Fetchall returns a list of Row objects (if row_factory is sqlite3.Row)
                results = items_cursor.fetchall()

            # Calculate total pages
            total_pages = ceil(total_items / results_per_page) if results_per_page > 0 and total_items > 0 else 0

            return results, total_pages, page, total_items

        # Catch DatabaseError potentially raised by self.execute_query
        except DatabaseError as e:
            logging.error(f"Database error in get_paginated_files for DB {self.db_path_str}: {e}", exc_info=True)
            # Re-raise the specific error for the caller to handle
            raise
        # Catch potential underlying SQLite errors if not wrapped by execute_query
        except sqlite3.Error as e:
            logging.error(f"SQLite error during pagination query in {self.db_path_str}: {e}", exc_info=True)
            raise DatabaseError(f"Failed pagination query: {e}") from e
        # Catch unexpected errors
        except Exception as e:
            logging.error(f"Unexpected error in get_paginated_files for DB {self.db_path_str}: {e}", exc_info=True)
            # Wrap unexpected errors in DatabaseError
            raise DatabaseError(f"Unexpected error during pagination: {e}") from e

    def add_media_chunk(self, media_id: int, chunk_text: str, start_index: int, end_index: int, chunk_id: str) -> Optional[Dict]:
        """
        Adds a single chunk record to the MediaChunks table for an active media item.

        Handles transaction, generates UUID, sets sync metadata, and logs a 'create' sync event.
        This is an instance method operating on the specific user's database.

        Args:
            media_id (int): The ID of the parent Media item.
            chunk_text (str): The text content of the chunk.
            start_index (int): Starting character index within the original content.
            end_index (int): Ending character index within the original content.
            chunk_id (str): The application-specific unique ID for this chunk within the media item.

        Returns:
            Optional[Dict]: A dictionary containing the new chunk's database 'id' and 'uuid'
                            on success, otherwise None or raises an exception.

        Raises:
            InputError: If media_id doesn't exist/is inactive, or chunk_text is empty.
            DatabaseError: For database errors during insertion or sync logging, including IntegrityErrors.
        """
        if not chunk_text:
            raise InputError("Chunk text cannot be empty.")

        logger.debug(f"Adding chunk for media_id {media_id}, chunk_id {chunk_id} using client {self.client_id}")

        # Prepare sync/metadata fields using instance attributes/methods
        client_id = self.client_id
        current_time = self._get_current_utc_timestamp_str()  # Use internal helper
        new_uuid = self._generate_uuid()  # Use internal helper
        new_sync_version = 1  # Initial version for a new chunk record

        try:
            # Use instance transaction method
            with self.transaction() as conn:
                # Optional: Check if parent media exists and is active
                cursor_check = conn.cursor()
                cursor_check.execute("SELECT uuid FROM Media WHERE id = ? AND deleted = 0", (media_id,))
                media_info = cursor_check.fetchone()
                if not media_info:
                    raise InputError(f"Cannot add chunk: Parent Media ID {media_id} not found or deleted.")
                media_uuid = media_info['uuid']  # Get parent UUID for context if needed

                # Prepare data for insert statement
                insert_data = {
                    'media_id': media_id,
                    'chunk_text': chunk_text,
                    'start_index': start_index,
                    'end_index': end_index,
                    'chunk_id': chunk_id,  # Keep the original chunk_id column
                    'uuid': new_uuid,  # Add the new UUID column
                    'last_modified': current_time,
                    'version': new_sync_version,
                    'client_id': client_id,
                    'deleted': 0,
                    'media_uuid': media_uuid  # For sync payload context
                }

                # Execute INSERT
                cursor_insert = conn.cursor()
                sql = """
                      INSERT INTO MediaChunks
                      (media_id, chunk_text, start_index, end_index, chunk_id, uuid, last_modified, version, client_id, \
                       deleted)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) \
                      """
                params = (
                    insert_data['media_id'], insert_data['chunk_text'], insert_data['start_index'],
                    insert_data['end_index'], insert_data['chunk_id'], insert_data['uuid'],
                    insert_data['last_modified'], insert_data['version'], insert_data['client_id'],
                    insert_data['deleted']
                )
                cursor_insert.execute(sql, params)
                chunk_pk_id = cursor_insert.lastrowid

                if not chunk_pk_id:
                    raise DatabaseError("Failed to get last row ID for new media chunk.")

                # Log sync event using instance method (passing connection)
                self._log_sync_event(conn, 'MediaChunks', new_uuid, 'create', new_sync_version, insert_data)

                logger.info(f"Successfully added chunk ID {chunk_pk_id} (UUID: {new_uuid}) for media {media_id}.")
                return {'id': chunk_pk_id, 'uuid': new_uuid}

        except sqlite3.IntegrityError as ie:
            logger.error(f"Integrity error adding chunk for media {media_id}: {ie}", exc_info=True)
            raise DatabaseError(f"Failed to add chunk due to constraint violation: {ie}") from ie
        except (InputError, DatabaseError) as e:
            logger.error(f"Error adding chunk for media {media_id}: {e}", exc_info=True)
            raise e
        except Exception as e:
            logger.error(f"Unexpected error adding chunk for media {media_id}: {e}", exc_info=True)
            raise DatabaseError(f"An unexpected error occurred while adding media chunk: {e}") from e

    def add_media_chunks_in_batches(self, media_id: int, chunks_to_add: List[Dict[str, Any]],
                                    batch_size: int = 100) -> int:
        """
        Processes a list of chunk dictionaries and adds them to the MediaChunks table in batches.
        This method adapts the input chunk format for the internal self.batch_insert_chunks method.
        It preserves the batching and logging behavior of the original standalone process_chunks function.

        Args:
            media_id (int): ID of the media these chunks belong to.
            chunks_to_add (List[Dict[str, Any]]): List of chunk dictionaries. Each dictionary must have
                                                  'text', 'start_index', and 'end_index' keys.
                                                  Example: [{'text': 'chunk1', 'start_index': 0, 'end_index': 10}, ...]
            batch_size (int): Number of chunks to process and pass to self.batch_insert_chunks in each iteration.

        Returns:
            int: The total number of chunks successfully processed and attempted for insertion.

        Raises:
            InputError: If essential keys are missing in `chunks_to_add` items, or if `media_id` is invalid
                        (this error would be propagated from the underlying `self.batch_insert_chunks` call).
            DatabaseError: For database errors during insertion (propagated from `self.batch_insert_chunks`).
            Exception: For other unexpected errors during the process.
        """
        # These log_counter and log_histogram calls assume they are available in the global scope
        # or otherwise accessible, as per the original function's structure.
        # If not, they would need to be passed to this method or the Database instance.
        log_counter("add_media_chunks_in_batches_attempt", labels={"media_id": media_id})
        start_time = time.time()
        total_chunks_in_input = len(chunks_to_add)
        successfully_processed_count = 0

        # Parent media_id validity will be checked within self.batch_insert_chunks.
        # If media_id is invalid, self.batch_insert_chunks will raise an InputError.

        try:
            for i in range(0, total_chunks_in_input, batch_size):
                current_batch_from_input = chunks_to_add[i:i + batch_size]

                # Adapt batch to the format expected by self.batch_insert_chunks:
                # [{'text': ..., 'metadata': {'start_index': ..., 'end_index': ...}}]
                adapted_batch_for_internal_method = []
                for chunk_item in current_batch_from_input:
                    try:
                        # Ensure 'text', 'start_index', 'end_index' are present
                        # The self.batch_insert_chunks method expects 'text' (or 'chunk_text')
                        # and 'metadata' containing 'start_index' and 'end_index'.
                        text_content = chunk_item['text']
                        start_idx = chunk_item['start_index']
                        end_idx = chunk_item['end_index']

                        adapted_chunk = {
                            'text': text_content,
                            'metadata': {
                                'start_index': start_idx,
                                'end_index': end_idx
                            }
                        }
                        adapted_batch_for_internal_method.append(adapted_chunk)
                    except KeyError as e:
                        # Using global 'logging' as per the style in Database class and original function
                        logging.error(
                            f"Media ID {media_id}: Skipping chunk due to missing key {e} in input data: {chunk_item}")
                        log_counter("add_media_chunks_in_batches_item_skip_key_error",
                                    labels={"media_id": media_id, "key": str(e)})
                        continue  # Skip this malformed chunk_item

                if not adapted_batch_for_internal_method:
                    if current_batch_from_input:  # Original batch had items, but all were malformed or skipped
                        logging.warning(
                            f"Media ID {media_id}: Batch starting at index {i} resulted in no valid chunks to process.")
                    continue  # Move to the next batch

                try:
                    # self.batch_insert_chunks is an existing method in your Database class.
                    # It handles its own transaction, generates UUIDs, sets sync metadata (version, client_id, etc.),
                    # and logs sync events for each chunk.
                    # It returns the number of chunks it prepared/attempted from the adapted_batch.
                    num_inserted_this_batch = self.batch_insert_chunks(media_id, adapted_batch_for_internal_method)

                    successfully_processed_count += num_inserted_this_batch
                    logging.info(
                        f"Media ID {media_id}: Processed {successfully_processed_count}/{total_chunks_in_input} chunks so far. Current batch (size {len(adapted_batch_for_internal_method)}) resulted in {num_inserted_this_batch} items attempted.")
                    log_counter("add_media_chunks_in_batches_batch_success", labels={"media_id": media_id})

                # Catch specific errors that self.batch_insert_chunks might raise
                except InputError as e:  # e.g., if media_id is invalid, or chunk structure within adapted_batch is wrong
                    logging.error(f"Media ID {media_id}: Input error during an internal batch insertion: {e}")
                    log_counter("add_media_chunks_in_batches_batch_error",
                                labels={"media_id": media_id, "error_type": "InputError"})
                    raise  # Re-raise to halt the entire operation
                except DatabaseError as e:  # For other database-related errors from self.batch_insert_chunks
                    logging.error(f"Media ID {media_id}: Database error during an internal batch insertion: {e}")
                    log_counter("add_media_chunks_in_batches_batch_error",
                                labels={"media_id": media_id, "error_type": "DatabaseError"})
                    raise  # Re-raise
                except Exception as e:  # Catch any other unexpected errors from self.batch_insert_chunks
                    logging.error(f"Media ID {media_id}: Unexpected error during an internal batch insertion: {e}",
                                  exc_info=True)
                    log_counter("add_media_chunks_in_batches_batch_error",
                                labels={"media_id": media_id, "error_type": type(e).__name__})
                    raise  # Re-raise

            logging.info(
                f"Media ID {media_id}: Finished processing chunk list. Total chunks from input: {total_chunks_in_input}. Successfully processed and attempted for insertion: {successfully_processed_count}.")
            duration = time.time() - start_time
            log_histogram("add_media_chunks_in_batches_duration", duration, labels={"media_id": media_id})
            log_counter("add_media_chunks_in_batches_success_overall", labels={"media_id": media_id})
            return successfully_processed_count

        except Exception as e:  # Catches errors from the outer loop logic or re-raised errors from the inner try-except block
            duration = time.time() - start_time
            # Log duration even if the overall process failed
            log_histogram("add_media_chunks_in_batches_duration", duration, labels={"media_id": media_id})
            log_counter("add_media_chunks_in_batches_error_overall",
                        labels={"media_id": media_id, "error_type": type(e).__name__})
            logging.error(f"Media ID {media_id}: Error processing the list of chunks: {e}", exc_info=True)
            raise  # Re-raise the caught exception to inform the caller

    def batch_insert_chunks(self, media_id: int, chunks: List[Dict]) -> int:
        """
        Inserts a batch of chunk records into the MediaChunks table for an active media item.

        Uses executemany for efficiency within a single transaction.
        Generates UUIDs, sets sync metadata, and logs a 'create' sync event for EACH chunk.
        This is an instance method operating on the specific user's database.

        Args:
            media_id (int): The ID of the parent Media item.
            chunks (List[Dict]): A list of dictionaries, where each dictionary represents a chunk.
                                 Expected keys in each dict: 'text' (or 'chunk_text'), and
                                 'metadata' dict containing 'start_index', 'end_index'.

        Returns:
            int: The number of chunks successfully prepared for insertion.

        Raises:
            InputError: If media_id doesn't exist/is inactive, or the chunks list is empty or invalid.
            DatabaseError: For database errors during insertion or sync logging, including IntegrityErrors.
            KeyError: If expected keys ('text', 'metadata', 'start_index', 'end_index') are missing in chunk dicts.
        """
        if not chunks:
            logger.warning(f"batch_insert_chunks called with empty list for media {media_id}.")
            return 0

        logger.info(f"Batch inserting {len(chunks)} chunks for media_id {media_id} using client {self.client_id}.")

        # Use instance attributes/methods
        client_id = self.client_id
        current_time = self._get_current_utc_timestamp_str()
        params_list = []
        sync_log_data = []

        try:
            # Prepare data for all chunks first
            for i, chunk_dict in enumerate(chunks):
                try:
                    chunk_text = chunk_dict.get('text', chunk_dict['chunk_text'])
                    metadata = chunk_dict['metadata']
                    start_index = metadata['start_index']
                    end_index = metadata['end_index']
                except KeyError as ke:
                    logger.error(f"Missing expected key {ke} in chunk data at index {i} for media {media_id}")
                    raise InputError(f"Invalid chunk data structure at index {i}: Missing key {ke}") from ke

                if not chunk_text:
                    logger.warning(f"Skipping chunk at index {i} for media {media_id} due to empty text.")
                    continue

                # Generate IDs and sync fields using instance methods
                chunk_id = f"{media_id}_chunk_{i + 1}"
                new_uuid = self._generate_uuid()
                new_sync_version = 1

                params = (
                    media_id, chunk_text, start_index, end_index, chunk_id, new_uuid,
                    current_time, new_sync_version, client_id, 0  # deleted=0
                )
                params_list.append(params)

                payload = {
                    'media_id': media_id, 'chunk_text': chunk_text, 'start_index': start_index,
                    'end_index': end_index, 'chunk_id': chunk_id, 'uuid': new_uuid,
                    'last_modified': current_time, 'version': new_sync_version,
                    'client_id': client_id, 'deleted': 0
                }
                sync_log_data.append((new_uuid, new_sync_version, payload))

            if not params_list:
                logger.warning(f"No valid chunks prepared for batch insert media {media_id}.")
                return 0

            # Perform insertion and logging within a transaction using instance method
            with self.transaction() as conn:
                cursor_check = conn.cursor()
                cursor_check.execute("SELECT 1 FROM Media WHERE id = ? AND deleted = 0", (media_id,))
                if not cursor_check.fetchone():
                    raise InputError(f"Cannot batch insert chunks: Parent Media ID {media_id} not found or deleted.")

                cursor_insert = conn.cursor()
                sql = """
                      INSERT INTO MediaChunks
                      (media_id, chunk_text, start_index, end_index, chunk_id, uuid, last_modified, version, client_id, \
                       deleted)
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) \
                      """
                cursor_insert.executemany(sql, params_list)

                inserted_count = len(params_list)
                logger.debug(f"Executed batch insert for {inserted_count} chunks media {media_id}.")

                # Log sync events using instance method
                for chunk_uuid_log, version_log, payload_log in sync_log_data:
                    self._log_sync_event(conn, 'MediaChunks', chunk_uuid_log, 'create', version_log, payload_log)

            logger.info(f"Successfully batch inserted {inserted_count} chunks for media {media_id}.")
            return inserted_count

        except sqlite3.IntegrityError as ie:
            logger.error(f"Integrity error batch inserting chunks for media {media_id}: {ie}", exc_info=True)
            raise DatabaseError(f"Failed to batch insert chunks due to constraint violation: {ie}") from ie
        except (InputError, DatabaseError, KeyError) as e:
            logger.error(f"Error batch inserting chunks for media {media_id}: {e}", exc_info=True)
            raise e
        except Exception as e:
            logger.error(f"Unexpected error batch inserting chunks for media {media_id}: {e}", exc_info=True)
            raise DatabaseError(f"An unexpected error occurred during batch chunk insertion: {e}") from e

    def process_chunks(self, media_id: int, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        Process chunks in batches and insert them into the MediaChunks table.

        This method is part of the Database class and works with the V2 schema
        for MediaChunks. It generates necessary IDs (a UUID for 'chunk_id' and
        another for 'uuid') and sync metadata for each chunk.

        Args:
            media_id (int): ID of the media these chunks belong to.
            chunks (List[Dict[str, Any]]): List of chunk dictionaries. Each dictionary is
                                           expected to have 'text', 'start_index',
                                           and 'end_index' keys.
            batch_size (int): Number of chunks to process in each database transaction.

        Raises:
            InputError: If the parent media_id is not found or is deleted, or if
                        a chunk dictionary is missing required keys.
            DatabaseError: If there's an error during database operations (e.g.,
                           integrity constraints) or sync logging.
            Exception: For other unexpected errors during processing.
        """
        log_counter("process_chunks_attempt", labels={"media_id": media_id})
        start_time = time.time()
        total_chunks_to_process = len(chunks)
        successfully_inserted_chunks = 0

        # Initial check for parent media_id existence and active status.
        # This uses a direct query. An alternative is self.get_media_by_id(media_id).
        conn_for_check = self.get_connection()
        cursor_check = conn_for_check.execute("SELECT 1 FROM Media WHERE id = ? AND deleted = 0", (media_id,))
        if not cursor_check.fetchone():
            logging.error(f"Parent Media ID {media_id} not found or is deleted. Cannot process chunks.")
            log_counter("process_chunks_error", labels={"media_id": media_id, "error_type": "ParentMediaNotFound"})
            duration = time.time() - start_time  # Log duration even for this early exit
            log_histogram("process_chunks_duration", duration, labels={"media_id": media_id})
            raise InputError(f"Parent Media ID {media_id} not found or is deleted.")

        try:
            for i in range(0, total_chunks_to_process, batch_size):
                batch_of_input_chunks = chunks[i:i + batch_size]

                db_insert_params_list = []
                # Store tuples of (entity_uuid, version, payload) for logging after successful insert
                sync_log_data_for_batch = []

                current_timestamp = self._get_current_utc_timestamp_str()
                # Assumes self.client_id is available from the Database instance
                client_id = self.client_id

                for input_chunk_dict in batch_of_input_chunks:
                    try:
                        chunk_text = input_chunk_dict['text']
                        start_index = input_chunk_dict['start_index']
                        end_index = input_chunk_dict['end_index']
                    except KeyError as e:
                        logging.warning(
                            f"Skipping chunk for media_id {media_id} due to missing key '{e}': {str(input_chunk_dict)[:100]}")
                        log_counter("process_chunks_item_skipped",
                                    labels={"media_id": media_id, "reason": "missing_key", "key": str(e)})
                        continue  # Skip this malformed chunk

                    # Generate fields required by the MediaChunks schema.
                    # MediaChunks.chunk_id has a TEXT UNIQUE constraint. We generate a UUID for it.
                    generated_chunk_id_for_db = self._generate_uuid()
                    # MediaChunks.uuid also has a TEXT UNIQUE NOT NULL constraint.
                    generated_uuid_for_db = self._generate_uuid()

                    chunk_version = 1  # Initial sync version for new records
                    deleted_status = 0  # New chunks are not deleted

                    # Parameters order must match the INSERT statement columns
                    params_tuple = (
                        media_id,
                        chunk_text,
                        start_index,
                        end_index,
                        generated_chunk_id_for_db,  # value for 'chunk_id' column
                        generated_uuid_for_db,  # value for 'uuid' column
                        current_timestamp,  # last_modified
                        chunk_version,  # version
                        client_id,  # client_id
                        deleted_status  # deleted
                    )
                    db_insert_params_list.append(params_tuple)

                    # Prepare data for sync logging (payload should reflect the inserted row)
                    sync_payload = {
                        'media_id': media_id,
                        'chunk_text': chunk_text,
                        'start_index': start_index,
                        'end_index': end_index,
                        'chunk_id': generated_chunk_id_for_db,
                        'uuid': generated_uuid_for_db,
                        'last_modified': current_timestamp,
                        'version': chunk_version,
                        'client_id': client_id,
                        'deleted': deleted_status
                        # prev_version and merge_parent_uuid are typically NULL/None on creation
                    }
                    # Store data needed for _log_sync_event: (entity_uuid, version, payload_dict)
                    sync_log_data_for_batch.append((generated_uuid_for_db, chunk_version, sync_payload))

                if not db_insert_params_list:  # If all chunks in the current batch were skipped
                    logging.info(
                        f"Batch starting at index {i} for media_id {media_id} resulted in no valid chunks to insert.")
                    continue

                try:
                    # Each batch is processed in its own transaction for atomicity of that batch
                    with self.transaction() as conn:  # `conn` is yielded by the transaction context manager
                        insert_sql = """
                                     INSERT INTO MediaChunks
                                     (media_id, chunk_text, start_index, end_index, chunk_id, uuid,
                                      last_modified, version, client_id, deleted)
                                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) \
                                     """
                        # self.execute_many is called within the transaction.
                        # The default commit=False for execute_many is correct here.
                        self.execute_many(insert_sql, db_insert_params_list)

                        # If execute_many succeeded, log sync events for this batch
                        for entity_uuid, version_val, payload_dict in sync_log_data_for_batch:
                            self._log_sync_event(
                                conn=conn,  # Pass the connection from the transaction
                                entity="MediaChunks",
                                entity_uuid=entity_uuid,  # The UUID of the MediaChunk record
                                operation="create",
                                version=version_val,  # The sync version of the MediaChunk
                                payload=payload_dict
                            )

                    successfully_inserted_chunks += len(db_insert_params_list)
                    logging.info(
                        f"Successfully processed batch for media_id {media_id}. Total inserted so far: {successfully_inserted_chunks}/{total_chunks_to_process}")
                    log_counter("process_chunks_batch_success", labels={"media_id": media_id})

                except sqlite3.IntegrityError as e:
                    # This could be a FOREIGN KEY constraint failure if media_id became invalid
                    # or a UNIQUE constraint failure.
                    logging.error(f"Database integrity error inserting chunk batch for media_id {media_id}: {e}")
                    log_counter("process_chunks_batch_error",
                                labels={"media_id": media_id, "error_type": "IntegrityError"})
                    # Re-raise to stop processing further batches, as this indicates a critical issue.
                    raise DatabaseError(
                        f"Integrity error during chunk batch insertion for media_id {media_id}: {e}") from e
                except Exception as e:  # Catch other errors from DB operation or sync logging
                    logging.error(f"Error processing chunk batch for media_id {media_id}: {e}", exc_info=True)
                    log_counter("process_chunks_batch_error",
                                labels={"media_id": media_id, "error_type": type(e).__name__})
                    raise  # Re-raise to be caught by the outer try-except, stopping further processing.

            logging.info(
                f"Finished processing all chunks for media_id {media_id}. Total successfully inserted: {successfully_inserted_chunks}")
            duration = time.time() - start_time
            log_histogram("process_chunks_duration", duration, labels={"media_id": media_id})
            log_counter("process_chunks_success", labels={"media_id": media_id})
            # No explicit return value, matching the original function's behavior.

        except Exception as e:  # Catches errors from loop setup or re-raised errors from batch processing
            duration = time.time() - start_time
            # Log duration even if the overall process failed or exited early
            log_histogram("process_chunks_duration", duration, labels={"media_id": media_id})
            log_counter("process_chunks_error", labels={"media_id": media_id, "error_type": type(e).__name__})
            logging.error(f"Overall error processing chunks for media_id {media_id}: {e}", exc_info=True)

            # Re-raise the exception so the caller is aware of the failure.
            # Wrap in DatabaseError if it's not already one of our specific DB errors.
            if not isinstance(e, (DatabaseError, InputError)):  # Check if e is already a known custom error
                raise DatabaseError(
                    f"An unexpected error occurred while processing chunks for media_id {media_id}: {e}") from e
            else:
                raise


# =========================================================================
# Standalone Functions (REQUIRE db_instance passed explicitly)
# =========================================================================
# These generally call instance methods now, which handle logging/FTS internally.

def get_document_version(db_instance: Database, media_id: int, version_number: Optional[int] = None, include_content: bool = True) -> Optional[Dict[str, Any]]:
    """
    Gets a specific document version or the latest active one for an active media item.

    Filters results to only include versions where both the DocumentVersion itself
    and the parent Media item are not soft-deleted (`deleted = 0`).

    Args:
        db_instance (Database): An initialized Database instance.
        media_id (int): The ID of the parent Media item.
        version_number (Optional[int]): The specific `version_number` to retrieve.
            If None, retrieves the latest (highest `version_number`) active version.
            Must be a positive integer if provided. Defaults to None.
        include_content (bool): Whether to include the 'content' field in the
                                result. Defaults to True.

    Returns:
        Optional[Dict[str, Any]]: A dictionary representing the document version
                                  if found and active, otherwise None.

    Raises:
        TypeError: If `db_instance` is not a Database object or `media_id` is not int.
        ValueError: If `version_number` is provided but is not a positive integer.
        DatabaseError: For database query errors.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("db_instance must be a Database object.")
    if not isinstance(media_id, int):
        raise TypeError("media_id must be an integer.")
    if version_number is not None and (not isinstance(version_number, int) or version_number < 1):
        raise ValueError("Version number must be a positive integer.")
    log_msg = f"Getting {'latest' if version_number is None else f'version {version_number}'} for media_id={media_id}"
    logger.debug(f"{log_msg} (active only) from DB: {db_instance.db_path_str}")
    try:
        select_cols_list = ["dv.id", "dv.uuid", "dv.media_id", "dv.version_number", "dv.created_at",
                           "dv.prompt", "dv.analysis_content", "dv.last_modified", "dv.version",
                           "dv.client_id", "dv.deleted"]
        if include_content:
            select_cols_list.append("dv.content")
        select_cols = ", ".join(select_cols_list)
        params = [media_id]
        query_base = "FROM DocumentVersions dv JOIN Media m ON dv.media_id = m.id WHERE dv.media_id = ? AND dv.deleted = 0 AND m.deleted = 0"
        order_limit = ""
        if version_number is None:
            order_limit = "ORDER BY dv.version_number DESC LIMIT 1"
        else:
            query_base += " AND dv.version_number = ?"
            params.append(version_number)
        final_query = f"SELECT {select_cols} {query_base} {order_limit}"
        cursor = db_instance.execute_query(final_query, tuple(params))
        result = cursor.fetchone()
        if not result:
            logger.warning(f"Active doc version {version_number or 'latest'} not found for active media {media_id}")
            return None
        return dict(result)
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error retrieving {log_msg} DB '{db_instance.db_path_str}': {e}", exc_info=True)
        raise DatabaseError(f"DB error retrieving version: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error retrieving {log_msg} DB '{db_instance.db_path_str}': {e}", exc_info=True)
        raise DatabaseError(f"Unexpected error retrieving version: {e}") from e


# Backup functions remain placeholders or need proper implementation
def create_incremental_backup(db_path, backup_dir):
    logger.warning("create_incremental_backup not implemented.")
    pass


def create_automated_backup(db_path, backup_dir):
    logger.warning("create_automated_backup not implemented.")
    pass


def rotate_backups(backup_dir, max_backups=10):
    logger.warning("rotate_backups not implemented.")
    pass


def check_database_integrity(db_path): # Standalone check is fine
    """
    Performs an integrity check on the specified SQLite database file.

    Connects in read-only mode and executes `PRAGMA integrity_check`.

    Args:
        db_path (str): The path to the SQLite database file.

    Returns:
        bool: True if the integrity check returns 'ok', False otherwise or if
              an error occurs during the check.
    """
    logger.info(f"Checking integrity of database: {db_path}")
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) # Read-only mode
        cursor = conn.execute("PRAGMA integrity_check;")
        result = cursor.fetchone()
        if result and result[0].lower() == 'ok':
            logger.info(f"Integrity check PASSED for {db_path}")
            return True
        else: logger.error(f"Integrity check FAILED for {db_path}: {result}")
        return False
    except sqlite3.Error as e:
        logger.error(f"Error during integrity check for {db_path}: {e}", exc_info=True)
        return False
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass


# Utility Checks
def is_valid_date(date_string: str) -> bool:
    """
    Checks if a string is a valid date in 'YYYY-MM-DD' format.

    Args:
        date_string (Optional[str]): The string to validate.

    Returns:
        bool: True if the string is a valid 'YYYY-MM-DD' date, False otherwise.
    """
    if not date_string:
        return False
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except (ValueError, TypeError):
        return False


def check_media_exists(db_instance: Database, media_id: Optional[int] = None, url: Optional[str] = None, content_hash: Optional[str] = None) -> Optional[int]:
    """
    Checks if an *active* (non-deleted) media item exists using ID, URL, or hash.

    Requires at least one identifier (media_id, url, or content_hash).
    Returns the ID of the first matching active media item found.

    Args:
        db_instance (Database): An initialized Database instance.
        media_id (Optional[int]): The media ID to check.
        url (Optional[str]): The media URL to check.
        content_hash (Optional[str]): The media content hash to check.

    Returns:
        Optional[int]: The integer ID of the existing active media item if found,
                       otherwise None.

    Raises:
        TypeError: If `db_instance` is not a Database object.
        ValueError: If none of `media_id`, `url`, or `content_hash` are provided.
        DatabaseError: For database query errors.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("db_instance required.")
    query_parts = []
    params = []
    if media_id is not None:
        query_parts.append("id = ?")
        params.append(media_id)
    if url:
        query_parts.append("url = ?")
        params.append(url)
    if content_hash:
        query_parts.append("content_hash = ?")
        params.append(content_hash)
    if not query_parts:
        raise ValueError("Must provide id, url, or content_hash to check.")
    query = f"SELECT id FROM Media WHERE ({' OR '.join(query_parts)}) AND deleted = 0 LIMIT 1"
    try:
        cursor = db_instance.execute_query(query, tuple(params))
        result = cursor.fetchone()
        return result['id'] if result else None
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error checking media existence DB '{db_instance.db_path_str}': {e}")
        raise DatabaseError(f"Failed check media existence: {e}") from e


def empty_trash(db_instance: Database, days_threshold: int) -> Tuple[int, int]:
    """
    Permanently removes items from the trash that are older than a threshold.

    Finds Media items where `is_trash = 1`, `deleted = 0`, and `trash_date`
    is older than `days_threshold` days ago. For each such item found, it calls
    `db_instance.soft_delete_media(media_id, cascade=True)` to perform the
    soft delete, log sync events, update FTS, and handle cascades.

    Args:
        db_instance (Database): An initialized Database instance.
        days_threshold (int): The minimum number of days an item must have been
                              in the trash (based on `trash_date`) to be emptied.
                              Must be a non-negative integer.

    Returns:
        Tuple[int, int]: A tuple containing:
            - processed_count (int): Number of items successfully moved from trash
                                     to the soft-deleted state.
            - remaining_count (int): Number of items still in the UI trash
                                     (`is_trash = 1`, `deleted = 0`) after the operation.
                                     Returns -1 for remaining_count if an error occurred
                                     during the final count query.

    Raises:
        TypeError: If `db_instance` is not a Database object.
        ValueError: If `days_threshold` is not a non-negative integer.
        DatabaseError: Can be raised by the underlying `soft_delete_media` calls if
                       they encounter issues beyond ConflictError. Errors during the
                       initial query or final count also raise DatabaseError.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("db_instance required.")
    if not isinstance(days_threshold, int) or days_threshold < 0:
        raise ValueError("Days must be non-negative int.")
    threshold_date_str = (datetime.now(timezone.utc) - timedelta(days=days_threshold)).strftime('%Y-%m-%dT%H:%M:%SZ')  # ISO Format
    processed_count = 0
    logger.info(f"Emptying trash older than {days_threshold} days ({threshold_date_str}) on DB {db_instance.db_path_str}")
    try:
        cursor_find = db_instance.execute_query("SELECT id, title FROM Media WHERE is_trash = 1 AND deleted = 0 AND trash_date <= ?", (threshold_date_str,))
        items_to_process = cursor_find.fetchall()
        if not items_to_process:
            logger.info("No items found in trash older than threshold.")
        else:
            logger.info(f"Found {len(items_to_process)} items to process.")
            for item in items_to_process:
                media_id, title = item['id'], item['title']
                logger.debug(f"Processing item ID {media_id} ('{title}') for sync delete from trash.")
                try:
                    success = db_instance.soft_delete_media(media_id=media_id, cascade=True)  # Instance method handles logging/FTS
                    if success:
                        processed_count += 1
                    else:
                        logger.warning(f"Failed process item ID {media_id} during trash emptying.")
                except ConflictError as e:
                    logger.warning(f"Conflict processing item ID {media_id} during trash emptying: {e}")
                except DatabaseError as e:
                    logger.error(f"DB error processing item ID {media_id} during trash emptying: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error processing item ID {media_id} during trash emptying: {e}", exc_info=True)
        cursor_remain = db_instance.execute_query("SELECT COUNT(*) FROM Media WHERE is_trash = 1 AND deleted = 0")
        remaining_count = cursor_remain.fetchone()[0]
        logger.info(f"Trash emptying complete. Processed (sync deleted): {processed_count}. Remaining in UI trash: {remaining_count}.")
        return processed_count, remaining_count
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error emptying trash DB '{db_instance.db_path_str}': {e}", exc_info=True)
        return 0, -1
    except Exception as e:
        logger.error(f"Unexpected error emptying trash DB '{db_instance.db_path_str}': {e}", exc_info=True)
        return 0, -1

# Deprecated check
def check_media_and_whisper_model(*args, **kwargs):
    logger.warning("check_media_and_whisper_model is deprecated.")
    return True, "Deprecated"

# Media processing state functions (unchanged logic, rely on DB fields)
def get_unprocessed_media(db_instance: Database) -> List[Dict]:
    """
    Retrieves media items marked as needing vector processing.

    Fetches active, non-trashed media items where `vector_processing = 0`.
    Returns a list of dictionaries containing basic info (id, uuid, content, type, title).

    Args:
        db_instance (Database): An initialized Database instance.

    Returns:
        List[Dict[str, Any]]: A list of media items needing processing. Empty if none.

    Raises:
        TypeError: If `db_instance` is not a Database object.
        DatabaseError: For database query errors.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("db_instance required.")
    try:
        query = "SELECT id, uuid, content, type, title FROM Media WHERE vector_processing = 0 AND deleted = 0 AND is_trash = 0 ORDER BY id"
        cursor = db_instance.execute_query(query)
        return [dict(row) for row in cursor.fetchall()]
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error getting unprocessed media DB '{db_instance.db_path_str}': {e}")
        raise DatabaseError("Failed get unprocessed media") from e


def mark_media_as_processed(db_instance: Database, media_id: int):
    """
    Marks a media item's vector processing status as complete (`vector_processing = 1`).

    Important: This function ONLY updates the `vector_processing` flag. It DOES NOT
    update the `last_modified` timestamp, increment the sync `version`, or log a
    sync event. It's intended for internal state tracking after a potentially long
    vector processing task, assuming a separate mechanism handles the main media
    updates and sync logging if content/vectors were added.

    Args:
        db_instance (Database): An initialized Database instance.
        media_id (int): The ID of the media item to mark as processed.

    Raises:
        TypeError: If `db_instance` is not a Database object.
        DatabaseError: For database query errors.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("db_instance required.")
    logger.debug(f"Marking media {media_id} vector_processing=1 on DB '{db_instance.db_path_str}'.")
    try:
        cursor = db_instance.execute_query("UPDATE Media SET vector_processing = 1 WHERE id = ? AND deleted = 0", (media_id,), commit=True)
        if cursor.rowcount == 0:
            logger.warning(f"Attempted mark media {media_id} processed, but not found/deleted.")
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error marking media {media_id} processed '{db_instance.db_path_str}': {e}")
        raise DatabaseError(f"Failed mark media {media_id} processed") from e

# Ingestion wrappers call instance methods
def ingest_article_to_db_new(db_instance: Database, *,
                             url: str, title: str,
                             content: str,
                             author: Optional[str] = None,
                             keywords: Optional[List[str]] = None,
                             summary: Optional[str] = None,
                             ingestion_date: Optional[str] = None,
                             custom_prompt: Optional[str] = None,
                             overwrite: bool = False) -> Tuple[Optional[int],
                            Optional[str], str]:
    """
    Wrapper function to add or update an article using `add_media_with_keywords`.

    Sets `media_type` to 'article'. Uses `summary` as `analysis_content` and
    `custom_prompt` as `prompt` for the initial document version.

    Args:
        db_instance (Database): An initialized Database instance.
        url (str): The URL of the article. Required.
        title (str): The title of the article. Required.
        content (str): The main content of the article. Required.
        author (Optional[str]): Author of the article.
        keywords (Optional[List[str]]): Keywords associated with the article.
        summary (Optional[str]): A summary or analysis of the article.
        ingestion_date (Optional[str]): ISO 8601 UTC timestamp string. Defaults to now.
        custom_prompt (Optional[str]): A prompt related to the article/summary.
        overwrite (bool): If True, update if article exists. Defaults to False.

    Returns:
        Tuple[Optional[int], Optional[str], str]: Result from `add_media_with_keywords`:
            (media_id, media_uuid, message).

    Raises:
        TypeError: If `db_instance` is not a Database object.
        InputError: If required fields (url, title, content) are missing/invalid.
        ConflictError: If overwrite=True and update fails due to version conflict.
        DatabaseError: For underlying database or sync/FTS errors.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("db_instance required.")
    if not url or not title or content is None:
        raise InputError("URL, Title, and Content are required.")
    return db_instance.add_media_with_keywords(
        url=url,
        title=title,
        media_type='article',
        content=content,
        keywords=keywords,
        prompt=custom_prompt,
        analysis_content=summary,
        author=author,
        ingestion_date=ingestion_date,
        overwrite=overwrite
    )


def import_obsidian_note_to_db(db_instance: Database, note_data: Dict[str, Any]) -> Tuple[Optional[int], Optional[str], str]:
    """
    Wrapper function to add or update an Obsidian note using `add_media_with_keywords`.

    Extracts relevant fields from the `note_data` dictionary. Uses Obsidian tags
    as keywords and YAML frontmatter (if present and valid) as `analysis_content`.
    Constructs a default URL like 'obsidian://note/TITLE'.

    Requires `pyyaml` to be installed to parse frontmatter.

    Args:
        db_instance (Database): An initialized Database instance.
        note_data (Dict[str, Any]): A dictionary containing note information.
            Expected keys: 'title' (str, required), 'content' (str, required).
            Optional keys: 'tags' (List[str|int]), 'frontmatter' (Dict),
            'file_created_date' (str, ISO 8601 UTC), 'overwrite' (bool).

    Returns:
        Tuple[Optional[int], Optional[str], str]: Result from `add_media_with_keywords`:
            (media_id, media_uuid, message).

    Raises:
        TypeError: If `db_instance` is not a Database object or `note_data` is not a dict.
        InputError: If required keys ('title', 'content') are missing or invalid in `note_data`.
        ConflictError: If overwrite=True and update fails due to version conflict.
        DatabaseError: For underlying database or sync/FTS errors.
        ImportError: If `yaml` library is needed but not installed.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("db_instance required.")
    required = ['title', 'content']
    missing = [k for k in required if k not in note_data or note_data[k] is None]
    if missing:
        raise InputError(f"Obsidian note missing required keys: {missing}")
    url_id = f"obsidian://note/{note_data['title']}"
    kw = note_data.get('tags', [])
    kw = [str(k) for k in kw if isinstance(k, (str, int))]
    fm_str = None
    fm = note_data.get('frontmatter')
    author = None
    if isinstance(fm, dict):
        author = fm.get('author')
        try:
            fm_str = yaml.dump(fm, default_flow_style=False)
        except Exception as e:
            logger.error(f"Error dumping frontmatter: {e}")
    return db_instance.add_media_with_keywords(url=url_id, title=note_data['title'], media_type='obsidian_note', content=note_data['content'], keywords=kw, author=author, prompt="Obsidian Frontmatter" if fm_str else None, analysis_content=fm_str, ingestion_date=note_data.get('file_created_date'), overwrite=note_data.get('overwrite', False))


# Read functions call instance methods or query directly with filters
def get_media_transcripts(db_instance: Database, media_id: int) -> List[Dict]:
    """
    Retrieves all active transcripts associated with an active media item.

    Filters results to only include transcripts where both the Transcript itself
    and the parent Media item are not soft-deleted (`deleted = 0`).
    Results are ordered by creation date descending (newest first).

    Args:
        db_instance (Database): An initialized Database instance.
        media_id (int): The ID of the parent Media item.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing an active
                              transcript. Returns an empty list if none are found.

    Raises:
        TypeError: If `db_instance` is not a Database object or `media_id` is not int.
        DatabaseError: For database query errors.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("db_instance required.")
    logger.debug(f"Fetching transcripts for media_id={media_id} DB: {db_instance.db_path_str}")
    try:
        query = "SELECT t.* FROM Transcripts t JOIN Media m ON t.media_id = m.id WHERE t.media_id = ? AND t.deleted = 0 AND m.deleted = 0 ORDER BY t.created_at DESC"
        cursor = db_instance.execute_query(query, (media_id,))
        return [dict(row) for row in cursor.fetchall()]
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error getting transcripts media {media_id} '{db_instance.db_path_str}': {e}")
        raise DatabaseError(f"Failed get transcripts {media_id}") from e


def get_latest_transcription(db_instance: Database, media_id: int) -> Optional[str]:
    """
    Retrieves the text content of the latest active transcript for an active media item.

    Filters for active transcripts and media, orders by creation date descending,
    and returns only the `transcription` field of the newest one.

    Args:
        db_instance (Database): An initialized Database instance.
        media_id (int): The ID of the parent Media item.

    Returns:
        Optional[str]: The transcription text if found, otherwise None.

    Raises:
        TypeError: If `db_instance` is not a Database object or `media_id` is not int.
        DatabaseError: For database query errors.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("db_instance required.")
    try:
        query = "SELECT t.transcription FROM Transcripts t JOIN Media m ON t.media_id = m.id WHERE t.media_id = ? AND t.deleted = 0 AND m.deleted = 0 ORDER BY t.created_at DESC LIMIT 1"
        cursor = db_instance.execute_query(query, (media_id,))
        result = cursor.fetchone()
        return result['transcription'] if result else None
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error get latest transcript {media_id} '{db_instance.db_path_str}': {e}")
        raise DatabaseError(f"Failed get latest transcript {media_id}") from e


def get_specific_transcript(db_instance: Database, transcript_uuid: str) -> Optional[Dict]:
    """
    Retrieves a specific active transcript by its UUID, ensuring parent media is active.

    Filters results to only include the transcript if both it and its parent
    Media item are not soft-deleted (`deleted = 0`).

    Args:
        db_instance (Database): An initialized Database instance.
        transcript_uuid (str): The UUID of the transcript to retrieve.

    Returns:
        Optional[Dict[str, Any]]: A dictionary representing the transcript if found
                                   and active, otherwise None.

    Raises:
        TypeError: If `db_instance` is not Database object or `transcript_uuid` not str.
        InputError: If `transcript_uuid` is empty.
        DatabaseError: For database query errors.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("db_instance required.")
    try:
        query = "SELECT t.* FROM Transcripts t JOIN Media m ON t.media_id = m.id WHERE t.uuid = ? AND t.deleted = 0 AND m.deleted = 0"
        cursor = db_instance.execute_query(query, (transcript_uuid,))
        result = cursor.fetchone()
        return dict(result) if result else None
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error get transcript UUID {transcript_uuid} '{db_instance.db_path_str}': {e}")
        raise DatabaseError(f"Failed get transcript {transcript_uuid}") from e


def get_specific_analysis(db_instance: Database, version_uuid: str) -> Optional[str]:
    """
    Retrieves the `analysis_content` from a specific active DocumentVersion.

    Ensures both the DocumentVersion and its parent Media item are active (`deleted=0`).

    Args:
        db_instance (Database): An initialized Database instance.
        version_uuid (str): The UUID of the DocumentVersion.

    Returns:
        Optional[str]: The analysis content string if found and active, otherwise None.

    Raises:
        TypeError: If `db_instance` is not Database object or `version_uuid` not str.
        InputError: If `version_uuid` is empty.
        DatabaseError: For database query errors.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("db_instance required.")
    try:
        query = "SELECT dv.analysis_content FROM DocumentVersions dv JOIN Media m ON dv.media_id = m.id WHERE dv.uuid = ? AND dv.deleted = 0 AND m.deleted = 0"
        cursor = db_instance.execute_query(query, (version_uuid,))
        result = cursor.fetchone()
        return result['analysis_content'] if result else None
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error get analysis UUID {version_uuid} '{db_instance.db_path_str}': {e}")
        raise DatabaseError(f"Failed get analysis {version_uuid}") from e


def get_media_prompts(db_instance: Database, media_id: int) -> List[Dict]:
    """
    Retrieves all non-empty prompts from active DocumentVersions for an active media item.

    Filters for active versions and media, excludes rows where `prompt` is NULL or empty,
    and orders by version number descending (newest first).

    Args:
        db_instance (Database): An initialized Database instance.
        media_id (int): The ID of the parent Media item.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing 'id', 'uuid',
                              'content' (the prompt text), 'created_at', and
                              'version_number' for matching prompts. Empty list if none.

    Raises:
        TypeError: If `db_instance` is not Database object or `media_id` not int.
        DatabaseError: For database query errors.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("db_instance required.")
    try:
        query = "SELECT dv.id, dv.uuid, dv.prompt, dv.created_at, dv.version_number FROM DocumentVersions dv JOIN Media m ON dv.media_id = m.id WHERE dv.media_id = ? AND dv.deleted = 0 AND m.deleted = 0 AND dv.prompt IS NOT NULL AND dv.prompt != '' ORDER BY dv.version_number DESC"
        cursor = db_instance.execute_query(query, (media_id,))
        return [{'id': r['id'], 'uuid': r['uuid'], 'content': r['prompt'], 'created_at': r['created_at'], 'version_number': r['version_number']} for r in cursor.fetchall()]
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error get prompts media {media_id} '{db_instance.db_path_str}': {e}")
        raise DatabaseError(f"Failed get prompts {media_id}") from e


def get_specific_prompt(db_instance: Database, version_uuid: str) -> Optional[str]:
    """
    Retrieves the `prompt` text from a specific active DocumentVersion.

    Ensures both the DocumentVersion and its parent Media item are active (`deleted=0`).

    Args:
        db_instance (Database): An initialized Database instance.
        version_uuid (str): The UUID of the DocumentVersion.

    Returns:
        Optional[str]: The prompt string if found and active, otherwise None.

    Raises:
        TypeError: If `db_instance` is not Database object or `version_uuid` not str.
        InputError: If `version_uuid` is empty.
        DatabaseError: For database query errors.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("db_instance required.")
    try:
        query = "SELECT dv.prompt FROM DocumentVersions dv JOIN Media m ON dv.media_id = m.id WHERE dv.uuid = ? AND dv.deleted = 0 AND m.deleted = 0"
        cursor = db_instance.execute_query(query, (version_uuid,))
        result = cursor.fetchone()
        return result['prompt'] if result else None
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error get prompt UUID {version_uuid} '{db_instance.db_path_str}': {e}")
        raise DatabaseError(f"Failed get prompt {version_uuid}") from e


# Specific deletes call instance methods
def soft_delete_transcript(db_instance: Database, transcript_uuid: str) -> bool:
    """
    Soft deletes a specific transcript by its UUID.

    Sets `deleted=1`, updates `last_modified`, increments sync `version`, and
    logs a 'delete' sync event for the `Transcripts` entity. Ensures the
    parent Media item is active before proceeding.

    Args:
        db_instance (Database): An initialized Database instance.
        transcript_uuid (str): The UUID of the transcript to soft delete.

    Returns:
        bool: True if successfully soft-deleted, False if not found or already deleted.

    Raises:
        TypeError: If `db_instance` is not a Database object.
        InputError: If `transcript_uuid` is empty or None.
        ConflictError: If the transcript's version changed concurrently.
        DatabaseError: For other database errors or sync logging failures.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("db_instance required.")
    if not transcript_uuid:
        raise InputError("Transcript UUID required.")

    current_time = db_instance._get_current_utc_timestamp_str()  # Get time via instance
    client_id = db_instance.client_id
    logger.debug(f"Attempting soft delete Transcript UUID: {transcript_uuid}")
    try:
        with db_instance.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT t.id, t.version, m.uuid as media_uuid FROM Transcripts t JOIN Media m ON t.media_id = m.id WHERE t.uuid = ? AND t.deleted = 0", (transcript_uuid,))
            info = cursor.fetchone()
            if not info:
                logger.warning(f"Transcript UUID {transcript_uuid} not found or already deleted.")
                return False
            t_id, current_version, media_uuid = info['id'], info['version'], info['media_uuid']
            new_version = current_version + 1

            # Pass current_time for last_modified
            cursor.execute("UPDATE Transcripts SET deleted=1, last_modified=?, version=?, client_id=? WHERE id=? AND version=?",
                    (current_time, new_version, client_id, t_id, current_version))
            if cursor.rowcount == 0:
                raise ConflictError("Transcripts", t_id)

            # Payload reflects the state *after* the update
            payload = {'uuid': transcript_uuid, 'media_uuid': media_uuid, 'last_modified': current_time, 'version': new_version, 'client_id': client_id, 'deleted': 1}
            db_instance._log_sync_event(conn, 'Transcripts', transcript_uuid, 'delete', new_version, payload)  # Call instance method for logging
            logger.info(f"Soft deleted Transcript UUID {transcript_uuid}. New ver: {new_version}")
            return True
    except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error soft delete Transcript UUID {transcript_uuid}: {e}", exc_info=True)
        # Re-raise specific errors, wrap general DB errors
        if isinstance(e, (InputError, ConflictError, DatabaseError)):
            raise e
        else:
            raise DatabaseError(f"Failed soft delete transcript: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected soft delete Transcript error UUID {transcript_uuid}: {e}", exc_info=True)
        raise DatabaseError(f"Unexpected transcript soft delete error: {e}") from e


# clear_specific_analysis/prompt call instance methods implicitly via update logic
def clear_specific_analysis(db_instance: Database, version_uuid: str) -> bool:
    """
    Clears the `analysis_content` field (sets to NULL) for a specific active DocumentVersion.

    Updates `last_modified`, increments sync `version`, and logs an 'update'
    sync event for the `DocumentVersions` entity. Ensures the version is active.

    Args:
        db_instance (Database): An initialized Database instance.
        version_uuid (str): The UUID of the DocumentVersion whose analysis to clear.

    Returns:
        bool: True if analysis was successfully cleared, False if version not found/deleted.

    Raises:
        TypeError: If `db_instance` is not a Database object.
        InputError: If `version_uuid` is empty or None.
        ConflictError: If the version's sync version changed concurrently.
        DatabaseError: For other database errors or sync logging failures.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("db_instance required.")
    if not version_uuid:
        raise InputError("Version UUID required.")

    current_time = db_instance._get_current_utc_timestamp_str()  # Get time via instance
    client_id = db_instance.client_id
    logger.debug(f"Clearing analysis for DocVersion UUID: {version_uuid}")
    try:
        with db_instance.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, version FROM DocumentVersions WHERE uuid = ? AND deleted = 0", (version_uuid,))
            info = cursor.fetchone()
            if not info:
                logger.warning(f"DocVersion UUID {version_uuid} not found or already deleted.")
                return False
            v_id, current_version = info['id'], info['version']
            new_version = current_version + 1

            # Pass current_time for last_modified
            cursor.execute("UPDATE DocumentVersions SET analysis_content=NULL, last_modified=?, version=?, client_id=? WHERE id=? AND version=?",
                           (current_time, new_version, client_id, v_id, current_version))
            if cursor.rowcount == 0:
                raise ConflictError("DocumentVersions", v_id)

            # Fetch full data for payload AFTER update
            cursor.execute("SELECT dv.*, m.uuid as media_uuid FROM DocumentVersions dv JOIN Media m ON dv.media_id = m.id WHERE dv.id = ?", (v_id,))
            payload = dict(cursor.fetchone())
            db_instance._log_sync_event(conn, 'DocumentVersions', version_uuid, 'update', new_version, payload)  # Call instance method for logging
            logger.info(f"Cleared analysis for DocVersion UUID {version_uuid}. New ver: {new_version}")
            return True
    except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error clearing analysis UUID {version_uuid}: {e}", exc_info=True)
        if isinstance(e, (InputError, ConflictError, DatabaseError)):
            raise e
        else:
            raise DatabaseError(f"Failed clear analysis: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error clearing analysis UUID {version_uuid}: {e}", exc_info=True)
        raise DatabaseError(f"Unexpected clear analysis error: {e}") from e


def clear_specific_prompt(db_instance: Database, version_uuid: str) -> bool:
    """
    Clears the `prompt` field (sets to NULL) for a specific active DocumentVersion.

    Updates `last_modified`, increments sync `version`, and logs an 'update'
    sync event for the `DocumentVersions` entity. Ensures the version is active.

    Args:
        db_instance (Database): An initialized Database instance.
        version_uuid (str): The UUID of the DocumentVersion whose prompt to clear.

    Returns:
        bool: True if prompt was successfully cleared, False if version not found/deleted.

    Raises:
        TypeError: If `db_instance` is not a Database object.
        InputError: If `version_uuid` is empty or None.
        ConflictError: If the version's sync version changed concurrently.
        DatabaseError: For other database errors or sync logging failures.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("db_instance required.")
    if not version_uuid:
        raise InputError("Version UUID required.")

    current_time = db_instance._get_current_utc_timestamp_str()  # Get time via instance
    client_id = db_instance.client_id
    logger.debug(f"Clearing prompt for DocVersion UUID: {version_uuid}")
    try:
        with db_instance.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, version FROM DocumentVersions WHERE uuid = ? AND deleted = 0", (version_uuid,))
            info = cursor.fetchone()
            if not info:
                logger.warning(f"DocVersion UUID {version_uuid} not found or already deleted.")
                return False
            v_id, current_version = info['id'], info['version']
            new_version = current_version + 1

            # Pass current_time for last_modified
            cursor.execute("UPDATE DocumentVersions SET prompt=NULL, last_modified=?, version=?, client_id=? WHERE id=? AND version=?",
                           (current_time, new_version, client_id, v_id, current_version))
            if cursor.rowcount == 0:
                raise ConflictError("DocumentVersions", v_id)

            # Fetch full data for payload AFTER update
            cursor.execute("SELECT dv.*, m.uuid as media_uuid FROM DocumentVersions dv JOIN Media m ON dv.media_id = m.id WHERE dv.id = ?", (v_id,))
            payload = dict(cursor.fetchone())
            db_instance._log_sync_event(conn, 'DocumentVersions', version_uuid, 'update', new_version, payload)  # Call instance method for logging
            logger.info(f"Cleared prompt for DocVersion UUID {version_uuid}. New ver: {new_version}")
            return True
    except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error clearing prompt UUID {version_uuid}: {e}", exc_info=True)
        if isinstance(e, (InputError, ConflictError, DatabaseError)):
            raise e
        else:
            raise DatabaseError(f"Failed clear prompt: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error clearing prompt UUID {version_uuid}: {e}", exc_info=True)
        raise DatabaseError(f"Unexpected clear prompt error: {e}") from e


# Other remaining functions
def get_chunk_text(db_instance: Database, chunk_uuid: str) -> Optional[str]:
    """
    Retrieves the text content (`chunk_text`) of a specific active chunk.

    Currently queries `UnvectorizedMediaChunks`. Ensures both the chunk and its
    parent Media item are active (`deleted=0`).

    Args:
     db_instance (Database): An initialized Database instance.
     chunk_uuid (str): The UUID of the chunk (from UnvectorizedMediaChunks).

    Returns:
     Optional[str]: The chunk text if found and active, otherwise None.

    Raises:
     TypeError: If `db_instance` is not Database object or `chunk_uuid` not str.
     InputError: If `chunk_uuid` is empty.
     DatabaseError: For database query errors.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("db_instance required.")
    target_table = "UnvectorizedMediaChunks"  # Assuming this table for text
    try:
        query = f"SELECT c.chunk_text FROM {target_table} c JOIN Media m ON c.media_id = m.id WHERE c.uuid = ? AND c.deleted = 0 AND m.deleted = 0"
        cursor = db_instance.execute_query(query, (chunk_uuid,))
        result = cursor.fetchone()
        return result['chunk_text'] if result else None
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error get chunk text UUID {chunk_uuid} '{db_instance.db_path_str}': {e}")
        raise DatabaseError(f"Failed get chunk text {chunk_uuid}") from e


def get_all_content_from_database(db_instance: Database) -> List[Dict[str, Any]]:
    """
    Retrieves basic identifying information for all active, non-trashed media items.

    Fetches `id`, `uuid`, `content`, `title`, `author`, `type`, `url`,
    `ingestion_date`, `last_modified` for items where `deleted = 0` and `is_trash = 0`.
    Ordered by `last_modified` descending.

    Args:
        db_instance (Database): An initialized Database instance.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing an active
                              media item. Empty list if none found.

    Raises:
        TypeError: If `db_instance` is not a Database object.
        DatabaseError: For database query errors.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("db_instance required.")
    try:
        cursor = db_instance.execute_query("SELECT id, uuid, content, title, author, type, url, ingestion_date, last_modified FROM Media WHERE deleted = 0 AND is_trash = 0 ORDER BY last_modified DESC")
        return [dict(item) for item in cursor.fetchall()]
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error retrieving all content DB '{db_instance.db_path_str}': {e}")
        raise DatabaseError("Error retrieving all content") from e


def permanently_delete_item(db_instance: Database, media_id: int) -> bool:
    """
        Performs a HARD delete of a media item and its related data via cascades.

        **DANGER:** This operation bypasses the soft delete mechanism and the sync log.
        It physically removes the row from the `Media` table. Foreign key constraints
        with `ON DELETE CASCADE` should automatically delete related rows in child
        tables (`Transcripts`, `MediaKeywords`, `DocumentVersions`, etc.). It also
        explicitly removes the corresponding FTS entry. Use with extreme caution,
        especially in synchronized environments, as this change will not be propagated
        through the sync log. Primarily intended for cleanup or specific admin tasks.

        Args:
            db_instance (Database): An initialized Database instance.
            media_id (int): The ID of the Media item to permanently delete.

        Returns:
            bool: True if the item was found and deleted, False otherwise.

        Raises:
            TypeError: If `db_instance` is not a Database object.
            DatabaseError: For database errors during deletion.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("db_instance required.")
    logger.warning(f"!!! PERMANENT DELETE initiated Media ID: {media_id} DB {db_instance.db_path_str}. NOT SYNCED !!!")
    try:
        with db_instance.transaction() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM Media WHERE id = ?", (media_id,))
            if not cursor.fetchone():
                logger.warning(f"Permanent delete failed: Media {media_id} not found.")
                return False
            # Hard delete - Cascades should handle children via FKs
            cursor.execute("DELETE FROM Media WHERE id = ?", (media_id,))
            deleted_count = cursor.rowcount
            # Manually delete from FTS (cascade should work, but belt-and-suspenders)
            db_instance._delete_fts_media(conn, media_id)
        if deleted_count > 0:
            logger.info(f"Permanently deleted Media ID: {media_id}. NO sync log generated.")
            return True
        else:
            logger.error(f"Permanent delete failed unexpectedly Media {media_id}.")
            return False
    except sqlite3.Error as e:
        logger.error(f"Error permanently deleting Media {media_id}: {e}", exc_info=True)
        raise DatabaseError(f"Failed permanently delete item: {e}") from e
    except Exception as e:
        (logger.error(f"Unexpected error permanently deleting Media {media_id}: {e}", exc_info=True))
        raise DatabaseError(f"Unexpected permanent delete error: {e}") from e


# Keyword read functions use instance methods or query directly
def fetch_keywords_for_media(media_id: int, db_instance: Database) -> List[str]:
    """
       Fetches all active keywords associated with a specific active media item.

       Filters results to only include keywords where both the Keyword itself and
       the parent Media item are not soft-deleted (`deleted = 0`).
       Results are sorted alphabetically (case-insensitive).

       Args:
           media_id (int): The ID of the Media item.
           db_instance (Database): An initialized Database instance.

       Returns:
           List[str]: A sorted list of active keyword strings linked to the media item.
                      Returns an empty list if none are found or if the media item
                      is inactive.

       Raises:
           TypeError: If `db_instance` is not Database object or `media_id` not int.
           DatabaseError: For database query errors.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("db_instance required.")
    logger.debug(f"Fetching keywords media_id={media_id} DB: {db_instance.db_path_str}")
    try:
        query = "SELECT k.keyword FROM Keywords k JOIN MediaKeywords mk ON k.id = mk.keyword_id JOIN Media m ON mk.media_id = m.id WHERE mk.media_id = ? AND k.deleted = 0 AND m.deleted = 0 ORDER BY k.keyword COLLATE NOCASE"
        cursor = db_instance.execute_query(query, (media_id,))
        return [row['keyword'] for row in cursor.fetchall()]
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Error fetching keywords media_id {media_id} '{db_instance.db_path_str}': {e}", exc_info=True)
        raise DatabaseError(f"Failed fetch keywords {media_id}") from e


def fetch_keywords_for_media_batch(media_ids: List[int], db_instance: Database) -> Dict[int, List[str]]:
    """
       Fetches active keywords for multiple active media items in a single query.

       Returns a dictionary mapping each requested `media_id` to a sorted list of
       its associated active keyword strings. Only includes media IDs that were
       found and are active.

       Args:
           media_ids (List[int]): A list of Media item IDs.
           db_instance (Database): An initialized Database instance.

       Returns:
           Dict[int, List[str]]: A dictionary where keys are the input `media_id`s
                                 (that are active and have keywords) and values are sorted
                                 lists of their active keyword strings. IDs not found,
                                 inactive, or without keywords will be omitted.

       Raises:
           TypeError: If `db_instance` is not Database object or `media_ids` not list.
           InputError: If `media_ids` contains non-integer values.
           DatabaseError: For database query errors.
    """
    if not isinstance(db_instance, Database):
        raise TypeError("db_instance required.")
    if not media_ids:
        return {}
    try:
        safe_media_ids = [int(mid) for mid in media_ids]
    except (ValueError, TypeError) as e:
        raise InputError(f"media_ids must be list of integers: {e}")
    if not safe_media_ids:
        return {}
    keywords_map = {media_id: [] for media_id in safe_media_ids}
    placeholders = ','.join('?' * len(safe_media_ids))
    query = f"SELECT mk.media_id, k.keyword FROM MediaKeywords mk JOIN Keywords k ON mk.keyword_id = k.id JOIN Media m ON mk.media_id = m.id WHERE mk.media_id IN ({placeholders}) AND k.deleted = 0 AND m.deleted = 0 ORDER BY mk.media_id, k.keyword COLLATE NOCASE"
    try:
        cursor = db_instance.execute_query(query, tuple(safe_media_ids))
        for row in cursor.fetchall():
            if row['media_id'] in keywords_map:
                keywords_map[row['media_id']].append(row['keyword'])
        return keywords_map
    except (DatabaseError, sqlite3.Error) as e:
        logger.error(f"Failed fetch keywords batch '{db_instance.db_path_str}': {e}", exc_info=True)
        raise DatabaseError("Failed fetch keywords batch") from e


#
# End of Media_DB_v2.py
#######################################################################################################################
