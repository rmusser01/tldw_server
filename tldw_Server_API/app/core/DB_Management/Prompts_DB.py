# Prompts_DB_v2.py
#########################################
# Prompts_DB_v2 Library
# Manages Prompts_DB_v2 operations for specific instances, handling sync metadata internally.
# Requires a client_id during Database initialization.
# Standalone functions require a PromptsDatabase instance passed as an argument.
#
# Manages SQLite database interactions for prompts and related metadata.
#
# This library provides a `PromptsDatabase` class to encapsulate operations for a specific
# SQLite database file. It handles connection management (thread-locally),
# schema initialization and versioning, CRUD operations, Full-Text Search (FTS)
# updates, and internal logging of changes for synchronization purposes via a
# `sync_log` table.
#
# Key Features:
# - Instance-based: Each `PromptsDatabase` object connects to a specific DB file.
# - Client ID Tracking: Requires a `client_id` for attributing changes.
# - Internal Sync Logging: Automatically logs creates, updates, deletes, links,
#   and unlinks to the `sync_log` table for external sync processing.
# - Internal FTS Updates: Manages associated FTS5 tables (`prompts_fts`, `prompt_keywords_fts`)
#   within the Python code during relevant operations.
# - Schema Versioning: Checks and applies schema updates upon initialization.
# - Thread-Safety: Uses thread-local storage for database connections.
# - Soft Deletes: Implements soft deletes (`deleted=1`) for Prompts and Keywords.
# - Transaction Management: Provides a context manager for atomic operations.
# - Standalone Functions: Offers utility functions that operate on a `PromptsDatabase`
#   instance (e.g., searching, fetching related data, exporting).
####
#
import json
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from typing import Any, Optional, Union

#
# Third-Party Libraries
from loguru import logger

from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    begin_immediate_if_needed,
    configure_sqlite_connection,
)
from tldw_Server_API.app.core.DB_Management.prompts_db_helpers import (
    build_structured_prompt_searchable_text,
    deserialize_prompt_record,
    normalize_keyword,
    normalize_text_for_search,
    serialize_prompt_definition,
)
from loguru import logger as logging

from tldw_Server_API.app.core.testing import is_test_mode

#
# Local Imports
#
########################################################################################################################
#
# Functions:

# --- Custom Exceptions (mirrors the legacy media DB shape) ---
class DatabaseError(Exception):
    """Base exception for database related errors."""
    pass


class SchemaError(DatabaseError):
    """Exception for schema version mismatches or migration failures."""
    pass


class InputError(ValueError):
    """Custom exception for input validation errors."""

    DEFAULT_SAFE_MESSAGE = "Invalid input."

    def __init__(self, message: str, safe_message: Optional[str] = None):
        super().__init__(message)
        self.original_message = str(message)
        self.safe_message = safe_message or self.DEFAULT_SAFE_MESSAGE


class ConflictError(DatabaseError):
    """Indicates a conflict due to concurrent modification (version mismatch)."""

    def __init__(self, message="Conflict detected: Record modified concurrently.", entity=None, identifier=None):
        super().__init__(message)
        self.entity = entity
        self.identifier = identifier

    def __str__(self):
        base = super().__str__()
        details = []
        if self.entity:
            details.append(f"Entity: {self.entity}")
        if self.identifier:
            details.append(f"ID: {self.identifier}")
        return f"{base} ({', '.join(details)})" if details else base


_PROMPTS_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    DatabaseError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeError,
    ValueError,
    json.JSONDecodeError,
    sqlite3.Error,
)


_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# --- Database Class ---
class PromptsDatabase:
    _CURRENT_SCHEMA_VERSION = 5

    _TABLES_SQL_V1 = """
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY NOT NULL
    );
    INSERT OR IGNORE INTO schema_version (version) VALUES (0);

    CREATE TABLE IF NOT EXISTS Prompts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        author TEXT,
        details TEXT,
        system_prompt TEXT, -- Renamed from 'system'
        user_prompt TEXT,   -- Renamed from 'user'
        uuid TEXT UNIQUE NOT NULL,
        last_modified DATETIME NOT NULL,
        version INTEGER NOT NULL DEFAULT 1,
        usage_count INTEGER NOT NULL DEFAULT 0,
        last_used_at DATETIME,
        client_id TEXT NOT NULL,
        deleted BOOLEAN NOT NULL DEFAULT 0,
        prev_version INTEGER,
        merge_parent_uuid TEXT
    );

    CREATE TABLE IF NOT EXISTS PromptKeywordsTable ( -- Renamed from Keywords to avoid clash if in same scope
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

    CREATE TABLE IF NOT EXISTS PromptKeywordLinks ( -- Renamed from PromptKeywords for clarity
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt_id INTEGER NOT NULL,
        keyword_id INTEGER NOT NULL,
        UNIQUE (prompt_id, keyword_id),
        FOREIGN KEY (prompt_id) REFERENCES Prompts(id) ON DELETE CASCADE,
        FOREIGN KEY (keyword_id) REFERENCES PromptKeywordsTable(id) ON DELETE CASCADE
    );

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
                      CREATE INDEX IF NOT EXISTS idx_prompts_name ON Prompts(name);
                      CREATE INDEX IF NOT EXISTS idx_prompts_author ON Prompts(author);
                      CREATE UNIQUE INDEX IF NOT EXISTS idx_prompts_uuid ON Prompts(uuid);
                      CREATE INDEX IF NOT EXISTS idx_prompts_last_modified ON Prompts(last_modified);
                      CREATE INDEX IF NOT EXISTS idx_prompts_usage_count ON Prompts(usage_count);
                      CREATE INDEX IF NOT EXISTS idx_prompts_last_used_at ON Prompts(last_used_at);
                      CREATE INDEX IF NOT EXISTS idx_prompts_deleted ON Prompts(deleted);

                      CREATE UNIQUE INDEX IF NOT EXISTS idx_promptkeywordstable_keyword ON PromptKeywordsTable(keyword);
                      CREATE UNIQUE INDEX IF NOT EXISTS idx_promptkeywordstable_uuid ON PromptKeywordsTable(uuid);
                      CREATE INDEX IF NOT EXISTS idx_promptkeywordstable_last_modified ON PromptKeywordsTable(last_modified);
                      CREATE INDEX IF NOT EXISTS idx_promptkeywordstable_deleted ON PromptKeywordsTable(deleted);

                      CREATE INDEX IF NOT EXISTS idx_promptkeywordlinks_prompt_id ON PromptKeywordLinks(prompt_id);
                      CREATE INDEX IF NOT EXISTS idx_promptkeywordlinks_keyword_id ON PromptKeywordLinks(keyword_id);

                      CREATE INDEX IF NOT EXISTS idx_sync_log_ts ON sync_log(timestamp);
                      CREATE INDEX IF NOT EXISTS idx_sync_log_entity_uuid ON sync_log(entity_uuid);
                      CREATE INDEX IF NOT EXISTS idx_sync_log_client_id ON sync_log(client_id); \
                      """

    _TRIGGERS_SQL_V1 = """
    DROP TRIGGER IF EXISTS prompts_validate_sync_update;
    CREATE TRIGGER prompts_validate_sync_update BEFORE UPDATE ON Prompts
    BEGIN
        SELECT RAISE(ABORT, 'Sync Error (Prompts): Version must increment by exactly 1.')
        WHERE NEW.version IS NOT OLD.version + 1;
        SELECT RAISE(ABORT, 'Sync Error (Prompts): Client ID cannot be NULL or empty.')
        WHERE NEW.client_id IS NULL OR NEW.client_id = '';
        SELECT RAISE(ABORT, 'Sync Error (Prompts): UUID cannot be changed.')
        WHERE NEW.uuid IS NOT OLD.uuid;
    END;

    DROP TRIGGER IF EXISTS promptkeywordstable_validate_sync_update;
    CREATE TRIGGER promptkeywordstable_validate_sync_update BEFORE UPDATE ON PromptKeywordsTable
    BEGIN
        SELECT RAISE(ABORT, 'Sync Error (PromptKeywordsTable): Version must increment by exactly 1.')
        WHERE NEW.version IS NOT OLD.version + 1;
        SELECT RAISE(ABORT, 'Sync Error (PromptKeywordsTable): Client ID cannot be NULL or empty.')
        WHERE NEW.client_id IS NULL OR NEW.client_id = '';
        SELECT RAISE(ABORT, 'Sync Error (PromptKeywordsTable): UUID cannot be changed.')
        WHERE NEW.uuid IS NOT OLD.uuid;
    END;
    """

    _FTS_TABLES_SQL = """
    CREATE VIRTUAL TABLE IF NOT EXISTS prompts_fts USING fts5(
        name,
        author,
        details,
        system_prompt,
        user_prompt,
        content='Prompts',
        content_rowid='id'
    );

    CREATE VIRTUAL TABLE IF NOT EXISTS prompt_keywords_fts USING fts5(
        keyword,
        content='PromptKeywordsTable',
        content_rowid='id'
    );
    """

    _COLLECTIONS_SQL_V2 = """
    CREATE TABLE IF NOT EXISTS PromptCollections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        uuid TEXT UNIQUE NOT NULL,
        created_at DATETIME NOT NULL,
        updated_at DATETIME NOT NULL
    );

    CREATE TABLE IF NOT EXISTS PromptCollectionItems (
        collection_id INTEGER NOT NULL,
        prompt_id INTEGER NOT NULL,
        sort_order INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (collection_id, prompt_id),
        FOREIGN KEY (collection_id) REFERENCES PromptCollections(id) ON DELETE CASCADE,
        FOREIGN KEY (prompt_id) REFERENCES Prompts(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_promptcollections_name ON PromptCollections(name);
    CREATE INDEX IF NOT EXISTS idx_promptcollectionitems_collection_order
        ON PromptCollectionItems(collection_id, sort_order, prompt_id);
    CREATE INDEX IF NOT EXISTS idx_promptcollectionitems_prompt_id
        ON PromptCollectionItems(prompt_id);
    """

    def __init__(self, db_path: Union[str, Path], client_id: str):
        """
        Initializes the PromptsDatabase instance, sets up the connection pool (via threading.local),
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
            raise ValueError("Client ID cannot be empty or None.")  # noqa: TRY003
        self.client_id = client_id

        # Ensure parent directory exists if it's a file-based DB
        if not self.is_memory_db:
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                # Catch potential errors creating the directory (e.g., permissions)
                raise DatabaseError(f"Failed to create database directory {self.db_path.parent}: {e}") from e  # noqa: TRY003

        logging.info(f"Initializing PromptsDatabase object for path: {self.db_path_str} [Client ID: {self.client_id}]")

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
            logging.critical(f"FATAL: Prompts DB Initialization failed for {self.db_path_str}: {e}", exc_info=True)
            # Attempt to clean up the connection before raising
            self.close_connection() # Important to call this if available
            # Re-raise as a DatabaseError to signal catastrophic failure
            raise DatabaseError(f"Prompts Database initialization failed: {e}") from e  # noqa: TRY003
        except Exception as e:
            # Catch any other unexpected errors during initialization
            logging.critical(f"FATAL: Unexpected error during Prompts DB Initialization for {self.db_path_str}: {e}", exc_info=True)
            # Attempt cleanup
            self.close_connection() # Important to call this
            # Re-raise as a DatabaseError
            raise DatabaseError(f"Unexpected prompts database initialization error: {e}") from e  # noqa: TRY003
        finally:
            # Log completion status based on the flag
            if initialization_successful:
                logging.debug(f"PromptsDatabase initialization completed successfully for {self.db_path_str}")
            else:
                # This path indicates an exception was caught and raised above.
                # Logging here provides context that the __init__ block finished, albeit with failure.
                logging.error(f"PromptsDatabase initialization block finished for {self.db_path_str}, but failed.")

    # --- Connection Management ---
    def _sqlite_journal_mode(self) -> str | None:
        if self.is_memory_db:
            return None
        return "WAL"

    def _get_thread_connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, 'conn', None)
        is_closed = True
        if conn:
            try:
                conn.execute("SELECT 1")
                is_closed = False
            except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                logging.warning(f"Thread-local connection to {self.db_path_str} was closed. Reopening.")
                is_closed = True
                try:
                    conn.close()
                except _PROMPTS_NONCRITICAL_EXCEPTIONS as e:
                    logging.warning(f"Failed to close database connection: {e}")
                self._local.conn = None

        if is_closed:
            try:
                conn = sqlite3.connect(
                    self.db_path_str,
                    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                    check_same_thread=False,  # Required for threading.local
                    timeout=1.0  # seconds; keep short to avoid long blocking on locks
                )
                conn.row_factory = sqlite3.Row
                journal_mode = self._sqlite_journal_mode()
                if journal_mode:
                    try:
                        conn.execute(f"PRAGMA journal_mode={journal_mode};")
                    except sqlite3.OperationalError as exc:
                        if "database is locked" not in str(exc).lower():
                            raise
                configure_sqlite_connection(
                    conn,
                    use_wal=False,
                    busy_timeout_ms=1000,
                )
                self._local.conn = conn
                logging.debug(
                    f"Opened/Reopened SQLite connection to {self.db_path_str} [Client: {self.client_id}, Thread: {threading.current_thread().name}]")
            except sqlite3.Error as e:
                logging.error(f"Failed to connect to database at {self.db_path_str}: {e}", exc_info=True)
                self._local.conn = None
                raise DatabaseError(f"Failed to connect to database '{self.db_path_str}': {e}") from e  # noqa: TRY003
        return self._local.conn

    def get_connection(self) -> sqlite3.Connection:
        return self._get_thread_connection()

    def close_connection(self):
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            try:
                conn = self._local.conn
                self._local.conn = None
                conn.close()
                logging.debug(f"Closed connection for thread {threading.current_thread().name}.")
            except sqlite3.Error as e:
                logging.warning(f"Error closing connection: {e}")
            finally:
                if hasattr(self._local, 'conn'): self._local.conn = None  # noqa: E701

    # Simple alias for test fixtures and callers expecting a generic close()
    def close(self):
        """Close any open SQLite connection held by this instance.

        Provided for compatibility with test fixtures that call `db.close()`.
        Internally delegates to `close_connection()` which manages the
        thread-local connection lifecycle.
        """
        try:
            self.close_connection()
        except _PROMPTS_NONCRITICAL_EXCEPTIONS as _e:
            # Be conservative: swallowing errors during close ensures test teardown
            # can proceed to unlink temporary files on platforms like Windows.
            logging.warning(f"PromptsDatabase.close() encountered an error: {_e}")

    def backup_database(self, backup_file_path: str) -> bool:
        """
        Creates a backup of the current database to the specified file path.

        Args:
            backup_file_path (str): The path to save the backup database file.

        Returns:
            bool: True if the backup was successful, False otherwise.
        """
        logger.info(f"Starting database backup from '{self.db_path_str}' to '{backup_file_path}'")
        backup_conn: Optional[sqlite3.Connection] = None
        try:
            # Ensure the backup file path is not the same as the source for file-based DBs
            if not self.is_memory_db and self.db_path.resolve() == Path(backup_file_path).resolve():
                logger.error("Backup path cannot be the same as the source database path.")
                raise ValueError("Backup path cannot be the same as the source database path.")  # noqa: TRY003, TRY301

            src_conn = self.get_connection()

            backup_db_path_obj = Path(backup_file_path)
            backup_db_path_obj.parent.mkdir(parents=True, exist_ok=True)

            backup_conn = sqlite3.connect(str(backup_db_path_obj))

            logger.debug(f"Source DB connection: {src_conn}")
            logger.debug(f"Backup DB connection: {backup_conn} to file {str(backup_db_path_obj)}")

            src_conn.backup(backup_conn, pages=0, progress=None)

            logger.info(f"Database backup successful from '{self.db_path_str}' to '{str(backup_db_path_obj)}'")
            return True  # noqa: TRY300
        except ValueError as ve:
            logger.error(f"ValueError during database backup: {ve}", exc_info=True)
            return False
        except sqlite3.Error as e:
            logger.error(f"SQLite error during database backup: {e}", exc_info=True)
            return False
        except _PROMPTS_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Unexpected error during database backup: {e}", exc_info=True)
            return False
        finally:
            if backup_conn:
                try:
                    backup_conn.close()
                    logger.debug("Closed backup database connection.")
                except sqlite3.Error as e:
                    logger.warning(f"Error closing backup database connection: {e}")
            # Source connection (src_conn) is managed by the thread-local mechanism.

    # --- Query Execution ---
    def execute_query(self, query: str, params: tuple = None, *, commit: bool = False) -> sqlite3.Cursor:
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            logging.debug(f"Executing Query: {query[:200]}... Params: {str(params)[:100]}...")
            cursor.execute(query, params or ())
            if commit:
                conn.commit()
                logging.debug("Committed.")
            return cursor  # noqa: TRY300
        except sqlite3.IntegrityError as e:
            msg = str(e).lower()
            if "sync error" in msg:  # From our custom triggers
                logging.error(f"Sync Validation Failed: {e}")
                raise
            else:
                logging.error(f"Integrity error: {query[:200]}... Error: {e}", exc_info=True)
                raise DatabaseError(f"Integrity constraint violation: {e}") from e  # noqa: TRY003
        except sqlite3.Error as e:
            logging.error(f"Query failed: {query[:200]}... Error: {e}", exc_info=True)
            raise DatabaseError(f"Query execution failed: {e}") from e  # noqa: TRY003

    def execute_many(self, query: str, params_list: list[tuple], *, commit: bool = False) -> Optional[sqlite3.Cursor]:
        conn = self.get_connection()
        if not isinstance(params_list, list):
            raise TypeError("params_list must be a list.")  # noqa: TRY003
        if not params_list:
            return None
        try:
            cursor = conn.cursor()
            logging.debug(f"Executing Many: {query[:150]}... with {len(params_list)} sets.")
            cursor.executemany(query, params_list)
            if commit:
                conn.commit()
                logging.debug("Committed Many.")
            return cursor  # noqa: TRY300
        except sqlite3.IntegrityError as e:
            logging.error(f"Integrity error during Execute Many: {query[:150]}... Error: {e}", exc_info=True)
            raise DatabaseError(f"Integrity constraint violation during batch: {e}") from e  # noqa: TRY003
        except sqlite3.Error as e:
            logging.error(f"Execute Many failed: {query[:150]}... Error: {e}", exc_info=True)
            raise DatabaseError(f"Execute Many failed: {e}") from e  # noqa: TRY003
        except TypeError as te:
            logging.error(f"TypeError during Execute Many: {te}. Check params_list format.", exc_info=True)
            raise TypeError(f"Parameter list format error: {te}") from te  # noqa: TRY003

    # --- Transaction Context ---
    @contextmanager
    def transaction(self):
        conn = self.get_connection()
        in_outer = conn.in_transaction
        try:
            if not in_outer:
                begin_immediate_if_needed(conn)
                logging.debug("Started transaction.")
            yield conn  # yield connection
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
            raise

    # --- Schema Initialization and Migration ---
    def _get_db_version(self, conn: sqlite3.Connection) -> int:
        try:
            cursor = conn.execute("SELECT version FROM schema_version LIMIT 1")
            result = cursor.fetchone()
            return result['version'] if result else 0
        except sqlite3.Error as e:
            if "no such table: schema_version" in str(e).lower():
                return 0
            else:
                raise DatabaseError(f"Could not determine schema version: {e}") from e  # noqa: TRY003

    _SCHEMA_UPDATE_VERSION_SQL_V1 = "UPDATE schema_version SET version = 1 WHERE version = 0;"
    _SCHEMA_UPDATE_VERSION_SQL_V2 = "UPDATE schema_version SET version = 2 WHERE version = 1;"
    _SCHEMA_UPDATE_VERSION_SQL_V3 = "UPDATE schema_version SET version = 3 WHERE version = 2;"
    _SCHEMA_UPDATE_VERSION_SQL_V4 = "UPDATE schema_version SET version = 4 WHERE version = 3;"
    _SCHEMA_UPDATE_VERSION_SQL_V5 = "UPDATE schema_version SET version = 5 WHERE version = 4;"

    def _apply_schema_v1(self, conn: sqlite3.Connection):
        logging.info(f"Applying initial schema (Version 1) to DB: {self.db_path_str}...")
        try:
            core_schema_script_with_version_update = f"""
                {self._TABLES_SQL_V1}
                {self._INDICES_SQL_V1}
                {self._TRIGGERS_SQL_V1}
                {self._SCHEMA_UPDATE_VERSION_SQL_V1}
            """
            with self.transaction():
                logging.debug("[Schema V1] Applying Core Schema + Version Update...")
                conn.executescript(core_schema_script_with_version_update)
                logging.debug("[Schema V1] Core Schema script (incl. version update) executed.")
                # Validation
                cursor = conn.execute("PRAGMA table_info(Prompts)")
                columns = {row['name'] for row in cursor.fetchall()}
                expected_cols = {
                    'id',
                    'name',
                    'author',
                    'details',
                    'system_prompt',
                    'user_prompt',
                    'uuid',
                    'last_modified',
                    'version',
                    'usage_count',
                    'last_used_at',
                    'client_id',
                    'deleted'
                }
                if not expected_cols.issubset(columns):
                    missing_cols = expected_cols - columns
                    raise SchemaError(f"Validation Error: Prompts table missing columns: {missing_cols}")  # noqa: TRY003
                logging.debug("[Schema V1] Prompts table structure validated.")
                cursor_check = conn.execute("SELECT version FROM schema_version LIMIT 1")
                version_in_tx = cursor_check.fetchone()
                if not version_in_tx or version_in_tx['version'] != 1:
                    raise SchemaError("Schema version update did not take effect within transaction.")  # noqa: TRY003
            logging.info(f"[Schema V1] Core Schema V1 applied and committed for DB: {self.db_path_str}.")
            try:
                logging.debug("[Schema V1] Applying FTS Tables...")
                conn.executescript(self._FTS_TABLES_SQL)
                conn.commit()  # Commit FTS creation separately
                logging.info("[Schema V1] FTS Tables created successfully.")
            except sqlite3.Error as fts_err:
                logging.error(f"[Schema V1] Failed to create FTS tables: {fts_err}", exc_info=True)
                # This might not be fatal if FTS is optional or can be rebuilt.
        except sqlite3.Error as e:
            logging.error(f"[Schema V1] Application failed: {e}", exc_info=True)
            raise DatabaseError(f"DB schema V1 setup failed: {e}") from e  # noqa: TRY003

    def _apply_schema_v2(self, conn: sqlite3.Connection):
        logging.info(f"Applying schema migration (Version 2) to DB: {self.db_path_str}...")
        try:
            migration_script = f"""
                {self._COLLECTIONS_SQL_V2}
                {self._SCHEMA_UPDATE_VERSION_SQL_V2}
            """
            with self.transaction():
                conn.executescript(migration_script)
                cursor_check = conn.execute("SELECT version FROM schema_version LIMIT 1")
                version_in_tx = cursor_check.fetchone()
                if not version_in_tx or version_in_tx["version"] != 2:
                    raise SchemaError("Schema V2 version update did not take effect within transaction.")  # noqa: TRY003
                collection_table = conn.execute("PRAGMA table_info(PromptCollections)").fetchall()
                item_table = conn.execute("PRAGMA table_info(PromptCollectionItems)").fetchall()
                if not collection_table or not item_table:
                    raise SchemaError("Schema V2 validation failed: collection tables missing.")  # noqa: TRY003
            logging.info(f"[Schema V2] Collection tables applied and committed for DB: {self.db_path_str}.")
        except sqlite3.Error as e:
            logging.error(f"[Schema V2] Application failed: {e}", exc_info=True)
            raise DatabaseError(f"DB schema V2 setup failed: {e}") from e  # noqa: TRY003

    def _apply_schema_v3(self, conn: sqlite3.Connection):
        logging.info(f"Applying schema migration (Version 3) to DB: {self.db_path_str}...")
        try:
            with self.transaction():
                existing_columns = {
                    row["name"] for row in conn.execute("PRAGMA table_info(Prompts)").fetchall()
                }
                if "usage_count" not in existing_columns:
                    conn.execute(
                        "ALTER TABLE Prompts ADD COLUMN usage_count INTEGER NOT NULL DEFAULT 0"
                    )
                if "last_used_at" not in existing_columns:
                    conn.execute("ALTER TABLE Prompts ADD COLUMN last_used_at DATETIME")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_prompts_usage_count ON Prompts(usage_count)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_prompts_last_used_at ON Prompts(last_used_at)"
                )
                conn.execute(self._SCHEMA_UPDATE_VERSION_SQL_V3)
                cursor_check = conn.execute("SELECT version FROM schema_version LIMIT 1")
                version_in_tx = cursor_check.fetchone()
                if not version_in_tx or version_in_tx["version"] != 3:
                    raise SchemaError("Schema V3 version update did not take effect within transaction.")  # noqa: TRY003
                prompts_columns = {
                    row["name"] for row in conn.execute("PRAGMA table_info(Prompts)").fetchall()
                }
                required = {"usage_count", "last_used_at"}
                missing = required - prompts_columns
                if missing:
                    raise SchemaError(
                        f"Schema V3 validation failed: missing prompt columns {sorted(missing)}."
                    )  # noqa: TRY003
            logging.info(f"[Schema V3] Usage tracking fields applied for DB: {self.db_path_str}.")
        except sqlite3.Error as e:
            logging.error(f"[Schema V3] Application failed: {e}", exc_info=True)
            raise DatabaseError(f"DB schema V3 setup failed: {e}") from e  # noqa: TRY003

    def _apply_schema_v4(self, conn: sqlite3.Connection):
        logging.info(f"Applying schema migration (Version 4) to DB: {self.db_path_str}...")
        try:
            with self.transaction():
                existing_columns = {
                    row["name"] for row in conn.execute("PRAGMA table_info(Prompts)").fetchall()
                }
                if "prompt_format" not in existing_columns:
                    conn.execute(
                        "ALTER TABLE Prompts ADD COLUMN prompt_format TEXT NOT NULL DEFAULT 'legacy'"
                    )
                if "prompt_schema_version" not in existing_columns:
                    conn.execute("ALTER TABLE Prompts ADD COLUMN prompt_schema_version INTEGER")
                if "prompt_definition_json" not in existing_columns:
                    conn.execute("ALTER TABLE Prompts ADD COLUMN prompt_definition_json TEXT")
                conn.execute(self._SCHEMA_UPDATE_VERSION_SQL_V4)
                cursor_check = conn.execute("SELECT version FROM schema_version LIMIT 1")
                version_in_tx = cursor_check.fetchone()
                if not version_in_tx or version_in_tx["version"] != 4:
                    raise SchemaError("Schema V4 version update did not take effect within transaction.")  # noqa: TRY003
                prompts_columns = {
                    row["name"] for row in conn.execute("PRAGMA table_info(Prompts)").fetchall()
                }
                required = {"prompt_format", "prompt_schema_version", "prompt_definition_json"}
                missing = required - prompts_columns
                if missing:
                    raise SchemaError(
                        f"Schema V4 validation failed: missing prompt columns {sorted(missing)}."
                    )  # noqa: TRY003
            logging.info(f"[Schema V4] Structured prompt fields applied for DB: {self.db_path_str}.")
        except sqlite3.Error as e:
            logging.error(f"[Schema V4] Application failed: {e}", exc_info=True)
            raise DatabaseError(f"DB schema V4 setup failed: {e}") from e  # noqa: TRY003

    def _apply_schema_v5(self, conn: sqlite3.Connection):
        logging.info(f"Applying schema migration (Version 5) to DB: {self.db_path_str}...")
        try:
            with self.transaction():
                conn.execute(self._SCHEMA_UPDATE_VERSION_SQL_V5)
                cursor_check = conn.execute("SELECT version FROM schema_version LIMIT 1")
                version_in_tx = cursor_check.fetchone()
                if not version_in_tx or version_in_tx["version"] != 5:
                    raise SchemaError("Schema V5 version update did not take effect within transaction.")  # noqa: TRY003
                self._rebuild_prompts_fts(conn)
            logging.info(f"[Schema V5] Prompt FTS index rebuilt for DB: {self.db_path_str}.")
        except sqlite3.Error as e:
            logging.error(f"[Schema V5] Application failed: {e}", exc_info=True)
            raise DatabaseError(f"DB schema V5 setup failed: {e}") from e  # noqa: TRY003

    def _initialize_schema(self):
        conn = self.get_connection()
        try:
            current_db_version = self._get_db_version(conn)
            target_version = self._CURRENT_SCHEMA_VERSION
            logging.info(f"Checking DB schema. Current: {current_db_version}, Code supports: {target_version}")

            if current_db_version > target_version:
                raise SchemaError(  # noqa: TRY003, TRY301
                    f"DB schema version ({current_db_version}) is newer than supported ({target_version}).")

            while current_db_version < target_version:
                if current_db_version == 0:
                    self._apply_schema_v1(conn)
                    current_db_version = self._get_db_version(conn)
                    continue
                if current_db_version == 1:
                    self._apply_schema_v2(conn)
                    current_db_version = self._get_db_version(conn)
                    continue
                if current_db_version == 2:
                    self._apply_schema_v3(conn)
                    current_db_version = self._get_db_version(conn)
                    continue
                if current_db_version == 3:
                    self._apply_schema_v4(conn)
                    current_db_version = self._get_db_version(conn)
                    continue
                if current_db_version == 4:
                    self._apply_schema_v5(conn)
                    current_db_version = self._get_db_version(conn)
                    continue
                raise SchemaError(  # noqa: TRY003
                    f"Migration needed from {current_db_version} to {target_version}, but no path defined."
                )

            if current_db_version != target_version:
                raise SchemaError(  # noqa: TRY003
                    f"Schema migration applied, but final DB version is {current_db_version}, expected {target_version}."
                )

            logging.info(f"Database schema initialized/migrated to version {target_version}.")
            try:
                conn.executescript(self._FTS_TABLES_SQL)
                conn.commit()
                logging.debug("Verified FTS tables exist.")
            except sqlite3.Error as fts_err:
                logging.warning(f"Could not verify/create FTS tables on correct schema: {fts_err}")
        except (DatabaseError, SchemaError, sqlite3.Error) as e:
            logging.error(f"Schema initialization/migration failed: {e}", exc_info=True)
            raise DatabaseError(f"Schema initialization failed: {e}") from e  # noqa: TRY003

    # --- Internal Helpers ---
    def _get_current_utc_timestamp_str(self) -> str:
        return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    def _generate_uuid(self) -> str:
        return str(uuid.uuid4())

    _serialize_prompt_definition = staticmethod(serialize_prompt_definition)
    _deserialize_prompt_record = staticmethod(deserialize_prompt_record)
    build_structured_prompt_searchable_text = staticmethod(build_structured_prompt_searchable_text)

    def _build_fts_details_text(self, details: Optional[str], prompt_definition: Any) -> str:
        detail_parts: list[str] = []
        if isinstance(details, str) and details.strip():
            detail_parts.append(details.strip())

        structured_text = self.build_structured_prompt_searchable_text(prompt_definition)
        if structured_text:
            detail_parts.append(structured_text)

        return "\n\n".join(detail_parts)

    _normalize_keyword = staticmethod(normalize_keyword)
    _normalize_text_for_search = staticmethod(normalize_text_for_search)

    def _get_next_version(self, conn: sqlite3.Connection, table: str, id_col: str, id_val: Any) -> Optional[
        tuple[int, int]]:
        try:
            if not (_SAFE_IDENTIFIER_RE.fullmatch(table or "") and _SAFE_IDENTIFIER_RE.fullmatch(id_col or "")):
                raise DatabaseError(  # noqa: TRY003
                    f"Unsafe identifier in version lookup: table={table!r}, column={id_col!r}"
                )
            cursor = conn.execute(f"SELECT version FROM {table} WHERE {id_col} = ? AND deleted = 0", (id_val,))  # nosec B608
            result = cursor.fetchone()
            if result:
                current_version = result['version']
                if isinstance(current_version, int):
                    return current_version, current_version + 1
                else:
                    logging.error(f"Invalid non-integer version '{current_version}' for {table} {id_col}={id_val}")
                    return None
        except sqlite3.Error as e:
            logging.error(f"DB error fetching version for {table} {id_col}={id_val}: {e}")
            raise DatabaseError(f"Failed to fetch current version: {e}") from e  # noqa: TRY003
        return None

    def _log_sync_event(self, conn: sqlite3.Connection, entity: str, entity_uuid: str, operation: str, version: int,
                        payload: Optional[dict] = None):
        if not entity or not entity_uuid or not operation:
            logging.error("Sync log attempt with missing entity, uuid, or operation.")
            return
        current_time = self._get_current_utc_timestamp_str()
        client_id = self.client_id
        payload_json = json.dumps(payload, separators=(',', ':')) if payload else None
        try:
            conn.execute("""
                         INSERT INTO sync_log (entity, entity_uuid, operation, timestamp, client_id, version, payload)
                         VALUES (?, ?, ?, ?, ?, ?, ?)
                         """, (entity, entity_uuid, operation, current_time, client_id, version, payload_json))
            logging.debug(f"Logged sync: {entity} {entity_uuid} {operation} v{version} at {current_time}")
        except sqlite3.Error as e:
            logging.error(f"Failed insert sync_log for {entity} {entity_uuid}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to log sync event: {e}") from e  # noqa: TRY003

    # --- FTS Helper Methods ---
    def _update_fts_prompt(self, conn: sqlite3.Connection, prompt_id: int, name: str, author: Optional[str],
                           details: Optional[str], system_prompt: Optional[str], user_prompt: Optional[str],
                           prompt_definition: Any = None):
        try:
            fts_details = self._build_fts_details_text(details, prompt_definition)
            conn.execute(
                "INSERT OR REPLACE INTO prompts_fts (rowid, name, author, details, system_prompt, user_prompt) VALUES (?, ?, ?, ?, ?, ?)",
                (prompt_id, name, author or "", fts_details, system_prompt or "", user_prompt or ""))
            logging.debug(f"Updated FTS for Prompt ID {prompt_id}")
        except sqlite3.Error as e:
            logging.error(f"Failed FTS update Prompt ID {prompt_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed FTS update Prompt ID {prompt_id}: {e}") from e  # noqa: TRY003

    def _rebuild_prompts_fts(self, conn: sqlite3.Connection) -> None:
        try:
            conn.executescript(self._FTS_TABLES_SQL)
            conn.execute("DELETE FROM prompts_fts")
            cursor = conn.execute(
                """
                SELECT id, name, author, details, system_prompt, user_prompt, prompt_definition_json
                FROM Prompts
                WHERE deleted = 0
                """
            )
            for row in cursor.fetchall():
                self._update_fts_prompt(
                    conn,
                    row["id"],
                    row["name"],
                    row["author"],
                    row["details"],
                    row["system_prompt"],
                    row["user_prompt"],
                    row["prompt_definition_json"],
                )
        except sqlite3.Error as e:
            logging.error(f"Failed to rebuild prompt FTS index: {e}", exc_info=True)
            raise DatabaseError(f"Failed to rebuild prompt FTS index: {e}") from e  # noqa: TRY003

    def _delete_fts_prompt(self, conn: sqlite3.Connection, prompt_id: int):
        try:
            conn.execute("DELETE FROM prompts_fts WHERE rowid = ?", (prompt_id,))
            logging.debug(f"Deleted FTS for Prompt ID {prompt_id}")
        except sqlite3.Error as e:
            logging.error(f"Failed FTS delete Prompt ID {prompt_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed FTS delete Prompt ID {prompt_id}: {e}") from e  # noqa: TRY003

    def _update_fts_prompt_keyword(self, conn: sqlite3.Connection, keyword_id: int, keyword: str):
        try:
            conn.execute("INSERT OR REPLACE INTO prompt_keywords_fts (rowid, keyword) VALUES (?, ?)",
                         (keyword_id, keyword))
            logging.debug(f"Updated FTS for PromptKeyword ID {keyword_id}")
        except sqlite3.Error as e:
            logging.error(f"Failed FTS update PromptKeyword ID {keyword_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed FTS update PromptKeyword ID {keyword_id}: {e}") from e  # noqa: TRY003

    # --- Version History ---
    def _fetch_prompt_versions_from_sync_log(self, prompt_uuid: str) -> list[dict[str, Any]]:
        """Build version entries for a prompt using sync_log snapshots."""
        if not prompt_uuid:
            return []
        try:
            cursor = self.execute_query(
                """SELECT change_id, version, timestamp, payload
                   FROM sync_log
                   WHERE entity = 'Prompts'
                     AND entity_uuid = ?
                     AND operation IN ('create', 'update')
                   ORDER BY version ASC, change_id ASC""",
                (prompt_uuid,)
            )
            versions_by_number: dict[int, dict[str, Any]] = {}
            for row in cursor.fetchall():
                row_dict = dict(row)
                ver = int(row_dict['version']) if isinstance(row_dict.get('version'), (int,)) else None
                if ver is None or ver in versions_by_number:
                    continue
                payload_raw = row_dict.get('payload')
                payload: Optional[dict[str, Any]] = None
                if payload_raw:
                    try:
                        payload = json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
                    except json.JSONDecodeError:
                        payload = None
                entry: dict[str, Any] = {
                    "version": ver,
                    "created_at": row_dict.get("timestamp"),
                    "comment": None,
                }
                if isinstance(payload, dict):
                    entry.update({
                        "name": payload.get("name"),
                        "author": payload.get("author"),
                        "details": payload.get("details"),
                        "system_prompt": payload.get("system_prompt"),
                        "user_prompt": payload.get("user_prompt"),
                        "prompt_format": payload.get("prompt_format"),
                        "prompt_schema_version": payload.get("prompt_schema_version"),
                        "prompt_definition": payload.get("prompt_definition"),
                    })
                versions_by_number[ver] = entry
            return [versions_by_number[v] for v in sorted(versions_by_number)]
        except (DatabaseError, sqlite3.Error) as e:
            logging.error(f"Error fetching version history for prompt UUID {prompt_uuid}: {e}")
            return []

    def get_prompt_versions(self, prompt_id: int) -> list[dict[str, Any]]:
        """Return version history for a prompt, enriched by sync_log when available."""
        try:
            prompt = self.fetch_prompt_details(prompt_id, include_deleted=True)
            if not prompt:
                return []
            versions = self._fetch_prompt_versions_from_sync_log(prompt.get("uuid"))
            if versions:
                return versions
            current_version = int(prompt.get("version", 1)) if isinstance(prompt.get("version"), (int,)) else 1
            return [
                {"version": v, "created_at": None, "comment": None}
                for v in range(1, max(1, current_version) + 1)
            ]
        except (DatabaseError, sqlite3.Error) as e:
            logging.error(f"Error building version history for prompt {prompt_id}: {e}")
            return []

    def _fetch_prompt_version_payload(self, prompt_uuid: str, version: int) -> Optional[dict[str, Any]]:
        """Fetch a specific prompt version payload from sync_log."""
        if not prompt_uuid:
            return None
        try:
            cursor = self.execute_query(
                """SELECT payload
                   FROM sync_log
                   WHERE entity = 'Prompts'
                     AND entity_uuid = ?
                     AND operation IN ('create', 'update')
                     AND version = ?
                   ORDER BY change_id ASC
                   LIMIT 1""",
                (prompt_uuid, version)
            )
            row = cursor.fetchone()
            if not row:
                return None
            row_dict = dict(row)
            payload_raw = row_dict.get('payload')
            if not payload_raw:
                return None
            if isinstance(payload_raw, dict):
                return payload_raw
            try:
                return json.loads(payload_raw)
            except json.JSONDecodeError:
                return None
        except (DatabaseError, sqlite3.Error) as e:
            logging.error(f"Error fetching prompt version payload for {prompt_uuid} v{version}: {e}")
            return None

    def restore_prompt_version(self, prompt_id: int, version: int) -> tuple[Optional[str], str]:
        """Restore a prompt to a previous version using sync_log snapshots."""
        if not isinstance(version, int) or version < 1:
            raise InputError("Version must be a positive integer.")  # noqa: TRY003
        prompt = self.fetch_prompt_details(prompt_id, include_deleted=True)
        if not prompt:
            raise InputError(f"Prompt with ID {prompt_id} not found.")  # noqa: TRY003
        payload = self._fetch_prompt_version_payload(prompt.get("uuid"), version)
        if not payload:
            raise InputError(f"Version {version} not found for prompt {prompt_id}.")  # noqa: TRY003

        update_data: dict[str, Any] = {}
        for field in (
            "name",
            "author",
            "details",
            "system_prompt",
            "user_prompt",
            "prompt_format",
            "prompt_schema_version",
            "prompt_definition",
        ):
            if field in payload:
                update_data[field] = payload.get(field)
        if "keywords" in payload and isinstance(payload.get("keywords"), list):
            update_data["keywords"] = payload.get("keywords")

        if not update_data:
            return None, f"No snapshot data available for version {version}."

        return self.update_prompt_by_id(int(prompt.get("id")), update_data)

    def _delete_fts_prompt_keyword(self, conn: sqlite3.Connection, keyword_id: int):
        try:
            conn.execute("DELETE FROM prompt_keywords_fts WHERE rowid = ?", (keyword_id,))
            logging.debug(f"Deleted FTS for PromptKeyword ID {keyword_id}")
        except sqlite3.Error as e:
            logging.error(f"Failed FTS delete PromptKeyword ID {keyword_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed FTS delete PromptKeyword ID {keyword_id}: {e}") from e  # noqa: TRY003

    # --- Public Mutating Methods ---
    def add_keyword(self, keyword_text: str) -> tuple[Optional[int], Optional[str]]:
        if not keyword_text or not keyword_text.strip():
            raise InputError("Keyword cannot be empty.")  # noqa: TRY003
        normalized_keyword = self._normalize_keyword(keyword_text)
        current_time = self._get_current_utc_timestamp_str()
        client_id = self.client_id

        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id, uuid, deleted, version FROM PromptKeywordsTable WHERE keyword = ?',
                               (normalized_keyword,))
                existing = cursor.fetchone()

                if existing:
                    kw_id, kw_uuid, is_deleted, current_version = existing['id'], existing['uuid'], existing['deleted'], \
                        existing['version']
                    if is_deleted:  # Undelete
                        new_version = current_version + 1
                        cursor.execute(
                            "UPDATE PromptKeywordsTable SET deleted=0, last_modified=?, version=?, client_id=? WHERE id=? AND version=?",
                            (current_time, new_version, client_id, kw_id, current_version))
                        if cursor.rowcount == 0: raise ConflictError(  # noqa: E701, TRY003, TRY301
                            "Failed to undelete keyword due to version mismatch or it was not found.",
                            "PromptKeywordsTable", kw_id)
                        cursor.execute("SELECT * FROM PromptKeywordsTable WHERE id=?", (kw_id,))
                        payload = dict(cursor.fetchone())
                        self._log_sync_event(conn, 'PromptKeywordsTable', kw_uuid, 'update', new_version, payload)
                        self._update_fts_prompt_keyword(conn, kw_id, normalized_keyword)
                        return kw_id, kw_uuid
                    else:  # Already active, just return its ID and UUID
                        logger.debug(
                            f"Keyword '{normalized_keyword}' already exists and is active. Reusing ID: {kw_id}, UUID: {kw_uuid}")
                        return kw_id, kw_uuid
                else:  # New keyword
                    new_uuid = self._generate_uuid()
                    new_version = 1
                    cursor.execute(
                        "INSERT INTO PromptKeywordsTable (keyword, uuid, last_modified, version, client_id, deleted) VALUES (?, ?, ?, ?, ?, 0)",
                        (normalized_keyword, new_uuid, current_time, new_version, client_id))
                    kw_id = cursor.lastrowid
                    if not kw_id: raise DatabaseError("Failed to get ID for new prompt keyword.")  # noqa: E701, TRY003, TRY301
                    cursor.execute("SELECT * FROM PromptKeywordsTable WHERE id=?", (kw_id,))
                    payload = dict(cursor.fetchone())
                    self._log_sync_event(conn, 'PromptKeywordsTable', new_uuid, 'create', new_version, payload)
                    self._update_fts_prompt_keyword(conn, kw_id, normalized_keyword)
                    return kw_id, new_uuid
        except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error in add_keyword (prompt) for '{keyword_text}': {e}", exc_info=True)
            if isinstance(e, (InputError, ConflictError, DatabaseError)):
                raise
            else:
                raise DatabaseError(f"Failed to add/update prompt keyword: {e}") from e  # noqa: TRY003

    def get_active_keyword_by_text(self, keyword_text: str) -> Optional[dict]:
        """
        Fetches an active (not deleted) keyword by its exact normalized text.

        Args:
            keyword_text: The keyword text to search for.

        Returns:
            A dictionary of the keyword's data if found and active, else None.
        """
        if not keyword_text or not keyword_text.strip():
            return None  # Or raise InputError if strictness is preferred here
        normalized_keyword = self._normalize_keyword(keyword_text)
        # Case-insensitive exact match using NOCASE collation
        query = "SELECT id, uuid, keyword, last_modified, version, client_id FROM PromptKeywordsTable WHERE keyword = ? COLLATE NOCASE AND deleted = 0"
        try:
            cursor = self.execute_query(query, (normalized_keyword,))
            result = cursor.fetchone()
            return dict(result) if result else None
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching active keyword by text '{normalized_keyword}': {e}")
            # Depending on desired strictness, could raise or return None
            # For a simple check, returning None on error is acceptable if the next step handles it.
            return None

    def add_prompt(self, name: str, author: Optional[str], details: Optional[str],
                   system_prompt: Optional[str] = None, user_prompt: Optional[str] = None,
                   prompt_format: str = "legacy",
                   prompt_schema_version: Optional[int] = None,
                   prompt_definition: Optional[Any] = None,
                   keywords: Optional[list[str]] = None, overwrite: bool = False) -> tuple[
        Optional[int], Optional[str], str]:
        if not isinstance(name, str) or name == "":
            raise InputError("Prompt name cannot be empty.")  # noqa: TRY003

        current_time = self._get_current_utc_timestamp_str()
        client_id = self.client_id
        prompt_definition_json = self._serialize_prompt_definition(prompt_definition)

        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, uuid, version, deleted FROM Prompts WHERE name = ?", (name,))
                existing = cursor.fetchone()

                prompt_id: Optional[int] = None
                prompt_uuid: Optional[str] = None
                action_taken: str = "skipped"

                if existing:
                    prompt_id, prompt_uuid, current_version, is_deleted = existing['id'], existing['uuid'], existing[
                        'version'], existing['deleted']
                    if is_deleted and not overwrite:  # Soft-deleted, treat as "exists" if not overwriting
                        return prompt_id, prompt_uuid, f"Prompt '{name}' exists but is soft-deleted. Use overwrite to restore/update."
                    if not overwrite and not is_deleted:
                        raise ConflictError(f"Prompt '{name}' already exists.")  # RAISE ERROR  # noqa: TRY003, TRY301
                        #return prompt_id, prompt_uuid, f"Prompt '{name}' already exists. Skipped."

                    # Overwrite or undelete-and-update
                    action_taken = "updated"
                    new_version = current_version + 1
                    update_data = {
                        'name': name, 'author': author, 'details': details, 'system_prompt': system_prompt,
                        'user_prompt': user_prompt,
                        'prompt_format': prompt_format,
                        'prompt_schema_version': prompt_schema_version,
                        'prompt_definition': prompt_definition,
                        'last_modified': current_time, 'version': new_version, 'client_id': client_id, 'deleted': 0,
                        'uuid': prompt_uuid
                    }
                    cursor.execute("""UPDATE Prompts
                                      SET author=?,
                                          details=?,
                                          system_prompt=?,
                                          user_prompt=?,
                                          prompt_format=?,
                                          prompt_schema_version=?,
                                          prompt_definition_json=?,
                                          last_modified=?,
                                          version=?,
                                          client_id=?,
                                          deleted=0
                                      WHERE id = ?
                                        AND version = ?""",
                                   (author, details, system_prompt, user_prompt, prompt_format, prompt_schema_version,
                                    prompt_definition_json, current_time, new_version, client_id,
                                    prompt_id, current_version))
                    if cursor.rowcount == 0:
                        # If it was deleted and overwrite is true, version check might fail if version wasn't for active.
                        # Or, a concurrent update happened.
                        # Re-fetch to check if it was deleted to adjust error message
                        cursor.execute("SELECT deleted, version FROM Prompts WHERE id=?", (prompt_id,))
                        refetched = cursor.fetchone()
                        if refetched and refetched['deleted'] and refetched['version'] == current_version:
                            # This means it was soft-deleted, and we tried to update with old version.
                            # We need to increment from its current soft-deleted version.
                            # For simplicity, we'll just tell user to handle undelete separately or ensure version matches.
                            # A more complex undelete+update would fetch its true current version first.
                            raise ConflictError(  # noqa: TRY003, TRY301
                                f"Prompt '{name}' (ID: {prompt_id}) was soft-deleted. Undelete first or ensure overwrite logic handles versioning correctly.",
                                "Prompts", prompt_id)
                        raise ConflictError(f"Failed to update prompt '{name}'.", "Prompts", prompt_id)  # noqa: TRY003, TRY301

                    self._log_sync_event(conn, 'Prompts', prompt_uuid, 'update', new_version, update_data)
                    self._update_fts_prompt(
                        conn,
                        prompt_id,
                        name,
                        author,
                        details,
                        system_prompt,
                        user_prompt,
                        prompt_definition,
                    )
                else:  # New prompt
                    action_taken = "added"
                    prompt_uuid = self._generate_uuid()
                    new_version = 1
                    insert_data = {
                        'name': name, 'author': author, 'details': details, 'system_prompt': system_prompt,
                        'user_prompt': user_prompt,
                        'prompt_format': prompt_format,
                        'prompt_schema_version': prompt_schema_version,
                        'prompt_definition': prompt_definition,
                        'usage_count': 0, 'last_used_at': None,
                        'uuid': prompt_uuid, 'last_modified': current_time, 'version': new_version,
                        'client_id': client_id, 'deleted': 0
                    }
                    cursor.execute(
                        """INSERT INTO Prompts (
                            name,
                            author,
                            details,
                            system_prompt,
                            user_prompt,
                            prompt_format,
                            prompt_schema_version,
                            prompt_definition_json,
                            usage_count,
                            last_used_at,
                            uuid,
                            last_modified,
                            version,
                            client_id,
                            deleted
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                        (
                            name,
                            author,
                            details,
                            system_prompt,
                            user_prompt,
                            prompt_format,
                            prompt_schema_version,
                            prompt_definition_json,
                            0,
                            None,
                            prompt_uuid,
                            current_time,
                            new_version,
                            client_id,
                        ),
                    )
                    prompt_id = cursor.lastrowid
                    if not prompt_id: raise DatabaseError("Failed to get ID for new prompt.")  # noqa: E701, TRY003, TRY301
                    self._log_sync_event(conn, 'Prompts', prompt_uuid, 'create', new_version, insert_data)
                    self._update_fts_prompt(
                        conn,
                        prompt_id,
                        name,
                        author,
                        details,
                        system_prompt,
                        user_prompt,
                        prompt_definition,
                    )

                if prompt_id:
                    # Apply provided keywords only; do not inject a default tag when none provided
                    eff_keywords = [k for k in (keywords or []) if isinstance(k, str)]
                    self.update_keywords_for_prompt(prompt_id, keywords_list=eff_keywords) # This is an instance method

                msg = f"Prompt '{name}' {action_taken} successfully."
                return prompt_id, prompt_uuid, msg

        except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error adding/updating prompt '{name}': {e}", exc_info=True)
            if isinstance(e, (InputError, ConflictError, DatabaseError)): raise  # noqa: E701
            else: raise DatabaseError(f"Failed to process prompt '{name}': {e}") from e  # noqa: E701, TRY003

    def update_keywords_for_prompt(self, prompt_id: int, keywords_list: list[str]):
        normalized_new_keywords = sorted({
            self._normalize_keyword(k) for k in keywords_list if k and k.strip()
        })
        # Do NOT auto-add a default keyword when clearing; allow empty keyword sets

        try:
            # This method is called within an existing transaction (e.g. from add_prompt)
            # So, use self.get_connection() but don't start a new transaction here.
            conn = self.get_connection()
            cursor = conn.cursor()

            # Get prompt_uuid for logging
            cursor.execute("SELECT uuid FROM Prompts WHERE id = ? AND deleted = 0", (prompt_id,))
            prompt_info = cursor.fetchone()
            if not prompt_info:
                raise InputError(f"Cannot update keywords: Prompt ID {prompt_id} not found or deleted.")  # noqa: TRY003, TRY301
            prompt_uuid = prompt_info['uuid']

            # Get current keywords for the prompt
            cursor.execute("""
                           SELECT pkl.keyword_id, pkw.keyword, pkw.uuid as keyword_uuid
                           FROM PromptKeywordLinks pkl
                                    JOIN PromptKeywordsTable pkw ON pkl.keyword_id = pkw.id
                           WHERE pkl.prompt_id = ? AND pkw.deleted = 0
                           """, (prompt_id,))
            current_keyword_links = {row['keyword_id']: {'text': row['keyword'], 'uuid': row['keyword_uuid']} for row in cursor.fetchall()}
            current_keyword_ids = set(current_keyword_links.keys())

            target_keyword_data: dict[int, dict[str,str]] = {} # {keyword_id: {'text': text, 'uuid': uuid}}
            if normalized_new_keywords:
                for kw_text in normalized_new_keywords:
                    # add_keyword is an instance method, it will use the existing transaction
                    kw_id, kw_uuid = self.add_keyword(kw_text)
                    if kw_id and kw_uuid:
                        target_keyword_data[kw_id] = {'text': kw_text, 'uuid': kw_uuid}
                    else:
                        # This should not happen if add_keyword is robust
                        raise DatabaseError(f"Failed to get/add keyword '{kw_text}' during prompt keyword update.")  # noqa: TRY003, TRY301

            target_keyword_ids = set(target_keyword_data.keys())

            ids_to_add = target_keyword_ids - current_keyword_ids
            ids_to_remove = current_keyword_ids - target_keyword_ids
            link_sync_version = 1 # For link/unlink operations, version is on the junction table itself if it had one, or just 1 for the event

            if ids_to_remove:
                remove_placeholders = ','.join('?' * len(ids_to_remove))
                cursor.execute(f"DELETE FROM PromptKeywordLinks WHERE prompt_id = ? AND keyword_id IN ({remove_placeholders})", (prompt_id, *list(ids_to_remove)))  # nosec B608
                for removed_id in ids_to_remove:
                    keyword_uuid = current_keyword_links[removed_id]['uuid']
                    link_composite_uuid = f"{prompt_uuid}_{keyword_uuid}" # Composite UUID for the link
                    payload = {'prompt_uuid': prompt_uuid, 'keyword_uuid': keyword_uuid}
                    self._log_sync_event(conn, 'PromptKeywordLinks', link_composite_uuid, 'unlink', link_sync_version, payload)

            if ids_to_add:
                insert_params = [(prompt_id, kid) for kid in ids_to_add]
                cursor.executemany("INSERT OR IGNORE INTO PromptKeywordLinks (prompt_id, keyword_id) VALUES (?, ?)", insert_params)
                for added_id in ids_to_add:
                    keyword_uuid = target_keyword_data[added_id]['uuid']
                    link_composite_uuid = f"{prompt_uuid}_{keyword_uuid}"
                    payload = {'prompt_uuid': prompt_uuid, 'keyword_uuid': keyword_uuid}
                    self._log_sync_event(conn, 'PromptKeywordLinks', link_composite_uuid, 'link', link_sync_version, payload)

            if ids_to_add or ids_to_remove:
                logging.debug(f"Keywords updated for prompt {prompt_id}. Added: {len(ids_to_add)}, Removed: {len(ids_to_remove)}.")
        except (InputError, DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error updating keywords for prompt {prompt_id}: {e}", exc_info=True)
            if isinstance(e, (InputError, DatabaseError)): raise  # noqa: E701
            else: raise DatabaseError(f"Keyword update failed for prompt {prompt_id}: {e}") from e  # noqa: E701, TRY003

    def update_prompt_by_id(self, prompt_id: int, update_data: dict[str, Any]) -> tuple[Optional[str], str]:
        """
        Updates an existing prompt identified by its ID.
        Handles name changes and ensures the new name doesn't conflict with other existing prompts.

        Args:
            prompt_id: The ID of the prompt to update.
            update_data: A dictionary containing fields to update (name, author, details, system_prompt, user_prompt).
                         Keywords are handled separately by `update_keywords_for_prompt`.

        Returns:
            A tuple (updated_prompt_uuid, message_string).

        Raises:
            InputError: If required fields like 'name' are missing or invalid in update_data.
            ConflictError: If a name change conflicts with another existing prompt, or version mismatch.
            DatabaseError: For other database issues.
        """
        if 'name' in update_data and (not update_data['name'] or not update_data['name'].strip()):
            raise InputError("Prompt name cannot be empty if provided for update.")  # noqa: TRY003

        current_time = self._get_current_utc_timestamp_str()
        client_id = self.client_id

        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                # Get current state of the prompt being updated
                cursor.execute("SELECT uuid, name, version, deleted FROM Prompts WHERE id = ?", (prompt_id,))
                existing_prompt_state = cursor.fetchone()

                if not existing_prompt_state:
                    return None, f"Prompt with ID {prompt_id} not found."  # Or raise InputError("Prompt not found")

                original_uuid = existing_prompt_state['uuid']
                original_name = existing_prompt_state['name']
                current_version = existing_prompt_state['version']
                is_deleted = existing_prompt_state['deleted']

                if is_deleted:  # Optional: decide if updating a soft-deleted prompt should undelete it.
                    # For now, let's assume we are updating an active prompt or an explicitly fetched soft-deleted one.
                    # If this method should also undelete, set 'deleted = 0' in the update.
                    pass

                new_name = update_data.get('name', original_name).strip()

                # If name is changing, check for conflict with *other* prompts
                if new_name != original_name:
                    cursor.execute("SELECT id FROM Prompts WHERE name = ? AND id != ? AND deleted = 0",
                                   (new_name, prompt_id))
                    conflicting_prompt = cursor.fetchone()
                    if conflicting_prompt:
                        raise ConflictError(  # noqa: TRY003, TRY301
                            f"Another active prompt with name '{new_name}' already exists (ID: {conflicting_prompt['id']}).")

                new_version = current_version + 1

                set_clauses = []
                params = []

                # Build SET clause dynamically
                if 'name' in update_data and update_data['name'].strip() != original_name:  # Only if actually changing
                    set_clauses.append("name = ?")
                    params.append(new_name)
                if 'author' in update_data:
                    set_clauses.append("author = ?")
                    params.append(update_data.get('author'))
                if 'details' in update_data:
                    set_clauses.append("details = ?")
                    params.append(update_data.get('details'))
                if 'system_prompt' in update_data:
                    set_clauses.append("system_prompt = ?")
                    params.append(update_data.get('system_prompt'))
                if 'user_prompt' in update_data:
                    set_clauses.append("user_prompt = ?")
                    params.append(update_data.get('user_prompt'))
                if 'prompt_format' in update_data:
                    set_clauses.append("prompt_format = ?")
                    params.append(update_data.get('prompt_format') or 'legacy')
                if 'prompt_schema_version' in update_data:
                    set_clauses.append("prompt_schema_version = ?")
                    params.append(update_data.get('prompt_schema_version'))
                if 'prompt_definition' in update_data:
                    set_clauses.append("prompt_definition_json = ?")
                    params.append(self._serialize_prompt_definition(update_data.get('prompt_definition')))
                if 'usage_count' in update_data:
                    usage_count = update_data.get('usage_count')
                    if usage_count is not None:
                        try:
                            usage_count = int(usage_count)
                        except (TypeError, ValueError) as exc:
                            raise InputError("usage_count must be an integer.") from exc  # noqa: TRY003
                        if usage_count < 0:
                            raise InputError("usage_count cannot be negative.")  # noqa: TRY003
                    else:
                        usage_count = 0
                    set_clauses.append("usage_count = ?")
                    params.append(usage_count)
                if 'last_used_at' in update_data:
                    set_clauses.append("last_used_at = ?")
                    params.append(update_data.get('last_used_at'))

                # Always update these
                set_clauses.extend(
                    ["last_modified = ?", "version = ?", "client_id = ?", "deleted = 0"])  # Ensure it's marked active
                params.extend([current_time, new_version, client_id])

                if not set_clauses:  # Nothing to update besides version/timestamp
                    return original_uuid, "No changes detected to update."

                sql_set_clause = ", ".join(set_clauses)
                update_sql = f"UPDATE Prompts SET {sql_set_clause} WHERE id = ? AND version = ?"  # nosec B608
                params.extend([prompt_id, current_version])

                cursor.execute(update_sql, tuple(params))

                if cursor.rowcount == 0:
                    raise ConflictError(f"Failed to update prompt ID {prompt_id} (version mismatch or record gone).",  # noqa: TRY003, TRY301
                                        "Prompts", prompt_id)

                # Log sync event
                # Fetch the full updated row for payload
                cursor.execute("SELECT * FROM Prompts WHERE id = ?", (prompt_id,))
                updated_payload = self._deserialize_prompt_record(dict(cursor.fetchone()))
                self._log_sync_event(conn, 'Prompts', original_uuid, 'update', new_version, updated_payload)

                # Update FTS
                self._update_fts_prompt(conn, prompt_id,
                                        updated_payload['name'], updated_payload.get('author'),
                                        updated_payload.get('details'), updated_payload.get('system_prompt'),
                                        updated_payload.get('user_prompt'),
                                        updated_payload.get('prompt_definition'))

                # Handle keywords if provided in update_data (assuming 'keywords' is a list of strings)
                if 'keywords' in update_data and isinstance(update_data['keywords'], list):
                    self.update_keywords_for_prompt(prompt_id, update_data['keywords'])  # Call existing method

                return original_uuid, f"Prompt ID {prompt_id} updated successfully to version {new_version}."

        except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error updating prompt ID {prompt_id}: {e}", exc_info=True)
            if isinstance(e, (InputError, ConflictError, DatabaseError)):
                raise
            raise DatabaseError(f"Failed to update prompt ID {prompt_id}: {e}") from e  # noqa: TRY003

    def record_prompt_usage(
        self, prompt_id_or_name_or_uuid: Union[int, str]
    ) -> Optional[dict[str, Any]]:
        current_time = self._get_current_utc_timestamp_str()
        client_id = self.client_id

        col_name = "id"
        identifier_value: Union[int, str] = prompt_id_or_name_or_uuid

        if isinstance(prompt_id_or_name_or_uuid, str):
            if prompt_id_or_name_or_uuid.isdigit():
                identifier_value = int(prompt_id_or_name_or_uuid)
                col_name = "id"
            else:
                try:
                    uuid.UUID(prompt_id_or_name_or_uuid, version=4)
                    col_name = "uuid"
                except ValueError:
                    col_name = "name"

        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, uuid, version, usage_count
                    FROM Prompts
                    WHERE {col_name} = ? AND deleted = 0
                    """.format_map(locals()),  # nosec B608
                    (identifier_value,),
                )
                prompt_info = cursor.fetchone()
                if not prompt_info:
                    return None

                prompt_id = int(prompt_info["id"])
                prompt_uuid = prompt_info["uuid"]
                current_version = int(prompt_info["version"])
                current_usage_count = int(prompt_info["usage_count"] or 0)
                new_usage_count = current_usage_count + 1
                new_version = current_version + 1

                cursor.execute(
                    """
                    UPDATE Prompts
                    SET usage_count = ?,
                        last_used_at = ?,
                        last_modified = ?,
                        version = ?,
                        client_id = ?
                    WHERE id = ? AND version = ?
                    """,
                    (
                        new_usage_count,
                        current_time,
                        current_time,
                        new_version,
                        client_id,
                        prompt_id,
                        current_version,
                    ),
                )
                if cursor.rowcount == 0:
                    raise ConflictError(
                        f"Failed to update usage for prompt ID {prompt_id}.",
                        "Prompts",
                        prompt_id,
                    )

                payload = {
                    "id": prompt_id,
                    "usage_count": new_usage_count,
                    "last_used_at": current_time,
                    "last_modified": current_time,
                    "version": new_version,
                    "client_id": client_id,
                }
                self._log_sync_event(
                    conn, "Prompts", prompt_uuid, "update", new_version, payload
                )

            return self.fetch_prompt_details(prompt_id, include_deleted=False)
        except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e:
            logger.error(
                f"Error recording usage for prompt '{prompt_id_or_name_or_uuid}': {e}",
                exc_info=True,
            )
            if isinstance(e, (InputError, ConflictError, DatabaseError)):
                raise
            raise DatabaseError(f"Failed to record prompt usage: {e}") from e  # noqa: TRY003

    def soft_delete_prompt(self, prompt_id_or_name_or_uuid: Union[int, str]) -> bool:
        current_time = self._get_current_utc_timestamp_str()
        client_id = self.client_id

        col_name = "id"
        identifier_value = prompt_id_or_name_or_uuid

        if isinstance(prompt_id_or_name_or_uuid, str):
            # First check if it's a numeric string (ID)
            if prompt_id_or_name_or_uuid.isdigit():
                identifier_value = int(prompt_id_or_name_or_uuid)
                col_name = "id"
            else:
                # Could be name or UUID. Check if it's a valid UUID format first.
                try:
                    uuid.UUID(prompt_id_or_name_or_uuid, version=4)
                    col_name = "uuid"
                except ValueError:
                    col_name = "name" # Assume it's a name if not a UUID

        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                # Fetch prompt to get its ID (if name/uuid provided), current version, and uuid
                # Also ensures it's not already deleted
                cursor.execute(f"SELECT id, uuid, version FROM Prompts WHERE {col_name} = ? AND deleted = 0", (identifier_value,))  # nosec B608
                prompt_info = cursor.fetchone()
                if not prompt_info:
                    logger.warning(f"Prompt '{prompt_id_or_name_or_uuid}' not found or already deleted.")
                    return False

                prompt_id, prompt_uuid, current_version = prompt_info['id'], prompt_info['uuid'], prompt_info['version']
                new_version = current_version + 1

                # Soft delete the prompt
                cursor.execute("UPDATE Prompts SET deleted=1, last_modified=?, version=?, client_id=? WHERE id=? AND version=?",
                               (current_time, new_version, client_id, prompt_id, current_version))
                if cursor.rowcount == 0:
                    raise ConflictError("Prompts", prompt_id)  # noqa: TRY301

                delete_payload = {'uuid': prompt_uuid, 'last_modified': current_time, 'version': new_version, 'client_id': client_id, 'deleted': 1}
                self._log_sync_event(conn, 'Prompts', prompt_uuid, 'delete', new_version, delete_payload)
                self._delete_fts_prompt(conn, prompt_id)

                # Rename the deleted prompt to free the original name for future creates
                # Strategy: prepend 'Deleted-' to the original name. If that collides, append an incrementing
                # counter in the style 'Deleted 2 - <name>', 'Deleted 3 - <name>', etc., until unique.
                try:
                    cursor.execute("SELECT name FROM Prompts WHERE id = ?", (prompt_id,))
                    rown = cursor.fetchone()
                    if rown and isinstance(rown['name'], str) and rown['name'].strip():
                        original_name = rown['name']
                        base_candidate = f"Deleted-{original_name}"
                        candidate = base_candidate
                        suffix = 1
                        # Ensure uniqueness across all records (active and deleted)
                        while True:
                            cursor.execute("SELECT id FROM Prompts WHERE name = ? AND id != ?", (candidate, prompt_id))
                            conflict = cursor.fetchone()
                            if not conflict:
                                break
                            suffix += 1
                            candidate = f"Deleted {suffix} - {original_name}"
                        try:
                            cursor.execute("UPDATE Prompts SET name = ? WHERE id = ?", (candidate, prompt_id))
                        except sqlite3.IntegrityError:
                            # In the unlikely event of a race, fall back to a UUID-suffixed name
                            fallback = f"Deleted-{original_name}-{prompt_uuid[:8]}"
                            cursor.execute("UPDATE Prompts SET name = ? WHERE id = ?", (fallback, prompt_id))
                except sqlite3.Error:
                    # Non-fatal: name renaming is a best-effort to free original name
                    pass

                # Explicitly unlink keywords and log those events
                cursor.execute("""
                               SELECT pkw.uuid AS keyword_uuid
                               FROM PromptKeywordLinks pkl
                                        JOIN PromptKeywordsTable pkw ON pkl.keyword_id = pkw.id
                               WHERE pkl.prompt_id = ? AND pkw.deleted = 0
                               """, (prompt_id,))
                keywords_to_unlink = cursor.fetchall()

                if keywords_to_unlink:
                    # The FK ON DELETE CASCADE on PromptKeywordLinks will remove rows.
                    # However, we want to log these 'unlink' events.
                    # So, we fetch them first, then rely on cascade or delete them explicitly.
                    # For clarity and explicit logging, let's delete them explicitly.
                    cursor.execute("DELETE FROM PromptKeywordLinks WHERE prompt_id = ?", (prompt_id,))
                    link_sync_version = 1
                    for kw_to_unlink in keywords_to_unlink:
                        keyword_uuid_val = kw_to_unlink['keyword_uuid']
                        link_composite_uuid = f"{prompt_uuid}_{keyword_uuid_val}"
                        unlink_payload = {'prompt_uuid': prompt_uuid, 'keyword_uuid': keyword_uuid_val}
                        self._log_sync_event(conn, 'PromptKeywordLinks', link_composite_uuid, 'unlink', link_sync_version, unlink_payload)
                    logging.debug(f"Unlinked {len(keywords_to_unlink)} keywords from soft-deleted prompt ID {prompt_id}.")

                logger.info(f"Soft deleted prompt '{prompt_id_or_name_or_uuid}' (ID: {prompt_id}, UUID: {prompt_uuid}).")
                return True
        except (ConflictError, DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error soft deleting prompt '{prompt_id_or_name_or_uuid}': {e}", exc_info=True)
            if isinstance(e, (ConflictError, DatabaseError)): raise  # noqa: E701
            else: raise DatabaseError(f"Failed to soft delete prompt: {e}") from e  # noqa: E701, TRY003

    def soft_delete_keyword(self, keyword_text: str) -> bool:
        if not keyword_text or not keyword_text.strip():
            raise InputError("Keyword to delete cannot be empty.")  # noqa: TRY003
        normalized_keyword = self._normalize_keyword(keyword_text)
        current_time = self._get_current_utc_timestamp_str()
        client_id = self.client_id

        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, uuid, version FROM PromptKeywordsTable WHERE keyword = ? AND deleted = 0", (normalized_keyword,))
                kw_info = cursor.fetchone()
                if not kw_info:
                    logger.warning(f"Prompt keyword '{normalized_keyword}' not found or already deleted.")
                    return False

                kw_id, kw_uuid, current_version = kw_info['id'], kw_info['uuid'], kw_info['version']
                new_version = current_version + 1

                cursor.execute("UPDATE PromptKeywordsTable SET deleted=1, last_modified=?, version=?, client_id=? WHERE id=? AND version=?",
                               (current_time, new_version, client_id, kw_id, current_version))
                if cursor.rowcount == 0:
                    raise ConflictError("PromptKeywordsTable", kw_id)  # noqa: TRY301

                delete_payload = {'uuid': kw_uuid, 'last_modified': current_time, 'version': new_version, 'client_id': client_id, 'deleted': 1}
                self._log_sync_event(conn, 'PromptKeywordsTable', kw_uuid, 'delete', new_version, delete_payload)
                self._delete_fts_prompt_keyword(conn, kw_id)

                # Explicitly unlink from prompts and log events
                cursor.execute("""
                               SELECT p.uuid AS prompt_uuid
                               FROM PromptKeywordLinks pkl
                                        JOIN Prompts p ON pkl.prompt_id = p.id
                               WHERE pkl.keyword_id = ? AND p.deleted = 0
                               """, (kw_id,))
                prompts_to_unlink = cursor.fetchall()

                if prompts_to_unlink:
                    # FK ON DELETE CASCADE will handle actual deletion from PromptKeywordLinks.
                    # Log these unlinks.
                    cursor.execute("DELETE FROM PromptKeywordLinks WHERE keyword_id = ?", (kw_id,))
                    link_sync_version = 1
                    for p_to_unlink in prompts_to_unlink:
                        prompt_uuid_val = p_to_unlink['prompt_uuid']
                        link_composite_uuid = f"{prompt_uuid_val}_{kw_uuid}"
                        unlink_payload = {'prompt_uuid': prompt_uuid_val, 'keyword_uuid': kw_uuid}
                        self._log_sync_event(conn, 'PromptKeywordLinks', link_composite_uuid, 'unlink', link_sync_version, unlink_payload)
                    logging.debug(f"Unlinked keyword ID {kw_id} from {len(prompts_to_unlink)} prompts during soft delete.")

                logger.info(f"Soft deleted prompt keyword '{normalized_keyword}' (ID: {kw_id}, UUID: {kw_uuid}).")
                return True
        except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error soft deleting prompt keyword '{keyword_text}': {e}", exc_info=True)
            if isinstance(e, (InputError, ConflictError, DatabaseError)): raise  # noqa: E701
            else: raise DatabaseError(f"Failed to soft delete prompt keyword: {e}") from e  # noqa: E701, TRY003

    # --- Prompt Collection Methods ---
    def create_prompt_collection(
        self,
        name: str,
        description: Optional[str] = None,
        prompt_ids: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        if not isinstance(name, str) or not name.strip():
            raise InputError("Collection name cannot be empty.")  # noqa: TRY003

        cleaned_name = name.strip()
        cleaned_description = description.strip() if isinstance(description, str) else description
        normalized_prompt_ids: list[int] = []
        for item in prompt_ids or []:
            if not isinstance(item, int) or item <= 0:
                raise InputError(f"Invalid prompt ID in collection payload: {item!r}")  # noqa: TRY003
            if item not in normalized_prompt_ids:
                normalized_prompt_ids.append(item)

        timestamp = self._get_current_utc_timestamp_str()
        collection_uuid = self._generate_uuid()

        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO PromptCollections (name, description, uuid, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (cleaned_name, cleaned_description, collection_uuid, timestamp, timestamp),
                )
                collection_id = cursor.lastrowid
                if not collection_id:
                    raise DatabaseError("Failed to create prompt collection.")  # noqa: TRY003

                if normalized_prompt_ids:
                    placeholders = ",".join("?" for _ in normalized_prompt_ids)
                    existing_rows = cursor.execute(
                        f"SELECT id FROM Prompts WHERE deleted = 0 AND id IN ({placeholders})",  # nosec B608
                        tuple(normalized_prompt_ids),
                    ).fetchall()
                    existing_prompt_ids = {int(row["id"]) for row in existing_rows}
                    missing = [pid for pid in normalized_prompt_ids if pid not in existing_prompt_ids]
                    if missing:
                        raise InputError(f"Prompt(s) not found or deleted: {missing}")  # noqa: TRY003

                    cursor.executemany(
                        """
                        INSERT INTO PromptCollectionItems (collection_id, prompt_id, sort_order)
                        VALUES (?, ?, ?)
                        """,
                        [(collection_id, prompt_id, idx) for idx, prompt_id in enumerate(normalized_prompt_ids)],
                    )

                return {
                    "collection_id": int(collection_id),
                    "name": cleaned_name,
                    "description": cleaned_description,
                    "prompt_ids": normalized_prompt_ids,
                }
        except (InputError, DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error creating prompt collection '{cleaned_name}': {e}", exc_info=True)
            if isinstance(e, (InputError, DatabaseError)):
                raise
            raise DatabaseError(f"Failed to create prompt collection: {e}") from e  # noqa: TRY003

    def get_prompt_collection_by_id(self, collection_id: int) -> Optional[dict[str, Any]]:
        if not isinstance(collection_id, int) or collection_id <= 0:
            raise InputError("Collection ID must be a positive integer.")  # noqa: TRY003

        try:
            collection_row = self.execute_query(
                """
                SELECT id, name, description
                FROM PromptCollections
                WHERE id = ?
                """,
                (collection_id,),
            ).fetchone()
            if not collection_row:
                return None

            prompt_rows = self.execute_query(
                """
                SELECT pci.prompt_id
                FROM PromptCollectionItems pci
                JOIN Prompts p ON p.id = pci.prompt_id
                WHERE pci.collection_id = ? AND p.deleted = 0
                ORDER BY pci.sort_order ASC, pci.prompt_id ASC
                """,
                (collection_id,),
            ).fetchall()

            return {
                "collection_id": int(collection_row["id"]),
                "name": collection_row["name"],
                "description": collection_row["description"],
                "prompt_ids": [int(row["prompt_id"]) for row in prompt_rows],
            }
        except (InputError, DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching prompt collection {collection_id}: {e}", exc_info=True)
            if isinstance(e, (InputError, DatabaseError)):
                raise
            raise DatabaseError(f"Failed to fetch prompt collection: {e}") from e  # noqa: TRY003

    def list_prompt_collections(self, limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
        if not isinstance(limit, int) or limit <= 0:
            raise InputError("Limit must be a positive integer.")  # noqa: TRY003
        if not isinstance(offset, int) or offset < 0:
            raise InputError("Offset must be a non-negative integer.")  # noqa: TRY003

        try:
            collection_rows = self.execute_query(
                """
                SELECT id, name, description
                FROM PromptCollections
                ORDER BY updated_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

            collections: list[dict[str, Any]] = []
            for collection_row in collection_rows:
                collection_id = int(collection_row["id"])
                prompt_rows = self.execute_query(
                    """
                    SELECT pci.prompt_id
                    FROM PromptCollectionItems pci
                    JOIN Prompts p ON p.id = pci.prompt_id
                    WHERE pci.collection_id = ? AND p.deleted = 0
                    ORDER BY pci.sort_order ASC, pci.prompt_id ASC
                    """,
                    (collection_id,),
                ).fetchall()
                collections.append(
                    {
                        "collection_id": collection_id,
                        "name": collection_row["name"],
                        "description": collection_row["description"],
                        "prompt_ids": [int(row["prompt_id"]) for row in prompt_rows],
                    }
                )

            return collections
        except (InputError, DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error listing prompt collections: {e}", exc_info=True)
            if isinstance(e, (InputError, DatabaseError)):
                raise
            raise DatabaseError(f"Failed to list prompt collections: {e}") from e  # noqa: TRY003

    def count_prompt_collections(self) -> int:
        try:
            row = self.execute_query(
                """
                SELECT COUNT(*) AS total
                FROM PromptCollections
                """
            ).fetchone()
            return int(row["total"] if row and row["total"] is not None else 0)
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error counting prompt collections: {e}", exc_info=True)
            if isinstance(e, DatabaseError):
                raise
            raise DatabaseError(f"Failed to count prompt collections: {e}") from e  # noqa: TRY003

    def update_prompt_collection(
        self,
        collection_id: int,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        prompt_ids: Optional[list[int]] = None,
    ) -> Optional[dict[str, Any]]:
        if not isinstance(collection_id, int) or collection_id <= 0:
            raise InputError("Collection ID must be a positive integer.")  # noqa: TRY003

        normalized_name: Optional[str] = None
        if name is not None:
            if not isinstance(name, str) or not name.strip():
                raise InputError("Collection name cannot be empty.")  # noqa: TRY003
            normalized_name = name.strip()

        normalized_description = description.strip() if isinstance(description, str) else description

        normalized_prompt_ids: Optional[list[int]] = None
        if prompt_ids is not None:
            normalized_prompt_ids = []
            for item in prompt_ids:
                if not isinstance(item, int) or item <= 0:
                    raise InputError(f"Invalid prompt ID in collection payload: {item!r}")  # noqa: TRY003
                if item not in normalized_prompt_ids:
                    normalized_prompt_ids.append(item)

        timestamp = self._get_current_utc_timestamp_str()

        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                collection_row = cursor.execute(
                    """
                    SELECT id, name, description
                    FROM PromptCollections
                    WHERE id = ?
                    """,
                    (collection_id,),
                ).fetchone()
                if not collection_row:
                    return None

                next_name = normalized_name if normalized_name is not None else collection_row["name"]
                next_description = (
                    normalized_description
                    if description is not None
                    else collection_row["description"]
                )

                cursor.execute(
                    """
                    UPDATE PromptCollections
                    SET name = ?, description = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (next_name, next_description, timestamp, collection_id),
                )

                if normalized_prompt_ids is not None:
                    if normalized_prompt_ids:
                        placeholders = ",".join("?" for _ in normalized_prompt_ids)
                        existing_rows = cursor.execute(
                            f"SELECT id FROM Prompts WHERE deleted = 0 AND id IN ({placeholders})",  # nosec B608
                            tuple(normalized_prompt_ids),
                        ).fetchall()
                        existing_prompt_ids = {int(row["id"]) for row in existing_rows}
                        missing = [pid for pid in normalized_prompt_ids if pid not in existing_prompt_ids]
                        if missing:
                            raise InputError(f"Prompt(s) not found or deleted: {missing}")  # noqa: TRY003

                    cursor.execute(
                        "DELETE FROM PromptCollectionItems WHERE collection_id = ?",
                        (collection_id,),
                    )

                    if normalized_prompt_ids:
                        cursor.executemany(
                            """
                            INSERT INTO PromptCollectionItems (collection_id, prompt_id, sort_order)
                            VALUES (?, ?, ?)
                            """,
                            [
                                (collection_id, prompt_id, idx)
                                for idx, prompt_id in enumerate(normalized_prompt_ids)
                            ],
                        )

            return self.get_prompt_collection_by_id(collection_id)
        except (InputError, DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error updating prompt collection {collection_id}: {e}", exc_info=True)
            if isinstance(e, (InputError, DatabaseError)):
                raise
            raise DatabaseError(f"Failed to update prompt collection: {e}") from e  # noqa: TRY003

    # --- Read Methods ---
    def get_prompt_by_id(self, prompt_id: int, include_deleted: bool = False) -> Optional[dict]:
        query = "SELECT * FROM Prompts WHERE id = ?"
        params = [prompt_id]
        if not include_deleted:
            query += " AND deleted = 0"
        try:
            cursor = self.execute_query(query, tuple(params))
            result = cursor.fetchone()
            return self._deserialize_prompt_record(dict(result)) if result else None
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching prompt by ID {prompt_id}: {e}")
            raise DatabaseError(f"Failed fetch prompt by ID: {e}") from e  # noqa: TRY003

    def get_prompt_by_uuid(self, prompt_uuid: str, include_deleted: bool = False) -> Optional[dict]:
        query = "SELECT * FROM Prompts WHERE uuid = ?"
        params = [prompt_uuid]
        if not include_deleted:
            query += " AND deleted = 0"
        try:
            cursor = self.execute_query(query, tuple(params))
            result = cursor.fetchone()
            return self._deserialize_prompt_record(dict(result)) if result else None
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching prompt by UUID {prompt_uuid}: {e}")
            raise DatabaseError(f"Failed fetch prompt by UUID: {e}") from e  # noqa: TRY003

    def get_prompt_by_name(self, name: str, include_deleted: bool = False) -> Optional[dict]:
        query = "SELECT * FROM Prompts WHERE name = ?"
        params = [name]
        if not include_deleted:
            query += " AND deleted = 0"
        try:
            cursor = self.execute_query(query, tuple(params))
            result = cursor.fetchone()
            return self._deserialize_prompt_record(dict(result)) if result else None
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching prompt by name '{name}': {e}")
            raise DatabaseError(f"Failed fetch prompt by name: {e}") from e  # noqa: TRY003

    def list_prompts(
        self,
        page: int = 1,
        per_page: int = 10,
        include_deleted: bool = False,
        sort_by: str = "last_modified",
        sort_order: str = "desc"
    ) -> tuple[list[dict], int, int, int]:
        if page < 1:
            raise ValueError("Page number must be >= 1")  # noqa: TRY003
        if per_page < 1:
            raise ValueError("Per page must be >= 1")  # noqa: TRY003
        sort_key = (sort_by or "last_modified").strip().lower()
        sort_dir = (sort_order or "desc").strip().lower()
        allowed_sort = {
            "last_modified": "last_modified",
            "name": "name",
            "author": "author",
            "id": "id",
            "usage_count": "usage_count",
            "last_used_at": "last_used_at",
        }
        if sort_key not in allowed_sort:
            raise ValueError(f"Unsupported sort_by value: {sort_by}")  # noqa: TRY003
        if sort_dir not in {"asc", "desc"}:
            raise ValueError(f"Unsupported sort_order value: {sort_order}")  # noqa: TRY003
        offset = (page - 1) * per_page

        where_clause = "WHERE deleted = 0" if not include_deleted else ""
        order_col = allowed_sort[sort_key]
        order_dir = "ASC" if sort_dir == "asc" else "DESC"
        tie_breaker = f"id {order_dir}"
        if sort_key in {"name", "author"}:
            order_clause = f"{order_col} COLLATE NOCASE {order_dir}, {tie_breaker}"
        else:
            order_clause = f"{order_col} {order_dir}, {tie_breaker}"

        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM Prompts {where_clause}")  # nosec B608
                total_items = cursor.fetchone()[0]

                results_data = []
                if total_items > 0:
                    # Select desired fields, e.g., id, name, uuid, author
                    query = """SELECT id, name, uuid, author, details, last_modified, usage_count, last_used_at FROM Prompts
                                {where_clause} ORDER BY {order_clause}
                                LIMIT ? OFFSET ?""".format_map(locals())  # nosec B608
                    cursor.execute(query, (per_page, offset))
                    results_data = [dict(row) for row in cursor.fetchall()]
                    # Normalize author formatting only (do not mutate other fields)
                    for it in results_data:
                        with suppress(_PROMPTS_NONCRITICAL_EXCEPTIONS):
                            it['author'] = (it.get('author') or '').strip()

                # Enrich each prompt with keywords for downstream filtering/searching that rely on list output
                # (kept outside of the above block to ensure empty lists are handled consistently)
                if results_data:
                    try:
                        for item in results_data:
                            pid = item.get('id')
                            if pid is not None:
                                item['keywords'] = self.fetch_keywords_for_prompt(int(pid), include_deleted=False)
                    except _PROMPTS_NONCRITICAL_EXCEPTIONS as e:
                        logging.debug(f"Could not enrich prompts with keywords: {e}")
                # Do not mutate keywords in list results; preserve exact values for export/roundtrip

            total_pages = ceil(total_items / per_page) if total_items > 0 else 0
            return results_data, total_pages, page, total_items  # noqa: TRY300
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error listing prompts: {e}")
            raise DatabaseError(f"Failed to list prompts: {e}") from e  # noqa: TRY003

    def fetch_prompt_details(self, prompt_id_or_name_or_uuid: Union[int, str], include_deleted: bool = False) -> Optional[dict]:
        prompt_data = None
        if isinstance(prompt_id_or_name_or_uuid, int):
            prompt_data = self.get_prompt_by_id(prompt_id_or_name_or_uuid, include_deleted)
        elif isinstance(prompt_id_or_name_or_uuid, str):
            # First, check if it's a numeric string (ID)
            if prompt_id_or_name_or_uuid.isdigit():
                prompt_data = self.get_prompt_by_id(int(prompt_id_or_name_or_uuid), include_deleted)
            else:
                try: # Check if UUID
                    uuid.UUID(prompt_id_or_name_or_uuid, version=4)
                    prompt_data = self.get_prompt_by_uuid(prompt_id_or_name_or_uuid, include_deleted)
                except ValueError: # Assume name
                    prompt_data = self.get_prompt_by_name(prompt_id_or_name_or_uuid, include_deleted)

        if not prompt_data:
            return None

        # Fetch keywords
        keywords = self.fetch_keywords_for_prompt(prompt_data['id'], include_deleted=include_deleted) # Pass prompt_id
        prompt_data_dict = dict(prompt_data)
        prompt_data_dict['keywords'] = keywords
        return prompt_data_dict

    def fetch_all_keywords(self, include_deleted: bool = False) -> list[str]:
        query = "SELECT keyword FROM PromptKeywordsTable"
        if not include_deleted: query += " WHERE deleted = 0"  # noqa: E701
        query += " ORDER BY keyword COLLATE NOCASE"
        try:
            cursor = self.execute_query(query)
            return [row['keyword'] for row in cursor.fetchall()]
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching all prompt keywords: {e}")
            raise DatabaseError("Failed to fetch all prompt keywords") from e  # noqa: TRY003

    def fetch_all_prompt_names(self, include_deleted: bool = True) -> list[str]:
        """
        Returns all prompt names from the Prompts table.

        Args:
            include_deleted: If True, include soft-deleted prompts; otherwise only active prompts.

        Returns:
            A list of prompt names (strings).

        Raises:
            DatabaseError: If the query fails.
        """
        query = "SELECT name FROM Prompts"
        if not include_deleted:
            query += " WHERE deleted = 0"
        query += " ORDER BY name COLLATE NOCASE"
        try:
            cursor = self.execute_query(query)
            return [row['name'] for row in cursor.fetchall()]
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching all prompt names: {e}")
            raise DatabaseError("Failed to fetch all prompt names") from e  # noqa: TRY003

    def fetch_keywords_for_prompt(self, prompt_id: int, include_deleted: bool = False) -> list[str]:
        # Note: include_deleted here refers to the keyword itself, not the link or prompt
        query = """SELECT k.keyword FROM PromptKeywordsTable k
                                             JOIN PromptKeywordLinks pkl ON k.id = pkl.keyword_id
                   WHERE pkl.prompt_id = ?"""
        params = [prompt_id]
        if not include_deleted: # Filter for active keywords
            query += " AND k.deleted = 0"
        query += " ORDER BY k.keyword COLLATE NOCASE"
        try:
            cursor = self.execute_query(query, tuple(params))
            return [row['keyword'] for row in cursor.fetchall()]
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching keywords for prompt ID {prompt_id}: {e}")
            raise DatabaseError(f"Failed to fetch keywords for prompt {prompt_id}") from e  # noqa: TRY003

    def search_prompts(self,
                       search_query: Optional[str],
                       search_fields: Optional[list[str]] = None,  # e.g. ['name', 'details', 'keywords']
                       page: int = 1,
                       results_per_page: int = 20,
                       include_deleted: bool = False
                       ) -> tuple[list[dict[str, Any]], int]:
        if page < 1: raise ValueError("Page must be >= 1")  # noqa: E701, TRY003
        if results_per_page < 1: raise ValueError("Results per page must be >= 1")  # noqa: E701, TRY003

        # Keep a copy of caller intent before we normalize fields
        original_fields = search_fields

        # Normalize fields
        if search_query and not search_fields:
            search_fields = ["name", "details", "system_prompt", "user_prompt", "author"]
        elif not search_fields:
            search_fields = []

        offset = (page - 1) * results_per_page

        base_select = "SELECT p.*"
        count_select = "SELECT COUNT(p.id)"
        from_clause = "FROM Prompts p"
        order_by_clause = "ORDER BY p.last_modified DESC, p.id DESC"
        conditions = []
        params = []

        if not include_deleted:
            conditions.append("p.deleted = 0")

        # Handle field-prefixed query patterns
        # Case-insensitive prefix handling for field-specific filters
        if isinstance(search_query, str) and ":" in search_query:
            _sq = search_query.lstrip()
            _pfx, _rest = _sq.split(":", 1)
            _pfx_norm = _pfx.strip().lower()
        else:
            _pfx_norm, _rest = None, None

        if _pfx_norm == "author":
            author_value = _rest
            if author_value is None:
                author_value = ""
            author_value = author_value.strip()
            # Exact author match (case-insensitive)
            cond = "p.author = ? COLLATE NOCASE"
            conditions = ["p.deleted = 0", cond] if not include_deleted else [cond]
            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            results_sql = f"{base_select} {from_clause} {where_clause} {order_by_clause} LIMIT ? OFFSET ?"
            count_sql = f"{count_select} {from_clause} {where_clause}"
            params = [author_value]
            try:
                total_matches = self.execute_query(count_sql, tuple(params)).fetchone()[0]
                results_list = []
                if total_matches > 0:
                    paginated_params = tuple(params + [results_per_page, offset])
                    results_cursor = self.execute_query(results_sql, paginated_params)
                    results_list = [dict(row) for row in results_cursor.fetchall()]
                    for res in results_list:
                        res['keywords'] = self.fetch_keywords_for_prompt(res['id'], include_deleted=False)
                return results_list, total_matches  # noqa: TRY300
            except (DatabaseError, sqlite3.Error) as e:
                logging.error(f"DB error during author filter search: {e}", exc_info=True)
                return [], 0

        if _pfx_norm == "keyword":
            kw_value_raw = (_rest or "")
            kw_value = self._normalize_keyword(kw_value_raw)
            try:
                # Match active keywords (case-insensitive exact match via NOCASE collation)
                kw_cursor = self.execute_query(
                    "SELECT id FROM PromptKeywordsTable WHERE keyword = ? COLLATE NOCASE AND deleted = 0",
                    (kw_value,),
                )
                kw_ids = [row['id'] for row in kw_cursor.fetchall()]
                if not kw_ids:
                    # Python-side normalization fallback for edge cases (control chars, locale quirks)
                    try:
                        all_kw_cur = self.execute_query("SELECT id, keyword FROM PromptKeywordsTable WHERE deleted = 0")
                        target_norm = self._normalize_keyword(kw_value_raw).casefold()
                        for r in all_kw_cur.fetchall():
                            if self._normalize_keyword(r['keyword']).casefold() == target_norm:
                                kw_ids.append(r['id'])
                    except _PROMPTS_NONCRITICAL_EXCEPTIONS:
                        pass
                if not kw_ids:
                    return [], 0
                placeholders = ','.join('?' * len(kw_ids))
                link_cursor = self.execute_query(
                    f"SELECT DISTINCT prompt_id FROM PromptKeywordLinks WHERE keyword_id IN ({placeholders})",  # nosec B608
                    tuple(kw_ids),
                )
                prompt_ids = [row['prompt_id'] for row in link_cursor.fetchall()]
                if not prompt_ids:
                    return [], 0
                id_placeholders = ','.join('?' * len(prompt_ids))
                conditions = []
                if not include_deleted:
                    conditions.append("p.deleted = 0")
                conditions.append(f"p.id IN ({id_placeholders})")
                where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
                results_sql = f"{base_select} {from_clause} {where_clause} {order_by_clause} LIMIT ? OFFSET ?"
                count_sql = f"{count_select} {from_clause} {where_clause}"
                total_matches = self.execute_query(count_sql, tuple(prompt_ids)).fetchone()[0]
                results_list = []
                if total_matches > 0:
                    paginated_params = tuple(prompt_ids + [results_per_page, offset])
                    results_cursor = self.execute_query(results_sql, paginated_params)
                    results_list = [dict(row) for row in results_cursor.fetchall()]
                    for res in results_list:
                        res['keywords'] = self.fetch_keywords_for_prompt(res['id'], include_deleted=False)
                return results_list, total_matches  # noqa: TRY300
            except (DatabaseError, sqlite3.Error) as e:
                logging.error(f"DB error during keyword filter search: {e}", exc_info=True)
                return [], 0

        # --- FTS search using subqueries, with robust fallback ---
        used_fts = False
        fts_error = False
        # Prefer Python-side normalization only in TEST_MODE and when caller didn't specify fields
        prefer_naive = (
            is_test_mode()
            and ((original_fields is None) or (isinstance(original_fields, list) and len(original_fields) == 0))
        )

        if search_query and search_fields and not prefer_naive:
            matching_prompt_ids = set()
            text_search_fields = {"name", "author", "details", "system_prompt", "user_prompt"}

            # Search in prompt text fields using column-qualified FTS query restricted to requested fields
            selected_text_fields = [f for f in search_fields if f in text_search_fields]
            if selected_text_fields:
                try:
                    fts_query = " OR ".join([f"{fld}:{search_query}" for fld in selected_text_fields])
                    cursor = self.execute_query("SELECT rowid FROM prompts_fts WHERE prompts_fts MATCH ?", (fts_query,))
                    used_fts = True
                    matching_prompt_ids.update(row['rowid'] for row in cursor.fetchall())
                except sqlite3.Error as e:
                    logging.warning(f"FTS search on prompts failed: {e}; will fallback to naive search.")
                    fts_error = True


            # Search in keywords
            if "keywords" in search_fields:
                try:
                    # 1. Find keyword IDs matching the query
                    kw_cursor = self.execute_query("SELECT rowid FROM prompt_keywords_fts WHERE prompt_keywords_fts MATCH ?", (search_query,))
                    used_fts = True
                    matching_keyword_ids = {row['rowid'] for row in kw_cursor.fetchall()}

                    # 2. Find prompt IDs linked to those keywords
                    if matching_keyword_ids:
                        placeholders = ','.join('?' * len(matching_keyword_ids))
                        link_cursor = self.execute_query(
                            f"SELECT DISTINCT prompt_id FROM PromptKeywordLinks WHERE keyword_id IN ({placeholders})",  # nosec B608
                            tuple(matching_keyword_ids)
                        )
                        matching_prompt_ids.update(row['prompt_id'] for row in link_cursor.fetchall())
                except sqlite3.Error as e:
                    logging.warning(f"FTS search on keywords failed: {e}; will fallback to naive search.")
                    fts_error = True

            if not matching_prompt_ids and not used_fts:
                return [], 0  # No FTS used and no matches requested

            # Add the final ID list to the main query conditions
            if matching_prompt_ids:
                id_placeholders = ','.join('?' * len(matching_prompt_ids))
                conditions.append(f"p.id IN ({id_placeholders})")
                params.extend(list(matching_prompt_ids))

        # If FTS was used but resulted in no matches, prevent unfiltered result set before fallback.
        if search_query and search_fields and used_fts and not matching_prompt_ids and ("author:" not in str(search_query).lower()) and ("keyword:" not in str(search_query).lower()):
            conditions.append("1 = 0")

        # --- Build and Execute Final Query ---
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        order_by_clause = "ORDER BY p.last_modified DESC, p.id DESC"

        try:
            # Get total count
            count_sql = f"{count_select} {from_clause} {where_clause}"
            total_matches = self.execute_query(count_sql, tuple(params)).fetchone()[0]

            results_list = []
            if total_matches > 0:
                # Get paginated results
                results_sql = f"{base_select} {from_clause} {where_clause} {order_by_clause} LIMIT ? OFFSET ?"
                paginated_params = tuple(params + [results_per_page, offset])
                results_cursor = self.execute_query(results_sql, paginated_params)
                results_list = [dict(row) for row in results_cursor.fetchall()]
                # Attach keywords to each result
                for res_dict in results_list:
                    res_dict['keywords'] = self.fetch_keywords_for_prompt(res_dict['id'], include_deleted=False)
            # Naive fallback conditions:
            # - FTS errored
            # - Prefer naive in test mode or when no fields were specified (original_fields None)
            # - FTS was used but returned zero matches (defensive fallback to avoid false negatives)
            do_naive = False
            if search_query:  # noqa: SIM102
                if fts_error or prefer_naive or (not used_fts and (not search_fields or len(search_fields) == 0)) or used_fts and total_matches == 0:
                    do_naive = True
            if do_naive:
                try:
                    fallback_items = []
                    base_where = "WHERE p.deleted = 0" if not include_deleted else ""
                    all_rows_cursor = self.execute_query(
                        f"SELECT p.* {from_clause} {base_where}"
                    )
                    q = self._normalize_text_for_search(search_query)
                    # Determine which fields to scan
                    text_fields = {"name", "author", "details", "system_prompt", "user_prompt"}
                    fields_to_scan = set(search_fields) if search_fields else (text_fields | {"keywords"})
                    for row in all_rows_cursor.fetchall():
                        rowd = dict(row)
                        kws = self.fetch_keywords_for_prompt(rowd['id'], include_deleted=False)
                        haystack_parts = []
                        if fields_to_scan & text_fields:
                            haystack_parts.append(
                                self._normalize_text_for_search(
                                    self.build_structured_prompt_searchable_text(
                                        rowd.get("prompt_definition_json")
                                    )
                                )
                            )
                        for f in fields_to_scan:
                            if f == "keywords":
                                haystack_parts.extend([self._normalize_text_for_search(k) for k in kws])
                            elif f in text_fields:
                                haystack_parts.append(self._normalize_text_for_search(rowd.get(f, "")))
                        if q in ' '.join(haystack_parts):
                            rowd['keywords'] = kws
                            fallback_items.append(rowd)
                    # Merge unique by id. If running in prefer_naive mode,
                    # ignore any prior unfiltered SQL results to avoid overcounting.
                    by_id = {} if prefer_naive else {item['id']: item for item in results_list}
                    for item in fallback_items:
                        by_id[item['id']] = item
                    combined = list(by_id.values())
                    combined.sort(key=lambda x: (x.get('last_modified', ''), x.get('id', 0)), reverse=True)
                    total_matches = len(combined)
                    results_list = combined[offset:offset + results_per_page]
                    return results_list, total_matches  # noqa: TRY300
                except _PROMPTS_NONCRITICAL_EXCEPTIONS as fe:
                    logging.warning(f"Naive fallback search failed: {fe}")
            return results_list, total_matches  # noqa: TRY300
        except (DatabaseError, sqlite3.Error) as e:
            logging.error(f"DB error during prompt search: {e}", exc_info=True)
            # Last-resort fallback to empty set
            return [], 0

    # --- Sync Log Access Methods ---
    def get_sync_log_entries(self, since_change_id: int = 0, limit: Optional[int] = None) -> list[dict]:
        query = "SELECT * FROM sync_log WHERE change_id > ? ORDER BY change_id ASC"
        params_list = [since_change_id]
        if limit is not None:
            query += " LIMIT ?"
            params_list.append(limit)
        try:
            cursor = self.execute_query(query, tuple(params_list))
            results = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                if row_dict.get('payload'):
                    try: row_dict['payload'] = json.loads(row_dict['payload'])  # noqa: E701
                    except json.JSONDecodeError:
                        logging.warning(f"Failed decode JSON payload for sync_log ID {row_dict.get('change_id')}")
                        row_dict['payload'] = None
                results.append(row_dict)
            return results  # noqa: TRY300
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching sync_log entries: {e}")
            raise DatabaseError("Failed to fetch sync_log entries") from e  # noqa: TRY003


    def delete_sync_log_entries(self, change_ids: list[int]) -> int:
        if not change_ids: return 0  # noqa: E701
        if not all(isinstance(cid, int) for cid in change_ids):
            raise ValueError("change_ids must be a list of integers.")  # noqa: TRY003
        placeholders = ','.join('?' * len(change_ids))
        query = f"DELETE FROM sync_log WHERE change_id IN ({placeholders})"  # nosec B608
        try:
            with self.transaction(): # Ensure commit happens
                cursor = self.execute_query(query, tuple(change_ids), commit=False) # commit handled by transaction
                deleted_count = cursor.rowcount
                logger.info(f"Deleted {deleted_count} sync log entries.")
                return deleted_count
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error deleting sync log entries: {e}")
            raise DatabaseError("Failed to delete sync log entries") from e  # noqa: TRY003


# =========================================================================
# Standalone Functions (REQUIRE db_instance passed explicitly)
# =========================================================================
# These functions now operate on a PromptsDatabase instance.

def add_or_update_prompt(db_instance: PromptsDatabase,
                         name: str, author: Optional[str], details: Optional[str],
                         system_prompt: Optional[str] = None, user_prompt: Optional[str] = None,
                         keywords: Optional[list[str]] = None) -> tuple[Optional[int], Optional[str], str]:
    """
    Adds a new prompt or updates an existing one (identified by name).
    If the prompt exists (even if soft-deleted), it will be updated/undeleted.
    """
    if not isinstance(db_instance, PromptsDatabase):
        raise TypeError("db_instance must be a PromptsDatabase object.")  # noqa: TRY003
    # `add_prompt` with overwrite=True handles both add and update logic.
    return db_instance.add_prompt(
        name=name, author=author, details=details,
        system_prompt=system_prompt, user_prompt=user_prompt,
        keywords=keywords, overwrite=True # Key change: always overwrite/update if exists
    )

def load_prompt_details_for_ui(db_instance: PromptsDatabase, prompt_name: str) -> tuple[str, str, str, str, str, str]:
    """
    Loads prompt details for UI display, fetching by name.
    Returns empty strings if not found.
    """
    if not isinstance(db_instance, PromptsDatabase):
        raise TypeError("db_instance must be a PromptsDatabase object.")  # noqa: TRY003
    if not prompt_name:
        return "", "", "", "", "", ""

    details_dict = db_instance.fetch_prompt_details(prompt_name, include_deleted=False) # Fetch active by name
    if details_dict:
        return (
            details_dict.get('name', ""),
            details_dict.get('author', "") or "", # Ensure empty string if None
            details_dict.get('details', "") or "",
            details_dict.get('system_prompt', "") or "",
            details_dict.get('user_prompt', "") or "",
            ', '.join(details_dict.get('keywords', [])) # keywords should be a list
        )
    return "", "", "", "", "", ""


def export_prompt_keywords_to_csv(db_instance: PromptsDatabase) -> tuple[str, str]:
    import csv
    import os
    import tempfile
    from datetime import datetime

    if not isinstance(db_instance, PromptsDatabase):
        raise TypeError("db_instance must be a PromptsDatabase object.")  # noqa: TRY003

    logging.debug(f"export_prompt_keywords_to_csv from DB: {db_instance.db_path_str}")
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f'prompt_keywords_export_{timestamp}.csv')

        # Query to get keywords with associated prompt info (names, authors, counts)
        # This requires joining Prompts, PromptKeywordsTable, PromptKeywordLinks
        query = """
                SELECT
                    pkw.keyword,
                    GROUP_CONCAT(DISTINCT p.name) AS prompt_names,
                    COUNT(DISTINCT p.id) AS num_prompts,
                    GROUP_CONCAT(DISTINCT p.author) AS authors
                FROM PromptKeywordsTable pkw
                         LEFT JOIN PromptKeywordLinks pkl ON pkw.id = pkl.keyword_id
                         LEFT JOIN Prompts p ON pkl.prompt_id = p.id AND p.deleted = 0 /* Only count links to active prompts */
                WHERE pkw.deleted = 0 /* Only export active keywords */
                GROUP BY pkw.id, pkw.keyword
                ORDER BY pkw.keyword COLLATE NOCASE \
                """
        cursor = db_instance.execute_query(query)
        results = cursor.fetchall()

        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Keyword', 'Associated Prompts', 'Number of Prompts', 'Authors'])
            for row in results:
                writer.writerow([
                    row['keyword'],
                    row['prompt_names'] or '',
                    row['num_prompts'],
                    row['authors'] or ''
                ])

        status_msg = f"Successfully exported {len(results)} active prompt keywords to CSV."
        logging.info(status_msg)
        return status_msg, file_path  # noqa: TRY300

    except (DatabaseError, sqlite3.Error) as e:
        error_msg = f"Database error exporting keywords: {e}"
        logging.error(error_msg, exc_info=True)
        return error_msg, "None"
    except _PROMPTS_NONCRITICAL_EXCEPTIONS as e:
        error_msg = f"Error exporting keywords: {e}"
        logging.error(error_msg, exc_info=True)
        return error_msg, "None"


def view_prompt_keywords_markdown(db_instance: PromptsDatabase) -> str:
    if not isinstance(db_instance, PromptsDatabase):
        raise TypeError("db_instance must be a PromptsDatabase object.")  # noqa: TRY003
    logging.debug(f"view_prompt_keywords_markdown from DB: {db_instance.db_path_str}")
    try:
        query = """
                SELECT pkw.keyword, COUNT(DISTINCT pkl.prompt_id) as prompt_count
                FROM PromptKeywordsTable pkw
                         LEFT JOIN PromptKeywordLinks pkl ON pkw.id = pkl.keyword_id
                         LEFT JOIN Prompts p ON pkl.prompt_id = p.id AND p.deleted = 0
                WHERE pkw.deleted = 0
                GROUP BY pkw.id, pkw.keyword
                ORDER BY pkw.keyword COLLATE NOCASE \
                """
        cursor = db_instance.execute_query(query)
        keywords_data = cursor.fetchall()

        if keywords_data:
            keyword_list_md = [f"- {row['keyword']} ({row['prompt_count']} active prompts)" for row in keywords_data]
            return "### Current Active Prompt Keywords:\n" + "\n".join(keyword_list_md)
        return "No active keywords found."  # noqa: TRY300
    except (DatabaseError, sqlite3.Error) as e:
        error_msg = f"Error retrieving keywords for markdown view: {e}"
        logging.error(error_msg, exc_info=True)
        return error_msg


def export_prompts_formatted(db_instance: PromptsDatabase,
                             export_format: str = 'csv', # 'csv' or 'markdown'
                             filter_keywords: Optional[list[str]] = None,
                             include_system: bool = True,
                             include_user: bool = True,
                             include_details: bool = True,
                             include_author: bool = True,
                             include_associated_keywords: bool = True, # Renamed for clarity
                             markdown_template_name: Optional[str] = "Basic Template" # Name of template
                             ) -> tuple[str, str]:
    import csv
    import os
    import tempfile
    import zipfile  # For markdown if multiple files
    from datetime import datetime

    if not isinstance(db_instance, PromptsDatabase):
        raise TypeError("db_instance must be a PromptsDatabase object.")  # noqa: TRY003

    logging.debug(f"export_prompts_formatted (format: {export_format}) from DB: {db_instance.db_path_str}")

    # --- Fetch Prompts Data ---
    # Build base query parts
    select_fields = ["p.id", "p.name", "p.uuid"] # Always include id, name, uuid
    if include_author: select_fields.append("p.author")  # noqa: E701
    if include_details: select_fields.append("p.details")  # noqa: E701
    if include_system: select_fields.append("p.system_prompt")  # noqa: E701
    if include_user: select_fields.append("p.user_prompt")  # noqa: E701

    query_sql = f"SELECT DISTINCT {', '.join(select_fields)} FROM Prompts p"  # nosec B608
    query_params = []

    # Keyword filtering
    if filter_keywords and len(filter_keywords) > 0:
        normalized_filter_keywords = [db_instance._normalize_keyword(k) for k in filter_keywords if k and k.strip()]
        if normalized_filter_keywords:
            placeholders = ','.join(['?'] * len(normalized_filter_keywords))
            query_sql += f"""
                JOIN PromptKeywordLinks pkl ON p.id = pkl.prompt_id
                JOIN PromptKeywordsTable pkw ON pkl.keyword_id = pkw.id
                WHERE p.deleted = 0 AND pkw.deleted = 0 AND pkw.keyword IN ({placeholders})
            """
            query_params.extend(normalized_filter_keywords)
        else: # No valid filter keywords, so just filter active prompts
            query_sql += " WHERE p.deleted = 0"
    else: # No keyword filter, just active prompts
        query_sql += " WHERE p.deleted = 0"

    query_sql += " ORDER BY p.name COLLATE NOCASE"

    try:
        cursor = db_instance.execute_query(query_sql, tuple(query_params))
        prompts_data = [dict(row) for row in cursor.fetchall()]

        if not prompts_data:
            return "No prompts found matching the criteria for export.", "None"

        # Fetch associated keywords for each prompt if needed
        if include_associated_keywords:
            for prompt_dict in prompts_data:
                prompt_dict['keywords_list'] = db_instance.fetch_keywords_for_prompt(prompt_dict['id'], include_deleted=False)

        # --- Perform Export ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_path = "None"

        if export_format == 'csv':
            temp_csv_file = os.path.join(tempfile.gettempdir(), f'prompts_export_{timestamp}.csv')
            header_row = ['Name', 'UUID'] # Start with common fields
            if include_author: header_row.append('Author')  # noqa: E701
            if include_details: header_row.append('Details')  # noqa: E701
            if include_system: header_row.append('System Prompt')  # noqa: E701
            if include_user: header_row.append('User Prompt')  # noqa: E701
            if include_associated_keywords: header_row.append('Keywords')  # noqa: E701

            with open(temp_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header_row)
                for p_data in prompts_data:
                    row_to_write = [p_data['name'], p_data['uuid']]
                    if include_author: row_to_write.append(p_data.get('author', ''))  # noqa: E701
                    if include_details: row_to_write.append(p_data.get('details', ''))  # noqa: E701
                    if include_system: row_to_write.append(p_data.get('system_prompt', ''))  # noqa: E701
                    if include_user: row_to_write.append(p_data.get('user_prompt', ''))  # noqa: E701
                    if include_associated_keywords:
                        row_to_write.append(', '.join(p_data.get('keywords_list', [])))
                    writer.writerow(row_to_write)
            output_file_path = temp_csv_file
            status_msg = f"Successfully exported {len(prompts_data)} prompts to CSV."

        elif export_format == 'markdown':
            temp_zip_dir = tempfile.mkdtemp()
            zip_file_path = os.path.join(tempfile.gettempdir(), f'prompts_export_markdown_{timestamp}.zip')

            templates = {
                "Basic Template": """# {name} ({uuid})
{author_section}
{details_section}
{system_section}
{user_section}
{keywords_section}
""",
                "Detailed Template": """# {name}
**UUID**: {uuid}

## Author
{author_section}

## Description
{details_section}

## System Prompt
```
{system_prompt_content}
```

## User Prompt
```
{user_prompt_content}
```

## Keywords
{keywords_section}
"""
            }
            chosen_template_str = templates.get(markdown_template_name, templates["Basic Template"])

            with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for p_data in prompts_data:
                    author_sec = f"**Author**: {p_data['author']}" if include_author and p_data.get('author') else ""
                    details_sec = f"**Details**: {p_data['details']}" if include_details and p_data.get('details') else ""
                    system_sec = f"**System Prompt**:\n```\n{p_data['system_prompt']}\n```" if include_system and p_data.get('system_prompt') else ""
                    user_sec = f"**User Prompt**:\n```\n{p_data['user_prompt']}\n```" if include_user and p_data.get('user_prompt') else ""
                    keywords_sec = f"**Keywords**: {', '.join(p_data['keywords_list'])}" if include_associated_keywords and p_data.get('keywords_list') else ""

                    md_content = chosen_template_str.format(
                        name=p_data['name'],
                        uuid=p_data['uuid'],
                        author_section=author_sec,
                        details_section=details_sec,
                        system_section=system_sec, # For Basic Template direct injection
                        system_prompt_content=p_data.get('system_prompt', ''), # For Detailed Template
                        user_section=user_sec, # For Basic Template direct injection
                        user_prompt_content=p_data.get('user_prompt', ''), # For Detailed Template
                        keywords_section=keywords_sec
                    ).strip() # Clean up extra newlines if sections are empty

                    safe_filename = re.sub(r'[^\w\-_ \.]', '_', p_data['name']) + ".md"
                    md_file_path_in_zip_dir = os.path.join(temp_zip_dir, safe_filename)
                    with open(md_file_path_in_zip_dir, 'w', encoding='utf-8') as md_file:
                        md_file.write(md_content)
                    zipf.write(md_file_path_in_zip_dir, arcname=safe_filename)

            output_file_path = zip_file_path
            status_msg = f"Successfully exported {len(prompts_data)} prompts to Markdown in a ZIP file."
        else:
            raise ValueError(f"Unsupported export_format: {export_format}. Must be 'csv' or 'markdown'.")  # noqa: TRY003, TRY301

        logging.info(status_msg)
        return status_msg, output_file_path  # noqa: TRY300

    except (DatabaseError, sqlite3.Error, ValueError) as e:
        error_msg = f"Error exporting prompts: {e}"
        logging.error(error_msg, exc_info=True)
        return error_msg, "None"
    except _PROMPTS_NONCRITICAL_EXCEPTIONS as e: # Catch any other unexpected error
        error_msg = f"Unexpected error exporting prompts: {e}"
        logging.error(error_msg, exc_info=True)
        return error_msg, "None"
