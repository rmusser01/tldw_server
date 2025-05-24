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
import hashlib  # For potential future use, not strictly needed for prompts text
import json
import sqlite3
import threading
import time
import uuid  # For UUID generation
import re  # For normalize_keyword
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from math import ceil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

# --- Logging Setup ---
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Custom Exceptions (Mirrors Media_DB_v2) ---
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
        self.identifier = identifier

    def __str__(self):
        base = super().__str__()
        details = []
        if self.entity:
            details.append(f"Entity: {self.entity}")
        if self.identifier:
            details.append(f"ID: {self.identifier}")
        return f"{base} ({', '.join(details)})" if details else base


# --- Database Class ---
class PromptsDatabase:
    _CURRENT_SCHEMA_VERSION = 1

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
            raise ValueError("Client ID cannot be empty or None.")
        self.client_id = client_id

        # Ensure parent directory exists if it's a file-based DB
        if not self.is_memory_db:
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                # Catch potential errors creating the directory (e.g., permissions)
                raise DatabaseError(f"Failed to create database directory {self.db_path.parent}: {e}") from e

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
            raise DatabaseError(f"Prompts Database initialization failed: {e}") from e
        except Exception as e:
            # Catch any other unexpected errors during initialization
            logging.critical(f"FATAL: Unexpected error during Prompts DB Initialization for {self.db_path_str}: {e}", exc_info=True)
            # Attempt cleanup
            self.close_connection() # Important to call this
            # Re-raise as a DatabaseError
            raise DatabaseError(f"Unexpected prompts database initialization error: {e}") from e
        finally:
            # Log completion status based on the flag
            if initialization_successful:
                logging.debug(f"PromptsDatabase initialization completed successfully for {self.db_path_str}")
            else:
                # This path indicates an exception was caught and raised above.
                # Logging here provides context that the __init__ block finished, albeit with failure.
                logging.error(f"PromptsDatabase initialization block finished for {self.db_path_str}, but failed.")

    # --- Connection Management ---
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
                except Exception:
                    pass
                self._local.conn = None

        if is_closed:
            try:
                conn = sqlite3.connect(
                    self.db_path_str,
                    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                    check_same_thread=False,  # Required for threading.local
                    timeout=10  # seconds
                )
                conn.row_factory = sqlite3.Row
                if not self.is_memory_db:
                    conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA foreign_keys = ON;")
                self._local.conn = conn
                logging.debug(
                    f"Opened/Reopened SQLite connection to {self.db_path_str} [Client: {self.client_id}, Thread: {threading.current_thread().name}]")
            except sqlite3.Error as e:
                logging.error(f"Failed to connect to database at {self.db_path_str}: {e}", exc_info=True)
                self._local.conn = None
                raise DatabaseError(f"Failed to connect to database '{self.db_path_str}': {e}") from e
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
                if hasattr(self._local, 'conn'): self._local.conn = None

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
            return cursor
        except sqlite3.IntegrityError as e:
            msg = str(e).lower()
            if "sync error" in msg:  # From our custom triggers
                logging.error(f"Sync Validation Failed: {e}")
                raise e
            else:
                logging.error(f"Integrity error: {query[:200]}... Error: {e}", exc_info=True)
                raise DatabaseError(f"Integrity constraint violation: {e}") from e
        except sqlite3.Error as e:
            logging.error(f"Query failed: {query[:200]}... Error: {e}", exc_info=True)
            raise DatabaseError(f"Query execution failed: {e}") from e

    def execute_many(self, query: str, params_list: List[tuple], *, commit: bool = False) -> Optional[sqlite3.Cursor]:
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

    # --- Transaction Context ---
    @contextmanager
    def transaction(self):
        conn = self.get_connection()
        in_outer = conn.in_transaction
        try:
            if not in_outer:
                conn.execute("BEGIN")
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
            raise e

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
                raise DatabaseError(f"Could not determine schema version: {e}") from e

    _SCHEMA_UPDATE_VERSION_SQL_V1 = "UPDATE schema_version SET version = 1 WHERE version = 0;"

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
                expected_cols = {'id', 'name', 'author', 'details', 'system_prompt', 'user_prompt', 'uuid',
                                 'last_modified', 'version', 'client_id', 'deleted'}
                if not expected_cols.issubset(columns):
                    missing_cols = expected_cols - columns
                    raise SchemaError(f"Validation Error: Prompts table missing columns: {missing_cols}")
                logging.debug("[Schema V1] Prompts table structure validated.")
                cursor_check = conn.execute("SELECT version FROM schema_version LIMIT 1")
                version_in_tx = cursor_check.fetchone()
                if not version_in_tx or version_in_tx['version'] != 1:
                    raise SchemaError("Schema version update did not take effect within transaction.")
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
            raise DatabaseError(f"DB schema V1 setup failed: {e}") from e

    def _initialize_schema(self):
        conn = self.get_connection()
        try:
            current_db_version = self._get_db_version(conn)
            target_version = self._CURRENT_SCHEMA_VERSION
            logging.info(f"Checking DB schema. Current: {current_db_version}, Code supports: {target_version}")

            if current_db_version == target_version:
                logging.debug("Database schema is up to date.")
                try:  # Ensure FTS tables exist
                    conn.executescript(self._FTS_TABLES_SQL)
                    conn.commit()
                    logging.debug("Verified FTS tables exist.")
                except sqlite3.Error as fts_err:
                    logging.warning(f"Could not verify/create FTS tables on correct schema: {fts_err}")
                return

            if current_db_version > target_version:
                raise SchemaError(
                    f"DB schema version ({current_db_version}) is newer than supported ({target_version}).")

            if current_db_version == 0:
                self._apply_schema_v1(conn)
                final_db_version = self._get_db_version(conn)
                if final_db_version != target_version:
                    raise SchemaError(
                        f"Schema migration applied, but final DB version is {final_db_version}, expected {target_version}.")
                logging.info(f"Database schema initialized/migrated to version {target_version}.")
            else:
                # Placeholder for future migrations from v1 to v2, etc.
                raise SchemaError(
                    f"Migration needed from {current_db_version} to {target_version}, but no path defined.")
        except (DatabaseError, SchemaError, sqlite3.Error) as e:
            logging.error(f"Schema initialization/migration failed: {e}", exc_info=True)
            raise DatabaseError(f"Schema initialization failed: {e}") from e

    # --- Internal Helpers ---
    def _get_current_utc_timestamp_str(self) -> str:
        return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    def _generate_uuid(self) -> str:
        return str(uuid.uuid4())

    def _normalize_keyword(self, keyword: str) -> str:
        return re.sub(r'\s+', ' ', keyword.strip().lower())

    def _get_next_version(self, conn: sqlite3.Connection, table: str, id_col: str, id_val: Any) -> Optional[
        Tuple[int, int]]:
        try:
            cursor = conn.execute(f"SELECT version FROM {table} WHERE {id_col} = ? AND deleted = 0", (id_val,))
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
            raise DatabaseError(f"Failed to fetch current version: {e}") from e
        return None

    def _log_sync_event(self, conn: sqlite3.Connection, entity: str, entity_uuid: str, operation: str, version: int,
                        payload: Optional[Dict] = None):
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
            raise DatabaseError(f"Failed to log sync event: {e}") from e

    # --- FTS Helper Methods ---
    def _update_fts_prompt(self, conn: sqlite3.Connection, prompt_id: int, name: str, author: Optional[str],
                           details: Optional[str], system_prompt: Optional[str], user_prompt: Optional[str]):
        try:
            conn.execute(
                "INSERT OR REPLACE INTO prompts_fts (rowid, name, author, details, system_prompt, user_prompt) VALUES (?, ?, ?, ?, ?, ?)",
                (prompt_id, name, author or "", details or "", system_prompt or "", user_prompt or ""))
            logging.debug(f"Updated FTS for Prompt ID {prompt_id}")
        except sqlite3.Error as e:
            logging.error(f"Failed FTS update Prompt ID {prompt_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed FTS update Prompt ID {prompt_id}: {e}") from e

    def _delete_fts_prompt(self, conn: sqlite3.Connection, prompt_id: int):
        try:
            conn.execute("DELETE FROM prompts_fts WHERE rowid = ?", (prompt_id,))
            logging.debug(f"Deleted FTS for Prompt ID {prompt_id}")
        except sqlite3.Error as e:
            logging.error(f"Failed FTS delete Prompt ID {prompt_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed FTS delete Prompt ID {prompt_id}: {e}") from e

    def _update_fts_prompt_keyword(self, conn: sqlite3.Connection, keyword_id: int, keyword: str):
        try:
            conn.execute("INSERT OR REPLACE INTO prompt_keywords_fts (rowid, keyword) VALUES (?, ?)",
                         (keyword_id, keyword))
            logging.debug(f"Updated FTS for PromptKeyword ID {keyword_id}")
        except sqlite3.Error as e:
            logging.error(f"Failed FTS update PromptKeyword ID {keyword_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed FTS update PromptKeyword ID {keyword_id}: {e}") from e

    def _delete_fts_prompt_keyword(self, conn: sqlite3.Connection, keyword_id: int):
        try:
            conn.execute("DELETE FROM prompt_keywords_fts WHERE rowid = ?", (keyword_id,))
            logging.debug(f"Deleted FTS for PromptKeyword ID {keyword_id}")
        except sqlite3.Error as e:
            logging.error(f"Failed FTS delete PromptKeyword ID {keyword_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed FTS delete PromptKeyword ID {keyword_id}: {e}") from e

    # --- Public Mutating Methods ---
    def add_keyword(self, keyword_text: str) -> Tuple[Optional[int], Optional[str]]:
        if not keyword_text or not keyword_text.strip():
            raise InputError("Keyword cannot be empty.")
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
                        if cursor.rowcount == 0: raise ConflictError(
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
                        return kw_id, kw_uuid  # THIS LINE IS REVERTED
                else:  # New keyword
                    new_uuid = self._generate_uuid()
                    new_version = 1
                    cursor.execute(
                        "INSERT INTO PromptKeywordsTable (keyword, uuid, last_modified, version, client_id, deleted) VALUES (?, ?, ?, ?, ?, 0)",
                        (normalized_keyword, new_uuid, current_time, new_version, client_id))
                    kw_id = cursor.lastrowid
                    if not kw_id: raise DatabaseError("Failed to get ID for new prompt keyword.")
                    cursor.execute("SELECT * FROM PromptKeywordsTable WHERE id=?", (kw_id,))
                    payload = dict(cursor.fetchone())
                    self._log_sync_event(conn, 'PromptKeywordsTable', new_uuid, 'create', new_version, payload)
                    self._update_fts_prompt_keyword(conn, kw_id, normalized_keyword)
                    return kw_id, new_uuid
        except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error in add_keyword (prompt) for '{keyword_text}': {e}", exc_info=True)
            if isinstance(e, (InputError, ConflictError, DatabaseError)):
                raise e
            else:
                raise DatabaseError(f"Failed to add/update prompt keyword: {e}") from e

    def get_active_keyword_by_text(self, keyword_text: str) -> Optional[Dict]:
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
        query = "SELECT id, uuid, keyword, last_modified, version, client_id FROM PromptKeywordsTable WHERE keyword = ? AND deleted = 0"
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
                   keywords: Optional[List[str]] = None, overwrite: bool = False) -> Tuple[
        Optional[int], Optional[str], str]:
        if not name or not name.strip():
            raise InputError("Prompt name cannot be empty.")
        name = name.strip()  # Use original case for name, but ensure no leading/trailing spaces

        current_time = self._get_current_utc_timestamp_str()
        client_id = self.client_id

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
                        raise ConflictError(f"Prompt '{name}' already exists.")  # RAISE ERROR
                        #return prompt_id, prompt_uuid, f"Prompt '{name}' already exists. Skipped."

                    # Overwrite or undelete-and-update
                    action_taken = "updated"
                    new_version = current_version + 1
                    update_data = {
                        'name': name, 'author': author, 'details': details, 'system_prompt': system_prompt,
                        'user_prompt': user_prompt,
                        'last_modified': current_time, 'version': new_version, 'client_id': client_id, 'deleted': 0,
                        'uuid': prompt_uuid
                    }
                    cursor.execute("""UPDATE Prompts
                                      SET author=?,
                                          details=?,
                                          system_prompt=?,
                                          user_prompt=?,
                                          last_modified=?,
                                          version=?,
                                          client_id=?,
                                          deleted=0
                                      WHERE id = ?
                                        AND version = ?""",
                                   (author, details, system_prompt, user_prompt, current_time, new_version, client_id,
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
                            raise ConflictError(
                                f"Prompt '{name}' (ID: {prompt_id}) was soft-deleted. Undelete first or ensure overwrite logic handles versioning correctly.",
                                "Prompts", prompt_id)
                        raise ConflictError(f"Failed to update prompt '{name}'.", "Prompts", prompt_id)

                    self._log_sync_event(conn, 'Prompts', prompt_uuid, 'update', new_version, update_data)
                    self._update_fts_prompt(conn, prompt_id, name, author, details, system_prompt, user_prompt)
                else:  # New prompt
                    action_taken = "added"
                    prompt_uuid = self._generate_uuid()
                    new_version = 1
                    insert_data = {
                        'name': name, 'author': author, 'details': details, 'system_prompt': system_prompt,
                        'user_prompt': user_prompt,
                        'uuid': prompt_uuid, 'last_modified': current_time, 'version': new_version,
                        'client_id': client_id, 'deleted': 0
                    }
                    cursor.execute(
                        """INSERT INTO Prompts (name, author, details, system_prompt, user_prompt, uuid, last_modified,
                                                version, client_id, deleted)
                                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                                   (name, author, details, system_prompt, user_prompt, prompt_uuid, current_time, new_version, client_id))
                    prompt_id = cursor.lastrowid
                    if not prompt_id: raise DatabaseError("Failed to get ID for new prompt.")
                    self._log_sync_event(conn, 'Prompts', prompt_uuid, 'create', new_version, insert_data)
                    self._update_fts_prompt(conn, prompt_id, name, author, details, system_prompt, user_prompt)

                if prompt_id and keywords is not None: # keywords can be empty list to remove all
                    self.update_keywords_for_prompt(prompt_id, keywords_list=keywords) # This is an instance method

                msg = f"Prompt '{name}' {action_taken} successfully."
                return prompt_id, prompt_uuid, msg

        except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error adding/updating prompt '{name}': {e}", exc_info=True)
            if isinstance(e, (InputError, ConflictError, DatabaseError)): raise e
            else: raise DatabaseError(f"Failed to process prompt '{name}': {e}") from e

    def update_keywords_for_prompt(self, prompt_id: int, keywords_list: List[str]):
        normalized_new_keywords = sorted(list(set([self._normalize_keyword(k) for k in keywords_list if k and k.strip()])))

        try:
            # This method is called within an existing transaction (e.g. from add_prompt)
            # So, use self.get_connection() but don't start a new transaction here.
            conn = self.get_connection()
            cursor = conn.cursor()

            # Get prompt_uuid for logging
            cursor.execute("SELECT uuid FROM Prompts WHERE id = ? AND deleted = 0", (prompt_id,))
            prompt_info = cursor.fetchone()
            if not prompt_info:
                raise InputError(f"Cannot update keywords: Prompt ID {prompt_id} not found or deleted.")
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

            target_keyword_data: Dict[int, Dict[str,str]] = {} # {keyword_id: {'text': text, 'uuid': uuid}}
            if normalized_new_keywords:
                for kw_text in normalized_new_keywords:
                    # add_keyword is an instance method, it will use the existing transaction
                    kw_id, kw_uuid = self.add_keyword(kw_text)
                    if kw_id and kw_uuid:
                        target_keyword_data[kw_id] = {'text': kw_text, 'uuid': kw_uuid}
                    else:
                        # This should not happen if add_keyword is robust
                        raise DatabaseError(f"Failed to get/add keyword '{kw_text}' during prompt keyword update.")

            target_keyword_ids = set(target_keyword_data.keys())

            ids_to_add = target_keyword_ids - current_keyword_ids
            ids_to_remove = current_keyword_ids - target_keyword_ids
            link_sync_version = 1 # For link/unlink operations, version is on the junction table itself if it had one, or just 1 for the event

            if ids_to_remove:
                remove_placeholders = ','.join('?' * len(ids_to_remove))
                cursor.execute(f"DELETE FROM PromptKeywordLinks WHERE prompt_id = ? AND keyword_id IN ({remove_placeholders})", (prompt_id, *list(ids_to_remove)))
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
            if isinstance(e, (InputError, DatabaseError)): raise e
            else: raise DatabaseError(f"Keyword update failed for prompt {prompt_id}: {e}") from e

    def update_prompt_by_id(self, prompt_id: int, update_data: Dict[str, Any]) -> Tuple[Optional[str], str]:
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
            raise InputError("Prompt name cannot be empty if provided for update.")

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
                        raise ConflictError(
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

                # Always update these
                set_clauses.extend(
                    ["last_modified = ?", "version = ?", "client_id = ?", "deleted = 0"])  # Ensure it's marked active
                params.extend([current_time, new_version, client_id])

                if not set_clauses:  # Nothing to update besides version/timestamp
                    return original_uuid, "No changes detected to update."

                sql_set_clause = ", ".join(set_clauses)
                update_sql = f"UPDATE Prompts SET {sql_set_clause} WHERE id = ? AND version = ?"
                params.extend([prompt_id, current_version])

                cursor.execute(update_sql, tuple(params))

                if cursor.rowcount == 0:
                    raise ConflictError(f"Failed to update prompt ID {prompt_id} (version mismatch or record gone).",
                                        "Prompts", prompt_id)

                # Log sync event
                # Fetch the full updated row for payload
                cursor.execute("SELECT * FROM Prompts WHERE id = ?", (prompt_id,))
                updated_payload = dict(cursor.fetchone())
                self._log_sync_event(conn, 'Prompts', original_uuid, 'update', new_version, updated_payload)

                # Update FTS
                self._update_fts_prompt(conn, prompt_id,
                                        updated_payload['name'], updated_payload.get('author'),
                                        updated_payload.get('details'), updated_payload.get('system_prompt'),
                                        updated_payload.get('user_prompt'))

                # Handle keywords if provided in update_data (assuming 'keywords' is a list of strings)
                if 'keywords' in update_data and isinstance(update_data['keywords'], list):
                    self.update_keywords_for_prompt(prompt_id, update_data['keywords'])  # Call existing method

                return original_uuid, f"Prompt ID {prompt_id} updated successfully to version {new_version}."

        except (InputError, ConflictError, DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error updating prompt ID {prompt_id}: {e}", exc_info=True)
            if isinstance(e, (InputError, ConflictError, DatabaseError)):
                raise e
            raise DatabaseError(f"Failed to update prompt ID {prompt_id}: {e}") from e

    def soft_delete_prompt(self, prompt_id_or_name_or_uuid: Union[int, str]) -> bool:
        current_time = self._get_current_utc_timestamp_str()
        client_id = self.client_id

        col_name = "id"
        if isinstance(prompt_id_or_name_or_uuid, str):
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
                cursor.execute(f"SELECT id, uuid, version FROM Prompts WHERE {col_name} = ? AND deleted = 0", (prompt_id_or_name_or_uuid,))
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
                    raise ConflictError("Prompts", prompt_id)

                delete_payload = {'uuid': prompt_uuid, 'last_modified': current_time, 'version': new_version, 'client_id': client_id, 'deleted': 1}
                self._log_sync_event(conn, 'Prompts', prompt_uuid, 'delete', new_version, delete_payload)
                self._delete_fts_prompt(conn, prompt_id)

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
            if isinstance(e, (ConflictError, DatabaseError)): raise e
            else: raise DatabaseError(f"Failed to soft delete prompt: {e}") from e

    def soft_delete_keyword(self, keyword_text: str) -> bool:
        if not keyword_text or not keyword_text.strip():
            raise InputError("Keyword to delete cannot be empty.")
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
                    raise ConflictError("PromptKeywordsTable", kw_id)

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
            if isinstance(e, (InputError, ConflictError, DatabaseError)): raise e
            else: raise DatabaseError(f"Failed to soft delete prompt keyword: {e}") from e

    # --- Read Methods ---
    def get_prompt_by_id(self, prompt_id: int, include_deleted: bool = False) -> Optional[Dict]:
        query = "SELECT * FROM Prompts WHERE id = ?"
        params = [prompt_id]
        if not include_deleted: query += " AND deleted = 0"
        try:
            cursor = self.execute_query(query, tuple(params))
            result = cursor.fetchone()
            return dict(result) if result else None
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching prompt by ID {prompt_id}: {e}")
            raise DatabaseError(f"Failed fetch prompt by ID: {e}") from e

    def get_prompt_by_uuid(self, prompt_uuid: str, include_deleted: bool = False) -> Optional[Dict]:
        query = "SELECT * FROM Prompts WHERE uuid = ?"
        params = [prompt_uuid]
        if not include_deleted: query += " AND deleted = 0"
        try:
            cursor = self.execute_query(query, tuple(params))
            result = cursor.fetchone()
            return dict(result) if result else None
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching prompt by UUID {prompt_uuid}: {e}")
            raise DatabaseError(f"Failed fetch prompt by UUID: {e}") from e

    def get_prompt_by_name(self, name: str, include_deleted: bool = False) -> Optional[Dict]:
        query = "SELECT * FROM Prompts WHERE name = ?"
        params = [name]
        if not include_deleted: query += " AND deleted = 0"
        try:
            cursor = self.execute_query(query, tuple(params))
            result = cursor.fetchone()
            return dict(result) if result else None
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching prompt by name '{name}': {e}")
            raise DatabaseError(f"Failed fetch prompt by name: {e}") from e

    def list_prompts(self, page: int = 1, per_page: int = 10, include_deleted: bool = False) -> Tuple[List[Dict], int, int, int]:
        if page < 1: raise ValueError("Page number must be >= 1")
        if per_page < 1: raise ValueError("Per page must be >= 1")
        offset = (page - 1) * per_page

        where_clause = "WHERE deleted = 0" if not include_deleted else ""

        try:
            with self.transaction() as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM Prompts {where_clause}")
                total_items = cursor.fetchone()[0]

                results_data = []
                if total_items > 0:
                    # Select desired fields, e.g., id, name, uuid, author
                    query = f"""SELECT id, name, uuid, author, last_modified FROM Prompts
                                {where_clause} ORDER BY last_modified DESC, id DESC
                                LIMIT ? OFFSET ?"""
                    cursor.execute(query, (per_page, offset))
                    results_data = [dict(row) for row in cursor.fetchall()]

            total_pages = ceil(total_items / per_page) if total_items > 0 else 0
            return results_data, total_pages, page, total_items
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error listing prompts: {e}")
            raise DatabaseError(f"Failed to list prompts: {e}") from e

    def fetch_prompt_details(self, prompt_id_or_name_or_uuid: Union[int, str], include_deleted: bool = False) -> Optional[Dict]:
        prompt_data = None
        if isinstance(prompt_id_or_name_or_uuid, int):
            prompt_data = self.get_prompt_by_id(prompt_id_or_name_or_uuid, include_deleted)
        elif isinstance(prompt_id_or_name_or_uuid, str):
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

    def fetch_all_keywords(self, include_deleted: bool = False) -> List[str]:
        query = "SELECT keyword FROM PromptKeywordsTable"
        if not include_deleted: query += " WHERE deleted = 0"
        query += " ORDER BY keyword COLLATE NOCASE"
        try:
            cursor = self.execute_query(query)
            return [row['keyword'] for row in cursor.fetchall()]
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching all prompt keywords: {e}")
            raise DatabaseError("Failed to fetch all prompt keywords") from e

    def fetch_keywords_for_prompt(self, prompt_id: int, include_deleted: bool = False) -> List[str]:
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
            raise DatabaseError(f"Failed to fetch keywords for prompt {prompt_id}") from e

    def search_prompts(self,
                       search_query: Optional[str],
                       search_fields: Optional[List[str]] = None, # e.g. ['name', 'details', 'keywords']
                       page: int = 1,
                       results_per_page: int = 20,
                       include_deleted: bool = False
                       ) -> Tuple[List[Dict[str, Any]], int]:
        if page < 1: raise ValueError("Page must be >= 1")
        if results_per_page < 1: raise ValueError("Results per page must be >= 1")

        if search_query and not search_fields:
            search_fields = ["name", "details", "system_prompt", "user_prompt", "author"] # Default FTS fields
        elif not search_fields:
            search_fields = []

        offset = (page - 1) * results_per_page

        base_select_parts = ["p.id", "p.uuid", "p.name", "p.author", "p.details",
                             "p.system_prompt", "p.user_prompt", "p.last_modified", "p.version", "p.deleted"]
        count_select = "COUNT(DISTINCT p.id)"
        base_from = "FROM Prompts p"
        joins = []
        conditions = []
        params = []

        if not include_deleted:
            conditions.append("p.deleted = 0")

        fts_search_active = False
        if search_query:
            fts_query_parts = []
            if "name" in search_fields: fts_query_parts.append("name")
            if "author" in search_fields: fts_query_parts.append("author")
            if "details" in search_fields: fts_query_parts.append("details")
            if "system_prompt" in search_fields: fts_query_parts.append("system_prompt")
            if "user_prompt" in search_fields: fts_query_parts.append("user_prompt")

            # FTS on prompt fields
            if fts_query_parts:
                fts_search_active = True
                if not any("prompts_fts fts_p" in j_item for j_item in joins):
                    joins.append("JOIN prompts_fts fts_p ON fts_p.rowid = p.id")
                # Build FTS query: field1:query OR field2:query ...
                # For simple matching, just use the query directly if FTS table covers all these.
                # The FTS table definition needs to match these fields.
                # Assuming prompts_fts has 'name', 'author', 'details', 'system_prompt', 'user_prompt'
                conditions.append("fts_p.prompts_fts MATCH ?")
                params.append(search_query) # User provides FTS syntax or simple terms

            # FTS on keywords (if specified in search_fields)
            if "keywords" in search_fields:
                fts_search_active = True
                # Join for keywords
                if not any("PromptKeywordLinks pkl" in j_item for j_item in joins):
                    joins.append("JOIN PromptKeywordLinks pkl ON p.id = pkl.prompt_id")
                if not any("PromptKeywordsTable pkw" in j_item for j_item in joins):
                    joins.append("JOIN PromptKeywordsTable pkw ON pkl.keyword_id = pkw.id AND pkw.deleted = 0")
                if not any("prompt_keywords_fts fts_k" in j_item for j_item in joins):
                    joins.append("JOIN prompt_keywords_fts fts_k ON fts_k.rowid = pkw.id")

                conditions.append("fts_k.prompt_keywords_fts MATCH ?")
                params.append(search_query) # Match against keywords

        order_by_clause_str = "ORDER BY p.last_modified DESC, p.id DESC"
        if fts_search_active:
            # FTS results are naturally sorted by relevance (rank) by SQLite.
            # We can select rank if needed for explicit sorting or display.
            if "fts_p.rank AS relevance_score" not in " ".join(base_select_parts) and "fts_p" in " ".join(joins) :
                base_select_parts.append("fts_p.rank AS relevance_score") # Add if fts_p is used
            elif "fts_k.rank AS relevance_score_kw" not in " ".join(base_select_parts) and "fts_k" in " ".join(joins):
                base_select_parts.append("fts_k.rank AS relevance_score_kw") # Add if fts_k is used
            # A more complex ranking might be needed if both prompt and keyword FTS are active.
            # For now, default sort or rely on SQLite's combined FTS rank if multiple MATCH clauses are used.
            order_by_clause_str = "ORDER BY p.last_modified DESC, p.id DESC" # Fallback, FTS rank is implicit

        final_select_stmt = f"SELECT DISTINCT {', '.join(base_select_parts)}"
        join_clause = " ".join(list(dict.fromkeys(joins))) # Unique joins
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        try:
            count_sql = f"SELECT {count_select} {base_from} {join_clause} {where_clause}"
            count_cursor = self.execute_query(count_sql, tuple(params))
            total_matches = count_cursor.fetchone()[0]

            results_list = []
            if total_matches > 0 and offset < total_matches:
                results_sql = f"{final_select_stmt} {base_from} {join_clause} {where_clause} {order_by_clause_str} LIMIT ? OFFSET ?"
                paginated_params = tuple(params + [results_per_page, offset])
                results_cursor = self.execute_query(results_sql, paginated_params)
                results_list = [dict(row) for row in results_cursor.fetchall()]
                # If keywords need to be attached to each result
                for res_dict in results_list:
                    res_dict['keywords'] = self.fetch_keywords_for_prompt(res_dict['id'], include_deleted=False)

            return results_list, total_matches
        except sqlite3.Error as e:
            if "no such table: prompts_fts" in str(e).lower() or "no such table: prompt_keywords_fts" in str(e).lower():
                logging.error(f"FTS table missing in {self.db_path_str}. Search may fail or be incomplete.")
                # Fallback to LIKE search or raise error
                # For now, let it fail and be caught by generic error.
            logging.error(f"DB error during prompt search in '{self.db_path_str}': {e}", exc_info=True)
            raise DatabaseError(f"Failed to search prompts: {e}") from e

    # --- Sync Log Access Methods ---
    def get_sync_log_entries(self, since_change_id: int = 0, limit: Optional[int] = None) -> List[Dict]:
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
                    try: row_dict['payload'] = json.loads(row_dict['payload'])
                    except json.JSONDecodeError:
                        logging.warning(f"Failed decode JSON payload for sync_log ID {row_dict.get('change_id')}")
                        row_dict['payload'] = None
                results.append(row_dict)
            return results
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error fetching sync_log entries: {e}")
            raise DatabaseError("Failed to fetch sync_log entries") from e

    def delete_sync_log_entries(self, change_ids: List[int]) -> int:
        if not change_ids: return 0
        if not all(isinstance(cid, int) for cid in change_ids):
            raise ValueError("change_ids must be a list of integers.")
        placeholders = ','.join('?' * len(change_ids))
        query = f"DELETE FROM sync_log WHERE change_id IN ({placeholders})"
        try:
            with self.transaction(): # Ensure commit happens
                cursor = self.execute_query(query, tuple(change_ids), commit=False) # commit handled by transaction
                deleted_count = cursor.rowcount
                logger.info(f"Deleted {deleted_count} sync log entries.")
                return deleted_count
        except (DatabaseError, sqlite3.Error) as e:
            logger.error(f"Error deleting sync log entries: {e}")
            raise DatabaseError("Failed to delete sync log entries") from e


# =========================================================================
# Standalone Functions (REQUIRE db_instance passed explicitly)
# =========================================================================
# These functions now operate on a PromptsDatabase instance.

def add_or_update_prompt(db_instance: PromptsDatabase,
                         name: str, author: Optional[str], details: Optional[str],
                         system_prompt: Optional[str] = None, user_prompt: Optional[str] = None,
                         keywords: Optional[List[str]] = None) -> Tuple[Optional[int], Optional[str], str]:
    """
    Adds a new prompt or updates an existing one (identified by name).
    If the prompt exists (even if soft-deleted), it will be updated/undeleted.
    """
    if not isinstance(db_instance, PromptsDatabase):
        raise TypeError("db_instance must be a PromptsDatabase object.")
    # `add_prompt` with overwrite=True handles both add and update logic.
    return db_instance.add_prompt(
        name=name, author=author, details=details,
        system_prompt=system_prompt, user_prompt=user_prompt,
        keywords=keywords, overwrite=True # Key change: always overwrite/update if exists
    )

def load_prompt_details_for_ui(db_instance: PromptsDatabase, prompt_name: str) -> Tuple[str, str, str, str, str, str]:
    """
    Loads prompt details for UI display, fetching by name.
    Returns empty strings if not found.
    """
    if not isinstance(db_instance, PromptsDatabase):
        raise TypeError("db_instance must be a PromptsDatabase object.")
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


def export_prompt_keywords_to_csv(db_instance: PromptsDatabase) -> Tuple[str, str]:
    import csv
    import tempfile
    import os
    from datetime import datetime

    if not isinstance(db_instance, PromptsDatabase):
        raise TypeError("db_instance must be a PromptsDatabase object.")

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
        return status_msg, file_path

    except (DatabaseError, sqlite3.Error) as e:
        error_msg = f"Database error exporting keywords: {e}"
        logging.error(error_msg, exc_info=True)
        return error_msg, "None"
    except Exception as e:
        error_msg = f"Error exporting keywords: {e}"
        logging.error(error_msg, exc_info=True)
        return error_msg, "None"


def view_prompt_keywords_markdown(db_instance: PromptsDatabase) -> str:
    if not isinstance(db_instance, PromptsDatabase):
        raise TypeError("db_instance must be a PromptsDatabase object.")
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
        return "No active keywords found."
    except (DatabaseError, sqlite3.Error) as e:
        error_msg = f"Error retrieving keywords for markdown view: {e}"
        logging.error(error_msg, exc_info=True)
        return error_msg


def export_prompts_formatted(db_instance: PromptsDatabase,
                             export_format: str = 'csv', # 'csv' or 'markdown'
                             filter_keywords: Optional[List[str]] = None,
                             include_system: bool = True,
                             include_user: bool = True,
                             include_details: bool = True,
                             include_author: bool = True,
                             include_associated_keywords: bool = True, # Renamed for clarity
                             markdown_template_name: Optional[str] = "Basic Template" # Name of template
                             ) -> Tuple[str, str]:
    import csv
    import tempfile
    import os
    import zipfile # For markdown if multiple files
    from datetime import datetime

    if not isinstance(db_instance, PromptsDatabase):
        raise TypeError("db_instance must be a PromptsDatabase object.")

    logging.debug(f"export_prompts_formatted (format: {export_format}) from DB: {db_instance.db_path_str}")

    # --- Fetch Prompts Data ---
    # Build base query parts
    select_fields = ["p.id", "p.name", "p.uuid"] # Always include id, name, uuid
    if include_author: select_fields.append("p.author")
    if include_details: select_fields.append("p.details")
    if include_system: select_fields.append("p.system_prompt")
    if include_user: select_fields.append("p.user_prompt")

    query_sql = f"SELECT DISTINCT {', '.join(select_fields)} FROM Prompts p"
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
            if include_author: header_row.append('Author')
            if include_details: header_row.append('Details')
            if include_system: header_row.append('System Prompt')
            if include_user: header_row.append('User Prompt')
            if include_associated_keywords: header_row.append('Keywords')

            with open(temp_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header_row)
                for p_data in prompts_data:
                    row_to_write = [p_data['name'], p_data['uuid']]
                    if include_author: row_to_write.append(p_data.get('author', ''))
                    if include_details: row_to_write.append(p_data.get('details', ''))
                    if include_system: row_to_write.append(p_data.get('system_prompt', ''))
                    if include_user: row_to_write.append(p_data.get('user_prompt', ''))
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
            raise ValueError(f"Unsupported export_format: {export_format}. Must be 'csv' or 'markdown'.")

        logging.info(status_msg)
        return status_msg, output_file_path

    except (DatabaseError, sqlite3.Error, ValueError) as e:
        error_msg = f"Error exporting prompts: {e}"
        logging.error(error_msg, exc_info=True)
        return error_msg, "None"
    except Exception as e: # Catch any other unexpected error
        error_msg = f"Unexpected error exporting prompts: {e}"
        logging.error(error_msg, exc_info=True)
        return error_msg, "None"