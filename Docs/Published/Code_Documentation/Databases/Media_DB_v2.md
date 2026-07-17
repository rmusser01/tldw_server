# MediaDatabase

## Overview

The package-native `media_db` layer provides a robust content database tailored for media content and related metadata. It is designed with a focus on multi-instance database management, where each `MediaDatabase` object corresponds to a distinct database (SQLite by default; PostgreSQL is supported via a backend abstraction). A key feature is its internal handling of synchronization metadata and Full-Text Search (FTS) updates, simplifying client-side logic.

Each `MediaDatabase` instance requires a `client_id` upon initialization, which is used to attribute all changes made through that instance. The library automatically logs create, update, delete, link, and unlink operations to an internal `sync_log` table. This log is intended for consumption by external synchronization mechanisms to keep multiple database instances consistent.

## Key Features

*   **Instance-Based:** Each `MediaDatabase` object encapsulates a connection and operations for a specific database file (or in-memory SQLite).
*   **Client ID Tracking:** All data modifications are attributed to a `client_id` provided during `Database` initialization.
*   **Internal Sync Logging:** Automatically records changes to a `sync_log` table, facilitating external synchronization processes. This includes changes to main entities and relationships (e.g., `MediaKeywords` links).
*   **Internal FTS Management:** Transparently manages updates to associated FTS5 tables (`media_fts`, `keyword_fts`) and maintains a `claims_fts` index via triggers.
*   **Schema Versioning:** The database bootstraps and migrates to the current schema version automatically (current version: 5).
*   **Backend Abstraction:** Uses a backend factory to support SQLite (default) and PostgreSQL. Selection is driven by `[Database]` settings in `Config_Files/config.txt` or `TLDW_CONTENT_*` environment variables (see Backend Selection Examples).
*   **Thread-Safety:** Utilizes `threading.local` to provide thread-local database connections, ensuring safe concurrent access from multiple threads.
*   **Soft Deletes:** Implements a soft-delete pattern (`deleted=1`) for most entities, allowing for data recovery and synchronization of deletions.
*   **Optimistic Concurrency Control:** Employs a `version` column in key tables. Update operations typically require the current version of the record, and increment it, failing if a concurrent modification has occurred (`ConflictError`).
*   **Transaction Management:** Provides a `transaction()` context manager for atomic database operations.
*   **Comprehensive CRUD Operations:** Offers methods for creating, reading, updating, and deleting media, keywords, transcripts, document versions (with `safe_metadata`), claims, chunking templates, and media chunks.
*   **Standalone Utility Functions:** Includes functions that operate on a `MediaDatabase` instance for tasks like searching, fetching related data, and database maintenance.

## Dependencies

*   Python Standard Library: `sqlite3`, `threading`, `uuid`, `datetime`, `pathlib`, `json`, `hashlib`, `logging`, `math`, `contextlib`, `typing`.
*   Third-Party:
    *   `PyYAML` (optional, used by `import_obsidian_note_to_db` for parsing frontmatter). Ensure it's installed if this functionality is used.
    *   PostgreSQL backend (optional): if enabled, requires the corresponding driver per backend implementation.

## Custom Exceptions

The library defines several custom exceptions to provide more specific error information:

*   **`DatabaseError(Exception)`**: Base exception for all database-related errors within this library.
*   **`SchemaError(DatabaseError)`**: Raised for schema version mismatches or errors during schema migration/initialization.
*   **`InputError(ValueError)`**: Custom exception for input validation errors (e.g., missing required parameters, invalid formats).
*   **`ConflictError(DatabaseError)`**: Indicates a conflict due to concurrent modification, typically when an entity's `version` number does not match the expected value during an update or delete operation.
    *   `entity` (Optional[str]): The name of the entity/table where the conflict occurred.
    *   `identifier` (Optional[Any]): The ID or UUID of the record that caused the conflict.

## `MediaDatabase` Class

The core of the library, managing all interactions with a specific content database (SQLite by default; PostgreSQL supported via backend).
```python
class MediaDatabase:
    _CURRENT_SCHEMA_VERSION = 5
    # ... schema SQL definitions and FTS/trigger setup
```

### Initialization

```python
def __init__(self, db_path: Union[str, Path], client_id: str, *, backend: Optional[DatabaseBackend] = None, config: Optional[ConfigParser] = None)
```

*   **Purpose**: Initializes a `MediaDatabase` instance, connecting to the specified database (SQLite file path or `':memory:'`). It sets up thread-local connection management and ensures the database schema is initialized/migrated to the current version. If a backend is not provided, a backend is resolved from configuration (defaults to SQLite; can select PostgreSQL when enabled).
*   **Args**:
    *   `db_path (Union[str, Path])`: Path to the database (SQLite file path or `':memory:'`).
    *   `client_id (str)`: A unique identifier for the client application or process instance making changes to this database. This ID is recorded in the `sync_log` and on modified records.
    *   `backend (Optional[DatabaseBackend])`: Explicit backend instance (primarily for tests/DI).
    *   `config (Optional[ConfigParser])`: Optional configuration to help resolve backend when not provided.
*   **Raises**:
    *   `ValueError`: If `client_id` is empty or `None`.
    *   `DatabaseError`: If initialization or schema setup fails.
    *   `SchemaError`: If the existing database schema version is newer than what the code supports, or if migration fails.

Backend selection:
- Default is SQLite.
- To select a backend via environment for content DBs, prefer the `TLDW_CONTENT_*` variables (handled by `content_backend.py`). Examples below.

Backend selection examples:

1) Configure PostgreSQL via DSN
```bash
export TLDW_CONTENT_DB_BACKEND=postgresql
export TLDW_CONTENT_PG_DSN="postgresql://tldw_user:secret@localhost:5432/tldw_content?sslmode=prefer"
```

2) Configure PostgreSQL via discrete parameters
```bash
export TLDW_CONTENT_DB_BACKEND=postgresql
export TLDW_CONTENT_PG_HOST=localhost
export TLDW_CONTENT_PG_PORT=5432
export TLDW_CONTENT_PG_DATABASE=tldw_content
export TLDW_CONTENT_PG_USER=tldw_user
export TLDW_CONTENT_PG_PASSWORD=secret
export TLDW_CONTENT_PG_SSLMODE=prefer
```

Then instantiate normally (backend is auto-resolved):
```python
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
db = MediaDatabase(db_path=str(DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id())), client_id="pg_client")
```

3) Programmatic backend (explicit override)
```python
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

cfg = DatabaseConfig(
    backend_type=BackendType.POSTGRESQL,
    pg_host="localhost", pg_port=5432,
    pg_database="tldw_content", pg_user="tldw_user",
    pg_password="secret", pg_sslmode="prefer",
)
backend = DatabaseBackendFactory.create_backend(cfg)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
db = MediaDatabase(db_path=str(DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id())), client_id="pg_client", backend=backend)
```

4) Configure via config.txt ([Database] section)
```ini
[Database]
# Select backend: sqlite or postgresql
type = postgresql

# Option A: Single DSN
pg_connection_string = postgresql://tldw_user:secret@localhost:5432/tldw_content?sslmode=prefer

# Option B: Discrete parameters (used when no pg_connection_string is provided)
pg_host = localhost
pg_port = 5432
pg_database = tldw_content
pg_user = tldw_user
pg_password = secret
pg_sslmode = prefer

# Pooling
pg_pool_size = 20
pg_max_overflow = 40
pg_pool_timeout = 30.0

# If using SQLite instead
# type = sqlite
# sqlite_path = <USER_DB_BASE_DIR>/<user_id>/Media_DB_v2.db
# sqlite_wal_mode = true
# sqlite_foreign_keys = true
# backup_path = ./tldw_DB_Backups/
```
    *   `SchemaError`: If the existing database schema version is newer than what the code supports, or if migration fails.

### Connection Management

These methods handle the SQLite connection lifecycle.

*   **`get_connection(self) -> sqlite3.Connection`**:
    Returns the active, thread-local SQLite connection. Enables `PRAGMA foreign_keys = ON` and `PRAGMA journal_mode=WAL` (for file-based DBs).
*   **`close_connection(self)`**:
    Closes the current thread's database connection, if open.

### Query Execution

Methods for executing SQL queries.

*   **`execute_query(self, query: str, params: tuple = None, *, commit: bool = False) -> sqlite3.Cursor`**:
    Executes a single SQL query.
    *   `commit`: If `True`, commits the transaction. Usually managed by the `transaction()` context manager.
    *   **Raises**: `DatabaseError` for general SQLite errors. Re-raises `sqlite3.IntegrityError` if a sync validation trigger (defined in the schema) fails, allowing specific handling of these optimistic locking failures.
*   **`execute_many(self, query: str, params_list: List[tuple], *, commit: bool = False) -> Optional[sqlite3.Cursor]`**:
    Executes a SQL query for multiple sets of parameters (e.g., batch inserts).
    *   **Raises**: `TypeError` for invalid `params_list`, `DatabaseError` for SQLite errors.

### Transaction Management

*   **`transaction(self)` (Context Manager)**:
    Provides an atomic block for database operations. Commits on successful completion, rolls back on any exception. Handles nested transactions gracefully (outermost transaction controls commit/rollback).
    *   **Yields**: `sqlite3.Connection`.

### Schema Management (Internal)

These methods are primarily for internal use during initialization.

*   `_initialize_schema(self)`: Checks DB schema version and applies `_apply_schema_v1` if necessary.
*   `_apply_schema_v1(self, conn: sqlite3.Connection)`: Applies Version 1 of the schema (tables, indices, FTS tables, validation triggers) and sets the schema version in the `schema_version` table.

### Internal Helpers (Internal)

These methods support the public API:

*   `_get_current_utc_timestamp_str(self) -> str`: Generates an ISO 8601 UTC timestamp string.
*   `_generate_uuid(self) -> str`: Generates a UUID v4 string.
*   `_get_next_version(self, conn: sqlite3.Connection, table: str, id_col: str, id_val: Any) -> Optional[Tuple[int, int]]`: Fetches current and next sync version for a record.
*   `_log_sync_event(self, conn: sqlite3.Connection, entity: str, entity_uuid: str, operation: str, version: int, payload: Optional[Dict] = None)`: Logs an operation to the `sync_log` table.
*   `_update_fts_media(self, conn: sqlite3.Connection, media_id: int, title: str, content: Optional[str])`: Updates/inserts into `media_fts`.
*   `_delete_fts_media(self, conn: sqlite3.Connection, media_id: int)`: Deletes from `media_fts`.
*   `_update_fts_keyword(self, conn: sqlite3.Connection, keyword_id: int, keyword: str)`: Updates/inserts into `keyword_fts`.
*   `_delete_fts_keyword(self, conn:sqlite3.Connection, keyword_id: int)`: Deletes from `keyword_fts`.

### Search Operations

*   **`search_media_db(self, search_query: Optional[str], search_fields: Optional[List[str]] = None, keywords: Optional[List[str]] = None, media_ids_filter: Optional[List[Union[int, str]]] = None, page: int = 1, results_per_page: int = 20, include_trash: bool = False, include_deleted: bool = False) -> Tuple[List[Dict[str, Any]], int]`**:
    Searches media items.
    *   Supports FTS on `title`, `content` via `media_fts`.
    *   Supports LIKE search on `author`, `type`.
    *   Filters by a list of required `keywords` (all must match).
    *   Filters by a list of `media_ids_filter` if provided.
    *   Applies `is_trash` and `deleted` filters.
    *   Implements pagination.
    *   **Returns**: A tuple: `(results_list, total_matches)`.
    *   **Raises**: `ValueError` for invalid pagination or `media_ids_filter` types. `DatabaseError` if `media_fts` is missing or other DB errors.

### Media and Metadata Management (Mutators)

These methods modify the database content. They typically operate within a transaction, update `last_modified` and `version` columns, log events to `sync_log`, and update FTS tables where appropriate.

*   **`add_keyword(self, keyword: str) -> Tuple[Optional[int], Optional[str]]`**:
    Adds a new keyword (case-insensitive, stored lowercase) or undeletes an existing one. Updates `keyword_fts`. Logs 'create' or 'update' to `sync_log`.
    *   **Returns**: `(keyword_id, keyword_uuid)`.
    *   **Raises**: `InputError`, `ConflictError`, `DatabaseError`.

*   **`add_media_with_keywords(self, *, url: Optional[str] = None, title: Optional[str] = None, media_type: Optional[str] = None, content: Optional[str] = None, keywords: Optional[List[str]] = None, prompt: Optional[str] = None, analysis_content: Optional[str] = None, safe_metadata: Optional[str] = None, transcription_model: Optional[str] = None, author: Optional[str] = None, ingestion_date: Optional[str] = None, overwrite: bool = False, chunk_options: Optional[Dict] = None, chunks: Optional[List[Dict[str, Any]]] = None) -> Tuple[Optional[int], Optional[str], str]`**:
    Adds a new media item or updates an existing one (if `overwrite=True`, based on URL or content hash).
    *   Generates a content hash.
    *   Associates keywords (via `update_keywords_for_media`).
    *   Creates an initial `DocumentVersion` (via `create_document_version`).
    *   If `chunks` are provided, they are saved as `UnvectorizedMediaChunks` (old ones deleted on overwrite).
    *   Updates `media_fts`. Logs 'create'/'update' for Media, and relies on called methods for other sync events.
    *   `ingestion_date`: Defaults to current time if `None`.
    *   **Returns**: `(media_id, media_uuid, status_message)`.
    *   **Raises**: `InputError`, `ConflictError`, `DatabaseError`.

*   **`create_document_version(self, media_id: int, content: str, prompt: Optional[str] = None, analysis_content: Optional[str] = None, safe_metadata: Optional[str] = None) -> Dict[str, Any]`**:
    Creates a new entry in `DocumentVersions`. Assigns the next `version_number` for the `media_id`. Logs 'create' to `sync_log`.
    *   **Returns**: Dict with new version's `id`, `uuid`, `media_id`, `version_number`.
    *   **Raises**: `InputError`, `DatabaseError`.

*   **`update_keywords_for_media(self, media_id: int, keywords: List[str])`**:
    Synchronizes keywords for a media item. Adds new keywords/links and removes outdated ones. Logs 'link'/'unlink' to `sync_log` for `MediaKeywords` changes. Calls `add_keyword` internally.
    *   **Raises**: `InputError`, `DatabaseError`, `ConflictError` (from `add_keyword`).

*   **`soft_delete_media(self, media_id: int, cascade: bool = True) -> bool`**:
    Soft-deletes a Media item (`deleted=1`). Deletes its `media_fts` entry. If `cascade=True`:
    *   Deletes `MediaKeywords` links (logs 'unlink').
    *   Soft-deletes associated `Transcripts`, `MediaChunks`, `UnvectorizedMediaChunks`, `DocumentVersions` (logs 'delete' for each).
    *   **Returns**: `True` on success.
    *   **Raises**: `ConflictError`, `DatabaseError`.

*   **`soft_delete_keyword(self, keyword: str) -> bool`**:
    Soft-deletes a keyword (`deleted=1`). Deletes its `keyword_fts` entry. Removes all `MediaKeywords` links (logs 'unlink').
    *   **Returns**: `True` on success.
    *   **Raises**: `InputError`, `ConflictError`, `DatabaseError`.

*   **`soft_delete_document_version(self, version_uuid: str) -> bool`**:
    Soft-deletes a `DocumentVersion`. Prevents deletion if it's the last active version for the media.
    *   **Returns**: `True` on success.
    *   **Raises**: `InputError`, `ConflictError`, `DatabaseError`.

*   **`mark_as_trash(self, media_id: int) -> bool`**:
    Marks a media item as trash (`is_trash=1`, sets `trash_date`). Logs 'update' to `sync_log`. Does not affect FTS.
    *   **Returns**: `True` on success.
    *   **Raises**: `ConflictError`, `DatabaseError`.

*   **`restore_from_trash(self, media_id: int) -> bool`**:
    Restores a media item from trash (`is_trash=0`, `trash_date=NULL`). Logs 'update' to `sync_log`.
    *   **Returns**: `True` on success.
    *   **Raises**: `ConflictError`, `DatabaseError`.

*   **`rollback_to_version(self, media_id: int, target_version_number: int) -> Dict[str, Any]`**:
    Rolls back Media content to a previous `DocumentVersion` state.
    *   Creates a *new* `DocumentVersion` with the old content.
    *   Updates the main `Media` record's content and `media_fts`.
    *   Logs 'create' for new `DocumentVersion`, 'update' for `Media`.
    *   Prevents rollback to the current latest version number.
    *   **Returns**: Dict with success/error message and new version details.
    *   **Raises**: `ValueError`, `InputError`, `ConflictError`, `DatabaseError`.

*   **`process_unvectorized_chunks(self, media_id: int, chunks: List[Dict[str, Any]], batch_size: int = 100)`**:
    Adds `UnvectorizedMediaChunks` in batches. Logs 'create' for each chunk.
    *   `chunks`: List of dicts, keys: `chunk_text` (or `text`), `chunk_index`, optional: `start_char`, `end_char`, `chunk_type`, `metadata`.
    *   **Raises**: `InputError`, `DatabaseError`, `TypeError` (for metadata JSON).

*   **`add_media_chunk(self, media_id: int, chunk_text: str, start_index: int, end_index: int, chunk_id: str) -> Optional[Dict]`**:
    Adds a single `MediaChunk` record. Logs 'create' to `sync_log`.
    *   **Returns**: Dict with new chunk's `id` and `uuid`.
    *   **Raises**: `InputError`, `DatabaseError`.

*   **`add_media_chunks_in_batches(self, media_id: int, chunks_to_add: List[Dict[str, Any]], batch_size: int = 100) -> int`**:
    Wrapper to add `MediaChunk` records in batches, adapting input format for `batch_insert_chunks`.
    *   `chunks_to_add`: List of dicts, keys: `text`, `start_index`, `end_index`.
    *   **Returns**: Total number of chunks attempted for insertion.
    *   **Raises**: Propagates errors from `batch_insert_chunks`.

*   **`batch_insert_chunks(self, media_id: int, chunks: List[Dict]) -> int`**:
    Inserts a batch of `MediaChunk` records. Logs 'create' for each.
    *   `chunks`: List of dicts, keys: `text` (or `chunk_text`), `metadata` (dict with `start_index`, `end_index`).
    *   **Returns**: Number of chunks prepared for insertion.
    *   **Raises**: `InputError`, `DatabaseError`, `KeyError`.

### Media and Metadata Retrieval (Readers)

These methods fetch data. They generally filter for active records (`deleted=0`) unless specified.

*   **`fetch_all_keywords(self) -> List[str]`**:
    Fetches all active (non-deleted) keywords, sorted.
    *   **Returns**: `List[str]` of keywords.

*   **`fetch_media_for_keywords(self, keywords: List[str], include_trash: bool = False) -> Dict[str, List[Dict[str, Any]]]`**:
    Fetches active, non-deleted media items associated with *each* provided active keyword.
    *   Media items filtered by `deleted=0` and `is_trash` (if `include_trash=False`).
    *   **Returns**: Dict mapping found keywords to lists of their associated media items (basic fields).
    *   **Raises**: `TypeError`, `DatabaseError`.

*   **`get_paginated_media_list(self, page: int = 1, results_per_page: int = 10) -> Tuple[List[Dict[str, Any]], int, int, int]`**:
    Fetches a paginated list of active, non-trashed media items (`id`, `title`, `type`, `uuid`).
    *   **Returns**: `(results_list, total_pages, current_page, total_items)`.
    *   **Raises**: `ValueError`, `DatabaseError`.

*   **`get_media_by_id(self, media_id: int, include_deleted=False, include_trash=False) -> Optional[Dict]`**:
    Retrieves a media item by ID.
*   **`get_media_by_uuid(self, media_uuid: str, include_deleted=False, include_trash=False) -> Optional[Dict]`**:
    Retrieves a media item by UUID.
*   **`get_media_by_url(self, url: str, include_deleted=False, include_trash=False) -> Optional[Dict]`**:
    Retrieves a media item by URL.
*   **`get_media_by_hash(self, content_hash: str, include_deleted=False, include_trash=False) -> Optional[Dict]`**:
    Retrieves a media item by content hash.
*   **`get_media_by_title(self, title: str, include_deleted=False, include_trash=False) -> Optional[Dict]`**:
    Retrieves the first media item (by `last_modified` DESC) matching a title.

*   **`get_all_document_versions(self, media_id: int, include_content: bool = False, include_deleted: bool = False, limit: Optional[int] = None, offset: Optional[int] = 0) -> List[Dict[str, Any]]`**:
    Retrieves document versions for an active media item, with pagination.
    *   **Raises**: `TypeError`, `ValueError`, `DatabaseError`.

*   **`get_paginated_files(self, page: int = 1, results_per_page: int = 50) -> Tuple[List[sqlite3.Row], int, int, int]`**:
    Similar to `get_paginated_media_list` but returns `sqlite3.Row` objects for `id`, `title`, `type`.
    *   **Returns**: `(results_list_of_rows, total_pages, current_page, total_items)`.

### Metadata Search

*   **`search_by_safe_metadata(self, filters: Dict[str, Any], *, logical_op: str = 'AND', case_sensitive: bool = False, limit: Optional[int] = None, offset: int = 0) -> List[Dict[str, Any]]`**:
    Searches `DocumentVersions.safe_metadata` (JSON) and related identifiers using simple equality/prefix filters with optional case sensitivity. Returns matching rows with basic fields.

### Sync Log Management

*   **`get_sync_log_entries(self, since_change_id: int = 0, limit: Optional[int] = None) -> List[Dict]`**:
    Retrieves `sync_log` entries newer than `since_change_id`. Payload is JSON-decoded.
*   **`delete_sync_log_entries(self, change_ids: List[int]) -> int`**:
    Deletes specific `sync_log` entries by `change_id`. Returns count of deleted entries.
*   **`delete_sync_log_entries_before(self, change_id_threshold: int) -> int`**:
    Deletes `sync_log` entries with `change_id <= change_id_threshold`. Returns count.

### Claims API

The `Claims` subsystem stores ingestion-time factual statements associated with media chunks and maintains an FTS index (`claims_fts`) via triggers.

- `upsert_claims(self, claims: List[Dict[str, Any]]) -> int`:
  - Inserts claims in batch. Each item should include: `media_id`, `chunk_index`, `span_start`, `span_end`, `claim_text`, `confidence`, `extractor`, `extractor_version`, `chunk_hash`.
  - Returns number of claims written. Triggers keep `claims_fts` in sync.

- `get_claims_by_media(self, media_id: int, *, limit: int = 100) -> List[Dict[str, Any]]`:
  - Fetches non-deleted claims for a media item (limited count).

- `soft_delete_claims_for_media(self, media_id: int) -> int`:
  - Soft-deletes all claims for a media item. Returns number affected.

- `rebuild_claims_fts(self) -> int`:
  - Rebuilds `claims_fts` from `Claims` table; returns row count in the rebuilt FTS.

Example (Claims):
```python
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
db = MediaDatabase(db_path=str(DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id())), client_id="example")

# Upsert a single claim
claims_written = db.upsert_claims([
    {
        "media_id": 123,
        "chunk_index": 0,
        "span_start": 0,
        "span_end": 42,
        "claim_text": "The video states the launch date is May 2025.",
        "confidence": 0.85,
        "extractor": "simple_rule",
        "extractor_version": "1.0",
        "chunk_hash": "abc123"
    }
])

# Fetch recent claims for the media item
claims = db.get_claims_by_media(123, limit=10)

# Soft-delete all claims for a media item
deleted_count = db.soft_delete_claims_for_media(123)

# After bulk ingest or when triggers were disabled, rebuild FTS
fts_rows = db.rebuild_claims_fts()
print(f"claims_fts rows: {fts_rows}")
```

### Chunking Templates API

Manage reusable chunking templates stored in the `ChunkingTemplates` table. Built-in templates cannot be modified or deleted.

- `create_chunking_template(self, name: str, template_json: str, description: Optional[str] = None, is_builtin: bool = False, tags: Optional[List[str]] = None, user_id: Optional[str] = None) -> Dict[str, Any]`:
  - Creates a template; `name` must be unique among non-deleted templates. Validates `template_json`.

- `get_chunking_template(self, template_id: Optional[int] = None, name: Optional[str] = None, uuid: Optional[str] = None, include_deleted: bool = False) -> Optional[Dict[str, Any]]`:
  - Fetches a template by ID, name, or UUID.

- `list_chunking_templates(self, include_builtin: bool = True, include_custom: bool = True, tags: Optional[List[str]] = None, user_id: Optional[str] = None, include_deleted: bool = False) -> List[Dict[str, Any]]`:
  - Lists templates with optional filters; sorts built-ins first then by name.

- `update_chunking_template(self, template_id: Optional[int] = None, name: Optional[str] = None, uuid: Optional[str] = None, template_json: Optional[str] = None, description: Optional[str] = None, tags: Optional[List[str]] = None) -> bool`:
  - Updates a custom template’s JSON/description/tags. Validates JSON. Returns `True` if updated.

- `delete_chunking_template(self, template_id: Optional[int] = None, name: Optional[str] = None, uuid: Optional[str] = None, hard_delete: bool = False) -> bool`:
  - Soft-deletes by default; `hard_delete=True` permanently removes. Not allowed for built-ins.

- `seed_builtin_templates(self, templates: List[Dict[str, Any]]) -> int`:
  - Seeds built-in templates or restores deleted ones. Returns number created/restored.

Example (Chunking Templates):
```python
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
db = MediaDatabase(db_path=str(DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id())), client_id="example")

# Create a custom template
tmpl = db.create_chunking_template(
    name="Sentences-512",
    template_json='{"method":"sentences","max_tokens":512}',
    description="Sentence chunking up to 512 tokens",
    tags=["default"],
)

# List templates
all_templates = db.list_chunking_templates()

# Update template metadata
db.update_chunking_template(uuid=tmpl["uuid"], description="Refined")

# Soft-delete the template
db.delete_chunking_template(uuid=tmpl["uuid"])  # soft delete
```

## Standalone Functions

These functions require a `MediaDatabase` instance to be passed as the `db_instance` argument. They generally call instance methods or perform read operations.

*   **`get_document_version(db_instance: MediaDatabase, media_id: int, version_number: Optional[int] = None, include_content: bool = True) -> Optional[Dict[str, Any]]`**:
    Gets a specific active document version or the latest active one for an active media item.
    *   **Raises**: `TypeError`, `ValueError`, `DatabaseError`.

*   **`create_incremental_backup(db_path, backup_dir)` / `create_automated_backup(db_path, backup_dir)` / `rotate_backups(backup_dir, max_backups=10)`**:
    Placeholder functions, currently not implemented (log warnings).

*   **`check_database_integrity(db_path: str) -> bool`**:
    Performs `PRAGMA integrity_check;` on the DB file (read-only).
    *   **Returns**: `True` if 'ok', `False` otherwise.

*   **`is_valid_date(date_string: str) -> bool`**:
    Checks if a string is a valid 'YYYY-MM-DD' date.

*   **`check_media_exists(db_instance: MediaDatabase, media_id: Optional[int] = None, url: Optional[str] = None, content_hash: Optional[str] = None) -> Optional[int]`**:
    Checks if an *active* media item exists by ID, URL, or hash.
    *   **Returns**: Media ID if found, else `None`.
    *   **Raises**: `TypeError`, `ValueError`, `DatabaseError`.

*   **`empty_trash(db_instance: MediaDatabase, days_threshold: int) -> Tuple[int, int]`**:
    Permanently removes items from trash older than `days_threshold` by calling `db_instance.soft_delete_media()` for each.
    *   **Returns**: `(processed_count, remaining_in_trash_count)`.
    *   **Raises**: `TypeError`, `ValueError`, `DatabaseError` (can be from `soft_delete_media`).

*   **`check_media_and_whisper_model(*args, **kwargs)`**:
    Deprecated function, logs a warning.

*   **`get_unprocessed_media(db_instance: MediaDatabase) -> List[Dict]`**:
    Retrieves active, non-trashed media items with `vector_processing = 0`.
    *   **Returns**: List of dicts (`id`, `uuid`, `content`, `type`, `title`).
    *   **Raises**: `TypeError`, `DatabaseError`.

*   **`mark_media_as_processed(db_instance: MediaDatabase, media_id: int)`**:
    Sets `vector_processing = 1` for a media item. **Does not update `last_modified`, `version`, or log to `sync_log`**.
    *   **Raises**: `TypeError`, `DatabaseError`.

*   **`ingest_article_to_db_new(db_instance: MediaDatabase, *, url: str, title: str, content: str, author: Optional[str] = None, keywords: Optional[List[str]] = None, summary: Optional[str] = None, ingestion_date: Optional[str] = None, custom_prompt: Optional[str] = None, overwrite: bool = False) -> Tuple[Optional[int], Optional[str], str]`**:
    Wrapper to add/update an article using `db_instance.add_media_with_keywords`. Sets `media_type='article'`.
    *   **Raises**: `TypeError`, `InputError`, `ConflictError`, `DatabaseError`.

*   **`import_obsidian_note_to_db(db_instance: MediaDatabase, note_data: Dict[str, Any]) -> Tuple[Optional[int], Optional[str], str]`**:
    Wrapper to add/update an Obsidian note using `db_instance.add_media_with_keywords`. Sets `media_type='obsidian_note'`. Uses `pyyaml` for frontmatter.
    *   `note_data`: Dict with `title`, `content`, optional `tags`, `frontmatter`, `file_created_date`, `overwrite`.
    *   **Raises**: `TypeError`, `InputError`, `ConflictError`, `DatabaseError`, `ImportError` (if `yaml` missing).

*   **`get_media_transcripts(db_instance: MediaDatabase, media_id: int) -> List[Dict]`**:
    Retrieves all active transcripts for an active media item.
    *   **Raises**: `TypeError`, `DatabaseError`.

*   **`get_latest_transcription(db_instance: MediaDatabase, media_id: int) -> Optional[str]`**:
    Retrieves text of the latest active transcript for an active media item.
    *   **Raises**: `TypeError`, `DatabaseError`.

*   **`get_specific_transcript(db_instance: MediaDatabase, transcript_uuid: str) -> Optional[Dict]`**:
    Retrieves a specific active transcript by UUID (ensuring parent media is active).
    *   **Raises**: `TypeError`, `InputError`, `DatabaseError`.

*   **`get_specific_analysis(db_instance: MediaDatabase, version_uuid: str) -> Optional[str]`**:
    Retrieves `analysis_content` from a specific active `DocumentVersion` (ensuring parent media is active).
    *   **Raises**: `TypeError`, `InputError`, `DatabaseError`.

*   **`get_media_prompts(db_instance: MediaDatabase, media_id: int) -> List[Dict]`**:
    Retrieves non-empty prompts from active `DocumentVersions` for an active media item.
    *   **Returns**: List of dicts (`id`, `uuid`, `content` (prompt), `created_at`, `version_number`).
    *   **Raises**: `TypeError`, `DatabaseError`.

*   **`get_specific_prompt(db_instance: MediaDatabase, version_uuid: str) -> Optional[str]`**:
    Retrieves `prompt` from a specific active `DocumentVersion` (ensuring parent media is active).
    *   **Raises**: `TypeError`, `InputError`, `DatabaseError`.

*   **`soft_delete_transcript(db_instance: MediaDatabase, transcript_uuid: str) -> bool`**:
    Soft-deletes a specific transcript. Calls instance methods for logging.
    *   **Returns**: `True` on success.
    *   **Raises**: `TypeError`, `InputError`, `ConflictError`, `DatabaseError`.

*   **`clear_specific_analysis(db_instance: MediaDatabase, version_uuid: str) -> bool`**:
    Clears `analysis_content` (sets to NULL) for an active `DocumentVersion`. Calls instance methods for logging.
    *   **Returns**: `True` on success.
    *   **Raises**: `TypeError`, `InputError`, `ConflictError`, `DatabaseError`.

*   **`clear_specific_prompt(db_instance: MediaDatabase, version_uuid: str) -> bool`**:
    Clears `prompt` (sets to NULL) for an active `DocumentVersion`. Calls instance methods for logging.
    *   **Returns**: `True` on success.
    *   **Raises**: `TypeError`, `InputError`, `ConflictError`, `DatabaseError`.

*   **`get_chunk_text(db_instance: MediaDatabase, chunk_uuid: str) -> Optional[str]`**:
    Retrieves `chunk_text` from an active `UnvectorizedMediaChunks` (ensuring parent media is active).
    *   **Raises**: `TypeError`, `InputError`, `DatabaseError`.

*   **`get_all_content_from_database(db_instance: MediaDatabase) -> List[Dict[str, Any]]`**:
    Retrieves various fields for all active, non-trashed media items.
    *   **Raises**: `TypeError`, `DatabaseError`.

*   **`permanently_delete_item(db_instance: MediaDatabase, media_id: int) -> bool`**:
    **DANGER**: Hard deletes a media item and its related data via DB cascades. Bypasses soft delete and sync log. Deletes FTS entry.
    *   **Returns**: `True` if deleted.
    *   **Raises**: `TypeError`, `DatabaseError`.

*   **`fetch_keywords_for_media(media_id: int, db_instance: MediaDatabase) -> List[str]`**:
    Fetches active keywords for a specific active media item.
    *   **Raises**: `TypeError`, `DatabaseError`.

*   **`fetch_keywords_for_media_batch(media_ids: List[int], db_instance: MediaDatabase) -> Dict[int, List[str]]`**:
    Fetches active keywords for multiple active media items.
    *   **Returns**: Dict mapping `media_id` to list of keywords.
    *   **Raises**: `TypeError`, `InputError`, `DatabaseError`.

## Schema Overview (v5)

The current database schema consists of the following main tables:

*   **`schema_version`**: Stores the current schema version of the database.
*   **`Media`**: Core table for media items (articles, videos, notes, etc.). Includes metadata like URL, title, content, author, timestamps, content hash, UUID, and flags for `is_trash`, `deleted`. Also tracks `chunking_status` and `vector_processing` states.
*   **`Keywords`**: Stores unique keywords (normalized to lowercase). Includes UUID and sync metadata.
*   **`MediaKeywords`**: Junction table linking `Media` and `Keywords` (many-to-many).
*   **`Transcripts`**: Stores transcriptions for media items, potentially from different models. Includes UUID and sync metadata.
*   **`MediaChunks`**: Stores processed, smaller segments (chunks) of media content, often used for vectorization or focused analysis. Includes UUID and sync metadata.
*   **`UnvectorizedMediaChunks`**: Stores raw chunks of media content before vectorization or further processing. Includes UUID and sync metadata, `chunk_index`, character offsets, and processing status.
*   **`DocumentVersions`**: Stores historical versions of a media item's content, with associated `prompt`, `analysis_content`, and `safe_metadata` (JSON). Includes UUID and sync metadata.
*   **`Claims`**: Stores factual claims extracted at ingestion time, attached to a media chunk (with indices/spans, extractor metadata, confidence). Maintained with FTS via triggers.
*   **`sync_log`**: Records all CUD (Create, Update, Delete) operations, as well as link/unlink operations on entities, along with `client_id`, `timestamp`, `version`, and an optional `payload` (JSON of changed data).
*   **`ChunkingTemplates`**: Stores reusable chunking templates (JSON) with metadata, tags, ownership, and soft-delete/version metadata.
*   **FTS Tables**:
    *   **`media_fts`**: FTS5 virtual table for full-text searching `Media.title` and `Media.content`.
    *   **`keyword_fts`**: FTS5 virtual table for full-text searching `Keywords.keyword`.
    *   **`claims_fts`**: FTS5 virtual table for full-text search over `Claims.claim_text` with triggers to maintain sync.

All primary entities (`Media`, `Keywords`, `Transcripts`, `MediaChunks`, `UnvectorizedMediaChunks`, `DocumentVersions`) include standard columns for synchronization and optimistic locking:
*   `uuid`: A universally unique identifier for the record.
*   `last_modified`: UTC timestamp of the last modification.
*   `version`: An integer incremented on each modification, used for optimistic concurrency.
*   `client_id`: Identifier of the client instance that made the last change.
*   `deleted`: Boolean flag (0 or 1) for soft deletes.
*   `prev_version`: Placeholder for previous version tracking (usage may vary).
*   `merge_parent_uuid`: Placeholder for tracking merge history (usage may vary).

Database triggers validate updates on major tables (`version` must increment by 1, `client_id` must be non-empty, `uuid` cannot change). Additional triggers maintain the `claims_fts` index.

## Example Usage (Conceptual)

```python
from pathlib import Path
from tldw_Server_API.app.core.DB_Management.media_db.errors import ConflictError, InputError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

# Initialize a MediaDatabase instance for a specific user/DB file
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
db_file = DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id())
client_1 = "my_desktop_app_instance_123"
try:
    db = MediaDatabase(db_path=db_file, client_id=client_1)
except Exception as e:
    print(f"Failed to initialize database: {e}")
    raise

# Add a new article
try:
    media_id, media_uuid, msg = db.add_media_with_keywords(
        url="http://example.com/article1",
        title="My First Article",
        media_type="article",
        content="This is the full content of the article...",
        keywords=["tech", "python", "database"],
        safe_metadata='{"source": "web", "lang": "en"}',
        author="John Doe",
        overwrite=False # Don't overwrite if it exists
    )
    print(f"{msg} - ID: {media_id}, UUID: {media_uuid}")

    # Search for it
    results, total = db.search_media_db(search_query="article", keywords=["tech"])
    for item in results:
        print(f"Found: {item['title']}")

except InputError as ie:
    print(f"Input Error: {ie}")
except ConflictError as ce:
    print(f"Conflict Error: {ce} - Record {ce.entity} ID {ce.identifier} was modified by another client.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    db.close_connection() # Close connection for the current thread

# Another client instance (e.g., on a different device or process)
# client_2 = "my_mobile_app_instance_456"
# db_sync_client = MediaDatabase(db_path=db_file, client_id=client_2)
#
# # Fetch sync log entries to process changes made by client_1
# changes = db_sync_client.get_sync_log_entries(since_change_id=0) # Get all changes
# for change in changes:
#     print(f"Sync Event: {change['entity']} {change['operation']} by {change['client_id']}")
#     # ... (logic to apply these changes to another system/DB) ...
#
# db_sync_client.close_connection()
```

## Troubleshooting

- Missing FTS tables (e.g., `media_fts`, `keyword_fts`, `claims_fts`)
  - Symptom: `DatabaseError` mentioning "no such table" or failed search.
  - Fix: Ensure schema initialization completed. Instantiate `MediaDatabase` once with valid config to bootstrap schema. For claims, you can run `rebuild_claims_fts()` after enabling the table.

- Permissions when creating DB directories
  - Symptom: Initialization fails creating the database directory.
  - Fix: Verify the `db_path` parent exists and has write permission, or choose a writable path. In containerized deployments, mount a writable volume.

- Chunking template JSON errors
  - Symptom: `InputError: Invalid template JSON` when creating/updating templates.
  - Fix: Provide valid JSON; prefer double quotes and escape inner quotes.

- Sync validation/optimistic concurrency errors
  - Symptom: `sqlite3.IntegrityError` with trigger messages (e.g., "Version must increment by exactly 1", "UUID cannot be changed").
  - Fix: Use the documented mutators that manage version increments and `last_modified`. Avoid direct SQL updates on versioned tables.

- PostgreSQL connection/config issues
  - Symptom: Connection refused/timeout or invalid DSN.
  - Fix: Use the `TLDW_CONTENT_*` env vars or `[Database]` config as shown in examples. Verify credentials, host reachability, and `pg_sslmode`.

- Upgrades and `safe_metadata`
  - Symptom: Older DBs may lack `safe_metadata` column; insert paths catch and adapt, but queries may fail.
  - Fix: Initialize via current code path to let migrations add new columns, or recreate the DB and re-ingest.

### Common Migration Tasks

- Verify schema version and bootstrap
  - Back up the DB file first.
  - Instantiate `MediaDatabase` once to run schema checks/migrations. Verify with:
    - SQLite: `sqlite3 <USER_DB_BASE_DIR>/<user_id>/Media_DB_v2.db "SELECT version FROM schema_version;"` → should be `5`.

- Add `safe_metadata` to older databases (manual)
  - If you cannot run the app to migrate, add the column manually:
    - SQLite: `sqlite3 <db> "ALTER TABLE DocumentVersions ADD COLUMN safe_metadata TEXT;"`

- Enable Claims and rebuild FTS (SQLite)
  - Initialization creates `Claims` and triggers; if claims_fts is empty or missing, rebuild:
    - Python: `db.rebuild_claims_fts()`
    - Or SQLite: `INSERT INTO claims_fts(rowid, claim_text) SELECT id, claim_text FROM Claims WHERE deleted = 0;`

- Rebuild FTS indices after bulk changes (SQLite)
  - `media_fts`: delete and reinsert rows from `Media`.
  - `keyword_fts`: delete and reinsert rows from `Keywords`.
  - The library updates FTS on mutators; for manual maintenance, script the rebuilds.

- Move to per-user layout
  - Move file to `<USER_DB_BASE_DIR>/<user_id>/Media_DB_v2.db` and update config (`USER_DB_BASE_DIR` in `tldw_Server_API.app.core.config`, override via environment variable or `Config_Files/config.txt`, or content DB settings).

- Migrate SQLite → PostgreSQL
  - Recommended: re-ingest via the library (mutators handle sync/versioning/metadata) into a Postgres-backed `MediaDatabase`.
  - Programmatic: read from SQLite and call `add_media_with_keywords`, `create_document_version`, `process_unvectorized_chunks`, `upsert_claims` on the PG instance.
  - Note: FTS virtual tables are SQLite-specific; Postgres backend uses LIKE/ILIKE fallbacks in search.
