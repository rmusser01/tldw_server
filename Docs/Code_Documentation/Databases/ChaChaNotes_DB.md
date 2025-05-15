# Characters, Chat & Notes Database

# LLM-generated documentation for the SQLite database used in the ChaChaNotes application. (Needs to be reviewed)

## Table of Contents
1.  [Overview](#overview)
2.  [Key Features](#key-features)
3.  [Getting Started](#getting-started)
4.  [Database Schema](#database-schema)
5.  [Core Concepts](#core-concepts)
    *   [Schema Versioning and Initialization](#schema-versioning-and-initialization)
    *   [Client ID](#client-id)
    *   [Optimistic Locking (Versioning)](#optimistic-locking-versioning)
    *   [Soft Deletes](#soft-deletes)
    *   [Full-Text Search (FTS5)](#full-text-search-fts5)
    *   [Synchronization Log](#synchronization-log)
    *   [Thread Safety](#thread-safety)
    *   [JSON Fields](#json-fields)
6.  [API Reference: `CharactersRAGDB` Class](#api-reference-charactersragdb-class)
    *   [Initialization](#initialization)
    *   [Connection Management](#connection-management)
    *   [Query Execution](#query-execution)
    *   [Transaction Management](#transaction-management)
    *   [Character Card Methods](#character-card-methods)
    *   [Conversation Methods](#conversation-methods)
    *   [Message Methods](#message-methods)
    *   [Keyword Methods](#keyword-methods)
    *   [Keyword Collection Methods](#keyword-collection-methods)
    *   [Note Methods](#note-methods)
    *   [Linking Table Methods](#linking-table-methods)
    *   [Sync Log Methods](#sync-log-methods)
7.  [Custom Exceptions](#custom-exceptions)
8.  [Logging](#logging)

---

## 1. Overview

`ChaChaNotes_DB.py` provides a Python library for managing a SQLite database designed to store Character Cards, Chat Conversations, Messages, and associated Notes and Keywords. It's built with features like schema versioning, full-text search, optimistic locking for concurrent modifications, and a synchronization log to facilitate data syncing between different clients or devices.

The library is intended for applications that require local, structured storage for rich text-based content, with capabilities for efficient searching and robust data management.

---

## 2. Key Features

*   **SQLite Backend:** Uses SQLite for a lightweight, file-based database.
*   **Schema Management:** Includes built-in schema definition (currently at version 3) and initialization.
*   **Thread Safety:** Designed for use in multi-threaded applications using thread-local connections.
*   **Optimistic Locking:** Implements a `version` field and `expected_version` checks for update and delete operations to prevent lost updates in concurrent environments.
*   **Soft Deletes:** Records are marked as `deleted` rather than being physically removed, allowing for potential recovery or audit.
*   **Full-Text Search (FTS5):** Provides FTS capabilities for `character_cards`, `conversations`, `messages`, `notes`, `keywords`, and `keyword_collections` via SQLite's FTS5 extension. FTS updates are primarily handled by SQL triggers.
*   **Synchronization Log:** Automatically logs changes (creates, updates, deletes) to main entity tables into a `sync_log` table using SQL triggers. Link table changes are logged manually by Python methods. This log is essential for implementing data synchronization strategies.
*   **Client ID Tracking:** Associates a `client_id` with data modifications, crucial for sync conflict resolution.
*   **UUIDs for IDs:** Uses UUIDs for primary keys in `conversations`, `messages`, and `notes` for globally unique identification.
*   **JSON Field Support:** Handles serialization and deserialization for specific fields (e.g., `tags`, `extensions` in character cards).
*   **Transaction Management:** Provides a context manager for database transactions.

---

## 3. Getting Started

To use the library, instantiate the `CharactersRAGDB` class, providing the path to the SQLite database file and a unique `client_id` for the application instance.

```python
from ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, InputError, ConflictError
import logging

# Configure logging for the library (optional, but recommended)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# For more detailed library logs:
# logging.getLogger("ChaChaNotes_DB").setLevel(logging.DEBUG)


try:
    # Initialize the database. If the file doesn't exist, it will be created.
    # If it exists, schema version will be checked/initialized.
    db = CharactersRAGDB(db_path="my_app_data.sqlite", client_id="my_unique_client_instance_001")

    # Example: Add a character card
    card_data = {
        "name": "Captain Eva",
        "description": "A fearless space explorer.",
        "personality": "Brave, curious, and witty.",
        "system_prompt": "You are Captain Eva."
    }
    char_id = db.add_character_card(card_data)
    if char_id:
        print(f"Added character 'Captain Eva' with ID: {char_id}")

    # ... perform other database operations ...

except CharactersRAGDBError as e:
    print(f"A database error occurred: {e}")
except InputError as e:
    print(f"Invalid input: {e}")
except ConflictError as e:
    print(f"A conflict occurred (e.g., version mismatch or unique constraint): {e}")
finally:
    if 'db' in locals() and db:
        db.close_connection() # Important to close when done with the instance for a thread.
                              # Or manage connections per-thread if db instance is long-lived.
```

---

## 4. Database Schema

The database schema (currently version 3) consists of the following main tables:

*   `db_schema_version`: Tracks the current schema version.
*   `character_cards`: Stores character profiles (name, description, personality, image, etc.). Includes FTS.
*   `conversations`: Represents chat conversations (title, character association, etc.). Includes FTS.
*   `messages`: Stores individual messages within conversations (sender, content, timestamp, ranking). Includes FTS.
*   `keywords`: Stores unique keywords. Includes FTS.
*   `keyword_collections`: Groups keywords into named collections, possibly hierarchical. Includes FTS.
*   `notes`: Stores general-purpose notes (title, content). Includes FTS.
*   **Linking Tables:**
    *   `conversation_keywords`: Many-to-many link between conversations and keywords.
    *   `collection_keywords`: Many-to-many link between keyword collections and keywords.
    *   `note_keywords`: Many-to-many link between notes and keywords.
*   `sync_log`: Records all data modifications (create, update, delete) with client ID, timestamp, version, and payload for synchronization purposes.

Each main entity table (`character_cards`, `conversations`, `messages`, `notes`, `keywords`, `keyword_collections`) includes:
*   `created_at`: Timestamp of creation.
*   `last_modified`: Timestamp of the last modification.
*   `deleted`: Boolean flag for soft deletes (0 = active, 1 = deleted).
*   `client_id`: Identifier of the client instance that last modified the record.
*   `version`: An integer incremented on each update, used for optimistic locking.

Associated FTS5 virtual tables (e.g., `character_cards_fts`) are used for full-text searching. SQL triggers automatically update these FTS tables and most `sync_log` entries upon CUD operations on the main tables.

---

## 5. Core Concepts

### Schema Versioning and Initialization
The library manages its database schema version via the `db_schema_version` table.
*   `_CURRENT_SCHEMA_VERSION` (currently 3) defines the version the code expects.
*   On `CharactersRAGDB` initialization:
    *   It checks the existing database schema version.
    *   If the DB is new (version 0), it applies the full schema for the `_CURRENT_SCHEMA_VERSION`.
    *   If the DB version matches `_CURRENT_SCHEMA_VERSION`, it proceeds.
    *   If the DB version is older and a migration path is not defined (e.g., code is V3, DB is V1 or V2 with no direct migration logic in `_initialize_schema`), a `SchemaError` is raised.
    *   If the DB version is newer than `_CURRENT_SCHEMA_VERSION`, a `SchemaError` is raised, as the code is not equipped to handle a future schema.

### Client ID
The `client_id` provided during `CharactersRAGDB` initialization is crucial for the `sync_log`. Every modification (create, update, delete) logged in `sync_log` (and written to the entity tables) is stamped with this `client_id`. This allows synchronization systems to identify the origin of changes and helps in conflict resolution strategies.

### Optimistic Locking (Versioning)
To prevent lost updates when multiple clients or threads might modify the same record, the library uses optimistic locking.
*   Each main entity record has a `version` column (integer).
*   When a record is created, its version is typically set to 1.
*   When updating or soft-deleting a record, the method (e.g., `update_character_card`, `soft_delete_message`) requires an `expected_version` parameter.
*   The SQL `UPDATE` statement will include `WHERE id = ? AND version = ?`.
    *   If the record's current version in the database matches `expected_version`, the update/delete proceeds, and the record's `version` is incremented.
    *   If they don't match (meaning another client/thread modified it), the `UPDATE` affects 0 rows. The method then detects this and raises a `ConflictError`.
*   The client application is responsible for fetching the latest version of a record before attempting an update and handling `ConflictError` (e.g., by re-fetching, re-applying changes, or informing the user).

### Soft Deletes
Records are generally not physically deleted from the database. Instead, they are "soft-deleted" by setting their `deleted` column to `1` (True).
*   Most `get_` and `list_` methods automatically filter out soft-deleted records (i.e., they only return records `WHERE deleted = 0`).
*   Search methods also exclude soft-deleted records.
*   Soft-deleting a record is an update operation that also increments its `version` and logs it to `sync_log` as a 'delete' operation type.
*   Some `add_` methods (e.g., `_add_generic_item` for keywords) may "undelete" an existing soft-deleted item if a new item with the same unique key is added, effectively reactivating and updating it.

### Full-Text Search (FTS5)
The library leverages SQLite's FTS5 extension for efficient searching of text content.
*   For tables like `character_cards`, `conversations`, `messages`, `notes`, `keywords`, and `keyword_collections`, corresponding FTS5 virtual tables (e.g., `character_cards_fts`) are created.
*   SQL triggers (e.g., `character_cards_ai`, `character_cards_au`, `character_cards_ad`) are defined in the schema. These triggers automatically synchronize the FTS tables when records in the main tables are inserted, updated, or deleted.
*   This means the Python methods for CUD operations don't need to manually update FTS tables; the database handles it.
*   `search_...` methods query these FTS tables using the `MATCH` operator.

### Synchronization Log
The `sync_log` table is central to enabling data synchronization.
*   **Purpose:** To record every significant change (create, update, delete) made to the data.
*   **Automatic Logging (Triggers):** For `character_cards`, `conversations`, `messages`, `notes`, `keywords`, and `keyword_collections`, SQL triggers automatically insert a new entry into `sync_log` whenever a record is inserted, updated (including soft delete/undelete), or (conceptually) hard deleted.
*   **Manual Logging (Python):** For linking tables (`conversation_keywords`, `collection_keywords`, `note_keywords`), changes (links/unlinks) are logged by the corresponding Python methods (`_manage_link` helper) because they don't have their own `version` or `client_id` columns suitable for complex triggers. These log entries use an operation type of 'create' for linking and 'delete' for unlinking.
*   **Log Entry Content:** Each log entry includes:
    *   `change_id`: Auto-incrementing primary key for the log.
    *   `entity`: The name of the table that was changed (e.g., "messages").
    *   `entity_id`: The ID of the record that was changed.
    *   `operation`: Type of operation ('create', 'update', 'delete').
    *   `timestamp`: When the change occurred.
    *   `client_id`: The ID of the client that made the change.
    *   `version`: The new version of the entity record after the change. For link tables, this is typically set to 1.
    *   `payload`: A JSON string containing the state of the record (or relevant parts for deletes) after the change. BLOB fields like images are typically excluded from the payload for size reasons.
*   **Usage:** A sync system can query `sync_log` entries since its last known `change_id` to get new changes and apply them to another data store.

### Thread Safety
The library is designed to be used in multi-threaded environments:
*   It uses `threading.local()` to store SQLite connections, ensuring each thread has its own independent connection.
*   When connecting, `check_same_thread=False` is used, which is necessary when connections are managed per-thread but might be created by a central manager.
*   The `PRAGMA journal_mode=WAL;` (Write-Ahead Logging) is set for non-memory databases, which improves concurrency and performance.

### JSON Fields
Certain table columns are designed to store structured data as JSON strings:
*   `character_cards`: `alternate_greetings`, `tags`, `extensions`.
*   The library provides helpers (`_ensure_json_string`, `_deserialize_row_fields`) to handle conversion between Python objects (lists/dicts) and JSON strings for these fields.
*   When adding or updating, Python lists/dicts for these fields are automatically converted to JSON strings.
*   When retrieving data, these JSON strings are automatically parsed back into Python lists/dicts.

---

## 6. API Reference: `CharactersRAGDB` Class

### Initialization

```python
class CharactersRAGDB:
    def __init__(self, db_path: Union[str, Path], client_id: str)
```
Initializes the database connection and schema.

*   **Parameters:**
    *   `db_path (Union[str, Path])`: Path to the SQLite database file. Can be `":memory:"` for an in-memory database.
    *   `client_id (str)`: A unique identifier for this client/application instance. Cannot be empty.
*   **Raises:**
    *   `ValueError`: If `client_id` is empty.
    *   `CharactersRAGDBError`: If database directory creation fails or any other initialization error occurs.
    *   `SchemaError`: If schema version mismatch or migration issues occur.

### Connection Management

```python
    def get_connection(self) -> sqlite3.Connection
```
Returns the thread-local SQLite connection. Manages opening or reopening if necessary.

*   **Returns:** `sqlite3.Connection` - The active connection for the current thread.
*   **Raises:** `CharactersRAGDBError` if connection fails.

```python
    def close_connection(self)
```
Closes the thread-local SQLite connection. If WAL mode is enabled, it attempts a `PRAGMA wal_checkpoint(TRUNCATE)` before closing.

### Query Execution

```python
    def execute_query(self, query: str, params: Optional[Union[tuple, Dict[str, Any]]] = None, *, commit: bool = False, script: bool = False) -> sqlite3.Cursor
```
Executes a single SQL query.

*   **Parameters:**
    *   `query (str)`: The SQL query string.
    *   `params (Optional[Union[tuple, Dict[str, Any]]])`: Parameters for the query.
    *   `commit (bool)`: If `True` and not in an explicit transaction, commits the change. Defaults to `False`.
    *   `script (bool)`: If `True`, executes the query as a script (using `executescript`). Defaults to `False`.
*   **Returns:** `sqlite3.Cursor` - The cursor object after execution.
*   **Raises:**
    *   `ConflictError`: If a unique constraint violation occurs.
    *   `CharactersRAGDBError`: For other SQLite errors or query execution failures.

```python
    def execute_many(self, query: str, params_list: List[tuple], *, commit: bool = False) -> Optional[sqlite3.Cursor]
```
Executes a SQL query multiple times with different parameter sets.

*   **Parameters:**
    *   `query (str)`: The SQL query string.
    *   `params_list (List[tuple])`: A list of parameter tuples.
    *   `commit (bool)`: If `True` and not in an explicit transaction, commits the changes. Defaults to `False`.
*   **Returns:** `sqlite3.Cursor` or `None` if `params_list` is empty.
*   **Raises:**
    *   `ConflictError`: If a unique constraint violation occurs during batch execution.
    *   `CharactersRAGDBError`: For other SQLite errors or execution failures.

### Transaction Management

```python
    def transaction(self) -> 'TransactionContextManager'
```
Returns a context manager for database transactions.

*   **Usage:**
    ```python
    with db.transaction() as conn: # conn is the sqlite3.Connection
        # ... execute queries using conn.execute(...) or db.execute_query(...)
        # On successful exit from 'with' block, transaction is committed.
        # If an exception occurs, transaction is rolled back.
    ```
*   **Returns:** `TransactionContextManager` instance.

### Character Card Methods

Handles operations for `character_cards` table.
JSON fields: `alternate_greetings`, `tags`, `extensions`.

```python
    def add_character_card(self, card_data: Dict[str, Any]) -> Optional[int]
```
Adds a new character card. `name` is required. `version` defaults to 1.
*   **Parameters:** `card_data (Dict[str, Any])` - Dictionary with card attributes.
*   **Returns:** `Optional[int]` - The ID of the newly created character card, or `None` on failure before insertion.
*   **Raises:** `InputError`, `ConflictError` (if name exists), `CharactersRAGDBError`.

```python
    def get_character_card_by_id(self, character_id: int) -> Optional[Dict[str, Any]]
```
Retrieves an active character card by its ID.
*   **Parameters:** `character_id (int)`
*   **Returns:** `Optional[Dict[str, Any]]` - Card data or `None` if not found/deleted.
*   **Raises:** `CharactersRAGDBError`.

```python
    def get_character_card_by_name(self, name: str) -> Optional[Dict[str, Any]]
```
Retrieves an active character card by its unique name.
*   **Parameters:** `name (str)`
*   **Returns:** `Optional[Dict[str, Any]]` - Card data or `None` if not found/deleted.
*   **Raises:** `CharactersRAGDBError`.

```python
    def list_character_cards(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]
```
Lists active character cards, ordered by name.
*   **Parameters:** `limit (int)`, `offset (int)`
*   **Returns:** `List[Dict[str, Any]]`
*   **Raises:** `CharactersRAGDBError`.

```python
    def update_character_card(self, character_id: int, card_data: Dict[str, Any], expected_version: int) -> bool
```
Updates an existing character card with optimistic locking. Increments `version`.
*   **Parameters:** `character_id (int)`, `card_data (Dict[str, Any])`, `expected_version (int)`
*   **Returns:** `bool` - `True` if successful.
*   **Raises:** `InputError`, `ConflictError` (version mismatch, not found, deleted, or name conflict if name is changed), `CharactersRAGDBError`.

```python
    def soft_delete_character_card(self, character_id: int, expected_version: int) -> bool
```
Soft-deletes a character card with optimistic locking. Increments `version`.
*   **Parameters:** `character_id (int)`, `expected_version (int)`
*   **Returns:** `bool` - `True` if successful or already deleted.
*   **Raises:** `ConflictError` (version mismatch, not found), `CharactersRAGDBError`.

```python
    def search_character_cards(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]
```
Searches character cards using FTS5 (name, description, personality, scenario, system_prompt).
*   **Parameters:** `search_term (str)`, `limit (int)`
*   **Returns:** `List[Dict[str, Any]]` - Matching active cards.
*   **Raises:** `CharactersRAGDBError`.

### Conversation Methods

Handles operations for `conversations` table. ID is UUID (string).

```python
    def add_conversation(self, conv_data: Dict[str, Any]) -> Optional[str]
```
Adds a new conversation. `character_id` is required. `id` (UUID) can be provided or will be generated. `root_id` defaults to `id` if not provided.
*   **Parameters:** `conv_data (Dict[str, Any])`
*   **Returns:** `Optional[str]` - The ID (UUID) of the new conversation.
*   **Raises:** `InputError`, `ConflictError` (if ID exists), `CharactersRAGDBError`.

```python
    def get_conversation_by_id(self, conversation_id: str) -> Optional[Dict[str, Any]]
```
Retrieves an active conversation by its ID (UUID).
*   **Parameters:** `conversation_id (str)`
*   **Returns:** `Optional[Dict[str, Any]]`
*   **Raises:** `CharactersRAGDBError`.

```python
    def get_conversations_for_character(self, character_id: int, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]
```
Lists active conversations for a given character, ordered by last modified.
*   **Parameters:** `character_id (int)`, `limit (int)`, `offset (int)`
*   **Returns:** `List[Dict[str, Any]]`
*   **Raises:** `CharactersRAGDBError`.

```python
    def update_conversation(self, conversation_id: str, update_data: Dict[str, Any], expected_version: int) -> bool
```
Updates an existing conversation (e.g., title, rating) with optimistic locking.
*   **Parameters:** `conversation_id (str)`, `update_data (Dict[str, Any])`, `expected_version (int)`
*   **Returns:** `bool` - `True` if successful.
*   **Raises:** `InputError`, `ConflictError`, `CharactersRAGDBError`.

```python
    def soft_delete_conversation(self, conversation_id: str, expected_version: int) -> bool
```
Soft-deletes a conversation with optimistic locking.
*   **Parameters:** `conversation_id (str)`, `expected_version (int)`
*   **Returns:** `bool` - `True` if successful or already deleted.
*   **Raises:** `ConflictError`, `CharactersRAGDBError`.

```python
    def search_conversations_by_title(self, title_query: str, character_id: Optional[int] = None, limit: int = 10) -> List[Dict[str, Any]]
```
Searches conversations by title using FTS5. Optionally filters by `character_id`.
*   **Parameters:** `title_query (str)`, `character_id (Optional[int])`, `limit (int)`
*   **Returns:** `List[Dict[str, Any]]`
*   **Raises:** `CharactersRAGDBError`.

### Message Methods

Handles operations for `messages` table. ID is UUID (string).

```python
    def add_message(self, msg_data: Dict[str, Any]) -> Optional[str]
```
Adds a new message to a conversation. `conversation_id`, `sender`, `content` are required. `id` (UUID) can be provided or will be generated.
*   **Parameters:** `msg_data (Dict[str, Any])`
*   **Returns:** `Optional[str]` - The ID (UUID) of the new message.
*   **Raises:** `InputError` (if required fields missing or conversation not found/deleted), `ConflictError` (if ID exists), `CharactersRAGDBError`.

```python
    def get_message_by_id(self, message_id: str) -> Optional[Dict[str, Any]]
```
Retrieves an active message by its ID (UUID).
*   **Parameters:** `message_id (str)`
*   **Returns:** `Optional[Dict[str, Any]]`
*   **Raises:** `CharactersRAGDBError`.

```python
    def get_messages_for_conversation(self, conversation_id: str, limit: int = 100, offset: int = 0, order_by_timestamp: str = "ASC") -> List[Dict[str, Any]]
```
Lists active messages for a conversation, ordered by timestamp.
*   **Parameters:** `conversation_id (str)`, `limit (int)`, `offset (int)`, `order_by_timestamp (str)` ("ASC" or "DESC")
*   **Returns:** `List[Dict[str, Any]]`
*   **Raises:** `InputError` (for invalid order_by_timestamp), `CharactersRAGDBError`.

```python
    def update_message(self, message_id: str, update_data: Dict[str, Any], expected_version: int) -> bool
```
Updates an existing message (content, ranking, parent_message_id) with optimistic locking.
*   **Parameters:** `message_id (str)`, `update_data (Dict[str, Any])`, `expected_version (int)`
*   **Returns:** `bool` - `True` if successful.
*   **Raises:** `InputError`, `ConflictError`, `CharactersRAGDBError`.

```python
    def soft_delete_message(self, message_id: str, expected_version: int) -> bool
```
Soft-deletes a message with optimistic locking.
*   **Parameters:** `message_id (str)`, `expected_version (int)`
*   **Returns:** `bool` - `True` if successful or already deleted.
*   **Raises:** `ConflictError`, `CharactersRAGDBError`.

```python
    def search_messages_by_content(self, content_query: str, conversation_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]
```
Searches messages by content using FTS5. Optionally filters by `conversation_id`.
*   **Parameters:** `content_query (str)`, `conversation_id (Optional[str])`, `limit (int)`
*   **Returns:** `List[Dict[str, Any]]`
*   **Raises:** `CharactersRAGDBError`.

### Keyword Methods

Handles operations for `keywords` table. ID is auto-incrementing integer. Keyword text is unique (case-insensitive).

```python
    def add_keyword(self, keyword_text: str) -> Optional[int]
```
Adds a new keyword or undeletes+updates an existing soft-deleted one.
*   **Parameters:** `keyword_text (str)`
*   **Returns:** `Optional[int]` - The ID of the keyword.
*   **Raises:** `InputError`, `ConflictError` (if active keyword text exists), `CharactersRAGDBError`.

```python
    def get_keyword_by_id(self, keyword_id: int) -> Optional[Dict[str, Any]]
```
Retrieves an active keyword by ID.
*   **Parameters:** `keyword_id (int)`
*   **Returns:** `Optional[Dict[str, Any]]`
*   **Raises:** `CharactersRAGDBError`.

```python
    def get_keyword_by_text(self, keyword_text: str) -> Optional[Dict[str, Any]]
```
Retrieves an active keyword by its text.
*   **Parameters:** `keyword_text (str)`
*   **Returns:** `Optional[Dict[str, Any]]`
*   **Raises:** `CharactersRAGDBError`.

```python
    def list_keywords(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]
```
Lists active keywords, ordered by text (case-insensitive).
*   **Parameters:** `limit (int)`, `offset (int)`
*   **Returns:** `List[Dict[str, Any]]`
*   **Raises:** `CharactersRAGDBError`.

```python
    def soft_delete_keyword(self, keyword_id: int, expected_version: int) -> bool
```
Soft-deletes a keyword with optimistic locking.
*   **Parameters:** `keyword_id (int)`, `expected_version (int)`
*   **Returns:** `bool` - `True` if successful or already deleted.
*   **Raises:** `ConflictError`, `CharactersRAGDBError`.

```python
    def search_keywords(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]
```
Searches keywords by text using FTS5.
*   **Parameters:** `search_term (str)`, `limit (int)`
*   **Returns:** `List[Dict[str, Any]]`
*   **Raises:** `CharactersRAGDBError`.

### Keyword Collection Methods

Handles operations for `keyword_collections` table. ID is auto-incrementing integer. Name is unique (case-insensitive).

```python
    def add_keyword_collection(self, name: str, parent_id: Optional[int] = None) -> Optional[int]
```
Adds a new keyword collection or undeletes+updates an existing soft-deleted one.
*   **Parameters:** `name (str)`, `parent_id (Optional[int])`
*   **Returns:** `Optional[int]` - The ID of the collection.
*   **Raises:** `InputError`, `ConflictError` (if active name exists), `CharactersRAGDBError`.

```python
    def get_keyword_collection_by_id(self, collection_id: int) -> Optional[Dict[str, Any]]
```
Retrieves an active keyword collection by ID.
*   **Parameters:** `collection_id (int)`
*   **Returns:** `Optional[Dict[str, Any]]`
*   **Raises:** `CharactersRAGDBError`.

```python
    def get_keyword_collection_by_name(self, name: str) -> Optional[Dict[str, Any]]
```
Retrieves an active keyword collection by name.
*   **Parameters:** `name (str)`
*   **Returns:** `Optional[Dict[str, Any]]`
*   **Raises:** `CharactersRAGDBError`.

```python
    def list_keyword_collections(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]
```
Lists active keyword collections, ordered by name (case-insensitive).
*   **Parameters:** `limit (int)`, `offset (int)`
*   **Returns:** `List[Dict[str, Any]]`
*   **Raises:** `CharactersRAGDBError`.

```python
    def update_keyword_collection(self, collection_id: int, update_data: Dict[str, Any], expected_version: int) -> bool
```
Updates a keyword collection (name, parent_id) with optimistic locking.
*   **Parameters:** `collection_id (int)`, `update_data (Dict[str, Any])`, `expected_version (int)`
*   **Returns:** `bool` - `True` if successful.
*   **Raises:** `InputError`, `ConflictError` (if name conflicts), `CharactersRAGDBError`.

```python
    def soft_delete_keyword_collection(self, collection_id: int, expected_version: int) -> bool
```
Soft-deletes a keyword collection with optimistic locking.
*   **Parameters:** `collection_id (int)`, `expected_version (int)`
*   **Returns:** `bool` - `True` if successful or already deleted.
*   **Raises:** `ConflictError`, `CharactersRAGDBError`.

```python
    def search_keyword_collections(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]
```
Searches keyword collections by name using FTS5.
*   **Parameters:** `search_term (str)`, `limit (int)`
*   **Returns:** `List[Dict[str, Any]]`
*   **Raises:** `CharactersRAGDBError`.

### Note Methods

Handles operations for `notes` table. ID is UUID (string).

```python
    def add_note(self, title: str, content: str, note_id: Optional[str] = None) -> Optional[str]
```
Adds a new note. `title` and `content` are required. `note_id` (UUID) can be provided or will be generated.
*   **Parameters:** `title (str)`, `content (str)`, `note_id (Optional[str])`
*   **Returns:** `Optional[str]` - The ID (UUID) of the new note.
*   **Raises:** `InputError`, `ConflictError` (if ID exists), `CharactersRAGDBError`.

```python
    def get_note_by_id(self, note_id: str) -> Optional[Dict[str, Any]]
```
Retrieves an active note by its ID (UUID).
*   **Parameters:** `note_id (str)`
*   **Returns:** `Optional[Dict[str, Any]]`
*   **Raises:** `CharactersRAGDBError`.

```python
    def list_notes(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]
```
Lists active notes, ordered by last modified descending.
*   **Parameters:** `limit (int)`, `offset (int)`
*   **Returns:** `List[Dict[str, Any]]`
*   **Raises:** `CharactersRAGDBError`.

```python
    def update_note(self, note_id: str, update_data: Dict[str, Any], expected_version: int) -> bool
```
Updates an existing note (title, content) with optimistic locking.
*   **Parameters:** `note_id (str)`, `update_data (Dict[str, Any])`, `expected_version (int)`
*   **Returns:** `bool` - `True` if successful.
*   **Raises:** `InputError`, `ConflictError`, `CharactersRAGDBError`.

```python
    def soft_delete_note(self, note_id: str, expected_version: int) -> bool
```
Soft-deletes a note with optimistic locking.
*   **Parameters:** `note_id (str)`, `expected_version (int)`
*   **Returns:** `bool` - `True` if successful or already deleted.
*   **Raises:** `ConflictError`, `CharactersRAGDBError`.

```python
    def search_notes(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]
```
Searches notes by title and content using FTS5.
*   **Parameters:** `search_term (str)`, `limit (int)`
*   **Returns:** `List[Dict[str, Any]]`
*   **Raises:** `CharactersRAGDBError`.

### Linking Table Methods

These methods manage associations in many-to-many linking tables. They **manually create `sync_log` entries** for these link/unlink operations.

**Conversation <-> Keyword**
```python
    def link_conversation_to_keyword(self, conversation_id: str, keyword_id: int) -> bool
    def unlink_conversation_from_keyword(self, conversation_id: str, keyword_id: int) -> bool
    def get_keywords_for_conversation(self, conversation_id: str) -> List[Dict[str, Any]]
    def get_conversations_for_keyword(self, keyword_id: int, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]
```

**Collection <-> Keyword**
```python
    def link_collection_to_keyword(self, collection_id: int, keyword_id: int) -> bool
    def unlink_collection_from_keyword(self, collection_id: int, keyword_id: int) -> bool
    def get_keywords_for_collection(self, collection_id: int) -> List[Dict[str, Any]]
    def get_collections_for_keyword(self, keyword_id: int, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]
```

**Note <-> Keyword**
```python
    def link_note_to_keyword(self, note_id: str, keyword_id: int) -> bool # note_id is str (UUID)
    def unlink_note_from_keyword(self, note_id: str, keyword_id: int) -> bool # note_id is str (UUID)
    def get_keywords_for_note(self, note_id: str) -> List[Dict[str, Any]] # note_id is str (UUID)
    def get_notes_for_keyword(self, keyword_id: int, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]
```
*   **Link/Unlink methods:**
    *   **Parameters:** IDs of the two entities to link/unlink.
    *   **Returns:** `bool` - `True` if the link was newly created or successfully removed, `False` if it already existed (for link) or didn't exist (for unlink).
    *   **Raises:** `CharactersRAGDBError` for database issues, `InputError` for invalid operations passed to internal helper.
*   **Get methods:**
    *   **Parameters:** ID of one entity, optional `limit` and `offset`.
    *   **Returns:** `List[Dict[str, Any]]` - List of associated active entities.
    *   **Raises:** `CharactersRAGDBError`.

### Sync Log Methods

```python
    def get_sync_log_entries(self, since_change_id: int = 0, limit: Optional[int] = None, entity_type: Optional[str] = None) -> List[Dict[str, Any]]
```
Retrieves entries from `sync_log` table.
*   **Parameters:**
    *   `since_change_id (int)`: Retrieve entries with `change_id` greater than this value.
    *   `limit (Optional[int])`: Maximum number of entries to return.
    *   `entity_type (Optional[str])`: Filter by entity table name (e.g., "messages").
*   **Returns:** `List[Dict[str, Any]]` - Sync log entries with 'payload' parsed as JSON.
*   **Raises:** `CharactersRAGDBError`.

```python
    def get_latest_sync_log_change_id(self) -> int
```
Gets the highest `change_id` from the `sync_log`.
*   **Returns:** `int` - The maximum `change_id`, or 0 if the log is empty.
*   **Raises:** `CharactersRAGDBError`.

---

## 7. Custom Exceptions

The library defines several custom exceptions:

*   `CharactersRAGDBError(Exception)`: Base exception for all library-specific errors.
    *   `SchemaError(CharactersRAGDBError)`: Raised for schema version mismatches or migration failures.
    *   `ConflictError(CharactersRAGDBError)`: Indicates a conflict, typically due to:
        *   Optimistic locking version mismatch during an update or delete.
        *   Attempting to create a record that violates a UNIQUE constraint (e.g., duplicate name).
        *   The `entity` and `entity_id` attributes may provide more context.
*   `InputError(ValueError)`: Raised for invalid input parameters to methods (e.g., missing required fields, invalid enum values).

---

## 8. Logging

The library uses Python's standard `logging` module. A logger instance is created as:
`logger = logging.getLogger(__name__)` (where `__name__` will be `ChaChaNotes_DB`).

To see logs from this library, configure the logging system in your application.
Example basic configuration:
```python
import logging
logging.basicConfig(level=logging.INFO, # Or logging.DEBUG for more verbose output
                    format='%(asctime)s - %(levelname)s - %(name)s - %(threadName)s - %(message)s')

# To specifically control the library's log level:
# logging.getLogger("ChaChaNotes_DB").setLevel(logging.DEBUG)
```
The library logs various events, including:
*   Initialization status.
*   SQL query execution (at DEBUG level).
*   Transaction begin/commit/rollback (at DEBUG level).
*   Successful operations (e.g., adding a card) at INFO level.
*   Warnings for non-critical issues (e.g., unknown fields in update payload).
*   Errors and critical failures.

---
```