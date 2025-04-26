Okay, let's architect and plan the database synchronization library. This will be a conceptual outline and structure, forming the basis for implementation.

**Phase 1: Architecture & Planning**

**1. Core Requirements & Goals:**

*   Synchronize the SQLite database (using the new schema) between multiple client instances.
*   Leverage the existing `sync_log` table populated by database triggers for local change detection.
*   Handle Create, Update, Delete, Link, Unlink operations.
*   Implement conflict resolution based on `version` and `last_modified`.
*   Be transport-agnostic (allow different ways to send/receive changes, e.g., HTTP API, file sync, P2P).
*   Manage sync state (know what changes have been processed).
*   Provide a clear API for the application to trigger synchronization.

**2. Key Architectural Decisions:**

*   **Centralized vs. P2P:** While P2P is possible, a **centralized approach** (clients syncing with a server/common point) is generally simpler to manage for conflict resolution and ensuring all clients eventually converge. We'll design with a central point in mind, but keep the transport abstract.
*   **Conflict Resolution Strategy:** Start with **Last Write Wins (LWW)** based primarily on `last_modified` timestamp, using `client_id` as a tie-breaker. This is relatively simple to implement with the given schema. More complex strategies (e.g., merging, manual resolution prompts) are outside the initial scope but could be added later.
*   **Change Granularity:** The triggers log changes at the row level. We'll process changes based on these log entries.
*   **State Management:** We need to track the "high water mark" of changes received from the central point (e.g., the timestamp of the last processed change) to avoid reprocessing. This state needs to be persistent.
*   **Library Boundary:**
    *   **In Scope:** Reading local `sync_log`, applying remote changes, basic conflict resolution, managing sync state, providing transport hooks.
    *   **Out of Scope:** The specific implementation of the *transport layer* (HTTP server/client, file watching), UI, user authentication.

**3. Core Components:**

*   **`Database` Class:** Your existing (refactored) class for interacting with the local SQLite DB.
*   **`SyncLogEntry`:** A data structure (like a `dataclass` or `NamedTuple`) to represent a parsed row from the `sync_log`.
*   **`SyncTransport` (Interface/ABC):** Defines *how* changes are fetched from and sent to the central point/peers.
    *   `fetch_changes(since: Optional[datetime]) -> List[SyncLogEntry]`
    *   `send_changes(changes: List[SyncLogEntry]) -> bool`
*   **`ConflictResolver` (Interface/Class):** Defines *how* conflicts are handled.
    *   `resolve(local_data: Optional[Dict], remote_change: SyncLogEntry) -> str`: Returns 'apply_remote', 'keep_local', or potentially 'error'.
*   **`SyncStateManager`:** Handles persistent storage of the last sync timestamp (or equivalent marker).
*   **`SyncManager`:** The central orchestrator class. It uses the other components to perform the sync cycle.

**4. Data Flow / Sync Cycle:**

```
+-----------------+        +-----------------+        +-----------------+
| Client A        |        | Central Point   |        | Client B        |
| (SyncManager)   |        | (API / DB / ...) |        | (SyncManager)   |
+-----------------+        +-----------------+        +-----------------+
        |                        ^        |                        ^
        | 1. Get Local Changes   |        |                        |
        |    (Read sync_log)     |        |                        |
        |----------------------->| 2. Send Changes                |
        |                        |    (via Transport.send_changes)|
        |                        |----------------------->        | Apply Logic
        |                        |                        |    (if received)
        |                        V        |                        |
        | 3. Fetch Remote Changes|        |                        |
        |    (via Transport.fetch|--------|<-----------------------|
        |     _changes(since))   |        | 4. Get & Send Changes  |
        |<-----------------------|        | (B does steps 1 & 2)   |
        |                        |        |                        |
        V                        V        V                        V
 5. Apply Remote Changes
    (Handle Conflicts,
     Update Local DB)

 6. Update Sync State
    (Save last processed
     timestamp)
```

**5. Conflict Resolution Logic (LWW Example):**

When applying a remote `update` or `create` (`SyncLogEntry` `remote_change`) for `entity_uuid`:

1.  Fetch the *current* local record (if any) with the same `entity_uuid`. Get its `local_version`, `local_last_modified`, `local_deleted` status.
2.  **Case 1: Local record doesn't exist.**
    *   If `remote_change.operation` is `delete`, ignore (nothing to delete).
    *   If `remote_change.operation` is `create` or `update` (undelete), apply the remote change (INSERT or UPDATE with `deleted=0`).
3.  **Case 2: Local record exists.**
    *   **Compare `last_modified` timestamps (UTC).**
    *   If `remote_change.timestamp > local_last_modified`: **Apply Remote.** (Update local record with remote payload, version, ts, client_id, deleted status).
    *   If `remote_change.timestamp < local_last_modified`: **Keep Local.** (Ignore remote change, log it).
    *   If `remote_change.timestamp == local_last_modified`:
        *   If `remote_change.operation == 'delete'` and `local_deleted == 0`: **Apply Remote Delete.** (Set `deleted=1`, update meta). Deletes often take precedence or are handled by LWW timestamp.
        *   If `remote_change.operation != 'delete'` and `local_deleted == 1`: **Apply Remote Update/Create.** (Undeletes take precedence if timestamps match).
        *   Otherwise (both updates/creates, same timestamp): Use `client_id` as tie-breaker (e.g., lexicographically higher client ID wins). Apply the winning change.
    *   **Important `deleted` flag interaction:** If the winning change sets `deleted=1`, ensure subsequent *older* updates for the same record are ignored.

**6. Handling Links/Unlinks (`MediaKeywords`):**

*   These require special handling in `apply_remote_changes`.
*   The `payload` contains `media_uuid` and `keyword_uuid`.
*   Look up the local `id` for `Media` based on `media_uuid`.
*   Look up the local `id` for `Keywords` based on `keyword_uuid`.
*   If both local IDs are found:
    *   `link`: `INSERT OR IGNORE INTO MediaKeywords (media_id, keyword_id) VALUES (?, ?)`
    *   `unlink`: `DELETE FROM MediaKeywords WHERE media_id = ? AND keyword_id = ?`
*   If parent records don't exist locally yet, the link/unlink might fail. The operation might need to be deferred or retried after parent records are synced.

**7. Sync State Management:**

*   Store the `timestamp` of the *latest successfully processed change* received from the transport layer.
*   When calling `transport.fetch_changes`, pass this timestamp (`since`) so the central point only sends newer changes.
*   Store this state persistently (e.g., in a simple config file, a dedicated `.sync_state` file, or a small table in the DB).

**Phase 2: Build Plan (Library Structure)**

```
sync_library/
├── __init__.py
├── core.py             # SyncManager class
├── exceptions.py       # Custom exceptions (SyncError, ConflictError, TransportError)
├── state.py            # SyncStateManager class/functions
├── transport.py        # SyncTransport ABC, HttpApiTransport example
├── conflict.py         # ConflictResolver ABC, LastWriteWinsStrategy example
└── models.py           # SyncLogEntry dataclass
```

**Implementation Steps:**

1.  **`models.py`:** Define `SyncLogEntry` dataclass, including parsing the JSON `payload` into a dict.
2.  **`exceptions.py`:** Define custom exceptions.
3.  **`transport.py`:** Define `SyncTransport` ABC. Implement `HttpApiTransport` (or a dummy `FileTransport` for testing) connecting to a placeholder API/file structure.
4.  **`conflict.py`:** Define `ConflictResolver` ABC. Implement `LastWriteWinsStrategy`.
5.  **`state.py`:** Implement `SyncStateManager` (e.g., using `json` file storage for simplicity first).
6.  **`core.py` (`SyncManager`):** This is the bulk of the work.
    *   Implement `__init__`.
    *   Implement `_load_sync_state`, `_save_sync_state`.
    *   Implement `get_local_changes`: Read `sync_log` table WHERE `change_id > last_processed_local_id`, parse rows into `SyncLogEntry` objects. Update `last_processed_local_id`.
    *   Implement `apply_remote_changes`:
        *   Sort incoming changes by timestamp.
        *   Start DB transaction.
        *   Loop through changes:
            *   Call internal `_apply_change(entry)`.
        *   Commit transaction.
        *   If successful, update `last_processed_remote_timestamp` using `SyncStateManager`.
        *   Handle errors and rollback.
    *   Implement `_apply_change(entry)`:
        *   Fetch local record based on `entry.entity` and `entry.entity_uuid`.
        *   Handle create/update/delete/link/unlink logic.
        *   Call `_handle_conflict(local_data, entry)` when necessary.
        *   Execute the appropriate SQL (INSERT/UPDATE/DELETE) based on conflict resolution outcome. Needs careful construction of SQL and parameters based on `entry.payload`. Handle lookups for link/unlink.
    *   Implement `_handle_conflict(local_data, remote_change)`: Call the injected `conflict_resolver.resolve()`.
    *   Implement `synchronize()`:
        1.  `last_sync = state_manager.load_last_sync_timestamp()`
        2.  `remote_changes = transport.fetch_changes(client_id, last_sync)`
        3.  `apply_remote_changes(remote_changes)` (updates last sync timestamp on success)
        4.  `local_changes = get_local_changes()` # Get changes made since *last successful send*
        5.  `transport.send_changes(client_id, local_changes)` (update local state tracking sent changes)
7.  **`__init__.py`:** Expose the main classes/functions for easy import (`SyncManager`, transports, resolvers, exceptions).
8.  **Testing:** Create extensive unit and integration tests. Mock the `SyncTransport`. Test conflict scenarios, different operation types, edge cases (empty logs, first sync), error handling, state persistence.
9.  **Documentation:** Write clear docstrings and potentially a README explaining usage.

This detailed plan provides a solid foundation for building your synchronization library. Remember that the `apply_remote_changes` logic, especially handling potential foreign key issues during application, is the most intricate part.





Architecture Summary:

    Core: A SyncManager orchestrates the process.

    Change Source: Reads the local sync_log table.

    Transport: Abstracted via a SyncTransport interface (e.g., HttpApiTransport).

    State: SyncStateManager tracks the last processed remote timestamp and last sent local change ID.

    Conflict Resolution: Abstracted via ConflictResolver (starting with LastWriteWinsStrategy).

    Local DB Interaction: Uses the existing Database class.

    Data Model: SyncLogEntry dataclass represents log entries.

    Cursors: Use change_id for local log processing, last_modified timestamp (UTC) for fetching remote changes.

    Cascades: Assuming Explicit Cascade is handled by the application calling soft_delete_media (which updates children/unlinks keywords). The sync library applies deletes as recorded.

    Deferred Links: A simple retry mechanism during the apply phase.

    Concurrency: Basic lock within SyncManager to prevent simultaneous sync runs on the same instance.