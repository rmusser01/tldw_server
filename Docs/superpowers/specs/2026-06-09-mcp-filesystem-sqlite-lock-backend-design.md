# MCP Filesystem SQLite Lock Backend Design

Date: 2026-06-09
Status: Draft for spec review
Backlog: TASK-2344

## Summary

Add a persistent SQLite-backed implementation of the existing MCP filesystem lock manager seam, while keeping the base `tldw_server` integration intentionally small. The default `tldw_server` behavior remains the existing in-memory lock manager. Operators opt into SQLite by setting `lock_manager_backend=sqlite` and providing a lock database path.

This is not a new lock framework in the host server. The durable backend belongs to the standalone `mcp_unified` package boundary so the same lock primitives can be used by the future standalone MCP gateway and by `tldw_server` without duplicating lock semantics.

## Goals

- Preserve the existing `fs.lock_acquire`, `fs.lock_release`, `fs.edit`, `fs.write`, and `fs.patch` behavior for memory-backed deployments.
- Move reusable lock models, exceptions, protocol, memory backend, and factory into `mcp_unified.filesystem_locks`.
- Add an optional `SQLiteFilesystemLockManager` under the standalone package using SQLAlchemy Core, not raw `sqlite3`.
- Keep `tldw_Server_API.app.core.MCP_unified.modules.implementations.filesystem_locks` as a compatibility re-export.
- Keep `FilesystemModule` integration to an import swap plus config passthrough.
- Support multiple processes on the same host coordinating through one SQLite database file.
- Keep all returned paths workspace-relative and keep conflict responses free of absolute paths, process IDs, or raw context metadata.

## Non-Goals

- No broad base-server DB refactor.
- No new `BaseModule` abstraction, generic backend registry, or host-wide persistence framework.
- No async lock-manager protocol churn. The filesystem module already offloads filesystem tool work through `asyncio.to_thread`, and lock-manager calls continue to run inside that worker thread.
- No distributed lock guarantee across hosts.
- No claim of correctness on unreliable network filesystems.
- No replacement for hash/read-receipt preimage checks. Locks reduce edit races; preimage checks remain authoritative.
- No admin inspection UI in this slice.

## Current Foundation

The merged lock-lease slice already provides:

- `FilesystemLockLease`
- `FilesystemLockManager`
- `FilesystemLockConflict`
- `FilesystemLockMissing`
- `InMemoryFilesystemLockManager`
- `create_filesystem_lock_manager(settings)`

These live today under:

`tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_locks.py`

`FilesystemModule` accepts an injected `lock_manager` and otherwise calls the factory from module settings. Tool execution for lock acquire/release and mutations is already offloaded with `asyncio.to_thread`, so a synchronous SQLAlchemy backend will not block the event loop when reached through the module.

## Package Layout

Add a small package namespace:

```text
mcp_unified/filesystem_locks/
  __init__.py
  models.py
  memory.py
  sqlite.py
```

Responsibilities:

- `models.py`: lease dataclass, protocol, conflict/missing exceptions, shared helpers.
- `memory.py`: existing in-memory implementation with unchanged semantics.
- `sqlite.py`: optional SQLAlchemy-backed implementation.
- `__init__.py`: lightweight exports for common models, memory backend, and the factory. The factory imports `sqlite.py` only when the selected backend is `sqlite`.

The existing `tldw_server` module path becomes a shim:

```python
from mcp_unified.filesystem_locks import (
    FilesystemLockConflict,
    FilesystemLockLease,
    FilesystemLockManager,
    FilesystemLockMissing,
    InMemoryFilesystemLockManager,
    create_filesystem_lock_manager,
)
```

That keeps existing imports and tests working while preventing new durable-backend complexity from living in the host server implementation package. The shim must not star-import a lazy SQLite export if that would force SQLAlchemy to import for memory-only deployments.

## Base Server Boundary

`tldw_server` changes must stay minimal:

- Import `FilesystemLockManager` and `create_filesystem_lock_manager` from `mcp_unified.filesystem_locks`.
- Keep the constructor injection point unchanged.
- Keep existing settings dictionary passthrough.
- Do not add host `DB_Management` dependencies for this standalone package backend.
- Do not add another server-level lock abstraction.
- Update lock tool descriptions from "process-local" to backend-neutral advisory wording.

The host server remains a consumer of the package-level lock API. It should not know whether the selected backend is memory or SQLite beyond validating settings through the package factory.

## Configuration

Supported settings:

- `lock_manager_backend`: `memory`, `in_memory`, or `sqlite`.
- `lock_manager_sqlite_path`: required non-empty path when backend is `sqlite`.
- `lock_manager_sqlite_timeout_seconds`: optional SQLite busy timeout, default `30.0`.
- `lock_manager_cleanup_interval`: optional opportunistic cleanup operation cadence.
- `lock_manager_cleanup_limit`: optional maximum expired rows removed during one cleanup pass.

Fail-closed behavior:

- Missing `lock_manager_backend` defaults to memory.
- Explicit blank, false, zero, or unsupported backend values raise `ValueError`.
- `sqlite` without a usable path raises `ValueError`.
- Missing SQLAlchemy for SQLite raises an actionable import error explaining the `mcp-unified[sqlite]` or `mcp-unified[gateway]` extra.

Path handling:

- SQLite path is expanded with `Path(...).expanduser()`.
- Parent directory is created when possible.
- `:memory:` is not a supported operator configuration for cross-process coordination. Tests may use file-backed temporary databases for shared-manager coverage.

## SQLite Schema

Use SQLAlchemy Core tables, not raw SQL strings or `sqlite3` calls.

Table: `mcp_filesystem_lock_leases`

- `workspace_key` string, primary key part.
- `path` string, primary key part.
- `lease_id` string, non-null.
- `owner` string, non-null.
- `expires_at_epoch_us` integer, non-null.
- `ttl_seconds` integer, non-null.
- `workspace_id` string, nullable.
- `session_id` string, nullable.
- `updated_at_epoch_us` integer, non-null.

Indexes:

- Composite primary key on `(workspace_key, path)`.
- Index on `expires_at_epoch_us` for cleanup.

Use Python wall-clock epoch microseconds for stored timestamps:

```python
now_us = int(time.time() * 1_000_000)
expires_at_epoch_us = now_us + ttl_seconds * 1_000_000
```

Returned `FilesystemLockLease.expires_at` remains float seconds to preserve the public dataclass contract.

`workspace_key` may continue to be the resolved workspace-root string used by the current module. That value is internal storage metadata and may include an absolute path. Tool responses, conflict payloads, logs intended for callers, and docs examples must continue to expose only workspace-relative file paths.

## Lock Semantics

The SQLite backend must match the memory backend behavior.

Acquire without `lease_id`:

- If no active lease exists, create one with a generated token.
- If only an expired lease exists, replace it with a new generated token.
- If an active lease exists, raise `FilesystemLockConflict`.

Acquire with `lease_id`:

- If an active lease exists with the same token, renew it and return `renewed=True`.
- If an active lease exists with a different token, raise `FilesystemLockConflict`.
- If no active lease exists, or the row is expired, raise `FilesystemLockMissing`.

Release:

- If the active row has the supplied token, delete it and return the released lease.
- If no active lease exists, or the row is expired, return `None`.
- If an active lease exists with a different token, raise `FilesystemLockConflict`.

Validate:

- If the active row has the supplied token, return the lease.
- If no active lease exists, or the row is expired, raise `FilesystemLockMissing`.
- If an active lease exists with a different token, raise `FilesystemLockConflict`.

Expired rows may be removed opportunistically, but cleanup is not correctness-critical. Every acquire, release, and validate decision must treat expired rows as inactive even if cleanup has not run.

## Atomicity

The SQLite acquire path must avoid unsafe read-then-write races.

Recommended implementation:

- For acquire without `lease_id`, generate a new token and use SQLite dialect upsert:
  - Insert a new row.
  - On `(workspace_key, path)` conflict, update only when `expires_at_epoch_us <= now_us`.
  - If the write affects one row, return the inserted/replaced lease.
  - If not, select the current row and classify it as active conflict, missing, or expired. The normal non-racing case is active conflict.
- For renewal with `lease_id`, use a conditional update:
  - Update only when key, token, and `expires_at_epoch_us > now_us` match.
  - If one row is updated, return the renewed lease.
  - Otherwise select the current row to distinguish active conflict from missing/expired.
- For release, select the current row inside a transaction, then delete conditionally by key and token. A concurrent already-released row is idempotent and returns `None`.
- For validate, select the current row and classify expired, missing, matching, and conflicting cases.

This keeps the correctness decision in the database write predicate for acquisition, where races matter most.

## Threading And Event Loop Behavior

The lock-manager protocol stays synchronous. `FilesystemModule.execute_tool()` already calls lock acquire/release and mutations inside `asyncio.to_thread`. The SQLite manager must not introduce async methods for this slice.

Tests should prove that `fs.lock_acquire` still executes lockable path checks off the event-loop thread. The SQLite backend itself can be tested synchronously at the package level.

## Packaging And Extras

Update the standalone package metadata:

- Add `mcp_unified.filesystem_locks` to `mcp_unified/pyproject.toml` explicit package list.
- Add matching package directory mapping.
- Keep SQLAlchemy only in the `sqlite` and `gateway` extras.
- Keep `mcp_unified.filesystem_locks` importable without SQLAlchemy installed.
- Update `mcp_unified/package_metadata.py` only if extras or public metadata change.

The root `pyproject.toml` package discovery should already include `mcp_unified.*`, but the standalone package gate depends on the nested `mcp_unified/pyproject.toml`.

## Documentation

Update package-local docs, not only host docs:

- `mcp_unified/USER_GUIDE.md`: document memory vs SQLite lock backends, config keys, and host-local limitation.
- `mcp_unified/README.md`: mention the optional SQLite lock backend in the relevant feature/config area.

The documentation must not claim distributed locking. It should say SQLite coordinates cooperating processes that point at the same local database file. It should also warn that the SQLite database path is operator configuration, not an agent-controlled filesystem tool path.

## Testing

Package-level tests:

- Memory backend still supports acquire, conflict, renewal, expiration, release, and validation.
- SQLite backend supports the same behavior with a temporary file-backed DB.
- Two SQLite manager instances sharing one database observe the same active lease.
- Expired rows do not block acquisition.
- Wrong-token renew and release classify as conflict when another active token exists.
- Expired-token renew returns missing.
- Importing `mcp_unified.filesystem_locks` without importing SQLAlchemy remains possible.

Filesystem module tests:

- Existing lock tests remain green through the compatibility shim.
- `lock_manager_backend=sqlite` plus path creates a SQLite manager through config.
- Unsupported, blank, false, and zero backend values fail closed.
- SQLite backend config missing path fails closed.
- Tool descriptions are backend-neutral.

Package boundary tests:

- `test_mcp_unified_standalone_pyproject_matches_release_metadata`
- Standalone artifact gate tests for metadata, sdist boundary, typed marker, and docs.
- A test or guard that confirms the new package is present in standalone artifacts.

Focused validation commands for the implementation plan:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_filesystem_module.py -q
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py -q
python -m pytest .github/tests/test_mcp_unified_artifact_gate.py -c mcp_unified/pytest-artifact-gate.ini -q
python -m py_compile mcp_unified/filesystem_locks/*.py tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_locks.py tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py
python -m bandit -r mcp_unified/filesystem_locks tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_locks.py tldw_Server_API/app/core/MCP_unified/modules/implementations/filesystem_module.py -f json -o /tmp/bandit_mcp_fs_sqlite_locks.json
git diff --check
```

If artifact-gate dependencies are unavailable locally, record the exact skipped command and reason in the Backlog task.

## Risks And Mitigations

Risk: the package extraction adds abstract complexity to `tldw_server`.

Mitigation: keep the host integration to an import swap, compatibility shim, and config passthrough. No new host framework or DB layer is added.

Risk: SQLite locking is mistaken for distributed locking.

Mitigation: docs, tool metadata, and config comments describe the backend as host-local/shared-file coordination only.

Risk: SQLAlchemy becomes a core package dependency.

Mitigation: keep SQLite imports lazy and covered by import-boundary tests.

Risk: atomic acquisition races under concurrent writers.

Mitigation: implement acquisition through conditional SQLite upsert/update predicates, not through an unlocked read followed by a write.

Risk: standalone artifacts omit the new subpackage.

Mitigation: update `mcp_unified/pyproject.toml` explicit package list and run the artifact gate.

## Open Decisions

None blocking for this slice. Future work may add admin inspection, lock pruning commands, or alternate durable backends, but those should be separate tasks after the SQLite backend is proven.
