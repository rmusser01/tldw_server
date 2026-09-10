# AuthNZ Request-Time UsersDB Schema DDL Design

**Task:** TASK-12020.43

**Date:** 2026-07-15

**Status:** Approved for implementation planning

## Problem

`AuthnzUsersRepo` creates a fresh `UsersDB` and calls `initialize()` for every
repository operation. `UsersDB.initialize()` opens a transaction and runs users
table DDL. During a live multi-user PostgreSQL workspace load, concurrent
authenticated requests entered this DDL path and PostgreSQL selected one as a
deadlock victim. The affected notes request returned HTTP 500 even though user
lookup is an ordinary read.

The shared `DatabasePool` already owns connection and base-schema startup.
Repository request paths should consume that ready pool, not repeat schema
bootstrap.

## Design

Extend `UsersDB.initialize()` with a keyword-only `ensure_schema` argument that
defaults to `True`.

- Existing explicit callers keep their current behavior and ensure the users
  table and indexes.
- `AuthnzUsersRepo` constructs `UsersDB` with its injected shared pool and calls
  `initialize(ensure_schema=False)`.
- Initialization still obtains the global pool when no pool was injected and
  marks the `UsersDB` instance ready.
- Queries continue through existing `UsersDB` methods and `DatabasePool`; no SQL
  is duplicated in the repository.

This is intentionally narrower than changing all `UsersDB` callers. Callers
that explicitly instantiate and initialize `UsersDB` remain schema-bootstrap
callers unless they deliberately opt out.

## Data Flow

Startup remains:

1. FastAPI lifespan obtains and initializes the shared `DatabasePool`.
2. `DatabasePool` and AuthNZ startup helpers ensure the configured backend
   schema.
3. Requests begin only after lifespan startup completes.

Authenticated request lookup becomes:

1. `AuthnzUsersRepo.from_pool()` obtains the shared initialized pool.
2. The repository creates a `UsersDB` adapter without running schema DDL.
3. `UsersDB` performs the requested query through the shared pool.

## Failure Behavior

Removing request-time DDL does not add retries or hide database errors. Missing
or invalid schema still fails through the existing `DatabaseError` handling,
which keeps operational failures visible and makes startup/schema management
the correct repair point.

## Alternatives Rejected

1. **Serialize DDL with a process lock.** This only masks unnecessary work,
   adds latency to every request, and cannot coordinate multiple server
   processes.
2. **Retry PostgreSQL deadlocks.** This treats the symptom while retaining DDL
   in ordinary authentication.
3. **Rewrite repository methods with direct SQL.** This duplicates mature
   `UsersDB` behavior and creates a larger compatibility surface.
4. **Mutate `UsersDB._initialized` from the repository.** This is shorter but
   couples the repository to private state and obscures why schema creation is
   skipped.

## Verification

1. Add a focused unit regression where repository lookup succeeds through a
   fake ready pool whose transaction method fails if schema DDL is attempted.
2. Keep existing SQLite and PostgreSQL `UsersDB.initialize()` tests green to
   prove default schema assurance is unchanged.
3. Run concurrent PostgreSQL repository lookups against the task fixture.
4. Restart the patched backend and rerun the recipient Chrome CDP flow.
5. Confirm the notes search no longer returns 500; classify the known shared
   workspace data-plane 404 under TASK-12020.40 rather than this task.

## Scope Boundaries

The AuthNZ scheduler timezone mismatch, MCP policy boolean/integer query, and
shared-workspace data-plane hydration are separate defects. They will be
tracked independently and are not bundled into this fix.
