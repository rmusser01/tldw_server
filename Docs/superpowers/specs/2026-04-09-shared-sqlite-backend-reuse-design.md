# Shared SQLite Backend Reuse Design

Date: 2026-04-09
Status: Approved for planning
Owner: Codex brainstorming session
Supersedes: `Docs/superpowers/specs/2026-04-06-sqlite-backend-log-noise-design.md`

## Summary

Reduce repeated SQLite backend churn by fixing the underlying resource lifecycle instead of aggregating logs. The design makes SQLite backend reuse a first-class responsibility of the shared backend factory, so repeated requests for the same normalized SQLite target resolve to one shared backend instance. Wrapper objects remain disposable, but SQLite backend pools become process-scoped shared resources with centralized cleanup.

## Problem

The current log noise is a symptom of repeated backend construction:

- [factory.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/backends/factory.py) still emits a `Creating sqlite backend` event for each `DatabaseBackendFactory.create_backend(...)` call, currently at `DEBUG` in this branch.
- The media request path already has a higher-level cache in [DB_Deps.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py), but repeated SQLite backend creation still occurs when that cache is bypassed, reset, or duplicated by other call sites.
- Multiple DB helpers, including [Collections_DB.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/Collections_DB.py) and [ChaChaNotes_DB.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py), call the shared factory directly and currently assume they own any SQLite backend they construct.

This creates two linked issues:

- low-value repeated backend creation log events
- unnecessary SQLite backend and pool churn for the same DB path/config

## Goals

- Reuse one SQLite backend per normalized SQLite target and effective SQLite policy.
- Reduce `Creating sqlite backend` log noise as a consequence of real reuse.
- Apply the reuse model to both the media DB request path and direct SQLite-using DB helpers.
- Centralize SQLite backend shutdown and reset behavior.
- Preserve existing PostgreSQL behavior.

## Non-Goals

- Redesign PostgreSQL backend ownership.
- Change higher-level wrapper lifetimes or make wrappers singleton objects.
- Introduce log aggregation or rate-limited duplicate summaries as the primary fix.
- Expand this work into unrelated DB abstraction refactors.

## Requirements Confirmed With User

- Do not optimize for log suppression alone.
- Target both the media DB request path and direct factory callers in one design.
- Share SQLite backends by normalized target and config.
- Treat centralized shutdown as the new ownership model.
- Allow helper `close()` paths to become release or no-op style for shared SQLite pools.

## Current State

### Shared reuse exists only in selected places

- [DB_Deps.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py) caches `MediaDbFactory` objects per user.
- [session.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/media_db/runtime/session.py) lets a `MediaDbFactory` own one SQLite backend for a single DB path.

This already proves the codebase can safely reuse SQLite backends in some paths, but the reuse boundary is inconsistent and incomplete.

### Many wrappers still build SQLite backends directly

Representative direct callers include:

- [Collections_DB.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/Collections_DB.py)
- [ChaChaNotes_DB.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py)
- [UserDatabase_v2.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/UserDatabase_v2.py)
- [Watchlists_DB.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/Watchlists_DB.py)
- [Workflows_Scheduler_DB.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/Workflows_Scheduler_DB.py)
- [media_db/runtime/backend_resolution.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/media_db/runtime/backend_resolution.py)

Because these paths all use `DatabaseBackendFactory.create_backend(...)`, the current factory log repeats even when the effective SQLite target is identical.

### Ownership semantics are local, not centralized

Some helpers currently assume that if they resolved the backend, they can close its pool directly in their own `close()` or teardown path. That assumption prevents safe process-wide reuse for SQLite.

## Approaches Considered

### Approach 1: Keep behavior the same and only suppress or aggregate logs

Pros:

- Smallest code change.
- Low risk to runtime behavior.

Cons:

- Does not address repeated backend creation.
- Keeps the system paying the churn cost.
- Turns an implementation issue into a logging policy workaround.

### Approach 2: Fix only the media DB request path

Pros:

- Narrower change set.
- Likely removes a large share of the repeated events.

Cons:

- Leaves direct SQLite helper construction untouched.
- Produces two incompatible ownership models in one subsystem.

### Approach 3: Centralize shared SQLite backend reuse in the factory

Pros:

- Solves both the hot request path and direct caller churn.
- Makes the factory responsible for the resource it creates.
- Reduces log noise naturally because creation only happens on cache miss.

Cons:

- Requires an ownership contract change for wrapper `close()` paths.
- Needs careful reset and shutdown handling.

## Recommendation

Use Approach 3.

The correct boundary is not “aggregate the message” but “stop constructing the same SQLite backend repeatedly.” The shared backend factory already sits at the point where all these callers converge, so it is the right place to define canonical reuse, locking, and cleanup.

## Proposed Design

### 1. Add a factory-owned shared SQLite registry

`DatabaseBackendFactory.create_backend(...)` should keep its current API but change SQLite behavior:

- For PostgreSQL, keep the current create-and-return behavior.
- For SQLite, compute a canonical signature and return the shared backend for that signature.

The SQLite signature should include:

- backend type
- normalized SQLite target
- effective SQLite settings that materially affect connection behavior

For this design, the implementation note is:

- include `sqlite_wal_mode`
- include `sqlite_foreign_keys`
- do not include non-functional correlation fields such as `client_id`
- do not include settings the current SQLite backend does not use to shape connection behavior, such as `pool_size` or `echo`

The signature should use normalized file paths or canonical SQLite URIs so logically identical configurations converge on one key.

### 2. Normalize SQLite identity before lookup

Add a dedicated normalization helper in [factory.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/backends/factory.py) so identity rules are not spread across call sites.

Normalization should distinguish:

- file-backed paths
- `:memory:` databases
- SQLite file URIs

It should preserve meaningful differences but collapse trivial textual ones such as equivalent absolute paths.

Concrete normalization examples:

- `/abs/path/db.sqlite` and `./db.sqlite` resolve to the same canonical file-backed target when they point to the same absolute path
- raw `:memory:` is never merged with `file:sharedmem?mode=memory&cache=shared`
- `file:sharedmem?mode=memory&cache=shared` and the same URI with reordered but equivalent query semantics must normalize to one identity only if the implementation can do so safely and deterministically; otherwise the planner should preserve exact URI identity for shared-cache memory URIs

In-memory policy must be explicit:

- raw `:memory:` remains per-construction and is excluded from shared reuse
- anonymous in-memory targets that only preserve process-local isolation semantics are also excluded from shared reuse
- explicit named shared-cache memory URIs may participate in reuse, but only when their normalized URI identity matches exactly

This keeps the new registry from breaking the current use of `:memory:` as a test and helper isolation boundary.

### 3. Change ownership from local to centralized

Wrapper objects remain short-lived, but the underlying SQLite backend becomes a shared process resource.

That means:

- helper constructors may still call `DatabaseBackendFactory.create_backend(...)`
- wrapper `close()` methods should stop closing shared SQLite pools directly
- wrapper teardown should still perform wrapper-local cleanup, but should release shared backend ownership instead of closing a shared pool
- actual pool shutdown happens only through centralized reset and shutdown paths

To support this cleanly, the factory should expose a release helper for managed backends so higher-level callers do not need to inspect private registry state.

The cleanup boundary must be explicit:

- wrapper-local cleanup remains the wrapper's responsibility
- shared backend pool shutdown becomes the factory/reset responsibility

Wrapper-local cleanup includes behavior such as:

- clearing thread-local references
- releasing request-local or persistent connections back to the pool
- rollback, checkpoint, or other connection-level safety work already performed by the wrapper

The implementation must preserve that local cleanup behavior even when the backend pool itself is shared.

### 4. Keep existing higher-level caches, but make them secondary

The per-user `MediaDbFactory` cache in [DB_Deps.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py) still provides value and should remain.

Its role changes from “the only thing stopping backend churn” to “a higher-level object cache on top of shared backend reuse.”

This gives two benefits:

- repeated request resolution stays cheap
- even if that higher-level cache misses or resets, the same SQLite backend can still be reused underneath

### 5. Make factory logging reflect actual creation

Once SQLite reuse is centralized, `Creating sqlite backend` should only be emitted when a new shared backend is actually created, not when the factory returns an existing one.

No separate aggregation layer is needed. Log noise reduction becomes a direct consequence of real resource reuse.

## Component Responsibilities

### Shared backend factory

Responsible for:

- canonical SQLite signature generation
- thread-safe lookup-or-create for SQLite backends
- managed backend release bookkeeping
- centralized bulk shutdown of managed backends

The new SQLite registry should coexist with the existing named backend cache in [factory.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/backends/factory.py), not create a second competing ownership model. `get_backend(name, ...)` may still cache named handles, but when it resolves a SQLite backend it should land on the same canonical shared backend instance used by direct `create_backend(...)` callers.

There must be one shutdown authority for factory-managed SQLite pools:

- the shared SQLite registry is the source of truth for pool ownership
- the named cache stores references to canonical shared backends but does not independently own them
- centralized shutdown must close each canonical shared SQLite backend once, then clear both the registry and any named references that point at those backends

The factory should also snapshot the effective SQLite config when creating a managed shared backend, rather than keeping a caller-owned mutable `DatabaseConfig` object by reference. The planner should treat the normalized SQLite target and effective SQLite policy as immutable backend identity once the shared backend is created.

### Wrapper and helper classes

Responsible for:

- asking the factory for a backend
- treating returned SQLite backends as shared unless explicitly injected otherwise
- releasing or ignoring shared backend cleanup locally
- keeping their existing schema/setup behavior

The first wave of ownership updates should be limited to the paths already confirmed to own or close SQLite resources directly:

- `MediaDbSession.release_context_connection()` and `MediaDbFactory.close()` in [session.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/media_db/runtime/session.py)
- `CollectionsDatabase.close()` in [Collections_DB.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/Collections_DB.py)
- the `CharactersRAGDB` close and pool-shutdown path in [ChaChaNotes_DB.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py)

Other direct factory callers such as `Watchlists_DB`, `UserDatabase_v2`, and scheduler-adjacent wrappers remain in scope for compatibility review, but they do not need to be part of the first ownership-migration slice unless they are confirmed to close managed shared pools directly.

For the first wave, the intended behavior is:

- `MediaDbSession.release_context_connection()` continues to perform request-local cleanup and must not be reduced to a total no-op
- `MediaDbFactory.close()` stops closing a shared SQLite pool directly and instead delegates to factory-managed release or centralized reset
- `CollectionsDatabase.close()` stops calling `pool.close_all()` for factory-managed SQLite backends
- `CharactersRAGDB` retains its connection-local rollback, checkpoint, and thread-local cleanup while avoiding direct shutdown of a factory-managed shared SQLite pool

Cross-wrapper compatibility must also account for same-thread connection sharing. For callers that use `backend.get_pool().get_connection()`, a shared SQLite backend may imply one underlying SQLite connection per backend per thread. The planner must treat this as an explicit compatibility constraint, not an incidental detail, and validate it before broadening reuse across helper classes that currently assume wrapper-local connection isolation.

### Reset and shutdown paths

Responsible for:

- closing each managed shared backend once
- clearing the registry
- preserving test isolation and process teardown semantics

## Data Flow

### Media DB request path

1. Request code resolves the current user in [DB_Deps.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py).
2. `MediaDbFactory` is fetched or created for that user.
3. The factory asks `DatabaseBackendFactory.create_backend(...)` for the SQLite backend for that user DB path.
4. The shared backend registry returns the existing backend if the signature matches.
5. Request-scoped wrapper objects are created on top of that shared backend.

### Direct helper paths

1. A helper such as `CollectionsDatabase` or `CharactersRAGDB` resolves a SQLite path.
2. It calls `DatabaseBackendFactory.create_backend(...)`.
3. The shared registry returns the canonical backend for that normalized target.
4. The helper uses the shared backend and later releases the handle without closing the shared pool.

## Lifecycle And Failure Handling

### Concurrency

The SQLite registry must be protected by a lock so concurrent lookup-or-create calls for the same signature cannot create duplicate pools.

### Failed initialization

If backend creation fails:

- no partial registry entry should remain
- the exception should propagate unchanged to the caller
- future attempts for that same signature should be able to retry cleanly

### Release behavior

Release bookkeeping may track active holders for diagnostics, but zero active holders does not automatically close the backend. Actual closure is centralized and explicit.

### Reset behavior

Central reset paths such as:

- [reset_media_db_cache()](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py)
- [close_all_backends()](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/backends/factory.py)

must support two reset modes:

- graceful runtime reset for reconfiguration, where canonical shared SQLite backends are logically evicted from caches first and closed after a short grace period so in-flight users can finish. The intended caller class is runtime reconfiguration entry points such as [reset_content_backend()](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/DB_Manager.py).
- hard reset for tests or final shutdown, where canonical shared SQLite backends may be closed immediately and all caches cleared. The intended caller class is test/reset or shutdown entry points such as [reset_media_db_cache()](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py) and [close_all_backends()](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/backends/factory.py).

This mirrors the current deferred-close pattern already used by the shared content backend in [content_backend.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/content_backend.py).

The reset contract must also be explicit about cache ordering:

- named backend cache entries are cleared as references
- canonical shared SQLite backends are closed exactly once by the registry owner
- no reset path should leave the named cache pointing at an evicted shared backend

## Testing

### Factory-level unit tests

- same normalized SQLite target returns the same backend instance
- different SQLite targets return different backend instances
- different effective SQLite policies that materially change connection behavior produce distinct instances
- failed SQLite creation does not poison the registry
- `get_backend(name, sqlite_config)` and direct `create_backend(sqlite_config)` resolve to the same canonical shared SQLite backend
- centralized reset clears both the canonical registry and named cache references atomically

### Ownership and cleanup tests

- helper `close()` paths no longer close shared SQLite pools directly
- centralized reset closes shared pools exactly once
- release of a shared backend does not break another active consumer using the same backend
- wrapper-local cleanup behavior such as thread-local clearing and connection-level rollback or checkpoint still runs after ownership centralization

### Media DB regression tests

- repeated media DB resolution for the same user reuses the same underlying SQLite backend
- resetting the media DB cache clears higher-level factories without leaving the shared backend registry in an inconsistent state

### Cross-wrapper regression tests

- one direct helper path and one media path can both resolve the same underlying SQLite backend safely when pointed at the same normalized target
- two wrappers on the same thread that share a canonical SQLite backend do not corrupt each other's transaction or connection state, or the implementation explicitly limits reuse for wrappers that require connection isolation

## Rollout Plan

1. Add the shared SQLite registry and canonical signature logic in [factory.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/backends/factory.py) with focused unit tests.
2. Update the first-wave ownership paths to stop directly closing managed shared SQLite pools: media runtime session/factory cleanup, `CollectionsDatabase.close()`, and the `CharactersRAGDB` close path.
3. Verify the media DB request path still behaves correctly under cache hits, cache resets, and test isolation.
4. Sweep remaining direct SQLite factory callers in `DB_Management` for ownership compatibility. The exit criterion is not “every caller changed,” but “every caller reviewed, and only callers that directly close or invalidate a factory-managed SQLite pool changed.”

## Risks

- A too-broad signature could incorrectly merge backends that should stay distinct.
- A too-narrow signature could reduce reuse and preserve some churn.
- Missing one direct close path could allow one wrapper to shut down a shared pool unexpectedly.
- Over-simplifying wrapper teardown into a no-op could remove required local cleanup even when shared-pool ownership is correct.
- Same-thread wrappers that share one backend may also share one pooled SQLite connection, which can couple transaction state if compatibility is not validated explicitly.
- Holding a caller-owned mutable `DatabaseConfig` object inside a canonical shared backend could allow later config mutation to alter shared backend behavior unexpectedly.
- Resetting shared SQLite backends without a graceful eviction path could break in-flight requests during runtime reconfiguration.
- Test helpers that assume per-instance teardown may need small updates to use centralized reset.

## Open Questions

None for planning. The key decisions were confirmed during brainstorming:

- fix the structural churn, not just the log line
- cover both media and direct helper paths
- share SQLite backends by normalized target and config
- centralize cleanup
