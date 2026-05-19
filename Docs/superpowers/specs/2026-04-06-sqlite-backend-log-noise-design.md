# SQLite Backend Log Noise Design

Date: 2026-04-06
Status: Approved for planning
Owner: Codex brainstorming session

## Summary

Reduce the repeated `Creating sqlite backend` log noise without changing SQLite backend lifetime semantics. The current low-level backend factory emits a generic `INFO` log every time a SQLite backend object is constructed, even though many callers intentionally create fresh backends and some higher-level paths already provide their own lifecycle context. The design keeps creation behavior unchanged, demotes generic factory success logging to a diagnostic concern, and shifts `INFO`-level observability to the owner objects that know what subsystem is being initialized and why.

## Problem

The current log stream is noisy because the common backend constructor emits an unconditional success message:

- [factory.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/backends/factory.py) logs `Creating sqlite backend` at `INFO` inside `DatabaseBackendFactory.create_backend(...)`.
- Many SQLite-using wrappers and helpers across `DB_Management` call `create_backend(...)` directly rather than going through a shared global cache.
- The resulting `INFO` stream answers only that a backend object was allocated, not which higher-level database object was initialized, for which user or path, or whether the event is operationally meaningful.

This creates two concrete problems:

- Operators see repeated low-value `INFO` logs during normal request and service activity.
- Meaningful higher-level initialization events are harder to spot because the common constructor floods the log stream first.

## Goals

- Eliminate the repeated generic `Creating sqlite backend` `INFO` noise.
- Preserve current SQLite backend creation semantics.
- Keep useful operational visibility for higher-level database object initialization.
- Make DB initialization logs answer what is being initialized, for whom, and with which effective target.
- Apply the logging contract across SQLite-using `DB_Management` owners, including media/content wrappers, auth/user DB wrappers, collections, watchlists, and scheduler-adjacent wrappers in this subsystem.

## Non-Goals

- Introduce process-wide shared SQLite backend instances.
- Change request, owner, or process lifetime semantics for existing SQLite backends.
- Hide real initialization failures, path issues, or cleanup errors.
- Refactor unrelated database behavior or add new backend caching mechanisms.
- Redesign PostgreSQL backend ownership or remove PostgreSQL-specific observability without an explicit follow-up decision.

## Requirements Confirmed With User

- Treat this as an observability problem, not a backend reuse redesign.
- Include all SQLite backend creation in the `DB_Management` subsystem, not only the media request path.
- Do not introduce process-wide shared SQLite backend instances keyed by path or config.
- Do not standardize on a new owner-scoped reuse invariant in this change.
- Keep current creation patterns intact and fix the log noise by improving the logging surface.
- Keep the scope centered on SQLite success logging; do not create accidental PostgreSQL observability regressions while changing the shared factory.
- Do not replace factory noise with a new layer of per-request owner `INFO` noise.

## Current State

### Factory logging is unconditional and too low-level

[factory.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/backends/factory.py) currently emits `logger.info(f"Creating {backend_type.value} backend")` inside `DatabaseBackendFactory.create_backend(...)` before returning a new backend instance.

That message is generic by design. It does not include enough context to be useful as a high-signal `INFO` event, but it sits on the shared constructor path used by many callers.

### Reuse already varies by caller

The subsystem does not use one consistent reuse model today:

- The media request path already has a higher-level cache in [DB_Deps.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py), where `_media_db_factories` caches a `MediaDbFactory` per user.
- That cached factory reuses one SQLite backend through [session.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/media_db/runtime/session.py) when `MediaDbFactory.for_sqlite_path(...)` is used.
- Other wrappers such as user/auth DB, watchlists, collections, workflow-related DB objects, migration helpers, and fallback resolution helpers still call `DatabaseBackendFactory.create_backend(...)` directly and allocate fresh backend instances as part of normal object construction.

The repeated log noise therefore reflects existing creation behavior, not necessarily a bug.

### Some higher-level logs already exist

Several owners already emit better lifecycle logs than the factory does. For example:

- [DB_Deps.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py) logs cache hits, cache misses, and `MediaDbFactory` initialization outcomes.
- [UserDatabase_v2.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/UserDatabase_v2.py) logs when `UserDatabase` initialization completes.

Those logs are more useful because they describe the subsystem object, not the internal backend allocation step.

## Approaches Considered

### Approach 1: Demote the factory `INFO` log to `DEBUG` and stop there

Change the factory success message from `INFO` to `DEBUG` and leave all owners unchanged.

Pros:

- Smallest code change.
- Immediately removes the repeated `INFO` noise.
- Preserves all runtime behavior.

Cons:

- Leaves the subsystem without a clear logging contract.
- Some owners may still lack meaningful `INFO` initialization logs.
- Future contributors may reintroduce generic low-value `INFO` logs elsewhere.

### Approach 2: Move `INFO` ownership to higher-level DB owners and keep the factory diagnostic-only

Demote or remove the generic factory success log, then standardize `INFO`-level initialization logs on the owner objects that know the DB role, scope, and effective target.

Pros:

- Removes the noisy low-value message.
- Retains meaningful operational visibility.
- Matches the existing code shape without changing backend lifetimes.
- Gives the subsystem a clear logging contract.

Cons:

- Slightly larger than a one-line log-level change.
- Requires touching multiple owner call sites for consistency.

### Approach 3: Keep factory `INFO`, but deduplicate or rate-limit it

Retain the shared low-level log and add deduplication, throttling, or sampling.

Pros:

- Preserves a shared `INFO` breadcrumb at the constructor layer.
- Reduces some noise without moving log ownership.

Cons:

- Adds stateful logging machinery for a message that is still low-value.
- Makes reasoning about missing or sampled logs harder.
- Solves presentation rather than the underlying observability boundary.

## Recommendation

Use Approach 2.

The factory should not be an `INFO`-level lifecycle surface. It is a low-level constructor and should log successful backend creation only at `DEBUG` when deep troubleshooting is needed. `INFO` logging should belong to owner objects such as `MediaDbFactory`, `UserDatabase`, watchlist/collection wrappers, and similar components that can explain what database object is being initialized and why that event matters.

Approach 2 can include the small demotion from Approach 1, but the important design choice is not merely “make the line quieter.” It is “move success logging to the correct abstraction layer.”

## Proposed Design

### 1. Define a logging ownership contract

Adopt the following subsystem rule:

- Low-level backend construction is a debug concern.
- Higher-level database object lifecycle is an info concern.

Under this rule:

- `DatabaseBackendFactory.create_backend(...)` may emit `DEBUG` details for successful construction.
- Higher-level owners decide whether a given initialization is meaningful enough for `INFO`.
- Repeated low-level backend allocations no longer flood `INFO`.

### 2. Make the backend factory diagnostic-only for successful creation

Update the common backend factory so that successful SQLite backend creation no longer emits a generic `INFO` line.

Expected factory behavior:

- Keep unsupported backend type errors and import/configuration failures at their current error severity.
- Log successful SQLite backend creation only at `DEBUG`, with safe context when available.
- Avoid generic success messages that do not identify the owning subsystem object.
- Leave non-SQLite success logging unchanged in this change unless the owner path already provides equivalent or better `INFO` visibility and the adjustment is explicitly justified.

Example acceptable `DEBUG` context:

- backend type
- normalized SQLite path when file-backed
- whether the target is `:memory:` or a SQLite URI

This is diagnostic detail, not an operator-facing lifecycle event.

### 3. Move meaningful `INFO` logs to owner initialization paths

Public owner initialization boundaries that create or resolve SQLite backends should emit or retain one meaningful `INFO` lifecycle log where initialization is operationally relevant.

In practice, that means `INFO` should be reserved for:

- cache misses that create shared or reused owner objects
- service startup or worker startup initialization
- long-lived process-owned DB wrappers
- explicit administrative or configuration transitions

Short-lived or frequently repeated wrapper construction should not gain a new success `INFO` log merely because it is an owner boundary. For those paths, either keep success logging at `DEBUG` or rely on existing surrounding context and failure logs.

Representative targets include:

- media/content DB factory creation and cache miss paths
- auth/user database initialization
- collections DB initialization
- watchlists DB initialization
- workflow and scheduler-adjacent SQLite DB wrappers in `DB_Management`
- any public constructor or factory boundary that currently relies on the factory log for visibility

Each owner-level `INFO` log should identify:

- the subsystem object being initialized
- the backend type
- concise target context at `INFO`, with the effective SQLite path only when it is already accepted operational context and materially helps distinguish the instance
- user or scope context when already part of existing operational logging

This replaces “a backend was created” with “this database-facing component initialized for this target.”

Internal helper methods such as backend resolvers should not gain new `INFO` logs unless they are the only meaningful public lifecycle boundary. The goal is to move logging to the right abstraction layer, not to relocate the same noise into `_resolve_backend(...)` style helpers.

Likewise, constructors or helpers that are called from hot request dependencies, such as per-request `for_user(...)` accessors, should default to no new success `INFO` log unless the event is rate-limited by an existing cache miss or otherwise represents an infrequent transition.

### 4. Standardize success log phrasing

Use a small, grep-friendly phrase set for owner lifecycle logs:

- `Initializing ...`
- `Initialized ...`
- `Closing ...`

This avoids every module inventing its own wording and makes DB lifecycle logs easier to scan.

The design does not require every owner to emit both start and completion logs. One concise `INFO` line per meaningful initialization path is enough unless the component already has a useful two-phase pattern.

Do not add a new owner-level `INFO` log if the component already emits an equally useful contextual initialization log. Prefer refining the existing log over adding a second line and recreating duplicate noise.

Severity guidance:

- `INFO`: cache miss, process startup, worker startup, long-lived shared object initialization, explicit reconfiguration
- `DEBUG`: repeated constructor calls, per-request wrappers, internal backend resolution detail

### 5. Preserve existing lifetime semantics

This design must not quietly introduce new caching or sharing behavior.

Specifically:

- No process-wide shared SQLite backend registry keyed by path or config.
- No change to the current media factory cache behavior beyond log wording.
- No new reuse invariant forced onto wrappers that currently create fresh backends.

The only behavioral change is the observability surface.

## Error Handling

The change must not suppress or demote real failures.

Keep warning/error logging for:

- unsupported backend types
- invalid configuration
- path resolution or directory creation failures
- connection and pool setup failures
- cleanup and close failures

Only successful generic factory creation moves out of `INFO`.

## Testing

### Unit tests

- Assert that successful SQLite creation through `DatabaseBackendFactory.create_backend(...)` no longer produces the generic `Creating sqlite backend` `INFO` log.
- Assert that the factory still raises and logs appropriately for unsupported backend types or real initialization failures.

### Targeted owner tests

- Add or update targeted tests for representative owners so they emit one meaningful `INFO` initialization log with subsystem context.
- Cover at least one content/media path and one non-media wrapper such as `UserDatabase`.
- Cover at least one owner that already has partial lifecycle logging so the implementation does not accidentally add duplicate `INFO` lines on a single successful initialization path.

### Regression coverage

- Add a focused regression test around repeated media DB access so the higher-level cached path still works and the noisy factory pattern does not reappear in the `INFO` stream.
- Optionally add a grep-style assertion that the exact generic `Creating sqlite backend` `INFO` line is absent from normal successful SQLite creation logs.
- Add at least one hot-path regression around repeated `for_user(...)` construction or request dependency resolution so the implementation does not shift the spam from the factory into collections/watchlists owner logs.

## Rollout Notes

- Start with the shared backend factory and the most visible owner paths.
- Sweep remaining direct SQLite owner constructors in `DB_Management` for consistency.
- Keep the rollout narrow: this is a logging contract cleanup, not a DB lifecycle redesign.
- Audit existing owner logs before adding new ones so modules like workflow/scheduler wrappers do not end up with two contextual `INFO` lines for the same initialization.
- Treat high-frequency constructors such as collections and watchlists `for_user(...)` accessors as likely `DEBUG`-or-silent paths unless an existing cache miss or service startup boundary makes the event infrequent.

## Open Questions

None. The design constraints were confirmed during brainstorming:

- no process-wide shared SQLite backends
- no reuse redesign
- full `DB_Management` subsystem scope
- owner-level `INFO`, factory-level `DEBUG`
