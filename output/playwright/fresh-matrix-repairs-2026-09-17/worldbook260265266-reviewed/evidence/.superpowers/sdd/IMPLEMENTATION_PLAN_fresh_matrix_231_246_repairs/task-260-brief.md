# UAT260 World Book timestamp disambiguation brief

## Stage 1: Establish the read contract

**Goal:** Keep timestamp provenance local to World Book read results.

**Success criteria:** SQLite-generated naive timestamps are treated as UTC; PostgreSQL naive `timestamp` values are projected using the active PostgreSQL session timezone; already-aware values retain their instant.

**Tests:** Add focused backend red tests using production World Book service and response routes.

**Status:** Complete

## Stage 2: Apply the narrow response normalization

**Goal:** Normalize list, get, create/update readback, and character-attached World Book timestamps without changing permissions, transactions, cache ownership, schema, pool, or global parsing.

**Success criteria:** Every `WorldBookResponse` path emits explicit-offset timestamps under the established SQLite and stable PostgreSQL session policies.

**Tests:** SQLite plus official PostgreSQL non-UTC/DST session controls.

**Status:** Complete

## Stage 3: Preserve UI and static controls

**Goal:** Keep the existing World Book formatter behavior for explicit and invalid values while proving a response timestamp renders consistently across browser timezones.

**Success criteria:** Explicit offsets, numeric and Date inputs, and invalid/unknown output retain existing behavior.

**Tests:** Focused backend/frontend suites, official PostgreSQL runner, lint/compiler/Bandit scoped checks.

**Status:** Complete

## Limitation

Naive PostgreSQL rows do not contain their historical writer offset. This task assumes a stable PostgreSQL writing/reading session timezone policy. If that policy changed or varied, the original instant cannot be recovered from stored wall-clock fields; a separate migration/provenance decision is required.

## Attempt checkpoint

1. The focused SQLite route test failed as designed: its `WorldBookResponse` timestamps were naive.
2. The first green rerun reached an unrelated direct-call fixture issue: FastAPI's `Query` default is an object unless `expected_version=None` is passed explicitly. The test was corrected without changing product behavior.
3. SQLite then passed. The official PostgreSQL fixture run passed four cases and failed the combined endpoint-readback case. The runner retained the detailed redacted fixture traceback, and the fixture owner supplied a sanitized cause before the scope expanded.
4. The supplied traceback identified `update_world_book` using the PostgreSQL-incompatible wrapper context manager. TASK13260.207 authorized its narrow replacement with `db.transaction()`. The portable update rollback/conflict control and the PostgreSQL update path then passed.
5. The next official PostgreSQL run passed six cases and exposed a separate pre-existing wrapper context-manager use in `attach_to_character`. TASK13260.208 authorized its narrow `db.transaction()` replacement, transaction-preserving validation prechecks, and rollback/idempotency controls. The final official PostgreSQL suite passed 45 cases with no skips.

## Separate transaction findings

- **TASK13260.207 / UAT265:** `update_world_book` used `get_connection()` as a context manager, which PostgreSQL's `BackendConnectionWrapper` does not support. It now uses the established `db.transaction()` API, retaining optimistic-conflict and caller-rollback behavior.
- **TASK13260.208 / UAT266:** `attach_to_character` had the same PostgreSQL wrapper misuse. It now uses `db.transaction()` and verifies the existing character and World Book within that transaction before the existing backend-specific upsert. Repeated attaches remain idempotent; invalid references return the pre-existing false result; a caller rollback removes an uncommitted attachment.
- Inventory found further raw `get_connection()` contexts in other World Book operations. They were not executed by this work and have no recorded failure, so they remain unchanged.
