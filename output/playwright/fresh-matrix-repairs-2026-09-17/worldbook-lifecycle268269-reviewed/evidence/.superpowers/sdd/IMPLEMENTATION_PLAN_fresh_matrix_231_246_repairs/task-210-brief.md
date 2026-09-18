# TASK13260.210 World Book lifecycle qualification plan

**Goal:** Qualify the supported World Book book, entry, and character-association lifecycle on actual SQLite and the official PostgreSQL fixture, then repair only reproducible local portability faults in `world_book_manager.py`.

**Scope:** `tldw_Server_API/app/core/Character_Chat/world_book_manager.py` and a new maintained backend lifecycle test. Preserve the committed UAT260/265/266 acceptance bytes. Do not change shared database wrappers, SQL guards, schema configuration, native/runtime code, Git, or task tracking.

## Stage 1: Causal lifecycle test (red)

**Goal:** Add a parameterized real-backend test using `CharactersRAGDB` and the official `pg_database_config` fixture.

**Success criteria:** One lifecycle covers book create/get/list/update and cache invalidation; entry create/list/get/update/delete; character attach/update-idempotency/detach; soft-delete visibility and supported hard-delete behavior. A second control covers caller-owned transaction rollback and existing optimistic-version conflict. Existing route permission controls remain adjacent coverage.

**Tests:** Run the new file directly for SQLite and through `run-pg-tests-fixture-database.mjs` for both SQLite and PostgreSQL. Retain the first actual failure before implementation changes.

**Status:** Complete

## Stage 2: Narrow portability repair (green)

**Goal:** Replace only each exercised unsupported wrapper-context write path with the existing `db.transaction()` boundary.

**Success criteria:** No manual commit inside a caller-owned transaction; existing return values, backend placeholders/upsert behavior, cache invalidation, version conflict behavior, and validation failures remain unchanged.

**Tests:** Re-run the exact causal test after each local repair. Stop and reassess after three failed repair attempts for one fault.

**Status:** Complete

## Stage 3: Lifecycle and static verification

**Goal:** Confirm the full supported lifecycle on real SQLite and official PostgreSQL without skips, then run adjacent permissions and scoped static checks.

**Success criteria:** Actual lifecycle and adjacent timestamp/init/read/permission controls pass; source compilation and scoped Bandit introduce no production finding. Report known unsupported restore behavior explicitly rather than inventing it.

**Tests:** Official fixture runner with a unique evidence label, maintained permission test, compile, Bandit, and diff check.

**Status:** Complete

## Baseline and limitation

Baseline is commit `0d7f2a23c9`. The service has no supported `restore_world_book` operation; the lifecycle test will verify soft-delete invisibility and the existing permanent-delete option rather than adding restore semantics. World Book storage is per `CharactersRAGDB` instance, so cross-user ownership remains covered by route/dependency permission controls rather than a new global ownership layer.
