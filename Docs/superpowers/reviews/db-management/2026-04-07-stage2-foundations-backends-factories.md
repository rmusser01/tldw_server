# Stage 2 Foundations Backends Factories

## Findings
1. Medium: shared PostgreSQL content backend resets replace cached backend objects without closing the superseded pools, so runtime reconfiguration can leak live connections until process exit.
   - Backend classification: PostgreSQL-specific
   - Confidence: High
   - Why it matters: `get_content_backend()` swaps `_cached_backend` in place and the reset path clears cache globals by assignment, but neither path closes the previous pool. `DB_Manager.shutdown_content_backend()` only closes the currently reachable backend, so reload-heavy tests or long-lived processes can strand connections to the old DSN/configuration.
   - File references: `tldw_Server_API/app/core/DB_Management/content_backend.py:153-205`, `tldw_Server_API/app/core/DB_Management/media_db/runtime/defaults.py:42-109`, `tldw_Server_API/app/core/DB_Management/DB_Manager.py:222-241`
2. Medium: the shared `DatabaseBackend` FTS surface is not actually syntax-parity across SQLite and PostgreSQL, even though an FTS translator exists specifically for that purpose.
   - Backend classification: Cross-backend parity concern
   - Confidence: High
   - Why it matters: `DatabaseBackend` exposes a single shared `fts_search(fts_query, ...)` contract, so callers using the abstraction have a reasonable expectation that one `FTSQuery.query_text` surface will behave comparably across backend implementations. SQLite `fts_search()` sends raw FTS5 `MATCH` text through unchanged, while PostgreSQL `fts_search()` passes raw `fts_query.query_text` directly to `to_tsquery('english', ...)`. Queries that are valid in SQLite FTS5 but require translation for PostgreSQL can therefore succeed on SQLite and fail or change meaning on PostgreSQL when callers use the backend abstraction directly.
   - File references: `tldw_Server_API/app/core/DB_Management/backends/sqlite_backend.py:543-590`, `tldw_Server_API/app/core/DB_Management/backends/postgresql_backend.py:1075-1120`, `tldw_Server_API/app/core/DB_Management/backends/fts_translator.py:1-216`
3. Low: the named singleton backend factory can silently return a stale backend for a different configuration because cache identity is only the caller-provided name.
   - Backend classification: Backend-agnostic factory risk
   - Confidence: High
   - Why it matters: `get_backend(name, config=...)` ignores the new config once a name is cached, so later calls can reuse the wrong database connection settings and bypass any expected recreation or validation step. I did not find this path on the main Media DB runtime path, but it remains part of the public factory surface.
   - File references: `tldw_Server_API/app/core/DB_Management/backends/factory.py:35-36`, `tldw_Server_API/app/core/DB_Management/backends/factory.py:219-254`

## Scope
Review the shared DB_Management entry points, backend abstraction, backend factories, and Media DB routing used by the compatibility surface in `DB_Manager.py` and `media_db/api.py`.

Limited transitive scope expansion:
- `tldw_Server_API/app/core/DB_Management/media_db/runtime/defaults.py`
- `tldw_Server_API/app/core/DB_Management/media_db/runtime/factory.py`
- These runtime modules were consulted deliberately because `DB_Manager.py` and `media_db/api.py` delegate into them for backend loading and Media DB construction. They were used only to trace those delegated paths, not to broaden Stage 2 into a general runtime-module review.

## Code Paths Reviewed
### Construction and routing map
- `content_backend.load_content_db_settings()` resolves backend type, DSN/host settings, SQLite path, and backup path.
- `content_backend.get_content_backend()` only constructs and caches a shared backend for PostgreSQL; SQLite intentionally returns `None` so callers use file-backed per-user paths instead.
- `media_db.runtime.defaults.ensure_content_backend_loaded()` lazily loads that shared backend, while `single_user_db_path` remains the fallback default path for SQLite mode.
- `DB_Manager.get_content_backend_instance()` and `DB_Manager.create_media_database()` are compatibility wrappers over the Media DB runtime defaults and runtime factory.
- `media_db.api.create_media_database()` routes through the same runtime factory, and `media_db.api.get_media_repository()` decides whether to keep a writer double as-is or wrap a DB-like object in `MediaRepository`.
- `media_db.runtime.factory.create_media_database()` is the final routing gate: it always computes a target path, but in PostgreSQL mode it refuses to construct a media DB unless the resolved backend is non-null and `BackendType.POSTGRESQL`.

### Query preparation and execution map
- `prepare_backend_statement()` and `prepare_backend_many_statement()` are the shared SQLite-to-PostgreSQL rewrite layer: placeholder conversion, `INSERT OR IGNORE` rewrite, `COLLATE NOCASE` removal, boolean rewrite, `randomblob()` rewrite, and optional `RETURNING id`.
- `SQLiteBackend.execute()` and `SQLiteBackend.execute_many()` run raw SQLite SQL; `SQLiteBackend.transaction()` explicitly issues `BEGIN IMMEDIATE` and only commits or rolls back when it started the outermost transaction.
- `PostgreSQLBackend.execute()` and `PostgreSQLBackend.execute_many()` always normalize SQL first, classify writes, auto-commit writes when outside a managed transaction, and explicitly roll back read-only implicit transactions to avoid idle-in-transaction sessions.
- `PostgreSQLBackend.transaction()` tracks nested transaction ownership via `_managed_tx_depths` and commits or rolls back only at the outermost depth.
- `AsyncDatabaseWrapper` moves synchronous `CharactersRAGDB` calls onto a shared thread pool; `transaction_utils.run_transaction()` is the safe single-threaded transactional path, while `db_transaction()` is the async wrapper over the synchronous transaction manager.

### Backend parity notes recorded during trace
- Placeholder handling differs by backend exactly as intended: SQLite keeps `?`, PostgreSQL converts to `%s`.
- Transaction boundaries differ materially: SQLite uses explicit `BEGIN IMMEDIATE`; PostgreSQL relies on implicit transactions plus outermost-depth commit/rollback rules.
- Pool ownership differs materially: SQLite keeps thread-local connections until pool close/pruning; PostgreSQL re-borrows from either `psycopg_pool` or a fallback mini-pool and closes overflow connections on return.
- FTS behavior diverges by backend: SQLite exposes native FTS5 virtual tables and `MATCH`; PostgreSQL exposes `tsvector` columns, GIN indexes, and raw `to_tsquery`.

## Tests Reviewed
- `tldw_Server_API/tests/DB_Management/test_backend_utils.py`: asserts placeholder conversion, SQLite-to-PostgreSQL SQL rewrites, runtime helper delegation, backend-specific case-insensitive search helpers, and boolean rewrite behavior. Coverage: both backends via shared utility layer. This upgrades query-rewrite claims from probable to confirmed.
- `tldw_Server_API/tests/DB_Management/test_content_backend_cache.py`: asserts the shared content backend cache key changes when PostgreSQL password or `sslmode` changes. Coverage: PostgreSQL cache path only. This confirms only part of the cache signature, not cleanup behavior or pool-tuning invalidation.
- `tldw_Server_API/tests/DB_Management/test_database_backends.py`: asserts SQLite backend creation/features/PRAGMA wiring/schema creation/transaction semantics/rank-expression sanitization, plus selected PostgreSQL behavior such as mixed-case `table_exists()`, failed-statement rollback before reuse, CTE write detection, and FTS source-table mapping. Coverage: both. The live-PostgreSQL integration assertions were inspected in source but not exercised in this validation run unless a live Postgres fixture was available.
- `tldw_Server_API/tests/DB_Management/test_db_manager_config_behavior.py`: asserts backup path precedence and `db_type` derivation across SQLite/PostgreSQL/unsupported values. Coverage: routing/config only. This confirms manager-level config discrimination, not backend cleanup.
- `tldw_Server_API/tests/DB_Management/test_db_manager_wrappers.py`: asserts wrapper functions require explicit DB instances, route through the Media DB API/runtime factory, accept request-scoped sessions, and preserve deprecated compatibility surfaces without going back through `Media_DB_v2`. Coverage: mostly SQLite-backed doubles plus forced Postgres mode branches. This confirms wrapper routing claims.
- `tldw_Server_API/tests/DB_Management/test_transaction_utils.py`: asserts async retry behavior, rollback rules, decorator behavior, and conversation/message transactional helpers. Coverage: backend-agnostic `CharactersRAGDB` transaction helpers, not the backend implementations themselves.
- `tldw_Server_API/tests/DB_Management/unit/test_postgres_placeholder_prepare.py`: asserts placeholder conversion ignores quoted literals/identifiers and that PostgreSQL query preparation preserves literals. Coverage: PostgreSQL utility path. This confirms the literal-safe placeholder rewrite.
- `tldw_Server_API/tests/DB_Management/unit/test_postgres_pool_fallback.py`: asserts fallback PostgreSQL pool closes overflow connections, closes free connections on shutdown, and deduplicates close calls. Coverage: PostgreSQL fallback pool only. This confirms the intended overflow cleanup path.
- `tldw_Server_API/tests/DB_Management/unit/test_postgres_transaction_manager.py`: asserts PostgreSQL outermost commit/rollback semantics, nested transaction depth behavior, statusless cursor handling, and driver exception wrapping. Coverage: PostgreSQL backend only. This confirms transaction-depth handling in isolation.
- `tldw_Server_API/tests/DB_Management/unit/test_sqlite_pool_pruning.py`: asserts stale thread-local SQLite connections are pruned once worker threads die. Coverage: SQLite pool only. This confirms the dead-thread cleanup rule for SQLite.

## Validation Commands
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_backend_utils.py tldw_Server_API/tests/DB_Management/test_content_backend_cache.py tldw_Server_API/tests/DB_Management/test_database_backends.py tldw_Server_API/tests/DB_Management/test_db_manager_config_behavior.py tldw_Server_API/tests/DB_Management/test_db_manager_wrappers.py tldw_Server_API/tests/DB_Management/test_transaction_utils.py tldw_Server_API/tests/DB_Management/unit/test_postgres_placeholder_prepare.py tldw_Server_API/tests/DB_Management/unit/test_postgres_pool_fallback.py tldw_Server_API/tests/DB_Management/unit/test_postgres_transaction_manager.py tldw_Server_API/tests/DB_Management/unit/test_sqlite_pool_pruning.py -v`
- Result on 2026-04-07: `101 passed, 5 skipped` in `33.11s`
- Skipped items were the live-PostgreSQL integration cases from `test_database_backends.py`, so claims that depend on full Postgres execution remain source-backed unless covered by the PostgreSQL unit tests above.

## Coverage Gaps
- PostgreSQL-specific: no targeted test exercises cache replacement or reset cleanup for the shared content backend, so the connection-leak finding is source-confirmed rather than test-confirmed.
- Cross-backend parity concern: no targeted test exercises backend-level FTS query translation for PostgreSQL with SQLite-style FTS operators; the current evidence is the code path mismatch plus the existence of the unused translator.
- Backend-agnostic factory risk: no targeted test covers `backends.factory.get_backend()` with conflicting configs under the same cache name.

## Improvements
- Close and dispose the previous shared content backend pool before overwriting cache globals in `content_backend` or `media_db.runtime.defaults`, then add a regression test that verifies old pools are closed on reset/reconfiguration.
- Either normalize backend-level `FTSQuery.query_text` inside `PostgreSQLBackend.fts_search()` or explicitly narrow the abstraction contract so backend callers know they must pre-normalize the query.
- Harden `backends.factory.get_backend()` by keying cache entries on config signature or raising when a different config is supplied for an existing name.

## Exit Note
- Stage 2 shared-foundation review is complete. The main Stage 2 risks identified are stale pooled-backend lifecycle, backend-level FTS parity drift, and stale-config singleton reuse outside the primary Media DB runtime path.
