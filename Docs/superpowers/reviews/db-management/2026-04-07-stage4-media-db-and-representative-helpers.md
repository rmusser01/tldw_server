# Stage 4 Audit: Media DB Runtime and Representative Helpers

## Findings
1. High: `UserDatabase_v2` can finish initialization with missing core auth columns, missing `registration_codes.role_id`, or missing default RBAC seed data because its normalization and seed paths swallow failures instead of failing closed.
   - Issue class: Correctness
   - Evidence: Source-confirmed; the Stage 4 helper tests exercised `Users_DB` and backend-detection behavior, not `UserDatabase_v2` bootstrap failure paths.
   - Confidence: High
   - Why it matters: `_seed_default_data()` skips failed role, permission, and role-permission inserts with debug-only logging, while `_ensure_core_columns()` catches all exceptions around critical `ALTER TABLE` / extension / backfill work and continues. A partially migrated auth database can therefore appear initialized even when lockout columns, UUIDs, registration-code role linkage, or baseline permissions are missing.
   - File references: `tldw_Server_API/app/core/DB_Management/UserDatabase_v2.py:1097-1189`, `tldw_Server_API/app/core/DB_Management/UserDatabase_v2.py:1195-1267`
2. Medium: package-level `media_db.api` lookup helpers convert backend/query failures into normal `False`/`None`/`[]` answers, so caller-facing retrieval silently degrades instead of surfacing Media DB faults.
   - Issue class: Correctness
   - Evidence: Source-confirmed; caller-level runtime evidence shows MCP media retrieval depends on these helpers for chunk and section resolution, while Stage 4 tests only covered happy-path doubles and green-path runtime behavior.
   - Confidence: High
   - Why it matters: `has_unvectorized_chunks()`, the unvectorized chunk navigation helpers, and the document-structure lookup helpers all swallow broad exceptions. Callers such as the MCP media module then quietly fall back from prechunked retrieval and structural location hints to weaker approximate behavior, masking DB regressions and making retrieval quality depend on silent failure rather than explicit state.
   - File references: `tldw_Server_API/app/core/DB_Management/media_db/api.py:175-189`, `tldw_Server_API/app/core/DB_Management/media_db/api.py:547-705`, `tldw_Server_API/app/core/DB_Management/media_db/api.py:858-928`, `tldw_Server_API/app/core/MCP_unified/modules/implementations/media_module.py:769-777`, `tldw_Server_API/app/core/MCP_unified/modules/implementations/media_module.py:903-960`
3. Low: trusted-path enforcement is inconsistent across the representative helper modules, leaving some SQLite stores dependent on caller discipline rather than DB-layer validation.
   - Issue class: Isolation
   - Evidence: Source-confirmed; current caller-level code constrains some entry paths, which reduces present impact, but the helper modules themselves do not enforce a uniform trust boundary.
   - Confidence: Medium
   - Why it matters: `VoiceRegistryDB` resolves and confines its DB path beneath the configured user DB base, but `TopicMonitoringDB` only rejects bare relative filenames and `watchlist_alert_rules_db` opens whatever `db_path` it is given. That inconsistency means future or non-endpoint call sites can accidentally create or mutate SQLite files outside the intended storage roots even though the safer path helper already exists elsewhere in `DB_Management`.
   - File references: `tldw_Server_API/app/core/DB_Management/Voice_Registry_DB.py:72-84`, `tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py:103-115`, `tldw_Server_API/app/core/DB_Management/watchlist_alert_rules_db.py:44-47`, `tldw_Server_API/app/core/DB_Management/watchlist_alert_rules_db.py:55-67`, `tldw_Server_API/app/core/DB_Management/watchlist_alert_rules_db.py:90-187`

## Scope
Review the `media_db` package-native runtime and the representative helper modules selected in the plan, with emphasis on request isolation, backend-resolution drift, bootstrap or migration behavior, and helper modules that still own custom schema or filesystem logic.

Limited transitive scope expansion:
- `tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py`
- `tldw_Server_API/app/core/MCP_unified/modules/implementations/media_module.py`
- `tldw_Server_API/app/core/Watchlists/alert_rules.py`
- `tldw_Server_API/app/api/v1/endpoints/watchlist_alert_rules.py`
- These were consulted only to confirm how `media_db` sessions are created and released, how chunk/section lookup helpers affect caller-visible retrieval behavior, and how representative helper modules currently receive their DB paths.

## Code Paths Reviewed
### `media_db` runtime flow
- `media_db.api.create_media_database()` delegates to `runtime.factory.create_media_database()` and returns the package-native `MediaDatabase`.
- `runtime.factory.create_media_database()` and `validate_postgres_content_backend()` mediate default config loading, backend loader use, content-backend validation, and schema/policy version checks.
- `runtime.bootstrap_lifecycle_ops.initialize_media_database()` sets path state, resolves the backend, initializes context-local connection state, and invokes `_initialize_schema()`.
- `runtime.backend_resolution._resolve_backend()` chooses SQLite vs PostgreSQL and can silently coerce PostgreSQL content mode back to SQLite in test mode when the DB path is not `:memory:`.
- `runtime.connection_lifecycle` owns persistent-connection reuse, request-context release, and Postgres scope reapplication on reused connections.
- `runtime.execution_ops` owns direct SQL execution, ephemeral SQLite connection behavior, backend error translation, and commit semantics.
- `runtime.scope_resolution_ops._resolve_scope_ids()` is the only package-owned org/team scope resolver for the native class.

### `media_db` caller-facing helpers
- `media_db.api` exposes lookup, chunk-navigation, chunk-template, and document-section helper functions that are reused by runtime wrappers and caller modules.
- `runtime.query_ops` is a thin delegation layer around those package-level helpers.
- `DB_Deps._get_or_create_media_db_factory()` caches one `MediaDbFactory` per user and chooses a shared backend strategy for request-scoped sessions.
- `DB_Deps._resolve_media_db_for_user()` forwards effective org/team scope to `MediaDbFactory.for_request()`.
- `MCP media_module` uses `has_unvectorized_chunks()`, `get_unvectorized_anchor_index_for_offset()`, `get_unvectorized_chunk_index_by_uuid()`, and `get_unvectorized_chunks_in_range()` to drive location hints and prechunked retrieval.

### Representative helper modules
- `UserDatabase_v2.py` still owns its own schema bootstrap, core-column normalization, and default RBAC seeding on top of the shared backend abstraction.
- `TopicMonitoring_DB.py` is a direct SQLite wrapper with manual schema evolution and hand-managed transactions.
- `Voice_Registry_DB.py` is a direct SQLite wrapper but explicitly validates containment beneath the configured user DB base directory.
- `Workflows_Scheduler_DB.py` uses the shared backend factory, own schema DDL, and a dedicated per-user workflows path.
- `watchlist_alert_rules_db.py` is a direct SQLite helper with table bootstrap and CRUD by raw filesystem path.

## Tests Reviewed
- `tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py`: verifies package exports and lightweight reader/writer compatibility. Coverage is broad for import surfaces and happy-path delegation, but it does not assert error propagation for the chunk-navigation or section-lookup helpers.
- `tldw_Server_API/tests/DB_Management/test_media_db_bootstrap_lifecycle_ops.py`: validates constructor rebinding, memory-DB bootstrap behavior, and constructor failure cleanup.
- `tldw_Server_API/tests/DB_Management/test_media_db_connection_cleanup.py`: validates Postgres connection scope reapplication on reused connections plus SQLite ephemeral-connection cleanup. It does not exercise package-level helper failure paths.
- `tldw_Server_API/tests/DB_Management/test_media_db_runtime_factory.py`: covers runtime-factory defaults and Postgres validator behavior at a mostly stubbed level.
- `tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py`: covers schema bootstrap behavior and a large slice of native-class/runtime wiring. Coverage is strong for green-path bootstrap.
- `tldw_Server_API/tests/DB_Management/test_media_db_scope_resolution_ops.py`: covers default scope fallback and `get_scope()` override semantics, but only at the helper level.
- `tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py`: covers request-scoped factory/session behavior and cache reuse for `DB_Deps`.
- `tldw_Server_API/tests/DB_Management/test_media_db_core_repositories.py`: covers repository-backed CRUD/read flows. Coverage is mostly functional and does not stress DB-failure propagation.
- `tldw_Server_API/tests/DB_Management/test_media_db_v2_regressions.py`: regression coverage for selected legacy/native compatibility edges.
- `tldw_Server_API/tests/DB_Management/test_media_db_migration_missing_scripts_error.py`: covers missing migration-script failure behavior.
- `tldw_Server_API/tests/DB_Management/test_media_db_postgres_rls_ops.py`: covers policy helper binding and selected PostgreSQL RLS SQL generation paths.
- `tldw_Server_API/tests/DB_Management/test_media_postgres_migrations.py`: provides real PostgreSQL integration coverage for media schema migrations, but those integration tests were skipped in this environment.
- `tldw_Server_API/tests/DB_Management/test_media_postgres_support.py`: covers a mix of stubbed PostgreSQL runtime behavior plus one PostgreSQL integration test; the integration slice was skipped here.
- `tldw_Server_API/tests/DB_Management/test_users_db_sqlite.py` and `tldw_Server_API/tests/DB_Management/unit/test_users_db_update_backend_detection.py`: cover the async `Users_DB` path, not `UserDatabase_v2`.
- `tldw_Server_API/tests/DB_Management/test_workflows_scheduler_db_paths.py`: confirms the scheduler DB uses its dedicated per-user SQLite path rather than inheriting global DB URLs.
- `tldw_Server_API/tests/AuthNZ/integration/test_authnz_backends_pytest.py`: covers AuthNZ backend smoke behavior; the PostgreSQL-marked slice was skipped when dependencies were unavailable.
- `tldw_Server_API/tests/test_watchlist_alert_rules.py`: covers alert-rule CRUD and endpoint behavior on a valid caller-supplied path.
- `tldw_Server_API/tests/test_watchlist_alert_rules_paths.py`: confirms the watchlist-alert endpoint helper rejects path-like user IDs before building the per-user DB path.
- `tldw_Server_API/tests/Claims/test_claims_service_backend_selection.py`, `tldw_Server_API/tests/Claims/test_ingestion_claims_sql.py`, `tldw_Server_API/tests/Media/test_media_reprocess_endpoint.py`, and `tldw_Server_API/tests/test_utils.py`: caller-level checks that constrain selected backend-choice and media entrypoint behavior around the reviewed modules.

## Validation Commands
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py tldw_Server_API/tests/DB_Management/test_media_db_bootstrap_lifecycle_ops.py tldw_Server_API/tests/DB_Management/test_media_db_connection_cleanup.py tldw_Server_API/tests/DB_Management/test_media_db_runtime_factory.py tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py tldw_Server_API/tests/DB_Management/test_media_db_scope_resolution_ops.py tldw_Server_API/tests/DB_Management/test_media_db_request_scope_isolation.py tldw_Server_API/tests/DB_Management/test_media_db_core_repositories.py tldw_Server_API/tests/DB_Management/test_media_db_v2_regressions.py tldw_Server_API/tests/DB_Management/test_media_db_migration_missing_scripts_error.py tldw_Server_API/tests/DB_Management/test_media_db_postgres_rls_ops.py tldw_Server_API/tests/DB_Management/test_media_postgres_migrations.py tldw_Server_API/tests/DB_Management/test_media_postgres_support.py -q`
- Result on 2026-04-07: `387 passed, 9 skipped` in `40.01s`
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_users_db_sqlite.py tldw_Server_API/tests/DB_Management/unit/test_users_db_update_backend_detection.py tldw_Server_API/tests/DB_Management/test_workflows_scheduler_db_paths.py tldw_Server_API/tests/AuthNZ/integration/test_authnz_backends_pytest.py tldw_Server_API/tests/test_watchlist_alert_rules.py tldw_Server_API/tests/Claims/test_claims_service_backend_selection.py tldw_Server_API/tests/Claims/test_ingestion_claims_sql.py tldw_Server_API/tests/Media/test_media_reprocess_endpoint.py tldw_Server_API/tests/test_utils.py -q`
- Result on 2026-04-07: `20 passed, 1 skipped` in `45.19s`
- Confidence boundary: core SQLite/native-runtime behavior is well runtime-validated in Stage 4, but several PostgreSQL migration/bootstrap checks remained skipped integration coverage in this environment, so some backend-parity conclusions remain partly source-traced.

## Coverage Gaps
- No Stage 4 test asserts that `media_db.api` chunk-navigation or section-lookup helpers surface backend failures instead of silently returning `False`/`None`/`[]`.
- No Stage 4 test exercises `UserDatabase_v2` startup under failing `ALTER TABLE`, failing `pgcrypto` extension creation, or failed role/permission seed writes.
- `runtime.backend_resolution._resolve_backend()` has no targeted test that documents or challenges the silent PostgreSQL-to-SQLite downgrade in test mode for non-`:memory:` DB paths.
- The representative-helper tests validate valid path usage, but no targeted test asserts that `TopicMonitoringDB` or `watchlist_alert_rules_db` enforce the same trusted-root containment that `VoiceRegistryDB` already applies.
- The skipped PostgreSQL integration tests mean Stage 4 can confirm a lot of migration SQL and RLS logic by source and stubbed execution, but not every end-to-end PostgreSQL bootstrap invariant here.

## Improvements
- Change the package-level `media_db.api` chunk and section helpers to raise a typed DB/read exception on backend failure, and let callers explicitly choose whether to fall back.
- Add regression tests for `has_unvectorized_chunks()`, `get_unvectorized_anchor_index_for_offset()`, `get_unvectorized_chunks_in_range()`, `lookup_section_for_offset()`, and `lookup_section_by_heading()` that assert DB failures are not silently normalized to benign results.
- Make `UserDatabase_v2` schema normalization and RBAC seeding transactional and fatal for required columns/roles/permissions, or retire the helper in favor of the better-covered AuthNZ `Users_DB` path.
- Unify direct SQLite helper path handling around `db_path_utils` trust-boundary helpers so representative modules do not each invent a different filesystem-safety rule.
- Add an explicit test that locks in whether test mode is allowed to coerce a requested PostgreSQL media backend back to SQLite, because that choice materially affects backend-parity confidence.

## Exit Note
- Stage 4 runtime/helper review is complete. Stage 5 should prioritize cross-stage synthesis around fail-open bootstrap and helper behavior, the weakly tested PostgreSQL parity edges, and the modules that still bypass the shared DB path/trust abstractions.
