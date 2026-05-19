# Stage 3 Audit: Paths, Tenancy, Migrations, and Backups

## Findings
1. High: PostgreSQL RLS installers can partially fail and still return success, so tenant isolation policy setup can silently diverge from the code's assumptions.
   - Issue class: Isolation
   - Evidence: Source-confirmed; not directly exercised by the targeted Stage 3 tests because PostgreSQL fixture setup failed in this environment.
   - Confidence: High
   - Why it matters: `ensure_prompt_studio_rls()` and `ensure_chacha_rls()` execute each `ALTER TABLE`/`CREATE POLICY` statement independently, swallow every exception as debug-only noise, and return `True` once any statement succeeded. A deployment can therefore end up with only some tables protected, or with `FORCE ROW LEVEL SECURITY` missing on some tables, while higher layers continue as if PostgreSQL tenancy is fully enforced.
   - File references: `tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py:24-215`, `tldw_Server_API/app/core/DB_Management/backends/pg_rls_policies.py:218-263`
2. High: the migration loader skips malformed or unreadable migration files instead of failing closed, and the upgrade path does not enforce contiguous version advancement.
   - Issue class: Correctness
   - Evidence: Source-confirmed; not reproduced by the targeted tests in this stage because the migration loader tests only covered the successful and duplicate-version cases.
   - Confidence: High
   - Why it matters: `load_migrations()` logs and continues when a `.json` or `.sql` migration cannot be parsed. `_migrate_to_version_locked()` then applies whatever later versions remain without checking that every intermediate version exists. A broken v2 file plus a valid v3 file can therefore advance the database to v3 without ever running v2, and the current targeted tests only exercise the happy path where duplicates are ignored intentionally.
   - File references: `tldw_Server_API/app/core/DB_Management/db_migration.py:255-301`, `tldw_Server_API/app/core/DB_Management/db_migration.py:344-366`, `tldw_Server_API/app/core/DB_Management/db_migration.py:543-645`
3. Medium: trusted SQLite path containment is lexical, not realpath-based, so a symlink placed under a trusted root can redirect writes outside that root.
   - Issue class: Isolation
   - Evidence: Source-confirmed; not directly exercised by the targeted Stage 3 tests because no symlink-escape case exists in the current path test set.
   - Confidence: High
   - Why it matters: `resolve_trusted_database_path()` normalizes with `normpath()` and accepts the first path that is textually relative to a trusted root, but it does not resolve symlinks before the containment check. Callers such as `PersonalizationDB` and the embeddings A/B test store rely on this helper as their trust boundary, so a symlinked subdirectory under a trusted base can escape to an arbitrary filesystem location while still passing validation.
   - File references: `tldw_Server_API/app/core/DB_Management/db_path_utils.py:120-178`, `tldw_Server_API/app/core/DB_Management/Personalization_DB.py:63-73`, `tldw_Server_API/app/core/Evaluations/embeddings_abtest_repository.py:663-668`
4. Low: the migration CLI advertises `--no-backup`, but the flag is never wired into the actual migrate call.
   - Issue class: Correctness
   - Evidence: Source-confirmed; the targeted Stage 3 tests did not directly exercise the CLI flag path.
   - Confidence: High
   - Why it matters: operators invoking `migrate_db.py migrate --no-backup` still take the default backup path because `main()` ignores `args.no_backup` and `migrate()` has no parameter for it. This is not a data-loss issue, but the CLI contract is inaccurate and can mislead automation or incident response workflows.
   - File references: `tldw_Server_API/app/core/DB_Management/migrate_db.py:79-122`, `tldw_Server_API/app/core/DB_Management/migrate_db.py:194-204`, `tldw_Server_API/app/core/DB_Management/migrate_db.py:228-236`

## Scope
Review the DB_Management path, tenancy, migration, policy, and backup surfaces listed in the Stage 3 task, with emphasis on containment checks, scope propagation, migration safety invariants, and backup/restore guardrails.

Limited transitive scope expansion:
- `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- `tldw_Server_API/app/api/v1/API_Deps/DB_Deps.py`
- `tldw_Server_API/app/core/DB_Management/backends/postgresql_backend.py`
- `tldw_Server_API/app/core/DB_Management/Personalization_DB.py`
- `tldw_Server_API/app/core/Evaluations/embeddings_abtest_repository.py`
- These were consulted only to confirm where `scope_context` is activated and consumed, and where `resolve_trusted_database_path()` is relied on as a trust boundary.

## Code Paths Reviewed
### Path resolution and containment map
- `db_path_utils._normalize_user_db_base_dir()` anchors relative `USER_DB_BASE_DIR` values to the project root and rejects relative escapes outside that root.
- `db_path_utils._resolve_user_id_for_storage()` and `_build_user_dir()` normalize per-user directory names, require explicit `user_id` in multi-user mode, and reject computed user directories that escape the selected base.
- `DatabasePaths.get_*_db_path()` helpers derive per-user roots for media, ChaCha, prompts, evaluations, research sessions, outputs, chatbooks, voices, and workflow storage, creating the needed subdirectories on demand.
- `normalize_output_storage_filename()` enforces flat output filenames and optional base-dir containment for absolute and relative paths.
- `resolve_trusted_database_path()` is the generic trust-boundary helper for SQLite DB paths, but it performs only lexical containment checks.
- `get_shared_audit_db_path()` and `get_shared_circuit_breaker_db_path()` resolve env-provided shared DB paths directly and do not reuse the trusted-path helper.

### Scope propagation and policy map
- `scope_context.set_scope()` stores per-request scope in the `content_scope_ctx` `ContextVar`; `get_scope()` is the retrieval point and `scoped_context()` is the temporary helper.
- `auth_deps._activate_scope_context()` populates that `ContextVar` from request/user state and stores the returned token on `request.state._content_scope_token`.
- `DB_Deps._resolve_media_db_for_user()` reads the active scope and forwards effective org/team IDs into the per-request media DB factory.
- `postgresql_backend._apply_scope_settings()` reads the same scope and writes `app.current_user_id`, `app.user_id`, `app.org_ids`, `app.team_ids`, and `app.is_admin` GUCs for PostgreSQL sessions.
- `sqlite_policy.py` only configures PRAGMAs and transaction entry (`BEGIN IMMEDIATE`); it does not emulate row-level tenant filtering in SQLite mode.
- `pg_rls_policies.py` builds Prompt Studio and ChaChaNotes RLS SQL around `current_setting('app.current_user_id', true)`, with no validation that the full policy set applied successfully.

### Migration and backup control points
- `db_migration.DatabaseMigrator` owns migration directory validation, schema version discovery, backup creation, migration execution, rollback-to-backup, and checksum verification.
- `migrations.MigrationManager` is a separate in-process SQLite migration framework used for evaluations DB schema evolution.
- `migrate_db.py` is the CLI wrapper around `DatabaseMigrator`.
- `migration_tools.py` handles SQLite-to-PostgreSQL content/workflows row copy, per-user truncation/copy, dependency ordering, and sequence synchronization.
- `content_migrate.py` is only a PostgreSQL content-backend validation CLI; it does not execute schema migrations itself.
- `DB_Backups.py` owns backup base resolution, allowed DB root resolution, SQLite URI handling, backup name validation, SQLite backup/restore via the backup API, and PostgreSQL `pg_dump`/`pg_restore` helpers.

## Tests Reviewed
- `tldw_Server_API/tests/DB_Management/test_db_path_utils.py`: covers per-user base resolution, single-user fallback, multi-user rejection, test-mode fallback isolation, prompt salt sanitization, relative base-dir escape rejection, and project-root anchoring for relative trusted DB paths. Coverage includes both happy and negative path cases. Gap: no symlink-escape case for `resolve_trusted_database_path()`.
- `tldw_Server_API/tests/DB_Management/test_db_path_utils_env.py`: covers absolute and relative `USER_DB_BASE_DIR` environment handling. Coverage is happy-path only.
- `tldw_Server_API/tests/DB_Management/test_db_paths_media_prompts_env.py`: covers media/prompts path placement under the env-configured base and directory creation. Coverage is happy-path only.
- `tldw_Server_API/tests/DB_Management/test_research_db_paths.py`: covers per-user research DB placement. Coverage is happy-path only.
- `tldw_Server_API/tests/DB_Management/test_output_storage_normalization.py`: covers safe filenames, absolute-in-base acceptance, `~` expansion, and rejection of nested, traversal, invalid-character, and outside-base paths. Coverage includes negative cases.
- `tldw_Server_API/tests/DB_Management/test_sqlite_policy.py`: covers SQLite PRAGMA application, WAL skip for in-memory DBs, outermost `BEGIN IMMEDIATE`, and async PRAGMA setup. Coverage is behavior-focused, but only for connection tuning.
- `tldw_Server_API/tests/DB_Management/test_sqlite_policy_integrations.py`: confirms many runtime DB wrappers actually apply the expected PRAGMAs and use `BEGIN IMMEDIATE` in selected mutating paths. Coverage is broad for SQLite connection policy, but it does not validate any tenancy isolation behavior.
- `tldw_Server_API/tests/DB_Management/test_db_migration_loader.py`: covers successful migration loading, version sorting, SQL baseline presence, and duplicate-version suppression. Coverage is happy-path only; malformed-file behavior is untested.
- `tldw_Server_API/tests/DB_Management/test_db_migration_path_validation.py`: covers allowed migration directories, rejection of external migration directories, rejection of rollback backups outside the backup dir, and in-memory DB rejection. Coverage includes negative cases.
- `tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py`: covers end-to-end SQLite-to-PostgreSQL content/workflows migration parity. In this environment the tests reached fixture setup and then failed because PostgreSQL on `127.0.0.1:5432` was unavailable, so they did not validate runtime assertions here.
- `tldw_Server_API/tests/DB_Management/test_migration_tools.py`: covers table truncation order, per-user truncation/copy scoping, identifier escaping, and sequence SQL escaping. Coverage includes some malformed-identifier hardening, but not partial-copy or transaction-failure cases.
- `tldw_Server_API/tests/DB_Management/test_backup_restore_verification.py`: covers end-to-end SQLite backup creation, restore, integrity checks, schema preservation, and WAL-mode backups. Coverage is mostly happy-path robustness.
- `tldw_Server_API/tests/DB_Management/test_db_backup_integrity.py`: covers missing-source backup failure, quoted paths, incremental backup directory creation, SQLite file URIs, rollback-to-backup, restore snapshot behavior, restore on busy targets, invalid backup names, flat backup layout, rejection of DB paths outside allowed roots, and rejection of traversal backup directories. Coverage includes many negative cases.
- `tldw_Server_API/tests/DB_Management/test_db_backup_name_validation.py`: covers backup label sanitization, backup basename validation, and PostgreSQL restore path confinement to the configured backup base. Coverage includes both happy and negative cases for name/path validation.

## Validation Commands
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_db_path_utils.py tldw_Server_API/tests/DB_Management/test_db_path_utils_env.py tldw_Server_API/tests/DB_Management/test_db_paths_media_prompts_env.py tldw_Server_API/tests/DB_Management/test_research_db_paths.py tldw_Server_API/tests/DB_Management/test_output_storage_normalization.py tldw_Server_API/tests/DB_Management/test_sqlite_policy.py tldw_Server_API/tests/DB_Management/test_sqlite_policy_integrations.py -v`
- Result on 2026-04-07: `54 passed` in `2.75s`
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_db_migration_loader.py tldw_Server_API/tests/DB_Management/test_db_migration_path_validation.py tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py tldw_Server_API/tests/DB_Management/test_migration_tools.py tldw_Server_API/tests/DB_Management/test_backup_restore_verification.py tldw_Server_API/tests/DB_Management/test_db_backup_integrity.py tldw_Server_API/tests/DB_Management/test_db_backup_name_validation.py -v`
- Result on 2026-04-07: `46 passed, 2 errors` in `4.10s`
- The two errors were `test_migration_cli_transfers_content_rows` and `test_migration_cli_transfers_workflow_rows`, both failing during PostgreSQL fixture setup rather than on migration assertions.
- Follow-up verification outside the sandbox: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_migration_cli_integration.py -v`
- Follow-up result on 2026-04-07: both integration tests still errored, now with `connection refused` on `127.0.0.1:5432`, confirming an environment dependency gap rather than a sandbox-only false positive.
- Confidence boundary: PostgreSQL-backed runtime behavior in Stage 3 is only partially runtime-validated here and remains partly source-traced because the PostgreSQL fixture setup failed before the migration assertions could run.

## Coverage Gaps
- No targeted Stage 3 test exercises `scope_context` activation/reset directly, `auth_deps._activate_scope_context()`, or `postgresql_backend._apply_scope_settings()`, so request-scope lifetime and GUC propagation are source-traced rather than test-confirmed.
- No targeted test validates that PostgreSQL RLS application fails closed; there is no direct coverage for partial `CREATE POLICY` / `ALTER TABLE ... FORCE ROW LEVEL SECURITY` failure.
- No targeted test exercises malformed or unreadable migration files, missing intermediate versions, or checksum drift after skipped versions.
- No targeted test exercises symlink escape attempts against `resolve_trusted_database_path()`.
- SQLite policy tests prove PRAGMA tuning and transaction mode only; they do not establish any SQLite-vs-PostgreSQL tenancy parity at the DB layer.
- The PostgreSQL migration integration tests require a live local PostgreSQL listener; without it, Stage 3 can only confirm unit-level migration logic and backup constraints.

## Improvements
- Make `ensure_prompt_studio_rls()` and `ensure_chacha_rls()` fail closed when any required statement fails, and add a regression test that asserts incomplete policy installation raises instead of returning success.
- Make `load_migrations()` fail on malformed migration artifacts or explicitly validate that pending migrations form a contiguous sequence before any upgrade begins.
- Change `resolve_trusted_database_path()` to compare resolved real paths, then add a symlink-escape regression test for both `PersonalizationDB` and embeddings A/B test path loading.
- Wire `--no-backup` through `migrate_db.py` so the CLI contract matches runtime behavior.
- Add focused tests for `scope_context` lifecycle and `postgresql_backend.apply_scope()` so Stage 4 does not have to infer scope propagation solely from source.

## Exit Note
- Stage 3 path/tenancy/migration/backup review is complete. Stage 4 should verify inside `media_db` that request scope is consistently consumed on read/write paths, that SQLite-mode filtering and PostgreSQL RLS-backed filtering do not drift semantically, and that the media DB runtime does not rely on the same fail-open assumptions identified here for policy installation or version advancement.
