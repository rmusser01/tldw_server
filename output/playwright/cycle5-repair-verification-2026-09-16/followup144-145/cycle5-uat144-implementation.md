# UAT144 / TASK13260.83 — explicit PostgreSQL single-user initialization

## Repair

`DatabasePool._should_use_postgres` previously accepted PostgreSQL only for multi-user mode or detected test processes. Normal single-user initialization followed its PostgreSQL URL, received a SQLite pool, then failed PostgreSQL schema bootstrap. Pytest exemptions masked the runtime path.

The bounded `database.py` correction honors explicit `TLDW_USER_DB_BACKEND=postgresql` or the already-supported `postgres` alias with a PostgreSQL URL. Its fallback helper honors the same selection. Absent, SQLite, and invalid selectors retain incidental-DSN fallback; SQLite URLs, multi-user selection, and existing test exemptions remain unchanged. The selector contract is documented in `Docs/AuthNZ/AUTHNZ_DATABASE_CONFIG.md:18,30,79` and implemented by the existing `AuthNZ/db_config.py:65–91` resolver. No auth/permission/schema relaxation.

## Regression and scope

New `tests/AuthNZ/integration/test_database_runtime_selection.py` runs normal Python subprocesses with a curated environment, private config/cwd, and no pytest/test-mode flags. Actual `setup_database` plus two `bootstrap_single_user_profile` calls run against fresh official `pg_temp_db` databases for both aliases. Checks include actual `current_database()`, one active verified admin, one correct primary-key hash/scope, and no fallback `Databases/users.db`. SQLite bootstrap and nine backend-selection cases provide controls. No mock database or operator/profile mutation.

Valid RED: **4 failed / 8 passed / 0 skipped**, `/private/tmp/cycle5-postgres-144-red-valid.redacted.log`. Both aliases failed actual setup and selected SQLite. An earlier preliminary test run used an invalid maximum pool size of2; this fixture mistake was corrected to the documented minimum5 before the valid RED and production edit.

Initial expanded verification: **39 passed / 3 failed / 0 skipped**; all12 new controls passed. The three pre-existing PostgreSQL tests attempted prohibited direct users writes. Baseline reproduction also yielded3 failures/1pass; tracked and repaired separately under UAT145/TASK13260.84, without changing this unit's production/new-test freeze.

Final combined verification: **42 passed / 0 failed / 0 skipped**,82 warnings,125.78s; `/private/tmp/cycle5-postgres-144-145-final-green.redacted.log`.

Exact command (official helper activates `.venv`, supplies private fixture environment, requires PostgreSQL, and disables Docker auto-start):

```sh
node /private/tmp/cycle5-postgres-fixture-run.mjs \
  tldw_Server_API/tests/AuthNZ/integration/test_database_runtime_selection.py \
  tldw_Server_API/tests/AuthNZ/integration/test_single_user_bootstrap_postgres.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_single_user_bootstrap_sqlite.py \
  tldw_Server_API/tests/AuthNZ/unit/test_database_openai_credential_lock_pool.py \
  tldw_Server_API/tests/AuthNZ/unit/test_database_pool_fetchone_sqlite_fallback.py \
  -q -rs --show-capture=no > /private/tmp/cycle5-postgres-144-145-final-green.log 2>&1
node /private/tmp/cycle5-postgres-report.mjs /private/tmp/cycle5-postgres-144-145-final-green.log --save
```

PostgreSQL fixture server: official isolated PostgreSQL18.6 on port55475. No required-PG skips accepted. Native holder databases and runtime profiles were untouched. Only fresh disposable fixture databases were initialized.

Static checks: project-venv Ruff0 for production/new test; production Bandit0 (also0 baseline); scoped `git diff --check` clean. Retained JSON: `/private/tmp/cycle5-uat144-{ruff-final,bandit-final}.json`.

Freeze: `/private/tmp/cycle5-uat144-code-freeze.json` (two code/test paths,16:04:46.600Z); final owned manifest also includes task83. No staging/commit. Independent review is clear:12/12 required-PG controls and matching hashes, `/private/tmp/cycle5-uat144-independent-review.md`. Root separately verified actual `AuthNZ.initialize --non-interactive` on a new empty PostgreSQL r2 profile: exit0, admin/key ensured, no users.db fallback or application test flags (`/private/tmp/cycle5-native-pg-r2-single-init.redacted.log`). Root owns subsequent API/native workflow validation; these subprocess checks are not full workflow signoff.
