# UAT150 — backend-appropriate cached ChaCha health

Task TASK-13260.89. Production: `tldw_Server_API/app/api/v1/API_Deps/ChaCha_Notes_DB_Deps.py`; new tests: `tldw_Server_API/tests/Chat/test_chacha_notes_db_deps_postgres_health.py`.

## Diagnosis and fix

`_health_check_instance` unconditionally called the SQLite policy helper. Its first statement is `PRAGMA database_list` (sqlite_policy.py:11); its error is caught by that helper, then `PRAGMA foreign_keys=ON` escapes. PostgreSQL's external pinned connection stays in a failed transaction. The cached dependency propagates BackendDatabaseError because that class was absent from its exception handler. This explains a first uncached multi-user read succeeding and following reads failing, while a startup-warmed single-user cache fails immediately.

A disposable real-PG driver observer captured SQLSTATE42601,42601 then25P02 on subsequent SELECT1. Explicit rollback restored reads. Confirmed private probe: `/private/tmp/cycle5-postgres-150-health-probe-confirmed.redacted.log` (1passed). The earlier diagnostic run expected one syntax exception rather than two and failed its observer-count assertion; it is not the permanent RED evidence.

The fix runs PG SELECT1 inside the existing ChaCha transaction manager. A failed owned probe rolls back and returnsFalse using the existing unhealthy-cache path; a successful probe settles its transaction. Existing nested transaction ownership is preserved. SQLite policy remains unchanged. The failure log still records only exception type.

No ChaCha implementation, backend, SQL translation, graph projection, RLS, schema, auth, or runtime changes. The graph suggestion FTS helper already branches for PostgreSQL and is not the origin in the actual trace.

## Verification

Permanent RED: **4failed,0skipped**,14.94s. `/private/tmp/cycle5-postgres-150-final-red.redacted.log`.

- Repeat actual PG liveness and a following query.
- Real division-by-zero abort; health must returnFalse, roll back, then subsequent query/probe succeeds.
- Actual cached async dependency twice returns the same healthy PG instance, without rebuilding.
- Health inside a caller transaction must not commit its pending insert; caller rollback removes it.

Final GREEN: **52passed,0skipped**,7files,16.41s,exit0. PostgreSQL18.6 official disposable fixtures, REQUIRED=1/NO_DOCKER=1. Existing SQLite tuning/health, sanitized errors, startup, PG transaction and session-scope controls included. Two existing warnings.

```sh
node /private/tmp/cycle5-postgres-fixture-run.mjs tldw_Server_API/tests/Chat/test_chacha_notes_db_deps_postgres_health.py tldw_Server_API/tests/Chat/test_chacha_notes_db_deps_sqlite_policy.py tldw_Server_API/tests/Chat/test_chacha_db_deps_error_mapping.py tldw_Server_API/tests/API_Deps/test_chacha_notes_db_deps_error_mapping.py tldw_Server_API/tests/Services/test_startup_chacha_warmup.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_transactions.py tldw_Server_API/tests/DB_Management/test_chacha_postgres_session_scope.py -q -rs > /private/tmp/cycle5-postgres-150-green.log 2>&1
node /private/tmp/cycle5-postgres-report.mjs /private/tmp/cycle5-postgres-150-green.log --save
```

Project venv Ruff production baseline0/final owned production+test0. Bandit production baseline0/final0. Logs `/private/tmp/cycle5-uat150-{ruff-baseline,ruff-final,bandit-baseline,bandit-final}.json`.

## Acceptance limits

Tests exercise real PostgreSQL and the actual cached dependency, but do not authenticate native HTTP requests. Repeated normal-runtime single/multi authenticated Characters/Chats/Notes reads and independent review remain parent-owned acceptance. Held r3 databases/profiles were not changed. Existing global cache eviction/rebuild code is unchanged. No broad DB-error masking or automatic transaction retry added.
