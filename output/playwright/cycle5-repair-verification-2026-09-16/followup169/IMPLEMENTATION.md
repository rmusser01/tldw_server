# UAT169 / TASK-13260.106 — authentication monitor timestamp binding

## Result

Frozen for independent review. One production expression converts the already-UTC five-minute cutoff to naive datetime when binding PostgreSQL `audit_logs.created_at`. Real PostgreSQL now completes the monitor and dispatches the expected alert. SQLite behavior stays covered. No migration, timezone helper, broad exception handling, or other monitor change.

## Cause and scope

- Actual current PGmulti logs repeatedly showed `asyncpg.exceptions.DataError` for `$2` at `_monitor_auth_failures`, caused by subtracting offset-naive and offset-aware datetimes while encoding the timestamp parameter.
- Canonical `pg_migrations_extra.py` creates `audit_logs.created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP`; the official AuthNZ fixture agrees. Existing `AuthnzMonitoringRepo` binds naive values for this table.
- The monitor already obtains UTC time explicitly. `.replace(tzinfo=None)` here removes timezone metadata after UTC time was selected, retaining its clock value. It is not a general-purpose offset conversion.
- Owned production path is only `tldw_Server_API/app/core/AuthNZ/scheduler.py`, one expression plus schema comment. Two new permanent test files are listed in `owned-paths.json`; snapshots and hashes are in `owned-manifest.json`.

## Meaningful tests

The permanent PostgreSQL test uses official `isolated_test_environment`, which provisions and removes a randomly named isolated test database, then a real `DatabasePool` on the test loop. SQLite uses temporary real `DatabasePool` schema initialization. SQL and database responses are not mocked. Clock and external alert delivery are substituted.

Each backend runs with PII redaction off and on. A fixed 00:03 UTC time makes the five-minute cutoff cross the date boundary. Ten matching failures one microsecond after cutoff must produce no alert. Matching failures exactly at cutoff and one microsecond before, plus unrelated recent actions, must not count. An eleventh valid failure triggers exactly one alert with 11 failures, three distinct IPs, high severity and the five-minute metadata. The real `_send_security_alert` path adds its source metadata, and its subject/message/log redaction are verified.

## Verification evidence

| Check | Result | Evidence |
|---|---|---|
| Initial sandboxed harness | 2 SQLite PASS, 2 PG setup ERROR before queries; not claimed RED | `uat169-red.redacted.log` |
| Baseline production meaningful RED | 2 PG FAIL with exact naive/aware asyncpg `$2` error; 2 SQLite PASS; 0 skips | `uat169-red-isolated.redacted.log` |
| Same tests after minimal fix | 4 PASS, 8 warnings, 0 skips | `uat169-green.redacted.log` |
| Adjacent repository backend selection + real SQLite monitoring | 5 PASS, 10 warnings, 0 skips | `adjacent-controls.log` |
| Ruff owned scope | One pre-existing I001 import-order diagnostic, identical baseline/current; new tests clean | `ruff-baseline.json`, `ruff-current.json`, `ruff-tests.json` |
| Bandit production | 0 findings, 0 errors | `bandit-production.json` |
| Bandit tests | 0 findings, 0 errors; B101 excluded only for pytest behavioral assertions | `bandit-tests.json` |
| `git diff --check` owned scope | PASS | independently rerunnable |

Required real PG command (from repo root, with project virtual environment active):

```sh
TLDW_UAT_EVIDENCE_LABEL=uat169-green node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs -q tldw_Server_API/tests/AuthNZ/integration/test_scheduler_auth_failure_monitor_postgres.py tldw_Server_API/tests/AuthNZ_SQLite/test_scheduler_auth_failure_monitor_sqlite.py --tb=short
```

The existing runner sets `TLDW_TEST_POSTGRES_REQUIRED=1` and `TLDW_TEST_NO_DOCKER=1`, reads credentials privately, and redacts its DSN/password from output. Connected runs required network sandbox escalation. They reuse the existing owned cluster; no service process or runtime database was changed. Command receipts are retained alongside logs.

Controls command:

```sh
python -m pytest -q tldw_Server_API/tests/AuthNZ/unit/test_authnz_monitoring_repo_backend_selection.py tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_monitoring_repo_sqlite.py --tb=short
```

## Limits

- This is real database verification of the actual scheduler method, not a live five-minute scheduler acceptance run. Parent owns runtime acceptance and independent review.
- SQLite controls exercise the existing ISO timestamp input convention; this repair does not broaden or migrate historical timestamp formats.
- The adjacent API-usage monitor is unchanged; its distinct timestamp contract was not included in this repair.
- A first attempted fixture choice was corrected before any connected database run: the private runner points its base DSN at `/postgres`, so the official random-name isolated fixture is used instead of the DSN-named `test_db_pool` fixture. The failed sandbox attempt performed no PostgreSQL setup operations.
- Baseline lint and pytest warnings remain explicitly recorded. No source/test/task/tracker edits outside the three owned paths; evidence only under this private packet. No browser, app runtime, service restart, staging, or commit actions.
