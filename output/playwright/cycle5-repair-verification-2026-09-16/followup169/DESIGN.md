# UAT169 / TASK-13260.106: scheduler audit cutoff

Parent approved this bounded design on 2026-09-16 before edits.

## Contract and fix

`audit_logs.created_at` is PostgreSQL `TIMESTAMP` without time zone in the canonical AuthNZ bootstrap and official fixtures. Authentication monitor computes its cutoff with `datetime.now(timezone.utc)`, but passes that aware value to asyncpg. The current runtime repeatedly raises the naive/aware timestamp encoding error.

Convert the already-UTC cutoff to a naive datetime only at `_monitor_auth_failures`'s PostgreSQL bind boundary. Keep the SQLite ISO parameter, SQL action allowlist, strict five-minute `>` comparison, 10-failure threshold, distinct-IP aggregation, and redaction behavior. No migration, generalized datetime helper, exception suppression, or adjacent API-usage monitor change.

## Owned paths

- `tldw_Server_API/app/core/AuthNZ/scheduler.py`
- `tldw_Server_API/tests/AuthNZ/integration/test_scheduler_auth_failure_monitor_postgres.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_scheduler_auth_failure_monitor_sqlite.py`

## Verification

Run new tests before source change. Real PostgreSQL uses the official `isolated_test_environment` fixture via the existing private required-PG runner, plus a real `DatabasePool` on the test loop. SQLite uses a temporary real `DatabasePool` with normal schema initialization. Fix time at UTC midnight crossover and seed rows one microsecond after, at, and one microsecond before the cutoff; include unrelated recent actions. Assert no alert for exactly ten, then a single real scheduler dispatch for eleven across three IPs, under both redaction settings. Only clock and external alert dispatcher are substituted. Retain RED and GREEN results; run adjacent monitoring regressions, scoped lint, and Bandit before review.

The initial sandboxed harness attempt produced two PostgreSQL setup errors and two passing SQLite controls, not a valid RED. On inspecting the private runner's `/postgres` DSN, the official isolated fixture was selected rather than `test_db_pool`, whose setup fixture recreates the DSN-named database. Parent confirmed isolated fixture choice before the connected run. No database operations were performed by the unavailable initial PostgreSQL setup.
