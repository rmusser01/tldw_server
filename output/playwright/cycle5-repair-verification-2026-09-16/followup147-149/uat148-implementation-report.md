# UAT148 / TASK13260.86 — MCP Media writable health

## Status
Frozen for independent review; task remains In Progress. No staging, commit, runtime, native profile, held database or inference changes. Parent owns startup acceptance and global documentation.

## Root cause and bounded correction
`MediaModule.check_health` ran SQLite-only `INSERT OR REPLACE` through the existing Media DB adapter. PostgreSQL rejects that syntax. The adapter translates driver failures into `media_db.errors.DatabaseError`, which was absent from the local read/write health catches, so the actual method raised instead of returning useful health flags.

The existing DB abstraction is retained: `db.transaction()` plus parameterized `db.execute_query()`. Each check now owns a UUID key and uses standard INSERT, then DELETE in the same transaction. Unique keys avoid overwriting/removing another check's row. Cleanup is mandatory; its failure rolls back and returns database_writable=false. Local read/write catches include DatabaseError. No shared extractor, SQL framework, DB-layer rewrite, migration or global exception change.

The existing empty `_mcp_healthcheck` table remains after successful checks; newly inserted probe rows do not. Existing unrelated rows are preserved. A failed read/write returns false for that dimension; the generic base health layer may report degraded rather than wholly unhealthy, but never healthy when a DB check fails.

## Owned files
- tldw_Server_API/app/core/MCP_unified/modules/implementations/media_module.py (19-line diff: 10 additions, 9 removals)
- tldw_Server_API/tests/DB_Management/test_mcp_media_health.py (new permanent actual-backend tests)
- TASK13260.86 updated only via official MCP; AC1/2 checked, AC3 remains pending.

SHA256/source freeze: `/private/tmp/uat148-owned-manifest.json`.

## RED / GREEN
Command prefix: `node /private/tmp/cycle5-postgres-fixture-run.mjs` with official `pg_database_config`/temporary database fixtures, PostgreSQL required, Docker autostart disabled. These are new disposable test DBs, not native held fixtures.

1. RED before production edit: `tldw_Server_API/tests/DB_Management/test_mcp_media_health.py -q --tb=short` → **9 failed, 1 passed**. Actual PG check_health throws translated DatabaseError; actual SQLite healthy control passes. Existing ping row is destroyed by the old SQLite upsert/delete; read/insert/cleanup failures escape.
2. GREEN same command → **10 passed, zero skips**.
3. Covering run adds `tldw_Server_API/tests/MCP_unified/test_mcp_hub_tool_registry.py` and `test_module_registry_sanitization.py` → **21 passed, zero skips**.

Permanent controls on BOTH real SQLite and PostgreSQL:
- Two successful checks with no leftover probe rows.
- Existing ping/other-probe row preserved exactly.
- Real INSERT CHECK-constraint failure returns not-writable and rolls back; a subsequent valid write succeeds.
- Narrow DB cleanup-failure injection returns not-writable and rolls back the actual inserted row.
- Real missing-table read error is translated/caught; connection flag false while the independent write probe remains usable.

Only disk free-space signal is stabilized to 5GB. No DB backend or healthy SQL response is mocked. Cleanup fault injection targets only DELETE; read failure invokes the real adapter against an absent table.

Redacted logs: `/private/tmp/uat148-red.redacted.log`, `/private/tmp/uat148-green.redacted.log`, `/private/tmp/uat148-covering.redacted.log`. Raw logs remain private and must not be surfaced. Runs report four existing warnings and unrelated pytest garbage-directory cleanup warnings for old Kokoro test artifacts; no files were removed to suppress those warnings.

## Quality checks
- Scoped Bandit on production file: **0 findings, 0 errors** (`/private/tmp/uat148-bandit.json`).
- Ruff baseline/current: **52 diagnostics each** (I001=1, UP045=51), identical code/message/source-line signatures after line shifts; new test **0**. No broad formatting churn. JSONs `/private/tmp/uat148-ruff-{baseline,current}.json`.
- Owned `git diff --check`: clear.

## Remaining verification / limits
Independent review and actual normal-runtime MCP startup/periodic health remain parent-owned and unclaimed. The regression constructs the actual Media database with each backend and calls the actual MediaModule.check_health; it does not claim a complete native startup pass. No broader UAT was run. Current probe still requires CREATE TABLE permission, as the existing contract did.
