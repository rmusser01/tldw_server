# UAT144 independent review

## Verdict

No actionable finding in the frozen two-file correction. The explicit PostgreSQL selector repair is approved for integration; fresh native initialization/startup remains a separate acceptance step.

## Scope and cause

Reviewed production `tldw_Server_API/app/core/AuthNZ/database.py` and new `tldw_Server_API/tests/AuthNZ/integration/test_database_runtime_selection.py`, frozen at 2026-09-16T16:04:46.600Z. Both hashes match; see /private/tmp/cycle5-uat144-independent-hashes.json.

The original non-test initialization evidence (/private/tmp/cycle5-native-pg-single-init-redacted.log) shows the pool choosing SQLite despite an explicit PostgreSQL backend, then initialization attempting PostgreSQL schema operations and failing. The prior test exemption masked this mismatch. The two new guards at database.py:403 and :985 honor the existing postgres/postgresql selector only for a PostgreSQL URL. Incidental DSN fallback, SQLite URLs, multi-user behavior, and existing test exemptions remain intact. Selector normalization matches the existing AuthNZ db_config contract. PostgreSQL connection failures still fail initialization; this change adds no automatic fallback or credential behavior.

## Independent verification

- Official fixture command: `node /private/tmp/cycle5-postgres-fixture-run.mjs tldw_Server_API/tests/AuthNZ/integration/test_database_runtime_selection.py -q -rs --show-capture=no`.
- Host-access run: **12 passed, zero skipped**, 45.36s, exit 0. Safe log: /private/tmp/cycle5-postgres-144-independent-host.redacted.log.
- The first restricted-sandbox run produced 10 passed / 2 required-PG reachability errors; this is retained separately at /private/tmp/cycle5-postgres-144-independent.redacted.log. No skips or fabricated database substitute were used.
- The permanent normal-process tests construct a private environment excluding inherited pytest/test flags, assert test mode is off, and exercise both explicit PG spellings on official disposable PostgreSQL databases. They verify current_database(), one fixed-ID active verified admin, the configured primary-key hash, two successful bootstrap calls, and no fallback SQLite file. SQLite bootstrap and nine selector combinations also pass. Precisely: schema setup runs once per process, profile bootstrap twice; this is not a claim of two complete schema setups or an application-server launch.
- Inspected author RED evidence: /private/tmp/cycle5-postgres-144-red-valid.redacted.log, 4 failed / 8 passed before the correction.
- Inspected author static artifacts: /private/tmp/cycle5-uat144-bandit-final.json has zero findings/errors; /private/tmp/cycle5-uat144-ruff-final.json has zero findings. These static checks were inspected, not independently rerun.

## Adjacent gate and limits

The author's expanded run had 39 passed / 3 failed / zero skipped. All three failures occurred in pre-existing PostgreSQL bootstrap fixtures at direct guarded users INSERTs, before their target assertions. /private/tmp/cycle5-postgres-144-baseline-existing-valid.redacted.log reproduces the same 3 failures / 1 pass against baseline production. They are tracked separately as UAT145/TASK13260.84, not attributed to this selector change; review and rerun of that correction are separate.

This review made no repository/task changes, launched no browser/application runtime or inference, and used only official disposable test databases. Fresh native startup, HTTP authentication, and broader PostgreSQL product flows are not certified by these bounded tests.
