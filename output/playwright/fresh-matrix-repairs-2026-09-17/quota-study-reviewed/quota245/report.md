# UAT245 — PostgreSQL storage quota bootstrap repair

Status: FROZEN for independent review. Final regression: **52 passed across six suites, zero skips**, 60 warnings, 89.85 seconds. Native acceptance remains pending.

## Cause and change

Normal PostgreSQL AuthNZ initialization omitted SQLite migration 051's shared org/team storage quota schema. The admission guard correctly failed closed when its real repository queried the absent table. Added the table and four indexes to the existing required, transactional `ensure_authnz_core_tables_pg` bootstrap sequence, after organization/team parents. The partial unique indexes match repository `ON CONFLICT ... WHERE ...` targets. Defaults, fractional usage, mutually exclusive scope, both-null compatibility, foreign keys, and cascade semantics follow the existing SQLite schema. Repeated initialization retains configured quotas and usage.

Only production file: `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py` (32 lines added). No request-time DDL, quota-policy changes, new grants, RLS changes, or runtime/profile mutations.

## Regression boundaries

Two new quota-specific files:
- `tldw_Server_API/tests/AuthNZ/integration/test_storage_quotas_bootstrap_postgres.py`: official disposable PostgreSQL fixture and direct restricted login; normal non-test subprocess initialization twice in both auth modes; actual repository CRUD/upsert/fractional usage; constraints/cascades; actual FastAPI quota dependency under/soft/hard/insufficient-space, absent-quota, missing-table fail-closed, and recovery. Repeated bootstrap preserves quota values. Restricted login explicitly has no superuser, BYPASSRLS, inheritance, memberships, create-role, create-database, or replication privileges. Role and scratch database cleanup use the official fixture; held native databases are untouched.
- `tldw_Server_API/tests/AuthNZ/unit/test_pg_storage_quotas_schema.py`: required quota table/index creation failure rejects canonical bootstrap before permission seed.

The admission test calls the real dependency with an isolated request and synthetic authenticated-user context. It does not claim full HTTP endpoint/auth middleware coverage or native ingest success. No LLM calls or browser actions.

## RED and harness corrections

`quota245-causal-red-final.redacted.log`: six failures, zero skips. All four real PostgreSQL cases reached absent-quota read after successful repeated initialization and raised `UndefinedTableError`; both required-DDL controls showed bootstrap incorrectly returning success without attempting quota DDL.

Earlier sandbox run could not reach PostgreSQL, so it is not causal evidence. Initial host fixture diagnostics failed on `SHOW server_version` rejected by the existing SQL guard; replaced with supported `SELECT current_setting`. First post-change run had five passes and one fixture-only parser failure from adjacent placeholders `$1,$2`; adding whitespace preserved the intended SQL and passed offline SQL parser controls. Neither correction changed production guards or quota behavior.

## Static checks

Scoped Ruff: zero baseline and final findings across owned production/tests. Scoped production Bandit: zero baseline and final findings. `git diff --check` passed. Receipts: `static-summary.json`, `ruff-{baseline,final}.json`, `bandit-{baseline,final}.json`.

## Official test command

```sh
source .venv/bin/activate && TLDW_UAT_EVIDENCE_LABEL=quota245-green-final node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/AuthNZ/integration/test_storage_quotas_bootstrap_postgres.py tldw_Server_API/tests/AuthNZ/unit/test_pg_storage_quotas_schema.py tldw_Server_API/tests/AuthNZ/unit/test_pg_migrations_authnz_core.py tldw_Server_API/tests/AuthNZ/integration/test_database_runtime_selection.py tldw_Server_API/tests/Admin/test_admin_storage_quotas.py tldw_Server_API/tests/Billing/test_storage_quota_guard.py -q -rs
```

Uses required official PostgreSQL fixture runner with host access; no independent container/database provisioning. Final command receipt and sanitized log are retained beside this report (`green-final-receipt.txt`, `quota245-green-final.redacted.log`, `quota245-green-final-command.json`). Root owns task updates, retention, independent review, native verification, and commits.
