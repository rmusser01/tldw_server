# UAT149 / TASK13260.87 — PostgreSQL Collections bootstrap

## Repair

Actual normal PostgreSQL startup logged duplicate `output_templates.metadata_json` and `outputs.deleted` errors and skipped reading-digest scheduler startup. Collections gathered existing columns only for SQLite, then blindly attempted legacy ALTER statements on fresh PostgreSQL tables. Its text-based duplicate handler cannot recognize deliberately sanitized PostgreSQL exceptions.

The minimal correction reuses the existing `_table_columns` helper for five existing table inventories and the content-items inventory after that table is created. PostgreSQL uses existing backend `get_table_info`; SQLite keeps its existing PRAGMA helper. No new helper, public error disclosure, schema redesign, broad suppression, or database fallback. Actual missing columns still receive their existing backfills; inspection errors propagate.

## Evidence

- Valid RED: `/private/tmp/cycle5-postgres-149-red-confirmed.redacted.log`,3failures,0skips. All stop at duplicate output-template metadata ALTER before target behavior.
- GREEN: `/private/tmp/cycle5-postgres-149-green-final.redacted.log`,7passed,0skips,exit0. Three new actual official-PG controls cover fresh plus explicitly repeated bootstrap preserving template content, a genuinely missing legacy column, and propagation of a bounded catalog-inspection fault. Existing SQLite schema controls and both PostgreSQL integration tests pass, including artifact/content round trips and notification-column upgrades.
- Exact command: `node /private/tmp/cycle5-postgres-fixture-run.mjs tldw_Server_API/tests/Collections/test_collections_postgres_bootstrap.py tldw_Server_API/tests/Collections/test_collections_schema_bootstrap.py tldw_Server_API/tests/Collections/test_collections_postgres_integration.py -q -rs --show-capture=no`, redirected to the named private log and sanitized with `/private/tmp/cycle5-postgres-report.mjs <log> --save`. Official required-PG fixtures; no skips or held native database changes.
- Two preliminary RED harness mistakes used a nonexistent factory method and called an instance key helper statically. They were corrected before the valid RED. The first GREEN reached a new test's missing required pagination arguments (1fail/6pass); the actual API signature and existing integration call were re-read and copied before final GREEN. These are test-author mistakes, not additional product defects. Private intermediate logs remain separate.
- Production Bandit0; Ruff signature counts unchanged versus the pre-edit baseline; new test Ruff0. Structured summary `/private/tmp/cycle5-uat149-static-summary.json`, raw baseline/final artifacts alongside it. Scoped diff-check clean.

Frozen two paths: `/private/tmp/cycle5-uat149-code-freeze.json`. Independent review and normal-runtime worker acceptance pending. Source coverage proves Collections initialization; the exact reading-digest worker consequence remains to be rechecked in the restarted full app. Existing optional startup warnings and pytest old-temp cleanup warnings are not claimed fixed.
