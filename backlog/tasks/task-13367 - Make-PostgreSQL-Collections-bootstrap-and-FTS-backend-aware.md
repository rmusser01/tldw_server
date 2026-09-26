---
id: TASK-13367
title: Make PostgreSQL Collections bootstrap and FTS backend-aware
status: Done
assignee: []
created_date: '2026-09-25 19:16'
updated_date: '2026-09-25 19:58'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A real PostgreSQL email upload reaches CollectionsDatabase.ensure_schema and fails on ALTER TABLE output_templates ADD COLUMN metadata_json because the column already exists. PostgreSQL column sets are left empty while SQLite introspects columns; backend redacts the duplicate-column error so the no-op handler cannot identify it. Use the existing backend table-info abstraction for PostgreSQL before backfills, with real integration coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PostgreSQL Collections bootstrap does not attempt ADD COLUMN for columns already in freshly created tables
- [x] #2 Existing PostgreSQL Collections integration test passes through bootstrap and CRUD
- [x] #3 Live synthetic PostgreSQL email upload gets past Collections schema sync
- [x] #4 A second PostgreSQL Collections adapter does not issue SQLite FTS5 writes after cached bootstrap
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Real HTTP upload stack identifies first failure at CollectionsDatabase.ensure_schema ALTER TABLE output_templates ADD COLUMN metadata_json on PostgreSQL. Existing _table_columns uses backend.get_table_info, but ensure_schema populates its initial column sets only for SQLite. Prior TASK-12910 covered a separate SQLite content_items timing issue.

Diagnostic full-app probe also found INSERT INTO content_items_fts from the second PostgreSQL Collections adapter. Constructor initializes _fts_available=True, but cached bootstrap skips ensure_schema, which normally resets it to False. This is part of the same PostgreSQL Collections readiness unit.

PostgreSQL Collections backfills use backend column introspection; cached adapters disable SQLite FTS5 writes. Real PostgreSQL round-trip passed; three schema unit tests passed; final live probe had zero PostgreSQL query failures. Bandit 0 findings; fatal Ruff clean. Skip: no scale test.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PostgreSQL Collections backfills use backend column introspection; cached adapters disable SQLite FTS5 writes. Real PostgreSQL round-trip passed; three schema unit tests passed; final live probe had zero PostgreSQL query failures. Bandit 0 findings; fatal Ruff clean. Skip: no scale test.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
