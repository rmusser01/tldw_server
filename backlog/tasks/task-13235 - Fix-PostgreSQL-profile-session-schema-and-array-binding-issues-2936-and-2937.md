---
id: TASK-13235
title: Fix PostgreSQL profile session schema and array binding (issues 2936 and 2937)
status: Done
assignee: []
created_date: '2026-09-10 03:35'
updated_date: '2026-09-10 04:08'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2936'
  - 'https://github.com/rmusser01/tldw_server/issues/2937'
documentation:
  - Docs/Design/ISSUES_2935_2938_REGRESSION_REPAIR.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair production sessions last_activity schema and profile override array parameters on dev. Audit related PostgreSQL ANY callers and session touch/refresh paths. Add behavior regressions using production schema and the real DatabasePool; validate available PostgreSQL fixtures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh and upgraded PostgreSQL sessions support listing, touch, and refresh without missing columns
- [x] #2 Org and team overrides preserve empty, single, and multiple ID array bindings through DatabasePool
- [x] #3 Related callers audited and confirmed defects fixed with regression tests
- [x] #4 Targeted tests, formatting, Bandit, and review recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Completed session schema repair, seven array-binding method repairs, regressions, and independent review. Design and final audit retained in Docs/Design/ISSUES_2935_2938_REGRESSION_REPAIR.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Array audit found three additional affected methods: ManagedSecretRefsRepo.list_refs_by_ids and organization/user scoped total counts. Seven affected methods now preserve a single array parameter through DatabasePool. Regression red: 16 failed, 14 passed; focused green including override readiness: 49 passed. Real PostgreSQL first run verified profile overrides and scoped counts; corrected missing metadata argument in the new managed-secret test and rerun is active. Independent review found no array production defect. Sessions migration work continues.

Final combined auth/profile suite: 86 passed, 78 warnings in 227.54s, including real PostgreSQL production-schema and array cases. Session worker: 24 focused unit tests and 4 PostgreSQL integration tests passed. SQLite compatibility: 8 passed. Existing PostgreSQL session tests now seed through UsersDB and isolated_test_environment; production write guards remain enabled. Independent review found no remaining defects. New test files pass Black and Ruff; session production scope passes Ruff. Array scope has one unchanged SIM118 finding, confirmed against base. Bandit for all eight touched auth/profile production files: zero findings. Broader migrations: 52 passed, 1 existing failure in test_sqlite_upgrade_preserves_custom_users_schema_objects_and_foreign_keys. Single-test rerun on exact base 751563a966 fails identically with seven extra users columns, starting with uuid; this is unrelated to migration98. Full backend suite was not run.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added PostgreSQL sessions.last_activity bootstrap/upgrade/default and safe historical backfill. Added SQLite migration98 and explicit activity initialization for new sessions after upgrade. Fixed all four profile override array methods plus managed-secret lookup and organization/user scoped counts without changing DatabasePool compatibility. Added real-driver, production-schema, property, refresh, and upgrade regressions; repaired older PostgreSQL session test setup to exercise current guarded writes.
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
