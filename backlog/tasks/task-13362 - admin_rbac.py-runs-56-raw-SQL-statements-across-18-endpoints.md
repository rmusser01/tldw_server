---
id: TASK-13362
title: admin_rbac.py runs 56 raw SQL statements across 18 endpoints
status: Done
assignee: []
created_date: '2026-09-23 17:27'
updated_date: '2026-09-23 19:33'
labels:
  - refactor
  - authnz
  - layering
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Last file in the raw-SQL ratchet (tests/lint/test_no_raw_sql_in_endpoints.py RAW_SQL_BASELINE). 18 async endpoints (tool permissions, roles matrix, permissions, role grants, user overrides, role effective permissions) run SQL directly on the AuthNZ pool/transaction connection. core/AuthNZ/repos/rbac_repo.py exists but is a synchronous wrapper over UserDatabase and covers only reads (effective permissions, user roles/overrides, role effective permissions). Route the endpoints through an async RBAC owner and drop the file from the baseline. Split from TASK-13317.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 admin_rbac.py leaves RAW_SQL_BASELINE
- [x] #2 RBAC writes and reads used by admin_rbac.py live in one async core owner
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
admin_rbac.py's 56 statements moved to core/AuthNZ/repos/rbac_admin_repo.py (async, one query per operation with ? placeholders, dual-backend via repos/_dual_backend.py). RAW_SQL_BASELINE is empty. Verified: migrated-SQLite repo test; RBAC endpoint tests 54 passed; Admin/Watchlists/Health and AuthNZ unit suites show no new failures vs HEAD; bandit clean. A suspected SQLite override-upsert call-shape bug was checked and is not a bug (the pool shim accepts positional params).
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
