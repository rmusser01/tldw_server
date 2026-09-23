---
id: TASK-13362
title: admin_rbac.py runs 56 raw SQL statements across 18 endpoints
status: To Do
assignee: []
created_date: '2026-09-23 17:27'
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
- [ ] #1 admin_rbac.py leaves RAW_SQL_BASELINE
- [ ] #2 RBAC writes and reads used by admin_rbac.py live in one async core owner
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
