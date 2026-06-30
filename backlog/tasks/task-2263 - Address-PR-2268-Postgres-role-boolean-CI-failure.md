---
id: TASK-2263
title: Address PR 2268 Postgres role boolean CI failure
status: Done
labels:
- pr-2268
- ci
- authnz
- postgres
modified_files:
- tldw_Server_API/app/core/AuthNZ/repos/rbac_repo.py
- tldw_Server_API/app/api/v1/endpoints/admin/admin_rbac.py
- tldw_Server_API/tests/AuthNZ/unit/test_rbac_repo_backend_selection.py
- tldw_Server_API/tests/AuthNZ/unit/test_admin_rbac_error_mapping.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the Postgres-compatible RBAC role queries surfaced by PR #2268 full-suite logs after rebasing on latest dev. Ensure boolean role columns use boolean defaults in Postgres while preserving SQLite integer defaults.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Postgres role queries no longer use integer 0 as the fallback for boolean is_system columns.
- [ ] #2 SQLite role queries retain existing 0 fallback behavior.
- [ ] #3 Focused tests cover backend-specific SQL generation or query execution shape.
- [ ] #4 Touched Python scope passes targeted tests and Bandit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2268 on origin/dev and fixed the remaining PR CI issue from the full-suite logs: Postgres RBAC role queries now use boolean FALSE fallbacks for is_system. Focused backend, frontend Mermaid, Bandit, and diff checks were run; live AuthNZ Postgres smoke skipped because the fixture could not provide PostgreSQL locally.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
