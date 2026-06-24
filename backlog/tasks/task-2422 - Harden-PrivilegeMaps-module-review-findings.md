---
id: TASK-2422
title: Harden PrivilegeMaps module review findings
status: Done
assignee: []
created_date: 2026-06-23 18:26
updated_date: 2026-06-24 19:49
labels:
- authnz
- privilege-maps
- review-fix
dependencies: []
references:
- tldw_Server_API/app/core/PrivilegeMaps/service.py
- tldw_Server_API/app/core/PrivilegeMaps/snapshots.py
- tldw_Server_API/app/core/PrivilegeMaps/trends.py
- tldw_Server_API/app/api/v1/endpoints/privileges.py
priority: high
modified_files:
- tldw_Server_API/app/core/PrivilegeMaps/db_utils.py
- tldw_Server_API/app/core/PrivilegeMaps/service.py
- tldw_Server_API/tests/Privileges/test_privilege_endpoints.py
- tldw_Server_API/tests/Privileges/test_privilege_service_sqlite.py
- tldw_Server_API/tests/Privileges/test_privilege_snapshot_store.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address code review findings in PrivilegeMaps: Postgres-safe snapshot/trend writes, RBAC parity for effective permissions, fail-closed user fetch behavior, collision-safe snapshot IDs, active membership filtering, bounded detail generation, org-scoped trends, and removal of an unused helper.
<!-- SECTION:DESCRIPTION:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused tests pass.
- [x] #8 Bandit runs on touched Python files.
- [x] #9 Backlog task records verification and final summary.
<!-- DOD:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Postgres-backed snapshot and trend stores use normalized parameter handling inside transactions.
- [x] #2 PrivilegeMaps effective permissions honor expired roles, expired overrides, and explicit deny overrides.
- [x] #3 Multi-user DB failures do not fall back to synthetic admin/full-scope privilege data.
- [x] #4 Snapshot creation uses collision-safe IDs and no longer overwrites concurrent sync snapshots.
- [x] #5 Org/team maps ignore inactive memberships and inactive teams/orgs.
- [x] #6 Detail views avoid unbounded matrix materialization beyond the configured cap.
- [x] #7 Org trend history is scoped by org_id when org filters are used.
- [x] #8 Unused PrivilegeMapService role helper is removed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation plan: IMPLEMENTATION_PLAN_privilege_maps_review_hardening_2422.md

Verification completed:
- source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Privileges -q -> 40 passed
- source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/PrivilegeMaps tldw_Server_API/app/api/v1/endpoints/privileges.py -f json -o /tmp/bandit_privilege_maps_2422.json -> 0 findings
Known skips/blockers: none.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened PrivilegeMaps against the review findings: transaction writes now normalize placeholders for raw PostgreSQL transactions; effective permissions honor expiry and explicit denies; multi-user loading fails closed; sync snapshot IDs are UUID-based; org/team filters ignore inactive memberships/entities; detail generation caps materialization; org trends carry org_id; the unused role helper was removed. Added regression coverage across service, endpoint, snapshot, trend, and role-resolution tests.
<!-- SECTION:FINAL_SUMMARY:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reopened to address PR #2461 review comments after initial push: rebase on latest dev, harden placeholder conversion around literal question marks, treat NULL active flags as inactive, clean up endpoint test formatting, and respond to the DB_Management boundary comment.
PR #2461 follow-up verification completed after rebasing on latest dev:
- source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Privileges/test_privilege_service_sqlite.py tldw_Server_API/tests/Privileges/test_privilege_snapshot_store.py tldw_Server_API/tests/Privileges/test_privilege_trends.py tldw_Server_API/tests/Privileges/test_privilege_role_normalization.py -q -> 15 passed
- source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Privileges -q -> 42 passed
- source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/PrivilegeMaps tldw_Server_API/app/api/v1/endpoints/privileges.py -f json -o /tmp/bandit_privilege_maps_2422_pr2461.json -> 0 findings
- git diff --check -> clean
Resolved review comments: robust PostgreSQL placeholder conversion skips SQL literals/identifiers/comments, NULL active flags are treated as inactive for users/teams/orgs, endpoint test signature was formatted. DB_Management boundary comment was addressed with PR discussion because a full PrivilegeMaps persistence relocation is broader than this hardening PR.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
