---
id: TASK-12139
title: Harden audit AuthNZ impersonation boundary
status: Done
created_date: 2026-07-04 00:44
labels:
- audit-remediation
- authnz
- security
priority: High
documentation:
- Docs/superpowers/reviews/2026-06-27-repo-audit/final-report.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/authnz-admin.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/remediation-backlog-draft.md
modified_files:
- Docs/superpowers/plans/2026-07-04-audit-auth-impersonation-remediation-plan.md
- tldw_Server_API/app/api/v1/endpoints/admin/admin_impersonation.py
- tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py
- tldw_Server_API/app/core/AuthNZ/jwt_service.py
- tldw_Server_API/app/core/AuthNZ/principal_model.py
- tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py
- tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py
- tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py
updated_date: 2026-07-04 00:54
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remediate comprehensive repository audit findings AUDIT-2026-06-27-AUTH-001, AUDIT-2026-06-27-AUTH-002, and AUDIT-2026-06-27-AUTH-003. Scope is the admin impersonation endpoint, impersonation token lifetime/claims, durable audit attribution, downstream auth context propagation, and backend-neutral database lookups.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Impersonation tokens use the explicit short-lived impersonation TTL and tests decode the JWT to verify the actual exp/iat interval.
- [x] #2 Impersonation issuance emits durable audit attribution for both the acting admin and impersonated subject using the existing admin audit patterns where feasible.
- [x] #3 Decoded impersonation tokens preserve actor metadata in downstream AuthContext/request auth state for later audit hooks.
- [x] #4 Impersonation user and role lookups use backend-neutral database helpers rather than raw PostgreSQL connections with SQLite placeholders.
- [x] #5 Focused AuthNZ impersonation tests pass and Bandit reports no new production findings on touched backend code.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created before repository edits per Backlog.md workflow. This branch starts from origin/dev f2d9be9864 in .worktrees/audit-auth-impersonation-2026-07-04.
Added the branch remediation plan at Docs/superpowers/plans/2026-07-04-audit-auth-impersonation-remediation-plan.md. Next step is RED regression coverage for token TTL, backend-neutral lookups, audit emission, and impersonation metadata propagation.
Implemented the AuthNZ impersonation remediation. The endpoint now uses DatabasePool.fetchone() for user/role lookups, passes the explicit 15 minute impersonation TTL into JWTService.create_access_token(), emits a durable AUTH_TOKEN_CREATED admin audit event with actor and target metadata, and decoded impersonation tokens populate AuthPrincipal/request.state impersonation fields. Verification: focused RED run first failed with 4 expected failures; after implementation, full touched AuthNZ test files passed with 27 passed. git diff --check passed. Bandit over touched production files wrote /tmp/bandit_audit_auth_impersonation.json and reported 13 LOW/MEDIUM B106 token-type string findings that pre-existed in AuthNZ token code; no endpoint findings and no high/medium severity findings were introduced.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Remediated audit AuthNZ impersonation findings AUDIT-2026-06-27-AUTH-001, AUDIT-2026-06-27-AUTH-002, and AUDIT-2026-06-27-AUTH-003. Impersonation tokens now use the explicit 15 minute TTL, endpoint user/role lookups use backend-neutral DatabasePool.fetchone(), issuance emits a durable AUTH_TOKEN_CREATED admin audit event with actor and target metadata, and decoded impersonation tokens populate AuthPrincipal plus request.state impersonation fields for downstream audit hooks. Verification: RED run failed first with the expected missing TTL/raw lookup/auth-context failures; final touched AuthNZ tests passed with 27 passed; git diff --check passed; Bandit wrote /tmp/bandit_audit_auth_impersonation.json with 13 LOW B106 token-type string findings that pre-existed in AuthNZ token code and no new endpoint/high/medium findings.
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
