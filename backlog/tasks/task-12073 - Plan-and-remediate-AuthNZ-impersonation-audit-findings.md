---
id: TASK-12073
title: Plan and remediate AuthNZ impersonation audit findings
status: In Progress
created_date: 2026-06-30 05:45
labels:
- audit
- remediation
- authnz
- impersonation
- wave-1
priority: high
references:
- AUDIT-2026-06-27-AUTH-001
- AUDIT-2026-06-27-AUTH-002
- AUDIT-2026-06-27-AUTH-003
documentation:
- https://github.com/rmusser01/tldw_server/pull/2556
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/authnz-admin.md
modified_files:
- Docs/superpowers/plans/2026-06-29-authnz-impersonation-boundary-remediation.md
- tldw_Server_API/app/api/v1/endpoints/admin/admin_impersonation.py
- tldw_Server_API/app/core/AuthNZ/jwt_service.py
- tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py
- tldw_Server_API/app/core/AuthNZ/principal_model.py
- tldw_Server_API/tests/AuthNZ/
updated_date: 2026-06-30 05:49
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track the current-dev remediation of the AuthNZ impersonation audit findings: short impersonation token lifetime, actor-plus-subject attribution propagation, durable audit evidence, and backend-neutral user/role lookups.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Implementation plan is written before production code changes.
- [ ] #2 Impersonation token lifetime matches the documented short TTL.
- [ ] #3 Actor and subject survive from token issuance into downstream request context.
- [ ] #4 Durable audit events capture impersonation issuance and impersonated actions.
- [ ] #5 PostgreSQL and SQLite lookup paths use backend-neutral query helpers.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation plan drafted at Docs/superpowers/plans/2026-06-29-authnz-impersonation-boundary-remediation.md after Wave 0 reconfirmation showed AUTH-001, AUTH-002, and AUTH-003 remain open on current origin/dev. The plan is test-first and splits remediation into JWT TTL support, AuthContext impersonation propagation, repository-backed endpoint lookups with mandatory issuance audit, and verification/Bandit closure evidence.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Token decode tests assert exp minus iat.
- [ ] #2 SQLite and PostgreSQL or fake-asyncpg impersonation tests cover user and role lookup.
- [ ] #3 Audit attribution tests assert actor and subject fields.
- [ ] #4 Focused AuthNZ tests pass.
- [ ] #5 Bandit runs over touched AuthNZ production paths.
- [ ] #6 Findings AUTH-001, AUTH-002, and AUTH-003 are closed or have residual risk recorded.
<!-- DOD:END -->
