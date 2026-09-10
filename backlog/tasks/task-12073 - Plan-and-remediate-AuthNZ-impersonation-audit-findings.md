---
id: TASK-12073
title: Plan and remediate AuthNZ impersonation audit findings
status: Done
assignee: []
created_date: '2026-06-30 05:45'
updated_date: '2026-06-30 22:54'
labels:
  - audit
  - remediation
  - authnz
  - impersonation
  - wave-1
dependencies: []
references:
  - AUDIT-2026-06-27-AUTH-001
  - AUDIT-2026-06-27-AUTH-002
  - AUDIT-2026-06-27-AUTH-003
documentation:
  - 'https://github.com/rmusser01/tldw_server/pull/2556'
  - Docs/superpowers/reviews/2026-06-27-repo-audit/domains/authnz-admin.md
modified_files:
  - Docs/superpowers/plans/2026-06-29-authnz-impersonation-boundary-remediation.md
  - tldw_Server_API/app/api/v1/endpoints/admin/admin_impersonation.py
  - tldw_Server_API/app/core/AuthNZ/jwt_service.py
  - tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py
  - tldw_Server_API/app/core/AuthNZ/principal_model.py
  - tldw_Server_API/app/services/admin_audit_service.py
  - tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py
  - tldw_Server_API/tests/AuthNZ/unit/test_impersonation_auth_context.py
  - tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track the current-dev remediation of the AuthNZ impersonation audit findings: short impersonation token lifetime, actor-plus-subject attribution propagation, durable audit evidence, and backend-neutral user/role lookups.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan is written before production code changes.
- [x] #2 Impersonation token lifetime matches the documented short TTL.
- [x] #3 Actor and subject survive from token issuance into downstream request context.
- [x] #4 Durable audit events capture impersonation issuance and impersonated actions.
- [x] #5 PostgreSQL and SQLite lookup paths use backend-neutral query helpers.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation plan drafted at Docs/superpowers/plans/2026-06-29-authnz-impersonation-boundary-remediation.md after Wave 0 reconfirmation showed AUTH-001, AUTH-002, and AUTH-003 remain open on current origin/dev. The plan is test-first and splits remediation into JWT TTL support, AuthContext impersonation propagation, repository-backed endpoint lookups with mandatory issuance audit, and verification/Bandit closure evidence.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Implemented and reviewed in isolated worktree .worktrees/audit-wave0-reconfirm-2026-06-29 on branch codex/audit-wave0-reconfirm-2026-06-29.

Implementation evidence:
- AUTH-001: JWTService.create_access_token now accepts a positive expires_delta override and create_impersonation_access_token delegates with impersonation=True, impersonated_by, and a 15-minute expires_delta. Tests decode tokens and assert exp - iat == 900 seconds.
- AUTH-002: verify_jwt_and_fetch_user now strictly validates impersonation/impersonated_by claims and propagates impersonation plus impersonated_by_user_id into request.state and request.state.auth.principal. Malformed claim values fail closed. Admin impersonation issuance now writes mandatory durable audit before returning success.
- AUTH-003: admin_impersonation uses AuthnzUsersRepo and AuthnzRbacRepo instead of raw endpoint SQL/pool.acquire lookups. RBAC lookup errors and malformed non-empty role rows fail closed before token creation/audit; empty role rows fall back to the legacy row role/user.
- Local integrated review hardening: impersonation actor IDs must now be positive integers at issuance and decode time, and signed tokens fail closed if actor attribution is present without impersonation=True.

Verification evidence:
- PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py tldw_Server_API/tests/AuthNZ/unit/test_impersonation_auth_context.py -q -> 49 passed, 207 warnings.
- PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tldw_Server_API/tests/AuthNZ tldw_Server_API/tests/AuthNZ_Unit -q -> 1306 passed, 175 skipped, 10846 warnings in 5574.39s.
- python -m bandit tldw_Server_API/app/api/v1/endpoints/admin/admin_impersonation.py tldw_Server_API/app/core/AuthNZ/jwt_service.py tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py tldw_Server_API/app/core/AuthNZ/principal_model.py tldw_Server_API/app/services/admin_audit_service.py -f json -o /tmp/bandit_authnz_impersonation.json -> exit 1 with 13 LOW B106 findings, all token-type string literal false positives such as access/refresh/api_key/magic_link/admin_reauth/password_reset/email_verification/service; no high/medium findings and no new security issue in the AuthNZ impersonation changes.
- git diff --check -> clean.
- Endpoint search for get_db_pool, pool.acquire, SELECT users/user_roles, and generic create_access_token in admin_impersonation.py -> no matches.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Remediated AUTH-001, AUTH-002, and AUTH-003 for the AuthNZ admin impersonation boundary. Impersonation tokens are now minted through a dedicated helper with a true 15-minute TTL, impersonation actor metadata is strictly validated and carried into AuthContext, issuance is durably audited with a mandatory AUTH_TOKEN_CREATED event before success is returned, and the endpoint no longer performs raw SQL/user-role lookups. A final local review also tightened impersonation claim invariants so actor attribution cannot appear without impersonation=True and actor IDs must be positive integers.

Closed findings:
- AUTH-001 closed: token decode tests assert the 900-second impersonation TTL.
- AUTH-002 closed for issuance and downstream request context: actor/subject metadata reaches request.state.auth.principal and issuance audit captures actor_id, target_user_id, TTL, resource/action, and required context.
- AUTH-003 closed: endpoint lookups use AuthnzUsersRepo/AuthnzRbacRepo and fail closed on RBAC errors or malformed role rows.

Residual risk recorded: step-up reauthentication for starting impersonation remains out of scope for this code slice because the existing endpoint has no request body/reauth contract. A future API-compatible design should add explicit step-up verification and, ideally, include impersonation token jti in the mandatory audit event for stronger correlation.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Token decode tests assert exp minus iat.
- [x] #2 SQLite and PostgreSQL or fake-asyncpg impersonation tests cover user and role lookup.
- [x] #3 Audit attribution tests assert actor and subject fields.
- [x] #4 Focused AuthNZ tests pass.
- [x] #5 Bandit runs over touched AuthNZ production paths.
- [x] #6 Findings AUTH-001, AUTH-002, and AUTH-003 are closed or have residual risk recorded.
<!-- DOD:END -->
