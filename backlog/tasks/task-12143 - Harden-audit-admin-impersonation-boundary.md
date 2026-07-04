---
id: TASK-12143
title: Harden audit admin impersonation boundary
status: In Progress
created_date: 2026-07-04 07:05
labels:
- audit
- remediation
- authnz
- admin
- security
priority: high
references:
- AUDIT-2026-06-27-AUTH-001
- AUDIT-2026-06-27-AUTH-002
- AUDIT-2026-06-27-AUTH-003
documentation:
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/authnz-admin.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/remediation-backlog-draft.md
modified_files:
- tldw_Server_API/app/api/v1/API_Deps/auth_deps.py
- tldw_Server_API/app/api/v1/endpoints/admin/admin_impersonation.py
- tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py
- tldw_Server_API/app/core/AuthNZ/jwt_service.py
- tldw_Server_API/app/core/AuthNZ/principal_model.py
- tldw_Server_API/app/services/admin_audit_service.py
- tldw_Server_API/tests/Admin/test_admin_account_audit_events.py
- tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py
- tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py
- tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py
updated_date: 2026-07-04 17:00
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the high-priority AuthNZ audit slice for admin impersonation: issued token lifetime must match the advertised short TTL, impersonation issuance must have durable actor/subject audit attribution, and user lookup should avoid SQLite-only placeholder usage on PostgreSQL-backed deployments.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Impersonation token expiry matches the advertised short-lived TTL and tests decode the token to verify exp minus iat.
- [x] #2 Impersonation issuance emits a durable audit event with both admin actor and impersonated subject identifiers.
- [x] #3 Impersonation claims are preserved into downstream auth context or request state so later audit layers can distinguish actor and subject.
- [x] #4 Target user lookup uses backend-neutral AuthNZ helper/abstraction instead of SQLite placeholders on raw PostgreSQL connections.
- [x] #5 Focused regression coverage protects the remediated impersonation behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current impersonation endpoint, JWT service, AuthContext creation, and tests on latest dev.
2. Add failing focused tests for token TTL, durable audit attribution, downstream impersonation metadata, and backend-neutral lookup behavior as scope allows.
3. Implement minimal AuthNZ changes using existing project abstractions and avoiding raw SQL placeholder drift.
4. Run focused AuthNZ impersonation tests, Bandit on touched production files, and diff whitespace validation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-04: Refreshed worktree to latest origin/dev before validation. Current base HEAD/origin/dev/merge-base: fd5c152b065c408e4e8ee5f08da41589f21cb7f5. Expanded related suite passed on that base: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py tldw_Server_API/tests/Admin/test_admin_account_audit_events.py -q => 40 passed, 145 warnings.
2026-07-04: Final validation on latest origin/dev fd5c152b065c408e4e8ee5f08da41589f21cb7f5: related suite passed with 41 passed, 149 warnings; git diff --check passed. Bandit over touched production files wrote /tmp/bandit_admin_impersonation.json and exited nonzero due to pre-existing low-severity B106 token-type literal findings. Comparison against clean dev report /tmp/bandit_admin_impersonation_base_fd5c.json: base_total=13, patch_total=13, added=[], removed=[].
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused AuthNZ impersonation tests pass.
- [x] #2 Bandit over touched production files reports no new issues.
- [x] #3 git diff --check passes.
- [ ] #4 Backlog task contains latest-dev base, verification evidence, final summary, and PR link if opened.
<!-- DOD:END -->
