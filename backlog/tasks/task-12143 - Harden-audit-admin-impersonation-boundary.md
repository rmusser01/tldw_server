---
id: TASK-12143
title: Harden audit admin impersonation boundary
status: Done
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
updated_date: 2026-07-05 00:27
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
2026-07-04: Refreshed worktree to then-current origin/dev before validation. Current base HEAD/origin/dev/merge-base: fd5c152b065c408e4e8ee5f08da41589f21cb7f5. Expanded related suite passed on that base: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py tldw_Server_API/tests/Admin/test_admin_account_audit_events.py -q => 40 passed, 145 warnings.
2026-07-04: Pre-rebase validation on origin/dev fd5c152b065c408e4e8ee5f08da41589f21cb7f5: related suite passed with 41 passed, 149 warnings; git diff --check passed. Bandit over touched production files wrote /tmp/bandit_admin_impersonation.json and exited nonzero due to pre-existing low-severity B106 token-type literal findings. Comparison against clean dev report /tmp/bandit_admin_impersonation_base_fd5c.json: base_total=13, patch_total=13, added=[], removed=[].
2026-07-04: Draft PR opened against dev: https://github.com/rmusser01/tldw_server/pull/2622. PR remains draft because AI-authored PRs require a human-written Change summary before merge.
Review follow-up before the later rebase from origin/dev fd5c152b065c408e4e8ee5f08da41589f21cb7f5: addressed PR #2622 inline review comments by accepting both dict-backed and object-backed AuthNZ user records in create_impersonation_token, and by rejecting JWTs with impersonation=true unless impersonated_by is present and integer-coercible. Added regressions test_success_accepts_user_model_object and test_verify_jwt_rejects_impersonation_without_valid_admin_actor. Also annotated token_type category literals with nosec B106 so the touched-scope Bandit gate reports real findings only. Verification: targeted regressions passed (3 passed); focused AuthNZ files passed (14 passed); Bandit over admin_impersonation.py and User_DB_Handling.py exited 0 with 0 findings in /tmp/bandit_admin_impersonation_review_latest_dev.json; git diff --check passed; branch merge-base matched origin/dev fd5c152b065c408e4e8ee5f08da41589f21cb7f5.
Post-rebase validation on current origin/dev 4c1ca5d8358bff2a5a7fb5c75d60d1bd6728e702: rebased codex/audit-admin-impersonation-2026-07-04 so merge-base equals current origin/dev. Fresh verification after rebase: focused admin impersonation/AuthNZ tests passed (14 passed, 66 warnings) using current paths tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py and tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py; git diff --check HEAD~1..HEAD passed. Bandit over review-touched production files tldw_Server_API/app/api/v1/endpoints/admin/admin_impersonation.py and tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py scanned 1471 LOC and reported 0 findings in /tmp/bandit_admin_impersonation_review_touched_rebased_dev.json. Broader branch-touched production Bandit scan also covered auth_deps.py, jwt_service.py, principal_model.py, and admin_audit_service.py; it reported only LOW B106 token-type literal findings in unchanged jwt_service.py lines, not in this branch's jwt_service diff.
2026-07-04 current-dev refresh: rebased `codex/audit-admin-impersonation-2026-07-04` onto `origin/dev` `09d9ec901e1d4548f7924f1c6bcefa963fadd9bd`; merge-base matches `origin/dev`. Validation: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py -q` passed with 14 tests; `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/admin/admin_impersonation.py tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py -f json -o /tmp/bandit_admin_impersonation_origin_dev_09d9ec.json` reported 0 findings over 1471 LOC; `git diff --check HEAD~1..HEAD` passed.
2026-07-04 latest-dev refresh: rebased and validated PR #2622 on origin/dev 6b727b221e55646eba663a03571e38302f7fafc2. Tested head 2f46788d2511. Verification: python -m pytest tldw_Server_API/tests/AuthNZ/test_admin_impersonation.py tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_jwt_membership.py -q => 14 passed, 66 warnings; Bandit over admin_impersonation.py and User_DB_Handling.py => 0 findings over 1471 LOC; git diff --check HEAD~1..HEAD => clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened admin impersonation and JWT membership handling. Final refresh validated against origin/dev 6b727b221e55646eba663a03571e38302f7fafc2 with focused AuthNZ tests passing, Bandit clean on touched production scope, and whitespace check clean.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused AuthNZ impersonation tests pass.
- [x] #2 Bandit over touched production files reports no new issues.
- [x] #3 git diff --check passes.
- [x] #4 Backlog task contains latest-dev base, verification evidence, final summary, and PR link if opened.
<!-- DOD:END -->
