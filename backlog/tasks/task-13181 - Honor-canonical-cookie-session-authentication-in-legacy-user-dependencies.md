---
id: TASK-13181
title: Honor canonical cookie-session authentication in legacy user dependencies
status: Done
assignee:
- '@codex'
created_date: 2026-09-05 15:53
updated_date: 2026-09-05 18:55
labels: []
dependencies: []
references:
- Docs/Reviews/MIGU_BUDDY_UAT_2026_09_05.md
priority: high
modified_files:
- tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py
- tldw_Server_API/tests/AuthNZ/integration/test_single_user_cookie_session.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Real quickstart Migu UAT mints a single-user cookie session (200) and loads users/me/profile (200), but notifications and ingestion capability requests return401 despite the browser sending its cookie. User_DB_Handling.get_request_user only accepts existing request principal or header credentials; it does not call the cookie-aware canonical resolver. Persona and DB dependencies use this path, so quickstart Persona remains unavailable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Valid single-user cookie sessions can access authorized Persona, ingestion capabilities, and notifications through their actual dependency chains without API-key headers.
- [x] #2 Expired, revoked, absent, or invalid sessions remain rejected and existing permission, user-isolation, CSRF, and origin checks remain effective.
- [x] #3 Real quickstart Migu builder and Buddy image UAT passes with cookie authentication; browser stream UAT uses an explicitly allowed origin.
- [x] #4 Quickstart connection readiness accepts an active same-origin cookie session and still rejects invalidated or cross-origin cookie metadata.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce cookie-only rejection through actual Persona, notifications, and ingestion dependencies using the existing real-session HTTP fixture.
2. Extend the legacy User adapter to invoke canonical principal resolution when header authentication is unavailable, retaining header precedence and all existing validation.
3. Verify absent, invalid, expired, revoked sessions, header precedence, CSRF, and per-user ownership with targeted tests; run touched-scope Bandit and review.
4. Root agent verifies real quickstart and Buddy browser UAT.
ADR required: no new ADR.
ADR path: Docs/ADR/018-resource-governance-endpoint-policy-and-route-map.md; Docs/ADR/019-security-request-edge-middleware.md.
Reason: bounded adapter repair implementing existing Principal Governance and Docs/superpowers/specs/2026-07-10-single-user-http-only-session-design.md; no new auth policy, identity owner, or middleware boundary.
Real browser follow-up: connection.tsx also checks only API keys before health probing. Reuse the existing active cookie-session config predicate for auth readiness, preserving invalidation and origin restrictions. Add a connection-store regression before changing readiness; this implements the existing July10 cookie contract, no new ADR.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Quickstart cookie authentication now reaches legacy dependencies and passes connection readiness. Real Persona builder, Buddy images, notifications, ingestion capabilities and browser WebSocket work without API-key headers. Invalidated/cross-origin metadata and revoked/invalid sessions remain rejected.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Cookie-only requests now reach Persona profiles, notifications, and ingestion capabilities through their real dependencies. get_request_user delegates headerless requests to the existing canonical principal resolver and returns its validated cached User; explicit Authorization/X-API-KEY header presence still prevents cookie fallback. No session validator, permission guard, CSRF middleware, origin policy, or storage owner changed.

Verification (server2 virtualenv; PYTHONPATH includes this worktree, packages/tldw_profile_core/src, apps/mcp-unified/src; TLDW_TEST_NO_DOCKER=1):
- RED: pytest tldw_Server_API/tests/AuthNZ/integration/test_single_user_cookie_session.py -k cookie_authenticates_legacy_user_routes -q --tb=short reproduced Persona and notifications 401 after real cookie mint. The minimal test app omitted the optional ingestion router; its regression now mounts the real router/dependencies.
- GREEN: complete test_single_user_cookie_session.py module: 32 passed (122 existing runtime/deprecation warnings), 209.96s. Includes actual route access, absent/invalid/expired/revoked/wrong-type persisted sessions, explicit invalid/blank header precedence, multi-user cookie rejection, Persona CSRF denial/success, canonical DB owner, and foreign-owner 404.
- Compatibility: test_authnz_invariants.py, test_user_db_handling_api_keys.py, test_user_db_handling_jwt_membership.py, test_single_user_session.py, test_websocket_session_auth.py, test_auth_principal_resolver.py: 78 passed (128 existing runtime/deprecation warnings), 156.51s, exit 0.
- Ruff lint on both touched Python files and Ruff formatting on the test module pass; git diff --check passes. Whole-file formatting of User_DB_Handling.py already fails at HEAD; retained existing style without unrelated churn.
- Bandit touched production/test scope: zero new production findings versus HEAD. Same three existing low B106 findings at 915/1023/1355; remaining full-scan findings are test assertions (B101). Report /private/tmp/task13181-final-bandit.json excludes B101; baseline /private/tmp/task13181-base-bandit.json.

No new ADR required. Existing Docs/ADR/018-resource-governance-endpoint-policy-and-route-map.md and Docs/ADR/019-security-request-edge-middleware.md apply, together with Docs/Product/Completed/AuthNZ-Refactor/Principal-Governance-PRD.md and Docs/superpowers/specs/2026-07-10-single-user-http-only-session-design.md. This repairs the existing adapter contract and adds no new authentication policy.

AC3 remains pending real quickstart Migu/Buddy and browser stream UAT by the parent agent. No full suite, commit, push, PR, or real runtime/browser operation performed by this subtask.
Real quickstart UAT now passes after both legacy-cookie auth and TASK13185 owner-admission fixes. Existing connection readiness also required an API key despite active cookie metadata; it now uses exact-origin cookie-session configuration plus invalidation state before URL normalization and authenticated health probing. Connection-store red-to-green matrix covers valid, invalidated and foreign-origin cookie metadata;36 store tests plus7 persona-state tests pass. Browser cookie-only users/me, Persona profiles, notifications, ingestion capabilities and docs-info all200; builder and Buddy decode96x96 protected blob images. Real cookie WebSocket receives the synthetic plan using explicitly allowed18385 origin. Evidence: Docs/Reviews/assets/migu-buddy-followups-2026-09-05/quickstart-builder.json and screenshot. TypeScript lint0errors; repository frontend typecheck retains80 unrelated diagnostics across the same6 baseline files. New public auth policy not introduced; ADR018/019 and July10 cookie design remain governing.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->