---
id: TASK-13192
title: Fix Resource Governor ingress identity for single-user session cookies
status: Done
created_date: 2026-09-05 19:13
assignee:
- '@codex'
documentation:
- Docs/ADR/044-cookie-session-governance-owner-preflight.md
modified_files:
- tldw_Server_API/app/core/Resource_Governance/middleware_simple.py
- tldw_Server_API/app/core/Resource_Governance/README.md
- tldw_Server_API/tests/Resource_Governance/test_middleware_cookie_owner.py
- Docs/ADR/044-cookie-session-governance-owner-preflight.md
- Docs/ADR/README.md
updated_date: 2026-09-05 19:13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Real quickstart Persona browsing is permanently rejected by request governance because cookie-only requests fall back to an IP scope excluded from the existing character-chat policy. Preserve configured request limits while accounting for the supported authentication transport.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cookie-only single-user requests reach their existing request policy bucket instead of permanent unmatched-scope denial.
- [x] #2 Explicit authenticated identity and API-key/Bearer header precedence remains unchanged; opaque cookie values never appear in entity keys.
- [x] #3 Governance preflight uses the canonical session resolver and shared request cache; multiple cookies for one owner share request quota, invalid sessions retain authentication errors, and exhaustion or missing policies still return 429.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: Docs/ADR/044-cookie-session-governance-owner-preflight.md (extends ADR018/019).
1. Reproduce cookie-only unmatched scope using real MemoryResourceGovernor and existing character-chat policy.
2. Before governed cookie-only ingress entity derivation, call the canonical AuthNZ principal resolver so its shared request-state cache provides the validated owner user ID. Preserve explicit-header precedence, reuse cached principal, return canonical auth errors and retain original rate policies.
3. Verify multiple sessions share the owner quota, invalid sessions fail authentication, and downstream auth reuses state; run focused entity/middleware regressions, touched-scope lint and Bandit.
4. Parent verifies isolated real browser after coordinated restart; document evidence and remaining UAT state.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Cookie-only Persona ingress previously derived IP before route authentication; character_chat.default excludes IP, so no bucket existed and every request returned 429/retry_after1. Added narrowly scoped canonical AuthNZ preflight for governed policies with user/api_key scopes and no global/ip/entity fallback. AuthContext and owner ID are shared with endpoint authentication; policy limits and secrets handling are unchanged. Anonymous-capable policies retain stale-cookie health and idempotent logout behavior.
ADR044 records the boundary and alternatives; RG README documents the exact criterion. Changed middleware_simple.py, added test_middleware_cookie_owner.py, and created Docs/ADR/044-cookie-session-governance-owner-preflight.md.
Evidence: initial regression2failed (valid cookie429 and invalidcookie429); anonymous-policy regression3failed then passed after narrowing. Final15new tests pass; existing middleware+entity19passed. Tests exercise real MemoryResourceGovernor/default YAML and canonical principal resolver, with session validation stubbed. Owner quota exhausts after60 across2cookies, invalid session401, explicit-header precedence, endpoint cache reuse, no permissive resolver failure, missing policy429, anonymous health/logout admission. Ruff check/test formatting, Bandit touched production code, and git diff --check pass. Broader real SQLite cookie integration suite and parent browser UAT pending; parent owns final task closure and commit.
Broader targeted verification finished: 58 passed, including 32 real SQLite single-user-cookie integration cases (mint, logout idempotence, revoked/expired sessions, CSRF, principal adapters). This run started before the final anonymous-policy narrowing; the final narrowed production snapshot then passed the dedicated 15 regressions and 19 existing middleware/entity tests. Added ADR044 to Docs/ADR/README.md. Parent browser UAT remains the final live verification.
Parent real-browser UAT: after restarting the actual FastAPI application with this middleware, Persona profile reads changed from persistent429 character_chat.default to200. Quickstart reports Connected; protected builder and Buddy images decode96x96; notifications/ingestion/users/me also200. Evidence: Docs/Reviews/assets/migu-buddy-followups-2026-09-05/quickstart-builder.json. No quota disablement, policy expansion, API-key injection or cookie-value logging. Independent frontend review caught missing per-turn stream correlation and stale connect completion guards; those are being repaired under TASK13180.
Originally allocated TASK-13185 during this work; renumbered to TASK-13192 before rebase because dev independently allocated 13185 to the llama.cpp snapshot plan. Original creation: 2026-09-05 18:30. Scope and verification unchanged.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Restored cookie-only Persona request admission using canonical cached owner authentication while preserving per-user quota ownership, explicit-header precedence, anonymous-capable policies and fail-closed missing-policy behavior. ADR044 documents scope; focused regression, cookie integration, Ruff and Bandit checks pass. Live browser verification and commit coordinated by parent.
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
