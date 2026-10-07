---
id: TASK-13534
title: Accept canonical cookie owners in required token-scope guards
status: In Progress
labels:
- chat
- auth
- ux
- webui
- extension
dependencies:
- TASK-13458
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Native latest-dev first-use Chat saved and renamed successfully, but typing a title in Find conversations returned HTTP 401 Authentication required; the header share-link list failed the same way. The identical endpoints accept the disposable X-API-KEY. Trace identifies the shared require_token_scope missing-header branch, used by TokenScopeGuard, which rejects a valid single-user cookie even though get_request_user and canonical AuthNZ already support it. Reuse canonical principal resolution at this shared boundary and retain explicit-header precedence, session revocation, CSRF, endpoint claims, and existing JWT/API-key scopes and quotas.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A causal HTTP regression uses a real minted single-user session and actual conversation search/share-link guards without overriding authentication.
- [x] #2 The shared guard accepts only validated canonical cookie owners; missing, revoked, unusable, wrong-mode and explicitly invalid credentials still fail closed.
- [x] #3 JWT/API-key scope, schedule, constraints and quota checks retain their existing behavior.
- [x] #4 Native latest-dev first saved chat can be named, searched, reopened and continued through the same cookie session.
- [ ] #5 Relevant tests, formatting, lint, touched-scope Bandit, independent review and task evidence are recorded; broad Stage5 and unrun surfaces stay open.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Base HEAD52ea5951b4 includes origin/dev5ec8c7939f; verified 2026-10-07. Native isolated runtime /private/tmp/chat-ux-native-latest-20261007-rbbx71sf from the exact worktree source uses API19767, local deterministic provider19768 and fresh WebUI19769. No browser interception or localStorage seeding; normal cookie-session bootstrap. Evidence /private/tmp/chat-native-search-refresh-failure-20261007.png and browser warnings for /api/v1/chats/conversations and /api/v1/chat/conversations/{id}/share-links. The same native API request with the disposable API key returns200 and the correct single result. This is a bounded Stage5 discovery; prior initial unavailable-default-provider probe is environment configuration evidence, not yet a proven frontend defect. ADR required: no; ADR-018/019/044 govern canonical claim-based authorization and cookie governance, ADR-049/050 history ownership, ADR-059 tracking. Restores existing authentication rules without changing credential authority or adding protocols. Plan: Docs/Design/IMPLEMENTATION_PLAN_chat_cookie_scope_guard_13534.md.
Shared root repair: require_token_scope now delegates only its required missing-explicit-key branch to get_auth_principal. JWT/API-key enforcement, optional requests and explicit-header precedence retain their existing branches. A real minted-cookie HTTP regression failed on both actual search and share-link aliases before the fix. Verified matrix /private/tmp/chat-cookie-scope-13534-verified-matrix.log: 155 passed, 374 warnings, no skips, 261.03s. Updated two existing missing-credential assertions to canonical AuthNZ detail; 401 and WWW-Authenticate remain required. Owned Black regions, AST compilation and git diff --check pass. Ruff baseline/current 3/3, zero additions. Bandit baseline/current 319/335; the 16 added LOW B101 results are pytest test assertions (15 new HTTP assertions and one authentication challenge assertion), with no new production or non-assert findings and no new exclusions. Reports: /private/tmp/chat-cookie-scope-13534-static/. Two existing independent reviewers found no blocker. Same-cookie native ordinary Chat title search, reopen, second turn and full reload succeed; proof /private/tmp/chat-native-cookie-search-reopen-13534.png and -server.json. Native imported Mira card selected through the Chat picker; committed greeting plus two H1 turns survived full reload with five actual server rows, character4 and resume_eligible true; proof /private/tmp/chat-native-mira-roleplay-reload-20261007.png and -server.json. Native runtime used 52ea5951b4 plus this source delta. Fresh fetch now reports origin/dev2c5f19d032; integrate and run affected checks before closing this bounded task. Separate manager launch readiness retirement discovered and independently traced; preserve it as follow-up, not solved by cookie authentication.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
