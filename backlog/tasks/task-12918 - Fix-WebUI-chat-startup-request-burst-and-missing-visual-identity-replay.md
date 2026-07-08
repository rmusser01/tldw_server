---
id: TASK-12918
title: Fix WebUI chat startup request burst and missing visual identity replay
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and fix the WebUI /chat startup request burst that can replay persona, character, and visual identity requests until Resource Governance returns 429. Evidence: stored/default character state on /chat triggers /persona/profiles, /persona/catalog, /characters?limit=1000&offset=0, /characters/3 404, and /visual-identities/bindings/resolve?actor_id=3 404. Scope: remove speculative background route prefetch, gate/dedupe first-run checks, and prevent missing visual identity bindings from being retried on remount churn.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Final summary: Fixed the WebUI /chat startup request burst by removing speculative route warm-prefetch from _app, gating the app-level first-run check until auth is resolved, single-flighting/caching successful first-run profile checks without caching failed responses, re-reading the local dismissal flag when consuming the cache, and caching missing visual identity 404s as no binding for the current JS session. Verification completed: focused Vitest suite passed (39 tests), frontend typecheck passed, Playwright e2e/chat-request-dedupe passed against live frontend/backend with RG enabled, seeded /chat probe against RG-enabled backend reported rateLimited=false and /persona/profiles=1, and backend log scan found no 429/rate_limited entries. Bandit: skipped because the touched implementation scope is frontend TypeScript/TSX only; no Python files were modified.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the WebUI /chat startup request burst by coalescing duplicate GET paths, gating first-run checks behind auth readiness, and preventing missing visual identity bindings from replaying on remount churn. Focused verification covered the changed frontend request paths; Bandit was skipped because the implementation scope is frontend TypeScript/TSX only.
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
