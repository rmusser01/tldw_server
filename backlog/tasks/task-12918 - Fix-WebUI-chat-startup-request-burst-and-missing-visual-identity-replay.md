---
id: TASK-12918
title: Fix WebUI chat startup request burst and missing visual identity replay
status: Done
assignee: []
created_date: ''
updated_date: '2026-09-16 20:51'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and fix the WebUI /chat startup request burst that can replay persona, character, and visual identity requests until Resource Governance returns 429. Evidence: stored/default character state on /chat triggers /persona/profiles, /persona/catalog, /characters?limit=1000&offset=0, /characters/3 404, and /visual-identities/bindings/resolve?actor_id=3 404. Scope: remove speculative background route prefetch, gate/dedupe first-run checks, and prevent missing visual identity bindings from being retried on remount churn.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Blank-key single-user Home does not dispatch authenticated persona profile reads before credentials are available.
- [x] #2 Authenticated single-user and multi-user cookie sessions retain the intended first-run personalization check, including auth transitions.
- [x] #3 Fresh native pre-auth and post-auth controls, scoped tests and independent review are retained.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Final summary: Fixed the WebUI /chat startup request burst by removing speculative route warm-prefetch from _app, gating the app-level first-run check until auth is resolved, single-flighting/caching successful first-run profile checks without caching failed responses, re-reading the local dismissal flag when consuming the cache, and caching missing visual identity 404s as no binding for the current JS session. Verification completed: focused Vitest suite passed (39 tests), frontend typecheck passed, Playwright e2e/chat-request-dedupe passed against live frontend/backend with RG enabled, seeded /chat probe against RG-enabled backend reported rateLimited=false and /persona/profiles=1, and backend log scan found no 429/rate_limited entries. Bandit: skipped because the touched implementation scope is frontend TypeScript/TSX only; no Python files were modified.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Reopened during fresh-install UAT on 2026-09-16 as UAT159. Blank-key single-user Home mounts FirstRunGate after auth resolution but before authentication; its hook sends an avoidable persona/profiles read, returning429 from anonymous identity scope denial. Authenticated reads200. Preserve backend fail-closed governance and multi-user cookie sessions. Native receipt: output/playwright/cycle5-repair-verification-2026-09-16/native-image118-157/. Separate bounded frontend correction and fresh no-key/authenticated controls required.

Approved UAT159 bounded design (2026-09-16): reuse established _app isAuthenticated for FirstRunGate bypass; make hook work conditional via enabled, preserving existing cookie-aware auth resolution and cancellation guards. retry031_repair owns only _app.tsx, FirstRunGate.tsx, useFirstRunCheck.ts and their focused tests. First write actual App→Gate→Hook transport-boundary REDs for blank-key Home/nonbypass route, deferred bootstrap and stale auth-transition responses; controls for manual/runtime credentials, quickstart cookies, hosted tokenless cookies and first setup routes. No backend, auth reimplementation, runtime/browser, tracker or commits. Caller inventory explicitly leaves unrelated ChatFirstRunNudge behavior unchanged absent a proven native no-key failure. Private plan/report/evidence: .tmp/uat159-first-run-auth-20260916.

UAT159 implementation ready for independent review: exactly3production files (_app, FirstRunGate, useFirstRunCheck) plus3existing test files. Permanent actual App→Gate→Hook RED recorded7failures/69controls passing; profile transport only mocked on the real boundary. Final95tests/4suites PASS, no skips; includes manual/runtime auth, quickstart single-user cookie, hosted tokenless multi-user cookie, rejected cookie auth, bootstrap/setup bypass and logout/re-entry late-response controls. Five older gate-active fixture cases now supply explicit authenticated env credentials, preserving their route expectations. ESLint6paths0errors0warnings; full compiler90baseline/90current0added (exit2 baseline); diff clean. Bandit3TS/TSX AST parse errors means no TypeScript security assurance. Existing ChatFirstRunNudge still invokes the hook with default enabled=true; it is not mounted by Home and remains outside this approved native159 scope. Report/evidence/manifest: .tmp/uat159-first-run-auth-20260916/. No runtime/browser/staging/commit/tracker edits. Root independent review and fresh native blank-key/post-auth acceptance pending.

Final159 self-review strengthened the auth transition test to retain real apiSend and mock only request-core: it freshly reproduced1RED/13controls because the path-only coalescer reused the old pending profile GET after re-enable. Root approved this read using existing coalesce:false, within the same3production files and without editing apiSend. Final110tests/5suitesPASS; stable authenticated rerenders/auth refreshes issue1request, enabled remount1newrequest, disabled remount0. Tradeoff documented: concurrent Gate/Nudge consumers now each send their one profile read instead of coalescing; no polling/retry loop added. Final scoped lint0/0, compiler90baseline90current0added, diffclean. Frozen owned-manifest.json/report and review-snapshot retained under .tmp/uat159-first-run-auth-20260916; independent review/native acceptance remain pending.

UAT159 final root acceptance: independent110tests/5suites pass; separate review clear with matching6filehashes. Fresh SQLite Home beforemanualkey sends0persona requests, authenticated Prompts request returns200; final-source fresh PostgreSQL single origin also0persona reads beforecredentials. Cookie controls are real App/hook transport tests, not native fullmultiUAT. Source/lint compiler baseline unchanged. StrictMode may issue2bounded reads; continuous authenticated-owner changes and independent composer eligibility remain outside scoped claim.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the WebUI /chat startup request burst by coalescing duplicate GET paths, gating first-run checks behind auth readiness, and preventing missing visual identity bindings from replaying on remount churn. Focused verification covered the changed frontend request paths; Bandit was skipped because the implementation scope is frontend TypeScript/TSX only.

UAT159 follow-up: gate unused or unauthenticated first-run reads using existing shell authentication; disable/re-enable gets a fresh request without joining an older owner response. Native blank-key and authenticated controls pass; no backend policy change.
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
