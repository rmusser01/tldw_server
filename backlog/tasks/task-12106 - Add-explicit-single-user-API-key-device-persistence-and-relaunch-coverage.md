---
id: TASK-12106
title: Add explicit single-user API key device persistence and relaunch coverage
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-12 15:37'
labels: []
dependencies:
  - TASK-12108
references:
  - TASK-12108
  - TASK-12030
  - TASK-12127
  - 'https://github.com/rmusser01/tldw_server/issues/2590'
documentation:
  - >-
    Docs/superpowers/specs/2026-07-10-single-user-api-key-device-persistence-design.md
  - >-
    docs/superpowers/plans/2026-07-10-remote-api-key-device-persistence-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add explicit device/session persistence for API keys only when users manually configure a remote single-user server in the WebUI or browser extension, with origin binding and browser/extension relaunch coverage. Same-origin runtime auth is handled without browser-readable keys by TASK-12108.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Manually configured remote single-user servers expose an explicit Remember on this device choice that defaults enabled for new setups.
- [x] #2 Session-only choice survives hard reload but does not persist the API key across a full browser restart.
- [x] #3 Remembered choice persists the origin-bound API key across a full browser restart until logout/reset.
- [x] #4 Remote WebUI regression coverage includes save then hard reload and save then close/reopen the same profile.
- [x] #5 Extension regression coverage includes save then close/reopen the same extension installation.
- [x] #6 Same-origin cookie-session/runtime credentials are never copied into browser-readable manual key storage.
- [x] #7 No browser password-manager behavior is required for API-key persistence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan: docs/superpowers/plans/2026-07-10-remote-api-key-device-persistence-implementation-plan.md. Execute after TASK-12108. Stages: credential metadata/storage policy; migration/hydration/save/clear; candidate-origin transition; onboarding/settings UX; WebUI and extension lifecycle verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
This task covers manually configured remote WebUI and browser-extension API-key persistence; same-origin runtime auth is TASK-12108 and never exposes or persists its key. Manual device persistence remains atomically in tldwConfig with complete manual/device/origin metadata so existing background and streaming readers continue working. Manual session keys use session storage. Candidate probes never inherit old auth; origin transitions are ordered; ambiguous or runtime-owned legacy values fail closed; persistence falls back device → session → memory. Design: Docs/superpowers/specs/2026-07-10-single-user-api-key-device-persistence-design.md

2026-07-11 Task 3 complete after feature-only rebase onto origin/dev d28c16bfa3. Candidate API-key probes use the submitted origin/key with credentials omitted; invalid URLs do not probe; old config remains unchanged on probe failure; onboarding/settings commit only after success; visible keys clear only across valid origin changes. Verification: focused Vitest 5 files/28 tests passed; broader auth/storage/bootstrap regression 10 files/134 tests passed; scoped ESLint 0 errors (48 pre-existing warnings).

2026-07-10 Task 2 complete: TldwApiClient migrates only confidently manual legacy keys, scrubs ambiguous keys, hydrates session credentials transiently, and provides explicit device/session/memory save plus fail-closed clear semantics. Logout and restart-onboarding reset clear manual credentials; updateConfig clears on auth-mode or normalized-origin changes and never writes hydrated session/env keys back to persistent storage. Device-write failures downgrade to session; session failures downgrade to memory. Persistent clear failures propagate. Verification: TDD RED observed; 46/46 focused tests; 18/18 extension bootstrap tests; repo-pinned ESLint 0 errors; git diff --check clean. TypeScript had 16 unrelated baseline errors. Spec and code-quality/security reviews approved. Bandit not applicable to TypeScript/TSX-only changes.

2026-07-11 Task 4 complete: onboarding and Settings now expose an accessible Remember on this device checkbox for manual single-user auth, default enabled for new entries and restored from saved scope. Unchecked saves to session scope. Device-to-session and session-to-memory fallbacks show truthful warnings. Cookie-session auth shows a server-held credential notice with no browser-readable key controls; changing to a different valid origin reveals manual controls. Verification: TDD RED observed; Task 4 Vitest 25/25; broader auth/storage/bootstrap regressions 134/134; scoped ESLint 0 errors; locale JSON valid; diff check clean. TypeScript has one unrelated baseline error in QuickIngestWizardModal.tsx:1813.

2026-07-11 Task 5 implementation and verification: added persistent-profile WebUI and unpacked-extension lifecycle suites using the real Settings UI, same profile, same unpacked extension path, and a local authenticated fixture. WebUI device save survives hard reload and full close/reopen; WebUI session save survives reload and clears on reopen. Extension device save survives close/reopen with the same extension ID; extension session storage clears on restart. During packaged testing, fixed two baseline blockers: Vite had compiled the locale glob map behind a runtime import.meta.glob guard that discarded it, and Settings initially applied hideHeader after mount, causing a deferred-route remount loop; focused RouteShell coverage now starts Settings with root chrome hidden. Review also caught that Plasmo's unspecified area defaults to chrome.storage.sync, which contradicted Remember on this device; all tldwConfig auth/request readers now use local storage, with tested legacy sync-to-local migration and removal. Verification: extension production build succeeded; packaged Settings smoke 1/1; extension lifecycle 2/2 in 9.8s; WebUI lifecycle 2/2; focused auth/request/route suites 96/96 plus final connection persistence suite 15/15; TypeScript noEmit completed cleanly; scoped ESLint 0 errors (existing warnings only); git diff check clean. Playwright 1.58.0, Chrome for Testing 145.0.7632.6. Known baseline build warnings remain duplicate imports, Rollup circular chunk notices, and stale Browserslist data. Bandit not applicable because Task 5 touches TypeScript/TSX, Playwright, Markdown, and task metadata only.

Verification correction: the final explicit TypeScript exit-code capture reports the same unrelated baseline error at apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx:1813 (overflowY string is not assignable to OverflowY); Task 5 touched files introduce no additional TypeScript errors. The existing apps/tldw-frontend/test-results artifact scan returned no known secret-marker matches; playwright-report was not generated.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed explicit manual remote single-user API-key persistence without relying on browser password managers. New setups default Remember on this device enabled; device credentials are origin-bound and stored in WebUI local storage or extension-local storage, while unchecked credentials use session storage and clear on full browser restart. Same-origin quickstart auth remains server-held behind HttpOnly sessions and never enters manual browser-readable storage. Added WebUI hard-reload/profile-reopen and extension same-installation/profile-reopen regression coverage for both device and session choices. Packaged extension verification also fixed Vite locale bootstrap and Settings root-shell remount blockers. Final evidence: Chrome extension production build; packaged Settings smoke 1/1; extension lifecycle 2/2; WebUI lifecycle 2/2; focused unit suites 96/96 plus migration/persistence suite 15/15; scoped ESLint 0 errors; diff and artifact secret-marker checks clean. Known unrelated baseline: QuickIngestWizardModal.tsx:1813 TypeScript overflowY type error. Bandit not applicable to this TypeScript/TSX/Markdown-only task.
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
