---
id: TASK-12106
title: Add explicit single-user API key device persistence and relaunch coverage
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-12 00:50'
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
- [ ] #1 Manually configured remote single-user servers expose an explicit Remember on this device choice that defaults enabled for new setups.
- [ ] #2 Session-only choice survives hard reload but does not persist the API key across a full browser restart.
- [ ] #3 Remembered choice persists the origin-bound API key across a full browser restart until logout/reset.
- [ ] #4 Remote WebUI regression coverage includes save then hard reload and save then close/reopen the same profile.
- [ ] #5 Extension regression coverage includes save then close/reopen the same extension installation.
- [ ] #6 Same-origin cookie-session/runtime credentials are never copied into browser-readable manual key storage.
- [ ] #7 No browser password-manager behavior is required for API-key persistence.
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
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
