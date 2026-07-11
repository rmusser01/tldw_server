---
id: TASK-12106
title: Add explicit single-user API key device persistence and relaunch coverage
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-07-10 22:44
labels: []
dependencies:
- TASK-12108
references:
- TASK-12108
- TASK-12030
- TASK-12127
- https://github.com/rmusser01/tldw_server/issues/2590
documentation:
- Docs/superpowers/specs/2026-07-10-single-user-api-key-device-persistence-design.md
- docs/superpowers/plans/2026-07-10-remote-api-key-device-persistence-implementation-plan.md
priority: high
modified_files:
- apps/packages/ui/src/services/tldw/TldwApiClient.ts
- apps/packages/ui/src/services/tldw/TldwAuth.ts
- apps/packages/ui/src/store/connection.tsx
- apps/packages/ui/src/services/__tests__/tldw-api-client.connection-sync.test.ts
- apps/packages/ui/src/services/__tests__/tldw-api-client.quickstart-auth.test.ts
- apps/packages/ui/src/services/__tests__/tldw-auth.refresh-rotation.test.ts
- apps/packages/ui/src/store/__tests__/connection.test.ts
- Docs/superpowers/plans/2026-07-10-remote-api-key-device-persistence-implementation-plan.md
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
This task covers manually configured remote WebUI and browser-extension API-key persistence; same-origin runtime auth is TASK-12108 and never exposes or persists its key. Manual device persistence remains atomically in tldwConfig with complete manual/device/origin metadata so existing background and streaming readers continue working. Manual session keys use session storage. Candidate probes never inherit old auth; origin transitions are ordered; ambiguous or runtime-owned legacy values fail closed; persistence falls back device → session → memory. Design: Docs/superpowers/specs/2026-07-10-single-user-api-key-device-persistence-design.md

2026-07-10 Task 2 complete: TldwApiClient now migrates only confidently manual legacy keys, scrubs ambiguous keys, hydrates session credentials transiently, and provides explicit device/session/memory save plus fail-closed clear semantics. Explicit logout and restart-onboarding reset clear manual credentials; updateConfig clears on auth-mode or normalized-origin changes and never writes hydrated session/env keys back to persistent storage. Device-write failures downgrade to session; session failures downgrade to memory and remove misleading persisted session metadata. Persistent clear failures propagate rather than falsely reactivating stale credentials. Verification: TDD RED observed; 46/46 focused tests; 18/18 extension bootstrap tests; repo-pinned ESLint 0 errors; git diff --check clean. TypeScript has 16 unrelated baseline errors outside Task 2 files. Spec and code-quality/security reviews approved. Bandit not applicable to TypeScript/TSX-only changes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
