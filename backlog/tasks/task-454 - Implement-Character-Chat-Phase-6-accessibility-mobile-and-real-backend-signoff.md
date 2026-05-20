---
id: TASK-454
title: Implement Character Chat Phase 6 accessibility mobile and real-backend signoff
status: Done
labels:
- chat
- characters
- role-play
- phase-6
- frontend
- e2e
- accessibility
priority: high
references:
- TASK-426
- TASK-428
- TASK-431
- TASK-438
- TASK-447
- TASK-449
- TASK-452
- Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md
documentation:
- Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md
- Docs/superpowers/plans/2026-05-20-character-chat-phase6-accessibility-mobile-backend-signoff-plan.md
modified_files:
- Docs/superpowers/plans/2026-05-20-character-chat-phase6-accessibility-mobile-backend-signoff-plan.md
- apps/packages/ui/src/components/Option/Playground/CharacterChatSessionsPanel.tsx
- apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx
- apps/packages/ui/src/components/Option/Playground/RolePlaySetupDrawer.tsx
- apps/packages/ui/src/components/Option/Playground/SavedRolePlaySetupsPanel.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/CharacterChatSessionsPanel.test.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.test.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/RolePlaySetupDrawer.test.tsx
- apps/packages/ui/src/services/tldw/TldwApiClient.ts
- apps/packages/ui/src/services/__tests__/tldw-api-client.character-persist.test.ts
- apps/packages/ui/src/services/__tests__/tldw-api-client.chat-debug.test.ts
- apps/tldw-frontend/e2e/workflows/journeys/character-chat-phase6.spec.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Phase 6 from the first-class Character Chat PRD: close the release-quality accessibility, mobile, and real-backend signoff gates for the /chat Character Chat workflow after Phase 5 sidepanel parity merged. Scope should focus on keyboard/focus semantics, screen-reader labels/live regions, narrow viewport behavior, and real backend E2E verification for create/select/send/resume without introducing a parallel character chat runtime.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Keyboard-only users can complete primary Character Chat setup and send/resume paths on /chat with logical focus order and no trapped focus.
- [x] #2 Screen-reader labels/live regions for mode switch, setup surface, sessions rail, selectors, composer, and destructive confirmations are covered by focused tests or browser accessibility snapshots.
- [x] #3 Desktop, tablet, and narrow/mobile viewports pass without horizontal overflow or hidden primary recovery actions for Character Chat mode, setup, sessions, and composer.
- [x] #4 Real backend E2E or documented real-backend browser verification covers character mode entry, character selection/restoration, send path readiness, and session resume using the actual backend API path rather than frontend-only mocks.
- [x] #5 Chat/character DB health release dependency is explicitly resolved, verified, or documented as a remaining release blocker with owner.
- [x] #6 Focused tests, real-browser evidence, diff hygiene, and Bandit applicability are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created Phase 6 implementation plan after verifying latest origin/dev includes PR #1888 and existing PRD phases 0-5. Current focus is release-quality accessibility, responsive/mobile signoff, and real-backend evidence rather than redoing already completed role-play remediation stages.

Implemented Phase 6 by adding screen-reader status/alert semantics to Character Chat sessions and setup loading/error states, adding saved setup list/listitem semantics, allowing the dense composer toolbar to wrap on non-desktop widths, adding desktop/tablet/mobile real-backend Playwright signoff for `/chat?mode=character`, and aligning monolithic `TldwApiClient` character stream/persist methods with the scoped domain client for real backend workspace-scoped calls.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verification recorded: focused Vitest suite passed 62 tests across Character Chat panels, composer toolbar, character mode contract, and TldwApiClient stream/persist regression coverage; real-backend Playwright passed 4/4 across character-chat-phase6.spec.ts and character-chat.spec.ts against http://127.0.0.1:8000; git diff --check passed. TypeScript checks were run for apps/packages/ui and apps/tldw-frontend and still fail on inherited baseline files, but the touched files and the previously in-scope useCharacterChatMode errors are absent from the fresh logs. Bandit was not run because no Python files were touched.
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
