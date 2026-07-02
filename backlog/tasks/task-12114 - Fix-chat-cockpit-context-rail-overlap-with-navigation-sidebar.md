---
id: TASK-12114
title: Fix chat cockpit context rail overlap with navigation sidebar
status: Done
modified_files:
- apps/packages/ui/src/components/Layouts/chat-rail-positioning.ts
- apps/packages/ui/src/components/Layouts/__tests__/chat-rail-positioning-contract.guard.test.ts
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-rail-restore.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The collapsed left context rail restore tab in the chat Playground is fixed at viewport left and can overlay the app-wide chat navigation side rail. Add a regression test and adjust the positioning contract so the tab stays attached to the chat content edge instead of covering navigation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Changed the collapsed chat cockpit context restore tab from viewport-fixed `left-0` positioning to absolute positioning within the chat cockpit shell. This keeps the tab attached to the chat content edge instead of overlaying the app-wide navigation sidebar. Updated the positioning contract and cockpit restore tests. Verification: `bunx vitest run src/components/Layouts/__tests__/chat-rail-positioning-contract.guard.test.ts src/components/Option/Playground/__tests__/Playground.cockpit-rail-restore.test.tsx` from `apps/packages/ui` passed (5 tests). Bandit skipped because the touched code is TypeScript UI/test-only.
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
