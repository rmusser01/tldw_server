---
id: TASK-517
title: Add chat cockpit rail regression guards
status: In Progress
labels:
- chat
- frontend
- test
priority: high
documentation:
- Docs/superpowers/plans/2026-05-27-chat-rails-ux-rebaseline-implementation-plan.md
modified_files:
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx
- backlog/tasks/task-517 - Add-chat-cockpit-rail-regression-guards.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add focused regression coverage that proves the main /chat cockpit shell, context rail, runtime inspector, mobile rail panels, focus mode, and character rail remain wired on the origin/dev chat baseline before user-facing UX fixes proceed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A source-level regression guard proves the /chat page still imports and renders the cockpit shell, context rail, runtime inspector, mobile rail surface, focus mode affordance, and character rail.
- [ ] #2 Existing cockpit component and real-server e2e coverage is run or updated to verify rail visibility on desktop and mobile.
- [ ] #3 Screenshot artifacts for rail-enabled /chat are copied into the review asset directory for the refreshed UX audit.
- [x] #4 Verification results, skips, and any baseline failures are recorded in this task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 2 implementation notes - 2026-05-27:
- Added `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts` as a source-level guard using `import.meta.url`, `fileURLToPath`, and `path.resolve` to read `../Playground.tsx` and `../PlaygroundCockpitShell.tsx`.
- Guard asserts `Playground.tsx` still imports/uses and renders `PlaygroundCockpitShell`, `PlaygroundContextRail`, `PlaygroundRuntimeInspector`, and `CharacterControlRail`.
- Guard asserts `PlaygroundCockpitShell.tsx` still exposes `playground-cockpit-shell`, left/right/mobile rail test ids, `Enter focus chat`, and `Show cockpit panels`.
- Guard asserts `Playground.tsx` keeps `playgroundChatLayoutMode`, `"cockpit"`, and `"focus"`.
- Fixed the Task 2 existing rail-suite blocker in `Playground.cockpit-controls.test.tsx` by adding focused harness mocks for `@/hooks/useServerChatHistory`, `@/services/tldw-server`, and `@/services/chat-settings`; the controls tests now cover representative tracked and plain server chat history without requiring React Query integration.
- The cockpit server-readiness tests now clear the default selected character/assistant before rendering so they assert the cockpit health strip instead of character-chat readiness behavior.

Verification:
- `cd apps/packages/ui && bun install` completed earlier to restore missing local Vitest dependencies after the first guard run failed to resolve `vitest/config`.
- `cd apps/packages/ui && bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts`: PASS, 1 file, 3 tests.
- Initial existing rail-suite run failed because `Playground.cockpit-controls.test.tsx` rendered `Playground`, which now reaches `CharacterControlRail` / character sessions and `useServerChatHistory`; that test file had no `@/hooks/useServerChatHistory` mock and no `QueryClientProvider`, so React Query threw `No QueryClient set` in 11 tests.
- `cd apps/packages/ui && bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx`: PASS, 1 file, 16 tests after the harness fix.
- `cd apps/packages/ui && bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx src/components/Option/Playground/__tests__/CharacterControlRail.test.tsx`: PASS, 6 files, 99 tests.

DoD notes:
- AC #1 is complete for this Task 2 slice.
- AC #2 is partially complete: the existing cockpit component rail set now passes; real-server/screenshot evidence remains for Task 3.
- AC #3 remains open for Task 3 screenshot/evidence work.
- AC #4 is complete for this Task 2 slice.
- Bandit is not applicable for this TypeScript test and Backlog-only slice; no Python code was touched.
- The prior existing rail-suite blocker is resolved; no new production-code changes were made.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 2 added the source-level cockpit rail wiring guard and fixed the cockpit-controls harness blocker exposed by the restored rails. The new guard passes, the isolated controls file passes, and the existing rail component set now passes. Screenshot/evidence criteria remain open for Task 3.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
