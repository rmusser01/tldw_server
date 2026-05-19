---
id: TASK-446
title: Fix Character Chat readiness for unconfigured model catalogs
status: Done
ordinal: 446
modified_files:
- apps/packages/ui/src/utils/chat-model-availability.ts
- apps/packages/ui/src/utils/__tests__/chat-model-availability.test.ts
- apps/packages/ui/src/components/Option/Playground/Playground.tsx
- apps/packages/ui/src/components/Option/Playground/PlaygroundStatusStrip.tsx
- apps/packages/ui/src/components/Option/Playground/playground-composition-preview.ts
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundStatusStrip.first-slice.test.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/playground-composition-preview.test.ts
- apps/packages/ui/src/components/Option/Playground/__tests__/CharacterChatReadinessPanel.test.tsx
- apps/packages/ui/src/services/tldw/TldwModels.ts
- apps/packages/ui/src/services/tldw/__tests__/TldwModels.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up after PR #1866 merged. Real-backend smoke on /chat?mode=character showed a selected catalog-only OpenAI gpt-4o route could still make Character Chat surfaces appear ready. This task carries the dev-based follow-up fix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Character Chat readiness does not report Ready when the selected chat model is catalog-only or explicitly unconfigured in backend model metadata.
- [x] #2 Readiness recovery copy points the user toward model/provider settings instead of allowing an unusable role-play session.
- [x] #3 Focused unit/integration tests cover catalog-only/unconfigured model metadata, stale unflagged descriptors, status propagation, and cache invalidation.
- [x] #4 Real-backend smoke verifies /chat?mode=character no longer shows contradictory no-provider/model-unavailable and Ready states.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Verification: focused Vitest suite passed with 9 files and 114 tests; git diff --check passed; real-backend browser smoke showed tldw:gpt-4o blocked with Model unavailable in the status strip and Error in the runtime rail. Full TypeScript still fails on existing repo-wide baseline debt, with no touched-file errors after fixing the local cockpit test typing issue. Bandit skipped because only frontend TypeScript/tests and Backlog docs were touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed Character Chat readiness so catalog-only or explicitly unconfigured backend model descriptors cannot make a role-play session look ready. The /chat cockpit now force-refreshes readiness-critical model metadata, fails closed on stale unflagged descriptors for Character Chat, invalidates the persisted model-cache schema, and propagates model-unavailable state into the readiness panel, composition preview, runtime rail, and bottom status strip.
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
