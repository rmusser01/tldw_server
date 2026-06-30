---
id: TASK-158.2
title: Address PR 1412 Buddy visual review feedback
status: Done
assignee: []
created_date: '2026-05-09 06:16'
updated_date: '2026-05-09 06:18'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1412'
parent_task_id: TASK-158
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve actionable review findings on PR #1412 for the Persona Buddy visual workflow entry point. Current actionable feedback: use react-router internal navigation instead of a raw anchor for the Open Visuals action, hide the action when no active persona id is available, and add a guard-case test for missing persona context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Open Visuals uses react-router internal navigation semantics instead of a raw href anchor.
- [x] #2 Open Visuals is not rendered when the Buddy shell lacks a concrete persona id.
- [x] #3 BuddyShellHost tests cover both valid persona id routing and missing persona id guard behavior.
- [x] #4 Focused related frontend tests pass and review-fix verification is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED: bunx vitest run src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx failed because Open Visuals still rendered with href /persona?tab=visuals when active_persona_id was null and no selectedAssistant fallback existed.
GREEN: bunx vitest run src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx passed with 17 tests after switching Open Visuals to react-router Link and hiding it without a normalized persona id.
RELATED VERIFICATION: bunx vitest run src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx src/components/Common/PersonaBuddy/__tests__/personaVisualState.test.ts src/components/Common/PersonaBuddy/__tests__/SpriteFrameRenderer.test.tsx src/utils/__tests__/persona-garden-route.test.ts passed with 38 tests.
HYGIENE: git diff --check passed.
BANDIT: not applicable; touched code is frontend TypeScript plus Backlog metadata only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1412 review feedback by changing the Buddy Open Visuals action from a raw internal anchor to react-router Link navigation and rendering it only when a normalized active persona id is available. Added a missing-persona-id guard test alongside the happy-path route-preservation test, with focused Buddy and visual-pack frontend tests passing.
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
