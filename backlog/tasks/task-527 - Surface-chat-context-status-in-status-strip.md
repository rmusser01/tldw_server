---
id: TASK-527
title: Surface chat context status in status strip
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-27 19:28'
labels:
  - chat
  - ux
  - status-strip
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address /chat UX rebaseline F5 by showing active Web search/context source state in the cockpit status strip before send. Keep scope limited to /chat status feedback and existing context rail/open Search & Context workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When Web search or other context sources are active, the /chat status strip exposes a visible concise source-status chip instead of only a generic Context action.
- [x] #2 The ready status strip remains compact and does not reintroduce routine session/save noise.
- [x] #3 The existing Open Search & Context action remains available when context is active.
- [x] #4 Regression coverage proves active Web search/context summaries are visible and inactive context does not add chips.
- [x] #5 Focused /chat status strip/cockpit tests pass and verification/known skips are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Starting focused /chat F5 status-feedback slice. Planned touched scope: PlaygroundStatusStrip and status-strip/cockpit tests only unless existing wiring requires a parent prop adjustment.

Implemented compact active-context source chips in PlaygroundStatusStrip using the existing contextSummary prop. Chips render only while hasContext is true, show up to four summaries, and add a +N more overflow label to keep the strip compact.

Verification: RED focused run failed as expected because Chat status lacked Web search/source text. GREEN focused run passed 74 tests across PlaygroundStatusStrip.first-slice, Playground.cockpit-controls, and Playground.cockpit-shell. TypeScript compiler gate still fails only on known baseline CharacterListContent.design-system.test.tsx GalleryCardDensity error. git diff --check passed. Bandit not run because touched code is TS/TSX UI only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Surface active Web search/context source state in the /chat status strip without reintroducing routine session noise. Existing Open Search & Context action remains available, inactive context suppresses stale summaries, and focused cockpit/status-strip coverage verifies the behavior.
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
