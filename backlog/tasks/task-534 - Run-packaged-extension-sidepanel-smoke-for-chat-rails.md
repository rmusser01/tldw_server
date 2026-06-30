---
id: TASK-534
title: Run packaged extension sidepanel smoke for chat rails
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-28 01:10'
labels:
  - chat
  - extension
  - ux
  - e2e
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build or load the packaged browser extension from the chat rails branch and smoke-test the sidepanel chat handoff into /chat. Keep scope to sidepanel chat, full-screen /chat handoff, route-only copy, role-play handoff, width/overflow fit, and absence of the removed CharacterControlRail.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Packaged extension sidepanel chat is opened at extension-like width and checked for horizontal overflow and clipped header/composer controls.
- [x] #2 Full-screen handoff opens /chat and route-only handoff copy remains visible or otherwise verified by source/runtime evidence.
- [x] #3 Role-play handoff still targets /chat and does not reintroduce CharacterControlRail.
- [x] #4 Screenshot or equivalent browser evidence is captured and the chat rails rebaseline doc is updated.
- [x] #5 Verification commands and any skips/blockers are recorded in the task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started from clean post-TASK-533 branch state. Scope is packaged extension sidepanel smoke only: sidepanel chat width/overflow, full-screen /chat handoff, route-only copy, role-play /chat handoff, and absence of CharacterControlRail. Unrelated untracked watchlist templates remain ignored.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Verification recorded: packaged extension sidepanel smoke passed 3 tests with /chat layout, /options.html#/chat handoff, no CharacterControlRail, and send/reply; focused SidepanelHeaderSimple and ControlRow role-play handoff Vitest passed 2 files / 6 tests; git diff --check and evidence JSON parse passed. Bandit skipped because this slice touched TypeScript E2E, Markdown, JSON, and PNG evidence only. During debugging, packaged active role-play state could not be reliably synthesized through extension storage after three variants, so role-play route intent is covered by the focused component contract rather than the packaged smoke.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Packaged extension sidepanel /chat smoke is now covered and documented. The E2E file verifies 390 px containment, explicit /chat sidepanel entry, route-only full-screen handoff to /options.html#/chat, no standalone CharacterControlRail, and send/reply. Role-play route intent remains verified by the focused ControlRow unit contract. Rebaseline docs and evidence now include the packaged sidepanel screenshot and verification commands.
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
