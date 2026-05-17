---
id: TASK-407.2
title: Implement chat role-play visible state and terminology cleanup
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-05-17 07:53'
labels:
  - chat
  - ux
  - roleplay
  - stage-2
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-17-main-chat-role-play-preset-remediation-implementation-plan.md
parent_task_id: TASK-407
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stage 2 implementation for the main /chat role-play preset plan: add derived role-play state, preserve behavior template identity, expose truthful active chips, and clean up terminology.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Applied behavior templates retain identity instead of collapsing to anonymous custom prompt state.
- [ ] #2 Active chips distinguish identity, behavior, scene, generation style, and context summary.
- [ ] #3 Terminology changes are routed through locale files.
- [ ] #4 Focused Stage 2 tests and browser verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stage 2 started after Stage 1 commit 6bd234f1d passed spec and code-quality review. Scope remains limited to visible role-play state, behavior template identity, active chips, and terminology cleanup on main /chat.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
