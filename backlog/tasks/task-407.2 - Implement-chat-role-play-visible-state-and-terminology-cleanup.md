---
id: TASK-407.2
title: Implement chat role-play visible state and terminology cleanup
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-17 15:59'
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
- [x] #1 Applied behavior templates retain identity instead of collapsing to anonymous custom prompt state.
- [x] #2 Active chips distinguish identity, behavior, scene, generation style, and context summary.
- [x] #3 Terminology changes are routed through locale files.
- [x] #4 Focused Stage 2 tests and browser verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stage 2 started after Stage 1 commit 6bd234f1d passed spec and code-quality review. Scope remains limited to visible role-play state, behavior template identity, active chips, and terminology cleanup on main /chat.

Stage 2 implementation completed in the dedicated chat role-play remediation worktree. Added a pure derived role-play state adapter, preserved applied behavior-template identity via the chat model settings store, rendered role-play active chips for identity/behavior/scene/generation/context summary, and routed terminology changes through the English locale files. Verification: focused Stage 2 Vitest suite passed (4 files, 39 tests); git diff --check passed; locale JSON parse check passed. Full tsc still fails on existing unrelated baseline files. Browser verification for 127.0.0.1:3001/chat was blocked by the in-app browser security policy; CDP was not used because the policy explicitly prohibited routing around the blocked target. Bandit is not applicable because this stage only touches frontend TypeScript/React/locale files.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 2 adds visible, derived role-play state for the main /chat composer. Behavior templates now retain identity instead of collapsing to anonymous custom prompts, edited templates are labeled as modified, role-play chips distinguish identity/behavior/scene/generation/context summary, and user-facing terminology now uses System prompts, Generation style, and Character / Scene labels via locale files. Focused tests pass; browser verification is recorded as blocked by the browser target policy.
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
