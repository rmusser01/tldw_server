---
id: TASK-407.1
title: 'Fix chat role-play starter crash, prompt recovery, and preset accessibility'
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-05-17 06:53'
labels:
  - chat
  - ux
  - roleplay
  - stage-1
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-17-main-chat-role-play-preset-remediation-implementation-plan.md
parent_task_id: TASK-407
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stage 1 implementation for the main /chat role-play preset plan: reproduce or retire the observed starter crash, harden prompt edit/clear recovery, and make compact generation presets accessible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Chat as a character starter and default/equivalent assistant selection do not crash /chat.
- [ ] #2 Custom system prompt edit and clear remain reachable even when the prompt library is empty.
- [ ] #3 Compact generation preset controls expose understandable accessible names and keyboard behavior.
- [ ] #4 Focused Stage 1 tests and browser verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Setup before implementation:
- Dedicated branch/worktree: codex/chat-role-play-preset-remediation at .worktrees/chat-role-play-preset-remediation, based on dev plus role-play spec/plan commits.
- Dependency setup: `bun install` hung after partial resolution and was stopped; `bun install --ignore-scripts` completed and left no tracked file changes.
- Baseline focused tests passed: `bunx vitest run ../packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx ../packages/ui/src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/ParameterPresets.guard.test.ts` (13 tests passed). PromptSelect test emitted a pre-existing React key warning from the test Dropdown mock.
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
