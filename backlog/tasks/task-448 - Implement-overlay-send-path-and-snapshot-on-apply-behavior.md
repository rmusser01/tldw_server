---
id: TASK-448
title: Implement overlay send path and snapshot-on-apply behavior
status: Done
labels:
- implementation
- chat
- frontend
- personas
- characters
priority: high
documentation:
- Docs/superpowers/specs/2026-05-22-chat-character-overlay-and-tracked-identity-design.md
- Docs/superpowers/plans/2026-05-22-chat-character-overlay-and-tracked-identity-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement snapshot-on-apply overlay resolution from full character/persona detail and route normal chat sends through overlay-aware prompt assembly without changing tracked chat semantics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 3 now separates overlay application from ordinary assistant selection, persists overlay snapshots in chat settings using resolved source detail, preserves tracked character/persona send paths, and prevents overlay changes from clearing the active conversation. Verification: `node node_modules/vitest/vitest.mjs --config vitest.config.ts run src/utils/__tests__/assistant-overlay.test.ts src/components/Common/__tests__/AssistantSelect.behavior.test.tsx src/components/Common/__tests__/AssistantSelect.tabs.test.tsx src/hooks/chat-modes/__tests__/normalChatMode.overlay.test.ts src/hooks/__tests__/useMessage.routing-mode.test.ts src/hooks/__tests__/useMessage.assistant-overlay.guard.test.ts src/hooks/__tests__/useMessageOption.assistant-overlay.test.tsx` -> 7 files passed, 29 tests passed. Bandit not applicable because the touched scope in this task is TypeScript/TSX only.
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
