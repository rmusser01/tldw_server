---
id: TASK-447
title: Add effective assistant mode resolver and remove destructive reset
status: Done
labels:
- implementation
- chat
- frontend
- state
priority: high
documentation:
- Docs/superpowers/specs/2026-05-22-chat-character-overlay-and-tracked-identity-design.md
- Docs/superpowers/plans/2026-05-22-chat-character-overlay-and-tracked-identity-implementation-plan.md
modified_files:
- apps/packages/ui/src/hooks/chat/effective-assistant-state.ts
- apps/packages/ui/src/hooks/useMessageOption.tsx
- apps/packages/ui/src/hooks/chat/useSelectServerChat.ts
- apps/packages/ui/src/hooks/chat/useChatActions.ts
- apps/packages/ui/src/hooks/chat/__tests__/effective-assistant-state.test.ts
- apps/packages/ui/src/hooks/__tests__/useMessageOption.assistant-overlay.test.tsx
- apps/packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx
- apps/packages/ui/src/hooks/chat/__tests__/useChatActions.persona.integration.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Introduce a pure tracked-vs-overlay assistant mode resolver and remove the destructive assistant-switch reset behavior from the /chat flow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed Task 2 on branch codex/chat-character-overlay-tracked-identity. Verification: `node node_modules/vitest/vitest.mjs --config vitest.config.ts run src/hooks/chat/__tests__/effective-assistant-state.test.ts src/hooks/__tests__/useMessageOption.assistant-overlay.test.tsx src/hooks/__tests__/useMessageOption.selected-model-sync.test.tsx src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx src/hooks/chat/__tests__/useChatActions.persona.integration.test.tsx` from `apps/packages/ui` -> 5 test files passed, 14 tests passed. Bandit is not applicable to this frontend-only TypeScript slice; running `python -m bandit -r apps/packages/ui/src/hooks/chat/effective-assistant-state.ts apps/packages/ui/src/hooks/useMessageOption.tsx apps/packages/ui/src/hooks/chat/useSelectServerChat.ts apps/packages/ui/src/hooks/chat/useChatActions.ts -f json -o /private/tmp/bandit_task447.json` produced parser errors on `.ts/.tsx` inputs and no actionable findings.
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
