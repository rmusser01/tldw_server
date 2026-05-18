---
id: TASK-428
title: Implement Character Chat Phase 0 contracts and real-backend harness
status: Done
labels:
- chat
- characters
- role-play
- phase-0
- tests
priority: High
references:
- Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md
- https://github.com/rmusser01/tldw_server/pull/1840
- TASK-429
documentation:
- Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md
modified_files:
- Docs/Development/Character_Chat_Real_Backend_E2E.md
- Docs/Development/Running_Chat_Tests.md
- apps/packages/ui/src/hooks/chat/__tests__/useCharacterChatMode.contract.test.ts
- apps/packages/ui/src/hooks/__tests__/useServerChatHistory.test.ts
- apps/packages/ui/src/services/__tests__/tldw-api-client.chat-debug.test.ts
- apps/tldw-frontend/e2e/workflows/journeys/character-chat.spec.ts
- backlog/tasks/task-428 - Implement-Character-Chat-Phase-0-contracts-and-real-backend-harness.md
- backlog/tasks/task-429 - Add-Character-Chat-DB-health-and-recovery-release-gate.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Phase 0 from the first-class Character Chat PRD. Scope: freeze current /chat character behavior through tests and documentation, verify character-scoped history and character streaming payload seams, document real-backend E2E profile, and decide/link the chat DB health GA dependency. Avoid production UI behavior changes except diagnostics/test hooks required for reliable tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tests verify current character chat creation and streaming payload shape at service or hook level.
- [ ] #2 Tests verify character-scoped chat history uses filterMode=character and backend character_scope mapping.
- [ ] #3 A real-backend E2E profile is documented and runnable locally for character select/send/resume using backend provider path, not frontend-only simulation.
- [ ] #4 The per-user chat DB corruption blocker has an owner, linked task, and release-gate decision documented for Character Chat GA.
- [ ] #5 No production behavior changes are introduced except optional test IDs or diagnostics needed for reliable tests.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Character Chat Phase 0 contracts and documentation. Verification: `bunx vitest run src/hooks/chat/__tests__/useCharacterChatMode.contract.test.ts src/hooks/__tests__/useServerChatHistory.test.ts src/services/__tests__/tldw-api-client.chat-debug.test.ts --maxWorkers=1` passed with 24 tests; `bunx playwright test e2e/workflows/journeys/character-chat.spec.ts --reporter=line --list` loaded and listed the real-backend journey; `git diff --check` passed. Live real-backend Playwright execution was not run in this slice because no running backend/model provider was started in-session. Bandit was not run because no Python code was touched.
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
