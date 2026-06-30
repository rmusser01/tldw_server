---
id: TASK-487
title: Add WebUI /chat end-to-end verification for overlay and tracked identity
status: Done
documentation:
- Docs/superpowers/specs/2026-05-22-chat-character-overlay-and-tracked-identity-design.md
- Docs/superpowers/plans/2026-05-22-chat-character-overlay-and-tracked-identity-implementation-plan.md
- Docs/superpowers/plans/2026-05-22-chat-overlay-webui-e2e-hardening.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add real WebUI `/chat` end-to-end coverage for the merged tracked-vs-overlay assistant identity model, focusing on the desktop rail workflow and preserving conversation continuity when overlays are applied, changed, or cleared mid-chat.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Add real-server `/chat` E2E coverage for applying, changing, and clearing assistant overlay identity without changing the active conversation.
- [x] Add real-server `/chat` E2E coverage proving tracked character chats remain linked to their character identity.
- [x] Add real-server `/chat` E2E coverage proving tracked persona chats remain linked to their persona identity.
- [x] Record focused unit/integration verification for overlay, tracked identity, greeting, persistence, and chat-settings behavior.
- [x] Record Bandit verification for touched backend and E2E Python-adjacent scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verification recorded for the focused `/chat` hardening slice: `bunx vitest run src/components/Option/Playground/__tests__/CharacterControlRail.test.tsx src/components/Option/Playground/hooks/__tests__/usePlaygroundPersistence.test.tsx src/hooks/__tests__/useCharacterGreeting.test.tsx src/hooks/__tests__/useMessage.routing-mode.test.ts src/hooks/__tests__/useSelectedAssistant.test.tsx src/hooks/__tests__/useServerChatLoader.test.ts src/hooks/chat/__tests__/effective-assistant-state.test.ts src/hooks/chat/__tests__/useChatActions.overlay.integration.test.tsx src/hooks/chat/__tests__/useChatActions.persona.integration.test.tsx src/hooks/utils/__tests__/messageHelpers.test.ts src/services/__tests__/chat-settings.overlay.test.ts src/types/__tests__/assistant-selection.test.ts` passed with 102 tests green; `python -m pytest tldw_Server_API/tests/Character_Chat/test_character_chat_endpoints.py -k "overlay or plain or assistant_overlay" -v` passed with 1 selected test green; `python -m pytest tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_settings_merge.py -k assistant_overlay -v` passed with 1 selected test green; `python -m pytest tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_session_create_schema.py -v` passed with 3 tests green; `bun run e2e:pw -- e2e/workflows/chat-cockpit.real-server.spec.ts --grep "same conversation while overlay changes|tracked character chat|tracked persona chat" --reporter=line` passed with 3 tests green against `http://127.0.0.1:8000`; `python -m bandit -r apps/tldw-frontend/e2e tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py -f json -o /tmp/bandit_task487_full.json` exited clean with only existing `# nosec` warnings and no findings. Residual risk: the real-server browser slice still depends on shared local rate-limit windows, so the helper paths now back off on transient 429s during fixture setup and overlay clear rather than assuming an isolated local environment.
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
