---
id: TASK-446
title: Add chat assistant overlay settings contract and validation
status: Done
labels:
- implementation
- chat
- frontend
- backend
- settings
priority: high
documentation:
- Docs/superpowers/specs/2026-05-22-chat-character-overlay-and-tracked-identity-design.md
- Docs/superpowers/plans/2026-05-22-chat-character-overlay-and-tracked-identity-implementation-plan.md
modified_files:
- apps/packages/ui/src/types/chat-session-settings.ts
- apps/packages/ui/src/services/chat-settings.ts
- apps/packages/ui/src/services/__tests__/chat-settings.overlay.test.ts
- apps/packages/ui/src/services/__tests__/chat-settings.sync.test.ts
- tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py
- tldw_Server_API/tests/Character_Chat/test_chat_settings_endpoints.py
- tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_settings_merge.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the assistantOverlay chat settings contract across frontend and backend validation, including local-first persistence before server chat creation and sync reconciliation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed Task 1 on branch codex/chat-character-overlay-tracked-identity in commit 4f81fbdc4. Verification: `bunx vitest run src/services/__tests__/chat-settings.overlay.test.ts src/services/__tests__/chat-settings.sync.test.ts` -> 12 passed; `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat/test_chat_settings_endpoints.py tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_settings_merge.py -k "assistant_overlay or overlay" -v` -> 8 passed, 25 deselected; `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py -f json -o /tmp/bandit_chat_overlay_task1.json` -> no new findings.
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
