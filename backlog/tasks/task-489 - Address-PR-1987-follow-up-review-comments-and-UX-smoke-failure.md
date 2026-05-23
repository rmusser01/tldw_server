---
id: TASK-489
title: Address PR-1987 follow-up review comments and UX smoke failure
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the latest PR-1987 follow-up review threads from CodeRabbit and the UX Smoke Gate failure. Scope includes validating each review comment, fixing only still-valid issues, updating Backlog task metadata, running focused regressions, and resolving/commenting on GitHub threads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Plain `/chat` server persistence requires no character, does not create a fallback character, and does not persist stale selected-character state as tracked identity.
- [x] Character workflow activation ignores stale `selectedCharacter` when the current selected assistant is not a tracked character.
- [x] `useSelectedCharacter` preserves an explicit incoming selection mode instead of overwriting it with the previous mode.
- [x] Server chat loader avoids reload churn from selected assistant object identity changes while preserving fallback assistant resolution.
- [x] Backend streaming metadata uses the module JSON import and the streaming autoexec suite terminates instead of heartbeating indefinitely.
- [x] Real-server cockpit E2E reload wait uses the persisted session server chat id instead of an undefined `plainChat.id`.
- [x] PR review housekeeping comments are addressed, including persona-test mock cleanup and TASK-487 completion metadata.
- [x] Focused frontend/backend verification, diff check, and Bandit are recorded before resolving the PR threads.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added RED frontend tests before production changes for plain-chat persistence and stale selected-character persistence.
- Added RED selection-mode coverage before changing `useSelectedCharacter`; added a plain server restore regression that was already green on the existing session persistence path.
- Confirmed the backend streaming timeout reproduced as a unified-stream heartbeat hang before the `_json.dumps` fix, then verified the streaming autoexec suite completes after the fix.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR-1987 follow-up review threads by requiring explicit tracked intent before character persistence, ignoring stale selected-character state when another assistant is selected, preserving explicit character selection modes, using a selected-assistant ref in the server chat loader, fixing the backend streaming metadata `_json.dumps` call, hardening the real-server cockpit reload wait, removing the redundant persona-test mock implementation, and completing TASK-487 metadata. Also updated the prompt cockpit E2E expectations to match the current actionable empty state while leaving unit focus-return coverage in place.

Verification recorded:
- `bunx vitest run src/components/Option/Playground/hooks/__tests__/usePlaygroundPersistence.test.tsx src/hooks/__tests__/useSelectedAssistant.test.tsx src/hooks/__tests__/usePlaygroundSessionPersistence.test.tsx src/hooks/__tests__/useServerChatLoader.test.ts src/hooks/chat/__tests__/useChatActions.persona.integration.test.tsx src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx` passed with 65 tests green.
- `python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py -q --timeout=60` passed with 6 tests green.
- `TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run e2e:pw -- e2e/workflows/chat-cockpit.real-server.spec.ts --grep "real prompt, model setting restore" --reporter=line --workers=1` passed with 1 Chromium test green against a local single-user backend.
- `python -m bandit -r tldw_Server_API/app/core/Chat/chat_service.py -f json -o /tmp/bandit_task489.json` exited clean with no findings.
- `git diff --check` exited clean.

CI note: GitHub reported the UX Smoke Gate failure while the workflow run was still in progress, so detailed job logs were not downloadable yet. Job metadata showed the failed smoke job stopped at frontend dependency install and backend health, while a later job in the same run successfully installed dependencies and passed backend health. Recheck CI after the follow-up commit is pushed.
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
