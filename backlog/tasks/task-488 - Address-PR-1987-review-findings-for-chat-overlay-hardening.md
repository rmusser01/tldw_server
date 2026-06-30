---
id: TASK-488
title: Address PR-1987 review findings for chat overlay hardening
status: Done
documentation:
- Docs/superpowers/plans/2026-05-23-pr-1987-review-fixes.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the actionable PR-1987 review findings for chat overlay hardening: fix stale `selectionMode` persistence in `AssistantSelect`, prevent scratch chat settings from leaking into unrelated hydrated server chats, remove the swallowed cleanup exception in the plain-chat endpoint test, and delete the dead E2E debug stub left in `useMessageOption`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] PR review thread for stale assistant selection mode is addressed with regression coverage.
- [x] PR review thread for scratch chat-settings leakage is addressed with regression coverage.
- [x] Cleanup exception swallow and dead E2E debug stub are removed.
- [x] Tracked character/persona reload blocker is fixed with unit and real-server browser coverage.
- [x] Verification and residual Bandit findings are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented PR review fixes and critical tracked identity reload hardening. Review fixes: captured active assistant selection mode before persistence, made scratch chat-settings fallback opt-in and only enabled it for first server-chat linking, removed the dead E2E debug stub, and replaced swallowed cleanup with direct ignore_errors cleanup. Critical reload fix: tracked character server-chat creation now immediately persists tracked identity into the playground session; tracked persona creation now writes assistant metadata after setServerChatId clears identity fields; session autosave enriches generic tracked snapshots from the richer persisted selection instead of overwriting display names with generic Persona/Assistant labels. Self-review also added a regression covering immediate persistence after an empty restore attempt and cleaned restore-branch formatting.

Verification run:
- bunx vitest run src/hooks/chat/__tests__/personaServerChat.test.ts src/hooks/chat/__tests__/useChatActions.persona.integration.test.tsx src/hooks/__tests__/usePlaygroundSessionPersistence.test.tsx -> 3 files, 15 tests passed.
- TLDW_WEB_AUTOSTART=false TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 TLDW_WEB_URL=http://127.0.0.1:8080 bun run e2e:pw -- e2e/workflows/chat-cockpit.real-server.spec.ts --grep "same conversation while overlay changes|tracked character chat|tracked persona chat" --reporter=line -> 3 passed. A sandboxed attempt failed because Chromium could not register the macOS Mach port; rerun outside sandbox passed.
- Broader touched UI slice -> 12 files, 70 tests passed.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Character_Chat/test_character_chat_endpoints.py -k plain_chat_session_without_tracked_identity -v -> 1 passed, 14 deselected.
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/tests/Character_Chat/test_character_chat_endpoints.py -f json -o /tmp/bandit_pr1987.json -> only low-severity B101 assert_used findings in pytest assertions; no errors or medium/high findings.
- git diff --check -> clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR-1987 review feedback is addressed and the critical tracked chat reload blocker is fixed. Overlay changes continue using the same conversation, tracked character/persona chats restore after reload with the correct tracked identity, scratch settings no longer leak into unrelated loaded chats, and the cleanup/debug-stub review comments are addressed. Residual Bandit output is limited to expected B101 assert_used warnings in the pytest file.
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
