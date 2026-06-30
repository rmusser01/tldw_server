---
id: TASK-2264
title: Harden Persona backend chat session metadata contract
status: Done
labels:
- persona
- chat
- backend
- tests
priority: Medium
references:
- https://github.com/rmusser01/tldw_server/issues/1908
- Docs/superpowers/plans/2026-05-22-persona-backed-chat-startup-hardening.md
- https://github.com/rmusser01/tldw_server/pull/1941
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the backend session metadata slice from the Persona-backed Chat Startup plan: verify Persona chat session creation requires assistant_id, preserves explicit read_only/read_write modes, rejects invalid modes, and rejects persona memory mode on Character chats.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Backend tests verify Persona chat creation requires assistant_id.
- [x] #2 Backend tests verify explicit read_only/read_write persona_memory_mode values are accepted and preserved where applicable.
- [x] #3 Backend tests verify persona_memory_mode is rejected for Character chats and invalid Persona memory modes are rejected.
- [x] #4 Verification commands and Bandit applicability are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implementation notes:
- Extended ChatSessionCreate schema contract coverage in tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_session_create_schema.py.
- Existing test already verifies Persona chat creation requires assistant_id.
- Added coverage for explicit read_only/read_write preservation.
- Added rejection coverage for persona_memory_mode on Character chats and invalid Persona memory modes.

Verification:
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_session_create_schema.py -q
- git diff --check

Bandit:
- Skipped: this slice changes tests and Backlog metadata only; no executable application code was changed.

Final summary:
- Persona backend chat session schema contract coverage now covers the remaining tracked memory-mode boundaries for the current startup hardening slice.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Persona backend chat session schema contract coverage now covers the remaining tracked memory-mode boundaries for the current startup hardening slice.
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
