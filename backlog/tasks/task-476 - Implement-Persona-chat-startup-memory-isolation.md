---
id: TASK-476
title: Implement Persona chat startup memory isolation
status: Done
labels:
- persona
- chat
- webui
- bugfix
priority: High
references:
- https://github.com/rmusser01/tldw_server/issues/1908
- https://github.com/rmusser01/tldw_server/pull/1929
- Docs/superpowers/plans/2026-05-22-persona-backed-chat-startup-hardening.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first Persona-backed Chat Startup hardening slice from #1908: prevent stale non-Persona chat state from carrying read_write memory mode into newly created Persona-backed chats.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persona server chat helper defaults new Persona-backed chats to explicit read_only after stale Character or different-Persona chat reset.
- [x] #2 Existing matching Persona-backed chats preserve read_write when reused.
- [x] #3 Focused WebUI tests cover stale Character and stale different-Persona memory isolation.
- [x] #4 Verification commands and Bandit applicability are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Updated `apps/packages/ui/src/hooks/chat/personaServerChat.ts` so `read_write` is preserved only when reusing an existing matching Persona-backed server chat.
- New Persona-backed chats after stale Character or different-Persona reset now start with explicit `read_only` and a fresh `in-progress` metadata payload instead of carrying stale topic/source/cluster/external-ref state.
- Updated `apps/packages/ui/src/hooks/chat/__tests__/personaServerChat.test.ts` to cover stale Character reset and stale different-Persona reset behavior.
- Verified with `bunx vitest run src/hooks/chat/__tests__/personaServerChat.test.ts` from `apps/packages/ui`.
- Verified with `git diff --check`.
- Bandit not applicable: no Python files changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the first Persona-backed chat startup hardening slice. The Persona server chat helper now isolates newly created Persona chats from stale non-matching assistant state, while preserving `read_write` only for an already matching Persona session.
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
