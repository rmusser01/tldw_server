---
id: TASK-479
title: Harden Persona chat resume metadata contract
status: Done
labels:
- persona
- chat
- webui
- tests
priority: Medium
references:
- https://github.com/rmusser01/tldw_server/issues/1908
- Docs/superpowers/plans/2026-05-22-persona-backed-chat-startup-hardening.md
- https://github.com/rmusser01/tldw_server/pull/1940
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the resume metadata slice from the Persona-backed Chat Startup plan: add focused coverage so Persona-backed loaded chats preserve Persona identity even with legacy character metadata and never synthesize read_write from invalid/missing memory mode metadata.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Loader tests cover Persona metadata with legacy character_id present and preserve Persona identity.
- [x] #2 Loader tests cover invalid and missing persona_memory_mode without escalating to read_write.
- [x] #3 Legacy Character-only resume behavior remains covered.
- [x] #4 Verification commands and Bandit applicability are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Updated `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts` so explicit Persona assistant metadata clears legacy `character_id` rather than preserving it into resumed Persona chat state.
- Extended `apps/packages/ui/src/hooks/__tests__/useServerChatLoader.test.ts` with Persona resume coverage for legacy `character_id`, missing `persona_memory_mode`, and existing invalid memory-mode handling.
- Verified red/green behavior: the new legacy `character_id` Persona case failed before the resolver change and passed after it.
- Verified with `bunx vitest run src/hooks/__tests__/useServerChatLoader.test.ts --reporter=verbose` from `apps/packages/ui`.
- Bandit not applicable: no Python files changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened Persona chat resume metadata resolution so Persona-backed loaded chats cannot leak legacy Character IDs into local server-chat state. Added focused loader coverage for legacy `character_id`, missing memory mode, invalid memory mode, and legacy Character fallback behavior.
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
