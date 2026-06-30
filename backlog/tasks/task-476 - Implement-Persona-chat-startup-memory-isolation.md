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
- PR #1935 review follow-up: added `serverChatMetaLoaded` to the Persona startup helper and callers so a restored `serverChatId` with assistant metadata still hydrating is reused instead of reset or recreated prematurely.
- PR #1935 review follow-up: brand-new Persona chats now also use fresh `in-progress` metadata even when stale conversation metadata remains in the store.
- Updated `apps/packages/ui/src/hooks/chat/__tests__/personaServerChat.test.ts` to cover stale Character reset, stale different-Persona reset, brand-new chat stale metadata isolation, and restored-chat metadata-hydration behavior.
- Verified red/green with `bunx vitest run src/hooks/chat/__tests__/personaServerChat.test.ts --reporter=verbose` from `apps/packages/ui`.
- Verified integration with `bunx vitest run src/hooks/chat/__tests__/personaServerChat.test.ts src/hooks/chat/__tests__/useChatActions.persona.integration.test.tsx --reporter=verbose` from `apps/packages/ui`.
- Verified with `git diff --check`.
- Ran frontend lint through the WebUI config; the only error reported in touched files was pre-existing `no-extra-boolean-cast` in `useChatActions.ts:3121`, outside this patch hunk. The WebUI-wide lint command exits 0 with existing warnings.
- `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` still fails on existing shared UI baseline type debt outside this slice and reports no diagnostics for the Persona startup changes.
- Bandit not applicable: no Python files changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed the PR #1935 review follow-up for Persona chat startup memory isolation. The helper now waits for server chat metadata before treating a restored chat ID as stale, always starts newly created Persona chats with fresh metadata, defaults unknown/unhydrated Persona startup memory to read_only, and preserves read_write only for loaded matching Persona sessions.
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
