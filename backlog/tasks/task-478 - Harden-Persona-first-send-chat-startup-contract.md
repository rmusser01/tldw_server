---
id: TASK-478
title: Harden Persona first-send chat startup contract
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
- https://github.com/rmusser01/tldw_server/pull/1935
- https://github.com/rmusser01/tldw_server/pull/1939
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first-send Persona chat startup slice from the Persona-backed Chat Startup plan: ensure normal chat first send creates Persona-backed server chats with explicit read_only metadata and no stale Character identity leakage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 First-send integration coverage asserts selected Persona creates a server chat with assistant_kind persona, selected assistant_id, explicit read_only memory mode, and fresh metadata.
- [x] #2 First-send integration coverage asserts stale Character identity is not forwarded into Persona chat state.
- [x] #3 Any production changes are minimal and only address failing contract cases.
- [x] #4 Verification commands and Bandit applicability are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Extended `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.persona.integration.test.tsx`.
- Tightened the existing first-send Persona assertion so it verifies the exact `createChat` payload: `assistant_kind: "persona"`, selected Persona `assistant_id`, explicit `read_only`, fresh `in-progress` state, and no inherited topic/cluster/source/external-ref metadata.
- Added a stale Character-state regression proving Persona first send resets stale Character chat metadata before creating the Persona-backed chat and does not forward the stale Character ID into the Persona path.
- No production implementation changes were needed; current code satisfies the new first-send contract after the prior helper hardening.
- Verified with `bunx vitest run src/hooks/chat/__tests__/useChatActions.persona.integration.test.tsx --reporter=verbose` from `apps/packages/ui`.
- Verified adjacent helper coverage with `bunx vitest run src/hooks/chat/__tests__/useChatActions.persona.integration.test.tsx src/hooks/chat/__tests__/personaServerChat.test.ts --reporter=verbose` from `apps/packages/ui`.
- Bandit not applicable: no Python files changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added focused first-send integration coverage for Persona-backed chat startup. The tests now lock explicit `read_only` Persona session creation, fresh startup metadata, normal chat handoff, and stale Character metadata isolation.
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
