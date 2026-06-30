---
id: TASK-474
title: Plan Persona-backed Chat Startup hardening
status: Done
labels:
- persona
- chat
- implementation-plan
- docs
priority: Medium
references:
- https://github.com/rmusser01/tldw_server/issues/1908
- https://github.com/rmusser01/tldw_server/pull/1909
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a repo-grounded implementation plan for Persona-backed ordinary chat startup hardening, focusing on Stage 1 contract audit/tests and the smallest Stage 2 startup/persistence fixes from the accepted PRD.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan maps current code/test coverage for Persona-backed chat startup.
- [x] #2 Plan defines scoped TDD tasks for assistant selection, Persona server chat helper, server chat loader/resume metadata, backend session metadata, and first-send behavior.
- [x] #3 Plan explicitly excludes Buddy animation/runtime, Workspace defaults, scheduled work, broad personalization memory, and design-system backlog work.
- [x] #4 Docs-only verification is recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
['Inspect current frontend and backend Persona-backed chat startup code/tests on dev.', 'Write a focused implementation plan under Docs/superpowers/plans with TDD tasks and validation commands.', 'Update Backlog task with evidence and verification, then commit plan-only changes.']
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created `Docs/superpowers/plans/2026-05-22-persona-backed-chat-startup-hardening.md`.
- Grounded the plan in the current WebUI/backend implementation:
  - `apps/packages/ui/src/types/assistant-selection.ts`
  - `apps/packages/ui/src/hooks/chat/personaServerChat.ts`
  - `apps/packages/ui/src/hooks/chat/useChatActions.ts`
  - `apps/packages/ui/src/hooks/useMessage.tsx`
  - `apps/packages/ui/src/hooks/chat/useServerChatLoader.ts`
  - `tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py`
  - existing Persona chat tests under `apps/packages/ui/src/hooks/**/__tests__`, `apps/packages/ui/src/components/Common/__tests__`, and `tldw_Server_API/tests/Character_Chat/`.
- Identified the first concrete hardening target: stale non-Persona chat state can currently carry `serverChatPersonaMemoryMode: read_write` into a newly created Persona chat because `ensurePersonaServerChat` computes the effective memory mode before stale chat reset.
- Kept scope strictly to Persona-backed ordinary chat startup and explicitly excluded Buddy runtime, Persona visual packs, Workspace defaults, scheduled work, broad personalization memory, and design-system backlog work.
- Docs/backlog-only change; Bandit is not applicable because no executable Python code changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a focused implementation plan for finishing the current Persona-backed ordinary chat startup slice from issue #1908. The plan maps existing code/test coverage, names the smallest risky behavior to fix first, and defines TDD tasks plus verification commands for frontend helper behavior, first-send startup, resume metadata, and backend session metadata.
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
