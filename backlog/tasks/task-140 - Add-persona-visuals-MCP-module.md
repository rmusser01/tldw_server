---
id: TASK-140
title: Add persona visuals MCP module
status: Done
assignee: []
created_date: '2026-05-09 01:22'
updated_date: '2026-05-09 01:30'
labels:
  - backend
  - frontend
  - mcp
  - persona
  - visuals
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 10 from Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md: internal persona_visuals MCP tools for visual pack capabilities, transient visual state overrides, draft pack/manifest updates, generation job enqueueing, Persona Live WebSocket propagation, frontend runtime override handling, and disabled module config.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 persona_visuals.capabilities returns active and draft visual pack summaries for the scoped persona
- [x] #2 persona_visuals.trigger_state requires persona/user context rejects unknown states and clamps duration between 100 and 30000 ms
- [x] #3 draft pack and manifest tools mutate only draft visual pack data and do not activate packs
- [x] #4 persona_visuals.enqueue_generation creates a persona visual generation Job for review
- [x] #5 module tool calls require user/persona context from MCP context or explicit persona arguments
- [x] #6 Persona Live emits visual_state_override WebSocket payloads when persona_visuals.trigger_state is invoked
- [x] #7 Frontend persona visual runtime store receives and expires visual_state_override payloads
- [x] #8 MCP module config is present and disabled by default
- [x] #9 Focused backend and frontend tests cover the MCP module and runtime override workflow
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented persona_visuals MCP module with capabilities, trigger_state, create_draft_pack, update_manifest, and enqueue_generation tools.

Added Persona Live visual_state_override payload extraction/emission and persisted the override payload on the tool result.

Confirmed existing frontend runtime override store and incoming payload hook cover receive/expiry workflow; no frontend code changes were needed for this slice.

Verification passed: python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_persona_visuals_module.py -v; bunx vitest run src/store/__tests__/persona-visual-runtime.test.ts src/routes/hooks/__tests__/usePersonaIncomingPayload.visuals.test.tsx; git diff --check; Bandit JSON at /tmp/bandit_persona_visuals_mcp.json with zero findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the disabled persona_visuals MCP module for persona-scoped visual pack capabilities, transient visual state overrides, draft-only pack/manifest edits, and Jobs-backed generation requests. Persona Live now recognizes successful persona_visuals.trigger_state tool results, emits bounded visual_state_override WebSocket payloads, and records the payload on the persisted tool result; existing UI runtime-store tests verify receive and expiry behavior.
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
