---
id: TASK-547
title: Plan sidepanel chat WebUI handoff implementation
status: Done
labels:
- chat
- extension
- planning
documentation:
- Docs/superpowers/plans/2026-05-29-sidepanel-chat-webui-handoff-implementation.md
modified_files:
- Docs/superpowers/plans/2026-05-29-sidepanel-chat-webui-handoff-implementation.md
priority: Medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a task-by-task implementation plan for the approved sidepanel chat Continue in WebUI handoff: fail-closed handoff storage, ControlRow action, /chat import and request inclusion, and packaged extension smoke.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan file documents exact implementation tasks for storage service, sidepanel action, WebUI import/request inclusion, and focused smoke verification.
- [x] #2 Plan names exact files to create or modify and exact focused test commands.
- [x] #3 Plan preserves design constraints from TASK-546, including route-only open behavior, explicit state transfer, fail-closed storage, no auto-send, and no fresh page-body capture.
- [x] #4 Plan records why subagent plan review was not dispatched in this session.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-29-sidepanel-chat-webui-handoff-implementation.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created a task-by-task implementation plan from the approved and hardened handoff design. The plan uses TDD slices, focused commits, exact file paths, and focused Vitest commands. Subagent plan review was not dispatched because the available subagent tool policy requires explicit user permission for delegation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implementation plan written for sidepanel chat WebUI handoff. It splits the work into storage, sidepanel action, WebUI import/request inclusion, and smoke verification, with explicit regression coverage for route-only behavior, role-play query preservation, imported-context request semantics, and stale handoff handling.
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
