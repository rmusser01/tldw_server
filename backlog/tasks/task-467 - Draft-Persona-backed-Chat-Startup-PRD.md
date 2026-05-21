---
id: TASK-467
title: Draft Persona-backed Chat Startup PRD
status: Done
labels:
- persona
- prd
- docs
priority: Medium
references:
- https://github.com/rmusser01/tldw_server/issues/1908
- https://github.com/rmusser01/tldw_server/issues/1902
modified_files:
- Docs/Product/Persona_Backed_Chat_Startup_PRD.md
- backlog/tasks/task-467 - Draft-Persona-backed-Chat-Startup-PRD.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Draft a repo-grounded PRD for making Personas first-class startup choices in ordinary chat flows, scoped separately from Buddy animation, Workspace defaults, and design-system backlog work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PRD is grounded in current chat and Persona contracts.
- [x] #2 Scope, non-goals, staged implementation, risks, and validation plan are documented.
- [x] #3 Issue #1908 and tracker #1902 are referenced.
- [x] #4 Docs-only verification is recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify PR #1905 merge and create issue #1908 for the Persona-backed Chat Startup PRD. 2. Inspect current chat/persona contracts to distinguish implemented substrate from missing product completion work. 3. Draft a repo-grounded PRD with scope, non-goals, staged implementation, risks, and validation. 4. Run docs-only verification and update task status.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the Persona-backed Chat Startup PRD and grounded it in the existing unified assistant selection, Persona catalog, persona server-chat helper, and chat session metadata contracts. Documented scope, non-goals, staged delivery, risks, open questions, and validation plan. Verification: git diff --check passed. Bandit skipped because this is docs/backlog only.
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
