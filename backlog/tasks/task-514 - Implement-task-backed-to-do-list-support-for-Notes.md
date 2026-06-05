---
id: TASK-514
title: Implement task-backed to-do list support for Notes
status: Done
references:
- TASK-512
- TASK-513
documentation:
- Docs/superpowers/specs/2026-06-05-notes-task-backed-todo-lists-design.md
- Docs/superpowers/plans/2026-06-05-notes-task-backed-todo-lists-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-05-notes-task-backed-todo-lists-implementation-plan.md
- backlog/tasks/task-514 - Implement-task-backed-to-do-list-support-for-Notes.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Parent execution task for implementing the approved Notes task-backed to-do list PRD across backend task foundation, MCP Unified tools, WebUI interactions, activity notices, autonomous-write gating, and final verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Parser, task storage, reconciliation, REST API, MCP Unified tools, WebUI interactions, activity notices, and final verification slices complete.
- [x] #2 Autonomous task writes remain disabled or approval-required until persistent activity notices are implemented and tested.
- [x] #3 All child slice tasks record verification results, commits, known skips, and final summaries.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute `Docs/superpowers/plans/2026-06-05-notes-task-backed-todo-lists-implementation-plan.md` using subagent-driven development. Task 0 has created this parent and child tasks before code edits. Each implementation slice must keep its Backlog task current, run the listed focused tests, run Bandit for touched backend code where applicable, and commit Backlog updates with the related code slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
All planned implementation slices completed in TASK-514.1 through TASK-514.5, with TASK-514.6 added to resolve the router contract failure found during final verification. Optional broader regression failures were triaged into TASK-514.7 and TASK-514.8 so they are tracked separately from the task-backed Notes to-do release gate.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented task-backed to-do list support for Notes across backend parser/storage/reconciliation/REST APIs, MCP Unified task tools and permissions, /notes task rendering and interaction, Notes Dock task interaction, persistent task activity notices, and scoped autonomous MCP task-write enablement. Final focused verification passed: backend notes/task/router suite 348 passed, frontend task-focused Vitest suite 7 files/20 tests passed, OpenAPI guard passed with 10 reviewed exceptions, and Bandit reported zero findings across touched backend task/router modules. Browser smoke remained skipped because this checkout has no root tests/e2e harness and no running WebUI dev server. Optional broad regressions were recorded in TASK-514.7 and TASK-514.8.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented: browser smoke skipped; optional broad regressions tracked in TASK-514.7 and TASK-514.8.
<!-- DOD:END -->
