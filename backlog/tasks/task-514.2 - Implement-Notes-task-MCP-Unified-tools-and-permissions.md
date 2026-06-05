---
id: TASK-514.2
title: Implement Notes task MCP Unified tools and permissions
status: To Do
parent_task_id: TASK-514
references:
- TASK-512
- TASK-513
- TASK-514
documentation:
- Docs/superpowers/specs/2026-06-05-notes-task-backed-todo-lists-design.md
- Docs/superpowers/plans/2026-06-05-notes-task-backed-todo-lists-implementation-plan.md
modified_files:
- tldw_Server_API/app/core/MCP_unified/modules/implementations/notes_module.py
- tldw_Server_API/tests/MCP_unified/test_notes_task_tools.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose Notes task-backed to-do list operations through MCP Unified with strict validation, permissions, approval-required handling, denied autonomous writes, idempotency, and scope enforcement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute plan Task 5. Keep autonomous writes denied or approval-required until the activity slice enables them after durable notice tests pass. Run `tldw_Server_API/tests/MCP_unified/test_notes_task_tools.py` and Bandit for touched backend scope.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
