---
id: TASK-514.5
title: Implement Notes task activity notices, autonomous write enablement, and final
  verification
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
- apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.task-activity.test.tsx
- apps/packages/ui/src/components/Common/NotesDock/__tests__/NotesDockPanel.task-activity.test.tsx
- apps/packages/ui/src/components/Notes
- apps/packages/ui/src/components/Common/NotesDock
- apps/packages/ui/src/store/notes-dock.tsx
- apps/packages/ui/src/public/_locales/en/option.json
- tldw_Server_API/app/core/MCP_unified/modules/implementations/notes_module.py
- tldw_Server_API/tests/MCP_unified/test_notes_task_tools.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement persistent per-user task activity notices, then enable scoped autonomous MCP task writes only after activity delivery tests pass, and run final focused backend/frontend/security/browser verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute plan Task 9 and Final Verification. Verify `task_event_read_state` read/dismiss behavior, allowed autonomous writes after activity delivery, denied autonomous policy still passing, focused backend/frontend tests, Bandit, OpenAPI guard, and browser smoke when available.
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
