---
id: TASK-514.5
title: Implement Notes task activity notices, autonomous write enablement, and final
  verification
status: Done
parent_task_id: TASK-514
references:
- TASK-512
- TASK-513
- TASK-514
documentation:
- Docs/superpowers/specs/2026-06-05-notes-task-backed-todo-lists-design.md
- Docs/superpowers/plans/2026-06-05-notes-task-backed-todo-lists-implementation-plan.md
modified_files:
- apps/packages/ui/src/components/Notes/TaskActivityNotice.tsx
- apps/packages/ui/src/components/Notes/NotesEditorPane.tsx
- apps/packages/ui/src/components/Notes/NotesManagerPage.tsx
- apps/packages/ui/src/components/Notes/hooks/useNotesEditorState.tsx
- apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.task-activity.test.tsx
- apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.task-backed-todos.test.tsx
- apps/packages/ui/src/components/Common/NotesDock/NotesDockPanel.tsx
- apps/packages/ui/src/components/Common/NotesDock/__tests__/NotesDockPanel.task-activity.test.tsx
- apps/packages/ui/src/components/Common/NotesDock/__tests__/NotesDockPanel.task-backed-todos.test.tsx
- apps/packages/ui/src/services/notes-tasks.ts
- tldw_Server_API/app/api/v1/endpoints/notes_tasks.py
- tldw_Server_API/app/api/v1/schemas/notes_tasks_schemas.py
- tldw_Server_API/app/core/MCP_unified/modules/implementations/notes_module.py
- tldw_Server_API/tests/MCP_unified/test_notes_task_tools.py
- tldw_Server_API/tests/Notes_NEW/integration/test_notes_tasks_api.py
- Docs/superpowers/plans/2026-06-05-notes-task-backed-todo-lists-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement persistent per-user task activity notices, then enable scoped autonomous MCP task writes only after activity delivery tests pass, and run final focused backend/frontend/security/browser verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Activity notices appear in `/notes` and Notes Dock for unread agent task events.
- [x] Dismissal uses persistent per-user `task_event_read_state` behavior.
- [x] Scoped autonomous MCP task writes succeed only with `notes.tasks.write.autonomous` capability.
- [x] Agent writes without approval and autonomous writes without scoped permission remain blocked.
- [x] Focused backend/frontend verification and Bandit/OpenAPI guard are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute plan Task 9 and Final Verification. Verify `task_event_read_state` read/dismiss behavior, allowed autonomous writes after activity delivery, denied autonomous policy still passing, focused backend/frontend tests, Bandit, OpenAPI guard, and browser smoke when available.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented shared Notes task activity notice UI for /notes and Notes Dock, exposed REST activity metadata fields (tool_name, policy_mode, approval_id), and enabled scoped autonomous MCP task writes when the effective policy grants notes.tasks.write.autonomous. Verification: focused UI activity red run failed before component existed; focused UI activity green 2/2; focused task UI suite 7 files/20 tests passed; focused backend suite 167 passed/7 warnings; Bandit touched backend scope exit 0 with zero findings/errors; OpenAPI guard passed with existing 10 reviewed exceptions. Optional root tests/e2e browser smoke unavailable because tests/e2e does not exist and no dev server was running. Broader backend command including test_router_groups_contract.py failed in unrelated dirty router-contract tests expecting missing router_groups.selection and route-key behavior outside this slice. Commit message: feat: surface notes task activity.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 9 completed. Added shared task activity notices for /notes and Notes Dock, including actor/tool/count/note context plus inspect and dismiss actions. REST activity responses now expose tool/policy/approval metadata used by the UI. MCP Notes task tools now allow autonomous writes only when the effective MCP policy includes notes.tasks.write.autonomous, while unapproved agent writes still require approval and autonomous writes without that capability still fail closed. Focused frontend/backend verification, Bandit, and OpenAPI guard were run; browser smoke was skipped because no root E2E harness/dev server was available; unrelated dirty router contract tests remain failing outside this slice.
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
