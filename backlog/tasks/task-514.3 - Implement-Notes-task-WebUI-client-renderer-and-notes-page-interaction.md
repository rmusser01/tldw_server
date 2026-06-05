---
id: TASK-514.3
title: Implement Notes task WebUI client, renderer, and notes page interaction
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
- apps/packages/ui/src/services/notes-tasks.ts
- apps/packages/ui/src/services/tldw/openapi-guard.ts
- apps/packages/ui/src/components/Notes/task-markdown.ts
- apps/packages/ui/src/components/Notes/TaskChecklistPreview.tsx
- apps/packages/ui/src/components/Notes/hooks/useNotesEditorState.tsx
- apps/packages/ui/src/components/Notes/NotesEditorPane.tsx
- apps/packages/ui/src/components/Notes/notes-manager-types.ts
- apps/packages/ui/src/public/_locales/en/option.json
- apps/packages/ui/src/services/__tests__/notes-tasks.test.ts
- apps/packages/ui/src/components/Notes/__tests__
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add typed WebUI task API helpers, shared task checklist renderer, `/notes` preview/split interaction, local dirty checkbox behavior, conflict copy, and task continuity copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute plan Tasks 6-7. Run focused Vitest suites for notes task services, markdown helpers, renderer, and NotesManagerPage task-backed todos; run OpenAPI guard if paths changed.
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
