---
id: TASK-481.2
title: Implement notes PR 2 list reliability and empty states
status: Done
labels:
- notes
- ux
- webui
- frontend
parent_task_id: TASK-481
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement PR 2 from the notes UX remediation plan: expose list query error/stale state, prevent stale totals from appearing fresh, distinguish empty/no-results/error states, and verify create/delete/restore/search list behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-27-notes-ux-remediation.md#pr-2-notes-list-reliability-and-empty-states
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented PR 2 in the stacked notes UX worktree after PR 1 commit `daf4968c8b`.

Changed files:
- `apps/packages/ui/src/components/Notes/hooks/useNotesListManagement.tsx`: exposes React Query `error`, `isError`, `isPlaceholderData`, and `refetch`.
- `apps/packages/ui/src/components/Notes/NotesManagerPage.tsx`: passes list error/stale state and retry into the sidebar.
- `apps/packages/ui/src/components/Notes/NotesSidebar.tsx`: forwards list health into the list panel and replaces authoritative header/filter counts with `Refresh failed` during list errors.
- `apps/packages/ui/src/components/Notes/NotesListPanel.tsx`: renders stale cached-result, failed-list, and no-results states distinctly.
- `apps/packages/ui/src/components/Notes/NotesListPanelEmptyStates.tsx`: adds no-results and error variants with clear-filters/retry/diagnostics actions.
- `apps/packages/ui/src/components/Notes/__tests__/NotesListPanel.stage46.empty-error-states.test.tsx`: covers filtered no-results and stale cached-results banner.
- `apps/packages/ui/src/components/Notes/__tests__/NotesSidebar.stage46.list-error-count.test.tsx`: covers list-error header/filter summary counts so stale totals are not presented as fresh.

Verification:
- RED: `NotesListPanel.stage46.empty-error-states.test.tsx` initially failed because filtered empty rendered `No notes yet` and cached stale rows had no warning banner.
- RED: `NotesSidebar.stage46.list-error-count.test.tsx` initially failed because list error rendered `0 of 1` / `Showing 0 of 1 notes` as fresh counts.
- GREEN: focused Vitest suite covering stage46, quick-save hint, selected-state accessibility, large-list, create/save, search/filtering, trash restore, and delete undo passed: 9 files, 20 tests.
- Browser verification against local Next dev server with mocked API: desktop filtered search showed `No notes match these filters` and `Clear filters`; mobile failed list request showed `Could not load notes` and `Refresh failed`; stale count strings `0 of 1` and `Showing 0 of 1 notes` were not visible.
- Bandit skipped/not applicable: PR 2 touched frontend TypeScript/React only.

Known notes:
- Browser verification required a temporary local Ant Design symlink repair because the tracked symlink target is absent from this dependency install; the symlink was restored before final status.
- Dev server required unsandboxed port binding approval after sandbox returned `EPERM` on `0.0.0.0:8080`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR 2 completed: /notes now distinguishes first-run empty, active-filter no-results, failed-list, and cached-stale-result states; retry/clear-filter actions are exposed; list query errors no longer leave stale header/filter totals looking fresh; existing create/search/delete/restore coverage remains green.
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
