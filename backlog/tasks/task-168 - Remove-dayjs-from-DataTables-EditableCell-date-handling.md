---
id: TASK-168
title: Remove dayjs from DataTables EditableCell date handling
status: In Progress
assignee: []
created_date: '2026-05-09 16:49'
updated_date: '2026-05-09 16:55'
labels:
  - webui
  - dependencies
  - issue-1346
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1346'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue issue #1346 dependency cleanup by removing the remaining dayjs usage from the shared WebUI DataTables editable date-cell path. This is a narrow compatibility slice: keep the data-table date editing/display behavior intact while replacing dayjs parsing/formatting with platform-native date handling so the remaining Ant Design DatePicker/Dayjs value-contract surfaces can stay deferred to a separate design slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 EditableCell no longer imports or references dayjs for date display or editing.
- [x] #2 Date cell display still renders valid date-like values as YYYY-MM-DD and preserves invalid raw values rather than throwing.
- [x] #3 Date editing still emits YYYY-MM-DD for selected dates and null/empty for cleared values.
- [x] #4 Focused tests cover display formatting, invalid values, and edit-change output for date cells.
- [x] #5 Issue #1346 audit notes are updated to reflect the reduced dayjs import count and the remaining deferred DatePicker value-contract surfaces.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Replace the DataTables EditableCell date display/editing path with native date formatting and a native date input while leaving other column types untouched.
2. Add focused Vitest coverage for valid date display, invalid raw values, native input output, and clearing behavior.
3. Update the WebUI dependency audit for the reduced dayjs import count and remaining blockers.
4. Run focused tests, exact dayjs import scan, lint/diff hygiene, and document the existing shared UI TypeScript baseline.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented in isolated worktree branch codex/webui-datatables-native-date-cell-1346. EditableCell now formats date cells via a local native Date helper and renders date edits with an accessible native input[type=date]. The test suite first failed against the old behavior: invalid values rendered as Invalid Date and edit mode exposed Ant Design DatePicker rather than the native input contract. After implementation, the focused tests pass. The audit now records shared UI dayjs package imports dropping from 7 to 6 and keeps the remaining Media, ReadingList, Items, and Kanban DatePicker/Dayjs surfaces deferred.

Verification: `bunx vitest run src/components/Option/DataTables/__tests__/EditableCell.date.test.tsx --maxWorkers=1` failed before implementation on invalid date rendering and missing native date input; after implementation it passed with 4 tests.

Exact shared UI dayjs package-import scan now reports 6 lines: Media/FilterPanel.tsx, ReadingItemsList.tsx type/runtime imports, ItemsWorkspace.tsx type/runtime imports, and KanbanPlayground/CardDetailPanel.tsx.

`git diff --check` exited 0. `bun run lint` from apps/tldw-frontend exited 0 with the existing 131-warning baseline. Full `./node_modules/.bin/tsc --noEmit --project tsconfig.json --pretty false` from apps/packages/ui exits 2 on existing repo-wide baseline diagnostics; filtering the same diagnostics for `EditableCell|DataTables` returned no matches.

Bandit skipped because this slice touches TypeScript, tests, docs, and Backlog metadata only; no Python files changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed dayjs from the DataTables EditableCell date path by replacing display formatting with native Date handling and replacing the Ant Design DatePicker editor with an accessible native date input. Added focused EditableCell date tests and updated the WebUI dependency audit to record the shared UI dayjs import count dropping from 7 to 6 while keeping the remaining DatePicker/Dayjs surfaces deferred.
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
