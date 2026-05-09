---
id: TASK-171
title: Remove dayjs from Kanban CardDetailPanel due-date editing
status: Done
assignee: []
created_date: '2026-05-09 17:46'
updated_date: '2026-05-09 17:55'
labels:
  - webui
  - dependencies
  - issue-1346
  - kanban
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1346'
  - 'https://github.com/rmusser01/tldw_server/pull/1427'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue issue #1346 by replacing the shared WebUI Kanban card due-date editor's dayjs/Ant Design DatePicker path with native platform date-time handling. Scope is limited to CardDetailPanel due-date form state, parsing/formatting helpers, focused tests, and the dependency audit. Leave Media, ReadingList, and Items RangePicker Dayjs value-contract surfaces for later design slices.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CardDetailPanel no longer imports or references dayjs or Ant Design DatePicker for due-date editing.
- [x] #2 The due-date editor uses a native datetime-local input that displays existing ISO due dates in local YYYY-MM-DDTHH:mm form and allows clearing the field.
- [x] #3 Saving a changed due date emits an ISO timestamp, saving a cleared due date emits null, and saving unrelated fields does not rewrite an unchanged existing due date.
- [x] #4 Focused tests cover native date-time formatting/parsing and changed/unchanged/cleared due-date update behavior.
- [x] #5 Issue #1346 audit notes are updated to reflect the reduced shared UI dayjs import count and remaining deferred DatePicker value-contract surfaces.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused regression coverage for Kanban due-date helpers and CardDetailPanel changed/cleared/unchanged save behavior.
2. Replace CardDetailPanel dayjs/Ant Design DatePicker state with native datetime-local state and ISO parse/format helpers.
3. Refresh the issue #1346 dependency audit to remove Kanban from remaining dayjs surfaces.
4. Run focused Vitest, dayjs import scan, lint, diff check, touched-slice TypeScript diagnostics, and document the Bandit skip because no Python files changed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented native Kanban due-date editing with a local helper module. CardDetailPanel now stores the datetime-local string plus an explicit touched flag so unchanged existing due dates are not rewritten when saving unrelated card fields. Changed due dates are parsed as local datetime-local values and emitted as ISO strings; cleared values emit null. Replaced the deprecated Drawer width prop with numeric size after tests surfaced the AntD 6 warning, preserving the 400px panel sizing.

Verification run from /Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/webui-kanban-native-due-date-1346: focused Vitest suite passed with 2 files and 8 tests after the native helper/input implementation; rerun after replacing Drawer width with size had no deprecation warnings.

Exact shared UI dayjs package-import scan now reports 5 remaining import lines in Media FilterPanel, ReadingList ReadingItemsList, and Items ItemsWorkspace; Kanban CardDetailPanel is no longer in the scan.

bun run lint from apps/tldw-frontend exited 0 with the existing 131-warning baseline. git diff --check exited 0. Filtered apps/packages/ui TypeScript diagnostics for CardDetailPanel, kanbanDateTime, and KanbanPlayground returned no matches.

Bandit skipped for this task because the touched scope is TypeScript, tests, documentation, and Backlog metadata only; no Python files changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the Kanban CardDetailPanel dependency on dayjs and Ant Design DatePicker for due-date editing. The panel now uses a native datetime-local input backed by small local ISO parse/format helpers, with an explicit touched flag so unchanged due dates are not rewritten when saving unrelated fields. Focused tests cover helper behavior plus changed, cleared, and unchanged due-date saves. The dependency audit now records the shared UI dayjs import count at 5 remaining lines, limited to Media, ReadingList, and Items surfaces. Verification passed with the focused Kanban Vitest suite, exact dayjs import scan, git diff --check, lint exiting 0 with the known warning baseline, and filtered touched-slice TypeScript diagnostics returning no matches. Bandit was skipped because this slice changed no Python files.
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
