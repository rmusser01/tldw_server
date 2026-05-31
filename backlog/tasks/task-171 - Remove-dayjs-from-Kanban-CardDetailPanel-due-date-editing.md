---
id: TASK-171
title: Remove dayjs from Kanban CardDetailPanel due-date editing
status: Done
assignee: []
created_date: '2026-05-09 17:46'
updated_date: '2026-05-09 18:43'
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

Expected impact estimate for this narrow replacement: reduce active shared UI dayjs package-import lines from 6 to 5 and remove one DatePicker/Dayjs runtime editing surface from Kanban. No direct manifest, lockfile, install-size, or bundle-size reduction is expected until dayjs is removed from the remaining Media, ReadingList, and Items surfaces, and Ant Design still owns dayjs transitively.
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

PR review follow-up plan for #1427:
1. Add a focused regression test proving a due-date save followed by an unrelated title save does not resend due_date while the drawer stays open.
2. Fix the touched-state reset behavior with the smallest local change that preserves existing CardDetailPanel/BoardView contracts.
3. Replace numeric Drawer size with an AntD 6-compatible pixel-width pattern and update verification notes.
4. Remove machine-specific local paths from Backlog notes, add the missing expected impact estimate, and run/record the WebUI build required by issue #1346.
5. Re-run focused tests, import scan, lint/build/type-filter/diff checks, push, and resolve/reply to review threads.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented native Kanban due-date editing with a local helper module. CardDetailPanel now stores the datetime-local string plus an explicit touched flag so unchanged existing due dates are not rewritten when saving unrelated card fields. Changed due dates are parsed as local datetime-local values and emitted as ISO strings; cleared values emit null.

PR review follow-up: verified the stale touched-state finding by adding a regression test that first failed because a due-date save followed by a title save in the same open drawer resent due_date. Fixed it by allowing CardDetailPanel.onSave to return a promise, awaiting the parent update, resetting dueDateTouched only after save success, and having BoardView use mutateAsync plus setSelectedCard(updatedCard) so the open panel receives the saved card. Also replaced numeric Drawer size with the repo's AntD 6 pixel-width pattern, styles.wrapper.width, preserving the 400px drawer width without using the deprecated width prop.

Verification run from the feature worktree (webui-kanban-native-due-date-1346): focused Vitest suite passed with 2 files and 9 tests after the review follow-up. The new stale touched-state regression first failed with the second save including due_date, then passed after the async save/reset fix.

Exact shared UI dayjs package-import scan reports 5 remaining import lines in Media FilterPanel, ReadingList ReadingItemsList, and Items ItemsWorkspace; Kanban CardDetailPanel is no longer in the scan.

WebUI build verification: NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run compile from apps/tldw-frontend exited 0. The build compiled successfully, generated 138 static pages, and the shared-token sync check reported OK for the generated CSS.

bun run lint from apps/tldw-frontend exited 0 with the existing 131-warning baseline. git diff --check exited 0. Filtered apps/packages/ui TypeScript diagnostics for CardDetailPanel, BoardView, kanbanDateTime, and KanbanPlayground returned no matches.

Bandit skipped for this task because the touched scope is TypeScript, tests, documentation, and Backlog metadata only; no Python files changed.

Final review-fix verification: focused Kanban Vitest passed 2 files and 9 tests; git diff --check exited 0; exact dayjs import scan still reports only the 5 deferred Media/ReadingList/Items lines; bun run lint exited 0 with the existing 131-warning baseline; filtered TypeScript diagnostics for CardDetailPanel, BoardView, kanbanDateTime, and KanbanPlayground returned no matches. The WebUI compile was run after the code fixes and exited 0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the Kanban CardDetailPanel dependency on dayjs and Ant Design DatePicker for due-date editing. The panel now uses a native datetime-local input backed by small local ISO parse/format helpers, with an explicit touched flag so unchanged due dates are not rewritten when saving unrelated fields. PR review follow-up fixed the stale touched-state case by awaiting the parent update, resetting dueDateTouched only after save success, and updating the selected card from the returned mutation result. The Drawer keeps its 400px pixel width through the existing AntD 6 styles.wrapper.width pattern. Focused tests cover helper behavior plus changed, cleared, unchanged, and same-session post-due-date-save behavior. The dependency audit records the shared UI dayjs import count at 5 remaining lines, limited to Media, ReadingList, and Items surfaces; no manifest/lockfile reduction is expected until those remaining surfaces are migrated. Verification passed with the focused Kanban Vitest suite, WebUI compile, exact dayjs import scan, git diff --check, lint exiting 0 with the known warning baseline, and filtered touched-slice TypeScript diagnostics returning no matches. Bandit was skipped because this slice changed no Python files.
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
