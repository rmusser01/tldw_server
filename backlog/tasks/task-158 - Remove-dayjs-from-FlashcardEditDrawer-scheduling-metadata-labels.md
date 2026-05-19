---
id: TASK-158
title: Remove dayjs from FlashcardEditDrawer scheduling metadata labels
status: Done
assignee: []
created_date: '2026-05-09 05:33'
updated_date: '2026-05-09 05:44'
labels:
  - webui
  - dependencies
  - cleanup
  - dayjs
  - flashcards
dependencies:
  - TASK-153
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1346'
  - 'https://github.com/rmusser01/tldw_server/pull/1411'
documentation:
  - Docs/Design/WebUI_Dependency_Audit.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue issue #1346 by replacing the display-only dayjs usage in FlashcardEditDrawer scheduling metadata labels with native date/time helpers. Scope is limited to the edit drawer's read-only due/last-reviewed labels and their existing tests; leave ReviewTab, ManageTab, and Ant Design Dayjs value-contract surfaces for separate slices.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 FlashcardEditDrawer no longer imports dayjs or dayjs/plugin/relativeTime for scheduling metadata display.
- [x] #2 Due-at and last-reviewed labels preserve the existing absolute YYYY-MM-DD HH:mm shape and a relative label for valid timestamps using native helpers.
- [x] #3 Existing FlashcardEditDrawer scheduling metadata coverage is updated to assert native-helper output without importing dayjs in the test.
- [x] #4 The WebUI dependency audit records the reduced shared UI dayjs import count and the remaining Flashcards/Ant Design surfaces.
- [x] #5 Focused Vitest, exact Flashcards dayjs import scan, frontend lint or focused lint rationale, git diff hygiene, and Bandit skip/run rationale are recorded.
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Add native date-formatting expectations to the existing FlashcardEditDrawer scheduling metadata test and verify the red failure while the helper does not exist.
2. Implement small Flashcards date display helpers for absolute YYYY-MM-DD HH:mm and relative due/last-reviewed labels.
3. Replace FlashcardEditDrawer dayjs usage and remove test-side dayjs imports.
4. Update the dependency audit and task notes with the new import count, verification, and remaining dayjs surfaces.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented native Flashcards date-display helpers for parsing timestamps, local YYYY-MM-DD HH:mm labels, and dayjs-compatible relative labels. Replaced FlashcardEditDrawer scheduling metadata dayjs usage with the helper and removed dayjs imports from the scheduling metadata test.

TDD red: focused Vitest failed on missing ../../utils/date-display and ../date-display imports before implementation. Green: bunx vitest run src/components/Flashcards/utils/__tests__/date-display.test.ts src/components/Flashcards/components/__tests__/FlashcardEditDrawer.scheduling-metadata.test.tsx passed with 2 files and 20 tests.

Verification: exact shared UI dayjs package-import scan now reports 11 remaining import lines; exact scan over Flashcards/components and Flashcards/utils returns no dayjs package imports. git diff --check passed. bun run lint in apps/tldw-frontend exited 0 with the existing 131-warning baseline. UI package tsc still exits 2 on the existing repo-wide baseline, but a filtered tsc diagnostic check for date-display and FlashcardEditDrawer.scheduling-metadata returned no touched-file diagnostics after adding review_prompt_side to the local test fixture. Bandit skipped because only TypeScript, documentation, and Backlog files changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed dayjs from FlashcardEditDrawer scheduling metadata labels by adding native Flashcards date-display helpers for local YYYY-MM-DD HH:mm and dayjs-compatible relative labels, rewiring the drawer to use them, updating scheduling metadata tests to avoid dayjs imports, and refreshing the WebUI dependency audit from 15 to 11 remaining shared UI dayjs package-import lines. Opened PR #1411 against dev.
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
