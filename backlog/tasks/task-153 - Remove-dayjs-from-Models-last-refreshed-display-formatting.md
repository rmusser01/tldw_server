---
id: TASK-153
title: Remove dayjs from Models last-refreshed display formatting
status: Done
assignee: []
created_date: '2026-05-09 04:59'
updated_date: '2026-05-09 05:09'
labels:
  - webui
  - dependencies
  - cleanup
  - dayjs
dependencies:
  - TASK-144
  - TASK-149
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1346'
  - 'https://github.com/rmusser01/tldw_server/pull/1405'
documentation:
  - Docs/Design/WebUI_Dependency_Audit.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue issue #1346 by replacing the display-only dayjs usage in apps/packages/ui/src/components/Option/Models/index.tsx with native Date formatting. Keep dayjs declared for remaining Flashcards display formatting and Ant Design Dayjs value-contract surfaces, and update the dependency audit with the reduced import count.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused tests or source guards cover the Models last-refreshed formatting behavior without relying on dayjs.
- [x] #2 apps/packages/ui/src/components/Option/Models/index.tsx no longer imports dayjs or dayjs/plugin/relativeTime.
- [x] #3 The WebUI dependency audit records the reduced dayjs import count and notes remaining blockers explicitly.
- [x] #4 Focused Vitest, git diff hygiene, frontend lint, and Bandit skip/run rationale are recorded.
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Add focused Models formatting coverage and verify dependency removal with an explicit import scan.
2. Replace the last-refreshed formatter with a native Date helper.
3. Update the dependency audit import count and completed follow-up notes.
4. Run focused Vitest, import scans, lint/diff checks, and record Bandit rationale before commit/PR.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
TDD red check: the focused Models display utility test first failed because modelsDisplayUtils did not exist. Dependency removal was verified separately with an exact package-import scan rather than a filesystem-reading unit-test guard.

Implemented a native Date/getHours/getMinutes helper for the Models last-refreshed HH:mm label and wired Models/index.tsx to use it instead of dayjs.

Updated the WebUI dependency audit to record the dayjs import count dropping from 17 to 15 and to keep remaining Flashcards display formatting plus Ant Design Dayjs value contracts explicit.

Verification: bunx vitest run src/components/Option/Models/__tests__/modelsDisplayUtils.test.ts passed with 3 tests; bunx vitest run src/components/Option/Models/__tests__ passed with 2 files and 5 tests; git diff --check passed; exact dayjs package-import scan listed 15 remaining shared UI import lines; bun run lint in apps/tldw-frontend exited 0 with the existing 131 warnings baseline and no touched-file warnings; Bandit skipped because only TypeScript/test/docs/Backlog files changed.

Opened PR #1405 against dev for this Models dayjs cleanup slice: https://github.com/rmusser01/tldw_server/pull/1405
<!-- SECTION:NOTES:END -->

## Final Summary
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed dayjs from the Models last-refreshed display label by replacing dayjs formatting with a native Date helper, added focused formatting tests, and updated the dependency audit to show the remaining dayjs import count dropping from 17 to 15 while preserving the remaining Flashcards and Ant Design Dayjs blockers.
<!-- SECTION:FINAL_SUMMARY:END -->
