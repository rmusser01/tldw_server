---
id: TASK-147
title: Remove dayjs from the WebUI humanizeMilliseconds utility
status: Done
assignee: []
created_date: '2026-05-09 03:47'
updated_date: '2026-05-09 03:55'
labels:
  - webui
  - dependencies
  - cleanup
  - dayjs
dependencies:
  - TASK-144
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1346'
  - 'https://github.com/rmusser01/tldw_server/pull/1395'
  - 'https://github.com/rmusser01/tldw_server/pull/1396'
documentation:
  - Docs/Design/WebUI_Dependency_Audit.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue GitHub issue #1346 by taking the first narrow dayjs reduction slice after TASK-144. Replace dayjs duration usage in the display-only humanizeMilliseconds utility with equivalent local arithmetic while leaving dayjs installed and declared for shared UI DatePicker/Dayjs value contracts. This should reduce active dayjs imports without changing the utility's public output shape.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A focused guard test fails before implementation because apps/packages/ui/src/utils/humanize-milliseconds.ts imports dayjs, then passes after the dependency is removed from that utility.
- [x] #2 humanizeMilliseconds preserves the existing display thresholds and output suffixes for millisecond, second, minute, hour, and day ranges.
- [x] #3 apps/packages/ui/src/utils/humanize-milliseconds.ts no longer imports dayjs or dayjs/plugin/duration.
- [x] #4 The WebUI dependency audit records the reduced dayjs import count and notes that dayjs remains declared for DatePicker/Dayjs contracts.
- [x] #5 Focused tests and lint/format checks for the touched scope pass, or blockers are documented with evidence; Bandit skip rationale is recorded if no Python files change.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the narrow dayjs reduction slice by replacing apps/packages/ui/src/utils/humanize-milliseconds.ts duration formatting with local millisecond arithmetic and adding src/utils/__tests__/humanize-milliseconds.test.ts. The test first failed on the dependency guard while dayjs imports remained, then passed after implementation.

Verification: bunx vitest run src/utils/__tests__/humanize-milliseconds.test.ts exited 0 from apps/packages/ui; git diff --check exited 0; bun run lint exited 0 from apps/tldw-frontend with existing unrelated warnings only; rg found no dayjs in humanize-milliseconds.ts and 19 remaining shared UI dayjs import lines. Bandit skipped because no Python files changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed dayjs duration usage from the display-only WebUI humanizeMilliseconds utility, covered the existing threshold behavior and no-dayjs source guard with a focused Vitest test, and updated the dependency audit to record the import count drop from 21 to 19 while keeping the remaining DatePicker/Dayjs contract blocker explicit.
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
