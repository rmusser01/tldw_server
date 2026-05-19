---
id: TASK-149
title: Remove dayjs from WorldBooks last-modified relative formatting
status: Done
assignee: []
created_date: '2026-05-09 04:06'
updated_date: '2026-05-09 04:15'
labels:
  - webui
  - dependencies
  - cleanup
  - dayjs
  - worldbooks
dependencies:
  - TASK-147
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1346'
  - 'https://github.com/rmusser01/tldw_server/pull/1396'
  - 'https://github.com/rmusser01/tldw_server/pull/1398'
documentation:
  - Docs/Design/WebUI_Dependency_Audit.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue issue #1346 by replacing the display-only dayjs relative-time usage in apps/packages/ui/src/components/Option/WorldBooks/worldBookListUtils.ts with a native/local helper. Keep the existing timestamp parsing and UTC absolute display behavior unchanged, and leave dayjs declared for remaining Flashcards/Models display formatting and Ant Design Dayjs value-contract surfaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A focused guard test fails before implementation because worldBookListUtils imports dayjs, then passes after the dependency is removed from that utility.
- [x] #2 WorldBooks last-modified formatting preserves timestamp parsing, unknown-safe display, UTC absolute formatting, and representative relative-time labels around existing behavior.
- [x] #3 apps/packages/ui/src/components/Option/WorldBooks/worldBookListUtils.ts no longer imports dayjs or dayjs/plugin/relativeTime.
- [x] #4 The WebUI dependency audit records the reduced dayjs import count and notes that dayjs remains declared for other display formatting and DatePicker/Dayjs contracts.
- [x] #5 Focused tests and lint/format checks pass or blockers are documented; Bandit skip rationale is recorded if no Python files change.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the WorldBooks display-only dayjs reduction by replacing dayjs.from(...) usage in apps/packages/ui/src/components/Option/WorldBooks/worldBookListUtils.ts with a local relative-time helper. Added table-driven tests for dayjs-compatible representative labels, preserved UTC absolute formatting coverage, and added a source guard that failed before implementation and passes after removing dayjs imports.

Verification: bunx vitest run src/components/Option/WorldBooks/__tests__/worldBookListUtils.test.ts exited 0 from apps/packages/ui; git diff --check exited 0; bun run lint exited 0 from apps/tldw-frontend with existing unrelated warnings only; rg found no dayjs in worldBookListUtils.ts and 17 remaining shared UI dayjs import lines. Bandit skipped because no Python files changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed dayjs relative-time usage from the WorldBooks last-modified formatter, covered representative relative labels and the no-dayjs source guard with focused Vitest tests, and updated the dependency audit to record the active dayjs import count dropping from 19 to 17 while keeping the remaining Flashcards/Models and DatePicker/Dayjs contract blockers explicit.
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
