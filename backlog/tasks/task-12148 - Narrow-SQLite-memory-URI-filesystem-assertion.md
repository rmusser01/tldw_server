---
id: TASK-12148
title: Narrow SQLite memory URI filesystem assertion
status: Done
assignee: []
created_date: '2026-07-04 18:36'
updated_date: '2026-07-04 19:33'
labels:
  - tests
  - db
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The remaining mid-slice stops at `test_sqlite_file_memory_uri_uses_memory` only in the broader process because unrelated lazy config loading can create a `Databases` directory under the test cwd. The backend behavior under test is that `file::memory:?cache=shared` does not create a SQLite database file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The SQLite memory URI test fails only if a filesystem file is created for the memory URI.
- [x] #2 Focused DB backend normalization test passes.
- [x] #3 The remaining mid-slice progresses past the DB normalization blocker.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the focused test passes and the broader run fails from unrelated `Databases` directory creation.
2. Narrow the assertion to check that no files are created under the temporary cwd, allowing unrelated directories.
3. Verify focused DB test and resume the remaining mid-slice; run Bandit/diff checks for touched test files.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Narrowed the SQLite memory URI assertion to look for created files, allowing unrelated directories produced by lazy config initialization in broad runs.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the DB backend normalization test by asserting that no filesystem files are created for file::memory:?cache=shared, rather than requiring an entirely empty temporary directory. Verification: focused touched-scope command passed (44 passed); Discord-to-Jobs slice passed (3247 passed, 156 skipped); git diff --check passed; Bandit on touched tests reported no findings.
<!-- SECTION:FINAL_SUMMARY:END -->

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
