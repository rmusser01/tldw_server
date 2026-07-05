---
id: TASK-12149
title: Fix DataTables job manager override isolation
status: Done
assignee: []
created_date: '2026-07-04 18:40'
updated_date: '2026-07-04 19:34'
labels:
  - tests
  - data-tables
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The DB-to-Jobs slice stops at `test_generate_job_create_error_marks_failed` with a 202 queued response because the test's top-level imported `get_job_manager` dependency key can become stale in the broad process. Focused execution passes, indicating an order-dependent reload/identity issue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 DataTables tests override the current endpoint module `get_job_manager` dependency key.
- [x] #2 Focused `test_generate_job_create_error_marks_failed` passes.
- [x] #3 The DB-to-Jobs slice progresses past the DataTables job-create-error blocker.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm focused pass and broad-process failure to identify stale dependency override identity.
2. Replace stale imported `get_job_manager` override keys with `data_tables_endpoint.get_job_manager` in the DataTables API tests.
3. Verify focused DataTables test and resume the DB-to-Jobs slice; run Bandit/diff checks for touched tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Replaced stale direct get_job_manager override keys with data_tables_endpoint.get_job_manager throughout the DataTables API tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the DataTables job-manager override isolation issue by using the current endpoint module dependency key for overrides and cleanup. Verification: focused touched-scope command passed (44 passed); DataTables directory passed (62 passed); Discord-to-Jobs slice passed (3247 passed, 156 skipped); git diff --check passed; Bandit on touched tests reported no findings.
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
