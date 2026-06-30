---
id: TASK-9926
title: Harden File Artifacts review findings 2026-06-23
status: Done
assignee: []
created_date: 2026-06-23 18:48
updated_date: 2026-06-25 03:54
labels:
- backend
- files
- review-hardening
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix current-code review findings in tldw_Server_API/app/core/File_Artifacts: duplicate-column JSON exports, caller-controlled limits, worker retry classification, XLSX sheet validation, URL export quota accounting, and non-finite image cfg_scale.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Duplicate data_table JSON export columns are rejected before lossy object serialization.
- [x] #2 Server-configured file artifact rows, cells, bytes, and export TTL caps cannot be raised by request options.
- [x] #3 Async file artifact worker retries transient export failures while keeping validation failures terminal.
- [x] #4 XLSX sheet names reject Excel-invalid characters.
- [x] #5 All URL export paths register generated files for quota/accounting.
- [x] #6 Image cfg_scale rejects NaN and infinities.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use focused regression tests, make scoped adapter/service/worker/storage fixes, run targeted pytest, compile touched production files, and run Bandit on touched scope.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented scoped File_Artifacts hardening for review findings: lossy duplicate-column JSON export rejection, invalid XLSX sheet-name validation, server-owned cap resolution for rows/cells/bytes/export TTL, retry-aware worker failure mapping, generated-file quota/accounting registration for non-image/non-spreadsheet URL exports, and finite cfg_scale validation.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verification in the PR worktree after rebase/comment fixes: py_compile passed for touched production files; focused pytest with --noconftest passed (13 passed, 2 warnings); Bandit touched scope completed with 0 findings in /tmp/bandit_file_artifacts_hardening_rebased.json. The branch was rebased onto origin/dev and PR review comments were addressed in the same fixup commit.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Rebased PR #2453 branch on origin/dev and addressed follow-up review comments: simplified cfg_scale validation issue handling, switched XLSX invalid-character detection to set isdisjoint, added immediate docstrings for modified helper methods, and propagated transient expiry metadata for image export registration with a regression assertion.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
