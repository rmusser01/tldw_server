---
id: TASK-12009
title: Harden Storage module review findings
status: Done
assignee: []
created_date: '2026-06-23 21:17'
updated_date: '2026-06-23 21:36'
labels:
  - backend
  - storage
  - review-hardening
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix current-code review findings in `tldw_Server_API/app/core/Storage`: filesystem path containment, quota failure behavior, generated-file quota preflight, atomic file writes, no-op cleanup, and Storage module cohesion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Filesystem storage rejects sibling-prefix path escapes for all path-based operations.
- [x] #2 Quota enforcement does not leak raw backend exception text and supports fail-closed behavior for managed/multi-user deployments.
- [x] #3 Generated-file helpers preflight quota/size checks before writing bytes to disk.
- [x] #4 Filesystem storage writes via same-directory temporary files and cleans up partial writes on failure.
- [x] #5 No-op generated-file helper code is removed.
- [x] #6 Backup schedule job helpers are moved out of `core/Storage` with a compatibility shim for existing imports.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use focused regression tests first, implement scoped Storage fixes, update imports/shim for backup schedule helpers, run targeted pytest, compile touched production files, and run Bandit on touched scope.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Work started after Backlog CLI task operations hung; user approved manual task-file fallback.

Implemented focused hardening for the review findings:
- Replaced filesystem path containment string-prefix checks with `Path.relative_to`.
- Switched filesystem writes to same-directory temporary files followed by atomic replace, with best-effort cleanup on failures.
- Closed default `retrieve_stream` file handles after streaming.
- Made quota repository failures fail closed by default, with sanitized error messages and explicit `STORAGE_QUOTA_FAIL_OPEN` opt-in.
- Added generated-file size/quota preflight before writing bytes.
- Removed the no-op TTS date-folder call.
- Moved backup schedule job helpers to `app/core/Admin_Backups` and left a Storage compatibility shim.

Verification:
- `python -m pytest tldw_Server_API/tests/Storage tldw_Server_API/tests/Image_Generation/test_reference_images.py tldw_Server_API/tests/Admin/test_admin_storage_quotas.py tldw_Server_API/tests/Admin/test_admin_backup_jobs.py tldw_Server_API/tests/Admin/test_admin_backup_scheduler.py -q` passed: 147 passed.
- `python -m py_compile` passed for touched production modules.
- `python -m bandit -r tldw_Server_API/app/core/Storage tldw_Server_API/app/core/Admin_Backups tldw_Server_API/app/services/admin_backup_jobs_worker.py tldw_Server_API/app/services/admin_backup_scheduler.py -f json -o /tmp/bandit_storage_hardening_12009.json` passed with 0 findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the Storage module review findings by fixing path containment, quota failure behavior, generated-file preflight checks, partial-write cleanup, stream handle cleanup, and backup schedule module ownership. Added regression coverage for the security and failure-mode cases and updated Storage README ownership notes.
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
