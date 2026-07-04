---
id: TASK-12145
title: Fix audit original file storage cleanup on DB registration failure
status: Done
created_date: 2026-07-04 18:38
labels:
- audit
- remediation
- media
- storage
priority: medium
references:
- AUDIT-2026-06-27-MEDIA-003
- https://github.com/rmusser01/tldw_server/pull/2612
documentation:
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/media-ingestion-storage.md
- Docs/superpowers/reviews/2026-06-27-repo-audit/remediation-backlog-draft.md
modified_files:
- tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_original_storage.py
updated_date: 2026-07-04 18:41
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remediate AUDIT-2026-06-27-MEDIA-003 by adding compensating cleanup when permanent original-file storage succeeds but MediaFiles row registration fails.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When original storage succeeds but db.insert_media_file raises, the stored permanent path is deleted through the storage backend.
- [x] #2 The affected media result reports original_file_stored as false and does not leave a retrievable original_file_path.
- [x] #3 Cleanup failure is logged without masking the original storage-registration failure handling.
- [x] #4 A focused unit test reproduces the DB registration failure and asserts the compensating delete.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added compensating cleanup after original-file storage succeeds but `db.insert_media_file` fails.
- The cleanup path calls `await storage.delete(storage_path)` before re-raising to the existing non-fatal storage failure handler, so the response still marks `original_file_stored` as false.
- Cleanup failures are logged for both exception and false-return storage backend behavior without masking the original registration failure.
- Added focused tests for registration failure cleanup and cleanup-failure logging.
- Verification: `python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_original_storage.py -q` passed with 15 tests; Bandit over `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py` reported 0 findings; `git diff --check` passed.
- 2026-07-04 latest-dev refresh: rebased `codex/audit-media-original-cleanup-2026-07-04` onto `origin/dev` `fd5c152b065c408e4e8ee5f08da41589f21cb7f5`; merge-base matches `origin/dev`.
- Tracking hygiene: moved this media cleanup audit record from colliding `TASK-12138` to `TASK-12145` because latest dev already contains other `TASK-12138` records.
- Latest-dev validation: `.venv/bin/python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_original_storage.py -q` passed with 15 tests; `.venv/bin/python -m bandit -r tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py -f json -o /tmp/bandit_media_original_cleanup_latest.json` reported 0 findings; `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed AUDIT-2026-06-27-MEDIA-003 by deleting permanently stored original-file blobs when MediaFiles registration fails after storage succeeds. The existing non-fatal ingestion behavior is preserved while preventing untracked permanent files, and tests now cover successful cleanup plus cleanup failure logging.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused original storage tests pass.
- [x] #2 Bandit runs clean over touched production code.
- [x] #3 git diff --check passes.
- [x] #4 AUDIT-2026-06-27-MEDIA-003 closure evidence is recorded in task notes.
<!-- DOD:END -->
