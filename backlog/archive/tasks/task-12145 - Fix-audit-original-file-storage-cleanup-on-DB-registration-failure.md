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
updated_date: 2026-07-10 06:10
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
- Tracking hygiene: moved this media cleanup audit record from colliding `TASK-12138` to `TASK-12145` because latest dev already contains other `TASK-12138` records.
- Review follow-up: widened the registration-failure cleanup path to catch ordinary `Exception` from `db.insert_media_file` and from `storage.delete`, while still re-raising the original registration error. This ensures cleanup is attempted for non-tuple database/library exceptions and cleanup failures do not mask the original registration failure.
- Final lint note: full-file Ruff still reports pre-existing import/type/name issues in the touched files; the new broad cleanup catches are covered by targeted `# noqa: BLE001` comments and `.venv/bin/python -m ruff check --select BLE001 tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_original_storage.py` passed before this current-dev refresh.
- Current-dev refresh: rebased `codex/audit-media-original-cleanup-2026-07-04` onto `origin/dev` `09d9ec901e1d4548f7924f1c6bcefa963fadd9bd`; merge-base matches `origin/dev`.
- Current-dev validation: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_original_storage.py -q` passed with 16 tests; `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py -f json -o /tmp/bandit_media_original_cleanup_origin_dev_09d9ec.json` reported 0 findings over 5627 LOC; `git diff --check HEAD~1..HEAD` passed.
2026-07-04 latest-dev refresh: rebased and validated PR #2612 on origin/dev 6b727b221e55646eba663a03571e38302f7fafc2. Tested head ab46b4e66ba1. Verification: python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_original_storage.py -q => 16 passed, 41 warnings; bandit -r tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py => 0 findings over 5627 LOC; git diff --check HEAD~1..HEAD => clean.
2026-07-09: origin/dev now contains a different active TASK-12145. This media cleanup record is superseded by unique TASK-12947 and archived before the current-dev rebase.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened media original-file cleanup behavior and storage edge-case coverage. Final refresh validated against origin/dev 6b727b221e55646eba663a03571e38302f7fafc2 with focused tests passing, Bandit clean on touched production scope, and whitespace check clean.
Superseded by TASK-12947 after the latest-dev refresh exposed an active task ID collision.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused original storage tests pass.
- [x] #2 Bandit runs clean over touched production code.
- [x] #3 git diff --check passes.
- [x] #4 AUDIT-2026-06-27-MEDIA-003 closure evidence is recorded in task notes.
<!-- DOD:END -->
