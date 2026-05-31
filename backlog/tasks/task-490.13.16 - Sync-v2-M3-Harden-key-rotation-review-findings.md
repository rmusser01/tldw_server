---
id: TASK-490.13.16
title: 'Sync v2 M3: Harden key rotation review findings'
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-23 23:59'
labels: []
dependencies: []
documentation:
  - Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
  - Docs/API/Sync_V2_M3.md
  - >-
    Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
parent_task_id: TASK-490.13
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address code-review findings for the Sync v2 key rotation preview/commit slice: make rotation commit derive the epoch boundary atomically with key state changes, persist/replay the exact source-key manifest, scope rotation idempotency to user and dataset, and redact secret-bearing validation errors.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Regression tests cover changed or invalid source sets on idempotent replay.
- [x] #2 Regression tests cover multi-source idempotent replay from persisted rotation source metadata.
- [x] #3 Regression tests cover same client rotation_id reused on two datasets without key-record collisions.
- [x] #4 Regression tests cover redacted 422 responses for secret-bearing key rotation commit input.
- [x] #5 Rotation commit computes active_from_server_sequence and retained range in the same storage transaction that inserts the new key and supersedes sources.
- [x] #6 Existing Sync v2 key rotation tests continue to pass, with Ruff, Bandit, and diff checks clean.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented review fixes for Sync v2 key rotation:
- Commit now derives active_from_server_sequence, retained envelope range, source selection, new-key insert, and source superseding inside one storage transaction.
- PostgreSQL rotation commit locks sync_envelopes and sync_key_records during sequencing; SQLite continues to rely on BEGIN IMMEDIATE.
- Rotation key IDs are scoped by user_id, dataset_id, and rotation_id.
- Committed rotations persist a canonical source-key manifest for multi-source idempotent replay.
- The key-rotation commit endpoint parses secret-bearing input through a redacted validator before returning 422.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the code review findings for key rotation integrity and redaction. Added regressions for interleaved envelope acceptance, multi-source replay, changed/missing replay source sets, scoped rotation IDs across datasets, and secret-bearing 422 validation responses. Verification: targeted red/green regressions passed; affected Sync files passed (223 passed); full Sync suite passed (393 passed, 6 warnings); Ruff passed; Bandit report /tmp/bandit_sync_v2_m3_key_rotation_review_fixes.json has 0 results; git diff --check passed.
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
