---
id: TASK-490.12
title: 'Sync v2 M2: Restore completeness and blobs'
status: Done
assignee:
- '@Codex'
created_date: ''
updated_date: 2026-05-23 18:40
labels:
- sync
- sync-v2
- m2
- roadmap
- attachments
dependencies: []
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
- Docs/Design/Sync_V2_M2_Restore_Completeness_and_Blobs.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m2-restore-completeness-blobs-implementation-plan.md
- Docs/API/Sync_V2_M2.md
parent_task_id: TASK-490
priority: medium
modified_files:
- Docs/API/Sync_V2_M2.md
- Docs/Design/Sync_V2_M2_Restore_Completeness_and_Blobs.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m2-restore-completeness-blobs-implementation-plan.md
- backlog/tasks/task-490.12 - Sync-v2-M2-Restore-completeness-and-blobs.md
- backlog/tasks/task-490.12.1 - Sync-v2-M2-Protocol-models-and-capabilities.md
- backlog/tasks/task-490.12.2 - Sync-v2-M2-Blob-storage-ledger-and-local-blob-store.md
- backlog/tasks/task-490.12.3 - Sync-v2-M2-Resumable-upload-API.md
- backlog/tasks/task-490.12.4 - Sync-v2-M2-Download-manifests-and-chunk-serving.md
- backlog/tasks/task-490.12.5 - Sync-v2-M2-Restore-completeness-and-selective-restore-status.md
- backlog/tasks/task-490.12.6 - Sync-v2-M2-Key-recovery-hardening.md
- backlog/tasks/task-490.12.7 - Sync-v2-M2-E2E-docs-and-final-verification.md
- tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Roadmap epic for Milestone 2 after M1 lands: attachment/blob transfer, restore completeness, resumable/chunked upload/download, quotas, checksums, availability status, selective restore controls, and server-unlocked key recovery hardening.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 M2 requirements are refined after M1 server contract stabilizes.
- [x] #2 Blob transfer design covers chunking/resume, quotas, checksums, and availability.
- [x] #3 Restore completeness criteria define when a new-device restore is complete versus metadata-only.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added Docs/Design/Sync_V2_M2_Restore_Completeness_and_Blobs.md to lock M2 design decisions after M1 stabilized.
- Added Docs/superpowers/plans/2026-05-23-sync-v2-m2-restore-completeness-blobs-implementation-plan.md with staged test-first implementation tasks.
- Completed child tasks TASK-490.12.1 through TASK-490.12.7 covering protocol models/capabilities, blob ledger/storage, resumable upload, download manifests, restore completeness, key recovery hardening, API docs, and final e2e verification.
- Added Docs/API/Sync_V2_M2.md and final e2e coverage for uploaded blob restore completeness.
- Final verification for the M2 track passed: Sync suite 313 passed, restore e2e 5 passed, Ruff passed, Bandit reported 0 findings at /tmp/bandit_sync_v2_m2_final.json, and git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Sync v2 M2 restore completeness and blob support is implemented and verified. The track now covers M2 protocol/capability models, blob storage ledger and local blob store, resumable upload API, download manifests and byte serving, restore completeness/selective restore status, key recovery hardening, API documentation, and uploaded-blob restore e2e coverage. Final verification passed with the Sync suite at 313 passed/6 warnings, restore e2e at 5 passed, Ruff clean for touched e2e code, Bandit 0 findings at /tmp/bandit_sync_v2_m2_final.json, and git diff --check clean.
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
