---
id: TASK-490.12
title: 'Sync v2 M2: Restore completeness and blobs'
status: In Progress
assignee:
- '@Codex'
labels:
- sync
- sync-v2
- m2
- roadmap
- attachments
priority: medium
parent_task_id: TASK-490
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
- Docs/Design/Sync_V2_M2_Restore_Completeness_and_Blobs.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m2-restore-completeness-blobs-implementation-plan.md
modified_files:
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

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `Docs/Design/Sync_V2_M2_Restore_Completeness_and_Blobs.md` to lock M2 design decisions after M1 stabilized.
- Added `Docs/superpowers/plans/2026-05-23-sync-v2-m2-restore-completeness-blobs-implementation-plan.md` with five implementation stages and test-first steps.
- Created child tasks:
  - `TASK-490.12.1` protocol models and capabilities.
  - `TASK-490.12.2` blob storage ledger and local blob store.
  - `TASK-490.12.3` resumable upload API.
  - `TASK-490.12.4` download manifests and chunk serving.
  - `TASK-490.12.5` restore completeness and selective restore status.
  - `TASK-490.12.6` key recovery hardening.
  - `TASK-490.12.7` E2E docs and final verification.
- Parent task remains `In Progress` as the M2 tracking epic while implementation children are open.
- Verification: `git diff --check` passed; `rg` verified the design, plan, and parent task contain the M2 server-trusted/resumable restore decisions and child task references; Backlog MCP lists all seven child tasks under `TASK-490.12`.
- Bandit: skipped for this planning pass because only Markdown/Backlog files were changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
M2 planning is refined and split into implementation child tasks. The M2 design locks personal attachment.ref blob transfer, resumable upload/download, quota and checksum policy, server-derived blob availability, profile-level restore completeness, selective restore controls, and server-unlocked key recovery hardening. Implementation is tracked by TASK-490.12.1 through TASK-490.12.7.
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
