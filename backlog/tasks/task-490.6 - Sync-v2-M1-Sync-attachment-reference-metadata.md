---
id: TASK-490.6
title: 'Sync v2 M1: Sync attachment reference metadata'
status: Done
assignee:
- '@Codex'
created_date: ''
updated_date: 2026-05-23 11:03
labels:
- sync
- sync-v2
- m1
- attachments
- backend
dependencies: []
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
parent_task_id: TASK-490
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement attachment.ref as metadata-only sync in M1, including validation, accepted envelope storage, pull visibility, idempotency/conflicts by payload_hash, and restore-preview missing-blob warnings while keeping binary blob transfer disabled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 attachment.ref envelopes validate required metadata and are accepted/pulled as an M1 domain.
- [x] #2 Duplicate attachment refs are idempotent by payload_hash and conflicting hashes do not overwrite history.
- [x] #3 Restore preview includes attachment ref summaries and missing blob warnings.
- [x] #4 Blob upload/download paths return sync_blob_transfer_not_supported for M1.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md#task-6-implement-attachment-ref-metadata-domain
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented attachment.ref metadata-only sync for M1 and addressed spec/quality review blockers. Added RED/GREEN regressions for non-JSON blob upload returning sync_blob_transfer_not_supported before M2 schema validation, rejecting object_id/payload.attachment_id mismatches, preventing stale upserts from resurrecting tombstoned attachment refs, and excluding tombstoned refs from restore-preview live attachment/missing-blob warnings. Verification: attachment refs passed 17 tests after RED; combined Task 6/domain plus endpoint/model smoke passed 83 tests; Bandit on touched production paths returned zero findings; git diff --check passed.
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
