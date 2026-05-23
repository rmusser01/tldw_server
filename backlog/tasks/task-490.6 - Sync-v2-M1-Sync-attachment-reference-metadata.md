---
id: TASK-490.6
title: 'Sync v2 M1: Sync attachment reference metadata'
status: To Do
assignee:
- '@Codex'
labels:
- sync
- sync-v2
- m1
- attachments
- backend
priority: high
parent_task_id: TASK-490
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement attachment.ref as metadata-only sync in M1, including validation, accepted envelope storage, pull visibility, idempotency/conflicts by payload_hash, and restore-preview missing-blob warnings while keeping binary blob transfer disabled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 attachment.ref envelopes validate required metadata and are accepted/pulled as an M1 domain.
- [ ] #2 Duplicate attachment refs are idempotent by payload_hash and conflicting hashes do not overwrite history.
- [ ] #3 Restore preview includes attachment ref summaries and missing blob warnings.
- [ ] #4 Blob upload/download paths return sync_blob_transfer_not_supported for M1.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md#task-6-implement-attachment-ref-metadata-domain
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
