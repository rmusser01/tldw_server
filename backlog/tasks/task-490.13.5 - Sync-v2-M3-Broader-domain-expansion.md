---
id: TASK-490.13.5
title: 'Sync v2 M3: Broader domain expansion'
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-24 00:29'
labels:
  - sync
  - sync-v2
  - m3
  - domains
dependencies: []
documentation:
  - Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
  - Docs/API/Sync_V2_M3.md
  - >-
    Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
parent_task_id: TASK-490.13
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expand Sync v2 domain coverage beyond personal Notes/Chat/attachment refs in reviewed tiers, starting with source cache and media metadata before derived content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each newly enabled domain defines stable identity, conflict rules, tombstones, projection ownership, restore inventory mapping, and redaction policy.
- [x] #2 Source cache and media metadata sync land before derived content domains such as transcripts, summaries, embeddings, or evaluation artifacts.
- [x] #3 Domain adapter, materializer, restore, replay/repair, and isolation tests cover each domain family.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed by reviewed domain-expansion subtasks TASK-490.13.9 through TASK-490.13.12. Source cache and media metadata domains now define stable identity, conflict/tombstone behavior, projection ownership, restore inventory, and redaction policy; derived content domains were reassessed and deferred unless promoted as source-of-truth later.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the Sync v2 M3 broader domain expansion stage. Source cache and media metadata domain families were promoted with adapter/materializer/restore/replay coverage, while transcripts, summaries, embeddings, and evaluation artifacts were explicitly classified as derived or deferred. Verification is recorded on the domain-family subtasks and Stage 5 is marked complete in the M3 implementation plan.
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
