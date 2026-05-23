---
id: TASK-490.13.5
title: 'Sync v2 M3: Broader domain expansion'
status: To Do
labels:
- sync
- sync-v2
- m3
- domains
priority: medium
parent_task_id: TASK-490.13
documentation:
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/API/Sync_V2_M3.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expand Sync v2 domain coverage beyond personal Notes/Chat/attachment refs in reviewed tiers, starting with source cache and media metadata before derived content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each newly enabled domain defines stable identity, conflict rules, tombstones, projection ownership, restore inventory mapping, and redaction policy.
- [ ] #2 Source cache and media metadata sync land before derived content domains such as transcripts, summaries, embeddings, or evaluation artifacts.
- [ ] #3 Domain adapter, materializer, restore, replay/repair, and isolation tests cover each domain family.
<!-- AC:END -->

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
