---
id: TASK-490.13.3
title: 'Sync v2 M3: Background sync policy and status'
status: To Do
labels:
- sync
- sync-v2
- m3
- background-sync
priority: medium
parent_task_id: TASK-490.13
documentation:
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/API/Sync_V2_M3.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add server-side policy, status, pause/resume intent, and advisory lease primitives for Chatbook-run background sync without replacing Sync v2 idempotent push/pull/blob APIs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Clients can fetch background sync policy hints and store pause/resume intent per dataset/device.
- [ ] #2 Advisory per-device sync leases prevent overlapping local workers without weakening idempotency guarantees.
- [ ] #3 Profile and per-domain background status reports last success, lag, conflicts, replayable failures, blob completeness, and quota pressure.
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
