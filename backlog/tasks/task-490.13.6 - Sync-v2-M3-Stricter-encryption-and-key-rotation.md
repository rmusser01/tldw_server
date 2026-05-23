---
id: TASK-490.13.6
title: 'Sync v2 M3: Stricter encryption and key rotation'
status: To Do
labels:
- sync
- sync-v2
- m3
- encryption
priority: medium
parent_task_id: TASK-490.13
documentation:
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/API/Sync_V2_M3.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add stricter dataset encryption policies and key rotation workflows while preserving the existing server_trusted_v1 mode and honest server-front-end limitations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Dataset policies validate and advertise server_trusted_v1, passphrase_wrapped_v1, device_wrapped_v1, and client_private_v1 capabilities honestly.
- [ ] #2 Key epochs, rewrap status, rotation preview/commit, and revoked/superseded key rejection are implemented without exposing key material.
- [ ] #3 Server-front-end editing is blocked or limited for opaque client-private fields that the server cannot materialize.
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
