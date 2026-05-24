---
id: TASK-490.13.6
title: 'Sync v2 M3: Stricter encryption and key rotation'
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-24 00:29'
labels:
  - sync
  - sync-v2
  - m3
  - encryption
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
Add stricter dataset encryption policies and key rotation workflows while preserving the existing server_trusted_v1 mode and honest server-front-end limitations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dataset policies validate and advertise server_trusted_v1, passphrase_wrapped_v1, device_wrapped_v1, and client_private_v1 capabilities honestly.
- [x] #2 Key epochs, rewrap status, rotation preview/commit, and revoked/superseded key rejection are implemented without exposing key material.
- [x] #3 Server-front-end editing is blocked or limited for opaque client-private fields that the server cannot materialize.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed by encryption and key-rotation subtasks TASK-490.13.13 through TASK-490.13.17. Policy metadata, key epoch storage, rotation preview/commit APIs, review hardening, and client-private server-front-end mutation gates are implemented and documented.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the Sync v2 M3 stricter encryption and key rotation stage. Dataset policies now validate and advertise server_trusted_v1, passphrase_wrapped_v1, device_wrapped_v1, and client_private_v1; key epochs and rotation APIs track rewrap status without exposing key material; revoked/superseded keys are rejected for new envelopes; and client-private server-front-end writes fail closed with an explicit 409 contract. Verification is recorded on the implementation subtasks, including the full Sync suite and Bandit for touched production code.
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
