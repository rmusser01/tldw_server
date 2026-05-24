---
id: TASK-490.13.17
title: 'Sync v2 M3: Gate client-private server-front-end mutations'
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-24 00:10
labels: []
dependencies: []
documentation:
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/API/Sync_V2_M3.md
parent_task_id: TASK-490.13
modified_files:
- Docs/API/Sync_V2_M3.md
- Docs/Design/Sync_V2_M3_Polished_Multi_Device.md
- Docs/superpowers/plans/2026-05-23-sync-v2-m3-polished-multi-device-implementation-plan.md
- tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py
- tldw_Server_API/app/api/v1/endpoints/character_messages.py
- tldw_Server_API/app/api/v1/endpoints/notes.py
- tldw_Server_API/app/api/v1/schemas/sync_v2_models.py
- tldw_Server_API/app/core/Sync/v2/models.py
- tldw_Server_API/app/core/Sync/v2/profile.py
- tldw_Server_API/app/core/Sync/v2/server_origin.py
- tldw_Server_API/app/core/Sync/v2/service.py
- tldw_Server_API/tests/Sync/test_sync_v2_profile_bootstrap.py
- tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py
- tldw_Server_API/tests/Sync/test_sync_v2_service.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 6 Step 4 from the Sync v2 M3 plan: prevent server-origin/server-front-end mutation paths from creating or materializing clear payloads for client_private_v1 personal datasets, while advertising the server-front-end limitation in status/capability responses.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Server-origin Notes/Chat capture refuses to mutate client_private_v1 datasets before appending a clear server-trusted envelope.
- [x] #2 Client-private profile/status or capabilities responses expose that server-front-end mutation is disabled for opaque fields.
- [x] #3 Server-trusted/passphrase/device-wrapped dataset behavior remains unchanged.
- [x] #4 Focused service/API tests cover the gate and advertised limitation.
- [x] #5 Targeted Sync tests, Ruff, Bandit, and git diff checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Stage 6 Step 4 for client-private server-front-end limitations. Added a shared blocker code/message, profile dataset/domain mutation flags, capabilities warning/compatibility flag, and a server-origin fail-closed exception that fires before envelope append or materialization. Notes and Chat endpoint error mappers now return 409 with the stable blocker code. Updated M3 API/design docs and roadmap status.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed the client-private server-front-end mutation gate for Sync v2 M3. Server-origin writes now refuse client_private_v1 datasets before appending clear server-trusted envelopes; profile/domain status and capabilities advertise the limitation; tests cover direct capture, normal Notes API mapping, profile status, and capabilities warnings. Verification: new red tests failed first for missing symbols, then passed; affected Sync service/profile/server-origin tests passed with 124 passed; full Sync suite passed with 397 passed and 6 warnings; Ruff passed for Sync core/test files and E9/F821 endpoint checks; Bandit report /tmp/bandit_sync_v2_m3_server_frontend_gate.json has 0 results; git diff --check passed.
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
