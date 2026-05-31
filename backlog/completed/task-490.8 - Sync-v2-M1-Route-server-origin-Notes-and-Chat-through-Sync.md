---
id: TASK-490.8
title: 'Sync v2 M1: Route server-origin Notes and Chat through Sync'
status: Done
assignee:
- '@Codex'
labels:
- sync
- sync-v2
- m1
- server-frontend
- backend
priority: high
parent_task_id: TASK-490
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
modified_files:
- tldw_Server_API/app/core/Sync/v2/server_origin.py
- tldw_Server_API/app/api/v1/endpoints/notes.py
- tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py
- tldw_Server_API/app/api/v1/endpoints/character_messages.py
- tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Route personal Notes and Chat mutations made through normal server APIs through Sync v2 when Sync is active so server-front-end writes are represented in the append-only envelope log before materialized projections exist.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Personal Notes and Chat server API writes create server-origin envelopes before projection writes occur.
- [x] #2 Envelope append failures prevent or roll back the normal API mutation so projections cannot exist without log entries.
- [x] #3 Materialization failures leave replayable failed envelopes and are visible in profile status.
- [x] #4 Offline-sync devices can pull server-origin envelopes by cursor/domain filter.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md#task-8-route-server-origin-notes-and-chat-changes-through-sync
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `tldw_Server_API/app/core/Sync/v2/server_origin.py` to append trusted server-origin envelopes, derive base state from Sync object state, materialize through the existing materializer path, and surface append/materialization failures without logging private content.
- Routed active personal Notes create/update/patch/delete, Chat conversation create/update/delete, and Chat message create/delete through Sync v2. Non-bootstrapped/inactive Sync keeps direct behavior.
- Kept workspace-scoped chat direct-write for M3 and rejected active-Sync binary chat message attachments because M1 excludes blobs.
- Preserved authenticated-user projection ownership for server-origin chat materialization.
- Review fix: server-origin note materialization now preserves authenticated user ownership, active-Sync note keyword mutations are rejected before direct keyword projection writes, chat deletion tombstones active child messages through Sync before the conversation tombstone, and accepted materialization conflicts are reported as accepted-not-applied envelopes rather than append failures.
- Re-review fix: active-Sync server-origin create/post paths now support `Idempotency-Key` by deriving deterministic stable keys, envelope IDs, and object IDs; conflicting key reuse returns `sync_server_origin_idempotency_conflict` without appending duplicate envelopes.
- Re-review fix: active-Sync Notes import and bulk create route through server-origin Sync capture, while keyword-bearing import/bulk/link/unlink mutations are rejected with `sync_v2_keywords_not_supported` before direct keyword writes.
- Re-review fix: active-Sync chat completion persist flows and message edits that M1 cannot represent are rejected before direct projection writes; workspace/direct inactive behavior remains unchanged.
- RED: `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py -q` initially failed 5 tests and passed 11, covering idempotency, Notes import/bulk/keyword bypasses, chat/message idempotency, chat completion persist, and message edit.
- Verification: `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py -q` passed 16 tests; `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -q` passed 29 tests; `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_store.py tldw_Server_API/tests/Sync/test_sync_v2_notes_materializer.py tldw_Server_API/tests/Sync/test_sync_v2_chat_materializer.py -q` passed 127 tests; Bandit report `/private/tmp/bandit_task490_8_rereview.json` has zero findings.
- Self-review fix: active-Sync `chat.message` idempotent append replay now ignores timestamp-only payload drift while preserving the original accepted envelope payload and still conflicting on changed content, sender, or conversation.
- Self-review fix: inactive/non-bootstrapped Notes create no longer derives server-origin deterministic IDs from `Idempotency-Key`; direct behavior stays UUID-based unless Sync is active.
- RED: `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py -q` failed 2 tests and passed 16 for timestamp-only chat message replay and inactive Notes `Idempotency-Key` ID generation.
- Verification: `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py -q` passed 18 tests; `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -q` passed 31 tests; `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_store.py tldw_Server_API/tests/Sync/test_sync_v2_notes_materializer.py tldw_Server_API/tests/Sync/test_sync_v2_chat_materializer.py -q` passed 127 tests; Bandit report `/private/tmp/bandit_task490_8_self_review.json` has zero findings.
- Quality re-review fix: active-Sync note restore is now rejected with `sync_v2_note_restore_not_supported` before direct projection writes; inactive Sync restore keeps existing direct behavior.
- RED: `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py -q` failed 1 test and passed 19 because active-Sync restore returned 200 and mutated the deleted projection.
- Verification: `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py -q` passed 20 tests; `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -q` passed 33 tests; `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_store.py tldw_Server_API/tests/Sync/test_sync_v2_notes_materializer.py tldw_Server_API/tests/Sync/test_sync_v2_chat_materializer.py -q` passed 127 tests; Bandit report `/private/tmp/bandit_task490_8_restore.json` has zero findings.
- Quality re-review fix: active-Sync chat conversation restore is now rejected with `sync_v2_chat_restore_not_supported` before direct projection writes; inactive Sync and workspace restore keep existing direct behavior, and already-active restore remains idempotent.
- RED: `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py -q --tb=short` failed 1 test and passed 21 because active-Sync chat restore returned 200 instead of the unsupported Sync v2 error.
- Verification: `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py -q` passed 22 tests; `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -q` passed 35 tests; `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_store.py tldw_Server_API/tests/Sync/test_sync_v2_notes_materializer.py tldw_Server_API/tests/Sync/test_sync_v2_chat_materializer.py -q` passed 127 tests; Bandit report `/private/tmp/bandit_task490_8_chat_restore.json` has zero findings.
- Quality re-review fix: active-Sync chat conversation hard-delete is now rejected with `sync_v2_chat_hard_delete_not_supported` after the existing trash/version preconditions and before direct projection removal; inactive Sync and workspace hard-delete keep existing direct behavior.
- RED: `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py -q --tb=short` failed 1 test and passed 23 because active-Sync chat hard-delete returned 204 and removed the deleted projection.
- Verification: `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py -q` passed 24 tests; `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_server_origin_capture.py tldw_Server_API/tests/Sync/test_sync_v2_endpoints.py -q` passed 37 tests; `python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_store.py tldw_Server_API/tests/Sync/test_sync_v2_notes_materializer.py tldw_Server_API/tests/Sync/test_sync_v2_chat_materializer.py -q` passed 127 tests; Bandit report `/private/tmp/bandit_task490_8_chat_hard_delete.json` has zero findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Server-origin Notes and Chat API mutations now create Sync v2 envelopes before projection writes when personal Sync is active, with idempotent active-Sync create/post retries and clear rejection for M1-unsupported keyword, note/chat restore, chat hard-delete, chat completion persist, and message edit mutations. Inactive/non-bootstrapped Sync and workspace chat paths keep direct behavior; append failures block projection writes, materialization failures remain replayable and visible through profile status, and offline devices can pull server-origin envelopes by cursor/domain filter.
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
