---
id: TASK-490.5
title: 'Sync v2 M1: Materialize Chat'
status: Done
assignee:
  - '@Codex'
created_date: ''
updated_date: '2026-05-23 10:37'
labels:
  - sync
  - sync-v2
  - m1
  - chat
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
Implement Chat domain materialization for chat.conversation metadata and chat.message append/tombstone behavior through DB_Management-owned ChaChaNotes helpers, including whole-object conversation conflicts and stable-message-ID dedupe/conflicts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 chat.conversation envelopes create/update/tombstone usable server chat metadata.
- [x] #2 chat.message envelopes append messages, dedupe same stable ID and payload_hash, and preserve both versions plus a conflict for same stable ID with different payload_hash.
- [x] #3 Message tombstones soft-delete messages without deleting conversations unless a conversation tombstone exists.
- [x] #4 Chat materializer and ChaChaNotes helper tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md#task-5-implement-chat-materialization
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added `ChatConversationMaterializer` and `ChatMessageMaterializer` using the Task 4 Notes apply-status/object-state pattern.
- Added ChaChaNotes-owned helpers for `upsert_conversation_from_sync`, `tombstone_conversation_from_sync`, `append_message_from_sync`, `tombstone_message_from_sync`, and stable-ID/include-deleted message fetches.
- Stored Sync v2 message identity metadata in `message_metadata.extra.sync_v2` so divergent stable message IDs can preserve both projection rows while keeping the stable ID and payload hash discoverable.
- Mapped portable conversation states such as `active` and `archived` into the existing local conversation state enum for server-usable metadata.
- Controller hardening after initial commit: divergent message conflict projection retries remain conflicts after partial apply-status failure, appends after message tombstones conflict without resurrecting object state, and message tombstones require matching base server cursor/revision/hash.
- Verification:
  - RED: focused Task 5 command initially failed during collection with missing `tldw_Server_API.app.core.Sync.v2.materializers.chat`.
  - RED: controller regression tests failed before fixes: `test_retry_after_failed_message_conflict_status_keeps_conflict` and `test_message_append_after_tombstone_conflicts_without_resurrecting_state` failed; `test_message_tombstone_requires_matching_base_state` failed with the tombstone accepted and the message deleted.
  - GREEN: focused Task 5 command passed, `25 passed, 5 warnings`.
  - Sync v2 smoke passed, `49 passed, 5 warnings`.
  - Bandit on touched production scope completed with 0 results.
- Known residual: a broad `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync -q` run from the initial implementation passed the new chat tests and other Sync v2 smoke suites but failed existing `test_sync_v2_service.py` checks outside this materializer slice.

- Review-fix regressions added after spec/quality rejection:
  - RED then GREEN: divergent stable message IDs conflict based on SyncObjectState even when existing message metadata is missing.
  - RED then GREEN: failed Sync v2 message metadata persistence leaves materialization replayable without duplicate rows.
  - RED then GREEN: message tombstones match the canonical base hash even when a conflict projection sorts first, then hide all projections for the stable message ID.
- Review-fix verification:
  - Targeted failed store test passed, `1 passed, 5 warnings`.
  - Focused Task 5 command passed, `28 passed, 5 warnings`.
  - Sync v2 smoke passed, `49 passed, 5 warnings`.
  - Bandit on touched production scope completed with 0 results and 0 errors.
  - `git diff --check` passed.

- Second quality review fix:
  - RED then GREEN: metadata-less physical-ID fallback no longer adopts an unrelated local ChaCha message that happens to share the Sync stable message ID.
  - `append_message_from_sync()` now backfills metadata-less fallback rows only when the stored projection fields and sync client match the incoming envelope, and chooses an unused conflict projection ID when a fallback row blocks the requested projection ID.
- Second review-fix verification:
  - Targeted false-adoption and metadata-replay regressions passed, `2 passed, 5 warnings`.
  - Focused Task 5 command passed, `29 passed, 5 warnings`.
  - Sync v2 smoke passed, `49 passed, 5 warnings`.
  - Bandit on touched production scope completed with 0 results and 0 errors.
  - `git diff --check` passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Sync v2 M1 chat materialization is implemented for `chat.conversation` and `chat.message`. Conversations project upsert/update/tombstone metadata into ChaChaNotes with whole-object conflict detection. Messages append into existing conversations, dedupe by stable ID plus payload hash, preserve divergent stable-ID versions as conflict projections, fail/replay safely when Sync v2 metadata persistence fails, refuse to adopt unrelated metadata-less local rows during physical-ID fallback replay, require matching base state for tombstones, and accepted tombstones hide all projections for the stable message ID without deleting the conversation. Review-fix regressions were verified red/green. Focused chat/ChaChaNotes tests pass (`29 passed`), Sync v2 smoke tests pass (`49 passed`), Bandit reported zero findings on touched production paths, and `git diff --check` passed. A broader Sync suite still has existing `test_sync_v2_service.py` failures unrelated to this chat materializer slice.
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
