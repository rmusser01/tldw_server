---
id: TASK-490.11
title: 'Sync v2 M1: Verify end to end and harden'
status: Done
assignee:
- '@Codex'
labels:
- sync
- sync-v2
- m1
- verification
priority: high
parent_task_id: TASK-490
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
modified_files:
- tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py
- tldw_Server_API/app/api/v1/endpoints/sync.py
- Docs/API/Sync_V2_M1.md
- Docs/Design/Sync_V2_M1_Implementation_Decisions.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
- backlog/tasks/task-490.11 - Sync-v2-M1-Verify-end-to-end-and-harden.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run final Sync v2 M1 verification and hardening, including two-device scenarios, server-front-end writes, restore previews, conflicts, tombstones, attachment refs, cross-user isolation, targeted tests, broader relevant tests, Bandit, and final documentation/backlog updates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 E2E scenario matrix covers two devices, server-origin writes, clean restore, non-empty conflicts, stable-message dedupe/conflicts, tombstones, attachment refs, and cross-user isolation.
- [x] #2 Targeted Sync and ChaChaNotes tests pass or documented pre-existing failures are recorded.
- [x] #3 Bandit runs on all touched production scope with no new findings.
- [x] #4 Backlog child tasks record touched files, verification, skips, and final summaries.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md#task-11-end-to-end-verification-and-hardening
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added final Sync v2 M1 E2E coverage for two-device pagination/echo suppression, server-origin Notes writes, clean restore, non-empty Note/conversation conflicts, stable chat message dedupe/conflict behavior, tombstones, attachment refs, and cross-user isolation.
- Removed unreachable legacy media sync processor code behind the `/api/v1/sync/send` and `/api/v1/sync/get` 410 replacement routes; the routes now only authenticate and return the stable replacement response.
- Updated API/design docs for repair status, `/sync/repair`, required `chat.message.sender` payloads, and verification/repair invariants.
- Verification:
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m ruff check tldw_Server_API/app/api/v1/endpoints/sync.py tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py` -> passed.
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Sync tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_conversation_store.py tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py tldw_Server_API/tests/e2e/test_chatbook_sync_v2_restore.py -q` -> 326 passed, 6 warnings.
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/ChaChaNotesDB tldw_Server_API/tests/e2e/test_chats_and_characters.py tldw_Server_API/tests/e2e/test_workspace_chat_scope.py -q` -> 312 passed, 5 skipped, 3 failed, 2 errors. Recorded as unrelated pre-existing broad-suite failures: the persona analytics test uses a fixed April 19, 2026 fixture outside the current 30-day query window on May 23, 2026; flashcard template tests monkeypatch read-only `CharactersRAGDB.backend_type` and fail during setup/teardown.
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r <touched production scope> -f json -o /tmp/bandit_sync_v2_m1.json` -> 0 findings.
  - `git diff --check` -> passed.
  - M1 contradiction scan -> reviewed remaining expected response `media_type` false positives, redaction-key names, and dormant future/compat adapters that are not in the M1 registry/capabilities.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 11 completed final Sync v2 M1 hardening. Targeted Sync/ChaChaNotes/E2E verification passes, Ruff passes on changed Python files, Bandit reports zero findings on touched production scope, and the broader relevant suite failures are recorded as unrelated pre-existing issues.
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
