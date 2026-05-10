---
id: TASK-224
title: Add Sync v2 server domain adapters
status: Done
assignee: []
created_date: '2026-05-10 06:23'
updated_date: '2026-05-10 06:36'
labels:
  - sync
  - adapters
  - server
dependencies:
  - TASK-222
references:
  - tldw_Server_API/app/core/Sync/v2/adapters.py
  - tldw_Server_API/app/core/Sync/v2/domain_adapters
  - tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
  - tldw_Server_API/tests/ChaChaNotesDB/test_chacha_note_store.py
  - tldw_Server_API/tests/ChaChaNotesDB/test_chacha_message_store.py
  - tldw_Server_API/tests/ChaChaNotesDB/test_workspace_sub_resources_db.py
documentation:
  - >-
    Docs/superpowers/plans/2026-05-10-chatbook-sync-engine-implementation-plan.md
  - Docs/superpowers/specs/2026-05-10-chatbook-sync-engine-prd-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 6 from the Chatbook sync engine implementation plan: add first-pass Sync v2 domain adapters for notes, chat, workspaces, and source cache using the current adapter registry/service shape.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Adds concrete notes, chat, workspace, and source-cache Sync v2 domain adapters registered through the existing adapter registry.
- [x] #2 Covers PRD merge/conflict rules for note metadata versus encrypted content, chat message append/id conflicts, workspace source membership, source cache content-hash coexistence, and delete-vs-update conflicts.
- [x] #3 Keeps the current SyncDomainAdapter protocol compatible with the existing service unless a narrowly justified extension is required.
- [x] #4 Keeps relevant ChaChaNotesDB tests passing or documents intentional non-applicability.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started implementation in worktree .worktrees/codex-sync-v2-schemas on branch codex/sync-v2-schemas. Current architecture requires a narrow adapter context plus store query for prior accepted envelopes; no ChaCha DB apply methods planned for this slice.

Implemented concrete notes, chat, workspaces, and source_cache Sync v2 adapters. Added SyncAdapterContext and a focused accepted-envelope lookup by entity_id/stable_key so adapters can compare prior accepted envelopes without applying domain data to ChaChaNotesDB. Verification passed: Sync adapter/service/endpoint tests 59 passed; ChaCha note/message/workspace tests 37 passed; Bandit on touched production files exited 0; git diff --check exited 0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added first-pass Sync v2 server domain adapters for notes, chat, workspaces, and source_cache, registered them in the default Sync v2 endpoint registry, and kept media on MediaCompatibilityAdapter. Extended the adapter protocol with an optional SyncAdapterContext backed by a focused accepted-envelope lookup, allowing envelope-level merge/conflict decisions without broad ChaCha DB apply methods. Added deterministic adapter and service-level conflict tests covering metadata merges, encrypted-content conflicts, append-only chat messages, source-ref membership, source cache coexistence, and delete-vs-update conflicts. No known skips or blockers.
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
