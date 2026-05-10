---
id: TASK-224
title: Add Sync v2 server domain adapters
status: Done
assignee: []
created_date: '2026-05-10 06:23'
updated_date: '2026-05-10 06:50'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add regression tests for linear note edits, stale note conflicts, linear delete-after-upsert, stale delete/update conflict behavior, and legacy adapter service compatibility.
2. Update domain adapter conflict helpers to compute the current accepted head from context and only treat content/delete divergence as concurrent when the incoming envelope is not based on that head.
3. Update Sync v2 service adapter invocation to detect whether an adapter accepts context or **kwargs and call legacy adapters without context.
4. Run focused Sync/ChaCha tests, Bandit on touched production files, git diff --check, update TASK-224, and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Review-fix pass started for Task 6 on branch codex/sync-v2-schemas. Blocking items: accept linear edits/deletes based on current head, preserve stale/concurrent conflicts, and keep old-shape SyncDomainAdapter implementations compatible with service context handling.

Review-fix pass completed for blocking Task 6 findings. Added head-version lineage handling so linear encrypted note edits and delete/update transitions based on the current accepted head are accepted, while stale/no-base divergent edits and deletes still conflict. Restored adapter protocol compatibility by keeping the protocol minimum at evaluate_envelope(envelope, *, dataset) and detecting context support in SyncV2Service before passing SyncAdapterContext. Verification: Sync adapter/service/endpoint tests 71 passed; ChaCha note/message/workspace tests 37 passed; Bandit on touched production files exited 0 with no findings; git diff --check exited 0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added first-pass Sync v2 server domain adapters for notes, chat, workspaces, and source_cache, registered them in the default Sync v2 endpoint registry, and kept media on MediaCompatibilityAdapter. Extended the adapter protocol with an optional SyncAdapterContext backed by a focused accepted-envelope lookup, allowing envelope-level merge/conflict decisions without broad ChaCha DB apply methods. Added deterministic adapter and service-level conflict tests covering metadata merges, encrypted-content conflicts, append-only chat messages, source-ref membership, source cache coexistence, and delete-vs-update conflicts. No known skips or blockers.

Review fixes: current-head lineage now distinguishes linear edits/deletes from stale concurrent changes across Sync v2 domain adapters, source-cache deletes bypass same-content payload mismatch checks after lineage acceptance, and SyncV2Service preserves old-shape adapter compatibility while still passing context to adapters that accept it.
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
