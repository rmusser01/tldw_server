---
id: TASK-224
title: Add Sync v2 server domain adapters
status: Done
assignee: []
created_date: '2026-05-10 06:23'
updated_date: '2026-05-10 07:00'
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
1. Keep the existing Sync v2 domain adapters registered for notes, chat, workspaces, source_cache, and media compatibility.
2. Preserve service compatibility with legacy adapters while passing context to adapters that support it.
3. For Task 6 review fixes, use accepted-envelope context for lineage-based delete/update and content conflict decisions.
4. For the final quality pass, make note encrypted-content conflict checks compare against the latest prior content-bearing note head instead of metadata-only heads, and require entity_id/stable_key identity for version-token dependency matches while preserving direct envelope/server-sequence references.
5. Verify with focused Sync tests, focused ChaChaNotesDB tests, Bandit on touched production files, and git diff --check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Review-fix pass started for Task 6 on branch codex/sync-v2-schemas. Blocking items: accept linear edits/deletes based on current head, preserve stale/concurrent conflicts, and keep old-shape SyncDomainAdapter implementations compatible with service context handling.

Review-fix pass completed for blocking Task 6 findings. Added head-version lineage handling so linear encrypted note edits and delete/update transitions based on the current accepted head are accepted, while stale/no-base divergent edits and deletes still conflict. Restored adapter protocol compatibility by keeping the protocol minimum at evaluate_envelope(envelope, *, dataset) and detecting context support in SyncV2Service before passing SyncAdapterContext. Verification: Sync adapter/service/endpoint tests 71 passed; ChaCha note/message/workspace tests 37 passed; Bandit on touched production files exited 0 with no findings; git diff --check exited 0.

Reopened for remaining Task 6 quality findings on codex/sync-v2-schemas. Plan: add regression tests for note content lineage across metadata-only heads and dependency matching without entity identity; verify they fail; patch note content conflict handling to compare encrypted content edits against the latest prior content-bearing head while preserving metadata-safe merge behavior; tighten version-token dependency matching so it requires entity_id/stable_key while keeping direct identifiers standalone; run focused Sync/ChaCha pytest suites, Bandit on touched production files, git diff --check, update this task, and commit.

Final quality pass completed. Added note regressions for content v1 plus metadata-only v2 followed by content v2 based on content v1, and for stale content after a metadata-only head. Added lineage helper coverage proving version-only dependencies do not reference the head, entity_id/stable_key plus version do, and direct server_sequence/client_envelope_id/envelope_id/base_envelope_id references still stand alone. Updated NotesDomainAdapter to compare encrypted content edits with the latest content-bearing note head, and tightened dependency entity matching for version tokens. Verification: Sync adapter/service/endpoint tests 75 passed; ChaCha note/message/workspace tests 37 passed; Bandit on touched production files exited 0 with no findings; git diff --check exited 0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added first-pass Sync v2 server domain adapters for notes, chat, workspaces, and source_cache, registered them in the default Sync v2 endpoint registry, and kept media on MediaCompatibilityAdapter. Extended the adapter protocol with an optional SyncAdapterContext backed by a focused accepted-envelope lookup, allowing envelope-level merge/conflict decisions without broad ChaCha DB apply methods. Added deterministic adapter and service-level conflict tests covering metadata merges, encrypted-content conflicts, append-only chat messages, source-ref membership, source cache coexistence, and delete-vs-update conflicts. No known skips or blockers.

Review fixes: current-head lineage now distinguishes linear edits/deletes from stale concurrent changes across Sync v2 domain adapters, source-cache deletes bypass same-content payload mismatch checks after lineage acceptance, and SyncV2Service preserves old-shape adapter compatibility while still passing context to adapters that accept it.

Final quality pass: separated note content lineage from metadata-only note heads, so metadata tag/status heads remain mergeable while encrypted content edits are checked against the latest content-bearing note head. Tightened Sync v2 lineage dependency matching so version-token references require matching entity_id or stable_key, while direct envelope/server-sequence references still work without entity identity. Added regression coverage for both behaviors and re-ran the required focused verification.
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
