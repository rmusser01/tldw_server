---
id: TASK-13003
title: Synchronize Notes keywords collections and folders
status: In Progress
assignee: []
created_date: '2026-08-08 20:21'
updated_date: '2026-08-09 01:13'
labels:
  - notes
  - sync-v2
  - parity
  - organization
dependencies:
  - TASK-13002
references:
  - Docs/ADR/031-notes-capability-sync-domains.md
  - Docs/ADR/032-durable-server-origin-sync-mutation-batches.md
  - >-
    Docs/ADR/033-canonical-folder-link-suppression-preserves-source-provenance.md
documentation:
  - Docs/superpowers/specs/2026-08-08-notes-organization-sync-design.md
  - Docs/API/Sync_V2_M1.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add first-class Sync v2 ownership for Notes keywords, keyword links and collections, folders, and note-folder membership so organization state survives offline and multi-device use instead of remaining REST-only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Capabilities advertise versioned notes.keyword, notes.keyword_link, notes.keyword_collection, notes.keyword_collection_link, notes.folder, and notes.folder_link domains with upsert/tombstone operations.
- [ ] #2 Domain adapters and materializers preserve stable object identity, hierarchy, membership, optimistic base state, and user ownership on SQLite and PostgreSQL.
- [ ] #3 Server-origin keyword, collection, folder, and membership REST mutations capture canonical envelopes when Sync v2 is active.
- [ ] #4 Keyword writes no longer return the active-sync unsupported error only when all required organization domains are enabled for the dataset.
- [ ] #5 Concurrent rename, hierarchy, membership, merge, and delete/update cases produce deterministic idempotent results or reviewable conflicts.
- [ ] #6 Restore preview, repair, and capability documentation include every organization domain.
- [ ] #7 Existing personal datasets bootstrap their current organization state before the six-domain group becomes write-ready, and interrupted bootstrap resumes without partial enrollment or data loss.
- [ ] #8 Upgrading one device does not deliver unsupported organization domains to legacy devices whose registered requested domains exclude them.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Detailed execution plan: Docs/superpowers/plans/2026-08-08-notes-organization-sync-implementation-plan.md

1. Define the six-domain public contract, strict payload schemas, and deterministic identities.
2. Migrate ChaChaNotes to schema v55 and add stable resource identities plus a focused projection seam, including canonical folder-link suppression that preserves local source provenance.
3. Add atomic mutation-group persistence to the Sync store.
4. Add batch preflight, durable append, ordered materialization, and resume semantics.
5. Implement strict organization adapters and conflict policy.
6. Implement SQLite/PostgreSQL materializers and production factory registration.
7. Bootstrap existing datasets and isolate legacy-device implicit pulls.
8. Route direct organization REST mutations through the coordinator.
9. Make compound note writes, effective folder provenance, and keyword merge lossless.
10. Integrate restore/repair, update docs, and record focused release evidence.

ADR required: yes
ADR paths: Docs/ADR/032-durable-server-origin-sync-mutation-batches.md; Docs/ADR/033-canonical-folder-link-suppression-preserves-source-provenance.md
Reason: This task adds durable cross-database mutation groups and a local derived suppression projection so canonical folder-link absence converges without deleting source-ingestion provenance.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
- [ ] #7 Focused Notes organization and Sync v2 suites pass on supported database backends.
- [ ] #8 Bandit and static checks pass for touched production files.
- [ ] #9 Public schemas and examples contain no note content or secret-bearing fixtures.
<!-- DOD:END -->
