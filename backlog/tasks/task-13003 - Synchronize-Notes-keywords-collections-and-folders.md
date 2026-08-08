---
id: TASK-13003
title: Synchronize Notes keywords collections and folders
status: In Progress
assignee: []
created_date: '2026-08-08 20:21'
updated_date: '2026-08-08 22:58'
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
