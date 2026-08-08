---
id: TASK-13004
title: Synchronize Notes relationships graph and restore lifecycle
status: To Do
assignee: []
created_date: '2026-08-08 20:23'
labels:
  - notes
  - sync-v2
  - parity
  - graph
dependencies:
  - TASK-13003
references:
  - >-
    https://github.com/rmusser01/tldw_chatbook/blob/dev/backlog/decisions/046-synchronized-database-notes-parity.md
documentation:
  - >-
    https://github.com/rmusser01/tldw_chatbook/blob/dev/Docs/Parity/2026-08-08-notes-server-capability-matrix.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add first-class Sync v2 ownership for explicit Notes relationships while keeping wikilink parsing, backlinks, and graph summaries deterministic projections, and complete the conflict-aware trash/restore lifecycle for synchronized notes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Capabilities advertise a versioned `notes.link` domain with upsert and tombstone operations.
- [ ] #2 Explicit relationship payloads preserve stable edge identity, source, target, type, label, properties, ownership, and optimistic base state across SQLite and PostgreSQL.
- [ ] #3 Wikilinks, backlinks, orphan reports, and graph summaries are rebuilt deterministically from synchronized notes and explicit links rather than synchronized as mutable duplicates.
- [ ] #4 Server-origin link mutations and note trash or restore mutations capture canonical envelopes when Sync v2 is active.
- [ ] #5 Delete-versus-update, restore-versus-recreate, and concurrent edge edits yield idempotent outcomes or reviewable conflicts with restore-preview evidence.
- [ ] #6 Graph and lifecycle endpoints remain paginated, bounded, and authorized for the selected dataset.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
- [ ] #7 Focused Notes relationship, graph, and lifecycle suites pass on supported database backends.
- [ ] #8 Bandit and static checks pass for touched production files.
- [ ] #9 Performance checks demonstrate bounded, paginated graph and orphan queries.
<!-- DOD:END -->
