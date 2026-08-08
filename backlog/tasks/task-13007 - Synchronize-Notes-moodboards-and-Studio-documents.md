---
id: TASK-13007
title: Synchronize Notes moodboards and Studio documents
status: To Do
assignee: []
created_date: '2026-08-08 20:25'
labels:
  - notes
  - sync-v2
  - parity
  - moodboards
  - studio
dependencies:
  - TASK-13006
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
Synchronize Notes moodboards, note placement, and persisted Studio document state so visual organization and accepted AI-assisted outputs survive offline and multi-device use without synchronizing transient generation requests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Capabilities advertise versioned notes.moodboard notes.moodboard_note and notes.studio_document domains with supported upsert and tombstone operations.
- [ ] #2 Moodboard payloads preserve stable identity name description canvas metadata ownership and optimistic base state while placement payloads preserve note identity position size order and display metadata.
- [ ] #3 Studio documents preserve stable identity source note title content document type revision metadata ownership and optimistic base state across SQLite and PostgreSQL.
- [ ] #4 Server-origin moodboard placement and Studio document mutations capture canonical envelopes when Sync v2 is active.
- [ ] #5 AI title suggestion summarization and Studio generation requests remain operations while only explicitly accepted persisted output enters synchronized state with provenance.
- [ ] #6 Concurrent board layout document revision note deletion restore and placement edits yield idempotent outcomes or reviewable conflicts with authorized bounded queries.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
- [ ] #7 Focused Notes moodboard placement Studio and accepted-output suites pass on supported database backends.
- [ ] #8 Bandit and static checks pass for touched production files.
- [ ] #9 Transient-request exclusion provenance conflict and pagination scenarios have automated evidence.
<!-- DOD:END -->
