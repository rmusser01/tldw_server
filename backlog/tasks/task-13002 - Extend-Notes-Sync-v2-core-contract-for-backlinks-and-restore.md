---
id: TASK-13002
title: Extend Notes Sync v2 core contract for backlinks and restore
status: To Do
assignee: []
created_date: '2026-08-08 20:21'
labels:
  - notes
  - sync-v2
  - parity
dependencies: []
references:
  - >-
    https://github.com/rmusser01/tldw_chatbook/blob/dev/backlog/decisions/046-synchronized-database-notes-parity.md
documentation:
  - >-
    https://github.com/rmusser01/tldw_chatbook/blob/dev/Docs/superpowers/specs/2026-08-08-notes-server-parity-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the existing notes.note domain a lossless production contract for title, content, conversation/message backlinks, tombstones, and base-aware restore so Chatbook can synchronize one personal Database Notes collection without server-side ambiguity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Capabilities and envelope schemas expose notes.note upsert/tombstone with title, content, conversation_id, and message_id under server_trusted_v1.
- [ ] #2 A restore-intent upsert can resurrect only the current tombstone head; stale ordinary updates against deleted notes remain whole-object conflicts.
- [ ] #3 Server-origin note create, update, delete, and restore produce the same canonical envelopes and materialized object state as client-origin mutations.
- [ ] #4 Accepted note title/content are preserved exactly within documented limits; validation never truncates, escapes, or rewrites canonical Markdown.
- [ ] #5 SQLite and PostgreSQL contract tests cover create, update, tombstone, stale conflict, restore, idempotency, and exact payload materialization.
- [ ] #6 Keyword writes remain explicitly blocked until their separately synchronized domain is enabled; no partial ownership is implied.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
- [ ] #7 Focused Sync v2 and Notes tests pass on supported database backends.
- [ ] #8 Bandit and static checks pass for touched production files.
- [ ] #9 ADR-031 and Sync v2 public contract documentation are updated.
<!-- DOD:END -->
