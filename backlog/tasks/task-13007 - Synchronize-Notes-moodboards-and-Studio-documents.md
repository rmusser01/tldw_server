---
id: TASK-13007
title: Synchronize Notes moodboards and Studio documents
status: In Progress
assignee: []
created_date: '2026-08-08 20:25'
updated_date: '2026-08-25 00:45'
labels:
  - notes
  - sync-v2
  - parity
  - moodboards
  - studio
dependencies:
  - TASK-13006
references:
  - Docs/ADR/031-notes-capability-sync-domains.md
  - Docs/ADR/040-synchronized-moodboards-and-studio-authority.md
documentation:
  - Docs/superpowers/specs/2026-08-24-notes-moodboard-studio-sync-design.md
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Publish the approved authority and lifecycle design in Docs/superpowers/specs/2026-08-24-notes-moodboard-studio-sync-design.md and ADR-040 before implementation.
2. Run an independent spec review and resolve every valid finding.
3. Obtain requester approval of the reviewed written spec.
4. Use the writing-plans workflow to add concrete per-child implementation plans before code changes.
5. Execute TASK-13007.1 through TASK-13007.4 in dependency order with TDD review verification and live PostgreSQL gates.

ADR required: yes
ADR path: Docs/ADR/040-synchronized-moodboards-and-studio-authority.md
Reason: The work changes persistent schema tenant scope sync authority lifecycle conflict policy provenance and capability activation boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approved design decomposed into TASK-13007.1 through TASK-13007.4. Existing Chatbook reference paths are stale and will be replaced by the server-local authoritative spec and ADR-040.

The approved interactive design was written to Docs/superpowers/specs/2026-08-24-notes-moodboard-studio-sync-design.md with proposed ADR-040. Independent review found portable-time, client compound-push, task-boundary, smart-match portability, closed Studio schema, encryption-policy, source-note authorization, and lifecycle-binding gaps. The spec, ADR, and child acceptance criteria were revised; independent re-review approved the result with no blocker, high, or medium findings. Requester approval of the reviewed written spec is the next gate. No implementation has started.
<!-- SECTION:NOTES:END -->

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
