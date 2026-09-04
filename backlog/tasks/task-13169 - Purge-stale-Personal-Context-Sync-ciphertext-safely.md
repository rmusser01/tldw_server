---
id: TASK-13169
title: Purge stale Personal Context Sync ciphertext safely
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-04 03:34'
labels:
  - personal-context
  - sync
  - security
  - retention
dependencies:
  - TASK-13168
references:
  - >-
    backlog/tasks/task-13161 -
    Relay-ordered-Personal-Context-authority-publications-through-Sync-V2.md
  - >-
    backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
  - Docs/superpowers/specs/2026-09-02-personal-context-ongoing-sync-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remediate TASK-13161 by binding cryptographic cleanup of stale Personal Context Sync rows to the authenticated explicit Delete Entire Profile path while keeping pull and relay recovery non-destructive.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Authenticated explicit direct full-profile purge cryptographically shreds or otherwise makes unrecoverable every old-generation Personal Context Sync authority and ingress payload and wrapped data key, including pending, orphaned, superseded, and history-key-protected rows.
- [ ] #2 Pull, relay recovery, compaction, listing, and ordinary mutation never trigger destructive profile cleanup.
- [ ] #3 Content-free tombstone and generation-fence evidence remains sufficient for retry and audit while current-generation data, other profiles, and other datasets are unaffected.
- [ ] #4 Cleanup is idempotent, restartable after partial failure, and covers retained backup artifacts within the documented profile-purge boundary.
- [ ] #5 Before explicit purge, non-destructive scans skip stale-generation rows without exposing them or allowing an orphaned pending row to block eligible current-generation delivery.
- [ ] #6 Real SQLite tests and DB, WAL, backup, and key-rotation canaries prove old ciphertext cannot be recovered after authorized purge.
- [ ] #7 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs direct-purge-only cryptographic shredding.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
