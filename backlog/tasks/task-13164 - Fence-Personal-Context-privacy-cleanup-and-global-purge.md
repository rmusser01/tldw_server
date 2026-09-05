---
id: TASK-13164
title: Fence Personal Context privacy cleanup and global purge
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-03 13:40'
labels:
  - personal-context
  - sync
  - privacy
  - deletion
dependencies:
  - TASK-13162
references:
  - >-
    backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
  - Docs/superpowers/specs/2026-09-02-personal-context-ongoing-sync-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make restrictive Personal Context mutations immediately authoritative at every server read boundary while tracking replayable cleanup acknowledgments, and implement delete-everywhere generation fencing so old ingress or authority history cannot resurrect purged content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every restrictive canonical mutation atomically records a version-bound content-free cleanup requirement before the new head is externally readable.
- [ ] #2 Derived context, search, summary, cache, tool, and index paths consult the restrictive head immediately and report cleanup complete only after server acknowledgment.
- [ ] #3 Global purge advances generation, destroys canonical and derived-readable content, terminalizes older source batches, and creates one deterministic new-generation barrier under canonical mutation serialization.
- [ ] #4 Older uncommitted ingress becomes stale_generation, older authority cannot stage behind the barrier, and Sync history is crypto-shredded to a minimal content-free acknowledgment ledger.
- [ ] #5 Server-initiated and client-initiated purge requests are signed, durable, idempotent, remain usable after ongoing-sync capability or continuity loss, and cannot recreate the old profile before acknowledgment.
- [ ] #6 Cleanup-interruption, cache invalidation, every purge interleaving, restart, stale-device, retention, and plaintext-canary tests pass.
- [ ] #7 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs privacy cleanup and purge fencing.
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
