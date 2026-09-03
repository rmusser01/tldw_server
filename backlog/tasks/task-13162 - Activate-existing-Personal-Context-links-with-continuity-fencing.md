---
id: TASK-13162
title: Activate existing Personal Context links with continuity fencing
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-03 13:40'
labels:
  - personal-context
  - sync
  - activation
  - security
dependencies:
  - TASK-13161
references:
  - >-
    backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
  - Docs/superpowers/specs/2026-09-02-personal-context-ongoing-sync-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the replayable activation journal that establishes a server baseline and publication checkpoint for existing and newly linked profiles before ongoing sync can run. Activation must preserve publication order, survive every cross-database interruption, and issue continuity proof that fails closed across capability gaps.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Preparation stores one encrypted exact-head baseline, digest, purge generation, and watermark at a whole publication-batch boundary.
- [ ] #2 Deterministic Sync installation and receipt verification precede a leased Personalization CAS that marks all covered batches covered_by_activation and advances content-free covered-through proof before compaction.
- [ ] #3 Per-device acknowledgments and activation state replay idempotently by activation ID and digest across Personalization and Sync without claiming cross-database atomicity.
- [ ] #4 A random activation epoch and continuity token are durable, generation-bound, echoed and validated on every version-1 exchange, and invalidated or write-fenced when journaling cannot be guaranteed.
- [ ] #5 Capability downgrade preserves links and queued work; restoration requires the same proven continuity pair or a fresh baseline.
- [ ] #6 Restart tests cover preparation, Sync installation, coverage CAS, compaction, acknowledgment, racing server writes, and first post-watermark relay.
- [ ] #7 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs activation and continuity.
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
