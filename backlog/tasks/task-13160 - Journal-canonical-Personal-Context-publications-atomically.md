---
id: TASK-13160
title: Journal canonical Personal Context publications atomically
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-03 13:39'
labels:
  - personal-context
  - sync
  - storage
  - security
dependencies:
  - TASK-13159
references:
  - >-
    backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
  - Docs/superpowers/specs/2026-09-02-personal-context-ongoing-sync-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the encrypted Personalization-owned journal that makes every eligible canonical Personal Context mutation publishable after commit. Canonical state, authoritative manifest advance, monotonic publication batch, ingress replay receipt, and continuity state must share one transaction while ongoing synchronization remains disabled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Idempotent Personalization schema additions store encrypted source-publication batches, per-profile sequence state, ingress replay receipts, continuity state, and content-free terminal metadata.
- [ ] #2 Every eligible API, migration, and authorized agent mutation atomically commits canonical state, one manifest advance, and a complete ordered publication batch; device-only data is never journaled.
- [ ] #3 Client-ingress replay identity and canonical payload digest return the original canonical version and batch without a second mutation after interruption.
- [ ] #4 Publication payloads and labels are encrypted at rest; indexes, logs, status, and retry metadata contain only bounded opaque or content-free values.
- [ ] #5 Pre-activation compaction preserves the latest exact canonical head per object and never removes work newer than an activation watermark.
- [ ] #6 Focused migration, transaction rollback, replay, concurrency, encryption, and plaintext-canary tests pass while ongoing_sync_version remains 0.
- [ ] #7 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs storage and publication authority.
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
