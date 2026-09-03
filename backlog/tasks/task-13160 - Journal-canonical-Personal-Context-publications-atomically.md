---
id: TASK-13160
title: Journal canonical Personal Context publications atomically
status: Done
assignee:
  - '@codex'
created_date: '2026-09-03 13:39'
updated_date: '2026-09-03 20:36'
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
- [x] #1 Idempotent Personalization schema additions store encrypted source-publication batches, per-profile sequence state, ingress replay receipts, continuity state, and content-free terminal metadata.
- [x] #2 Every eligible API, migration, and authorized agent mutation atomically commits canonical state, one manifest advance, and a complete ordered publication batch; device-only data is never journaled.
- [x] #3 Client-ingress replay identity and canonical payload digest return the original canonical version and batch without a second mutation after interruption.
- [x] #4 Publication payloads and labels are encrypted at rest; indexes, logs, status, and retry metadata contain only bounded opaque or content-free values.
- [x] #5 Pre-activation compaction preserves the latest exact canonical head per object and never removes work newer than an activation watermark.
- [x] #6 Focused migration, transaction rollback, replay, concurrency, encryption, and plaintext-canary tests pass while ongoing_sync_version remains 0.
- [x] #7 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs storage and publication authority.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. Existing backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs the storage, authority, encryption, and publication-journal boundaries.
1. Add failing journal schema, atomicity, replay, rollback, concurrency, encryption, and pre-activation compaction tests.
2. Add idempotent Personalization publication-profile, batch, row, and ingress-receipt schema objects and indexes.
3. Implement the transaction-scoped encrypted publication journal using the canonical repository key and AEAD boundary.
4. Integrate every eligible record, scope, proposal, manifest-only, migration, and authorized agent mutation into its existing canonical transaction while excluding device-only state.
5. Add apply_sync_ingress replay semantics with bounded digest-conflict errors and stable canonical receipts.
6. Run the four targeted Personalization suites, Ruff, Bandit for touched code, plaintext-canary checks, and diff hygiene.
7. Complete acceptance criteria and implementation notes, then commit the task as one atomic implementation unit.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

Implemented the encrypted, transaction-owned Personal Context source-publication journal and ingress replay receipts. Canonical record, scope, proposal, manifest, bootstrap/migration, and purge-manifest writes now append ordered batches inside their owning transaction; server runtime policy and device-only inputs remain excluded. Pre-activation compaction marks only superseded encrypted rows at or below a caller watermark, preserving their bytes until the activation owner performs later terminal cleanup. Added focused atomicity, replay, conflict, concurrency, key-rotation, rollback, plaintext-canary, and compaction coverage. ADR required: no new ADR; ADR-002 governs this implementation. Verification: 59 targeted Personalization tests, Ruff, Bandit, and diff hygiene passed. Known blockers/skips: none.
