---
id: TASK-13161
title: Relay ordered Personal Context authority publications through Sync V2
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-03 13:39'
updated_date: '2026-09-04 03:20'
labels:
  - personal-context
  - sync
  - relay
  - security
dependencies:
  - TASK-13160
references:
  - >-
    backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
  - Docs/superpowers/specs/2026-09-02-personal-context-ongoing-sync-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Recoverably relay server canonical publication batches into Sync V2 and materialize client ingress without turning transport state into application authority. Both relay entry points must share global per-profile ordering, and Personal Context pull must expose only verified home-authority egress.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After-commit and pull-time relay share one recoverable per-profile lease and claim only the earliest nonterminal publication sequence.
- [ ] #2 Semantic authority envelopes become durable before their manifest sibling; deterministic IDs and ordinal receipts make every crash point idempotent.
- [ ] #3 Client-authored envelopes remain ingress-only and become applied only after the canonical Personalization receipt is verified; accepted ingress is never pull-visible.
- [ ] #4 Personal Context pull filters to verified applied home-authority egress, separates raw scan watermarks from delivered/application checkpoints, and cannot be spoofed by a registered client.
- [ ] #5 Pull-time recovery stops at the fixed 100-row/100-millisecond budget and returns personal_context_relay_pending without skipping unavailable authority data.
- [ ] #6 SQLite interruption, interleaved-relay, cursor-pagination, reserved-identity, poisoned-batch, and privacy tests pass.
- [x] #7 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs ordered cross-database relay.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. Existing backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs globally ordered cross-database relay, reserved authority identity, and ingress visibility.
1. Add failing relay interruption, materializer replay, role-visibility, cursor, reserved-identity, poison, privacy, and interleaving tests.
2. Implement one recoverable per-profile lease that claims only the earliest nonterminal publication sequence and stages semantic rows before manifest.
3. Materialize client ingress through the canonical Personalization replay receipt before marking Sync apply status.
4. Add an internal-only deterministic home-authority server-origin insertion seam and reject public spoofing.
5. Recover and filter Personal Context pull within the fixed row/time budget using separate raw scan and delivered/application checkpoints.
6. Run the five targeted Sync suites, Ruff, Bandit for touched code, and diff hygiene while ongoing_sync_version remains 0.
7. Complete acceptance criteria and implementation notes, then commit the task as one atomic implementation unit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Round-5 targeted verification completed successfully (331 focused tests, Ruff, Bandit, and diff hygiene), but final review rejected completion. TASK-13161 remains In Progress because crash-orphan cleanup, poison-versus-retry classification, exact shared recovery-budget enforcement, legacy empty-wire receipt backfill safety, compatibility of Personal Context conflict proof gates with unrelated Sync conflicts, production-factory/HTTP recovery evidence, and stale ciphertext retention/cleanup are not yet resolved to the acceptance standard. [ADR-002](../decisions/002-personal-context-profile-authority-sync-and-encryption.md) remains the governing decision. No additional fix work is included in this housekeeping change.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
