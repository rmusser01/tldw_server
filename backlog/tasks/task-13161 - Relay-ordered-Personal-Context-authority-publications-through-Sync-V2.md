---
id: TASK-13161
title: Relay ordered Personal Context authority publications through Sync V2
status: Done
assignee:
  - '@codex'
created_date: '2026-09-03 13:39'
updated_date: '2026-09-04 03:02'
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
- [x] #1 After-commit and pull-time relay share one recoverable per-profile lease and claim only the earliest nonterminal publication sequence.
- [x] #2 Semantic authority envelopes become durable before their manifest sibling; deterministic IDs and ordinal receipts make every crash point idempotent.
- [x] #3 Client-authored envelopes remain ingress-only and become applied only after the canonical Personalization receipt is verified; accepted ingress is never pull-visible.
- [x] #4 Personal Context pull filters to verified applied home-authority egress, separates raw scan watermarks from delivered/application checkpoints, and cannot be spoofed by a registered client.
- [x] #5 Pull-time recovery stops at the fixed 100-row/100-millisecond budget and returns personal_context_relay_pending without skipping unavailable authority data.
- [x] #6 SQLite interruption, interleaved-relay, cursor-pagination, reserved-identity, poisoned-batch, and privacy tests pass.
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
Implemented the [ADR-002](../decisions/002-personal-context-profile-authority-sync-and-encryption.md) boundary as one leased, deterministic Personal Context publication relay shared by after-commit and pull-time recovery. Canonical journal rows remain encrypted at rest; client ingress is receipt-gated and permanently hidden, while only restored, applied home-authority envelopes can egress after exact persisted activation/link proof.

Recovery now uses one absolute 100-row/100-millisecond budget across incremental journal and raw Sync scans for legacy, signed, PC-only, mixed, and domain-subset pulls. Exact self-head confirmation handles the canonical after-commit callback while matching client ingress is pending or applied, without relaxing ordinary lineage validation. Legacy empty wire identities backfill only after the original receipt and authenticated source row match; durable poison attention commits separately from failed decryption. Lease/generation CAS checks, invisible pending rows, deterministic compensation, stale-generation filtering, and restart/interruption coverage prevent stale authority exposure.

Verification: the five required Sync suites plus the full Sync endpoint and Personalization publication suites passed (**331 tests, 52 warnings**). Ruff passed for all touched files, Bandit passed for all touched production files with only existing parser/nosec warnings, and `git diff --check` passed. Per instruction, no full-repository suite was run. Ongoing Sync remains advertised as version 0; tests activate only through persisted internal state, and no TASK-13162 activation route was added.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
