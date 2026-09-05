---
id: TASK-13161
title: Relay ordered Personal Context authority publications through Sync V2
status: Done
assignee:
  - '@codex'
created_date: '2026-09-03 13:39'
updated_date: '2026-09-05 02:09'
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
The ordered Personal Context authority relay is complete through the independently approved remediation chain: TASK-13166 head 37d2c4abdc bound confirmation identity and receipts; TASK-13167 head 7190ead127 hardened legacy empty-wire receipt backfill; TASK-13168 head 4f54f80621 made staging and compensation crash-safe; TASK-13169 head ae5eb04151 enforced authorized purge and stale-ciphertext custody; TASK-13170 final head 4261bfd1ac unified the exact 100-row and 100-millisecond recovery budget and immutable authority lineage; TASK-13171 head cabdaf36f2 enforced selected-operation exchange proof while preserving Sync V2 compatibility; TASK-13172 approved candidate 8ee4c2227df533dbff3dea303c7838c1ba01d4d6 certified the production factory, restart, PostgreSQL race, single-dataset invariant, privacy, and exact-once paths. Final evidence is 25/25 certification with genuine PostgreSQL and no skip, 773/773 exact 14-file matrix, and 241 affected passes plus 2 existing skips with eight approved-head baselines deselected. Ruff, Bandit, git diff, and artifact scans passed. Every child received final independent specification and code or security approval; the final TASK-13172 rechecks reported no Critical, Important, or Minor findings. Canonical SDD reports and progress are updated, and incident-backed lessons are retained in backlog/docs/lessons-testing-evidence.md; closure produced no new incident requiring another lesson. ADR-002 remains governing with no new ADR. Custody guarantees cover application-owned active SQLite databases, WAL and SHM files, controlled diagnostics, logs, migration snapshots, and application-owned backup fixtures; they do not claim physical deletion from external or operator backups, exported recovery bundles, or prior-process memory. The one known fake-PostgreSQL link-binding baseline and seven legacy bootstrap activation-proof baselines remain explicitly documented; the genuine PostgreSQL race passed. Protocol version remains 0 and no schema or public API was added.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TASK-13161 is complete: ordered recoverable Personal Context authority publication, verified ingress materialization, filtered encrypted egress, bounded recovery, crash and purge safety, and compatibility gates are implemented and independently approved through TASK-13166 to TASK-13172.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
