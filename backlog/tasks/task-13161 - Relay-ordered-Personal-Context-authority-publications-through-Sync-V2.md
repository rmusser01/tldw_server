---
id: TASK-13161
title: Relay ordered Personal Context authority publications through Sync V2
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-03 13:39'
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
- [ ] #5 Pull-time recovery stops at an actual post-filter lookahead or the fixed 100-row/100-millisecond budget and returns personal_context_relay_pending without skipping unavailable authority data.
- [ ] #6 SQLite interruption, interleaved-relay, cursor-pagination, reserved-identity, poisoned-batch, and privacy tests pass.
- [ ] #7 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs ordered cross-database relay.
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
