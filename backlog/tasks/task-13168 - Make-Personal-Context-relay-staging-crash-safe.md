---
id: TASK-13168
title: Make Personal Context relay staging crash-safe
status: Done
assignee:
  - '@codex'
created_date: '2026-09-04 03:33'
updated_date: '2026-09-04 08:24'
labels:
  - personal-context
  - sync
  - security
  - relay
dependencies:
  - TASK-13166
  - TASK-13167
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
Remediate TASK-13161 by making hidden authority staging, source acknowledgement, finalization, retry, and compensation recoverable across crashes and concurrent relay instances.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A hidden pending authority insert remains discoverable from deterministic source identity even if the process crashes before the publication journal records the staged row.
- [x] #2 Lease acquisition and release, staging, acknowledgement, poison, completion, and finalization transitions use owner-token, batch, status, generation, and affected-row compare-and-set checks appropriate to each transition.
- [x] #3 Authority becomes pull-visible only after durable source acknowledgement, and restart after crashes immediately after insert, acknowledgement, or finalization converges exactly once.
- [x] #4 Lease loss or authenticated source purge compensates any hidden pending authority row and current-head barrier without exposing content or destructively deleting unrelated history.
- [x] #5 Durable poison is reserved for authenticated source corruption; head contention, database errors, lease loss, and other retryable failures remain pending.
- [x] #6 Real cross-instance SQLite tests cover each crash boundary, restart after purge, slow relay with lease contention, corrupt source restart, and captured DB, WAL, and log content canaries.
- [x] #7 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs hidden staging and authority publication ordering.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED crash, CAS, and failure-classification tests. 2. Recover deterministic hidden staging from source identity. 3. Fence all relay transitions and compensate lost claims. 4. Keep only authenticated source corruption poisonable. 5. Run targeted security and concurrency checks. 6. Self-review and close the task. ADR required: no new ADR; ADR-002 governs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented deterministic hidden-row recovery with structured stage receipts, acknowledgement-before-finalization ordering, exact source/Sync CAS fencing, and narrowly authorized purge compensation. Review round 1 then hardened stale-owner recovery, fenced staging and finalization through the actual Sync commit using the existing Sync-then-source lock order, and made terminal cleanup cover cursor-bearing identities while durably advancing past safely reconciled bounded prefixes. Applied history and unrelated rows are never deleted; cursor/identity disagreement, adapter rejection, head contention, database exceptions, and lease races remain retryable. Persisted two-database tests cover every durable boundary, two relay instances, purge, slow stage/finalize commit interleavings, corrupt-source restart, and plaintext scans across DB/WAL/SHM/log artifacts. No schema, migration, dependency, protocol-version, new ADR, or later-task activation was required. Focused verification: 104 passed; Ruff, Bandit, and diff check passed. Sandbox-only cache/log-buffer permission warnings are the only known noise; no blockers.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Personal Context authority relay staging now converges exactly once across every durable and commit boundary while keeping authority hidden until source acknowledgement. Exact source/Sync guards prevent stale takeover, purge, and finalization races; bounded terminal cleanup preserves applied and unrelated history.
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
