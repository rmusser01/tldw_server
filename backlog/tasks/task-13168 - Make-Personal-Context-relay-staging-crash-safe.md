---
id: TASK-13168
title: Make Personal Context relay staging crash-safe
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-04 03:33'
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
- [ ] #1 A hidden pending authority insert remains discoverable from deterministic source identity even if the process crashes before the publication journal records the staged row.
- [ ] #2 Lease acquisition and release, staging, acknowledgement, poison, completion, and finalization transitions use owner-token, batch, status, generation, and affected-row compare-and-set checks appropriate to each transition.
- [ ] #3 Authority becomes pull-visible only after durable source acknowledgement, and restart after crashes immediately after insert, acknowledgement, or finalization converges exactly once.
- [ ] #4 Lease loss or authenticated source purge compensates any hidden pending authority row and current-head barrier without exposing content or destructively deleting unrelated history.
- [ ] #5 Durable poison is reserved for authenticated source corruption; head contention, database errors, lease loss, and other retryable failures remain pending.
- [ ] #6 Real cross-instance SQLite tests cover each crash boundary, restart after purge, slow relay with lease contention, corrupt source restart, and captured DB, WAL, and log content canaries.
- [ ] #7 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs hidden staging and authority publication ordering.
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
