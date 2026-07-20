---
id: TASK-13000
title: Design UserProfiles Stage 2 single-update pipeline
status: In Progress
created_date: 2026-07-20 18:48
labels:
- userprofiles
- refactor
- design
priority: High
references:
- 'PR #2529'
- Historical UserProfiles contract-refactor workstream TASK-12016.1 through TASK-12016.11
documentation:
- Docs/superpowers/specs/2026-07-20-userprofiles-single-update-pipeline-stage2-design.md
modified_files:
- Docs/superpowers/specs/2026-07-20-userprofiles-single-update-pipeline-stage2-design.md
updated_date: 2026-07-20 18:57
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Capture the approved Stage 2 architecture for replacing the transitional UserProfiles single-update path with a typed planner, transaction-owning command service, storage-bound executors, mutation-coupled effect dispatch, and caller-specific compatibility mappers. This task is design-only; implementation planning begins only after human review of the written specification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The design specifies component responsibilities and dependency boundaries for contracts, shared policy, planner, executors, effects, command service, and response mappers.
- [x] #2 The design specifies transaction ownership, optimistic concurrency, execution ordering, same-transaction membership writes, commit behavior, and cross-system atomicity limits.
- [x] #3 The design specifies transport-neutral errors and results, caller-specific compatibility mapping, effect semantics, privacy constraints, migration sequencing, and explicit out-of-scope work.
- [x] #4 The design includes a verification strategy covering characterization, parity, deterministic concurrency, SQLite and PostgreSQL behavior, import boundaries, metrics, and security checks.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design decisions were developed and approved interactively. Work is isolated in .worktrees/userprofiles-stage2-design on branch codex/userprofiles-stage2-design. Baseline focused tests passed: 19 tests across planner, command-service, and v2 compatibility suites. The Backlog allocator reused a completed ID, and the historical UserProfiles parent ID now belongs to an active Embeddings task after tracker cleanup. Those uncommitted invalid records were removed; this standalone explicit ID is authoritative.
Written-spec self-review completed. The review corrected stale Backlog/document references, constrained the temporary connection bridge so post-commit effects cannot run before commit confirmation, documented composite-version linearizability limits, required one monotonic user-row version touch per non-empty plan, required transaction-time membership authorization rechecks, and made the legacy runtime-failure compatibility conflict explicit. The narrow correction rolls back all database mutations and returns an empty applied list rather than reporting rolled-back writes. Structural checks passed: no placeholders, balanced Markdown fences/inline backticks, ASCII-only content, required files present, and git diff --check clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Approved design is written to Docs/superpowers/specs/2026-07-20-userprofiles-single-update-pipeline-stage2-design.md.
- [x] #2 The written specification is self-reviewed for contradictions, missing compatibility behavior, unsafe transaction assumptions, and scope creep.
- [ ] #3 Documentation and Backlog.md records are committed on the isolated design branch.
- [ ] #4 The requester is asked to review the written specification before implementation planning begins.
<!-- DOD:END -->
