---
id: TASK-13421
title: Design Chunker hierarchical subsystem refactor
status: Done
created_date: 2026-10-03 01:27
labels:
- chunking
- design
- refactor
priority: high
references:
- backlog/completed/task-13112 - Design-Chunker-hierarchical-subsystem-refactor.md
documentation:
- Docs/superpowers/specs/2026-08-23-chunker-hierarchical-subsystem-refactor-design.md
- Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md
updated_date: 2026-10-03 01:29
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Completed and user-approved design for extracting Chunker hierarchical spans, leaves, builder, grouping, flattening, and live service coordination without changing public contracts. This replacement record resolves the old design TASK-13112 collision with unrelated upstream work introduced by the October 2 rebase; no design decisions are being reopened.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The design and implementation plan are approved by the requester
- [x] #2 Public signatures, malformed-tree, exception, offset, shallow identity, grouping and call-time contracts are defined
- [x] #3 Implementation uses isolated worktree, per-task TDD/reviews, full Chunking/static/security gates, and human-written Change summary merge gate
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-02 tracking-only reconciliation: original TASK-13112 design record and approved artifacts remain provenance. The upstream now contains unrelated TASK-13112 records, making the old ID ambiguous. Original design completion and repeated user approvals remain valid; only unique tracking identity is replaced. No production behavior or scope changed. Old record contents are preserved at the explicit reference path pending the scoped archive exception.
Design-only verification and completion criteria remain satisfied by the original approved record and committed artifacts. Bandit is inapplicable to this metadata-only migration; no production code changes. No unresolved design blocker remains. Historical implementation IDs are provenance, not active tracking targets.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Approved compatibility-first decomposition into passive models, shared spans, leaves, tree builder, grouping, flattening, and live-context service. The incremental test/review gates preserve public wrapper composition, source offsets, fallback/logging behavior and shallow aliasing; no behavior corrections are pre-approved.
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
