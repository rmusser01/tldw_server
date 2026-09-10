---
id: TASK-13186
title: Add private llama.cpp snapshot storage and compatibility checks
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 02:19'
updated_date: '2026-09-05 02:55'
labels: []
dependencies: []
documentation:
  - Docs/Design/2026-09-04-llamacpp-manual-slot-snapshots.md
  - Docs/ADR/043-managed-llamacpp-manual-slot-snapshots.md
  - Docs/superpowers/plans/2026-09-04-llamacpp-manual-snapshots.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve sensitive runtime cache artifacts with verifiable integrity and conservative restore compatibility.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Private versioned storage rejects traversal and symlinks, publishes only verified saves, and preserves existing snapshots on failure.
- [x] #2 Compatibility fails closed and retention keeps newest committed snapshots with default 10 and range 1 to 1000.
- [x] #3 Unit and property tests exercise corruption, crash publication, disk errors, and incompatible fingerprints.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Task 1 of Docs/superpowers/plans/2026-09-04-llamacpp-manual-snapshots.md with TDD, targeted verification, Bandit and independent review. ADR required: yes; ADR path: Docs/ADR/043-managed-llamacpp-manual-slot-snapshots.md; reason: sensitive runtime snapshot ownership.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented storage, metadata, compatibility and profile defaults in 88d0de09c6, with review fixes 05c1ef71c7 and f9ba63c5ac. Independent spec/quality review approved after two fix rounds. 50 targeted tests with cached Hypothesis; Ruff, compileall, Bandit and whitespace checks clean. Existing seven bootstrap warnings documented in report. ADR043 accepted from requester design/execution approval. Minor descriptor cleanup noted for integration/final review; runtime guards remain TASK-13187.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage1 storage reviewed and complete; one non-blocking descriptor cleanup follow-up recorded in execution ledger.
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
