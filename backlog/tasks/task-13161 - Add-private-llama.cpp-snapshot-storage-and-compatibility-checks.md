---
id: TASK-13161
title: Add private llama.cpp snapshot storage and compatibility checks
status: To Do
assignee: []
created_date: '2026-09-05 02:19'
updated_date: '2026-09-05 02:27'
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
- [ ] #1 Private versioned storage rejects traversal and symlinks, publishes only verified saves, and preserves existing snapshots on failure.
- [ ] #2 Compatibility fails closed and retention keeps newest committed snapshots with default 10 and range 1 to 1000.
- [ ] #3 Unit and property tests exercise corruption, crash publication, disk errors, and incompatible fingerprints.
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
