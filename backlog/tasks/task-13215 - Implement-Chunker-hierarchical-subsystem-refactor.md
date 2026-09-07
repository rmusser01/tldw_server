---
id: TASK-13215
title: Implement Chunker hierarchical subsystem refactor
status: In Progress
assignee: []
created_date: 2026-09-07 19:26
updated_date: 2026-09-07 19:49
labels:
- chunking
- refactor
- implementation
dependencies:
- TASK-13112
references:
- TASK-13112
documentation:
- Docs/superpowers/specs/2026-08-23-chunker-hierarchical-subsystem-refactor-design.md
- Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md
priority: high
modified_files:
- Docs/superpowers/specs/2026-08-23-chunker-hierarchical-subsystem-refactor-design.md
- Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md
- backlog/completed/task-13112 - Design-Chunker-hierarchical-subsystem-refactor.md
- backlog/archive/tasks/task-13113 - Implement-Chunker-hierarchical-subsystem-refactor.md
- backlog/tasks/task-13215 - Implement-Chunker-hierarchical-subsystem-refactor.md
- tldw_Server_API/app/core/Chunking/chunker.py
- tldw_Server_API/app/core/Chunking/process_text/models.py
- tldw_Server_API/app/core/Chunking/process_text/dispatch.py
- tldw_Server_API/app/core/Chunking/hierarchical/
- tldw_Server_API/tests/Chunking/
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Chunker hierarchical subsystem design in an isolated worktree based on current origin/dev. Extract spans, leaves, tree building, grouping, flattening, models, and service coordination while preserving the approved public, output, fallback, aliasing, logging, and call-trace contracts. No production edits begin until TASK-13112 is approved and finalized.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The implementation branch is reconciled with current origin/dev and focused baseline characterization is rerun before production edits
- [ ] #2 Frozen characterization tests cover option handling, leaf call traces, spans, malformed trees, flatten aliasing, logging, signatures, and import boundaries
- [ ] #3 The hierarchical package is extracted with the approved component interfaces and dependency direction while public hierarchy behavior remains compatible
- [ ] #4 The approved private span and header-title helpers are removed and process_text imports the shared span function directly
- [ ] #5 Focused and complete Chunking tests, compileall, Ruff, scoped Black, Bandit, and git diff --check pass with results recorded
- [ ] #6 The PR remains non-merge-ready until the human requester supplies the required Change summary explaining what changed and why
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md task-by-task: (1) reconcile current origin/dev and rerun the focused baseline before production edits; (2) freeze public signatures, composition, option, call-trace, logging, malformed-tree, aliasing, and regex contracts; (3) extract passive models and shared paragraph spans while migrating process_text; (4) extract and activate leaf construction; (5) extract the tree builder and per-call service coordination; (6) extract and activate grouping; (7) extract flattening and complete public delegation; (8) enforce dependency boundaries, run the complete Chunking/static/security gates, obtain final code review, and prepare the PR handoff against dev. Each structural stage uses red-green tests and a focused commit. Any behavior correction must satisfy the approved correction gate and land in a separate fix commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
This record supersedes the colliding active TASK-13113 after rebasing onto origin/dev introduced the unrelated completed TASK-13113 record for PR 2808. The approved design and detailed implementation plan remain unchanged. 2026-09-07 Task 1 baseline reconciliation: refreshed origin/dev at 01b516d4805ddf5eb7cddac35a11ef0e062facc5; the required scoped diff from 4958cfed65d3c6e9baa43ea47e2b155fed204e13 through current origin/dev across tldw_Server_API/app/core/Chunking and tldw_Server_API/tests/Chunking was empty, so no approved hierarchy/process_text contract was affected. Rebased the isolated branch without conflicts. The focused baseline suite collected 92 tests and completed with 91 passed, 1 skipped, 0 failures, and 196 warnings in 3.01s. Environmental output was limited to the existing no-.env fallback to config.txt, legacy single-user API-key warning, isolated test database fallback, and emitted OpenTelemetry spans. No production files were edited; Bandit is not applicable to this documentation/tracking-only task. Tracking reconciliation completed on 2026-09-07: Backlog allocated TASK-13215, the official CLI archived the old active implementation TASK-13113 without modifying the unrelated completed PR-2808 TASK-13113, and the approved spec, implementation plan, and completed TASK-13112 design record now reference TASK-13215.

2026-09-07 second baseline reconciliation: origin/dev was e3174f1ad9f6dd0b11e4ecb20d48c1c4090d3bfe; the required scoped diff from 01b516d4805ddf5eb7cddac35a11ef0e062facc5 through current origin/dev across tldw_Server_API/app/core/Chunking and tldw_Server_API/tests/Chunking was empty. Rebasing preserved the five workstream commits, including the completed Task 1 commit now at fa93165f7b, and produced the required 0/5 branch relationship. The exact focused seven-file suite collected 92 tests and completed with 91 passed, 1 skipped, 0 failures, and 196 warnings in 1.86s. Environmental output remained limited to the existing no-.env fallback to config.txt, legacy single-user API-key warning, isolated test database fallback, and emitted OpenTelemetry spans. No production files were edited; Task 1 checkboxes remain complete.
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
