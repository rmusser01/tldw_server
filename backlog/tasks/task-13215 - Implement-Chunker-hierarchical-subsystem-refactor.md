---
id: TASK-13215
title: Implement Chunker hierarchical subsystem refactor
status: In Progress
assignee: []
created_date: 2026-09-07 19:26
updated_date: 2026-09-07 20:57
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-09-07 Task 2 contract freeze completed. Added tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py, tldw_Server_API/tests/Chunking/test_hierarchy_malformed_contracts.py, and tldw_Server_API/tests/Chunking/test_hierarchical_spans.py (29 test functions, 54 collected cases) and updated only Task 2 in Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md. Focused pytest: 54 passed, 0 failed, 119 warnings in 1.35s. Ruff: All checks passed. Black --check: 3 files would be left unchanged. git diff --check: exit 0 with no output. Bandit on the three pytest files reported only the expected B101 assert-use rule (57 low-severity findings, 0 medium/high); rerun excluding B101 completed with 0 findings and 0 errors. Self-review confirmed deterministic instance fakes prevent leaf, rewrite, LLM, or external calls; every temporary Loguru sink is removed in finally; metadata fixtures are read-only and preserve the identities under test; malformed/span expectations match the observed unextracted baseline. No baseline discrepancy was found and no production code was changed.
2026-09-07 Task 2 specification-review correction: acceptance criterion #2 remains unchanged in wording but is now unchecked because it also includes import boundaries, which belong to Task 3 and have not been implemented. Task 2's characterization portion is complete and its implementation-plan checkboxes remain accurately checked; this tracking correction does not start Task 3. Strengthened the Task 2 tests to record sanitize_output truthiness before method resolution, cover sanitize_output removal on the semantic rewrite branch as well as metadata and ordinary branches, and assert identity for template and method_options forwarded by the public flat wrapper.
Task 2 specification-review verification: focused three-file suite collected 56 cases and completed with 56 passed, 0 failed, and 123 warnings in 1.17s. Ruff reported All checks passed. Black --check reported all 3 files would be left unchanged. Bandit excluding the expected pytest B101 assertion rule completed with 0 findings and 0 errors. git diff --check exited 0 with no output. The change scope contains only tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py and this official TASK-13215 record; no production file or Task 3 file changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
