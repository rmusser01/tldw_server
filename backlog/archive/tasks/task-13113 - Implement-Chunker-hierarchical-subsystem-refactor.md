---
id: TASK-13113
title: Implement Chunker hierarchical subsystem refactor
status: In Progress
created_date: 2026-08-24 07:10
dependencies:
- TASK-13112
labels:
- chunking
- refactor
- implementation
priority: High
references:
- TASK-13112
documentation:
- Docs/superpowers/specs/2026-08-23-chunker-hierarchical-subsystem-refactor-design.md
- Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md
modified_files:
- Docs/superpowers/specs/2026-08-23-chunker-hierarchical-subsystem-refactor-design.md
- Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md
- backlog/tasks/task-13113 - Implement-Chunker-hierarchical-subsystem-refactor.md
- tldw_Server_API/app/core/Chunking/chunker.py
- tldw_Server_API/app/core/Chunking/process_text/models.py
- tldw_Server_API/app/core/Chunking/process_text/dispatch.py
- tldw_Server_API/app/core/Chunking/hierarchical/
- tldw_Server_API/tests/Chunking/
updated_date: 2026-08-25 05:47
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Chunker hierarchical subsystem design in an isolated worktree based on current origin/dev. Extract spans, leaves, tree building, grouping, flattening, models, and service coordination while preserving the approved public, output, fallback, aliasing, logging, and call-trace contracts. No production edits begin until TASK-13112 is approved and finalized.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The implementation branch is reconciled with current origin/dev and focused baseline characterization is rerun before production edits
- [ ] #2 Frozen characterization tests cover option handling, leaf call traces, spans, malformed trees, flatten aliasing, logging, signatures, and import boundaries
- [ ] #3 The hierarchical package is extracted with the approved component interfaces and dependency direction while public hierarchy behavior remains compatible
- [ ] #4 The approved private span and header-title helpers are removed and process_text imports the shared span function directly
- [ ] #5 Focused and complete Chunking tests, compileall, Ruff, scoped Black, Bandit, and git diff --check pass with results recorded
- [ ] #6 The PR remains non-merge-ready until the human requester supplies the required Change summary explaining what changed and why
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute `Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md` task-by-task: (1) reconcile current origin/dev and rerun the focused baseline before production edits; (2) freeze public signatures, composition, option, call-trace, logging, malformed-tree, aliasing, and regex contracts; (3) extract passive models and shared paragraph spans while migrating process_text; (4) extract and activate leaf construction; (5) extract the tree builder and per-call service coordination; (6) extract and activate grouping; (7) extract flattening and complete public delegation; (8) enforce dependency boundaries, run the complete Chunking/static/security gates, obtain final code review, and prepare the PR handoff against dev. Each structural stage uses red-green tests and a focused commit. Any behavior correction must satisfy the approved correction gate and land in a separate fix commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created during the second spec-review pass to separate implementation tracking from design-only TASK-13112. User approved the final design on 2026-08-24; TASK-13112 was finalized and moved to completed storage, and this task moved to In Progress. Detailed implementation planning completed on 2026-08-24 through the writing-plans workflow. The linked plan uses eight reviewable tasks, incremental active wiring, frozen compatibility characterizations, exact malformed-tree and leaf-call matrices, AST dependency checks, scoped formatting/type/security gates, and an explicit human-written PR Change summary merge blocker. Production code remains untouched pending execution handoff.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
