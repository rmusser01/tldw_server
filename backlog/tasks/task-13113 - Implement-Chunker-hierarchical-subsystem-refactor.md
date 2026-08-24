---
id: TASK-13113
title: Implement Chunker hierarchical subsystem refactor
status: To Do
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
modified_files:
- Docs/superpowers/specs/2026-08-23-chunker-hierarchical-subsystem-refactor-design.md
- Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md
- backlog/tasks/task-13113 - Implement-Chunker-hierarchical-subsystem-refactor.md
- tldw_Server_API/app/core/Chunking/chunker.py
- tldw_Server_API/app/core/Chunking/process_text/models.py
- tldw_Server_API/app/core/Chunking/process_text/dispatch.py
- tldw_Server_API/app/core/Chunking/hierarchical/
- tldw_Server_API/tests/Chunking/
updated_date: 2026-08-24 07:15
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
Pending final approval of TASK-13112 and creation of the detailed implementation plan through the writing-plans workflow.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created during the second spec-review pass to separate implementation tracking from design-only TASK-13112. This task remains To Do and blocked on final design approval. The implementation plan must use the approved component interfaces, concrete malformed-tree matrix, logging contract, scoped quality gates, baseline reconciliation procedure, and human-owned PR Change summary gate.
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
