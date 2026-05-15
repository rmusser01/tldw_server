---
id: TASK-244.8
title: Implement Backlog.md Python clone safe mutation core
status: Done
assignee:
  - codex
created_date: '2026-05-10 23:50'
updated_date: '2026-05-11 00:17'
labels: []
dependencies:
  - TASK-244.7
references:
  - 'https://github.com/MrLesk/Backlog.md'
documentation:
  - >-
    Docs/superpowers/specs/2026-05-10-backlog-md-python-compatibility-clone-design.md
  - >-
    Docs/superpowers/plans/2026-05-10-backlog-md-python-compatibility-clone-implementation-plan.md
parent_task_id: TASK-244
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 7 from the Backlog.md Python compatibility clone implementation plan. Add the safe mutation core for task create/edit operations with path containment, atomic writes, validation-before-write behavior, section-scoped edits, CLI/MCP adapters, and disabled-by-default onStatusChange handling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Task create writes a valid new task in a temp Backlog fixture repo
- [x] #2 Task edit can update description/notes/final summary and checklist states without rewriting unowned sections
- [x] #3 Invalid checklist indexes, duplicate IDs, circular dependencies, and path traversal are rejected before writes
- [x] #4 Writes are atomic and leave no partial task file when validation fails
- [x] #5 CLI and pure MCP mutation adapters expose the safe core without shell execution
- [x] #6 onStatusChange remains disabled by default with explicit disabled/not-implemented behavior
- [x] #7 Focused mutation tests, full backlog-py tests, Bandit, diff checks, and two-stage review are completed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write failing tests for path containment and mutation behavior in test_security_paths.py and test_task_mutations.py.
2. Run the focused tests to verify the expected red failure against the current read-only core.
3. Implement path containment, validation-before-write mutation services, same-directory temp-file plus os.replace atomic commits, duplicate/circular dependency guards, section-scoped task updates, and disabled onStatusChange handling.
4. Add CLI task create/edit mutation options and pure MCP task_create/task_edit adapters over the safe core.
5. Run focused mutation tests, the full tools/backlog-py test suite, Bandit on tools/backlog-py/src, git diff checks, and two-stage review before finalizing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Worker added RED mutation/security tests and captured missing `backlog_py.security` / `MutableRepository` collection failures.
- Implemented safe path containment, validation-before-write task creation/editing, same-directory temp-file writes committed with `os.replace`, duplicate ID/circular dependency/checklist index guards, section-scoped edits, and disabled-by-default `onStatusChange` errors.
- Added CLI `task create` / `task edit` adapter options and pure MCP `task_create` / `task_edit` paths over the same core. Unsupported MCP argument shapes still raise explicit not-implemented errors for this slice.
- Verification run by worker: focused mutation/security tests passed; full `tools/backlog-py/tests` suite passed. Controller still owns Bandit, diff check, and two-stage review.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Controller review updates 2026-05-11: added missing fail-closed validation for nonexistent dependencies and unknown statuses on create/edit, allowed MCP title-only create through the repository ID allocator, added AC/DoD uncheck support across repository/CLI/MCP, and fixed symlink containment so task directory/file symlinks cannot redirect writes outside the lexical backlog path. Spec review and code-quality review both approved after fixes.

Final verification: focused mutation/security tests -> 21 passed; full tools/backlog-py tests -> 61 passed; Bandit on tools/backlog-py/src -> exit 0 with results []; git diff --check -> clean; pyproject dependency diff -> empty.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the safe task mutation core for the Backlog.py compatibility package. The implementation introduces lexical path containment, same-directory temporary writes with os.replace, validation-before-write guards for invalid IDs, duplicate IDs, unknown statuses, nonexistent and circular dependencies, invalid AC/DoD indexes, path traversal, symlink escapes, and disabled onStatusChange. Task create/edit now support section-scoped description, notes, final summary, AC/DoD check and uncheck mutations through the repository, CLI, and pure MCP adapters without shell execution or dependency additions. Verification covered red/green mutation tests, 21 focused mutation/security tests, 61 full backlog-py tests, Bandit with no findings, clean diff checks, and approved spec/code-quality reviews.
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
