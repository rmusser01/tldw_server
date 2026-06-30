---
id: TASK-244.6
title: Implement Backlog.md Python clone read-only repository and CLI
status: Done
assignee:
  - codex
created_date: '2026-05-10 23:13'
updated_date: '2026-05-10 23:29'
labels: []
dependencies:
  - TASK-244.5
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
Implement Task 5 from the Backlog.md Python compatibility clone implementation plan. Add read-only repository operations and CLI commands for task list/view/search/board and config list while preserving the backlog-py command name and proving commands do not mutate repository data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Read-only repository can list tasks, view TASK-1, search fixture task content, and group board data by status
- [x] #2 CLI implements top-level --cwd and --help plus task list --plain, task <id> --plain, search <query> --plain, board, and config list
- [x] #3 Read-only operations and CLI tests are written red-first and pass after implementation
- [x] #4 Live read-only smoke against this repo exits 0 and leaves backlog status unchanged
- [x] #5 Verification and Bandit results are recorded before completion
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write failing read-only repository and CLI tests.
2. Implement repository/search functions and Click commands without adding mutation paths or renaming the console script.
3. Run focused repository/CLI tests and accumulated focused suite.
4. Run live read-only smoke against this repo with before/after backlog status comparison.
5. Run Bandit, diff checks, and two-stage review before finalizing the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Controller verification 2026-05-10:
- Re-ran focused read-only repository/CLI tests: 12 passed before controller regression, then added explicit BACKLOG_CWD isolation coverage.
- Added a red regression test proving ReadOnlyRepository.from_path was incorrectly redirected by BACKLOG_CWD; it failed with TASK-99 instead of fixture TASK-1.
- Fixed ReadOnlyRepository.from_path to call discover_project with explicit_cwd.
- Re-ran focused read-only repository/CLI tests: 13 passed.
- Re-ran package-local read-only repository/CLI tests from tools/backlog-py: 13 passed.
- Re-ran accumulated focused suite: inventory + oracle + project + parser + read-only repository + CLI -> 32 passed.
- Re-ran Bandit: python -m bandit -r tools/backlog-py/src -f json -o /tmp/bandit_backlog_py_task5.json -> exit 0 with results: [].
- Re-ran live smoke after the explicit-cwd fix: task list --plain, task TASK-244.2 --plain, search Backlog.md --plain, board, and config list all exited 0. Before/after git status --short -- backlog snapshots matched exactly.
- Re-ran git diff --check -> exit 0.

Review closeout 2026-05-10:
- Spec-compliance review approved with no missing Task 5 requirements or extra scope.
- Code-quality review approved with no blockers. Deferred non-blocking polish: add explicit _project return type, translate missing task KeyError to a ClickException, and consider avoiding duplicate list_tasks parsing in board() if repo size becomes noticeable.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the read-only repository and CLI slice for the Backlog.md Python compatibility clone. Added task list/view/search/board repository operations, deterministic search, top-level --cwd/help, read-only CLI commands for task list/view/search/board/config list, and tests for no-write behavior, dotted IDs, package-local execution, and BACKLOG_CWD isolation. Latest verification: read-only repository/CLI tests passed 13/13, package-local read-only tests passed 13/13, the accumulated focused suite passed 32/32, live smoke for task list/view/search/board/config exited 0 with unchanged backlog status snapshots, Bandit reported no findings, diff checks passed, and both spec/code-quality reviews approved.
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

## Notes

<!-- SECTION:NOTES:BEGIN -->
- 2026-05-10: Wrote red repository/CLI tests first. Initial focused run failed on missing `backlog_py.core.repository`, then implementation made the focused tests pass.
- 2026-05-10: Live smoke found dotted task ID sorting failure (`TASK-244.6` mixed sort key). Added regression test `test_repository_sorts_dotted_task_ids`, verified it failed, fixed the sort key, and reran verification.
- 2026-05-10: Verification passed: focused read-only tests 12 passed; accumulated focused suite 31 passed; package-local read-only tests 12 passed; Bandit `/tmp/bandit_backlog_py_task5.json` reported 0 results; `git diff --check` exited 0.
- 2026-05-10: Live read-only smoke used this worktree with `task list --plain`, `task TASK-244.2 --plain`, `search "Backlog.md" --plain`, and `config list`; before/after `git status --short -- backlog` snapshots matched exactly.
- 2026-05-10: Known skip: no commit created, per controller instruction. Final summary intentionally left unchecked for controller finalization.
<!-- SECTION:NOTES:END -->
