---
id: TASK-244.2
title: Implement Backlog.md Python clone inventory scaffold
status: Done
assignee:
  - codex
created_date: '2026-05-10 21:13'
updated_date: '2026-05-10 21:54'
labels: []
dependencies:
  - TASK-244.1
references:
  - 'https://github.com/MrLesk/Backlog.md'
  - 'https://raw.githubusercontent.com/MrLesk/Backlog.md/main/CLI-INSTRUCTIONS.md'
  - 'https://raw.githubusercontent.com/MrLesk/Backlog.md/main/package.json'
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
Implement Task 1 from the Backlog.md Python compatibility clone implementation plan. Create the isolated `tools/backlog-py` Python package skeleton, minimal importable CLI entrypoint, built-in compatibility inventory, and focused inventory tests. This first executable slice must not cut over the `backlog` command or mutate live Backlog.md data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `tools/backlog-py` package skeleton exists with importable `backlog_py` package and `backlog-py` console script target
- [x] #2 Built-in compatibility inventory includes agent-critical CLI/MCP entries and browser/interactive deferrals from the plan
- [x] #3 Focused inventory tests are written red-first and pass after implementation
- [x] #4 No live `backlog` command cutover or live Backlog.md mutation is introduced
- [x] #5 Verification and Bandit results are recorded before completion
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Set up an isolated worktree branch for this implementation slice, preserving the dirty main checkout.
2. Follow Task 1 from Docs/superpowers/plans/2026-05-10-backlog-md-python-compatibility-clone-implementation-plan.md using TDD: write inventory tests first, verify they fail, then create the package skeleton and inventory implementation.
3. Keep the console command as `backlog-py`; do not alter the existing `backlog` command or mutate live Backlog.md data.
4. Run focused tests, Bandit on `tools/backlog-py/src`, and diff checks.
5. Run spec-compliance and code-quality review before finalizing the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation update 2026-05-10:
- Created isolated tools/backlog-py package scaffold with importable backlog_py package, backlog-py console script target, minimal Click CLI, and package README.
- Added built-in compatibility inventory covering agent-critical plain CLI commands, MCP workflow/search entries, and explicit browser/interactive deferrals.
- Verification: backlog-py --help exits 0; python -m pytest tools/backlog-py/tests/test_inventory.py -v reports 2 passed; Bandit JSON at /tmp/bandit_backlog_py_task1.json reports 0 findings; git diff --check exits 0.
- TDD note: original implementer red-phase output was unavailable after subagent shutdown, so controller performed a controlled red/green check by temporarily removing mcp:task-search, observing test_inventory_starts_with_agent_critical_commands fail, restoring the item, and rerunning the focused tests green.
- No live backlog command cutover or live Backlog.md mutation path was added; only the local task record was updated through the Backlog CLI fallback because MCP has no worktree selector.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the first Backlog.md Python compatibility clone slice under tools/backlog-py. The package is importable as backlog_py, exposes only a backlog-py console script, includes a built-in compatibility inventory for the initial agent-critical CLI/MCP commands plus explicit browser/interactive deferrals, and has focused inventory tests. Verification completed with CLI help, controlled red/green test sensitivity, focused pytest, Bandit with zero findings, diff checks, and spec/code-quality review approvals. No live backlog command cutover or live data mutation was introduced.
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
