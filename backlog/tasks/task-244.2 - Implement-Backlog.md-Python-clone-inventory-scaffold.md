---
id: TASK-244.2
title: Implement Backlog.md Python clone inventory scaffold
status: In Progress
assignee:
  - codex
created_date: '2026-05-10 21:13'
updated_date: '2026-05-10 21:13'
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
- [ ] #1 `tools/backlog-py` package skeleton exists with importable `backlog_py` package and `backlog-py` console script target
- [ ] #2 Built-in compatibility inventory includes agent-critical CLI/MCP entries and browser/interactive deferrals from the plan
- [ ] #3 Focused inventory tests are written red-first and pass after implementation
- [ ] #4 No live `backlog` command cutover or live Backlog.md mutation is introduced
- [ ] #5 Verification and Bandit results are recorded before completion
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Set up an isolated worktree branch for this implementation slice, preserving the dirty main checkout.
2. Follow Task 1 from Docs/superpowers/plans/2026-05-10-backlog-md-python-compatibility-clone-implementation-plan.md using TDD: write inventory tests first, verify they fail, then create the package skeleton and inventory implementation.
3. Keep the console command as `backlog-py`; do not alter the existing `backlog` command or mutate live Backlog.md data.
4. Run focused tests, Bandit on `tools/backlog-py/src`, and diff checks.
5. Run spec-compliance and code-quality review before finalizing the task.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
