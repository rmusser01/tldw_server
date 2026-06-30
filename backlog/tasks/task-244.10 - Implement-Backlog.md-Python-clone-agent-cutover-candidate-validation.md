---
id: TASK-244.10
title: Implement Backlog.md Python clone agent cutover candidate validation
status: Done
assignee: []
created_date: '2026-05-11 00:59'
labels: []
dependencies:
  - TASK-244.9
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
Implement Task 9 from the Backlog.md Python compatibility clone implementation plan. Add the agent-critical parity matrix, expand inventory and oracle manifest coverage, and validate that every agent-critical CLI/MCP operation is either covered by a fixture-backed golden requirement or explicitly documented as deferred.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Agent-critical parity document lists CLI and MCP operations with status and fixture coverage
- [x] #2 Inventory and oracle manifest represent every agent-critical operation
- [x] #3 Matrix regression test fails when a golden-required operation lacks fixture coverage
- [x] #4 README documents the agent cutover validation gate
- [x] #5 Full tests Bandit diff checks copied-repo mutation smoke and review are completed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `tools/backlog-py/docs/agent-critical-parity.md` with implemented golden requirements and explicit deferred blockers for browser, interactive, completion, hook, and git behavior.
- Expanded `CompatibilityItem` metadata with expected operation text, status, fixture name, and deferred reason.
- Expanded the oracle manifest so implemented agent-critical operations are fixture-backed `golden-required` entries and deferred blockers are explicit non-agent-critical entries.
- Added `tools/backlog-py/tests/test_agent_critical_matrix.py` to lock the expected cutover scope, fixture coverage, manifest/inventory consistency, and matrix doc coverage.
- Updated `tools/backlog-py/README.md` with the agent cutover validation gate and mutation-smoke warning.
- RED command: `source .venv/bin/activate && python -m pytest tools/backlog-py/tests/test_agent_critical_matrix.py -v`; result: expected failures for missing inventory entries and missing matrix doc.
- Focused GREEN command: `source .venv/bin/activate && python -m pytest tools/backlog-py/tests/test_agent_critical_matrix.py tools/backlog-py/tests/test_inventory.py tools/backlog-py/tests/test_oracle_manifest.py -v`; result: `8 passed, 2 warnings`.
- Full validation command: `source .venv/bin/activate && python -m pytest tools/backlog-py/tests -v`; result: `92 passed, 2 warnings`.
- Security regression command: `source .venv/bin/activate && python -m pytest tools/backlog-py/tests/test_security_paths.py -v`; result: `4 passed, 2 warnings`.
- Bandit command: `source .venv/bin/activate && python -m bandit -r tools/backlog-py/src -f json -o /tmp/bandit_backlog_py_task9_final.json`; result: zero findings.
- Diff check command: `git diff --check`; result: clean.
- Copied-repo mutation smoke command used a `mktemp -d` copy of `backlog`, ran `backlog-py --cwd "$tmpdir" task create "Temporary smoke task" --status "To Do" --plain` and `backlog-py --cwd "$tmpdir" task list --plain`; result: exit 0 and no live repository smoke task file.
- Self-review checked the inventory, oracle manifest, matrix doc, README, plan update, and task record for consistency with Task 9 acceptance criteria.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Backlog.md Python clone agent cutover candidate validation gate. The slice now has a documented parity matrix, expanded inventory and oracle manifest coverage, regression tests for matrix drift and fixture coverage, README guidance, copied-repo mutation smoke evidence, and passing pytest/Bandit/diff verification.
<!-- SECTION:FINAL_SUMMARY:END -->
