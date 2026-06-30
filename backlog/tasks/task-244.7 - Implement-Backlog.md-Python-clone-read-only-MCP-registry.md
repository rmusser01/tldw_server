---
id: TASK-244.7
title: Implement Backlog.md Python clone read-only MCP registry
status: Done
assignee:
  - codex
created_date: '2026-05-10 23:30'
updated_date: '2026-05-10 23:44'
labels: []
dependencies:
  - TASK-244.6
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
Implement Task 6 from the Backlog.md Python compatibility clone implementation plan. Add pure read-only MCP resource and tool registry functions backed by the read-only repository, while avoiding unverified MCP SDK/package dependencies because the SDK is not currently installed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCP dependency availability is checked and dependency changes are avoided when unavailable
- [x] #2 Resource registry returns workflow overview and task-workflow alias content
- [x] #3 Pure task_search and task_view tool functions return fixture-backed read-only data
- [x] #4 Unsupported mutation tools return explicit not-implemented errors
- [x] #5 Verification and Bandit results are recorded before completion
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record MCP SDK availability check result and do not add dependencies when unavailable.
2. Write failing pure registry tests for workflow resources, alias, task search/view, and unsupported mutation errors.
3. Implement resource and tool registry functions backed by the read-only repository.
4. Add a server adapter stub with a clear missing-SDK message rather than importing an unavailable dependency.
5. Run focused MCP tests, accumulated focused suite, Bandit, diff checks, and two-stage review before finalizing the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- MCP SDK availability was supplied by the controller as unavailable (`importlib.util.find_spec("mcp") is not None` printed `False`), so no dependency or pyproject changes were made.
- TDD red step: `source .venv/bin/activate && python -m pytest tools/backlog-py/tests/test_mcp_resources.py -v` failed during collection with `ModuleNotFoundError: No module named 'backlog_py.mcp'`.
- Implemented pure read-only MCP resource/tool registry functions and a server stub that checks SDK availability without importing `mcp` at module import time.
- Verification passed:
  - `source .venv/bin/activate && python -m pytest tools/backlog-py/tests/test_mcp_resources.py -v` -> 8 passed, 2 warnings.
  - `source .venv/bin/activate && python -m pytest tools/backlog-py/tests/test_inventory.py tools/backlog-py/tests/test_oracle_manifest.py tools/backlog-py/tests/test_project_discovery.py tools/backlog-py/tests/test_task_parser.py tools/backlog-py/tests/test_readonly_repository.py tools/backlog-py/tests/test_cli_readonly.py tools/backlog-py/tests/test_mcp_resources.py -v` -> 40 passed, 2 warnings.
  - From `tools/backlog-py`: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tests/test_mcp_resources.py -v` -> 8 passed, 2 warnings.
  - `source .venv/bin/activate && python -m bandit -r tools/backlog-py/src -f json -o /tmp/bandit_backlog_py_task6.json` -> exit 0; JSON summary results 0, errors 0.
  - `git diff --check` -> exit 0.
- Known skips/blockers: no live MCP stdio server adapter was implemented because the MCP SDK is unavailable and Task 6 is intentionally pure/read-only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Controller verification 2026-05-10:
- Confirmed pyproject.toml has no diff; no MCP dependency was added because the SDK check returned False.
- Hardened server stub test so it does not permanently assume the SDK is absent; it now verifies either the current missing-SDK message or the future adapter-not-implemented message depending on availability.
- Re-ran focused MCP tests from repo root: 8 passed.
- Re-ran focused MCP tests from tools/backlog-py: 8 passed.
- Re-ran accumulated focused suite: inventory + oracle + project + parser + read-only repository + CLI + MCP -> 40 passed.
- Re-ran Bandit: python -m bandit -r tools/backlog-py/src -f json -o /tmp/bandit_backlog_py_task6.json -> exit 0 with results: [].
- Re-ran git diff --check -> exit 0.

Two-stage review completed 2026-05-10: spec compliance reviewer approved the staged Task 6 scope, confirmed no MCP dependency was added, and found tests sufficient for this stage. Code-quality reviewer approved with no findings and confirmed the MCP package remains pure/read-only without importing mcp at module import time.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a pure read-only MCP registry for the Backlog.py compatibility package: workflow resources, task-workflow alias support, task_search/task_view helpers over ReadOnlyRepository, explicit mutation NotImplementedError stubs, and an SDK-availability server stub. No mcp dependency was added because the SDK is unavailable in the current environment. Verification covered focused MCP tests from repo root and package cwd (8 passed each), the accumulated focused Backlog.py suite (40 passed), Bandit on tools/backlog-py/src with no findings, diff checks, and approved spec/code-quality reviews.
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
