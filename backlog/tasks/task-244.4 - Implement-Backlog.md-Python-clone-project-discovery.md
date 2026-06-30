---
id: TASK-244.4
title: Implement Backlog.md Python clone project discovery
status: Done
assignee:
  - codex
created_date: '2026-05-10 22:07'
updated_date: '2026-05-10 22:56'
labels: []
dependencies:
  - TASK-244.3
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
Implement Task 3 from the Backlog.md Python compatibility clone implementation plan. Add project/config dataclasses, YAML config loading with snake_case and camelCase compatibility, and project discovery for supported Backlog.md config shapes with BACKLOG_CWD and explicit cwd precedence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Project/config dataclasses exist for loaded Backlog.md project metadata
- [x] #2 Config loader safely reads YAML and maps snake_case plus camelCase keys with defaults
- [x] #3 Project discovery supports backlog.config.yml, backlog/config.yml, .backlog/config.yml, BACKLOG_CWD, and explicit cwd precedence
- [x] #4 Focused project discovery tests are written red-first and pass after implementation
- [x] #5 Verification and Bandit results are recorded before completion
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write focused project discovery/config tests and verify the missing storage module failure.
2. Implement core model dataclasses and config/project storage functions.
3. Cover root config, backlog/config.yml, .backlog/config.yml, snake_case and camelCase keys, no-git style flags, BACKLOG_CWD, and explicit cwd precedence.
4. Run focused tests, Bandit on touched source, and diff checks.
5. Run spec-compliance and code-quality review before finalizing the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- 2026-05-10: Red-first project discovery test run captured the expected missing package failure: `ModuleNotFoundError: No module named 'backlog_py.storage'`.
- 2026-05-10: Added `BacklogConfig` and `BacklogProject` dataclasses plus read-only YAML config loading with snake_case and camelCase key aliases.
- 2026-05-10: Added project discovery for root `backlog.config.yml`, `backlog/config.yml`, `.backlog/config.yml`, `BACKLOG_CWD`, and explicit cwd precedence.
- 2026-05-10 verification: `source .venv/bin/activate && python -m pytest tools/backlog-py/tests/test_project_discovery.py -v` passed: 8 passed, 2 warnings.
- 2026-05-10 verification: `source .venv/bin/activate && python -m pytest tools/backlog-py/tests/test_inventory.py tools/backlog-py/tests/test_oracle_manifest.py tools/backlog-py/tests/test_project_discovery.py -v` passed: 12 passed, 2 warnings.
- 2026-05-10 verification: `source .venv/bin/activate && python -m bandit -r tools/backlog-py/src -f json -o /tmp/bandit_backlog_py_task3.json` passed with 0 results and 0 errors.
- 2026-05-10 verification: `git diff --check` passed with no output.
- Known skip: no commit created per user instruction; Task 3 plan Step 6 remains unchecked.

Controller verification 2026-05-10:
- Re-ran focused project discovery tests from repo root: 8 passed before parser hardening.
- Added a red test for malformed quoted boolean config values; it failed because bool("false") did not raise.
- Replaced bool/int/string coercion with strict typed config helpers, then re-ran project discovery tests: 9 passed.
- Verified package-directory execution from tools/backlog-py: python -m pytest tests/test_project_discovery.py -q -> 9 passed.
- Re-ran inventory + oracle + project discovery focused tests -> 13 passed.
- Re-ran Bandit: python -m bandit -r tools/backlog-py/src -f json -o /tmp/bandit_backlog_py_task3.json -> exit 0 with results: [].
- Re-ran git diff --check -> exit 0.

Review closeout 2026-05-10:
- Spec-compliance review approved with no missing Task 3 requirements or extra scope.
- Code-quality review approved with no blockers. Deferred non-blocking hardening: reject non-string list items instead of coercing, consider tuple fields for deeper immutability, and add nested-cwd discovery regression later when CLI/MCP entrypoints depend on discovery.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented project discovery and config loading for the Backlog.md Python compatibility clone. Added `BacklogConfig`/`BacklogProject`, safe YAML config loading with snake_case and camelCase aliases, strict scalar/boolean/integer validation, project discovery for `backlog.config.yml`, `backlog/config.yml`, `.backlog/config.yml`, `BACKLOG_CWD`, and explicit cwd precedence. Latest verification: project discovery tests passed 9/9 from both repo root and package directory, the inventory+oracle+project focused suite passed 13/13, Bandit reported no findings, diff checks passed, and both spec/code-quality reviews approved.
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
