---
id: TASK-368.1
title: Implement llama.cpp admin config facade
status: Done
assignee: []
created_date: '2026-05-15 03:42'
updated_date: '2026-05-15 04:42'
labels:
  - implementation
  - backend
  - llamacpp
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-15-llamacpp-server-management-webui-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-llamacpp-server-management-webui-implementation-plan.md
parent_task_id: TASK-368
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first backend slice from the implementation plan: typed llama.cpp admin config endpoints, saved-vs-active runtime state, restart-required semantics, environment override reporting, comment-preserving config writes, and binary validation. Do not implement inventory, provider wiring, hardware, logs, or frontend changes in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 GET and PUT /api/v1/llamacpp/config expose saved config, active config, restart-required reasons, warnings, and env override state.
- [x] #2 Config updates use the existing comment-preserving setup config writer and refresh config caches.
- [x] #3 POST /api/v1/llamacpp/validate reports binary validation results without starting a server.
- [x] #4 Focused backend tests for the config facade and existing management API pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started subagent-driven implementation for the backend config facade slice in worktree .worktrees/codex-llamacpp-webui-management on branch codex/llamacpp-webui-management. Baseline focused tests passed: test_llamacpp_management_api.py and test_llamacpp_handler.py, 37 passed with existing Loguru shutdown noise after pytest exit.

Implemented across commits 85d0a29a1, e3c5aa6d7, 0c11c38d1, and 74dc285b6. Review gates found and fixed binary probe hardening, malformed config parsing, explicit nullable integer clears, safe config-write errors, and restart-required honesty for runtime config fields.

Verification recorded: focused backend pytest 39 passed / 5 warnings for test_llamacpp_admin_config_api.py and test_llamacpp_management_api.py; implementer Bandit runs on touched backend paths were clean; git diff --check passed. Code-quality reviewer approved Task 1 at head 74dc285b6.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the llama.cpp admin config facade with typed GET/PUT config and POST validate endpoints. The facade distinguishes saved config from active handler state, reports restart reasons and environment overrides, persists updates through the existing comment-preserving setup writer, refreshes config caches, and keeps binary validation stat-only unless an explicitly requested probe targets the saved/active configured executable. Follow-up review fixes hardened arbitrary binary execution, malformed saved config handling, nullable scalar clears through setup validation, safe write-error responses, and restart-required detection for runtime/security launch fields.

Verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_admin_config_api.py tldw_Server_API/tests/LLM_Local/test_llamacpp_management_api.py -q -> 39 passed, 5 warnings. Bandit on touched backend paths clean in implementer runs. git diff --check passed. Spec and code-quality review gates both approved.
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
