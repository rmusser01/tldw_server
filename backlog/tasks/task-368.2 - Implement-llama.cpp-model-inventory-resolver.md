---
id: TASK-368.2
title: Implement llama.cpp model inventory resolver
status: Done
assignee: []
created_date: '2026-05-15 03:42'
updated_date: '2026-05-15 06:02'
labels:
  - implementation
  - backend
  - llamacpp
dependencies:
  - TASK-368.1
documentation:
  - Docs/superpowers/specs/2026-05-15-llamacpp-server-management-webui-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-llamacpp-server-management-webui-implementation-plan.md
parent_task_id: TASK-368
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the inventory/start-by-model backend slice from the implementation plan. Add safe recursive GGUF inventory, registered local model paths, stable model IDs, and a handler path-start helper while preserving the existing filename-based start_server endpoint.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 GET /api/v1/llamacpp/inventory returns bounded recursive GGUF inventory with stable model IDs and warnings.
- [x] #2 POST /api/v1/llamacpp/models/register-path persists explicit local GGUF paths safely through allowed config keys.
- [x] #3 POST /api/v1/llamacpp/start-by-model resolves model_id to a validated path and starts through the managed handler.
- [x] #4 Existing /api/v1/llamacpp/start_server filename behavior and path hardening continue to pass tests.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started subagent-driven implementation for the backend inventory resolver and start-by-model slice after TASK-368.1 passed spec and code-quality review. Scope is limited to Task 2 backend files and tests from the implementation plan.

Implemented across commits b0e41d827, dc2c5fe41, 883eecb3e, and 7585e96d3. Review gates found and fixed start-by-model allowlist validation, safe canonicalization for pathological paths, registered-path visibility under scan limits, shared llama.cpp config write locking, config-owned lock path, stale env override documentation, and sanitized start-by-model startup errors.

Verification recorded: Task 2 focused pytest passed at 49 tests / 6 warnings for inventory, handler, and hardening suites; Task 1 regression pytest passed at 41 tests / 5 warnings in implementer run and 39 tests / 5 warnings before the final lock patch, with no failures. Bandit on touched backend source paths was clean in implementer runs. git diff --check passed. Spec review and fresh code-quality re-review approved Task 2 at head 7585e96d3.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the llama.cpp model inventory resolver and start-by-model backend flow. The backend now scans configured and registered local GGUF paths with stable inventory IDs, filename-derived metadata, warning-based item health, and bounded recursive traversal; registered paths persist through the existing setup config writer; and start-by-model resolves inventory IDs to validated allowed GGUF paths before calling the managed handler. The handler gained a path-based start helper while preserving the existing filename-based start_server behavior and traversal hardening.

Review fixes tightened the safety boundary: outside-allowlist inventory rows remain visible but cannot launch, pathological path resolution returns warnings or safe errors, explicit registrations remain visible under scan caps, llama.cpp config writes share a config-owned file lock, stale registered-path env override claims were removed, and start-by-model no longer leaks raw startup stderr to clients.

Verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py tldw_Server_API/tests/Local_LLM/test_llamacpp_handler.py tldw_Server_API/tests/Local_LLM/test_llamacpp_hardening.py -q -> 49 passed, 6 warnings. Task 1 regression pair passed in implementer verification at 41 passed, 5 warnings. Bandit on touched backend paths clean in implementer runs. git diff --check passed. Spec and code-quality review gates both approved.
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
