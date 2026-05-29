---
id: TASK-368.1
title: Implement llama.cpp admin config facade
status: Done
assignee:
  - Codex
created_date: '2026-05-15 03:42'
updated_date: '2026-05-29 04:25'
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
references:
  - https://github.com/rmusser01/tldw_server/pull/1727
modified_files:
  - tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py
  - tldw_Server_API/app/core/Local_LLM/llamacpp_config_service.py
  - tldw_Server_API/app/api/v1/endpoints/llamacpp.py
  - tldw_Server_API/tests/LLM_Local/test_llamacpp_admin_config_api.py
  - tldw_Server_API/tests/LLM_Local/test_llamacpp_management_api.py
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
Stale tracker closeout after PR #2116 merged: the llama.cpp admin config facade already exists on `origin/dev` from PR #1727 (`726958be39 Improve llama.cpp WebUI server management`). Provenance check: `git show --stat --oneline --no-renames 726958be39 -- tldw_Server_API/app/core/Local_LLM/llamacpp_config_service.py tldw_Server_API/app/api/v1/endpoints/llamacpp.py tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py tldw_Server_API/tests/LLM_Local/test_llamacpp_admin_config_api.py` shows the config service, admin schemas, llama.cpp endpoint expansion, and focused config API tests were added there.

Verified current `origin/dev` behavior in a fresh worktree. `GET /api/v1/llamacpp/config`, `PUT /api/v1/llamacpp/config`, and `POST /api/v1/llamacpp/validate` are implemented through `llamacpp_config_service`, typed schemas, and the existing llama.cpp endpoint router. Config writes go through `setup_manager.update_config()` under the llama.cpp config write lock and refresh config caches; binary validation checks the selected local binary without starting a managed server.

Verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_admin_config_api.py tldw_Server_API/tests/LLM_Local/test_llamacpp_management_api.py -q` passed with 46 tests and 5 warnings.

Bandit was not rerun for this closeout-only PR because this slice changes only Backlog metadata. PR #1727 carried the implementation; this closeout records current focused test evidence.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TASK-368.1 is closed as already implemented and currently verified. The backend config facade, typed config update path, environment override reporting, restart-required state, and binary validation endpoints are present on `origin/dev`; the focused llama.cpp config and management API tests pass.
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
