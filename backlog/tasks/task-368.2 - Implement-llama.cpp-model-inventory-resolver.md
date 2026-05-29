---
id: TASK-368.2
title: Implement llama.cpp model inventory resolver
status: Done
assignee:
  - Codex
created_date: '2026-05-15 03:42'
updated_date: '2026-05-29 04:29'
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
references:
  - https://github.com/rmusser01/tldw_server/pull/1727
  - https://github.com/rmusser01/tldw_server/pull/1764
modified_files:
  - tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py
  - tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py
  - tldw_Server_API/app/core/Local_LLM/LlamaCpp_Handler.py
  - tldw_Server_API/app/api/v1/endpoints/llamacpp.py
  - tldw_Server_API/app/core/config.py
  - tldw_Server_API/Config_Files/config.txt
  - tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py
  - tldw_Server_API/tests/Local_LLM/test_llamacpp_handler.py
  - tldw_Server_API/tests/Local_LLM/test_llamacpp_hardening.py
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

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
This is a stale tracker closeout after PR #2117 merged. The llama.cpp model inventory resolver implementation is already present on current `origin/dev`; this task record now reflects the shipped state rather than introducing new backend code.

Implementation provenance:
- PR #1727 (`726958be39 Improve llama.cpp WebUI server management`) added the llama.cpp admin endpoint foundation, `start_server_by_path`, and handler hardening tests.
- PR #1764 (`560c8e17b3 Implement llama.cpp asset inventory v2`) added `llamacpp_inventory_service.py`, inventory schemas, `/api/v1/llamacpp/inventory`, `/api/v1/llamacpp/models/register-path`, `/api/v1/llamacpp/start-by-model`, and focused API tests.

Verified behavior:
- `GET /api/v1/llamacpp/inventory` returns bounded GGUF inventory with stable IDs and warning channels.
- `POST /api/v1/llamacpp/models/register-path` persists validated explicit GGUF paths.
- `POST /api/v1/llamacpp/start-by-model` resolves `model_id` to a validated path and starts through the managed handler.
- Legacy `/api/v1/llamacpp/start_server` filename-based behavior and hardening tests remain covered.

Verification command:
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py tldw_Server_API/tests/Local_LLM/test_llamacpp_handler.py tldw_Server_API/tests/Local_LLM/test_llamacpp_hardening.py -q`
- Result: 58 passed, 6 warnings in 4.00s.

Known skips and noise:
- After pytest reported pass, Loguru emitted cleanup-time `ValueError: I/O operation on closed file` messages from `handler_utils.py`. This appears to be post-test logging cleanup noise and was not introduced by this Backlog-only closeout.
- Bandit was not rerun for this closeout branch because it changes only Backlog metadata. The implementation code shipped in the referenced PRs.
<!-- SECTION:NOTES:END -->

## Final Summary
<!-- SECTION:SUMMARY:BEGIN -->
Closed `TASK-368.2` against the implementation already merged into `dev`. The current backend exposes llama.cpp inventory, explicit path registration, and start-by-model workflows with focused regression coverage.
<!-- SECTION:SUMMARY:END -->
