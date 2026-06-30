---
id: TASK-368.3
title: Implement llama.cpp provider wiring and diagnostics
status: Done
assignee:
  - Codex
created_date: '2026-05-15 03:43'
updated_date: '2026-05-29 04:34'
labels:
  - implementation
  - backend
  - llamacpp
dependencies:
  - TASK-368.2
documentation:
  - Docs/superpowers/specs/2026-05-15-llamacpp-server-management-webui-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-llamacpp-server-management-webui-implementation-plan.md
references:
  - https://github.com/rmusser01/tldw_server/pull/1727
  - https://github.com/rmusser01/tldw_server/pull/1848
modified_files:
  - tldw_Server_API/app/api/v1/endpoints/llamacpp.py
  - tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py
  - tldw_Server_API/app/core/Local_LLM/llamacpp_provider_service.py
  - tldw_Server_API/app/core/Local_LLM/llamacpp_hardware_service.py
  - tldw_Server_API/tests/LLM_Local/test_llamacpp_provider_and_logs_api.py
  - tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py
  - tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py
parent_task_id: TASK-368
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the provider diagnostics backend slice from the implementation plan. Add explicit use-in-chat provider wiring, bounded managed log tailing, best-effort hardware snapshot, and permission coverage. Do not change frontend behavior in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 POST /api/v1/llamacpp/use-in-chat explicitly updates only the llama.cpp provider endpoint after a managed server is running.
- [x] #2 GET /api/v1/llamacpp/logs/tail returns bounded managed logs and cannot read arbitrary paths.
- [x] #3 GET /api/v1/llamacpp/hardware returns best-effort RAM/CPU/GPU data with structured warnings and no hard dependency on NVIDIA hardware.
- [x] #4 New endpoints retain admin-only permission coverage and focused backend tests pass.
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
This is a stale tracker closeout after PR #2119 merged. The llama.cpp provider wiring and diagnostics backend slice is already present on current `origin/dev`; this task record now reflects the shipped state rather than introducing runtime code.

Implementation provenance:
- PR #1727 (`726958be39 Improve llama.cpp WebUI server management`) added `llamacpp_provider_service.py`, `llamacpp_hardware_service.py`, `/api/v1/llamacpp/use-in-chat`, `/api/v1/llamacpp/logs/tail`, `/api/v1/llamacpp/hardware`, response schemas, and focused provider/logs/hardware API tests.
- PR #1848 (`8e23caecaa Lock llama.cpp runtime API compatibility`) added runtime compatibility assertions and explicit permission coverage for the llama.cpp diagnostics endpoints.

Verified behavior:
- `POST /api/v1/llamacpp/use-in-chat` persists only `Local-API.llama_api_IP` after a managed runtime is running and reports env override warnings when applicable.
- `GET /api/v1/llamacpp/logs/tail` returns bounded redacted managed logs, rejects arbitrary path reads by design, and uses active managed runtime log state.
- `GET /api/v1/llamacpp/hardware` returns best-effort CPU/RAM/GPU information with structured warnings when optional GPU probing is unavailable.
- New endpoints require admin permissions.

Verification command:
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_provider_and_logs_api.py tldw_Server_API/tests/LLM_Local/test_llamacpp_runtime_api.py tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py -q`
- Result: 88 passed, 5 warnings in 4.76s.

Known skips:
- Bandit was not rerun for this closeout branch because it changes only Backlog metadata. The implementation code shipped in the referenced PRs.
<!-- SECTION:NOTES:END -->

## Final Summary
<!-- SECTION:SUMMARY:BEGIN -->
Closed `TASK-368.3` against the implementation already merged into `dev`. The backend exposes managed llama.cpp chat-provider wiring, bounded log tailing, best-effort hardware diagnostics, and admin-only permission coverage with focused regression tests.
<!-- SECTION:SUMMARY:END -->
