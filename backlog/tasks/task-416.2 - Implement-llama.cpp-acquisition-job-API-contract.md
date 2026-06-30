---
id: TASK-416.2
title: Implement llama.cpp acquisition job API contract
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-17 21:56
labels:
- llamacpp
- backend
- local-llm
- jobs
dependencies: []
documentation:
- Docs/superpowers/plans/2026-05-16-llamacpp-model-acquisition-import-workflows-plan.md
parent_task_id: TASK-416
priority: high
modified_files:
- tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py
- tldw_Server_API/app/core/Local_LLM/llamacpp_acquisition_service.py
- tldw_Server_API/app/core/Local_LLM/llamacpp_acquisition_jobs.py
- tldw_Server_API/app/api/v1/endpoints/llamacpp.py
- tldw_Server_API/app/core/Jobs/manager.py
- tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_service.py
- tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_api.py
- tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py
- backlog/tasks/task-416.2 - Implement-llama.cpp-acquisition-job-API-contract.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 2 from the llama.cpp model acquisition/import workflow plan: add safe remote download request validation plus admin-only Jobs-backed acquisition API/status/cancel endpoints without adding the download worker yet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Acquisition service rejects unsupported schemes, unsafe local/private sources by default, traversal/delimiter destinations, and unsafe filenames.
- [x] #2 Acquisition service redacts credentials/secrets from source labels and resolves destinations only under configured models_dir or allowed_paths.
- [x] #3 Acquisition service exposes partial-path, completed-download validation, and completed asset registration helpers for the later worker slice.
- [x] #4 Acquisition job helper creates/list/status/cancel mappings for domain llama.cpp acquisition jobs without storing raw credentials.
- [x] #5 Admin-only API endpoints create/list/status/cancel acquisition jobs and AuthNZ permission coverage includes all new endpoints.
- [x] #6 Focused acquisition service/API/AuthNZ tests, git diff checks, and Bandit on touched Python scope are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-16-llamacpp-model-acquisition-import-workflows-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the llama.cpp asset acquisition job API contract without adding a download worker. Remote acquisition requests now validate allowed schemes, private/local host policy, destination allowlists, filename safety, size/checksum metadata, and credential/secret URL policy before creating a Jobs row. Admin endpoints can create, list, inspect, and cancel acquisition jobs, and tests cover service validation, API job storage, credential hygiene, and admin-only access.

Verification: source .venv/bin/activate && python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_service.py tldw_Server_API/tests/LLM_Local/test_llamacpp_acquisition_api.py tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py -q --tb=short (61 passed, 5 warnings); git diff --check; Bandit on touched production paths wrote /tmp/bandit_llamacpp_acquisition_review_fix.json with 0 results.
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
