---
id: TASK-416.1
title: Implement llama.cpp local import preview contract
status: Done
labels:
- llamacpp
- backend
- local-llm
priority: high
parent_task_id: TASK-416
documentation:
- Docs/superpowers/plans/2026-05-16-llamacpp-model-acquisition-import-workflows-plan.md
modified_files:
- tldw_Server_API/app/api/v1/schemas/llamacpp_admin_schemas.py
- tldw_Server_API/app/core/Local_LLM/llamacpp_inventory_service.py
- tldw_Server_API/app/api/v1/endpoints/llamacpp.py
- tldw_Server_API/tests/LLM_Local/test_llamacpp_asset_inventory_service.py
- tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py
- tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 1 from the llama.cpp model acquisition/import workflow plan: add a non-mutating local asset-folder import preview service/API, richer preview result contract, and permission coverage without changing profile creation/start/use-in-chat behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Service preview summarizes GGUF/mmproj/folder assets, warnings, counts, and scan-limit state without persisting imported folders.
- [x] #2 Preview fails closed for non-existent paths, file paths, and folders outside allowed llama.cpp paths.
- [x] #3 API endpoint POST /api/v1/llamacpp/assets/import-folder/preview returns the preview response, maps ServerError to HTTP 400, and does not mutate config.
- [x] #4 AuthNZ permission coverage includes the new admin-only preview endpoint.
- [x] #5 Focused inventory service/API/AuthNZ tests, git diff checks, and Bandit on touched Python scope are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-16-llamacpp-model-acquisition-import-workflows-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added `LlamaCppAssetImportPreviewResponse` plus `preview_import_asset_folder()` using the existing canonicalization, config safety, allowlist, asset classification, mmproj pairing, warning, and bounded scan helpers. Added `POST /api/v1/llamacpp/assets/import-folder/preview` as an admin-only, rate-limited endpoint that runs inventory work in the threadpool and maps `ServerError` to HTTP 400. Added service/API/AuthNZ coverage for non-mutating preview behavior, fail-closed path handling, endpoint auth, and threadpool offload.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the local asset-folder import preview contract for llama.cpp without changing the existing persistent import path. Verification: focused pytest set passed (60 passed); git diff --check passed; Bandit on touched production Python scope passed with zero findings. Known skips/blockers: none.
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
