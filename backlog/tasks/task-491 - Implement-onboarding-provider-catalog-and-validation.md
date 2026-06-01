---
id: TASK-491
title: Implement onboarding provider catalog and validation
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-31 09:54
labels: []
dependencies: []
references:
- TASK-489
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md
documentation:
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md#task-3-provider-catalog-config-writes-and-validation
modified_files:
- tldw_Server_API/app/core/Setup/provider_catalog.py
- tldw_Server_API/app/core/Setup/provider_validation.py
- tldw_Server_API/app/core/Setup/setup_manager.py
- tldw_Server_API/app/core/config.py
- tldw_Server_API/app/api/v1/schemas/setup_schemas.py
- tldw_Server_API/app/api/v1/endpoints/setup.py
- tldw_Server_API/tests/Setup/test_setup_provider_catalog.py
- tldw_Server_API/tests/Setup/test_setup_provider_validation.py
- tldw_Server_API/tests/Setup/test_setup_manager_provider_field_insertion.py
- tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py
- tldw_Server_API/tests/Config/test_config_providers_endpoints.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task 3 slice from the unified onboarding plan. Add backend-generated setup provider catalog, provider save response contract, local endpoint validation, and provider setup endpoints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Provider catalog covers PRD provider keys and marks local endpoint providers correctly
- [x] #2 Provider save endpoint masks secrets and returns a typed saved/failed response
- [x] #3 Local OpenAI-compatible endpoint validation maps unreachable/auth/API-shape failures safely
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Task 3 subagent-driven slice after TASK-490 cleared spec and code-quality review at 3bd11d4e3f805cce849759319c0407c8711fb7dd. Scope: backend-generated setup provider catalog, provider save/validation schemas and endpoints, local OpenAI-compatible endpoint diagnostics, and regression coverage. Task 3 must reuse `_require_first_run_write_access` from Task 2 for setup writes.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 3 complete. Added backend-generated first-run provider catalog and provider save/validate endpoints for hosted and local providers; provider saves mask secrets, reject blank hosted keys, refresh runtime config caches, and write new catalog fields into the correct config section. Added runtime config mappings for Moonshot, Z.AI, Kobold.cpp, and TabbyAPI; added local endpoint validation for OpenAI-compatible providers plus native Kobold.cpp validation; added SSRF-style target guarding for local provider validation. Hardened legacy /setup/config and /setup/complete behind first-run write/completion gates so they share backend-authoritative state and cannot bypass first-chat completion. Verification: provider/config pytest set passed with 110 passed; setup guard/state/masking set passed with 34 passed; focused provider-field regression passed with 4 passed; Ruff passed on changed files; Bandit passed on latest touched production files; full touched-scope Bandit only reported pre-existing config.py low B105 findings outside changed lines; git diff --check passed. Spec and code-quality reviews are approved.
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
