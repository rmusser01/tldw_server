---
id: TASK-491
title: Implement onboarding provider catalog and validation
status: To Do
references:
- TASK-489
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md
documentation:
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md#task-3-provider-catalog-config-writes-and-validation
modified_files:
- tldw_Server_API/app/core/Setup/provider_catalog.py
- tldw_Server_API/app/core/Setup/provider_validation.py
- tldw_Server_API/app/api/v1/schemas/setup_schemas.py
- tldw_Server_API/app/api/v1/endpoints/setup.py
- tldw_Server_API/tests/Setup/test_setup_provider_catalog.py
- tldw_Server_API/tests/Setup/test_setup_provider_validation.py
- tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py
- tldw_Server_API/tests/Config/test_config_providers_endpoints.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task 3 slice from the unified onboarding plan. Add backend-generated setup provider catalog, provider save response contract, local endpoint validation, and provider setup endpoints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Provider catalog covers PRD provider keys and marks local endpoint providers correctly
- [ ] #2 Provider save endpoint masks secrets and returns a typed saved/failed response
- [ ] #3 Local OpenAI-compatible endpoint validation maps unreachable/auth/API-shape failures safely
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
