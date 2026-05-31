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
- tldw_Server_API/app/api/v1/schemas/setup_schemas.py
- tldw_Server_API/app/api/v1/endpoints/setup.py
- tldw_Server_API/tests/Setup/test_setup_provider_catalog.py
- tldw_Server_API/tests/Setup/test_setup_provider_validation.py
- tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py
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

<!-- SECTION:NOTES:BEGIN -->
Started Task 3 subagent-driven slice after TASK-490 cleared spec and code-quality review at 3bd11d4e3f805cce849759319c0407c8711fb7dd. Scope: backend-generated setup provider catalog, provider save/validation schemas and endpoints, local OpenAI-compatible endpoint diagnostics, and regression coverage. Task 3 must reuse `_require_first_run_write_access` from Task 2 for setup writes.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 3 spec review fixes completed. Added hosted provider credential presence/syntax validation without external API calls, rejected blank hosted API keys on first-run provider save before config writes, mapped malformed local endpoint URLs to sanitized typed validation failures, and hardened secret masking so one- and two-character nonempty secrets are never fully exposed. Verification: provider/setup/config pytest group passed with 84 passed; setup guard/state/masking pytest group passed with 33 passed; Ruff passed; Bandit JSON reported zero findings; git diff --check passed.
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
