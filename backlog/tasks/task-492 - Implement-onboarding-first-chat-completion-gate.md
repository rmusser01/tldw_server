---
id: TASK-492
title: Implement onboarding first-chat completion gate
status: In Progress
references:
- TASK-489
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md
documentation:
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md#task-4-first-chat-verification-and-completion-gate
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md#task-5-ingest-defaults-audio-defaults-and-optional-advanced-api-shape
modified_files:
- tldw_Server_API/app/core/Setup/first_chat_verifier.py
- tldw_Server_API/app/api/v1/schemas/setup_schemas.py
- tldw_Server_API/app/api/v1/endpoints/setup.py
- tldw_Server_API/app/core/Setup/setup_manager.py
- tldw_Server_API/tests/Setup/test_setup_first_chat_completion.py
- tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py
- tldw_Server_API/tests/Setup/test_setup_manager_user_db_base_dir_validation.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task 4-5 backend slice from the unified onboarding plan. Add first-chat verification, completion endpoint requiring first chat plus required acknowledgements, and lightweight ingest/audio/advanced first-run setting endpoints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 First-chat verification returns safe success/failure response shapes
- [ ] #2 Setup completion rejects missing first chat and missing required acknowledgements
- [ ] #3 Ingest/audio/optional-advanced first-run endpoints persist safe state without blocking first chat
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
