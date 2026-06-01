---
id: TASK-492
title: Implement onboarding first-chat completion gate
status: Done
references:
- TASK-489
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md
documentation:
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md#task-4-first-chat-verification-and-completion-gate
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md#task-5-ingest-defaults-audio-defaults-and-optional-advanced-api-shape
modified_files:
- tldw_Server_API/app/core/Setup/first_chat_verifier.py
- tldw_Server_API/app/core/Setup/first_run_state.py
- tldw_Server_API/app/core/Setup/setup_manager.py
- tldw_Server_API/app/api/v1/schemas/setup_schemas.py
- tldw_Server_API/app/api/v1/endpoints/setup.py
- tldw_Server_API/tests/Setup/test_first_run_state.py
- tldw_Server_API/tests/Setup/test_setup_first_chat_completion.py
- tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task 4-5 backend slice from the unified onboarding plan. Add first-chat verification, completion endpoint requiring first chat plus required acknowledgements, and lightweight ingest/audio/advanced first-run setting endpoints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 First-chat verification returns safe success/failure response shapes
- [x] #2 Setup completion rejects missing first chat and missing required acknowledgements
- [x] #3 Ingest/audio/optional-advanced first-run endpoints persist safe state without blocking first chat
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented first-chat verification through the chat completion path, first-run completion gating, and save endpoints for ingest defaults, audio defaults, and optional advanced choices. Completion now requires a successful first chat plus persisted required wizard step data; bare acknowledgement lists cannot manufacture missing step data.

Hardened public first-run state persistence and projection: secrets, token-like values, arbitrary local paths, unsafe nested keys/values, unsafe first-chat metadata, and unsafe skip reasons are rejected, stripped, or hidden. Ingest allowed local roots are validated by setup_manager but not persisted or returned in first-run state. Generic /setup/config writes now reject setup lifecycle flags so setup_completed and enable_first_time_setup cannot bypass onboarding gates.

Completion finalization writes first-run completed state before setting the legacy setup flag, under the first-run state lock, and rolls first-run state back if the legacy write fails.

Verification: focused setup suite passed 115 tests; adjacent setup/provider suite passed 141 tests; Ruff passed on touched setup production/test files; Bandit on touched setup production files reported 0 findings; git diff --check passed. Final spec review and final code-quality/security review both passed.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added backend first-chat verification and first-run completion endpoints for the solo onboarding flow. First-run completion is now backend-authoritative and requires an actual successful chat plus persisted wizard defaults for setup path, privacy/security, provider, ingest, audio, and optional advanced choices. The setup state surface now filters sensitive data consistently and blocks config-level lifecycle bypasses.

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
