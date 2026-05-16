---
id: TASK-410
title: Implement llama.cpp supervisor mmproj launch wiring
status: Done
labels:
- llamacpp
- backend
- profiles
modified_files:
- tldw_Server_API/app/core/Local_LLM/llamacpp_profile_capabilities.py
- tldw_Server_API/app/core/Local_LLM/llamacpp_supervisor_service.py
- tldw_Server_API/tests/LLM_Local/test_llamacpp_supervisor_service.py
references:
- https://github.com/rmusser01/tldw_server/pull/1788
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 2 from Docs/superpowers/plans/2026-05-16-llamacpp-model-family-mmproj-profile-wiring-plan.md: resolve managed llama.cpp profiles at supervisor start time, inject resolved mmproj args without mutating persisted profiles, reject invalid vision/mmproj launches, and add focused supervisor/runtime API coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-16-llamacpp-model-family-mmproj-profile-wiring-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Supervisor starts now use the llama.cpp profile capability resolver, injecting resolved mmproj paths at launch time and rejecting invalid vision/mmproj configurations before process start. Added focused async supervisor coverage for valid mmproj injection, missing projector, wrong-kind projector asset, outside-allowlist manual mmproj, resolved_args propagation, and non-mutation of persisted profile server_args. Verification: RED check failed as expected before implementation; focused pytest passed with 41 tests; git diff --check passed; Bandit on touched backend files passed with zero findings.
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
