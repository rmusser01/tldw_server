---
id: TASK-439
title: Add VZ host reboot evidence directory helpers
status: Done
documentation:
- Docs/superpowers/plans/2026-05-19-vz-helper-host-reboot-validation.md
modified_files:
- tools/macos-vz-helper/scripts/vz-helperctl.py
- tools/macos-vz-helper/Tests/test_vz_helperctl.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 1 from Docs/superpowers/plans/2026-05-19-vz-helper-host-reboot-validation.md: add host reboot evidence directory validation helpers and preserve helper generation details in ping state. Scope is limited to vz-helperctl.py and its focused tests; no pre/post manifest writing or CLI wiring.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented host reboot evidence directory validation helpers in vz-helperctl.py and added focused tests for private-directory enforcement, volatile-root rejection, and string-only ping helper details preservation. Verification: RED tests failed before implementation for missing helper/constant/details support; GREEN focused pytest passed 3 selected tests; full helperctl test file passed 145 tests with 6 skips; Bandit on the touched script reported 0 results.
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
