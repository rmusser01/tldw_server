---
id: TASK-12106
title: Add explicit single-user API key device persistence and relaunch coverage
status: In Progress
priority: High
references:
- TASK-12030
- TASK-12127
- https://github.com/rmusser01/tldw_server/issues/2590
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provision same-origin single-user WebUI auth from runtime configuration, add explicit opt-in persistent storage for manually configured remote servers, and cover hard reload plus browser/extension relaunch persistence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Same-origin single-user WebUI deployments with runtime auth do not ask users to enter an API key.
- [ ] #2 Manually configured remote single-user servers expose an explicit Remember on this device choice.
- [ ] #3 Session-only choice does not persist the API key across a full browser restart.
- [ ] #4 Remembered choice persists the API key across a full browser restart until logout/reset.
- [ ] #5 WebUI regression coverage includes save then hard reload and save then close/reopen same profile.
- [ ] #6 Extension regression coverage includes save then close/reopen same extension installation.
- [ ] #7 No browser password-manager behavior is required for API key persistence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved design: runtime-provisioned same-origin auth remains automatic; manual single-user setup defaults Remember on this device to enabled; unchecked credentials are session-only; keys are origin-bound; runtime keys are never persisted; extension device storage must remain local rather than synced.
Design specification: docs/superpowers/specs/2026-07-10-single-user-api-key-device-persistence-design.md
Spec review iteration 1 found origin-transition and legacy session-bridge ownership ambiguities. Revised the design to require explicit no-inherited-auth candidate probes, an ordered post-validation origin transition, a strict legacy manual-session classifier, and device → session → memory fallback semantics.
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
