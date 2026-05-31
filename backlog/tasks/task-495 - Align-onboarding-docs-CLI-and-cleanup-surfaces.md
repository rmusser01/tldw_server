---
id: TASK-495
title: Align onboarding docs CLI and cleanup surfaces
status: To Do
references:
- TASK-489
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md
documentation:
- Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md#task-9-docs-cli-makefile-and-onboarding-cleanup
modified_files:
- Makefile
- tldw_Server_API/cli/wizard/cli.py
- tldw_Server_API/cli/wizard/profile_verify.py
- tldw_Server_API/cli/wizard/profiles.py
- Docs/Getting_Started/README.md
- Docs/Getting_Started/Profile_Docker_Single_User.md
- Docs/Getting_Started/Profile_Local_Single_User.md
- Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md
- Docs/Getting_Started/onboarding_manifest.yaml
- Docs/Published/Getting_Started/README.md
- Docs/Published/Getting_Started/onboarding_manifest.yaml
- tldw_Server_API/tests/Docs/
- tldw_Server_API/tests/Utils/
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task 9 slice from the unified onboarding plan. Align Getting Started docs, profile docs, Makefile/CLI messaging, onboarding manifest, and published parity around peer solo setup paths, WebUI first chat, and post-onboarding first source.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Docs present Docker and local single-user as peer paths and multi-user as an operator exit
- [ ] #2 CLI/Makefile output points users to WebUI first-time setup and does not claim first-run completion without backend first-chat state
- [ ] #3 Conflicting setup copy is redirected, demoted, or aligned with the unified lifecycle
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
