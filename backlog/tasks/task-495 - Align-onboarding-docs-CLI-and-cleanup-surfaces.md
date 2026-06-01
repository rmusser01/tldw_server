---
id: TASK-495
title: Align onboarding docs CLI and cleanup surfaces
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-31 17:15'
labels: []
dependencies: []
references:
  - TASK-489
  - >-
    Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md
documentation:
  - >-
    Docs/superpowers/plans/2026-05-31-first-time-solo-user-onboarding-implementation-plan.md#task-9-docs-cli-makefile-and-onboarding-cleanup
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task 9 slice from the unified onboarding plan. Align Getting Started docs, profile docs, Makefile/CLI messaging, onboarding manifest, and published parity around peer solo setup paths, WebUI first chat, and post-onboarding first source.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Docs present Docker and local single-user as peer paths and multi-user as an operator exit
- [x] #2 CLI/Makefile output points users to WebUI first-time setup and does not claim first-run completion without backend first-chat state
- [x] #3 Conflicting setup copy is redirected, demoted, or aligned with the unified lifecycle
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed as fulfilled by replacement docs/startup task TASK-498. TASK-498 aligned source and published Getting Started docs, onboarding manifest parity, Makefile/start messaging, and CLI profile verification around WebUI first-chat completion and multi-user operator exit.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Superseded and fulfilled by TASK-498. Docs, Makefile, CLI verification, and cleanup requirements are implemented and verified in the completed replacement task.
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
