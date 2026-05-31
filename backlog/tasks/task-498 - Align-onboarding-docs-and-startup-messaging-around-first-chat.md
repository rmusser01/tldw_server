---
id: TASK-498
title: Align onboarding docs and startup messaging around first chat
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-31 15:48'
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
Align the Getting Started docs, published onboarding manifest parity, CLI verification copy, and Make quickstart messaging around the WebUI-first solo onboarding journey whose completion gate is a successful first chat.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Published Getting Started docs and onboarding manifest remain in parity with source docs.
- [x] #2 Getting Started docs present Docker and local single-user as peer solo setup choices with WebUI first-chat completion.
- [x] #3 Multi-user setup routes operators to the multi-user guide/checklist instead of the solo wizard.
- [x] #4 Make quickstart/start/verify messaging points users to the WebUI as the next setup action.
- [x] #5 CLI profile verification reports backend first-run chat state without marking setup complete unless the backend does.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Aligned source and published Getting Started docs around one obvious start command, peer local/Docker solo choices, first chat as the setup completion gate, and adding the first source as the immediate next milestone.
Updated Makefile status copy and profile verification to surface backend first-run chat state.
Verification: focused docs/Makefile/CLI tests passed; full planned docs/Makefile/CLI test set passed; adjacent docs contract tests passed; Bandit reported zero findings for profile_verify.py.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Docs, Makefile output, published Getting Started parity, and profile verification now point first-time solo users from start command to WebUI setup to first successful chat, then first source. Multi-user remains an operator path routed to the guide/checklist.
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
