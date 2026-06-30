---
id: TASK-418.3
title: Plan WebUI chat composer and global chrome implementation
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-17 18:54'
labels:
  - ux
  - design
  - webui
  - extension
  - planning
  - chat
  - navigation
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
  - >-
    Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md
  - >-
    Docs/superpowers/plans/2026-05-17-webui-chat-global-chrome-implementation-plan.md
parent_task_id: TASK-418
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Documentation-only child implementation plan for the approved WebUI/extension UX remediation program Task 6. Scope maps findings F6, F8 support, F13, F2 support, and F15 support into a reviewable chat composer-first, command target, and route-specific global chrome implementation plan without product code changes in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Child implementation plan saved at Docs/superpowers/plans/2026-05-17-webui-chat-global-chrome-implementation-plan.md
- [x] #2 Plan covers findings F6, F8 support, F13, F2 support, and F15 support
- [x] #3 Plan maps /chat, /quick-chat-popout, /knowledge, /media, /sources, /settings, /mcp-hub, /stt, and /tts route responsibilities
- [x] #4 Plan identifies exact files, tests, verification commands, acceptance criteria, rollback, and out-of-scope boundaries
- [x] #5 No product frontend or backend code changed in this planning task
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created the WP6 documentation-only implementation plan for chat, composer, command palette, and global chrome remediation.

Verification recorded for the planning artifact:
- Placeholder scan passed for blocked marker and weak-language tokens.
- ASCII and trailing-whitespace scan passed.
- git diff --check passed for the plan file.
- Required-scope coverage check passed for findings, routes, files, and test targets.

Bandit skip: documentation-only change; no Python or executable product code touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and verified the documentation-only child plan for WebUI chat composer and global chrome remediation. Product code was not modified.
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
