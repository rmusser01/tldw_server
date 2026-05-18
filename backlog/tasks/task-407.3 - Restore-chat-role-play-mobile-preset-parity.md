---
id: TASK-407.3
title: Restore chat role-play mobile preset parity
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-17 16:16'
labels:
  - chat
  - ux
  - roleplay
  - stage-3
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-17-main-chat-role-play-preset-remediation-implementation-plan.md
parent_task_id: TASK-407
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stage 3 implementation for the main /chat role-play preset plan: expose behavior templates, generation style, and active role-play recovery in the mobile composer/overflow path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Mobile/casual /chat users can reach behavior templates and generation style controls.
- [x] #2 Active role-play chips expose clear/change actions without desktop-only dependencies.
- [x] #3 Composer remains usable at narrow viewport widths.
- [x] #4 Focused Stage 3 tests and browser verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stage 3 started after Stage 2 commit 0e2203094. Scope remains limited to mobile/casual access for main /chat role-play behavior templates, generation style, and active role-play recovery controls.

Stage 3 implementation completed in the dedicated chat role-play remediation worktree. Mobile overflow now exposes System prompts and Generation style actions, routes them through reusable role-play action callbacks, and carries an optional Role-play setup callback for Stage 4. Active role-play chips remain visible and actionable on mobile without relying on desktop-only advanced controls. Verification: focused Stage 3 Vitest suite passed (4 files, 31 tests); locale JSON parse check passed; git diff --check passed. Full tsc still fails on existing unrelated baseline files. Browser verification for 127.0.0.1:3001/chat remains blocked by the in-app browser security policy; CDP was not used because the policy explicitly prohibited routing around the blocked target. Bandit is not applicable because this stage only touches frontend TypeScript/React/locale files.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 3 restores mobile parity for the main /chat role-play preset workflow. Mobile users can open System prompts and Generation style from the overflow menu, active role-play chips remain actionable in the mobile context strip, and the overflow now has a reusable Role-play setup callback hook for Stage 4. Focused tests pass; browser verification is recorded as blocked by the browser target policy.
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
