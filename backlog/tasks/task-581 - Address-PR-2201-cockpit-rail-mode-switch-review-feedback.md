---
id: TASK-581
title: Address PR 2201 cockpit rail mode-switch review feedback
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-01 00:24
labels:
- frontend
- chat
- ux
- review-feedback
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/pull/2201
- TASK-578
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix PR #2201 review feedback: preserve intentionally collapsed cockpit rail visibility across focus-to-cockpit mode switches now that both-collapsed cockpit state is supported by edge-mounted restore tabs. Remove or relax the legacy auto-restore branch in handleChatLayoutModeChange and add regression coverage for both rails collapsed across focus/cockpit toggles.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Switching from focus mode back to cockpit preserves intentionally collapsed Context and Runtime rails when both are hidden.
- [x] #2 Legacy auto-restore no longer reopens both rails solely because cockpit mode is selected.
- [x] #3 Regression coverage proves both edge restore tabs remain available after focus-to-cockpit toggling.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Verified the issue against `handleChatLayoutModeChange`: storage already normalizes unset rail visibility to visible, so the legacy both-hidden auto-restore was only overriding explicit user intent.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the legacy cockpit-mode auto-restore from `handleChatLayoutModeChange` so explicit `false` rail visibility persists across focus/cockpit mode toggles. Added a red/green regression in `Playground.cockpit-shell.test.tsx` covering both rails collapsed, focus mode, return to cockpit, and both edge restore tabs still visible. Verification: focused test file failed before the handler change, then passed after; combined cockpit/rail/offline suite passed with 54 tests; shared UI TypeScript passed with the heap override; `git diff --check` passed; targeted ESLint `--quiet` exited 0 with the existing Next pages-directory warning. Bandit skipped because touched code is TypeScript/TSX plus Backlog metadata only.
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
