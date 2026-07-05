---
id: TASK-12171
title: 'Fix dark-theme visual drift in Chat, extension chat, character chat, and Notes'
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-05 18:31'
labels:
  - frontend
  - theme
  - webui
  - extension
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Walk through the WebUI chat page, browser extension chat, character-card chat, and Notes page. Reproduce dark-theme light-surface leaks, apply the smallest token/component fixes, and verify the affected menus/options in rendered UI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Normal chat dark-mode walkthrough has no large light-surface leaks.
- [x] #2 Character chat dark-mode walkthrough has no large light-surface leaks.
- [x] #3 Extension sidepanel chat dark-mode walkthrough has no large light-surface leaks.
- [x] #4 Notes dark-mode walkthrough has no large light-surface leaks, including Ant Design select wrappers and overflow menu.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-05-chat-notes-dark-theme-visual-fidelity-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a focused Playwright smoke regression that seeds dark mode, mocks target API shapes, captures screenshots, and scans normal chat, character chat, extension sidepanel chat, Notes, and available overflow/tool menus for large light surfaces and low-contrast dark text. Fixed the confirmed Notes leaks by adding .ant-select to the shared Ant Design token bridge and covering Ant text buttons/radio-button wrappers in apps/packages/ui/src/assets/tailwind-shared.css.

PR review follow-up: excluded Ant link buttons from the generic text-color override and collapsed the visual scanner into one DOM traversal per checkpoint with opacity and viewport-bound checks.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the dark-theme visual drift found during the walkthrough by bridging Ant Design Select wrapper surfaces, text-style buttons, disabled buttons, and radio-button wrappers to shared theme tokens. Verified with a red/green Playwright dark-theme visual-fidelity smoke pass and git diff --check on the touched files. Bandit skipped because the touched code is frontend CSS and Playwright TypeScript only. Remaining environment note: backend was unavailable, so the rendered walkthrough used route-level API mocks and the extension chat was verified through the sidepanel debug route.

PR review follow-up excluded .ant-btn-link from the generic button color bridge and made the smoke scanner ignore transparent/off-viewport nodes while doing one DOM pass per checkpoint.
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
