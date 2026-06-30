---
id: TASK-12079
title: Apply chat cockpit rail tab alignment to extension
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-01 00:02'
labels:
  - webui
  - extension
  - chat
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend the WebUI collapsed chat rail alignment fix to the browser extension surface so the Chats tab sits higher and the context rail restore tab is edge-attached instead of floating.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Extension chat cockpit uses the same upper-edge collapsed Chats tab placement as the WebUI.
- [x] #2 Extension context rail restore tab is attached to the side edge and remains clear of the composer.
- [x] #3 Regression tests or source guards cover extension/shared extension behavior where practical.
- [x] #4 Rendered validation or the closest available extension workflow verifies the corrected layout.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR: https://github.com/rmusser01/tldw_server/pull/2562

Implemented shared rail positioning constants in packages/ui and consumed them from WebLayout, Layout, and PlaygroundCockpitShell so WebUI and extension option surfaces use the same collapsed Chats and context rail placement. Verification: focused Vitest WebUI/shared tests passed; extension Vitest config guard passed; WebUI Playwright chat rail collapse spec passed; extension build smoke passed and built bundle contains the new edge-positioning classes/test IDs. Extension Playwright smoke skipped at runtime because no service worker target appeared in this environment. Bandit ran on touched files with zero findings; TS/TSX files reported AST parse errors because Bandit is Python-oriented.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Applied the chat cockpit rail alignment fix across the WebUI and extension-facing shared UI. Added a shared positioning contract, wired both collapsed rail triggers to it, added guard coverage, and verified through focused tests, WebUI Playwright, extension build smoke, built bundle string checks, formatting, ESLint, and Bandit documentation.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Bandit run for touched code when applicable or documented skip
- [x] #4 Final summary added
- [x] #5 Known skips or blockers documented
<!-- DOD:END -->
