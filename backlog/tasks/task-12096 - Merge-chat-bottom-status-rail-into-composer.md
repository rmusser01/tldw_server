---
id: TASK-12096
title: Merge chat bottom status rail into composer
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-02 06:23'
labels:
  - webui
  - chat
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the separate bottom status rail on the chat cockpit surface, merge its status/model/message/context indicators into the composer control row, and dock the composer to the bottom of the viewport.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No standalone bottom status rail is rendered below the chat surface.
- [x] #2 Composer is anchored to the bottom edge of the chat viewport.
- [x] #3 Readiness, saved/persistence state, model/provider label, message count, and context action remain visible from the composer area.
- [x] #4 Mobile layout keeps the composer usable without cramped controls or horizontal overflow.
- [x] #5 Focused tests and browser QA cover the rail removal and composer merge.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation notes:
- Removed the standalone chat cockpit status strip from PlaygroundCockpitShell/Playground.
- Merged model health, character/context, message count, token usage, saved state, and advanced controls into the composer context row.
- Kept desktop composer status visible even when advanced options are collapsed; kept mobile compact to avoid cramped always-on status chrome.
- Updated the hidden-header /chat WebLayout wrapper to remove px/py padding for viewport-constrained chat routes so the composer reaches the viewport bottom.

Verification:
- bun run test __tests__/components/layout/WebLayout.chat-scroll-contract.test.tsx
- bun run test src/components/Option/Playground/__tests__/Playground.responsive-parity.guard.test.ts src/components/Option/Playground/__tests__/ComposerToolbar.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx
- bun run test:playground:device-matrix
- Browser QA at 1440x960 and 390x844: composer region bottom gap 0, status strip absent, merged context row visible.
- git diff --check
- Bandit on touched frontend scope wrote /tmp/bandit_task12096.json with zero findings (0 Python LOC in touched TS scope).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Merged the chat status rail into the composer, removed the standalone bottom rail, and adjusted the hidden-header chat shell so the composer docks to the viewport bottom on desktop and mobile. Focused tests, device-matrix tests, browser measurements, diff check, and Bandit completed.
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
