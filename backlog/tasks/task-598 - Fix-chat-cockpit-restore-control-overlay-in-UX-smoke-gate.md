---
id: TASK-598
title: Fix chat cockpit restore control overlay in UX smoke gate
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 20:35'
labels:
  - bug
  - frontend
  - ci
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the PR UX Smoke Gate failure where the chat sidebar edge expand button intercepts clicks on the playground cockpit left-rail restore button in the real-server cockpit test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Real-server chat cockpit restore interaction is clickable when both rails are collapsed.
- [x] #2 Focused chat cockpit UX smoke regression passes locally.
- [x] #3 Changes remain minimal and do not affect unrelated routes or backend code.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented a focused chat cockpit overlay fix. Added a Playwright regression proving the cockpit context restore tab remains clickable while the app chat rail edge is collapsed. Kept the app chat edge at its original z-index so ordinary page content does not intercept it, shifted that edge handle above the cockpit restore midpoint in both WebLayout copies, and raised cockpit restore tabs above peer edge controls. Verification: red focused regression reproduced the CI pointer-interception failure; green focused regression passed; component/guard tests passed (2 files, 6 tests); git diff --check passed. Full chat-rails-collapse local run still has an isolated pre-existing medium/mobile artifact close interception failure unrelated to this change because chat-sidebar-edge-expand is not rendered at that viewport.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Backlog task reflects touched files and verification.
- [x] #2 Regression coverage added or adjusted before implementation.
- [x] #3 Focused test passes after the fix.
- [x] #4 No Python Bandit run required for TS/TSX-only changes, or Bandit is run if Python is touched.
<!-- DOD:END -->
