---
id: TASK-13228
title: Keep the default Buddy placement clear of primary composer controls
status: Done
created_date: 2026-09-09 05:16
priority: high
references:
- TASK-13227
documentation:
- Docs/Reviews/2026-09-09-buddy-v1-qualification.md
assignee:
- '@codex'
updated_date: 2026-09-09 06:40
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-13227 live WebUI qualification at1090×990 found the initial Buddy position covering Console Send: the drag handle intercepted the pointer until Shift+ArrowUp moved the Buddy. Define a safe default or docking behavior while retaining intentional user positioning and keyboard access.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh/default and reset Buddy placement leave Console send, stop, and composer input usable at supported desktop/compact sizes.
- [x] #2 User-selected placement remains editable by pointer and keyboard and stays in viewport across resizing/navigation.
- [x] #3 Real pointer hit-testing verifies the primary controls alongside the visible Buddy; the chosen placement policy is documented before implementation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md. Reason: routine default placement repair within existing saved positions, keyboard and reset contract. Policy: use the upper-right area below app navigation for fresh/reset placement instead of the lower composer region; preserve explicitly saved positions and existing viewport clamping. Add focused default/reset, resize and manual movement regressions; verify actual composer input/Send/Stop pointer hit targets at the supported desktop and compact viewports through the allowed in-app browser. Avoid new persisted positioning state or automatic movement of user-selected positions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented y=96 for both shared default position buckets. Existing saved positions, drag/keyboard movement, and reset behavior remain intact. The store/layout subset passed 44 tests; the final UI gate passed 94. At a live 1090×991 viewport, Home reset placed the Buddy at (942,96), and actual input/Send center hit-tests were unobstructed. The planned live compact pointer check was replaced by automated clamping/layout checks because no supported compact browser control was available; no compact native pass is claimed. Changed-line formatting/lint comparison found no new issues. Independent review complete. No Python changed for this task, so Bandit is inapplicable. Existing ADR-005 applies. User guide and Docs/Reviews/2026-09-09-buddy-v1-qualification.md record policy, evidence and limits.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fresh installs and Reset position place the Buddy at y=96, preserving saved user placements and viewport movement. Verified with 94 focused UI tests and a live desktop composer hit-test. Compact geometry is automated; a live compact pointer walkthrough remains part of broader qualification. Bandit is inapplicable to these TypeScript-only edits. Existing ADR-005 and user/evidence docs updated; PR #2934.
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
