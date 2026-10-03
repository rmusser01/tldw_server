---
id: TASK-13423
title: Keep Persona Buddy visible when the web pane narrows
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The floating Persona Buddy disappears in the desktop browser when its pane becomes narrower than 1024 pixels, even though the live voice session still works. Keep the active Buddy available in compact panes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An enabled active Persona Buddy renders in a web viewport below 1024 pixels and remains within the viewport.
- [x] #2 Resizing across 1024 pixels preserves the mounted artwork, unsent control draft and live session without another automatic visual-pack or session load.
- [x] #3 Existing disabled, inactive-surface and independent-Buddy ownership gates continue to suppress the legacy shell.
- [x] #4 Targeted regressions, scoped static checks and an actual compact-browser verification pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
## Implementation Plan

ADR required: no
ADR path: backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md (existing guidance)
Reason: This bounded responsive correction keeps the existing render ownership, session, persistence and loading boundaries. It removes the legacy Phase 2 desktop-width limit for an active browser pane and uses existing viewport clamping.

1. Add failing regressions for rendering below 1024 pixels and retaining the same shell, artwork and unsent draft across the breakpoint.
2. Remove the web width suppression while retaining enable, surface and independent-ownership gates.
3. Run targeted tests and scoped static checks; verify the actual 858-pixel browser pane and publish through PR 3091.
Implemented the compact-pane correction by removing the legacy web <1024 suppression from BuddyShellHost. Enablement, active-surface and independent-Buddy ownership gates are unchanged; the existing viewport clamp keeps artwork and controls accessible. The legacy Track B design now records that TASK-13423 supersedes its desktop-width restriction. Existing ADR005 applies; no new ownership, persistence, session or loader boundary.

Validation: both narrow-render and resize-retention regressions failed before the fix. The final seven-file targeted run passed 128 tests, including three narrow-pane ownership gates. Scoped ESLint has no errors and the existing line-361 dependency warning; scoped TypeScript has the same four dependency declaration errors as baseline and no owned diagnostics. The added test region passes formatting using the existing shared-file style; whole-file Prettier warnings also occur on HEAD. Bandit is not applicable to this TypeScript/documentation-only change. Self review and independent review found no actionable code defects.

Actual browser verification at 1024, 1023, 390 and 320 pixels preserved the unsent draft and the exact connected session. Loaded sprite frames remained visible; the natural 858×981 pane was restored. API access counts added zero visual-pack list/detail or session reads during resizing. The disposable test draft was cleared without sending, controls closed, and position reset through the UI. Browser console warnings/errors were empty. Runtime source is byte-identical to the repaired source. Private receipts: /private/tmp/buddy-replay-fix-uat-20261003/responsive-browser-receipts.json, responsive-request-comparison.json, responsive-source-verification.json and buddy-visible-natural-pane.png. Tests/static receipts: /private/tmp/buddy-remaining-fixes-20261003/buddy-responsive-final-128.log, buddy-responsive-red.log, responsive-type-baseline-comparison.json and responsive-format-verification.json.

Tooling note: the hosted Backlog MCP was unresponsive. Its local canonical Python implementation supports safe checklist/note mutation but does not expose a separate plan field; the Implementation Plan heading was recorded through that API before code changes, preserving the supported task markers.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Buddy remains rendered in compact web panes and retains its artwork, unsent draft and connected session across resizing. 128 targeted tests passed; actual 858-pixel and 320/390/1023/1024 browser checks passed with no extra pack/session reads. Existing static-tool baseline findings are documented.
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
