---
id: TASK-13175
title: Keep expanded Buddy controls inside the browser viewport
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 14:55'
updated_date: '2026-09-05 15:58'
labels: []
dependencies: []
references:
  - Docs/Reviews/MIGU_BUDDY_UAT_2026_09_05.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live Migu UAT at 1280x720: opening Buddy at the default lower-right position placed its popover at x1104 y818 size220x478, entirely below the viewport. Choose/Change Buddy could not be clicked. Dragging to x704 y17 made it reachable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Opening or changing Buddy content keeps the composer and navigation controls reachable at supported desktop widths.
- [x] #2 Expanding and collapsing preserves a usable position across reload and viewport changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the existing dock overflow with focused component geometry regressions.
2. Re-clamp on open/content size changes and bound scrolling on short viewports while preserving stored drag positions.
3. Run focused Buddy shell Vitest regressions and scoped lint; record evidence.
ADR required: no
ADR path: N/A
Reason: Routine bug fix preserving the existing floating-shell geometry contract in Docs/superpowers/specs/2026-03-31-persona-buddy-track-b-floating-shell-design.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The floating shell now clamps before paint on expansion/context changes and observes dock size changes, so asynchronous diagnostics and visual/live content cannot leave the shell at its compact lower-right coordinates. Position updates remain equality-guarded; dragging and persisted surface buckets retain their existing behavior. The dock has a viewport-height bound; its popover scrolls within the remaining flex space, with outer overflow as a fallback.
Modified BuddyShellHost.tsx, BuddyShellDock.tsx, BuddyShellPopover.tsx, and BuddyShellHost.test.tsx.
TDD: two new regressions failed on the original code at (1104,609) instead of (1044,17), and on unobserved content growth. Focused host/popover/position-store Vitest run: 46 passed across 3 files. Regression coverage includes opening, content growth, unchanged-size notifications without repeated position writes, collapse/remount position, and viewport resizing. Scoped ESLint passes using the WebUI config from apps/ (Next page-directory lookup disabled because these are shared components); git diff --check passes. Existing localStorage/i18next test-environment warnings remain.
ADR required: no; routine correction of the existing Track B floating-shell geometry contract. Bandit skipped: touched code is TypeScript/TSX, not Python. Live browser verification of short-height scrolling is pending in the coordinating UAT; no shared browser/runtime or UAT report was modified.

Real Chromium UAT at 1280x720: expanded Migu dock x938.55 y229 w325.45 h416; bottom645 within720. At1280x360 dock y16 h328 bottom344; popover client219/content331 with overflow auto; bottom Choose/Change Buddy link reachable after scrolling. Screenshots and DOM geometry captured for the coordinated UAT report.

Coordinated final validation: 265 focused frontend tests, 54 backend tests, production Bandit0 findings, scoped frontend ESLint0 errors (warnings documented), unchanged Python lint baseline, real browser evidence and limitations recorded in Docs/Reviews/MIGU_BUDDY_UAT_2026_09_05.md. Repository-wide typechecking remains limited by80 diagnostics across6 unchanged unrelated files; no full suite run.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Expanded and content-resized Buddy controls remain within desktop viewports and scroll at short heights. Real pointer drag moved the repaired dock by200px left/40px down; final stop completed.46 geometry/popover/store tests pass within the265-test final frontend run.
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
