---
id: TASK-349.3.1
title: Stage 6A Watchlist constrained navigation shell
status: To Do
labels:
- watchlists
- stage6
- frontend
- ux
priority: medium
parent_task_id: TASK-349.3
documentation:
- Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage6-extension-management-plan.md
- Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
- Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage5-defensible-reports-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the shared constrained-viewport foundation and /watchlists shell navigation so extension-sized users can switch Watchlists and reach Overview, Feeds, Monitors, Alerts, Updates, Activity, Reports, Templates, and Settings without relying on desktop tabs or clipped controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Shared Watchlists constrained-viewport helper classifies 390x844 and 420x760 as constrained, 768px+ as desktop, and handles SSR/no-window safely.
- [ ] #2 `/watchlists` constrained shell exposes Watchlist switching plus Overview, Feeds, Monitors, Alerts, Updates, Activity, Reports, Templates, and Settings without a desktop tab bar dependency.
- [ ] #3 Existing desktop tab behavior, deep-link tab mapping, orientation guidance, and selected Watchlist scoping keep working.
- [ ] #4 Focused Vitest coverage proves constrained navigation, desktop regression, and secondary-tab mapping behavior.
- [ ] #5 No backend behavior changes are introduced; Bandit is recorded as not applicable unless Python files are unexpectedly touched.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Follow `Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage6-extension-management-plan.md` Task 1. Use TDD: write the viewport and constrained navigation tests before implementing the shell changes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
