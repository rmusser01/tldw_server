---
id: TASK-349.3.1
title: Stage 6A Watchlist constrained navigation shell
status: Done
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
- [x] #1 Shared Watchlists constrained-viewport helper classifies 390x844 and 420x760 as constrained, 768px+ as desktop, and handles SSR/no-window safely.
- [x] #2 `/watchlists` constrained shell exposes Watchlist switching plus Overview, Feeds, Monitors, Alerts, Updates, Activity, Reports, Templates, and Settings without a desktop tab bar dependency.
- [x] #3 Existing desktop tab behavior, deep-link tab mapping, orientation guidance, and selected Watchlist scoping keep working.
- [x] #4 Focused Vitest coverage proves constrained navigation, desktop regression, and secondary-tab mapping behavior.
- [x] #5 No backend behavior changes are introduced; Bandit is recorded as not applicable unless Python files are unexpectedly touched.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Stage 6A implementation added a shared Watchlists constrained viewport helper, replaced the constrained mobile tab select with grouped drawer navigation, preserved desktop tabs and direct secondary-destination navigation in constrained mode, and added focused Vitest coverage. Verification: bunx vitest run src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.extension-navigation.test.tsx src/components/Option/Watchlists/shared/__tests__/useWatchlistsViewport.test.ts src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.first-class.test.tsx src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.orientation-guidance.test.ts passed with 4 files / 12 tests. bun run test:watchlists:typecheck passed with 1 file / 3 tests. JSON locale parse passed for src/assets/locale/en/watchlists.json and src/public/_locales/en/watchlists.json. git diff --check passed. Bandit not applicable because no Python/backend files were touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 6A shipped the constrained Watchlists navigation shell. It adds a shared viewport helper, a grouped drawer navigation for extension-sized viewports, direct constrained navigation to secondary management destinations, locale-backed mobile navigation copy, and focused regression coverage. Verification completed: Stage 6A Vitest suite passed, Watchlists static guard passed, locale JSON parse passed, git diff --check passed, and Bandit was not applicable because only frontend/Markdown task files changed.
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
