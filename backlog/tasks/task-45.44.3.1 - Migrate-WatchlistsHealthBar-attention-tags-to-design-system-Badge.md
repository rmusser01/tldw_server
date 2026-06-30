---
id: TASK-45.44.3.1
title: Migrate WatchlistsHealthBar attention tags to design-system Badge
status: Done
assignee:
- Codex
labels:
- design-system
- webui
- product-state
- watchlists
priority: medium
parent_task_id: TASK-45.44.3
references:
- apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsHealthBar.tsx
- apps/packages/ui/src/components/Option/Watchlists/shared/__tests__/WatchlistsHealthBar.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace WatchlistsHealthBar's AntD Tag attention chips with shared design-system Badge primitives while preserving clickable navigation for sources, runs, outputs, and jobs attention items.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WatchlistsHealthBar no longer imports or renders AntD Tag for attention product-state UI.
- [x] #2 Sources, runs, outputs, and jobs attention items render through shared design-system Badge variants that preserve warning/error severity.
- [x] #3 Focused tests prove attention badges render design-system markers and still navigate to the matching watchlists tab.
- [x] #4 The design-system product-state baseline no longer contains WatchlistsHealthBar AntD Tag exceptions and verifier results are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused WatchlistsHealthBar coverage that renders attention sources/runs/outputs/jobs and asserts the current implementation lacks design-system Badge markers for those attention items.
2. Replace AntD Tag attention chips with a small local button wrapper around the shared Badge primitive, mapping warning items to variant=warning and failed runs to variant=danger.
3. Remove WatchlistsHealthBar entries from the product-state baseline.
4. Verify with focused Vitest, product-state guard tests, bun run verify:design-system-state, git diff --check, and document Bandit as not applicable for this UI-only slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented WatchlistsHealthBar attention chips with a local button wrapper around the shared design-system Badge primitive. Sources, outputs, and jobs use warning badges; failed runs use the danger badge. The focused regression first failed because the existing AntD Tag text had no design-system Badge marker, then passed after migration while preserving tab navigation.

Removed the four WatchlistsHealthBar AntD Tag entries from the product-state baseline. Verification: focused WatchlistsHealthBar test passed; LlamacppRuntimePanel/WatchlistsHealthBar/product-state guard suite passed 56/56; bun run verify:design-system-state passed with 406 remaining allowed legacy exceptions; git diff --check passed; section marker sanity check passed. Full UI TypeScript still exits 2 on existing repo-wide type debt outside touched files, with no touched-file errors observed in the output. Bandit is not applicable because this slice touches frontend TypeScript, JSON baseline data, and Backlog metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated WatchlistsHealthBar attention chips from AntD Tag to shared design-system Badge primitives, kept each chip as a keyboard-addressable button that navigates to the matching watchlists tab, added focused coverage for Badge markers and navigation, and removed the four WatchlistsHealthBar baseline exceptions.
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
