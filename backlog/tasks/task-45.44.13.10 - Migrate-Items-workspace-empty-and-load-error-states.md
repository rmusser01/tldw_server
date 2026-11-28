---
id: TASK-45.44.13.10
title: Migrate Items workspace empty and load-error states
status: Done
labels:
- design-system
- webui
- extension
- product-state
references:
- TASK-45.44.13
- https://github.com/rmusser01/tldw_server/issues/1670
- Docs/Design/tldw_web_design_system_visual_qa_checklist.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue GitHub #1670 by migrating the Items workspace empty, filtered no-results, and list failure states to shared product-state components. Preserve the existing filter controls, request flow, and Retry action.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Use shared EmptyState and StatePanel components for the Items empty and failed-load states.
- [x] #2 Cover pending load, empty list, filtered results, and failed-load recovery with focused tests.
- [x] #3 Remove the one Items workspace legacy allowance and record desktop and phone WebUI checks.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
No ADR is required. The changes use existing shared product-state conventions and keep the Items request and recovery flow. The open workspace recovery draft #3020 removes the separate Shared With Me component.
The route is WebUI-only in route-metadata.ts. Desktop and phone WebUI captures cover the consuming route.
Verified the base against remote dev at eb48892630330498346153332d6d3951fe942f41. Items allowances fell from one to zero, and the standalone #1670 path count fell from fourteen to thirteen. The repository-wide verifier exits 1 on 97 blocked findings and 28 stale entries present on the base.
All 39 focused Vitest tests passed across five files. Type checking and scoped ESLint passed. Bandit found no Python files in the Items scope. Independent review verified that late search responses and failures are ignored, and an undo action refreshes the current filters.
Before and after Chromium captures at 1440×900 and 390×844 show the empty results and failed-load panel on the WebUI Items route. The tests verify filter reset, a visible keyboard focus indicator on Retry, and recovery with Enter. Both viewports fit the error detail without horizontal scrolling.
Read-only Git merge previews flagged adjacent baseline hunks with #3671 and #3673. A merge of the baseline by entry ID preserves the independent removals from all four slices. The merge preview with #3672 passed cleanly.
Pull request https://github.com/rmusser01/tldw_server/pull/3674
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the Items workspace to shared EmptyState and StatePanel components. Empty results announce when the active request settles. Load failure keeps its detail and Retry, and delayed refreshes use the current filters.
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
