---
id: TASK-45.44.3.9
title: Migrate ReportBuilderDrawer alert to design system
status: Done
labels:
- design-system
- webui
- extension
- product-state
- watchlists
priority: medium
parent_task_id: TASK-45.44.3
references:
- https://github.com/rmusser01/tldw_server/issues/1660
documentation:
- Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
- apps/packages/ui/scripts/design-system-product-state-baseline.json
modified_files:
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/ReportBuilderDrawer.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/ReportBuilderDrawer.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the remaining AntD Alert product-state exception in Watchlists ReportBuilderDrawer to the shared design-system Alert primitive and record before/after baseline evidence for TASK-45.44.3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ReportBuilderDrawer run-required and preflight readiness warnings render through the shared design-system Alert primitive.
- [x] #2 The ReportBuilderDrawer AntD Alert baseline exception is removed from design-system-product-state-baseline.json.
- [x] #3 Focused ReportBuilderDrawer coverage and design-system product-state guard pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
RED: `bunx vitest run src/components/Option/Watchlists/OutputsTab/__tests__/ReportBuilderDrawer.test.tsx` failed because the run-required and blocking-warning notices did not have a `data-ds-component="Alert"` ancestor.

GREEN: migrated ReportBuilderDrawer warnings from AntD Alert props to the shared design-system Alert `variant` API and removed the stale baseline entry.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the ReportBuilderDrawer readiness notices from AntD Alert to the shared design-system Alert primitive, added regression assertions for the design-system Alert wrapper, and removed the stale product-state baseline exception. Baseline evidence: total product-state exceptions 256 -> 255; Jobs/Scheduler/Watchlists exceptions 21 -> 20; ReportBuilderDrawer target rows 1 -> 0. Verification: `bunx vitest run src/components/Option/Watchlists/OutputsTab/__tests__/ReportBuilderDrawer.test.tsx` passed (6 tests); `bun run verify:design-system-state` passed with 255 baseline exceptions. Bandit not applicable because the touched implementation scope is TypeScript/TSX plus JSON baseline metadata only.
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
