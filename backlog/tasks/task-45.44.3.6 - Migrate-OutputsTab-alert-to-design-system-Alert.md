---
id: TASK-45.44.3.6
title: Migrate OutputsTab alert to design-system Alert
status: Done
assignee: []
created_date: '2026-05-23 22:17'
updated_date: '2026-05-23 22:17'
labels:
  - design-system
  - webui
  - extension
  - product-state
  - watchlists
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1660'
  - apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputsTab.tsx
  - apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
  - 'https://github.com/rmusser01/tldw_server/pull/2012'
parent_task_id: TASK-45.44.3
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the Watchlists OutputsTab AntD Alert product-state callout with the shared design-system Alert primitive, preserving user-facing copy and removing the migrated baseline exception.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 OutputsTab delivery-issues banner renders through the shared design-system Alert primitive instead of AntD Alert.
- [x] #2 Focused OutputsTab coverage proves the migrated banner preserves copy/actions and exposes the canonical Alert marker.
- [x] #3 The OutputsTab Alert baseline exception is removed without introducing new product-state verifier findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused failing OutputsTab assertion requiring the delivery-issues banner to render with the design-system Alert marker.
2. Replace the OutputsTab AntD Alert usage with the shared Alert primitive while preserving copy, warning semantics, and remediation actions.
3. Remove the migrated OutputsTab Alert entry from design-system-product-state-baseline.json and run focused tests plus design-system verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TDD red/green completed. Added a focused OutputsTab assertion requiring the delivery-issues banner to carry data-ds-component="Alert"; the red run failed because the AntD Alert mock did not expose the marker. Replaced the banner with the shared design-system Alert primitive, preserved the warning title, description, Show failed only and Open failed runs actions, and removed the single OutputsTab baseline exception. Verification: focused OutputsTab advanced-filters test passed 6/6; product-state guard passed 54/54; bun run verify:design-system-state passed with 258 total baseline exceptions and 23 Jobs/Scheduler/Watchlists exceptions; baseline parse reported OutputsTab target rows 1 -> 0; git diff --check passed. Bandit skipped because this slice touched frontend TSX/test/JSON/Backlog markdown only.

TypeScript: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false still exits 2 on 347 existing diagnostics; no diagnostics mention OutputsTab, OutputsTab.advanced-filters, the baseline, or TASK-45.44.3.6.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the OutputsTab delivery-issues banner from AntD Alert to the design-system Alert primitive. Focused coverage now verifies the banner uses data-ds-component="Alert" while preserving the remediation actions, and the product-state baseline no longer contains the OutputsTab Alert exception.
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
