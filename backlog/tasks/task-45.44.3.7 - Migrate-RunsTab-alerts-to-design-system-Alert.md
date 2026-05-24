---
id: TASK-45.44.3.7
title: Migrate RunsTab alerts to design-system Alert
status: Done
assignee: []
created_date: '2026-05-23 22:34'
updated_date: '2026-05-23 22:34'
labels:
  - design-system
  - webui
  - extension
  - product-state
  - watchlists
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1660'
  - apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunsTab.tsx
  - apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
  - 'https://github.com/rmusser01/tldw_server/pull/2013'
parent_task_id: TASK-45.44.3
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the Watchlists RunsTab AntD Alert product-state callouts with the shared design-system Alert primitive, preserving user-facing copy/actions and removing the migrated baseline exception.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 RunsTab reliability-attention and load-error banners render through the shared design-system Alert primitive instead of AntD Alert.
- [x] #2 Focused RunsTab coverage proves the migrated banners preserve copy/actions and expose the canonical Alert marker.
- [x] #3 The RunsTab Alert baseline exception is removed without introducing new product-state verifier findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused failing RunsTab assertions requiring the reliability-attention and load-error banners to render with the design-system Alert marker.
2. Replace the RunsTab AntD Alert usages with the shared Alert primitive while preserving warning/error semantics and remediation actions.
3. Remove the migrated RunsTab Alert entry from design-system-product-state-baseline.json and run focused tests plus design-system verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TDD red/green completed. Added focused RunsTab assertions requiring both the load-error retry banner and the reliability-attention banner to be wrapped in data-ds-component="Alert"; the red run failed on the missing design-system Alert marker for both banners. Replaced the two RunsTab AntD Alert callouts with the shared design-system Alert primitive, preserved the load-error retry action plus reliability attention view/filter actions, and removed the single RunsTab Alert baseline exception. Verification: focused RunsTab load-error retry test passed 2/2; RunsTab advanced-filters test passed 5/5; product-state guard passed 54/54; bun run verify:design-system-state passed with 257 total baseline exceptions and 22 Jobs/Scheduler/Watchlists exceptions; RunsTab target rows 1 -> 0; git diff --check passed. Bandit skipped because this slice touched frontend TSX/test/JSON/Backlog markdown only.

TypeScript: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false still exits 2 on 347 existing diagnostics; no diagnostics mention RunsTab.tsx, RunsTab.load-error-retry.test.tsx, the baseline, or TASK-45.44.3.7.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the RunsTab load-error and reliability-attention banners from AntD Alert to the design-system Alert primitive. Focused coverage now verifies both banners use data-ds-component="Alert" while preserving retry, view failed run, and show failed runs actions, and the product-state baseline no longer contains the RunsTab Alert exception.
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
