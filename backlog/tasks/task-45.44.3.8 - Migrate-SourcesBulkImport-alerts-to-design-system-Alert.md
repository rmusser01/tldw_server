---
id: TASK-45.44.3.8
title: Migrate SourcesBulkImport alerts to design-system Alert
status: Done
assignee: []
created_date: '2026-05-24 00:34'
updated_date: '2026-05-24 00:34'
labels:
  - design-system
  - webui
  - extension
  - product-state
  - watchlists
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1660'
  - >-
    apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesBulkImport.tsx
  - >-
    apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourcesBulkImport.preflight-commit.test.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44.3
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the Watchlists SourcesBulkImport AntD Alert product-state callouts with the shared design-system Alert primitive, preserving preflight/commit copy and removing the migrated baseline exception.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SourcesBulkImport preflight/commit banners render through the shared design-system Alert primitive instead of AntD Alert.
- [x] #2 Focused SourcesBulkImport coverage proves the migrated banners preserve copy and expose the canonical Alert marker.
- [x] #3 The SourcesBulkImport Alert baseline exception is removed without introducing new product-state verifier findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused failing SourcesBulkImport assertions requiring the preflight/commit banners to render with the design-system Alert marker.
2. Replace the SourcesBulkImport AntD Alert usages with the shared Alert primitive while preserving warning/error/success semantics and user-facing copy.
3. Remove the migrated SourcesBulkImport Alert entry from design-system-product-state-baseline.json and run focused tests plus design-system verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TDD red/green completed. Added focused SourcesBulkImport assertions requiring the preflight summary and import summary banners to be wrapped in data-ds-component="Alert"; the red run failed on the missing design-system Alert marker for the preflight banner. Replaced the SourcesBulkImport AntD Alert callouts with the shared design-system Alert primitive, preserved loading/preflight/import summary copy and info/warning/error/success semantics, and removed the single SourcesBulkImport baseline exception. Verification: focused SourcesBulkImport preflight/commit test passed 5/5; product-state guard passed 54/54; bun run verify:design-system-state passed with 256 total baseline exceptions and 21 Jobs/Scheduler/Watchlists exceptions; SourcesBulkImport target rows 1 -> 0; git diff --check passed. Bandit skipped because this slice touched frontend TSX/test/JSON/Backlog markdown only.

TypeScript: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false still exits 2 on 347 existing diagnostics; no diagnostics mention SourcesBulkImport, SourcesBulkImport.preflight-commit, the baseline, or TASK-45.44.3.8.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the SourcesBulkImport loading, preflight, and import summary banners from AntD Alert to the design-system Alert primitive. Focused coverage now verifies the preflight and import summary banners use data-ds-component="Alert", and the product-state baseline no longer contains the SourcesBulkImport Alert exception.
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
