---
id: TASK-45.44.11.5
title: Migrate Admin StatusBanner feedback states to design-system primitives
status: Done
assignee: []
created_date: '2026-05-29'
updated_date: '2026-05-29 17:32'
labels:
  - design-system
  - webui
  - admin
  - product-state
dependencies: []
references:
  - apps/packages/ui/src/components/Option/Admin/StatusBanner.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44.11
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate Admin StatusBanner's product-state AntD Alert and Spin usages to canonical design-system feedback primitives while preserving status, retry, and quick-action behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 StatusBanner loading state renders through the design-system LoadingState primitive without changing the visible loading copy.
- [x] #2 StatusBanner error state renders through the design-system Alert primitive and preserves sanitized error copy and retry behavior.
- [x] #3 Focused regression coverage proves the migrated loading and error states use design-system primitives.
- [x] #4 The StatusBanner Alert and Spin product-state baseline exceptions are removed, focused guard coverage passes, and the full guard log has no StatusBanner findings.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TDD red: StatusBanner focused Vitest failed because the loading branch lacked a data-ds-component="LoadingState" ancestor and the error branch lacked a data-ds-component="Alert" ancestor.

Green: replaced the AntD Spin loading branch with the design-system LoadingState primitive and replaced the AntD Alert error branch with the design-system Alert primitive while preserving sanitized error text and Retry callback behavior. Removed the two matching StatusBanner product-state baseline exceptions.

Verification:
- bunx vitest run src/components/Option/Admin/__tests__/StatusBanner.test.tsx => 3 tests passed. Bun emitted its existing localStorage warning.
- bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot => 54 tests passed. Bun emitted its existing localStorage warning.
- rg -n "StatusBanner" /tmp/design-system-status-banner.log => no output after running the full guard, confirming the remaining full-guard failures are not from StatusBanner.
- rg -n "antd-product-state-import:src/components/Option/Admin/StatusBanner" apps/packages/ui/scripts/design-system-product-state-baseline.json => no output.
- NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false with touched-path filter for StatusBanner/baseline/task => exit 0 with no touched-path output. The default-heap broad tsc attempt OOMed before diagnostics.
- git diff --check => exit 0.

Known blocker: bun run verify:design-system-state currently exits 1 on unrelated current-dev product-state drift in IntegrationPolicyPanel, WritingActionBar, Notes, and ResearchWorkspace plus stale IntegrationPolicyPanel baseline entries. This slice keeps that broader cleanup out of scope.

Bandit: skipped because this slice only changes frontend TypeScript/TSX, JSON baseline data, and Backlog markdown; no Python runtime code was touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated Admin StatusBanner loading and error feedback states from AntD Spin/Alert to the canonical design-system LoadingState and Alert primitives. Added focused regression coverage for both migrated branches, preserved sanitized error and Retry behavior, and removed the two obsolete StatusBanner product-state baseline exceptions.
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
