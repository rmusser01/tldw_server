---
id: TASK-45.22
title: Adapt Collections StatusBadge to shared Badge
status: Done
assignee: []
created_date: '2026-05-09 04:03'
updated_date: '2026-05-09 04:11'
labels:
  - design-system
  - webui
  - guard
dependencies: []
references:
  - apps/packages/ui/src/components/Option/Collections/common/StatusBadge.tsx
  - apps/packages/ui/src/components/ui/primitives/Badge.tsx
  - apps/packages/ui/src/design-system/states.ts
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the Collections reading-status badge adapter from AntD Tag to the shared design-system Badge primitive with explicit canonical state mapping, then remove its local-status-badge baseline exception.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Collections StatusBadge renders through the shared Badge primitive while preserving reading-status labels, icons, and small/default sizing.
- [x] #2 Reading statuses map through the design-system state registry before selecting Badge variants.
- [x] #3 The local-status-badge baseline exception for src/components/Option/Collections/common/StatusBadge.tsx is removed without introducing new unbaselined findings.
- [x] #4 Focused StatusBadge tests, product-state guard tests, the design-system verifier, and diff checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the Collections StatusBadge adapter migration: replaced AntD Tag usage with the shared Badge primitive, mapped reading statuses through getDesignSystemState, preserved icon labels and compact sizing, and removed the old local-status-badge baseline exception.

Verification: watched the new focused adapter test fail before implementation because the previous component did not expose the shared Badge marker or icon test ids; after implementation, bunx vitest run src/components/Option/Collections/common/__tests__/StatusBadge.design-system.test.tsx --reporter=dot passed 5/5 tests.

Verification: bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed 46/46 tests; bun run verify:design-system-state passed and reports 513 baseline exceptions with local-status-badge reduced to 7; git diff --check passed.

Verification caveat: bunx tsc --noEmit --pretty false still exits 2 on the existing frontend TypeScript baseline, but the latest output contains no errors for src/components/Option/Collections/common/StatusBadge.tsx or its new focused test.

Bandit not run: touched implementation and test files are TypeScript/JSON/Backlog task metadata only, so the Python security scanner is not applicable to this slice.

Post-rebase verification on origin/dev after commit 370706dde: focused StatusBadge test passed 5/5, product-state guard test passed 46/46, design-system verifier passed with local-status-badge still at 7, and git diff --check passed.

Post-rebase TypeScript caveat: bunx tsc --noEmit --pretty false still exits 2 on unrelated existing frontend baseline errors; the rebased output contains no src/components/Option/Collections/common/StatusBadge.tsx errors.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Adapted Collections StatusBadge to the shared Badge primitive and design-system state registry, added focused regression coverage for labels/icons/shared Badge rendering and compact sizing, and removed the old local-status-badge baseline exception. Full frontend TypeScript remains blocked by unrelated baseline errors; focused tests, guard tests, verifier, and diff checks pass on the rebased branch.
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
