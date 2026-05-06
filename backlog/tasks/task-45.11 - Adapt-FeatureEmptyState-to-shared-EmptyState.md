---
id: TASK-45.11
title: Adapt FeatureEmptyState to shared EmptyState
status: Done
assignee: []
created_date: '2026-05-06 05:50'
updated_date: '2026-05-06 06:05'
labels:
  - design-system
  - frontend
  - library
  - ingestion
dependencies: []
documentation:
  - Docs/Design/tldw_web_design_system_inventory.md
  - Docs/Design/tldw_web_design_system_contract.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next Ingestion/Library design-system migration slice by turning the widely used Common/FeatureEmptyState compatibility component into a thin adapter around the canonical components/ui/feedback/EmptyState primitive. This should improve Media/Review/Knowledge empty-state consistency without broad consumer rewrites, Button migration, page-shell migration, or AntD mechanics changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 FeatureEmptyState renders the canonical EmptyState design-system marker while preserving its existing public props, accessible title/description/example text, actions, disabled states, className passthrough, and icon support.
- [x] #2 Focused tests cover the FeatureEmptyState adapter behavior and at least one Media/Review or Knowledge consumer path that relies on the compatibility component.
- [x] #3 The product-state guard passes without adding unexplained baseline debt; any stale baseline entry caused by migrating FeatureEmptyState is removed or documented as intentional.
- [x] #4 The PR stays scoped to the compatibility adapter and does not migrate every Media/Review/Knowledge empty state, direct AntD Empty, Button ownership, page shells, modal footers, or status chips.
- [x] #5 Focused Vitest coverage, design-system state verification, git diff checks, and Bandit applicability are recorded before completion.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Ingestion/Library design-system slice from current origin/dev. Scope: adapt Common/FeatureEmptyState to canonical EmptyState as a compatibility layer before high-volume Media/Review/Knowledge empty-state migrations.

Implemented FeatureEmptyState as a compatibility adapter over components/ui/feedback/EmptyState, preserving legacy action/title/className/icon behavior. Added focused adapter coverage, MediaTrashPage consumer coverage, and a product-state guard regression for canonical EmptyState adapters. Baseline cleanup removes migrated/stale local-empty-state entries and reconciles existing MonitoringDashboard stale IDs found by the verifier.

Verification: bunx vitest run src/components/Common/__tests__/FeatureEmptyState.test.tsx src/components/Review/__tests__/MediaTrashPage.connection.test.tsx src/design-system/__tests__/product-state-guard.test.ts --maxWorkers=1 --reporter=dot passed 43/43; bun run verify:design-system-state exited 0 with 525 allowed legacy exceptions; git diff --check exited 0. Full package tsc remains blocked by existing unrelated package-wide type errors. Bandit is not applicable to this frontend-only TypeScript/JSON slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Adapted Common/FeatureEmptyState into a thin design-system compatibility adapter over the canonical EmptyState primitive, added the small EmptyState action title passthrough needed for legacy parity, covered both adapter and MediaTrashPage usage, and taught the product-state guard not to flag adapters that genuinely render canonical EmptyState. Reconciled the guard baseline by removing migrated/stale empty-state debt and refreshing pre-existing MonitoringDashboard entries required for the verifier to stay green on current dev.
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
