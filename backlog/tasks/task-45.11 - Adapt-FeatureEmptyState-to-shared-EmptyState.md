---
id: TASK-45.11
title: Adapt FeatureEmptyState to shared EmptyState
status: Done
assignee: []
created_date: '2026-05-06 05:50'
updated_date: '2026-05-06 15:12'
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

PR review pass started for PR #1341. Actionable findings: Qodo guard suppression should be scoped per owner instead of file-wide; CodeRabbit baseline canonical-state-label entries should be grouped with canonical labels.

PR review fixes: scoped canonical EmptyState guard suppression to the owner component instead of a file-wide flag, added the sibling LegacyEmptyState regression, and moved MonitoringDashboard canonical-state-label baseline entries into the canonical-label group.

Review-fix verification: bunx vitest run src/components/Common/__tests__/FeatureEmptyState.test.tsx src/components/Review/__tests__/MediaTrashPage.connection.test.tsx src/design-system/__tests__/product-state-guard.test.ts --maxWorkers=1 --reporter=dot passed 44/44; bun run verify:design-system-state exited 0 with 525 allowed legacy exceptions; git diff --check exited 0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Adapted FeatureEmptyState to the shared EmptyState primitive and completed PR review fixes. The product-state guard now suppresses local-empty-state only for the component that actually renders canonical EmptyState, with regression coverage for sibling empty-state components, and the baseline canonical-state-label entries are grouped with the canonical-label section.
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
