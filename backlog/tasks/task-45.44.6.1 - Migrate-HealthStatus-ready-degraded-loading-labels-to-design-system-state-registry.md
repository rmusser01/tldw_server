---
id: TASK-45.44.6.1
title: >-
  Migrate HealthStatus ready degraded loading labels to design-system state
  registry
status: Done
assignee:
  - Codex
created_date: '2026-05-14 19:46'
updated_date: '2026-05-14 19:51'
labels:
  - design-system
  - webui
  - extension
  - product-state
dependencies: []
references:
  - apps/packages/ui/src/components/Option/Settings/health-status.tsx
  - >-
    apps/packages/ui/src/components/Option/Settings/__tests__/health-status.design-system.test.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
documentation:
  - >-
    Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
parent_task_id: TASK-45.44.6
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the remaining HealthStatus hardcoded ready, degraded, and loading product-state labels with the canonical design-system state registry while preserving existing translated copy behavior, health-check rendering, and product-state baseline enforcement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 HealthStatus resolves ready, degraded, and loading labels through getDesignSystemState instead of hardcoded canonical literals.
- [x] #2 Focused HealthStatus coverage proves mocked design-system labels render for summary and per-check status labels.
- [x] #3 The six HealthStatus canonical-state-label baseline entries are removed and the design-system product-state verifier passes.
- [x] #4 Verification records focused Vitest, product-state guard/verifier status, diff check, and TypeScript/Bandit applicability.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend the focused HealthStatus design-system test to mock getDesignSystemState for ready, degraded, and loading labels and assert those mocked labels render in both summary and per-check status surfaces.
2. Run the focused test to confirm it fails before implementation because HealthStatus still uses hardcoded fallback labels.
3. Import getDesignSystemState in health-status.tsx and create module-scope state label constants for ready, degraded, and loading. Use those labels as translation fallbacks in describeStatus and the per-check Tag labels.
4. Remove the six HealthStatus canonical-state-label baseline entries from design-system-product-state-baseline.json.
5. Verify with the focused HealthStatus Vitest, product-state guard tests, bun run verify:design-system-state, git diff --check, and document Bandit as skipped because the touched runtime scope is UI TypeScript/JSON/Backlog only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation replaced HealthStatus ready/degraded/loading translation fallback literals with getDesignSystemState labels and added focused coverage that mocks registry labels for summary plus per-check status surfaces.

Verified red before implementation: focused HealthStatus design-system test failed for per-check Ready, per-check Degraded, and Loading registry labels while HealthStatus still used hardcoded fallbacks.

Verification passing: bunx vitest run src/components/Option/Settings/__tests__/health-status.design-system.test.tsx --reporter=dot; bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot; bun run verify:design-system-state; git diff --check.

TypeScript note: bunx tsc --noEmit --pretty false exits 2 on existing package-wide type debt in unrelated tests/modules; no touched HealthStatus files appeared in the reported errors.

Bandit not run: touched runtime scope is UI TypeScript plus JSON baseline and Backlog metadata, with no Python execution path.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated HealthStatus ready, degraded, and loading product-state label fallbacks to the design-system state registry, added focused registry-label assertions, removed the six resolved HealthStatus canonical-state-label baseline entries, and refreshed four shifted HealthStatus AntD Alert baseline ids caused by nearby line movement.
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
