---
id: TASK-45.45
title: >-
  Migrate MonitoringDashboardPage sandbox readiness labels to design-system
  state registry
status: Done
assignee:
  - Codex
created_date: '2026-05-14 07:30'
updated_date: '2026-05-14 07:35'
labels:
  - design-system
  - frontend
  - product-state
dependencies: []
references:
  - apps/packages/ui/src/components/Option/Admin/MonitoringDashboardPage.tsx
  - >-
    apps/packages/ui/src/components/Option/Admin/__tests__/MonitoringDashboardPage.test.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
documentation:
  - Docs/Design/tldw_web_design_system_contract.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the admin MonitoringDashboardPage sandbox runtime summary labels for ready and unavailable readiness counts with the canonical design-system state registry labels while preserving the existing diagnostics layout and count values. This continues the design-system product-state migration against the current green verifier baseline on origin/dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MonitoringDashboardPage resolves ready and unavailable sandbox readiness labels through getDesignSystemState instead of hardcoded product-state literals.
- [x] #2 Focused component coverage proves mocked design-system ready and unavailable labels render in the sandbox runtime isolation summary.
- [x] #3 The canonical-state-label baseline entries for MonitoringDashboardPage Ready and Unavailable are removed and the design-system product-state verifier passes.
- [x] #4 Verification records focused Vitest, product-state guard/verifier status, diff check, and TypeScript/Bandit applicability.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused MonitoringDashboardPage coverage that mocks getDesignSystemState('ready') and getDesignSystemState('unavailable') to distinct labels and expects those labels in the sandbox runtime isolation summary.
2. Confirm the focused test fails before implementation because MonitoringDashboardPage still renders hardcoded Ready/Unavailable labels.
3. Import getDesignSystemState in MonitoringDashboardPage and use registry labels for the sandbox diagnostics summary items, with empty-string fallbacks so hardcoded canonical labels do not remain in product-state code.
4. Remove the two MonitoringDashboardPage canonical-state-label baseline entries and refresh the existing MonitoringDashboardPage AntD Alert baseline IDs that changed only because this touched file line numbers moved.
5. Verify with focused MonitoringDashboardPage Vitest, product-state guard tests, bun run verify:design-system-state, git diff --check, and broad TypeScript touched-file review; document Bandit as skipped because the final touched scope is UI TypeScript/JSON/Backlog only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the MonitoringDashboardPage readiness summary through getDesignSystemState for ready and unavailable. The RED Vitest failed on the mocked registry labels before implementation, then passed after wiring the labels through the registry. The product-state verifier also required removing hardcoded fallback literals and refreshing existing MonitoringDashboardPage AntD Alert baseline IDs that changed due to line movement; no AntD Alert migration was included in this slice.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the admin MonitoringDashboardPage sandbox readiness summary labels for ready and unavailable counts to the design-system state registry. Added focused test coverage with mocked registry labels so the component is no longer coupled to hardcoded product-state literals. Removed the two MonitoringDashboardPage canonical-state-label baseline entries and refreshed only the existing MonitoringDashboardPage AntD Alert baseline IDs whose generated hashes changed because the file moved lines.

Verification: RED focused Vitest failed on the mocked registry labels before implementation; GREEN focused Vitest passed 9/9; product-state guard tests passed 52/52; bun run verify:design-system-state passed with 494 allowed legacy exceptions and canonical-state-label reduced to 15; git diff --check passed. Broad bunx tsc --noEmit --pretty false still exits 2 on existing repo-wide UI/test type debt, with no observed MonitoringDashboardPage, design-system baseline, or touched-file errors in the output. Bandit skipped because this slice only touches UI TypeScript, JSON baseline data, and Backlog metadata.
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
