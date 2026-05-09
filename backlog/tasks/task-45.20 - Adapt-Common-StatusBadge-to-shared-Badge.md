---
id: TASK-45.20
title: Adapt Common StatusBadge to shared Badge
status: Done
assignee: []
created_date: '2026-05-09 00:49'
updated_date: '2026-05-09 00:56'
labels:
  - design-system
  - webui
  - guard
dependencies: []
references:
  - apps/packages/ui/src/components/Common/StatusBadge.tsx
  - apps/packages/ui/src/components/ui/primitives/Badge.tsx
  - apps/packages/ui/src/design-system/states.ts
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the shared Common StatusBadge adapter onto the design-system Badge primitive with explicit state-registry mapping, then remove its local-status-badge baseline exception.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Common StatusBadge renders through the shared Badge primitive while preserving demo/warning/error labels.
- [x] #2 The product-state guard recognizes status-badge compatibility adapters only when they return Badge and use the design-system state registry.
- [x] #3 The local-status-badge baseline exception for src/components/Common/StatusBadge.tsx is removed without introducing new unbaselined findings.
- [x] #4 Focused StatusBadge tests, product-state guard tests, the design-system verifier, and diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing focused tests for Common StatusBadge rendering through shared Badge and for the guard allowing only registry-backed Badge adapters.
2. Update the product-state guard to recognize status-badge compatibility adapters that directly return Badge and use getDesignSystemState.
3. Migrate Common/StatusBadge.tsx to Badge plus design-system state mapping while preserving its public API.
4. Remove the Common StatusBadge baseline entry and verify the focused suite.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red verification was performed before implementation: the new StatusBadge design-system test failed because Common/StatusBadge did not render the shared Badge primitive; the new guard test failed because StatusBadge adapters returning Badge were still reported as local-status-badge.

Implemented Common/StatusBadge as a compatibility adapter over the shared Badge primitive with variant mapping through getDesignSystemState. The product-state guard now only allows status-badge adapters when the same owner directly returns Badge and uses the state registry.

Fresh focused verification passed: bunx vitest run src/components/Common/__tests__/StatusBadge.design-system.test.tsx --reporter=dot (3 tests); bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot (45 tests); bun run verify:design-system-state (baseline 515, local-status-badge 9); git diff --check.

Ran bunx tsc --noEmit --pretty false; it failed with existing unrelated package-wide TypeScript errors in audio, chat composer, flashcards, playground, services, routes, and store tests. No visible errors were from the touched StatusBadge, guard, or design-system test files.

Bandit was not run because this slice only touches TypeScript/TSX/JavaScript/JSON and Backlog task metadata; no Python files were changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Adapted Common/StatusBadge to render through the shared Badge primitive while preserving the existing demo/warning/error public API. The migration maps status variants through the design-system state registry, removes the Common StatusBadge local-status-badge baseline exception, and tightens the product-state guard so only same-owner adapters that directly return Badge and use getDesignSystemState are treated as canonical compatibility adapters. Focused StatusBadge tests, product-state guard tests, the design-system verifier, and diff checks pass; the broader TypeScript check remains blocked by unrelated existing package-wide errors.
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
