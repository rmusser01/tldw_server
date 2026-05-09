---
id: TASK-45.21
title: Adapt Evaluations StatusBadge to shared Badge
status: Done
assignee: []
created_date: '2026-05-09 03:33'
updated_date: '2026-05-09 03:36'
labels:
  - design-system
  - webui
  - guard
dependencies: []
references:
  - >-
    apps/packages/ui/src/components/Option/Evaluations/components/StatusBadge.tsx
  - apps/packages/ui/src/components/ui/primitives/Badge.tsx
  - apps/packages/ui/src/design-system/states.ts
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the Evaluations run-status badge adapter from AntD Tag to the shared design-system Badge primitive with explicit canonical state mapping, then remove its local-status-badge baseline exception.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Evaluations StatusBadge renders through the shared Badge primitive while preserving known run status labels and spinner behavior for running status.
- [x] #2 Run status styling is mapped through the design-system state registry before selecting Badge variants.
- [x] #3 The local-status-badge baseline exception for src/components/Option/Evaluations/components/StatusBadge.tsx is removed without introducing new unbaselined findings.
- [x] #4 Focused StatusBadge tests, product-state guard tests, the design-system verifier, and diff checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red verification was performed before implementation: bunx vitest run src/components/Option/Evaluations/components/__tests__/StatusBadge.design-system.test.tsx --reporter=dot failed 5 tests because the existing component did not render the shared Badge primitive.

Migrated Evaluations StatusBadge from AntD Tag to the shared Badge primitive. Known run statuses now map through getDesignSystemState before selecting Badge variants: pending -> loading, running -> retrying, completed -> ready, failed -> error, cancelled -> degraded, and unknown statuses -> empty. Running status keeps the Loader2 animate-spin affordance.

Removed the src/components/Option/Evaluations/components/StatusBadge.tsx local-status-badge baseline exception.

Fresh focused verification passed: bunx vitest run src/components/Option/Evaluations/components/__tests__/StatusBadge.design-system.test.tsx --reporter=dot (6 tests); bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot (46 tests); bun run verify:design-system-state (baseline 514, local-status-badge 8); git diff --check.

Ran bunx tsc --noEmit --pretty false. It still fails on unrelated existing package-wide TypeScript errors in audio, chat composer, flashcards, playground, services, routes, store, etc.; after fixing the touched-file StatusBadge config type, no visible remaining errors are from the touched Evaluations StatusBadge files.

Bandit was not run because this slice only touches TypeScript/TSX/JSON and Backlog metadata; no Python files were changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Adapted the Evaluations run StatusBadge to render through the shared Badge primitive with canonical design-system state mapping and removed its baseline exception. Added focused regression coverage proving known statuses render through Badge and running status keeps its spinner. Focused tests, guard tests, the design-system verifier, and diff checks pass; broad local tsc remains blocked by unrelated existing package-wide errors.
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
