---
id: TASK-45.27
title: Adapt Layouts ConnectionStatus dot to shared Badge
status: Done
assignee: []
created_date: '2026-05-09 16:46'
updated_date: '2026-05-09 16:50'
labels:
  - design-system
  - webui
dependencies: []
references:
  - apps/packages/ui/src/components/Layouts/ConnectionStatus.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the Layouts ConnectionStatus local StatusDot product-state indicator to the shared Badge primitive and design-system state registry. This targets the remaining local-status-badge baseline entry for src/components/Layouts/ConnectionStatus.tsx without changing navigation or health-diagnostics behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ConnectionStatus renders its status indicator through the shared Badge primitive while preserving the existing clickable health diagnostics button behavior
- [x] #2 StatusKind values map through getDesignSystemState before selecting Badge variants
- [x] #3 The local-status-badge baseline exception for src/components/Layouts/ConnectionStatus.tsx is removed without introducing new guard findings
- [x] #4 Focused component coverage and design-system guard verification are recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-05-09: Added a focused red test for Layouts/ConnectionStatus requiring the dot indicator to render through data-ds-component="Badge" across connected, checking, unconfigured, and offline states while preserving custom click handling and default /settings/health navigation. Initial run failed because the existing indicator was a raw span with no connection-status-dot-badge element.

2026-05-09: Migrated StatusDot to return the shared Badge primitive. ConnectionStatus now maps core connection status through getDesignSystemState using ready, retrying, setup_required, and unavailable before selecting Badge variants. Removed the ConnectionStatus local-status-badge baseline exception.

Verification: bunx vitest run src/components/Layouts/__tests__/ConnectionStatus.design-system.test.tsx --reporter=dot -> 6 passed. bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot -> 46 passed. bun run verify:design-system-state -> passed; baseline exceptions 509 and local-status-badge 3. git diff --check -> passed. bunx tsc --noEmit --pretty false | rg touched files -> no touched-file diagnostics (rg exit 1/no matches).

Bandit skip: touched runtime/test files are TypeScript/TSX plus JSON Backlog metadata; no Python security surface changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated Layouts ConnectionStatus status dot to the shared Badge primitive and design-system state registry. The component now derives canonical state keys for connected, checking, unconfigured, and offline server states, renders the compact dot inside a Badge, keeps diagnostics click behavior intact, and removes the obsolete baseline exception.
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
