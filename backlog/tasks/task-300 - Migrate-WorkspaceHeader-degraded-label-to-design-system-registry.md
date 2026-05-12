---
id: TASK-300
title: Migrate WorkspaceHeader degraded label to design-system registry
status: Done
assignee: []
created_date: '2026-05-12 14:26'
labels:
  - design-system
  - frontend
  - workspace-playground
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the remaining hardcoded WorkspaceHeader degraded product-state label with the design-system registry value so the shared product-state guard baseline can shrink without changing the visible Workspace Playground status behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WorkspaceHeader renders the degraded status label from getDesignSystemState("degraded").label instead of a hardcoded string.
- [x] #2 Focused test coverage fails before the migration and passes after the registry-backed label is wired.
- [x] #3 The obsolete WorkspaceHeader canonical-state-label baseline entry is removed and the design-system product-state guard passes.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

- Replaced WorkspaceHeader's degraded connection label with `getDesignSystemState("degraded").label`.
- Added a registry-mock regression test that fails while the telemetry path uses the hardcoded degraded label.
- Removed the WorkspaceHeader `canonical-state-label` exception from the product-state baseline.
- Updated stale shortcut-modal assertions from `Focus sources/chat/studio` to the current `Focus sources/chat/studio pane` labels after the focused suite exposed the mismatch on current dev.

## Verification

- Red: `bunx vitest run src/components/Option/WorkspacePlayground/__tests__/WorkspaceHeader.test.tsx --reporter=dot` failed as expected for the new registry-label test with `to: "degraded"` instead of `to: "registry degraded"`; it also exposed the stale shortcut text assertion.
- Green: `bunx vitest run src/components/Option/WorkspacePlayground/__tests__/WorkspaceHeader.test.tsx --reporter=dot` passed 24 tests.
- `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed 52 tests.
- `bun run verify:design-system-state` passed with baseline exceptions reduced to 507 and canonical-state-label exceptions reduced to 27.
- `git diff --check` passed.
- `bunx tsc --noEmit --pretty false 2>&1 | rg -n "WorkspaceHeader|design-system-product-state-baseline"` returned no touched-path diagnostics after typing the connection fixture as `ConnectionState`.
- Bandit skipped: touched code is frontend TypeScript, JSON baseline, and task documentation only.

## Final Summary

WorkspaceHeader now gets its degraded product-state label from the design-system registry, the guard baseline no longer carries the obsolete WorkspaceHeader exception, and focused tests cover the registry-backed telemetry path.
