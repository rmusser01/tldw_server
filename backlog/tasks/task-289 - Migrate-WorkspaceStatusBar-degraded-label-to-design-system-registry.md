---
id: TASK-289
title: Migrate WorkspaceStatusBar degraded label to design-system registry
status: Done
assignee: []
created_date: '2026-05-12 04:26'
labels:
  - design-system
  - frontend
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the hardcoded degraded connection label in WorkspacePlayground WorkspaceStatusBar with the design-system state registry fallback while preserving connection tone and retry behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The degraded connection indicator in WorkspaceStatusBar displays the design-system degraded state label.
- [x] #2 Focused tests prove the degraded connection label comes from the registry without source-string assertions.
- [x] #3 The matching canonical-state-label baseline exception is removed and the design-system state guard passes.
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused WorkspaceStatusBar test that mocks the design-system registry and verifies degraded connection status uses the registry-provided label.
2. Replace the hardcoded degraded connection label with `getDesignSystemState("degraded").label` while preserving tone classes, details, and retry behavior.
3. Remove the matching `canonical-state-label` baseline exception and verify the product-state guard still passes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Red test first: `bunx vitest run src/components/Option/WorkspacePlayground/__tests__/WorkspaceStatusBar.test.tsx --reporter=dot` failed because the degraded connection indicator still rendered `Degraded` instead of the mocked registry label `Registry Degraded`.
- The degraded connection tone now reads its label from the design-system state registry; connected and disconnected labels remain unchanged.
- Removed the `canonical-state-label:src/components/Option/WorkspacePlayground/WorkspaceStatusBar.tsx:Degraded` baseline entry.
- Tightened the existing connection-state test mock typing to `ConnectionState` and corrected the auth error test fixture from `error_auth` to the valid `auth` error kind.
- Bandit skipped: touched implementation is frontend TypeScript/test JSON only, with no Python code path.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated WorkspaceStatusBar's degraded connection indicator label to the design-system state registry by resolving `getDesignSystemState("degraded")` and rendering its label for degraded connection UX states. Added focused coverage that mocks the registry while preserving real design-system module exports, proving the degraded indicator uses the registry-provided label without source-string assertions.

Removed the now-obsolete canonical state label baseline exception. Verification passed with the focused WorkspaceStatusBar test, the product-state guard unit suite, the design-system guard CLI, `git diff --check`, and an exact touched-path TypeScript error filter.
<!-- SECTION:FINAL_SUMMARY:END -->
