---
id: TASK-312
title: Migrate WritingPlayground diagnostics Ready label to design-system registry
status: Done
assignee: []
created_date: '2026-05-13 03:53'
updated_date: '2026-05-13 04:05'
labels:
  - design-system
  - frontend
  - writing-playground
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the remaining hardcoded WritingPlaygroundDiagnosticsPanel Ready product-state label with the canonical design-system state registry value. Scope is limited to the diagnostics panel ready status label and the matching product-state baseline entry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused regression coverage fails before the migration and passes after the registry-backed Ready label is wired.
- [x] #2 The matching WritingPlaygroundDiagnosticsPanel canonical-state-label baseline exception is removed and the design-system product-state guard passes.
- [x] #3 WritingPlaygroundDiagnosticsPanel renders the ready status label from getDesignSystemState("ready").label instead of a local hardcoded product-state string.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused failing regression test proving the ready diagnostics label comes from getDesignSystemState("ready").label. 2. Replace the hardcoded ready diagnostics label with the design-system registry value without changing warning/busy behavior. 3. Remove the matching WritingPlaygroundDiagnosticsPanel canonical-state-label baseline exception. 4. Run focused WritingPlayground test coverage, the product-state guard test, bun run verify:design-system-state, JSON/diff checks, and touched-path TypeScript filtering before opening a PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented WritingPlaygroundDiagnosticsPanel ready-label migration to getDesignSystemState("ready").label with optional fallback. Added focused Vitest coverage that mocks the registry ready label to "Registry Ready" and verifies the panel renders it. Removed the matching canonical-state-label baseline entry and refreshed the two existing AntD Alert baseline IDs shifted by the import.

Verification: bun run test src/components/Option/WritingPlayground/__tests__/WritingPlaygroundDiagnosticsPanel.design-system-state.test.tsx --reporter=dot (pass); bun run test src/design-system/__tests__/product-state-guard.test.ts --reporter=dot (pass); bun run verify:design-system-state (pass, 504 baseline exceptions / 24 canonical-state-label); node JSON parse for design-system-product-state-baseline.json (pass); git diff --check (pass); touched-path TypeScript filter over bunx tsc --noEmit --pretty false (tsc exited 2 for existing repo-wide diagnostics, no diagnostics matched touched paths).

Bandit: skipped because touched implementation/test files are frontend TypeScript/JSON plus this task record, with no Python runtime surface.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the WritingPlayground diagnostics Ready label to the design-system state registry and added a focused regression test proving the rendered label follows getDesignSystemState("ready").label. Removed the corresponding canonical-state-label baseline exception while preserving the existing AntD product-state baseline entries for later migration. Verification passed for focused Vitest, product-state guard, design-system state verifier, baseline JSON parse, diff check, and touched-path TypeScript filtering.
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
