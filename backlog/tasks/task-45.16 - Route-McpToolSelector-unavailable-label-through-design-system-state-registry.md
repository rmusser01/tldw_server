---
id: TASK-45.16
title: Route McpToolSelector unavailable label through design-system state registry
status: Done
assignee: []
created_date: '2026-05-08 02:55'
updated_date: '2026-05-08 02:55'
labels:
  - design-system
  - ui
  - product-state
dependencies: []
references:
  - apps/packages/ui/src/components/Common/McpToolSelector.tsx
  - apps/packages/ui/src/components/Common/__tests__/McpToolSelector.test.tsx
  - apps/packages/ui/scripts/verify-design-system-product-state.mjs
documentation:
  - Docs/Design/tldw_web_design_system_contract.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the current design-system product-state guard regression on dev by replacing McpToolSelector's hardcoded Unavailable status fallback with the canonical design-system state registry label while preserving existing translated status behavior for consumers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 McpToolSelector derives its unavailable status fallback from getDesignSystemState("unavailable") instead of hardcoding the canonical label.
- [x] #2 Existing McpToolSelector rendering and translation behavior remains intact.
- [x] #3 bun run verify:design-system-state passes on the branch.
- [x] #4 Focused McpToolSelector and product-state guard tests pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red check: bun run verify:design-system-state failed on origin/dev with blocked canonical-state-label finding for src/components/Common/McpToolSelector.tsx (Unavailable). Implementation imports getDesignSystemState and uses getDesignSystemState("unavailable").label as the statusUnavailable translation fallback, preserving the translation key and existing rendered label.

Verification: bun run verify:design-system-state passed with 520 baseline exceptions and no blocked findings. bunx vitest run src/components/Common/__tests__/McpToolSelector.test.tsx src/design-system/__tests__/product-state-guard.test.ts --maxWorkers=1 --reporter=dot passed with 2 files and 42 tests. git diff --check passed. Bandit skipped because touched code is frontend TypeScript and a Backlog task file, not Python.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Routed McpToolSelector's unavailable status fallback through the design-system state registry while preserving existing i18n behavior. This restores the product-state guard on current dev after the MCP tool selector introduced a hardcoded Unavailable label. Verified the design-system state guard, focused McpToolSelector/product-state guard tests, and diff whitespace check.
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
