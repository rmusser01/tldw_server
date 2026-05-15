---
id: TASK-45.44.4.1
title: Migrate MCPHub governance pack Blocked labels to design-system state registry
status: Done
assignee: []
created_date: '2026-05-15 02:03'
updated_date: '2026-05-15 02:14'
labels:
  - design-system
  - webui
  - product-state
dependencies: []
references:
  - apps/packages/ui/src/components/Option/MCPHub/GovernancePacksTab.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44.4
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the remaining GovernancePacksTab hardcoded Blocked canonical-state-label exceptions by routing them through the design-system state registry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 GovernancePacksTab Blocked verdict and upgrade labels use the canonical blocked state label.
- [x] #2 The product-state baseline no longer contains GovernancePacksTab canonical-state-label Blocked entries.
- [x] #3 Focused MCPHub tests and design-system product-state verification pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented on branch codex/design-system-next-slice-4 after rebasing onto origin/dev at e66f1e75e. GovernancePacksTab now uses BLOCKED_STATE_LABEL from the design-system registry for blocked dry-run verdicts and blocked upgrade plans.

Baseline update: removed the two GovernancePacksTab canonical-state-label Blocked exceptions. The remaining baseline after verification is 484 allowed legacy exceptions: 479 antd-product-state-import and 5 canonical-state-label.

Verification passing: bunx vitest run src/components/Option/MCPHub/__tests__/GovernancePacksTab.test.tsx src/design-system/__tests__/states.test.ts src/design-system/__tests__/product-state-guard.test.ts --reporter=dot; bun run verify:design-system-state; git diff --check.

TypeScript note: bunx tsc --noEmit --pretty false still exits 2 on existing package-wide type debt, including unrelated tests and existing Playground type errors.

Bandit not run: touched runtime scope is UI TypeScript plus JSON baseline and Backlog metadata, with no Python execution path.

PR link: https://github.com/rmusser01/tldw_server/pull/1712
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated MCPHub GovernancePacksTab blocked verdict and upgrade labels to the design-system state registry, added regression coverage, and removed the two corresponding canonical-state-label baseline exceptions. Focused tests and the design-system verifier pass; package-wide tsc remains blocked by existing unrelated type debt.
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
