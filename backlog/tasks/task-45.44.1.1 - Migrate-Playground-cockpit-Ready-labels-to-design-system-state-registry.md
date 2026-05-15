---
id: TASK-45.44.1.1
title: Migrate Playground cockpit Ready labels to design-system state registry
status: Done
assignee:
  - Codex
created_date: '2026-05-15 01:10'
updated_date: '2026-05-15 01:15'
labels:
  - design-system
  - webui
  - extension
  - product-state
dependencies: []
references:
  - >-
    apps/packages/ui/src/components/Option/Playground/playground-cockpit-summaries.ts
  - apps/packages/ui/src/components/Option/Playground/Playground.tsx
  - apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
documentation:
  - >-
    Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
parent_task_id: TASK-45.44.1
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the newly unbaselined Playground cockpit Ready product-state labels with the canonical design-system state registry so the product-state verifier stays closed on current dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Playground cockpit session Ready fallbacks resolve through getDesignSystemState instead of hardcoded canonical literals.
- [x] #2 The current three unbaselined Playground Ready findings are eliminated without adding baseline debt.
- [x] #3 Design-system product-state verifier passes after the Playground and ExtensionStartPanel migrations.
- [x] #4 Verification records focused Vitest, product-state guard/verifier status, diff check, and TypeScript/Bandit applicability.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Use the product-state verifier failure as the red check for the three current Playground Ready labels.
2. Add READY_STATE_LABEL constants from getDesignSystemState('ready') in the affected Playground cockpit modules.
3. Replace only the three Ready translation/default fallbacks called out by the verifier.
4. Verify with focused existing Playground cockpit tests where available, product-state guard tests, bun run verify:design-system-state, and diff checks.
5. Record verification and skip notes in Backlog.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation replaced the three current unbaselined Playground cockpit Ready fallbacks with getDesignSystemState('ready').label in playground-cockpit-summaries.ts, Playground.tsx, and PlaygroundContextRail.tsx.

Verified red before implementation: bun run verify:design-system-state reported three blocked Playground Ready canonical-state-label findings on current origin/dev.

Verification passing: bunx vitest run src/components/Option/PresentationStudio/__tests__/ExtensionStartPanel.design-system.test.tsx src/components/Option/Playground/__tests__/playground-cockpit-summaries.test.ts src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx --reporter=dot; bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot; bun run verify:design-system-state; git diff --check.

TypeScript note: bunx tsc --noEmit --pretty false exits 2 on existing package-wide type debt, including unrelated current Playground errors outside the touched lines.

Bandit not run: touched runtime scope is UI TypeScript plus JSON baseline and Backlog metadata, with no Python execution path.

PR link: https://github.com/rmusser01/tldw_server/pull/1709
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated three Playground cockpit Ready fallbacks to the design-system state registry and eliminated the new unbaselined Playground canonical-state-label verifier findings without adding baseline debt.
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
