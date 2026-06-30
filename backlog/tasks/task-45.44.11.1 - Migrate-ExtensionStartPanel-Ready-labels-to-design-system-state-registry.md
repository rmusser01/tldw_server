---
id: TASK-45.44.11.1
title: Migrate ExtensionStartPanel Ready labels to design-system state registry
status: Done
assignee:
  - Codex
created_date: '2026-05-15 01:07'
updated_date: '2026-05-15 01:14'
labels:
  - design-system
  - webui
  - extension
  - product-state
dependencies: []
references:
  - >-
    apps/packages/ui/src/components/Option/PresentationStudio/ExtensionStartPanel.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
documentation:
  - >-
    Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
parent_task_id: TASK-45.44.11
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the remaining ExtensionStartPanel hardcoded Ready product-state labels with the canonical design-system state registry while preserving existing Empty copy and launch-option rendering.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ExtensionStartPanel resolves seeded narration and image Ready labels through getDesignSystemState instead of hardcoded canonical literals.
- [x] #2 Focused coverage proves mocked design-system Ready label renders for narration and image seed status rows.
- [x] #3 The two ExtensionStartPanel canonical-state-label baseline entries are removed and the design-system product-state verifier passes.
- [x] #4 Verification records focused Vitest, product-state guard/verifier status, diff check, and TypeScript/Bandit applicability.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused ExtensionStartPanel design-system coverage that mocks getDesignSystemState('ready') and asserts the mocked Ready label appears for narration and image seed status rows while Empty remains unchanged.
2. Run the focused test red to confirm current hardcoded Ready labels fail the registry assertion.
3. Import getDesignSystemState in ExtensionStartPanel and use a module-level READY_STATE_LABEL for the two Ready status fallbacks.
4. Remove the two ExtensionStartPanel canonical-state-label baseline entries.
5. Verify with focused Vitest, product-state guard tests, bun run verify:design-system-state, git diff --check, and document Bandit/TypeScript applicability.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation replaced ExtensionStartPanel seeded narration and image Ready labels with getDesignSystemState('ready').label while preserving Empty copy.

Verified red before implementation: focused ExtensionStartPanel design-system test failed because seeded status rows rendered literal Ready instead of the mocked registry label.

Verification passing: bunx vitest run src/components/Option/PresentationStudio/__tests__/ExtensionStartPanel.design-system.test.tsx src/components/Option/Playground/__tests__/playground-cockpit-summaries.test.ts src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx --reporter=dot; bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot; bun run verify:design-system-state; git diff --check.

TypeScript note: bunx tsc --noEmit --pretty false exits 2 on existing package-wide type debt, including unrelated current Playground errors outside the touched lines.

Bandit not run: touched runtime scope is UI TypeScript plus JSON baseline and Backlog metadata, with no Python execution path.

PR link: https://github.com/rmusser01/tldw_server/pull/1709
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated ExtensionStartPanel Ready status labels to the design-system state registry, added focused registry-label coverage, and removed the two resolved ExtensionStartPanel canonical-state-label baseline entries.
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
