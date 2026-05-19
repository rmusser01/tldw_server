---
id: TASK-45.33
title: Migrate LowQualityRecoveryBanner to RecoveryCallout
status: Done
assignee: []
created_date: '2026-05-09 21:49'
updated_date: '2026-05-10 00:06'
labels:
  - design-system
  - ui
  - product-state
  - recovery
dependencies: []
references:
  - >-
    apps/packages/ui/src/components/Option/KnowledgeQA/panels/LowQualityRecoveryBanner.tsx
  - >-
    apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/LowQualityRecoveryBanner.test.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
  - apps/packages/ui/scripts/design-system-product-state-rules.mjs
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the Knowledge QA low-quality answer recovery banner from bespoke recovery markup to the shared design-system RecoveryCallout primitive. This should remove exactly one remaining local-recovery-banner baseline exception while preserving the existing recovery actions and dismissal behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 LowQualityRecoveryBanner renders the shared RecoveryCallout design-system primitive.
- [x] #2 Existing refine, enable web, select sources, and dismiss interactions remain covered by focused tests.
- [x] #3 The LowQualityRecoveryBanner local-recovery-banner baseline entry is removed and the design-system verifier passes with the remaining expected baseline debt.
- [x] #4 Verification records focused component tests, product-state guard tests, design-system verifier, syntax/whitespace checks, and any known TypeScript/Bandit skips.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused failing test that LowQualityRecoveryBanner renders RecoveryCallout while preserving existing interaction tests. 2. Replace bespoke warning panel markup with RecoveryCallout plus shared action mapping and explicit dismiss control. 3. Remove the LowQualityRecoveryBanner local-recovery-banner baseline exception. 4. Run focused tests, product-state guard tests, design-system verifier, syntax/whitespace checks, and a touched-file TypeScript filter.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the LowQualityRecoveryBanner migration to the shared RecoveryCallout primitive. Added a focused test that first failed because no data-ds-component marker existed, then replaced the bespoke warning panel and custom buttons with RecoveryCallout primary/secondary actions while preserving refine, enable web, select sources, and dismiss callbacks. Removed the stale local-recovery-banner baseline entry for LowQualityRecoveryBanner.

Verification: RED run of bunx vitest run src/components/Option/KnowledgeQA/__tests__/LowQualityRecoveryBanner.test.tsx --reporter=dot failed the new RecoveryCallout marker assertion because closest([data-ds-component]) returned null. GREEN/final run passed 6/6. bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed 52/52. bun run verify:design-system-state passed with Baseline exceptions: 509 and local-recovery-banner: 2. git diff --check passed. bunx tsc --noEmit --pretty false exited 2 with 236 lines of existing unrelated UI type errors; touched-file filter for LowQualityRecoveryBanner, design-system-product-state-baseline, product-state-guard, RecoveryCallout, and task-45.33 returned no diagnostics. Bandit skipped because touched files are UI TS/TSX, JSON baseline, and Markdown task metadata only.

PR review follow-up: Qodo flagged that the RecoveryCallout migration dropped the prior live-region status semantics and the dismiss action's specific accessible name. Gemini also requested reducing repeated action-label markup. Reopening the task to address those review comments before re-verifying and pushing.

Review fix implementation: added regression tests proving the RecoveryCallout banner exposes role=status with polite/atomic live-region semantics and that the dismiss action keeps the visible label "Dismiss" while exposing the accessible name "Dismiss recovery suggestions". Added StatePanel live-region passthrough props, StateAction ariaLabel passthrough through ActionGroup, and a local ActionLabel helper to remove repeated icon label markup.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Change summary: migrated Knowledge QA LowQualityRecoveryBanner from bespoke recovery panel markup to the shared RecoveryCallout primitive, then addressed PR review accessibility regressions by restoring live-region status semantics and the contextual dismiss accessible name. The shared StatePanel now supports optional live-region passthrough props, ActionGroup supports per-action ariaLabel, and LowQualityRecoveryBanner uses a local ActionLabel helper for the repeated icon+label markup.

Why: the migration should reduce local recovery-banner debt without weakening the banner's previous accessibility behavior. Keeping these as optional primitive hooks preserves the design-system path while allowing conditionally mounted state UI to remain announceable and context-rich for assistive tech.

Verification: RED review-regression tests failed before implementation for missing role=status and missing "Dismiss recovery suggestions" accessible name. Final focused LowQualityRecoveryBanner tests passed 7/7, state primitive tests passed 7/7, product-state guard tests passed 52/52, design-system verifier passed with 509 baseline exceptions and 2 remaining local-recovery-banner entries, git diff whitespace check passed, and full UI type-check still reports only unrelated existing baseline diagnostics with no touched-file hits for LowQualityRecoveryBanner, ActionGroup, StatePanel, RecoveryCallout, state-primitives, or task-45.33. Bandit is not applicable for this UI-only TS/TSX/Markdown change.
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
