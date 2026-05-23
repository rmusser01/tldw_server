---
id: TASK-45.44.12.11
title: Migrate WritingPlayground advanced settings alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- product-state
- writing
priority: medium
parent_task_id: TASK-45.44.12
references:
- apps/packages/ui/src/components/Option/WritingPlayground/index.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/WritingActionBar.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.shell-design-system-alert.test.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingActionBar.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- Docs/Design/tldw_web_design_system_contract.md
modified_files:
- apps/packages/ui/src/components/Option/WritingPlayground/index.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/WritingActionBar.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.shell-design-system-alert.test.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingActionBar.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- apps/packages/ui/src/assets/locale/en/option.json
- apps/packages/ui/src/public/_locales/en/option.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the remaining WritingPlayground advanced-settings product-state AntD Alert usages to the shared design-system Alert primitive and close out the Writing/Review product-state baseline rows for this file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The logprobs-unavailable advanced-settings info state renders through the shared design-system Alert primitive.
- [x] #2 The unsupported advanced sampler info state renders through the shared design-system Alert primitive.
- [x] #3 The logit-bias validation error state renders through the shared design-system Alert primitive.
- [x] #4 The remaining three WritingPlayground advanced-settings Alert exceptions are removed from the product-state baseline, leaving zero WritingPlayground rows.
- [x] #5 Focused tests and design-system guard verification are recorded in this task.
- [x] #6 Current-dev WritingActionBar product-state guard findings are resolved without adding new baseline exceptions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
- [x] Add focused failing tests for the three advanced-settings alert branches and assert each uses the shared design-system Alert marker.
- [x] Replace only the remaining WritingPlayground advanced-settings AntD Alert usages with the shared design-system Alert primitive while preserving copy and branch behavior.
- [x] Resolve current-dev WritingActionBar product-state guard blockers by using the canonical ready-state label and design-system Alert primitive.
- [x] Remove the three remaining WritingPlayground baseline rows from the product-state baseline.
- [x] Run focused tests, product-state guard/unit verifier checks, baseline JSON parse, TypeScript check, and git diff whitespace checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added focused coverage for the three remaining WritingPlayground advanced-settings alert branches: logprobs unavailable, unsupported advanced sampler controls, and invalid logit-bias JSON. Each test asserts the rendered copy is inside `[data-ds-component="Alert"]`.
- Replaced the three remaining `WritingPlayground/index.tsx` advanced-settings AntD Alert usages with the shared design-system Alert primitive and removed the final three WritingPlayground baseline rows.
- Current `dev` introduced two unbaselined `WritingActionBar` guard blockers in this surface. This slice resolves them by using `READY_STATE_LABEL` for the available tag and the design-system Alert primitive for whole-document confirmation warnings, with focused coverage for both paths.
- Verification run before PR creation: `bunx vitest run src/components/Option/WritingPlayground/__tests__/WritingPlayground.shell-design-system-alert.test.tsx --reporter=dot` (7 passed), `bunx vitest run src/components/Option/WritingPlayground/__tests__/WritingActionBar.test.tsx --reporter=dot` (9 passed), `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` (54 passed), `bun run verify:design-system-state` (passed; 265 baseline exceptions), baseline parse check (`total: 265`, `writingRows: 0`), and `git diff --check` (passed).
- `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` exits 2 with existing repo-wide TypeScript debt. The touched source-file diagnostics are on unchanged lines for existing revision-context/stop-string typing in `WritingPlayground/index.tsx` and existing AntD input ref typing in `WritingActionBar.tsx`; this migration does not introduce those diagnostics.
- PR review follow-up: verified the reported missing `DesignSystemAlert` import was already present in `WritingPlayground/index.tsx`, localized the `WritingActionBar` generation-unavailable and broad-target fallback labels via `react-i18next`, added matching English locale entries, and tightened the AntD input ref types. After the ref cleanup, the TypeScript check still exits 2 on repo-wide debt, but no longer reports `WritingActionBar.tsx`; remaining touched-file diagnostics are the pre-existing `WritingPlayground/index.tsx` revision typing issues.
- Bandit is not applicable for this slice because no Python code was touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR: https://github.com/rmusser01/tldw_server/pull/1998

Migrated the final WritingPlayground advanced-settings product-state Alert branches to the shared design-system Alert primitive, removed the last three WritingPlayground baseline entries, and resolved the current-dev WritingActionBar guard blockers without adding new baseline exceptions. Focused tests and product-state verification passed on the rebased branch. TypeScript still exits nonzero on existing repo-wide debt; touched-file diagnostics are on unchanged pre-existing lines and are documented in Implementation Notes.

PR review follow-up localized the `WritingActionBar` fallback labels, added English locale entries, and fixed the pre-existing AntD input ref typing in `WritingActionBar`. The `DesignSystemAlert` import comment was verified as already satisfied by the current branch.
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
