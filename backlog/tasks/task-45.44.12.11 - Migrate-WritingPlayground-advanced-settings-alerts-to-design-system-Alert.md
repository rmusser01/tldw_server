---
id: TASK-45.44.12.11
title: Migrate WritingPlayground advanced settings alerts to design-system Alert
status: In Progress
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
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the remaining WritingPlayground advanced-settings product-state AntD Alert usages to the shared design-system Alert primitive and close out the Writing/Review product-state baseline rows for this file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The logprobs-unavailable advanced-settings info state renders through the shared design-system Alert primitive.
- [ ] #2 The unsupported advanced sampler info state renders through the shared design-system Alert primitive.
- [ ] #3 The logit-bias validation error state renders through the shared design-system Alert primitive.
- [ ] #4 The remaining three WritingPlayground advanced-settings Alert exceptions are removed from the product-state baseline, leaving zero WritingPlayground rows.
- [ ] #5 Focused tests and design-system guard verification are recorded in this task.
- [ ] #6 Current-dev WritingActionBar product-state guard findings are resolved without adding new baseline exceptions.
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
- Bandit is not applicable for this slice because no Python code was touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
