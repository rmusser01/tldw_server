---
id: TASK-45.44.12.6
title: Migrate WritingPlaygroundTokenInspectorCard alerts to design-system Alert
status: In Progress
labels:
- design-system
- webui
- product-state
- writing
priority: medium
parent_task_id: TASK-45.44.12
references:
- apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundTokenInspectorCard.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- Docs/Design/tldw_web_design_system_contract.md
modified_files:
- apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundTokenInspectorCard.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlaygroundTokenInspectorCard.design-system-alert.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the token inspector unavailable/error product-state AntD Alerts to the shared design-system Alert primitive. This reduces the Writing/Review product-state baseline while preserving token inspector status copy and actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The token inspector unavailable state renders through the shared design-system Alert primitive.
- [x] #2 The token inspector error state renders through the shared design-system Alert primitive.
- [x] #3 The component keeps existing token inspector copy, actions, and result tag behavior.
- [x] #4 The `WritingPlaygroundTokenInspectorCard` AntD Alert exceptions are removed from the product-state baseline.
- [x] #5 Focused tests and design-system guard verification are recorded in this task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
- [x] Add focused failing tests that render the token inspector unavailable and error states and assert each alert uses the shared design-system Alert marker.
- [x] Replace the guarded AntD Alert usages in `WritingPlaygroundTokenInspectorCard` with the shared design-system Alert primitive while preserving copy and behavior.
- [x] Remove the migrated `WritingPlaygroundTokenInspectorCard` Alert rows from the product-state baseline.
- [x] Run focused tests, product-state guard/unit verifier checks, baseline JSON parse, TypeScript check, and git diff whitespace checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `WritingPlaygroundTokenInspectorCard.design-system-alert.test.tsx`; red state failed because the AntD Alerts did not provide the `data-ds-component="Alert"` marker.
- Replaced the token inspector unavailable/error AntD `Alert` usages with the shared design-system `Alert` using `variant="info"` and `variant="error"`. Existing message text and action/result tag behavior are unchanged.
- Removed both `WritingPlaygroundTokenInspectorCard` Alert baseline rows. Baseline count is now 285 total exceptions, with Writing and Review surfaces at 15.
- Verification: `bunx vitest run src/components/Option/WritingPlayground/__tests__/WritingPlaygroundTokenInspectorCard.design-system-alert.test.tsx --reporter=dot` passed.
- Verification: `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed.
- Verification: `bun run verify:design-system-state` passed and reported 285 baseline exceptions / 15 Writing and Review exceptions.
- Verification: baseline JSON parse and absence check for `src/components/Option/WritingPlayground/WritingPlaygroundTokenInspectorCard.tsx` passed.
- Verification: `git diff --check` passed.
- TypeScript: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` still exits 2 on existing repo-wide UI type debt; `/tmp/tldw_writing_token_inspector_tsc.log` contains no diagnostics for the touched component or new test.
- Bandit: skipped because this slice only touches frontend TypeScript/TSX and JSON task metadata.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
