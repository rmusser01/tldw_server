---
id: TASK-45.44.12.5
title: Migrate WritingPlaygroundActiveSessionGuard alert to design-system Alert
status: Done
labels:
- design-system
- webui
- product-state
- writing
priority: medium
parent_task_id: TASK-45.44.12
references:
- apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundActiveSessionGuard.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- Docs/Design/tldw_web_design_system_contract.md
- https://github.com/rmusser01/tldw_server/pull/1967
modified_files:
- apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundActiveSessionGuard.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlaygroundActiveSessionGuard.design-system-alert.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the single product-state AntD Alert in `WritingPlaygroundActiveSessionGuard` to the shared design-system Alert primitive. This reduces the Writing/Review product-state baseline while preserving the session-settings load failure copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] The active-session load failure alert renders through the shared design-system Alert primitive.
- [x] The component keeps the existing session-settings error copy and child/empty/loading behavior.
- [x] The `WritingPlaygroundActiveSessionGuard` AntD Alert exception is removed from the product-state baseline.
- [x] Focused tests and design-system guard verification are recorded in this task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
- [x] Add a focused failing test that renders the active-session load failure state and asserts the alert uses the shared design-system Alert marker.
- [x] Replace the guarded AntD Alert usage in `WritingPlaygroundActiveSessionGuard` with the shared design-system Alert primitive while preserving copy and behavior.
- [x] Remove the migrated `WritingPlaygroundActiveSessionGuard` Alert row from the product-state baseline.
- [x] Run focused tests, product-state guard/unit verifier checks, baseline JSON parse, TypeScript check, and git diff whitespace checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `WritingPlaygroundActiveSessionGuard.design-system-alert.test.tsx`; red state failed because the AntD Alert did not provide the `data-ds-component="Alert"` marker.
- Replaced the load-failure AntD `Alert` with the shared design-system `Alert` using `variant="error"`. The existing translated title and loading/empty/children branches are unchanged.
- Removed the `WritingPlaygroundActiveSessionGuard` Alert baseline row. Baseline count is now 287 total exceptions, with Writing and Review surfaces at 17.
- Verification: `bunx vitest run src/components/Option/WritingPlayground/__tests__/WritingPlaygroundActiveSessionGuard.design-system-alert.test.tsx --reporter=dot` passed.
- Verification: `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed.
- Verification: `bun run verify:design-system-state` passed and reported 287 baseline exceptions / 17 Writing and Review exceptions.
- Verification: baseline JSON parse and absence check for `src/components/Option/WritingPlayground/WritingPlaygroundActiveSessionGuard.tsx` passed.
- Verification: `git diff --check` passed.
- TypeScript: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` still exits 2 on existing repo-wide UI type debt; `/tmp/tldw_writing_active_session_tsc.log` contains no diagnostics for the touched component or new test.
- Bandit: skipped because this slice only touches frontend TypeScript/TSX and JSON task metadata.
- PR: https://github.com/rmusser01/tldw_server/pull/1967
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the `WritingPlaygroundActiveSessionGuard` settings-load error state to the shared design-system Alert primitive, added focused marker coverage, and removed the retired product-state baseline exception. The slice reduces the product-state baseline to 287 total exceptions and Writing/Review to 17.
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
