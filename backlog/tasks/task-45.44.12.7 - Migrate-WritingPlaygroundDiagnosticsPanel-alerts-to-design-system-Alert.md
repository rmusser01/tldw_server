---
id: TASK-45.44.12.7
title: Migrate WritingPlaygroundDiagnosticsPanel alerts to design-system Alert
status: In Progress
labels:
- design-system
- webui
- product-state
- writing
priority: medium
parent_task_id: TASK-45.44.12
references:
- apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundDiagnosticsPanel.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- Docs/Design/tldw_web_design_system_contract.md
modified_files:
- apps/packages/ui/src/components/Option/WritingPlayground/WritingPlaygroundDiagnosticsPanel.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlaygroundDiagnosticsPanel.design-system-alert.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the diagnostics panel offline and unsupported product-state AntD Alerts to the shared design-system Alert primitive. This reduces the Writing/Review product-state baseline while preserving diagnostics copy and child-card behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The diagnostics offline/server-required state renders through the shared design-system Alert primitive.
- [x] #2 The diagnostics unsupported/playground-unavailable state renders through the shared design-system Alert primitive.
- [x] #3 The component keeps existing diagnostics copy, status tag, empty state, and child-card behavior.
- [x] #4 The `WritingPlaygroundDiagnosticsPanel` AntD Alert exceptions are removed from the product-state baseline.
- [x] #5 Focused tests and design-system guard verification are recorded in this task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
- [x] Add focused failing tests that render the offline and unsupported diagnostics states and assert each alert uses the shared design-system Alert marker.
- [x] Replace the guarded AntD Alert usages in `WritingPlaygroundDiagnosticsPanel` with the shared design-system Alert primitive while preserving copy and behavior.
- [x] Remove the migrated `WritingPlaygroundDiagnosticsPanel` Alert rows from the product-state baseline.
- [x] Run focused tests, product-state guard/unit verifier checks, baseline JSON parse, TypeScript check, and git diff whitespace checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `WritingPlaygroundDiagnosticsPanel.design-system-alert.test.tsx`; red state failed because the AntD Alerts did not provide the `data-ds-component="Alert"` marker.
- Replaced the diagnostics offline/unsupported AntD `Alert` usages with the shared design-system `Alert` using `variant="warning"` and `variant="info"`. Existing titles, body copy, status tag, empty state, and child-card rendering are unchanged.
- Removed both `WritingPlaygroundDiagnosticsPanel` Alert baseline rows. Baseline count is now 283 total exceptions, with Writing and Review surfaces at 13.
- Verification: `bunx vitest run src/components/Option/WritingPlayground/__tests__/WritingPlaygroundDiagnosticsPanel.design-system-alert.test.tsx src/components/Option/WritingPlayground/__tests__/WritingPlaygroundDiagnosticsPanel.design-system-state.test.tsx --reporter=dot` passed.
- Verification: `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed.
- Verification: `bun run verify:design-system-state` passed and reported 283 baseline exceptions / 13 Writing and Review exceptions.
- Verification: baseline JSON parse and absence check for `src/components/Option/WritingPlayground/WritingPlaygroundDiagnosticsPanel.tsx` passed.
- Verification: `git diff --check` passed.
- TypeScript: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` still exits 2 on existing repo-wide UI type debt; `/tmp/tldw_writing_diagnostics_tsc.log` contains no diagnostics for the touched component or new test.
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
