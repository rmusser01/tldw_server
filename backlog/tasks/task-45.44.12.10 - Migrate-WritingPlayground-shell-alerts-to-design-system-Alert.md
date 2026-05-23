---
id: TASK-45.44.12.10
title: Migrate WritingPlayground shell alerts to design-system Alert
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
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- Docs/Design/tldw_web_design_system_contract.md
modified_files:
- apps/packages/ui/src/components/Option/WritingPlayground/index.tsx
- apps/packages/ui/src/components/Option/WritingPlayground/__tests__/WritingPlayground.shell-design-system-alert.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the WritingPlayground shell/session/editor product-state AntD Alert usages to the shared design-system Alert primitive. This narrows the remaining Writing/Review product-state baseline while leaving advanced settings alerts for a follow-up slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The offline shell state renders through the shared design-system Alert primitive.
- [x] #2 The unsupported shell state renders through the shared design-system Alert primitive.
- [x] #3 The sessions-load error state renders through the shared design-system Alert primitive.
- [x] #4 The active editor session-load error state renders through the shared design-system Alert primitive.
- [x] #5 The four migrated WritingPlayground shell/session/editor Alert exceptions are removed from the product-state baseline while advanced settings Alert exceptions remain for a follow-up slice.
- [x] #6 Focused tests and design-system guard verification are recorded in this task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
- [x] Add focused failing tests that render offline, unsupported, sessions-load error, and active-session editor error branches and assert each uses the shared design-system Alert marker.
- [x] Replace only those WritingPlayground shell/session/editor AntD Alert usages with the shared design-system Alert primitive while preserving copy and branch behavior.
- [x] Remove the four migrated WritingPlayground baseline rows from the product-state baseline, leaving the three advanced settings rows in place.
- [x] Run focused tests, product-state guard/unit verifier checks, baseline JSON parse, TypeScript check, and git diff whitespace checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added focused test coverage for the four targeted WritingPlayground shell/session/editor product states: offline shell, unsupported shell, sessions-load error, and active-session editor error.
- Confirmed the red check before implementation: the focused Vitest file failed all four design-system marker assertions while the branches still rendered AntD `Alert`.
- Migrated only the four targeted WritingPlayground shell/session/editor alerts to the shared design-system `Alert` primitive. The three advanced-settings AntD alerts remain as the follow-up slice.
- Removed the four migrated product-state baseline rows and refreshed the three remaining advanced-settings baseline IDs after the import shifted line numbers. Baseline now reports 268 total exceptions and 3 Writing/Review exceptions.
- Verification: `bunx vitest run src/components/Option/WritingPlayground/__tests__/WritingPlayground.shell-design-system-alert.test.tsx --reporter=dot` passed, with existing AntD Drawer deprecation warnings only.
- Verification: `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed.
- Verification: `bun run verify:design-system-state` passed, reporting `Baseline exceptions: 268` and `Writing and Review surfaces: 3`.
- Verification: baseline JSON parse/remaining check passed with exactly 3 `WritingPlayground/index.tsx` rows.
- Verification: `git diff --check` passed.
- TypeScript: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` exited 2 on existing repo-wide TypeScript debt; `/tmp/tldw_writing_shell_tsc.log` has 314 lines and `rg` found no diagnostics for the touched files.
- Bandit skipped: this slice touches UI TypeScript/TSX, product-state baseline JSON, and Backlog metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the four targeted `WritingPlayground` shell/session/editor product-state alerts to the shared design-system `Alert` primitive and added focused coverage for each branch. Removed the four migrated product-state baseline exceptions while preserving the three advanced-settings exceptions for a follow-up slice. PR: https://github.com/rmusser01/tldw_server/pull/1979
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
