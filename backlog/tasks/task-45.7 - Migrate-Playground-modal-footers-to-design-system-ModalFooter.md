---
id: TASK-45.7
title: Migrate Playground modal footers to design-system ModalFooter
status: Done
assignee:
  - codex
created_date: '2026-05-05 18:16'
updated_date: '2026-05-05 19:52'
labels:
  - design-system
  - frontend
  - playground
dependencies: []
documentation:
  - Docs/Design/tldw_web_design_system_contract.md
  - Docs/Design/tldw_web_design_system_inventory.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next Chat/Playground design-system inventory slice after status chips. Scope is limited to tldw-owned Playground modal footer action rows in PlaygroundStartupTemplateModal, PlaygroundContextWindowModal, PlaygroundImageGenModal, PlaygroundRawRequestModal, PlaygroundMcpSettingsModal, and Common/Playground/DocumentGeneratorDrawer. Keep AntD modal mechanics, forms, tables, popovers, and broad Button migration unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Target Playground modal footer action rows render through the shared ModalFooter primitive or an existing tldw-owned wrapper around it without changing modal open/close behavior.
- [x] #2 Primary, secondary, destructive, loading, and disabled footer actions preserve their accessible names and callbacks.
- [x] #3 AntD modal mechanics and non-footer controls remain unchanged in this slice.
- [x] #4 Focused tests cover representative migrated footers and assert design-system markers plus preserved actions/states.
- [x] #5 Verification includes focused Vitest coverage, targeted lint/diff checks, and Bandit is skipped or documented as not applicable for frontend-only changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused Vitest coverage for representative Playground footer migrations before production edits: the shared ModalFooter marker/action behavior, Startup template modal delete/cancel/apply callbacks, Raw request modal refresh/extra/copy/close actions, Image generation modal loading/disabled footer states, MCP settings close footer, and the Document generator drawer action row.
2. Extend apps/packages/ui/src/components/ui/layout/ModalFooter.tsx conservatively so it exposes data-ds-component="ModalFooter" and can render ordered leftActions/right actions with the existing Common/Button while preserving existing primaryAction/secondaryAction/onCancel compatibility.
3. Migrate only the scoped Playground surfaces to ModalFooter: PlaygroundStartupTemplateModal, PlaygroundRawRequestModal, PlaygroundContextWindowModal, PlaygroundImageGenModal, PlaygroundMcpSettingsModal, and Common/Playground/DocumentGeneratorDrawer. Keep AntD Modal/Drawer mechanics, forms, tables, selects, and non-footer controls unchanged.
4. Run focused Vitest for the new/updated tests, then run targeted lint/diff checks. Document that Bandit is skipped because this is a frontend-only TypeScript/React slice.
5. Update TASK-45.7 with verification notes/final summary, mirror the Backlog task file into the clean worktree, commit, push, and open the PR against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red/green cycle complete: the new focused tests first failed because the scoped Playground footers had no ModalFooter markers, then passed after migrating the target action rows to ModalFooter.

Implemented ModalFooter ordered left/right action support plus data-ds-component="ModalFooter" while preserving existing primaryAction, secondaryAction, onCancel and leftContent compatibility.

Verification: bunx vitest run src/components/Option/Playground/__tests__/PlaygroundModalFooters.design-system.test.tsx src/components/Common/Playground/__tests__/DocumentGeneratorDrawer.design-system.test.tsx src/components/Option/Playground/__tests__/PlaygroundForm.reference-image.integration.test.tsx passed with 8 tests. ESLint targeted touched files exited 0 with no errors; remaining warnings are pre-existing no-explicit-any warnings in older Playground component signatures. git diff --check passed. Package tsc is blocked by unrelated existing errors across ui tests/source; filtered tsc output for touched filenames was empty. Bandit skipped because this is frontend-only TypeScript/React code.

Reopened for PR #1323 review feedback. Unresolved threads: standardize ModalFooter action aria label key to aria-label while adapting to the existing Common/Button ariaLabel prop, add variant plumbing to the Playground test Button mock, and make ImageGen submit/refine footer actions respect busy. Keep changes limited to the reviewed files and focused tests.

PR #1323 review feedback addressed: ModalFooterAction now uses the standard aria-label action key while adapting to Common/Button's ariaLabel prop, the Playground footer test Button mock carries variant through data-variant, and ImageGen refine/generate actions respect busy. Verification: focused Playground footer Vitest suite passed 3 files / 8 tests; tsc --noEmit passed; git diff --check passed; targeted ESLint exited 0 with only existing no-explicit-any warnings in PlaygroundImageGenModal.tsx.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the scoped Playground modal/drawer footer action rows to the shared ModalFooter primitive and addressed PR #1323 review feedback. ModalFooter now exposes a standard aria-label action key while adapting to the existing Common/Button ariaLabel prop, the Playground design-system test mock preserves variant/type attributes, and the ImageGen footer submit/refine actions now respect the modal busy lock. Focused tests cover footer markers, action ordering, callbacks, disabled/loading states, and busy-lock click prevention. Verification passed for the focused Vitest suite, frontend tsc --noEmit, git diff --check, and targeted ESLint with no errors; ESLint still reports pre-existing no-explicit-any warnings in PlaygroundImageGenModal.tsx. Bandit is not applicable for this frontend-only TypeScript/React slice.
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
