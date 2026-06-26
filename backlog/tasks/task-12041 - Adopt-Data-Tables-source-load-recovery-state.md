---
id: TASK-12041
title: Adopt Data Tables source-load recovery state
status: Done
created_date: 2026-06-26 03:26
references:
- TASK-420
- TASK-418.11
- TASK-12040
documentation:
- Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
modified_files:
- Docs/superpowers/plans/2026-06-26-webui-stage12-data-tables-source-recovery-plan.md
- apps/packages/ui/src/components/Option/DataTables/SourceSelector.tsx
- apps/packages/ui/src/components/Option/DataTables/__tests__/SourceSelector.recovery.test.tsx
updated_date: 2026-06-26 03:29
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the deferred WebUI capability/error-state follow-up for the Data Tables source picker. Replace the toast-only failed chat/document source load state with an inline shared recovery state while preserving the existing source type tabs, search input, selected-source tags, RAG query flow, and retry behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Data Tables source picker shows one shared unavailable recovery state when chat/document source loading fails.
- [x] #2 The raw source-loading error is kept in diagnostics instead of being the primary source picker copy.
- [x] #3 The recovery state offers a retry action wired to the existing source query refetch.
- [x] #4 Existing source type selector, selected-source tags, and RAG query flow remain unchanged.
- [x] #5 Focused component tests cover the failed source-load recovery state.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused SourceSelector regression test for failed source loading and run it red.
2. Render a shared StatePanel unavailable recovery state for non-RAG source query errors, with diagnostics and retry.
3. Re-run the focused SourceSelector test, touched-file lint, and diff whitespace checks.
4. Record verification and final summary on this task before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented with a test-first pass:
- Added a focused `SourceSelector.recovery.test.tsx` regression for failed chat source loading. The first useful red run failed because `data-tables-source-load-recovery` was absent and the picker fell back to the existing empty-list behavior.
- Added a shared `StatePanel` unavailable state for non-RAG source query failures. The panel keeps selected-source tags and search context visible, moves raw error details into diagnostics, and wires the primary retry action to `sourcesQuery.refetch()`.
- Kept the existing transient toast/log behavior for source-load failures so users still get immediate feedback, while adding stable inline recovery for the route.

Verification:
- `bun run test:run ../packages/ui/src/components/Option/DataTables/__tests__/SourceSelector.recovery.test.tsx` from `apps/tldw-frontend`: PASS, 1 test.
- `bun apps/node_modules/.bun/eslint@9.39.2+288993669ddeca06/node_modules/eslint/bin/eslint.js -c apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Option/DataTables/SourceSelector.tsx apps/packages/ui/src/components/Option/DataTables/__tests__/SourceSelector.recovery.test.tsx`: PASS with no lint errors; only the known Next pages-directory notice.
- `git diff --check`: PASS.
- `bun scripts/verify-design-system-product-state.mjs` from `apps/packages/ui`: unable to start because the local shared-UI install cannot resolve the `typescript` package from the guard script; recorded as an environment/dependency-layout skip.
- Bandit: not applicable; this slice touches TS/TSX/docs/Backlog only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The Data Tables source picker now shows a shared unavailable StatePanel when chat or document source loading fails. The state preserves selected sources and search context, keeps raw error details in diagnostics, and provides a retry action backed by the existing source query refetch. A focused SourceSelector regression test covers the recovery state and retry behavior.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused Data Tables source selector test passes
- [x] #8 Touched-file lint check run or documented
- [x] #9 git diff whitespace check run
- [x] #10 Bandit run for touched code when applicable or documented as not applicable
<!-- DOD:END -->
