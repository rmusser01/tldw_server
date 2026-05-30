---
id: TASK-45.44.5.4
title: Migrate Evaluations RAG recipe config alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- product-state
- evaluations
priority: medium
parent_task_id: TASK-45.44.5
references:
- TASK-45.44.5
- https://github.com/rmusser01/tldw_server/issues/1662
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- https://github.com/rmusser01/tldw_server/pull/2138
- https://github.com/rmusser01/tldw_server/issues/1662#issuecomment-4581909874
modified_files:
- apps/packages/ui/src/components/Option/Evaluations/tabs/recipe-configs/RagAnswerQualityConfig.tsx
- apps/packages/ui/src/components/Option/Evaluations/tabs/recipe-configs/RagRetrievalTuningConfig.tsx
- apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/RagAnswerQualityConfig.test.tsx
- apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/RagRetrievalTuningConfig.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the remaining Evaluations product-state baseline debt by migrating the two RAG recipe configuration components from AntD Alert to the shared design-system Alert primitive. Keep the user-facing success, error, and info messages unchanged while adding tests that verify the messages are rendered by the design-system Alert container.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 RagAnswerQualityConfig AntD Alert product-state findings are migrated to the shared design-system Alert primitive without changing user-facing behavior.
- [x] #2 RagRetrievalTuningConfig AntD Alert product-state findings are migrated to the shared design-system Alert primitive without changing user-facing behavior.
- [x] #3 Matching Evaluations RAG recipe config baseline exceptions are removed and the remaining Evaluations count is documented.
- [x] #4 Focused tests pass and scoped product-area guard evidence confirms the migrated Evaluations RAG config findings do not reappear.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused tests around the RAG recipe config success, error, and info alert states that assert the rendered messages live inside the design-system Alert container.
2. Migrate RagAnswerQualityConfig and RagRetrievalTuningConfig from AntD Alert to the shared Alert primitive without changing copy or state conditions.
3. Remove the matching Evaluations RAG recipe config baseline exceptions and verify the Evaluations product-area count is zero.
4. Run focused Vitest coverage, TypeScript, design-system verifier, and diff hygiene; record any unrelated baseline verifier debt separately.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- TDD red pass: the focused tests initially failed after dependency install because the alert text existed but `closest('[data-ds-component="Alert"]')` was null for the remaining AntD Alert instances.
- Migrated `RagAnswerQualityConfig` success, error, and answer-anchor info alerts to `DsAlert` while preserving title/body copy and visibility conditions.
- Migrated `RagRetrievalTuningConfig` success, error, corpus-scope info, and approved-synthetic-query reminder alerts to `DsAlert` while preserving title/body copy and visibility conditions.
- Addressed PR review feedback by wrapping the approved-synthetic-query reminder title in `t("evaluations:syntheticQueriesReviewReminder", ...)` for i18n consistency.
- Removed the 7 matching Evaluations RAG recipe config exceptions from `design-system-product-state-baseline.json`; the Evaluations product-area count is now 0.
- Verification: `bun run test src/components/Option/Evaluations/tabs/__tests__/RagAnswerQualityConfig.test.tsx src/components/Option/Evaluations/tabs/__tests__/RagRetrievalTuningConfig.test.tsx --maxWorkers=1 --no-file-parallelism` -> 2 files passed, 14 tests passed. LocalStorage experimental warnings only.
- Verification: `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` -> exit 0 with no diagnostics.
- Verification: scoped product-state guard over `src/components/Option/Evaluations` using `runGuardOnSources` -> `No product-state guard issues found`.
- Full repo guard caveat: `bun run verify:design-system-state` remains blocked by unrelated global baseline debt outside Evaluations, so this PR records the scoped product-area result instead of claiming repo-wide baseline closure.
- Verification: `git diff --check` -> clean.
- Bandit skipped: touched implementation is TS/TSX plus JSON/Backlog metadata; no Python backend code changed in this slice.
- PR opened: https://github.com/rmusser01/tldw_server/pull/2138.
- GitHub issue #1662 updated with PR #2138 and zero-Evaluations-count evidence: https://github.com/rmusser01/tldw_server/issues/1662#issuecomment-4581909874.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the remaining Evaluations RAG recipe configuration product-state alerts from AntD Alert to the shared design-system Alert primitive. The change covers `RagAnswerQualityConfig` and `RagRetrievalTuningConfig`, preserving the existing success, error, and info copy while moving the state UI onto the shared primitive.

Added focused tests that assert the RAG recipe config alert messages render inside `[data-ds-component="Alert"]`, wrapped the remaining approved-synthetic-query reminder title in the existing i18n helper, and removed the 7 matching Evaluations baseline exceptions. A scoped product-state guard over `src/components/Option/Evaluations` now reports `No product-state guard issues found`.

Verification recorded: focused Vitest passed with 14 tests, TypeScript passed with no diagnostics, `git diff --check` was clean, and Bandit was skipped because this slice did not touch Python code. The full repo design-system verifier remains blocked by unrelated baseline debt outside Evaluations.
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
