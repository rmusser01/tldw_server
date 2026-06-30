---
id: TASK-145.4
title: Implement embeddings recipe frontend services and hooks
status: Done
assignee:
  - Codex
created_date: '2026-05-09 05:13'
labels:
  - evaluations
  - embeddings
  - rag
  - frontend
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-09-embeddings-rag-recipe-webui-implementation-plan.md
parent_task_id: TASK-145
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 3 from the embeddings RAG recipe implementation plan: add typed frontend service calls, React Query hooks, hook tests, and OpenAPI path guard entries for the embeddings_model_selection candidate and apply-preview helper APIs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 useRecipes tests cover stable candidate query loading and apply-preview mutation payloads.
- [x] #2 evaluations.ts exports types and functions for getEmbeddingRecipeCandidates and previewRecipeRecommendationApply.
- [x] #3 useRecipes.ts exports React Query hooks using stable keys and expected payload mapping.
- [x] #4 openapi-guard ClientPath includes the new candidate and apply-preview paths.
- [x] #5 Focused frontend hook tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing hook tests in useRecipes.test.tsx for loading embedding recipe candidates and posting apply-preview requests.
2. Add typed service response/request shapes and service functions in evaluations.ts.
3. Add useEmbeddingRecipeCandidates and usePreviewRecipeRecommendationApply hooks in useRecipes.ts.
4. Add the new API paths to the OpenAPI guard union.
5. Run the focused Vitest hook test and commit the frontend slice.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
- Red verification: `cd apps/packages/ui && bunx vitest run src/components/Option/Evaluations/hooks/__tests__/useRecipes.test.tsx` failed with the expected missing hook exports: `useEmbeddingRecipeCandidates is not a function` and `usePreviewRecipeRecommendationApply is not a function`.
- Green verification: `cd apps/packages/ui && bunx vitest run src/components/Option/Evaluations/hooks/__tests__/useRecipes.test.tsx` passed with 6 tests.
- Hygiene verification: `cd apps/packages/ui && bun run verify:openapi` passed; `git diff --check` passed.
- Bandit skipped: frontend-only TypeScript/service/hook changes, no Python touched in this task.
- Known skips/blockers: no package typecheck script exists in `apps/packages/ui/package.json`; focused Vitest and OpenAPI guard verification were run instead.
- Follow-up red verification: `cd apps/packages/ui && bunx vitest run src/components/Option/Evaluations/hooks/__tests__/useRecipes.test.tsx` failed with candidate `{ ok: false }` not reaching `isError` and apply-preview `{ ok: false }` resolving instead of rejecting.
- Follow-up green verification: same focused Vitest command passed with 8 tests after wrapping the embeddings candidate query and apply-preview mutation in `ensureOk`.
- Follow-up hygiene verification: `cd apps/packages/ui && bun run verify:openapi` passed; `git diff --check` passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added typed frontend service calls and React Query hooks for the embeddings recipe candidate discovery and recommendation apply-preview APIs. Covered the hooks with focused tests for stable candidate query loading, null-normalized apply-preview payload mapping, and `ensureOk` error propagation for failed candidate and apply-preview responses. Added the new client paths to the OpenAPI guard.
<!-- SECTION:FINAL_SUMMARY:END -->
