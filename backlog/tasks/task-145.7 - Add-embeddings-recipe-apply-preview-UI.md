---
id: TASK-145.7
title: Add embeddings recipe apply preview UI
status: Done
assignee: []
created_date: '2026-05-09 06:01'
updated_date: '2026-05-09 06:11'
labels:
  - evals
  - frontend
dependencies: []
references:
  - >-
    Docs/superpowers/plans/2026-05-09-embeddings-rag-recipe-webui-implementation-plan.md
parent_task_id: TASK-145
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 6 from the embeddings RAG recipe implementation plan: add a server-backed apply-preview modal for eligible embeddings recommendations and show copy-config fallback when live apply is unavailable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Eligible embeddings recommendation buttons call the apply-preview hook with runId, slotName, and candidateRunId and render the server-normalized preview.
- [x] #2 Preview modal shows current/proposed embedding provider and model, affected config keys, run id, warnings, reindex requirement, and copy-config fallback when apply_available=false.
- [x] #3 No live apply mutation is implemented; if apply_available=true the UI only shows a disabled placeholder until Task 7.
- [x] #4 Focused RecipesTab tests pass with apply-preview success and fallback coverage.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing RecipesTab tests for apply-preview modal and copy-config fallback. 2. Run focused launch test to confirm the missing modal/action fails. 3. Wire usePreviewRecipeRecommendationApply into RecipesTab eligible recommendation buttons. 4. Render preview modal from server response without exposing secrets and without live apply mutation. 5. Re-run focused tests, diff check, review, and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED: `bunx vitest run src/components/Option/Evaluations/tabs/__tests__/RecipesTab.launch.test.tsx` failed after adding Task 6 tests because the preview hook was not called and the apply-preview modal content was absent. GREEN: focused launch suite passed with 24 tests. Required combined verification passed with 2 files and 25 tests. `git diff --check` passed. Bandit skipped: frontend-only TypeScript/TSX tests and Backlog metadata, no Python touched. Known blockers: none.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the embeddings recommendation apply-preview UI for Task 6. Eligible recommendation actions now call `usePreviewRecipeRecommendationApply` with runId, slotName, and candidateRunId, then render a server-driven modal with current/proposed embedding model details, affected config keys, run id, warnings, reindex status, and copy_config JSON. `apply_available=false` shows the Copy config change fallback; `apply_available=true` shows only a disabled Apply config change placeholder for Task 7.
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
