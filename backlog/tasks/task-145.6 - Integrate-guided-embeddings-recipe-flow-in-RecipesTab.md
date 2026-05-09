---
id: TASK-145.6
title: Integrate guided embeddings recipe flow in RecipesTab
status: Done
assignee: []
created_date: '2026-05-09 05:42'
updated_date: '2026-05-09 05:50'
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
Implement Task 5 from the embeddings RAG recipe implementation plan: wire the guided embeddings_model_selection config component into RecipesTab and render recommendation-first embeddings report cards while preserving generic recipe and raw JSON behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Embeddings launch flow uses the guided config component and keeps the create payload dataset/runConfig shape compatible with the backend.
- [x] #2 Embeddings report results show recommendation-first cards for best overall/local/cheap slots with model, metrics, confidence/warnings, and raw report details still available.
- [x] #3 Preview action is shown only when server metadata marks a recommendation apply_eligible; blocked recommendations show the server reason without offering apply.
- [x] #4 Focused RecipesTab and EvaluationsPage recipe tab tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Update RecipesTab launch tests for the guided component labels and recommendation-first report expectations. 2. Run focused tests to confirm the existing flow fails. 3. Integrate EmbeddingsModelSelectionConfig for embeddings_model_selection while preserving raw JSON advanced editing and generic recipes. 4. Add recommendation card rendering for embeddings reports. 5. Re-run focused tests, diff check, and review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED: bunx vitest run src/components/Option/Evaluations/tabs/__tests__/RecipesTab.launch.test.tsx failed after Task 5 test updates. Failures showed RecipesTab still lacked Expected media IDs 1 from the guided embeddings component and recommendation-first embeddings report cards.

GREEN: bunx vitest run src/components/Option/Evaluations/tabs/__tests__/RecipesTab.launch.test.tsx src/components/Option/Evaluations/__tests__/EvaluationsPage.recipe-tab.test.tsx passed 22 tests. Component label touch verified with bunx vitest run src/components/Option/Evaluations/tabs/__tests__/EmbeddingsModelSelectionConfig.test.tsx passing 4 tests.

Verification: git diff --check exited 0. Bandit skipped because this task only touched frontend TypeScript/TSX tests and Backlog task metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Integrated EmbeddingsModelSelectionConfig into RecipesTab for embeddings_model_selection inline and saved modes, keeping Advanced JSON available. Added embeddings recommendation-first report cards for best overall/local/cheap slots with model/provider, Recall metrics, confidence warnings, eligible preview button gating, blocked apply reasons, and raw report details. Updated launch tests to cover guided payload shape and report rendering.
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
