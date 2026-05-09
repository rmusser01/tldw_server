---
id: TASK-145.5
title: Implement guided embeddings recipe config component
status: Done
assignee:
  - Codex
created_date: '2026-05-09 05:26'
updated_date: '2026-05-09 05:38'
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
Implement Task 4 from the embeddings RAG recipe implementation plan: create the dedicated guided embeddings_model_selection recipe configuration component with query/source/model controls and focused component tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Component serializes query rows and expected media IDs into the recipe dataset shape.
- [x] #2 Candidate readiness UI lists ready/disallowed candidates and only uses ready candidates for run config selection/prefill.
- [x] #3 Media search source selection stores integer media IDs only in expected_ids.
- [x] #4 Component keeps run config media_ids/candidates/top_k/hybrid_alpha in the expected backend shape.
- [x] #5 Focused component tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing component tests for guided query serialization, candidate readiness selection, and media search source selection.
2. Create EmbeddingsModelSelectionConfig.tsx with compact Corpus, Queries, Expected sources, Models, and Run review controls.
3. Use media IDs only for guided expected sources, keep advanced media ID entry, and defensively normalize media search results.
4. Use useEmbeddingRecipeCandidates for model hints and only auto-prefill ready candidates without overwriting user edits.
5. Run focused component Vitest, diff hygiene, and commit the component slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED: bunx vitest run src/components/Option/Evaluations/tabs/__tests__/EmbeddingsModelSelectionConfig.test.tsx failed before implementation with unresolved ../recipe-configs/EmbeddingsModelSelectionConfig import.

GREEN: bunx vitest run src/components/Option/Evaluations/tabs/__tests__/EmbeddingsModelSelectionConfig.test.tsx passed 3 tests after implementation.

Verification: git diff --check exited 0. Bandit skipped because this task only touched frontend TypeScript/TSX and Backlog task metadata.

Review fix RED: bunx vitest run src/components/Option/Evaluations/tabs/__tests__/EmbeddingsModelSelectionConfig.test.tsx failed with the new source_id_contract regression because onRunConfigChange emitted the manifest object instead of "media_id".

Review fix GREEN: bunx vitest run src/components/Option/Evaluations/tabs/__tests__/EmbeddingsModelSelectionConfig.test.tsx passed 4 tests after normalizing emitted source_id_contract to "media_id". git diff --check exited 0.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the dedicated guided embeddings model selection recipe config component and focused tests. The component serializes inline query/source labels as media ID expected_ids, searches media defensively through tldwClient.searchMedia, preserves backend-friendly run_config fields, and uses useEmbeddingRecipeCandidates so only ready embedding candidates can be selected or prefilling candidates when empty.

Follow-up review fix: source_id_contract is now serialized as the backend string "media_id" while the UI can still display manifest contract tags.
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
