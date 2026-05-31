---
id: TASK-145.8
title: Evaluate gated live apply for embeddings recipe recommendations
status: Done
assignee: []
created_date: '2026-05-09 06:15'
updated_date: '2026-05-09 16:32'
labels:
  - evals
  - backend
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
Follow-up for Task 7: only implement live config mutation for embeddings recipe recommendations after explicit approval. Scope includes POST apply endpoint, config override safety checks, audit metadata, frontend apply hook/button, and focused tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation is started only after explicit approval for live RAG/embeddings config mutation.
- [x] #2 Endpoint writes only [Embeddings] embedding_provider and embedding_model through setup_manager with backup/audit metadata.
- [x] #3 Mutation refuses to apply when environment variables override the effective Embeddings provider/model values.
- [x] #4 Frontend shows live apply only when server preview returns apply_available=true and confirmation matches proposed provider/model.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Keep this as a gated follow-up. Start with backend failing tests for the apply endpoint and env override refusal, then add service/hook/UI only after backend contract is stable.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
User replied continue after the preview/copy-config V1 completion summary, which I treated as explicit approval to proceed with the gated live RAG/embeddings config mutation slice. Implemented POST /api/v1/evaluations/recipe-runs/{run_id}/apply behind EVALS_MANAGE. Live apply calls preview first, requires confirmed provider/model to match the server preview, writes only [Embeddings] embedding_provider and embedding_model through setup_manager.update_config(create_backup=True), refuses env overrides from EMBEDDINGS_DEFAULT_PROVIDER, EMBEDDINGS_PROVIDER, EMBEDDINGS_DEFAULT_MODEL, and EMBEDDINGS_MODEL, and records embedding_recipe_apply_audit metadata on the recipe run. Audit metadata is persisted before config mutation as pending, finalized as applied on success, and marked failed on config errors so config mutation does not happen without a durable audit trail. Preview apply availability is now permission-aware: read-only eval users can still preview/copy config but see apply_available=false and the apply endpoint remains 403. Frontend service/hook and modal action show Apply to RAG config only when apply_available=true and post the confirmation payload.

Verification: backend focused pytest passed 39 tests with 5 warnings; frontend focused Vitest passed 33 tests; bun run verify:openapi passed with 259 ClientPath entries and existing 10 reviewed exceptions; Bandit on touched backend source wrote /tmp/bandit_embeddings_live_apply.json with results 0/errors 0/skipped 0; git diff --check passed after task-note refresh. Known note: RecipesTab launch test suite is slow, so the test file sets a 60s per-test timeout.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the gated live apply path for embeddings recipe recommendations. The backend adds a manage-permission apply endpoint that reuses preview validation, checks confirmed provider/model, refuses environment overrides, writes only [Embeddings] embedding_provider and embedding_model with setup-manager backup support, and stores embedding_recipe_apply_audit metadata before and after config mutation. The preview endpoint remains readable but only advertises apply availability to users with EVALS_MANAGE. The WebUI adds applyRecipeRecommendation/useApplyRecipeRecommendation and upgrades the apply-preview modal so eligible previews can apply to RAG config and show backup/audit feedback while copy fallback remains for unavailable live apply.
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
