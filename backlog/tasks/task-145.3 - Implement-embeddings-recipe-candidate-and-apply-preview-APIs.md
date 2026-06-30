---
id: TASK-145.3
title: Implement embeddings recipe candidate and apply preview APIs
status: Done
assignee:
  - Codex
created_date: '2026-05-09 04:42'
updated_date: '2026-05-09 05:09'
labels:
  - evaluations
  - embeddings
  - rag
  - backend
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-09-embeddings-rag-recipe-webui-implementation-plan.md
parent_task_id: TASK-145
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 2 from the embeddings RAG recipe implementation plan: backend helper functions, Pydantic schemas, candidate discovery endpoint, and recommendation apply-preview endpoint for the embeddings_model_selection recipe. Use TDD, keep payloads secret-free, do not mutate live config, and keep live apply unavailable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Candidate hints return current/default models with ready, missing_key, disallowed_provider, and disallowed_model status classification using existing policy helpers.
- [x] #2 Apply preview resolves completed embeddings recommendation slots into secret-free proposed config, copy_config, warnings, and apply_available=false.
- [x] #3 Recipe helper API schemas forbid unexpected fields and expose the Task 2 response/request contracts.
- [x] #4 Integration tests cover candidates endpoint, apply-preview success, and rejection for non-embeddings recipe runs.
- [x] #5 Focused backend tests, Bandit on touched production code, and git diff hygiene checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused failing helper tests for candidate readiness, policy status, missing-key handling, and apply-preview copy config.
2. Add recipe hint/apply-preview schemas matching the Task 2 contract.
3. Implement embeddings_recipe_hints.py using existing simplified config and embeddings policy helper wrappers, with dict and RecipeRunReport normalization.
4. Add API tests for candidates, apply-preview success, and non-embeddings recipe rejection using existing integration fixtures.
5. Wire endpoints in evaluations_recipes.py without calling endpoint functions from helpers.
6. Run focused pytest, Bandit on touched production files, and diff hygiene checks before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Task 2 backend candidate hints and apply-preview support. Red evidence: helper pytest initially failed during collection because embeddings_recipe_hints.py did not exist; endpoint tests then failed on missing candidates/apply-preview route wiring. Green evidence: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Evaluations/test_recipe_embeddings_hints.py tldw_Server_API/tests/Evaluations/integration/test_recipe_runs_api.py -q -> 29 passed, 5 warnings. Bandit: python -m bandit -r touched production files -f json -o /tmp/bandit_embeddings_recipe_task2.json -> 0 results, 0 errors. Hygiene: git diff --check -> clean.

Follow-up policy semantics fix: added red tests proving non-enforced allowlists do not block candidate hints and proving model allowlist patterns only support exact or trailing-* prefix matching. Updated _classify_candidate to gate provider/model allowlist blocks on should_enforce_embedding_policy(user), and aligned _model_allowed with embeddings_abtest_service exact/trailing-star semantics. Verification: python -m pytest tldw_Server_API/tests/Evaluations/test_recipe_embeddings_hints.py tldw_Server_API/tests/Evaluations/integration/test_recipe_runs_api.py -q -> 31 passed, 5 warnings. Bandit on embeddings_recipe_hints.py -> 0 results, 0 errors. git diff --check -> clean.

Follow-up hardening fix: added red tests for apply preview slots missing candidate_run_id and for current/default embedding candidates missing from configured provider model lists. The red run failed with candidates empty for drifted config and apply_eligible=true for missing candidate_run_id. Fixed apply preview to block slots without candidate_run_id and fixed candidate dedupe to insert current/default first when no configured candidate matches. Verification: python -m pytest tldw_Server_API/tests/Evaluations/test_recipe_embeddings_hints.py tldw_Server_API/tests/Evaluations/integration/test_recipe_runs_api.py -q -> 33 passed, 5 warnings. Bandit on embeddings_recipe_hints.py -> 0 results, 0 errors. git diff --check -> clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added embeddings recipe candidate hints and apply-preview backend APIs. The helper normalizes simplified embeddings config without exposing secrets, classifies candidate readiness against allowlists and key availability, supports dict/Pydantic recipe reports, and returns copy-config previews with live apply unavailable. Added Pydantic API contracts plus GET /recipes/embeddings_model_selection/candidates and POST /recipe-runs/{run_id}/apply-preview. Added focused helper tests and recipe API integration coverage for candidate discovery, copy-config preview, and non-embeddings rejection.
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
