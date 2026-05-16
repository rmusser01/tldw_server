---
id: TASK-408
title: Implement Persona Visual recipe-backed generation contract
status: Done
labels:
- persona
- persona-visual
- backend
- issue-1774
modified_files:
- tldw_Server_API/app/api/v1/endpoints/persona.py
- tldw_Server_API/app/api/v1/schemas/persona.py
- tldw_Server_API/app/core/Persona/visual_generation_recipes.py
- tldw_Server_API/app/core/Persona/visual_jobs.py
- tldw_Server_API/app/core/Persona/visual_jobs_worker.py
- tldw_Server_API/tests/Persona/test_persona_visual_jobs.py
- tldw_Server_API/tests/Persona/test_persona_visuals_api.py
- Docs/superpowers/plans/2026-05-16-persona-visual-recipe-contract-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1774 as a persona-only backend/API slice. Extend Persona Visual generation requests with optional recipe intent fields and request/correlation IDs, validate starter_pack_id plus recipe_output against bundled starter production_recipe.animation_outputs, compose bounded effective prompts, add recipe_intent metadata to the existing persona_visual_generate_candidate Jobs payload, keep prompt-only behavior unchanged, update idempotency, add trace-safe logging, and cover valid/invalid recipe-backed requests with focused backend tests. Non-goals: no Buddy animation runtime, no WebUI/extension changes, no final art generation, no automatic activation, no renderer expansion, no MCP provider execution/download, no marketplace/shared-library behavior, no VN/CYOA behavior, and no DB migration unless unavoidable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Generation request/response schemas support optional recipe intent and request/correlation IDs while preserving prompt-only behavior.
- [x] #2 Recipe-backed requests validate starter_pack_id and recipe_output against bundled starter production_recipe.animation_outputs and fail closed for missing pairs, unknown starters, or unknown outputs.
- [x] #3 Recipe-backed jobs use the existing persona_visual_generate_candidate job type and include bounded recipe_intent metadata plus normalized correlation identifiers.
- [x] #4 Idempotency distinguishes prompt-only and recipe-backed requests, including starter/output differences.
- [x] #5 Focused backend tests cover valid recipe-backed jobs, paired-field validation, unknown starter/output, overlong composed prompt, correlation IDs, and unchanged prompt-only behavior.
- [x] #6 Verification records focused pytest, compile/diff checks, and Bandit touched-scope results.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-16-persona-visual-recipe-contract-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `visual_generation_recipes.py` for trace-safe request ID normalization, starter/output validation against bundled Persona Visual starter fixtures, and bounded effective prompt composition.
- Extended `PersonaVisualGenerationRequest` / `PersonaVisualGenerationJobResponse` with request/recipe fields, while prompt-only requests continue to queue the existing job type and now receive a generated request ID.
- Extended `persona_visual_generate_candidate` job payloads/idempotency with optional `request_id` and `recipe_intent` metadata.
- Added structured log events for recipe validation and job creation without logging raw generated assets, credentials, or unbounded prompt data.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_jobs.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q` -> 64 passed, 5 warnings.
- Verification: `py_compile` for touched Python files -> passed.
- Verification: `git diff --check` -> passed.
- Verification: Bandit touched Python scope -> no findings in `/tmp/bandit_persona_visual_recipe_contract_1774.json`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Persona Visual recipe-backed generation contract Slice 1 for issue #1774. Generation requests now support request IDs plus optional starter_pack_id/recipe_output pairs, validate recipe intent against bundled starter production recipes, compose bounded effective prompts, queue the existing persona_visual_generate_candidate job with recipe_intent metadata, include correlation IDs in job payloads/responses, distinguish recipe-backed idempotency, and reject unsafe request IDs before queueing. Added trace-safe logging for recipe request validation, job creation, and candidate creation. Focused backend tests cover prompt-only behavior, valid recipe-backed requests, missing recipe pairs, unknown starter/output, overlong composed prompts, unsafe request IDs, correlation payloads, and idempotency.
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
