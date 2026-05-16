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
- tldw_Server_API/app/core/Jobs/manager.py
- tldw_Server_API/app/core/Persona/visual_generation_recipes.py
- tldw_Server_API/app/core/Persona/visual_jobs.py
- tldw_Server_API/app/core/Persona/visual_jobs_worker.py
- tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_sqlite.py
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
Review follow-up for PR #1778 completed. Fixed still-valid review findings by making recipe idempotency ignore request/correlation IDs, returning persisted request IDs on idempotent replays, forwarding request IDs into Jobs, preserving request/trace IDs in SQLite and Postgres idempotent return paths, replacing string-matched recipe errors with typed error codes, and simplifying request ID validation. Added a final regression that falls back to a stable job identifier before a fresh fallback request ID when persisted job correlation is absent. The worker long-line comment was already addressed by the prior request-ID hardening commit. Verification: focused pytest -> 72 passed, 5 warnings; py_compile touched Python files -> passed; git diff --check -> passed; Bandit touched implementation scope -> 0 findings in /tmp/bandit_persona_visual_recipe_contract_1778_review.json.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented PR #1778 review follow-up for the Persona Visual recipe-backed generation contract. The API/job path now keeps idempotent replays aligned with the persisted job request ID, avoids treating trace-only identifiers as generation intent, persists request IDs through the Jobs layer, uses typed recipe validation errors, and retains focused regression coverage for prompt-only and recipe-backed replay behavior.
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
