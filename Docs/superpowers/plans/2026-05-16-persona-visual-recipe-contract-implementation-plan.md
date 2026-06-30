## Stage 1: Baseline And Contract Tests
**Goal**: Ground the Slice 1 implementation in the existing Persona Visual generation job path and add failing tests for recipe-backed request/job behavior.
**Success Criteria**: Existing prompt-only generation tests pass before edits; new tests fail for missing recipe fields, missing validation, missing request/correlation IDs, missing recipe_intent payloads, and unchanged idempotency behavior.
**Tests**: Focused `test_persona_visual_jobs.py` and `test_persona_visuals_api.py` cases.
**Status**: Complete

## Stage 2: Recipe Intent Helpers
**Goal**: Add backend helpers that validate bundled starter recipe intent, compose bounded effective prompts, normalize correlation IDs, and build safe recipe_intent metadata.
**Success Criteria**: Valid starter/output pairs produce bounded prompt and recipe metadata; missing pairs, unknown starters, unknown outputs, and overlong composed prompts fail closed.
**Tests**: New helper coverage through job/API tests, with direct unit coverage if endpoint setup becomes too broad.
**Status**: Complete

## Stage 3: API And Job Wiring
**Goal**: Extend Persona Visual generation schemas and endpoint/job creation to carry request_id, starter_pack_id, recipe_output, and recipe_intent through the existing `persona_visual_generate_candidate` job type.
**Success Criteria**: Prompt-only behavior remains unchanged; recipe-backed requests queue the same job type with deterministic effective prompt, recipe_intent metadata, request/correlation IDs, updated idempotency, and trace-safe logs.
**Tests**: API tests for successful and failing recipe-backed requests plus job helper idempotency tests.
**Status**: Complete

## Stage 4: Verification And Packaging
**Goal**: Validate the focused backend slice and prepare a reviewable PR.
**Success Criteria**: Focused pytest passes; compile checks pass for touched production files; Bandit reports no new findings on touched Python scope; `git diff --check` passes; TASK-408 records verification and final status.
**Tests**: Focused Persona Visual pytest, `py_compile` or `compileall`, Bandit, and diff whitespace checks.
**Status**: Complete
