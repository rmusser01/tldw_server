# Research Workspace Artifact Claim Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add backend-internal Claims-module verification to Research Workspace artifact generation so quizzes, flashcards, audio summaries, data tables, slides, and mindmaps are real assets and factually good enough against the exact selected sources.

**Architecture:** Add one internal Claims artifact-verification helper, call it inside backend generation before success/persistence, and move any Research Workspace frontend-only `createChatCompletion` artifact paths behind backend orchestration before enabling the gate.

**Tech Stack:** FastAPI, Pydantic, existing ClaimsEngine/VerificationReport, existing LLM adapter registry, SQLite/PostgreSQL Media DB helpers, Next.js/React UI package, Vitest/Playwright, pytest/Hypothesis.

---

Task: TASK-12146
Spec: `Docs/superpowers/specs/2026-07-04-research-workspace-artifact-claim-verification-design.md`

## Review Findings Addressed

- [x] Do not call public `/api/v1/claims/*` endpoints from Research Workspace artifact generation.
- [x] Verify inside backend generation, before an artifact is marked successful.
- [x] Move frontend-only Research Workspace artifact generators backend-side before adding claim verification.
- [x] Preserve generation provider/model by default when extracting and verifying claims, including local llama.cpp workflows.
- [x] Allow a separate user-configured claims-verification provider/model and make non-default verifier use visible to the user.
- [x] Keep FVA out of the default gate; use document-only ClaimsEngine verification against selected source excerpts.
- [x] Preserve per-artifact unit IDs through claim extraction, verification, and report display.
- [x] Use existing Claims statuses/reporting instead of inventing a parallel frontend fact-check vocabulary.
- [x] Store verification reports through each artifact family's existing persistence path, adding only narrow storage where no suitable metadata field exists.

## File Structure

New backend files:

- `tldw_Server_API/app/core/Claims_Extraction/artifact_verification.py`
- `tldw_Server_API/tests/Claims/test_artifact_verification.py`
- `tldw_Server_API/tests/Claims/test_artifact_verification_properties.py`

Likely backend edits:

- `tldw_Server_API/app/core/Claims_Extraction/claims_engine.py`
- `tldw_Server_API/app/api/v1/schemas/claims_schemas.py`
- `tldw_Server_API/app/api/v1/endpoints/flashcards.py`
- `tldw_Server_API/app/api/v1/schemas/flashcards.py`
- `tldw_Server_API/app/services/quiz_generator.py`
- `tldw_Server_API/app/api/v1/schemas/quizzes.py`
- `tldw_Server_API/app/api/v1/endpoints/slides.py`
- `tldw_Server_API/app/core/Data_Tables/jobs_worker.py`
- `tldw_Server_API/app/api/v1/endpoints/data_tables.py`
- `tldw_Server_API/app/api/v1/schemas/data_tables_schemas.py`
- `tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py`
- `tldw_Server_API/app/core/DB_Management/media_db/schema/postgres_data_table_structures.py`
- `tldw_Server_API/app/core/DB_Management/media_db/runtime/data_table_metadata_ops.py`
- `tldw_Server_API/app/core/DB_Management/media_db/runtime/data_table_generation_ops.py`

New backend orchestration for frontend-only artifact types:

- `tldw_Server_API/app/core/Research_Workspace/artifact_generation.py`
- `tldw_Server_API/app/api/v1/endpoints/research_workspace.py`
- `tldw_Server_API/app/api/v1/schemas/research_workspace_artifacts.py`
- `tldw_Server_API/tests/ResearchWorkspace/test_artifact_generation_service.py`

Likely frontend edits:

- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/TraceableArtifactDetail.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/ArtifactModalContent.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useStudioDerivedState.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage1.test.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx`
- `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/__tests__/TraceableArtifactDetail.test.tsx`
- `apps/packages/ui/src/services/flashcards.ts`
- `apps/packages/ui/src/types/workspace.ts`

## Stage 1: Internal Claims Artifact Verification Helper

**Goal:** Build the reusable backend helper without touching public Claims HTTP endpoints.

**Success Criteria:** Artifact units can be verified against explicit `Document` inputs with deterministic verdicts and preserved unit IDs.

**Tests:** `tldw_Server_API/tests/Claims/test_artifact_verification.py`, `tldw_Server_API/tests/Claims/test_artifact_verification_properties.py`

**Status:** Complete

- [x] Create `ArtifactVerificationUnit`, `ArtifactUnitResult`, and `ArtifactVerificationResult` data models in `artifact_verification.py`.
- [x] Implement `verify_generated_artifact_against_sources(...)` with document-only verification against supplied `Document` objects.
- [x] Reuse `ClaimsEngine.extract_claims_only(...)`, `ClaimsEngine.verify_claims_only(..., doc_only_mode=True)`, and `generate_verification_report(...)`.
- [x] Add helper-level caps for unit count, claim count, and text length by artifact type.
- [x] Map existing `VerificationStatus` values to only `grounded`, `needs_revision`, or `failed` artifact verdicts.
- [x] Return non-grounded results when verification is missing or capped.
- [x] Keep FVA out of this helper.
- [x] Unit test failed statuses: `refuted`, `hallucination`, `numerical_error`, `misquoted`, and strong `citation_not_found`.
- [x] Unit test review statuses: `unverified`, `contested`, weak evidence, and budget/cap hits.
- [x] Property test that unit IDs round-trip through claim extraction and result mapping.
- [x] Test that capped/truncated verification cannot produce `grounded`.

## Stage 2: Provider/Model Propagation

**Goal:** Ensure the verification helper uses the same generation provider/model context rather than silently falling back to global Claims settings.

**Success Criteria:** Tests can bind a fake llama.cpp-style provider/model into extraction and verification calls and prove no global Claims provider is used.

**Tests:** Extend `test_artifact_verification.py`; add focused regression coverage for generation-bound provider/model metadata.

**Status:** Complete

- [x] Add a narrow generation-bound analyze callable used only by artifact verification.
- [x] Pass `generation_provider` and `generation_model` through extraction and verification by default.
- [x] Add optional `verification_provider` and `verification_model` inputs for user-configured claims-verification LLMs.
- [x] Use configured verification provider/model for ClaimsEngine extraction and verification when present.
- [x] Record `generation_provider`, `generation_model`, `verification_provider`, `verification_model`, and whether the verifier differs from generation in result metadata.
- [x] Treat missing verifier configuration as generation-model default; provider/model resolution errors from the internal call fail the generation path instead of silently using global Claims settings.
- [x] Add tests proving no public Claims endpoint is called and no broader retrieval happens.
- [x] Add tests proving local-provider settings such as llama.cpp are propagated into both claim extraction and verification calls.
- [x] Add tests proving explicit claims-verification provider/model overrides are used and marked as non-default in metadata.

## Stage 3: Backend-Generated Artifact Gates

**Goal:** Wire the helper into artifact types that already have backend generation paths, with verify-before-success behavior.

**Success Criteria:** Slides, flashcard drafts, and quiz generation verify generated claims before returning/persisting a clean success.

**Tests:** Existing backend endpoint/service tests plus new focused integration tests for each artifact type.

**Status:** Complete

- [x] Add response fields for `claim_verification`/`claimVerification` that wrap `ArtifactVerificationResult` without changing Claims module vocabulary.
- [x] Flashcards: build units from generated front/back text in `/api/v1/flashcards/generate`.
- [x] Flashcards: return drafts only when verdict is `grounded`; return 422 with report details when verdict is `needs_revision` or `failed`.
- [x] Quiz: build units from each generated question, answer, options, and explanation in `quiz_generator.py`.
- [x] Quiz: run verification before `_persist_generated_quiz(...)`.
- [x] Quiz: fail generation before persistence on any non-grounded verdict.
- [x] Slides: build units from normalized titles, bullets, and speaker notes after `_normalize_slides(...)`.
- [x] Slides: run verification before `db.create_presentation(...)`.
- [x] Slides: store grounded reports in `studio_data.claimVerification`.
- [x] Slides: return 422 before creating a completed presentation when verdict is non-grounded.
- [x] Add tests proving non-grounded slide verification does not create a presentation row.
- [x] Add tests proving non-grounded quiz verification does not persist a quiz.
- [x] Add tests proving non-grounded flashcard verification does not return clean draft cards.

## Stage 4: Data Tables Persistence Gate

**Goal:** Verify generated table rows before a data table job is marked ready.

**Success Criteria:** Data table worker verifies row/cell claims before `ready`, failed verification marks the table/job failed, and successful reports are retrievable.

**Tests:** `tldw_Server_API/tests/DataTables/test_data_tables_worker.py`, `test_data_tables_jobs_integration.py`, and DB metadata helper tests.

**Status:** Not used for this PR

- [ ] Existing Data Tables job persistence is not the path used by the Research Workspace inline data-table artifact in this PR; Research Workspace data-table generation is covered by Stage 5.
- [ ] Add a narrow `claim_verification_json` column to `data_tables` if the Research Workspace UI is later moved to the persistent Data Tables job path.
- [ ] Add SQLite schema, PostgreSQL late-column ensure logic, create/update/list/detail serialization, and DB helper coverage for `claim_verification_json`.
- [ ] Build table units from row values, prioritizing entities, numbers, dates, relationships, and source-backed cells.
- [ ] Verify rows after normalization and before `persist_data_table_generation(...)`.
- [ ] On `failed`, update the table status to `failed`, store the report JSON, set `last_error` to a concise verification failure code, and fail the job.
- [ ] On `needs_revision`, persist the table as ready with the report and review metadata so the UI can flag it.
- [ ] On `grounded`, persist the table as ready with the report.
- [ ] Add worker tests for failed, needs-revision, grounded, cancelled-before-verify, and cancelled-after-verify paths.

## Stage 5: Move Remaining Research Workspace Generators Backend-Side

**Goal:** Eliminate frontend-only Research Workspace artifact generation for the target artifact types before applying the claim gate.

**Success Criteria:** Research Workspace no longer uses `tldwClient.createChatCompletion` directly for quizzes, data tables, audio summaries, or mindmaps; backend generation runs verification internally.

**Tests:** `tldw_Server_API/tests/ResearchWorkspace/test_artifact_generation_api.py` plus existing StudioPane tests.

**Status:** Complete

- [x] Add a backend Research Workspace artifact generation service for source selection, prompt construction, draft generation, unit normalization, verification, and response shaping.
- [x] Add backend endpoints for Research Workspace mindmap, data-table, and audio-summary draft generation.
- [x] Switch Research Workspace quiz generation to the existing backend quiz endpoint/service rather than the frontend `createChatCompletion` path.
- [x] Switch Research Workspace data-table generation to a backend synchronous draft service for inline results.
- [x] Move audio summary script generation backend-side and verify the script before returning it for TTS.
- [x] Move mindmap Mermaid generation backend-side, normalize nodes/edges into units, and verify before returning a diagram.
- [x] Preserve source media IDs and claim-verification metadata in the backend response.
- [x] Return failed or needs-revision verification as an artifact-generation error, not a completed artifact.
- [x] Update StudioPane tests that currently mock `createChatCompletion` for these artifact types to mock backend services instead.
- [x] Add a regression check that Research Workspace generation does not call public Claims endpoints.

## Stage 6: Frontend Display and Review State

**Goal:** Surface verification results in the full Research Workspace app without adding a duplicate fact-check model.

**Success Criteria:** Completed artifacts show claim-verification status, needs-revision artifacts are visibly reviewable, and failed generation reports are shown as generation errors.

**Tests:** StudioPane and TraceableArtifactDetail tests.

**Status:** Complete

- [x] Add frontend service response support for the backend `claim_verification` wrapper.
- [x] Add an advanced Research Workspace setting for claims-verification provider/model, defaulting to the generation provider/model.
- [x] Store backend verification output at `artifact.data.claimVerification`.
- [x] Show the claims-verification provider/model whenever it differs from the generation LLM.
- [x] Return non-grounded generations as errors before completed artifacts are created.
- [x] Show artifact-level verifier metadata in `TraceableArtifactDetail`.
- [x] Keep display labels derived from Claims statuses; do not introduce a separate `factCheck` vocabulary.
- [x] Add tests for grounded metadata display, non-grounded backend errors, and missing-verification legacy artifacts.

## Stage 7: Verification and PR Completion

**Goal:** Prove the implementation works and update the PR cleanly.

**Success Criteria:** Focused backend/frontend tests pass, touched-code Bandit is clean, and PR #2633 is updated against `dev`.

**Tests:** Focused pytest, focused Vitest, `git diff --check`, Bandit on touched backend paths.

**Status:** In Progress

- [x] Run focused Claims tests.
- [x] Run focused quiz, flashcard, slides, data-table, and Research Workspace backend tests.
- [x] Run focused flashcard backend tests.
- [x] Run focused StudioPane frontend tests.
- [x] Run frontend typecheck.
- [x] Run Bandit on touched backend paths through the project virtual environment.
- [x] Run `git diff --check`.
- [x] Update Backlog task `TASK-12146` with touched files, verification results, blockers, and PR link.
- [ ] Commit the plan and implementation work in logical increments.
- [ ] Push branch `codex/issue-2605-research-workspace-uat`.
- [ ] Update PR #2633 with a human-owned change summary requirement note.

Latest verification evidence:

- `python -m pytest tldw_Server_API/tests/Claims/test_artifact_verification.py tldw_Server_API/tests/Claims/test_artifact_verification_properties.py -q`: 15 passed.
- `python -m pytest tldw_Server_API/tests/ResearchWorkspace/test_artifact_generation_service.py tldw_Server_API/tests/Slides/test_slides_api.py tldw_Server_API/tests/Slides/test_slides_endpoint_sanitization.py -q`: 85 passed.
- `python -m pytest tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py tldw_Server_API/tests/Quizzes/test_quiz_generate_endpoint_multi_source.py tldw_Server_API/tests/Quizzes/test_quiz_generator_test_mode.py -q`: 170 passed.
- `bun run typecheck`: passed.
- `bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage1.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx ../packages/ui/src/components/Option/ResearchWorkspace/StudioPane/__tests__/TraceableArtifactDetail.test.tsx --reporter=dot`: 112 passed.
- `git diff --check`: passed.
- `python -m bandit -r <touched backend source paths> -f json -o /tmp/bandit_research_workspace_claims.json`: 0 results.
