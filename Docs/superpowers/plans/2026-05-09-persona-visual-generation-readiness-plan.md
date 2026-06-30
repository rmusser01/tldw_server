# Persona Visual Generation Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Persona/Buddy visual asset generation readiness visible and actionable before users enqueue background generation jobs.

**Architecture:** Add a small pack-scoped readiness endpoint that reports the persona visual worker flag and configured image generation backends. The WebUI consumes that readiness signal, classifies it into user-facing states, and disables enqueue when Jobs or image provider setup cannot support generation.

**Tech Stack:** FastAPI, Pydantic, Jobs, image generation adapter registry, React, Vitest, Testing Library.

---

## File Structure

- Modify `tldw_Server_API/app/api/v1/schemas/persona.py` to add `PersonaVisualGenerationReadinessResponse`.
- Modify `tldw_Server_API/app/api/v1/endpoints/persona.py` to add a pack-scoped readiness endpoint beside `generation-jobs`.
- Modify `tldw_Server_API/tests/Persona/test_persona_visuals_api.py` to cover disabled worker and available backend readiness responses.
- Modify `apps/packages/ui/src/types/persona-visuals.ts` to add the readiness response type.
- Modify `apps/packages/ui/src/services/persona-visuals.ts` to add `getPersonaVisualGenerationReadiness`.
- Create `apps/packages/ui/src/components/PersonaGarden/personaVisualGenerationReadiness.ts` for the UI classifier.
- Create `apps/packages/ui/src/components/PersonaGarden/__tests__/personaVisualGenerationReadiness.test.ts` for classifier coverage.
- Modify `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx` to load readiness, render setup state, and gate enqueue.
- Modify `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx` to cover gated and ready enqueue flows.
- Update `backlog/tasks/task-180 - Improve-Persona-Visuals-generation-setup-and-unavailable-provider-states.md` with verification and final status.

## Stage 1: Backend Readiness Contract

**Goal:** Expose current generation setup without enqueuing a job.

**Success Criteria:** The endpoint distinguishes Jobs worker enablement from image backend availability and remains scoped to the persona/pack owner.

**Tests:** `test_visual_generation_readiness_reports_disabled_worker_and_missing_provider` and `test_visual_generation_readiness_reports_available_backend`.

**Status:** Complete

- [x] Add failing API tests in `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`.
- [x] Run the two new tests and verify they fail because the endpoint/schema do not exist.
- [x] Add `PersonaVisualGenerationReadinessResponse`.
- [x] Add `GET /profiles/{persona_id}/visual-packs/{pack_id}/generation-readiness`.
- [x] Re-run the two API tests and verify they pass.

## Stage 2: Frontend Readiness Classifier

**Goal:** Keep setup-state rules testable outside the large editor component.

**Success Criteria:** Classifier returns separate messages for worker disabled, no image providers, missing default backend, unavailable requested backend, loading, error, and ready states.

**Tests:** `personaVisualGenerationReadiness.test.ts`.

**Status:** Complete

- [x] Add failing classifier tests.
- [x] Run classifier tests and verify they fail because the helper does not exist.
- [x] Implement the minimal classifier helper.
- [x] Re-run classifier tests and verify they pass.

## Stage 3: VisualPackEditor Gating

**Goal:** Show setup/unavailable states before enqueue and preserve ready enqueue plus review.

**Success Criteria:** The Queue button is disabled for known unavailable readiness states, copy separates Jobs/backend from image provider/model setup, and the existing ready enqueue/review test continues to pass.

**Tests:** `VisualPackEditor.test.tsx` focused cases.

**Status:** Complete

- [x] Add failing editor tests for disabled worker and missing provider states.
- [x] Update the existing enqueue test to mock ready readiness.
- [x] Run editor tests and verify the new tests fail before implementation.
- [x] Wire readiness loading into `VisualPackEditor.tsx`.
- [x] Render readiness state near generation controls and disable enqueue when classified unavailable.
- [x] Re-run editor tests and verify they pass.

## Stage 4: Verification and PR Packaging

**Goal:** Leave the branch reviewable and tied back to issue `#1431`.

**Success Criteria:** Focused frontend/backend tests pass, Bandit is run on touched backend files, Backlog task is updated, and a PR is opened against `dev`.

**Tests:** Focused Vitest, focused pytest, Bandit touched scope.

**Status:** Complete

- [x] Run focused Vitest for Persona Visuals editor/classifier/diagnostics tests.
- [x] Run focused pytest for Persona visual API readiness tests.
- [x] Run Bandit on touched backend files.
- [x] Update Backlog acceptance criteria and verification notes.
- [x] Commit, push, and open the PR linked to `#1431`.
