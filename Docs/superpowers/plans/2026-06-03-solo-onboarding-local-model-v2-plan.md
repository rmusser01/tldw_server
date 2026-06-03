# Solo Onboarding Local Model V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement PR4 from the solo onboarding V2 roadmap: make local OpenAI-compatible/Ollama setup a first-class guided alternative with repeatable UAT coverage.

**Architecture:** Keep setup/readiness/completion backend-authoritative. Extend the existing onboarding UAT harness first, then adjust the setup provider validation contract and WebUI provider step so local endpoints can succeed through discovered-model and manual-model paths without implying the app installs local runtimes.

**Tech Stack:** FastAPI setup endpoints, Python provider validation tests, React setup wizard in `apps/packages/ui`, Vitest, Playwright onboarding UAT harness, repo `mock_openai_server`.

**Backlog:** `TASK-601`

---

## File Map

- Modify: `apps/tldw-frontend/scripts/__tests__/onboarding-uat-runner.test.ts`
  - Fix the stale hosted-success fixture assertion introduced by the post-rebase PR feedback commit.
- Modify: `tldw_Server_API/app/core/Setup/provider_validation.py`
  - Return an accepted/manual fallback result when a local OpenAI-compatible endpoint is reachable but `/models` is unsupported or returns an unsupported shape.
- Modify: `tldw_Server_API/tests/Setup/test_setup_provider_validation.py`
  - Add/adjust backend tests for discovered models, manual fallback, auth failure, unreachable endpoint, and sanitized output.
- Modify: `tldw_Server_API/app/api/v1/schemas/setup_schemas.py`
  - Only if needed, add typed metadata to `SetupProviderValidationResponse` for manual fallback guidance.
- Modify: `apps/packages/ui/src/types/setup-onboarding.ts`
  - Mirror any backend validation response shape additions.
- Modify: `apps/packages/ui/src/components/Option/Onboarding/steps/ProviderSetupStep.tsx`
  - Add clearer local endpoint guidance, model discovery picker behavior, manual entry fallback copy, and inline recovery buttons/copy.
- Modify: `apps/packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx`
  - Cover local manual fallback, endpoint recovery copy/actions, model unavailable guidance, and provider switch state isolation.
- Modify: `apps/tldw-frontend/e2e/onboarding-uat/scenarios.ts`
  - Add explicit PR4 local scenarios.
- Modify/Create: `apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/*.json`
  - Add static mock configs for models unsupported and local model unavailable.
- Modify: `apps/tldw-frontend/e2e/onboarding-uat/setup-happy-path.spec.ts`
  - Split/strengthen discovered-model and manual-model local happy paths.
- Modify: `apps/tldw-frontend/e2e/onboarding-uat/recovery.spec.ts`
  - Add local endpoint unreachable, model unavailable, and switch-back recovery assertions as needed.
- Modify: `backlog/tasks/task-601 - Implement-solo-onboarding-local-model-guided-alternative-V2.md`
  - Keep plan, touched files, verification, and final summary current.

## Stage 1: Baseline Harness Cleanup

**Goal:** Restore the focused onboarding harness unit baseline before PR4 behavior changes.

**Success Criteria:** `onboarding-uat-runner.test.ts` passes with the merged hosted-success fixture contract.

**Tests:** `bunx vitest run scripts/__tests__/onboarding-uat-runner.test.ts --reporter=dot`

**Status:** Complete

- [x] **Step 1: Confirm the existing red baseline**

  Run: `bunx vitest run scripts/__tests__/onboarding-uat-runner.test.ts --reporter=dot`

  Expected: FAIL at the hosted-success `require_auth` assertion.

- [x] **Step 2: Update the stale assertion**

  In `apps/tldw-frontend/scripts/__tests__/onboarding-uat-runner.test.ts`, change the hosted-success auth assertion so it matches `apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/hosted-success.json`.

  Expected shape:

  ```ts
  expect(configs["hosted-success.json"].server?.require_auth).toBe(false)
  ```

- [x] **Step 3: Verify green**

  Run: `bunx vitest run scripts/__tests__/onboarding-uat-runner.test.ts --reporter=dot`

  Expected: PASS.

- [x] **Step 4: Commit**

  Commit message: `test(onboarding): align UAT hosted fixture contract`

## Stage 2: Backend Local Validation Manual Fallback

**Goal:** Make local endpoint validation distinguish "reachable but model listing unsupported" from "endpoint broken" so manual model entry can be first-class.

**Success Criteria:** Local OpenAI-compatible validation returns:

- `ready` + `live_non_generative` + discovered models when `/models` returns OpenAI shape.
- `accepted` + a non-generative/manual fallback validation level when the endpoint is reachable but `/models` shape is unsupported.
- `failed` for unreachable endpoint, auth failure, disallowed target, invalid JSON, or HTTP error.

**Tests:** `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_setup_provider_validation.py -q`

**Status:** Complete

- [x] **Step 1: Write failing backend tests**

  Add tests in `tldw_Server_API/tests/Setup/test_setup_provider_validation.py`:

  - `test_openai_local_models_unsupported_accepts_manual_model_fallback`
  - `test_openai_local_empty_models_accepts_manual_model_fallback`
  - Keep `test_unsupported_api_shape_maps_to_unsupported_api_shape` or rename it if its desired contract changes.

  Expected new response:

  ```python
  assert response.status == "accepted"
  assert response.models == []
  assert response.validation_level == "live_endpoint_shape"
  assert response.can_gate_first_chat is True
  assert response.failure_category == "models_unavailable"
  ```

- [x] **Step 2: Run red tests**

  Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_setup_provider_validation.py -q`

  Expected: New fallback tests fail against current hard-failed unsupported shape.

- [x] **Step 3: Implement minimal backend fallback**

  In `tldw_Server_API/app/core/Setup/provider_validation.py`, add a small accepted response helper for reachable local endpoints whose `/models` response cannot be used for discovery.

  Keep auth, transport, invalid JSON, disallowed target, and HTTP error behavior failed/sanitized.

- [x] **Step 4: Verify backend tests**

  Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_setup_provider_validation.py -q`

  Expected: PASS.

- [x] **Step 5: Commit**

  Commit message: `feat(setup): accept manual local model fallback`

## Stage 3: Frontend Local Provider Guidance And Recovery

**Goal:** Make the provider step practical for local setup: clear endpoint examples, discovered model selection, manual fallback copy, and explicit inline actions.

**Success Criteria:**

- Local endpoint cards explain expected OpenAI-compatible base URLs and common examples.
- Discovered models are selectable without removing manual model entry.
- Manual model entry remains available when validation says model discovery is unavailable but the endpoint shape is acceptable.
- Failure copy distinguishes service not running, wrong host/port, missing `/v1`, model not pulled/loaded, and auth/token required where backend categories allow it.
- Recovery controls include retry, edit endpoint, switch provider, continue with manual model when allowed, and local setup guide/docs.
- Hosted and local validation/save state remains isolated after switching defaults.

**Tests:** `bunx vitest run ../packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx --reporter=dot`

**Status:** Complete

- [x] **Step 1: Write failing frontend tests**

  Add tests in `ProviderSetupStep.test.tsx` for:

  - Accepted local manual fallback copy enables save/continue after manual model entry.
  - `model_discovery_unavailable` copy offers manual model continuation and retry/edit/switch actions.
  - `local_provider_unreachable` copy offers retry and edit endpoint without enabling continue.
  - Switching from local default back to hosted requires current hosted validation/save and does not reuse stale local validation.

- [x] **Step 2: Run red frontend tests**

  Run: `bunx vitest run ../packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx --reporter=dot`

  Expected: New tests fail against current generic copy/actions.

- [x] **Step 3: Implement UI copy/actions**

  In `ProviderSetupStep.tsx`, keep the existing card structure but add small helper functions for local validation copy and recovery actions. Avoid a large refactor.

- [x] **Step 4: Verify frontend tests**

  Run: `bunx vitest run ../packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx --reporter=dot`

  Expected: PASS.

- [x] **Step 5: Commit**

  Commit message: `feat(onboarding): improve local provider recovery UI`

## Stage 4: Harness-First Local UAT Scenarios

**Goal:** Prove PR4 behavior through the real backend, real WebUI, and repo mock OpenAI-compatible server.

**Success Criteria:**

- UAT scenario exists for local reachable `/models` discovery path.
- UAT scenario exists for reachable endpoint with `/models` unsupported, manual model entry, and first chat success.
- UAT scenario exists for local endpoint unreachable recovery.
- UAT scenario exists for local model unavailable recovery.
- UAT scenario exists for local default switched back to hosted without stale state.
- Mobile local setup is covered.

**Tests:** `bun run e2e:onboarding:uat --scenario <scenario-id> --viewport <desktop|mobile>`

**Status:** Not Started

- [ ] **Step 1: Add failing/expanded UAT scenario declarations**

  Update `apps/tldw-frontend/e2e/onboarding-uat/scenarios.ts` with explicit PR4 scenario IDs, for example:

  - `local-openai-discovered-model-first-chat`
  - `local-openai-manual-model-first-chat`
  - `local-openai-model-unavailable-recovery`
  - `local-to-hosted-switch-state-isolated`
  - mobile coverage for local setup

- [ ] **Step 2: Add static mock configs**

  Add configs under `apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/` for:

  - Models unsupported or failing on `/v1/models`.
  - Chat failure for a selected local model.
  - Any provider-switch path that needs deterministic behavior.

- [ ] **Step 3: Write/extend Playwright UAT tests**

  Update `setup-happy-path.spec.ts` and `recovery.spec.ts` using existing helpers. Do not use Playwright route mocks for provider behavior.

- [ ] **Step 4: Verify targeted UAT scenarios**

  Run each new scenario manually with the harness. Preserve artifact paths in the Backlog task.

- [ ] **Step 5: Commit**

  Commit message: `test(onboarding): cover guided local model setup UAT`

## Stage 5: Final Verification And PR Prep

**Goal:** Produce a reviewable one-PR branch with four staged commits plus verification evidence.

**Success Criteria:** Focused tests pass; UAT evidence is recorded; Bandit is run for touched backend code; Backlog task has final notes.

**Tests:**

- `bunx vitest run scripts/__tests__/onboarding-uat-runner.test.ts --reporter=dot`
- `bunx vitest run ../packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx --reporter=dot`
- `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_setup_provider_validation.py -q`
- `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Setup/provider_validation.py tldw_Server_API/app/api/v1/endpoints/setup.py tldw_Server_API/app/api/v1/schemas/setup_schemas.py -f json -o /tmp/bandit_onboarding_local_model_v2.json`
- Targeted `bun run e2e:onboarding:uat` scenarios from Stage 4.
- `git diff --check`

**Status:** Not Started

- [ ] **Step 1: Run all focused verification**
- [ ] **Step 2: Update `TASK-601` with touched files, verification results, known skips, and final summary**
- [ ] **Step 3: Review diff for secrets, raw paths, and unrelated churn**
- [ ] **Step 4: Push branch and prepare PR summary**
