---
id: TASK-584
title: Implement onboarding confidence flow follow-up
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-06-01 06:14
labels:
- onboarding
- webui
- setup
dependencies: []
documentation:
- Docs/superpowers/specs/2026-06-01-onboarding-confidence-flow-design.md
- Docs/superpowers/plans/2026-06-01-onboarding-confidence-flow-implementation-plan.md
priority: high
modified_files:
- apps/packages/ui/src/hooks/useSetupReadinessSummary.ts
- apps/packages/ui/src/hooks/__tests__/useSetupReadinessSummary.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved onboarding confidence flow plan as one PR with four staged commits: provider validation gate, setup readiness panel, first-chat recovery actions, and first-source guided milestone. Use current dev contracts, preserve backend-authoritative setup/readiness state, and do not replay stale unified-solo-onboarding worktree changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Default first-chat provider requires manual validation and save before continuing; non-default providers can be saved unverified.
- [x] #2 Setup readiness panel uses existing readiness APIs/types and remains non-blocking for optional lanes.
- [ ] #3 First-chat failures render inline recovery actions: retry, edit provider, switch provider, skip setup, and endpoint recovery where relevant.
- [ ] #4 Post-onboarding first-source milestone offers Web URL, File upload, and Paste text, and only offers grounded chat after source readiness is confirmed.
- [ ] #5 Focused backend, frontend, E2E, Bandit, and diff verification are recorded before PR closeout.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Implementation task opened before code edits per AGENTS.md. Controller will dispatch Task 1 provider validation gate implementer first, then run spec and code-quality reviews before moving to Task 2.
- Task 1 provider validation gate implemented:
  - Backend validation responses now include non-secret `validation_level` and `can_gate_first_chat` metadata.
  - Hosted providers return `accepted` with `local_syntax` gate metadata after local syntax/presence checks.
  - Local OpenAI-compatible `/models` validation returns `ready` with `live_non_generative` gate metadata and discovered models.
  - Native Kobold endpoint shape validation returns `ready` with `live_endpoint_shape` gate metadata.
  - Failed validation keeps `can_gate_first_chat` false by schema default.
  - Frontend provider setup now requires the default provider to be manually validated and saved before Continue is enabled.
  - Non-default selected providers can still be saved without validation.
  - Editing default provider API key, base URL, or model invalidates the stored validation result via a non-secret fingerprint.
  - Validation fingerprints use provider key, local base URL, selected model, a non-secret user-edit revision, and secret-present boolean only.
  - Validation state is transient; raw API keys are cleared after save while masked saved-key state can preserve validation validity.
  - Local model discovery renders model choices while preserving manual model entry.
- Task 1 spec compliance fix:
  - Added a non-secret per-provider edit revision that increments only for user-initiated API key, base URL, and model edits.
  - Save-induced raw API key clearing does not increment the edit revision, so validated-and-saved masked key state remains current.
  - Added regression coverage for API key edits after save and local endpoint base URL edits invalidating validation.
- Task 1 back-navigation fix:
  - Lifted the provider setup cache to `UnifiedSetupWizard` so saved provider responses, validation responses, and non-secret edit revisions survive normal step unmount/remount.
  - Back navigation from ingest defaults to provider setup now keeps a validated-and-saved hosted default provider eligible to continue without re-pasting the raw API key.
  - Validation cache remains frontend-local/transient and contains no raw secrets.
  - Added wizard-level regression coverage for validate/save/continue/back/continue without revalidation.
- Task 1 saved hosted revalidation fix:
  - Added a manual-only hosted validation fallback for saved providers with a masked saved-key marker and no raw API key in the form.
  - Model-only edits still stale validation and require the user to click Validate again before Continue.
  - Revalidation after save records an `accepted` local-syntax result with first-chat-verifies copy without sending or storing raw secrets.
  - Local endpoints and unsaved hosted providers still use the backend validation path.
- Task 1 no-secret hosted save/resume fix:
  - Added backend support for hosted model/default saves without a raw API key when the provider already has a non-placeholder configured secret.
  - Added a sanitized `credential_configured` save response marker so the UI can preserve saved-key presence without raw secrets or fake masked strings.
  - Frontend validation fingerprints and saved-hosted fallback now use `credential_configured` or a real `masked_api_key` as the non-secret credential marker.
  - Provider setup progress records `default_provider_credential_configured` and resumed first-chat back navigation can seed a saved hosted provider marker before manual revalidation.
- Task 1 review fix:
  - First-run step redaction now preserves the exact boolean `default_provider_credential_configured` marker while still redacting secret-like credential, token, API key, password, and auth fields.
  - Unified setup resume no longer infers saved hosted credentials from completed provider steps or first-chat state; hosted saved-credential fallback requires an explicit persisted marker, a real masked key, or an actual save response marker.
  - Saved-hosted validation fallback remains manual-only and constrained to hosted providers that are already saved, have backend-confirmed credential presence, and have no raw API key typed.
  - Provider validation gating now requires `ready` or `accepted` status and honors `can_gate_first_chat` only as an additional positive gate flag.
  - Default model selection no longer falls back to catalog config field names such as `openai_model` or `ollama_model`; validation/save require a real typed or discovered model value.
- Task 1 verification:
  - RED backend: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_setup_provider_validation.py -q` failed with 3 expected missing metadata attribute failures.
  - GREEN backend: same pytest command passed, `25 passed, 5 warnings`.
  - RED frontend: package-local Vitest run failed on missing Validate action / validation gate behavior.
  - GREEN frontend: `bunx vitest run src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx` from `apps/packages/ui` passed, `19 passed`.
  - Requested root-level `bunx vitest run apps/packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx` fails before tests because temp `bunx` Vitest cannot resolve `jsdom`; the repo-local UI command above was used for functional verification.
  - Bandit: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/schemas/setup_schemas.py tldw_Server_API/app/core/Setup/provider_validation.py -f json -o /tmp/bandit_task584_task1.json` passed with 0 results.
- Task 1 fix verification:
  - RED frontend: `bunx vitest run src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx` from `apps/packages/ui` failed on the new API-key edit invalidation regression while the new base URL invalidation regression passed.
  - GREEN frontend: same package-local Vitest command passed, `21 passed`.
  - Backend not rerun for the spec compliance fix because backend files were not touched.
- Task 1 back-navigation fix verification:
  - RED frontend: same package-local Vitest command failed on the new back-navigation regression because Continue was disabled after returning from ingest defaults.
  - GREEN frontend: same package-local Vitest command passed, `22 passed`.
  - Backend not rerun for the back-navigation fix because backend files were not touched.
- Task 1 saved hosted revalidation fix verification:
  - RED frontend: same package-local Vitest command failed on the new validate/save/continue/back/change-model/validate regression with `provider_api_key_required` and Continue disabled.
  - GREEN frontend: same package-local Vitest command passed, `23 passed`.
  - Backend not rerun for the saved hosted revalidation fix because backend files were not touched.
- Task 1 no-secret hosted save/resume fix verification:
  - RED backend: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_setup_provider_validation.py -q` failed on the new configured-secret hosted save regression with `provider_api_key_required`.
  - RED frontend: package-local focused Vitest command failed on post-save model save and resumed first-chat provider edit regressions.
  - GREEN backend: same pytest command passed, `27 passed, 6 warnings`.
  - GREEN frontend: same package-local focused Vitest command passed, `24 passed`.
  - Bandit: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/setup.py tldw_Server_API/app/api/v1/schemas/setup_schemas.py tldw_Server_API/app/core/Setup/setup_manager.py -f json -o /tmp/bandit_task584_task1_no_secret_save.json` passed with 0 results.
- Task 1 review fix verification:
  - RED backend: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_first_run_state.py tldw_Server_API/tests/Setup/test_setup_provider_validation.py -q` failed because `default_provider_credential_configured` was redacted to `********`.
  - RED frontend: package-local focused Vitest command failed on permissive failed-response gating, catalog `model_field` fallback, and resumed saved-credential inference without an explicit marker.
  - GREEN backend: same focused backend pytest command passed, `49 passed, 6 warnings`.
  - GREEN frontend: `bunx vitest run src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx` from `apps/packages/ui` passed, `29 passed`.
  - Bandit: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Setup/first_run_state.py -f json -o /tmp/bandit_task584_task1_redaction_fix.json` passed with 0 results.
- Task 1 commit status: amended existing Task 1 commit as `feat: validate onboarding providers before first chat`.
- Task 1 recovery review fix:
  - Kept the RED regressions for resumed hosted model edits and post-save API key edits requiring Save again after validation.
  - Added saved-payload fingerprints to the provider setup cache so Continue requires the current default provider payload to match both the current validation fingerprint and the current saved fingerprint.
  - Fingerprints remain non-secret: provider key, effective local base URL, selected model, make_default, edit revision, and secret-present state only.
  - Replaced loop-prone initial saved-state seeding with guarded effects that only hydrate missing initial saved fingerprints when the current form still matches the backend-confirmed saved selection and no user edit revision exists.
  - Backend was not touched; Bandit/backend tests not rerun for this recovery.
- Task 1 recovery verification:
  - `bun run test:run ../packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx --reporter=dot` from `apps/tldw-frontend` passed, `31 passed`.
  - `git diff --check` passed.
  - `git diff --cached --check` passed.
- Task 1 default-churn review fix:
  - Added explicit `savedDefaultProvider` provider-step state, lifted through `UnifiedSetupWizard`, so Continue requires the current default provider to also be the last successfully saved default provider.
  - Preserved the existing saved payload fingerprint and transient validation gates; raw secrets are still cleared after save and are not stored in the new state.
  - Added regression coverage for saving OpenAI as default, saving Ollama as the new default after deselecting OpenAI, then reselecting OpenAI; Continue stays disabled until OpenAI is saved again as default.
- Task 1 default-churn verification:
  - RED frontend: `bun run test:run ../packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx --reporter=dot` from `apps/tldw-frontend` failed on the new default-churn regression because Continue was enabled after reselecting the stale OpenAI default.
  - GREEN frontend: `bun run test:run ../packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx --reporter=dot` from `apps/tldw-frontend` passed, `18 passed`.
  - Focused frontend: `bun run test:run ../packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx --reporter=dot` from `apps/tldw-frontend` passed, `32 passed`.
  - Backend was not touched; backend tests and Bandit were not rerun.
- Task 1 backend review-fix provider marker:
  - Allowed exact providers-step `default_provider_credential_configured` boolean marker through setup endpoint state validation and public-state projection.
  - Kept real credential-like provider step data rejected; added endpoint regression coverage for `provider_credential` rejection.
  - RED endpoint: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py::test_first_run_state_persists_provider_credential_configured_marker -q` failed with 400 before the fix.
  - GREEN focused endpoint: same marker test plus `test_first_run_state_rejects_real_provider_credential_step_data` passed, `2 passed`.
  - Required backend suite: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py::test_first_run_state_persists_provider_credential_configured_marker tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py::test_first_run_state_rejects_real_provider_credential_step_data tldw_Server_API/tests/Setup/test_first_run_state.py tldw_Server_API/tests/Setup/test_setup_provider_validation.py -q` passed, `51 passed, 3 warnings`.
  - Bandit production scope: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/setup.py -f json -o /tmp/bandit_task584_task1_provider_marker_app.json` passed with 0 results. Including the integration test file separately returned baseline pytest assert findings plus an existing test secret-string fixture warning, so production scope result is the recorded security gate.
  - `git diff --check` passed before staging.
- Task 1 frontend concurrency review fix:
  - Provider setup controlled object-state callbacks now accept and forward `React.SetStateAction` values so functional updates resolve against the latest parent/internal state instead of render-captured snapshots.
  - Applied the same pass-through functional-update pattern to validation state, saved providers, saved payload fingerprints, and provider edit revisions; saved default provider remains direct assignment.
  - Added a regression with deferred concurrent OpenAI/Ollama validation promises resolved out of order, asserting both validation entries remain visible.
  - Backend was not touched; backend tests and Bandit were not rerun.
- Task 1 frontend concurrency review verification:
  - RED frontend: `bun run test:run ../packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx --reporter=dot` from `apps/tldw-frontend` failed on the new concurrent-validation regression because Ollama's earlier validation entry was overwritten by OpenAI's later result.
  - GREEN frontend: same ProviderSetupStep focused command passed, `19 passed`.
  - Required focused frontend: `bun run test:run ../packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx --reporter=dot` from `apps/tldw-frontend` passed, `33 passed`.
  - `git diff --check` passed.
  - `git diff --cached --check` passed.
- Task 1 final backend test expectation fix:
  - Updated `test_first_run_provider_validate_returns_typed_response_without_token_echo` to assert the full provider validation response schema, including `validation_level` and `can_gate_first_chat`, after the final spec review found the stale exact response assertion.
  - Focused backend suite: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Setup/test_first_run_state.py tldw_Server_API/tests/Setup/test_setup_provider_validation.py tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py` passed, `134 passed, 5 warnings`.
- Task 2 setup readiness panel implemented:
  - Added `useSetupReadinessSummary` as a thin first-run wrapper around `getSetupReadinessStatus({ mode: "first-run" })`.
  - The hook loads on mount, exposes `status`, `loading`, sanitized fallback `error`, and `refresh`, and ignores stale refresh results when a newer request has already resolved.
  - Added `SetupReadinessPanel` for the wizard shell using existing readiness response/lane types and backend lane labels/statuses.
  - The panel renders compact Chat, Embeddings/RAG, and Speech lanes, readable overlay labels, collapsed warnings/blockers/effects details, and retry/loading/error states.
  - Chat failed/blocked state is presented as first-chat blocking; Embeddings/RAG and Speech remain visibly deferrable for not configured, skipped, and overlay-blocked optional states.
  - `UnifiedSetupWizard` now renders the readiness panel below the shell header and above the active step card.
  - Wizard setup-changing actions refresh the readiness summary after provider validation/save, ingest defaults save, audio defaults save, optional advanced save, setup completion, and skip attempts.
  - Backend files were not touched, so Bandit was not rerun for Task 2.
- Task 2 verification:
  - RED frontend: `bun run test:run ../packages/ui/src/hooks/__tests__/useSetupReadinessSummary.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/SetupReadinessPanel.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx --reporter=dot` from `apps/tldw-frontend` failed on missing hook/panel modules and missing wizard readiness rendering/refresh wiring.
  - GREEN frontend: same focused command passed, `25 passed`.
  - Post-cleanup frontend: same focused command passed, `25 passed`.
  - `git diff --check` passed.
  - `git diff --cached --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Task 2 spec-fix worker: useSetupReadinessSummary now fetches first-run status and profiles together, preserving status fields while filling missing lanes, lane ids, supported metadata, profiles, and active/other overlays from profiles when the status payload is empty. Added hook regression coverage for status-without-lanes plus profiles-with-lanes producing panel-ready lanes.

Task 2 spec-fix worker: added wizard readiness refresh wiring coverage for ingest defaults save, audio defaults save, optional advanced save, first-chat completion, failed skip attempts, and the SetupReadinessPanel retry button. No backend files changed; Bandit not required.

Task 2 spec-fix verification: RED hook focused run failed before implementation because getSetupReadinessProfiles was not called and status-without-lanes left lanes undefined. GREEN hook focused run passed, 5 passed. Requested focused frontend run from apps/tldw-frontend passed: 3 files, 32 tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 2 code-quality fix:
- useSetupReadinessSummary now treats first-run readiness status as authoritative and only fails when the status request fails; profile enrichment is best-effort and profile failures keep the successful status visible with no hook error.
- UnifiedSetupWizard no longer refreshes first-run readiness after terminal setup completion, avoiding a transient readiness error during route handoff; parent first-run state refresh/onComplete behavior remains covered.
- Added regressions for status-success/profile-failure and completion without readiness refresh.
- Focused frontend verification: `bun run test:run ../packages/ui/src/hooks/__tests__/useSetupReadinessSummary.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/SetupReadinessPanel.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx --reporter=dot` from `apps/tldw-frontend` passed, 3 files / 33 tests.
- Backend files were not touched; Bandit not required.

Task 2 quality fix:
- Made UnifiedSetupWizard readiness refresh fire-and-forget after setup-changing provider validate/save, ingest defaults save, audio defaults save, optional advanced save, and skip attempts so primary button promises settle from their own action results instead of waiting for readiness status/profile requests.
- Kept SetupReadinessPanel retry explicitly wired to the readiness refresh action and preserved no readiness refresh after terminal first-chat completion.
- Added a deferred-readiness regression proving provider validation and provider save UI settle while readiness refresh promises remain unresolved.
- RED frontend: `bun run test:run ../packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx -t "does not keep provider validation or save pending while readiness refresh is unresolved" --reporter=dot` from `apps/tldw-frontend` failed because the Validate button remained pending on unresolved readiness.
- GREEN frontend: same targeted regression command passed, 1 passed / 20 skipped.
- Focused frontend verification: `bun run test:run ../packages/ui/src/hooks/__tests__/useSetupReadinessSummary.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/SetupReadinessPanel.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx --reporter=dot` from `apps/tldw-frontend` passed, 3 files / 34 tests.
- `git diff --check` passed before staging. `git diff --cached --check` passed after staging. Backend files were not touched; Bandit not required.
Task 2 StrictMode quality fix:
- Reset the `useSetupReadinessSummary` mounted guard during effect setup so React StrictMode setup-cleanup-setup cycles do not leave the hook permanently unmounted.
- Added hook regression coverage rendering under `<React.StrictMode>` and asserting readiness reaches loaded state.
- Preserved request-id stale result handling, best-effort profile fallback, and status-authoritative error semantics.
- RED frontend: `bun run test:run ../packages/ui/src/hooks/__tests__/useSetupReadinessSummary.test.tsx --reporter=dot` from `apps/tldw-frontend` failed on the StrictMode regression with loading stuck true after an initial test-wrapper import fix.
- GREEN focused frontend: same hook command passed, 1 file / 7 tests.
- Required frontend suite: `bun run test:run ../packages/ui/src/hooks/__tests__/useSetupReadinessSummary.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/SetupReadinessPanel.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx --reporter=dot` from `apps/tldw-frontend` passed, 3 files / 35 tests.
- Backend files were not touched; Bandit not required.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
