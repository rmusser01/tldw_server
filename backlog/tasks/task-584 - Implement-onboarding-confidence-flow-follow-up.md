---
id: TASK-584
title: Implement onboarding confidence flow follow-up
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-01 15:28
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
- apps/packages/ui/src/components/Option/Onboarding/SetupReadinessPanel.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/SetupReadinessPanel.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/steps/ProviderSetupStep.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/steps/FirstChatStep.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/FirstSourceMilestonePrompt.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx
- apps/packages/ui/src/routes/option-index.tsx
- apps/packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx
- apps/packages/ui/src/utils/quick-ingest-open.ts
- apps/packages/ui/src/utils/__tests__/quick-ingest-open.test.ts
- apps/packages/ui/src/store/quick-ingest-session.ts
- apps/packages/ui/src/components/Common/QuickIngest/IngestWizardContext.tsx
- apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx
- apps/packages/ui/src/components/Common/QuickIngest/QueueTab/FileDropZone.tsx
- apps/packages/ui/src/components/Common/QuickIngest/__tests__/FileDropZone.acceptance.test.tsx
- apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx
- apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx
- apps/packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx
- apps/tldw-frontend/e2e/smoke/smoke.setup.ts
- apps/tldw-frontend/e2e/workflows/onboarding-ingestion-first.spec.ts
- apps/tldw-frontend/e2e/workflows/unified-first-run-onboarding.spec.ts
- tldw_Server_API/app/api/v1/endpoints/setup.py
- tldw_Server_API/app/api/v1/schemas/setup_schemas.py
- tldw_Server_API/app/core/Setup/provider_validation.py
- tldw_Server_API/app/core/Setup/setup_manager.py
- tldw_Server_API/app/core/Setup/first_run_state.py
- tldw_Server_API/tests/Setup/test_setup_provider_validation.py
- tldw_Server_API/tests/Setup/test_setup_first_chat_completion.py
- tldw_Server_API/tests/Setup/test_first_run_state.py
- tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py
- Docs/superpowers/specs/2026-06-01-onboarding-confidence-flow-design.md
- Docs/superpowers/plans/2026-06-01-onboarding-confidence-flow-implementation-plan.md
references:
- https://github.com/rmusser01/tldw_server/pull/2214
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved onboarding confidence flow plan as one PR with four staged commits: provider validation gate, setup readiness panel, first-chat recovery actions, and first-source guided milestone. Use current dev contracts, preserve backend-authoritative setup/readiness state, and do not replay stale unified-solo-onboarding worktree changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Default first-chat provider requires manual validation and save before continuing; non-default providers can be saved unverified.
- [x] #2 Setup readiness panel uses existing readiness APIs/types and remains non-blocking for optional lanes.
- [x] #3 First-chat failures render inline recovery actions: retry, edit provider, switch provider, skip setup, and endpoint recovery where relevant.
- [x] #4 Post-onboarding first-source milestone offers Web URL, File upload, and Paste text, and only offers grounded chat after source readiness is confirmed.
- [x] #5 Focused backend, frontend, E2E, Bandit, and diff verification are recorded before PR closeout.
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
- Task 2 spec and quality fixes:
  - `useSetupReadinessSummary` now fetches first-run status and profiles together, keeps first-run status authoritative, treats profile enrichment as best-effort, and handles React StrictMode effect setup/cleanup correctly.
  - Wizard readiness refreshes are fire-and-forget after setup-changing provider/default actions, so primary buttons are not blocked by readiness refresh latency.
  - Added coverage for status-without-lanes profile enrichment, profile failure fallback, retry wiring, skipped completion refresh, deferred readiness refresh, and StrictMode loading behavior.
  - Focused frontend verification for the final Task 2 state passed from `apps/tldw-frontend`, 3 files / 35 tests.
- Task 3 first-chat recovery actions:
  - `FirstChatStep` normalizes backend failure aliases and categories into recovery buckets, keeps failed attempts visible across retries, and shows explicit actions for retry, edit provider, switch provider, skip setup, and endpoint checks where relevant.
  - Added backend category coverage for `auth_failed`, `rate_limited`, `network_error`, `provider_error`, `request_invalid`, `configuration_error`, and `empty_response`.
  - Added request-exception handling so thrown retry attempts do not leave stale previous failure copy, and added a wizard-level pending guard to prevent duplicate skip submissions.
  - Focused frontend verification for Task 3 passed from `apps/tldw-frontend`, including FirstChatStep and UnifiedSetupWizard recovery coverage.
- Task 4 first-source guided milestone:
  - `FirstSourceMilestonePrompt` offers Web URL, File upload, and Paste text, and shows idle, processing, error, and ready states before grounded chat is offered.
  - OptionIndex scopes first-source state to durable quick-ingest sessions with first-source open detail and no longer unlocks grounded chat from unrelated global quick-ingest success or processing state.
  - Quick ingest now persists the first-source add mode, preserves first-source open detail during wizard sync, focuses the selected URL/file/paste entry, and queues pasted text as a `text/plain` `pasted-text.txt` file.
  - RED coverage failed first on missing picker/metadata/ready states, unrelated global quick-ingest leakage, ignored persisted first-source summaries, missing add-mode handoff, lost first-source open detail, and paste mode routing through URL input.
  - GREEN coverage passed for the targeted Task 4 suites, then the broader focused onboarding frontend suite passed from `apps/tldw-frontend`, 11 files / 155 tests.
- Final closeout verification after rebase:
  - `bun run test:run ../packages/ui/src/hooks/__tests__/useSetupReadinessSummary.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/SetupReadinessPanel.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/ProviderSetupStep.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx ../packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx ../packages/ui/src/utils/__tests__/quick-ingest-open.test.ts ../packages/ui/src/hooks/__tests__/usePostOnboardingMediaReadiness.test.tsx ../packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx ../packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx --reporter=dot` passed, 11 files / 155 tests.
  - `bun run e2e:pw e2e/workflows/unified-first-run-onboarding.spec.ts --project=chromium --reporter=line` passed, 3 tests, with the local Next test server run outside the sandbox after sandbox port binding failed earlier.
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_setup_provider_validation.py tldw_Server_API/tests/Setup/test_setup_first_chat_completion.py tldw_Server_API/tests/Setup/test_first_run_state.py tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py -q` passed, 143 tests.
  - `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/setup.py tldw_Server_API/app/api/v1/schemas/setup_schemas.py tldw_Server_API/app/core/Setup/provider_validation.py tldw_Server_API/app/core/Setup/setup_manager.py tldw_Server_API/app/core/Setup/first_run_state.py -f json -o /tmp/bandit_task584_onboarding_confidence_flow.json` reported 0 results.
  - `git diff --check origin/dev..HEAD` and final unstaged `git diff --check` passed.
- PR opened as draft for review: https://github.com/rmusser01/tldw_server/pull/2214. Human-written Change summary is still required before marking ready or merging per repo policy.
- PR review follow-up after rebase:
  - Rebased `codex/onboarding-confidence-flow` on latest `origin/dev`.
  - Setup readiness refresh now preserves the last known status on refresh failure, labels non-ready chat states as first-chat blockers, avoids duplicate detail keys, and guards wizard retry refresh failures.
  - First-chat retry clears stale response/error/category state before a new attempt.
  - First-source recovery copy no longer points users to a hidden Add Source action, radio choices expose a keyboard focus ring, and retry after reload uses the persisted first-source add mode.
  - Quick ingest preserves raw pasted-text whitespace, clears persisted first-source mode on reset, and file-drop autofocus is one-shot across disabled toggles.
  - The first-source quick-ingest open-detail guard now has an honest predicate type for legacy `firstSource: true` markers instead of narrowing them to milestone-only source details.
  - Backend first-run provider save handling now preserves existing hosted provider keys without overwriting secrets, reports existing local endpoint token configuration, and keeps rejected raw provider credentials out of persisted first-run state.
  - Smoke and first-source E2E fixtures were updated to match the completed first-run app shell and current first-source milestone flow.
- PR review follow-up verification:
  - Focused frontend review suite: `bun run test:run ../packages/ui/src/hooks/__tests__/useSetupReadinessSummary.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/SetupReadinessPanel.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx ../packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx ../packages/ui/src/components/Common/QuickIngest/__tests__/FileDropZone.acceptance.test.tsx ../packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx ../packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx --reporter=dot` passed, 9 files / 139 tests.
  - Expanded frontend review suite with quick-ingest open-detail guard coverage: `bun run test:run ../packages/ui/src/hooks/__tests__/useSetupReadinessSummary.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/SetupReadinessPanel.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/FirstSourceMilestonePrompt.test.tsx ../packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx ../packages/ui/src/routes/__tests__/option-index.unified-setup.test.tsx ../packages/ui/src/utils/__tests__/quick-ingest-open.test.ts ../packages/ui/src/components/Common/QuickIngest/__tests__/FileDropZone.acceptance.test.tsx ../packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.integration.test.tsx ../packages/ui/src/components/Common/QuickIngest/__tests__/QuickIngestWizardModal.session.test.tsx --reporter=dot` passed, 10 files / 146 tests.
  - Stage 6 interaction smoke: `bun run e2e:smoke:interaction:stage1 --project=chromium --reporter=line` passed, 2 tests, after rerunning outside the sandbox because the sandboxed server could not bind `0.0.0.0:8080`.
  - Onboarding first-source E2E: `bun run e2e:onboarding --project=chromium --reporter=line` passed, 2 tests, after rerunning outside the sandbox for the same port-binding restriction.
  - Focused backend setup suite: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Setup/test_setup_provider_validation.py tldw_Server_API/tests/Setup/test_setup_first_chat_completion.py tldw_Server_API/tests/Setup/test_first_run_state.py tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py -q` passed, 145 tests with 5 warnings.
  - Bandit production scope: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/setup.py -f json -o /tmp/bandit_task584_pr2214_review_fixes.json` reported 0 results.
  - `git diff --check` passed after the final review-fix edits.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the onboarding confidence flow follow-up as a rebased four-stage feature branch: provider validation before first chat, a backend-readiness panel inside the setup shell, inline first-chat recovery actions, and a post-onboarding first-source milestone integrated with quick ingest. PR review feedback was addressed after rebasing on latest `origin/dev`, including readiness refresh resilience, first-chat retry state cleanup, first-source recovery/focus/retry fixes, quick-ingest paste/autofocus/session/type-guard cleanup fixes, hosted/local provider credential handling fixes, and refreshed smoke/first-source E2E coverage. Final verification after review fixes: expanded frontend review suite passed (10 files / 146 tests), Stage 6 smoke passed (2 tests), first-source onboarding E2E passed (2 tests), focused backend setup suite passed (145 tests, 5 warnings), Bandit on touched production setup endpoint code reported 0 results, and git diff whitespace checks passed. No known blockers remain.
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
