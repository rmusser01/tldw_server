# Onboarding Confidence Flow Design

Date: 2026-06-01
Status: Ready for user review
Backlog: TASK-583

## Summary

Extend the unified first-time solo onboarding flow with a focused confidence pass before and after the first-chat completion gate. The PR should ship as one coherent change with four staged commits:

1. Manual provider validation.
2. Compact setup readiness panel.
3. Inline first-chat recovery.
4. Guided first-source milestone.

The goal is to make the first successful chat more reliable, make setup posture visible without adding another blocking step, and carry the user from "chat works" to "I added a source and got value."

This work must start from current `origin/dev`. The old `.worktrees/unified-solo-onboarding` branch is only a reference artifact; it must not be rebased or replayed wholesale because it predates newer setup readiness modules now present on `dev`.

## Source Of Truth

Current `dev` already contains the unified onboarding baseline and the setup readiness architecture. The next PR should preserve these contracts:

- Backend first-run state remains authoritative for setup progress, skip, first-chat success, and completion.
- Existing setup readiness modules remain authoritative for readiness lanes, statuses, overlays, previews, provisioning, and persisted readiness snapshots.
- WebUI local state remains temporary: unsaved form values, expanded rows, transient validation responses, and dismissed UI.
- `/setup` remains the backend/operator recovery surface and should share the same APIs and vocabulary as the WebUI.
- Completion still requires a real successful first-chat response unless the user explicitly skips setup.

## Goals

- Prevent avoidable first-chat failures by validating the default provider before continuing.
- Keep validation manual-first so the user controls hosted API calls and local endpoint checks.
- Surface setup readiness in the wizard shell without creating another mandatory step.
- Turn first-chat failures into clear recovery actions instead of generic error text.
- Convert the post-completion "add your first source" prompt into a guided first-value milestone.
- Preserve the existing first-run setup state and readiness APIs rather than adding a parallel onboarding state system.

## Non-Goals

- Do not manage or install Ollama, llama.cpp, or other local runtimes.
- Do not require validation for every non-default provider configured during setup.
- Do not make RAG, embeddings, speech, or storage path readiness mandatory for first-chat completion.
- Do not replace backend `/setup`.
- Do not redesign the entire app shell or settings area.
- Do not rebase or resurrect broad unrelated changes from the stale onboarding worktree.

## Stage 1: Provider Validation

Provider validation is manual-first. The user enters provider settings, clicks `Validate`, reviews the result, then saves and continues.

Rules:

- The selected default provider for first chat must have a non-failed manual validation attempt before `Continue` is enabled.
- Default-provider validation can satisfy the gate in two ways:
  - `ready`: live non-generative validation, local endpoint reachability, or model discovery/check succeeded.
  - `accepted`: the provider has no safe live preflight in this setup path, or the current backend only supports syntax/presence checks. The UI must label this plainly as "format accepted; first chat verifies the provider."
- `failed`, `blocked`, or missing validation results cannot satisfy the default-provider gate.
- Non-default selected providers can be saved as `unverified`.
- Validation should not run automatically on every keystroke, blur, or debounce.
- Hosted providers use non-generative validation when available: model list, auth check, or provider-specific metadata endpoint.
- Hosted providers without safe preflight can be saved as `unverified` when non-default. If selected as the default, the backend must return an explicit `accepted` validation result rather than letting the frontend infer success.
- Local and OpenAI-compatible providers should attempt reachability and model discovery when supported.
- Local providers keep manual model entry fallback when model discovery fails or is unsupported.
- Validation must mask secrets in requests, responses, state, logs, and test fixtures.

UI behavior:

- Each selected provider shows a `Validate` action and a validation state.
- Possible visible states: `Not validated`, `Checking`, `Ready`, `Saved unverified`, `Auth failed`, `Endpoint unreachable`, `Model unavailable`, `Unsupported API shape`, `Rate limited`, `Provider validation unavailable`.
- The default-provider radio/selection remains explicit.
- If a validated provider is edited after validation, its validation state is invalidated until revalidated.
- Model discovery should populate a model picker when available, while keeping a manual model input available.

Backend/API behavior:

- Reuse the existing provider catalog and provider save contracts where practical.
- Add or extend setup provider validation so responses include provider key, status, validation level, failure category, safe message, discovered models if available, and whether the result can satisfy the default-provider gate.
- Keep validation result vocabulary explicit. `ready` means the backend checked the live provider or endpoint without generating tokens. `accepted` means the backend performed only local syntax/presence checks or the provider lacks safe preflight. `failed` means the provider cannot continue as the first-chat default until fixed or changed.
- Do not persist raw provider secrets in first-run state. Persist only masked or derived status where needed.
- Save operations should stay about persistence. Prefer separate validation metadata, or frontend validation cache tied to a non-secret fingerprint, over widening the existing save status enum unless backend persistence truly needs it.

## Stage 2: Setup Readiness Panel

The readiness panel is a compact always-visible summary inside the focused setup shell. It is not a separate wizard step and does not replace first-chat completion.

Panel behavior:

- Show lane summaries for Chat, Embeddings/RAG, and Speech.
- Use backend readiness statuses and overlays only.
- Surface blockers and warnings in expandable lane details.
- Show restart, admin-required, remote-setup-blocked, downloads-disabled, package-installs-disabled, and network-unavailable overlays when returned by the backend.
- Provide next actions without forcing the user out of the wizard.
- Keep optional lanes visibly deferrable.

Placement:

- On desktop, the summary can live in the setup shell header/sidebar area.
- On mobile, the summary should collapse into a compact disclosure above the active step.
- It should not expose normal global app navigation before onboarding completion.

Readiness semantics:

- Chat readiness should reflect the selected provider/default model and validation outcome where available.
- Embeddings/RAG readiness can remain warning/skipped/deferred unless the user explicitly configures it.
- Speech readiness can reflect audio/STT/TTS defaults and setup readiness recommendations but must remain optional.
- Readiness panel status must never mark setup complete; only first-chat completion can do that.

## Stage 3: First-Chat Recovery

When first chat fails, the failed attempt stays visible inline on the first-chat step. The UI should explain the failure category and offer explicit recovery actions.

Recovery actions:

- `Retry`
- `Edit provider`
- `Switch provider`
- `Skip setup`
- `Check endpoint` for local/OpenAI-compatible endpoints

Failure categories should map to practical copy and actions:

- `auth_failed`: update credentials or switch provider.
- `quota_or_rate_limit`: retry later or switch provider.
- `endpoint_unreachable`: check endpoint URL, process, port, firewall, or local service.
- `unsupported_api_shape`: verify OpenAI-compatible path and provider type.
- `model_unavailable`: pick a discovered model or enter a different model.
- `config_write_failed`: retry save or open operator setup recovery.
- `provider_unvalidated`: return to provider validation.
- `unknown`: retry, edit provider, or view operator diagnostics.

The first-chat response text remains visible on success. If setup completion fails after chat succeeds, show that as a separate completion-state problem rather than implying the model response failed.

## Stage 4: First-Source Milestone

After first chat succeeds and setup is completed, the next guided milestone is adding a first source. V1 should upgrade the current prompt into a small guided picker.

Source options:

- `Web URL` as the default focused option.
- `File upload`.
- `Paste text`.

Flow:

1. Show the source picker immediately after onboarding completion.
2. Let the user add exactly one first source through the selected path.
3. Show ingest progress and readiness in plain language.
4. Offer `Ask a question about this source` only after the source is queryable or the backend marks it ready for grounded chat.
5. If ingest fails, keep the chosen source visible and show retry/edit actions.
6. If ingest succeeds but the source is still processing, show processing/readiness status and a safe next action such as viewing the source.
7. Let the user dismiss the milestone; dismissal is frontend-local unless the backend already has a suitable milestone state.

This milestone is post-onboarding. It must not block setup completion and must not require RAG/storage tuning during first-run setup.

## Data And State

Backend-authoritative:

- First-run setup status and current step.
- Completed/acknowledged setup steps.
- Skip state and skip reason.
- First-chat verification and completion state.
- Provider config writes and masked provider status.
- Setup readiness status, lanes, overlays, preview, and provisioning state.

Frontend-local:

- Current unsaved provider form values.
- Expanded readiness lanes.
- Temporary validation response cache for currently edited fields.
- First-source milestone dismissal.
- Unsaved first-source picker state.

State invalidation:

- Editing a validated provider field invalidates that provider's validation result.
- Validation results should be keyed by a non-secret provider fingerprint: provider key, base URL, model, and whether a secret was present. Never include the raw secret in the fingerprint.
- Changing the default provider/model requires validation for the new default.
- A failed first-chat attempt should not erase prior provider validation, but it can annotate the provider with the observed failure category.
- Readiness panel data should refresh after provider validation, provider save, audio defaults save, optional advanced save, first-chat success, and setup completion.

## API Shape

Prefer narrow additions to existing setup API domains:

- Provider catalog: no new catalog source.
- Provider validation: extend the existing validation endpoint/client if available.
- Provider save: add explicit verified/unverified response semantics if needed.
- Readiness status/profiles: consume existing readiness endpoints from the wizard. Do not create a second readiness client or duplicate readiness types.
- First-chat verification: preserve the existing completion gate and enrich failure categories if needed.
- First-source milestone: reuse existing quick-ingest/media APIs and post-onboarding media readiness helpers rather than adding a new ingestion backend.

Any new response fields should be additive, optional where possible, and covered by tests on both backend and client types.

## Error Handling

User-facing errors should be categorized and actionable. Raw exception text, stack traces, filesystem paths, request headers, and secrets should not appear in primary UI.

Provider validation and first-chat errors should use the same category vocabulary where practical. Operator-only diagnostics can expose deeper details behind guarded setup or admin surfaces.

## Testing Strategy

Backend tests:

- Default provider validation passes for safe mocked hosted validation.
- Hosted provider without safe preflight returns an explicit `accepted` validation result that can gate only when the UI labels first-chat verification honestly.
- Local endpoint model discovery success returns model choices.
- Local endpoint discovery failure still allows manual model fallback.
- Validation failures return safe categories and do not leak secrets.
- Validation fingerprints invalidate when provider key, base URL, model, or secret presence changes without storing raw secrets.
- Readiness endpoints remain compatible with the existing lane and overlay contracts.
- First-chat completion still rejects completion without a successful chat.

Frontend unit tests:

- Default provider `Continue` is disabled until validation passes.
- Non-default providers can be saved as unverified.
- Editing provider fields invalidates validation.
- Discovered models populate the picker and manual entry remains available.
- Readiness panel renders lane statuses, blockers, warnings, overlays, and collapsed mobile state.
- First-chat failure renders inline recovery actions without auto-navigation.
- First-source picker defaults to Web URL and supports file/paste choices.

E2E tests:

- Happy path: configure provider, validate, first chat succeeds, setup completes, first-source milestone appears.
- Hosted auth failure: validation blocks default provider continuation and shows credential recovery.
- Hosted provider with syntax-only validation shows `accepted`, allows first-chat verification, and does not claim live readiness.
- Local endpoint failure: validation shows endpoint recovery and model fallback.
- First-chat failure after validated provider: inline recovery allows editing provider and retry.
- Post-onboarding first source: URL ingest succeeds, waits for queryable readiness, then offers a grounded-question follow-up.

## PR And Commit Plan

One PR, four staged commits:

1. `feat: validate onboarding providers before first chat`
2. `feat: show setup readiness in onboarding wizard`
3. `fix: add first-chat onboarding recovery actions`
4. `feat: guide first source after onboarding`

Each commit should keep tests close to the changed contract. If a stage reveals stale old-worktree code that conflicts with current `dev`, prefer reimplementing against current contracts over cherry-picking.

## Risks

- Provider validation can accidentally become generative or quota-consuming.
  - Mitigation: manual-first, non-generative validation by default, and explicit unverified status for providers without safe preflight.
- Readiness panel can add cognitive load.
  - Mitigation: compact summary by default, details behind disclosure, optional lanes marked deferrable.
- First-source milestone can grow into a second onboarding wizard.
  - Mitigation: one-source V1, three input types, and one next action after ingest.
- Old worktree rebase can delete newer readiness architecture.
  - Mitigation: start fresh from `origin/dev`; old branch is reference only.

## Acceptance Criteria

- Manual provider validation is required for the default first-chat provider.
- Non-default providers can be saved as unverified.
- Local providers discover models when supported and keep manual fallback.
- Hosted validation avoids token-generating calls where possible.
- Syntax-only hosted validation is labeled as `accepted`, not live-ready.
- Setup readiness is visible in the wizard shell using backend lanes and overlays.
- First-chat failures stay inline and expose explicit recovery actions.
- Post-completion first-source milestone supports Web URL, file upload, and pasted text, and waits for queryable source readiness before offering a grounded question.
- The implementation does not remove or replace current `dev` setup readiness modules.
- Tests cover backend validation categories, WebUI gating, readiness rendering, first-chat recovery, and first-source milestone behavior.
