# First-Time Model Readiness Setup Design

Date: 2026-05-18
Owner: Codex brainstorming session
Status: Ready for user review
Backlog: TASK-426

## Summary

This design extends first-time setup so new users can choose and provision the
models they need before they start ingesting media, searching, running RAG, or
transcribing audio.

The recommended direction is a unified readiness wizard:

- Backend `/setup` remains the authoritative setup and recovery surface.
- The WebUI gets a native first-run setup screen backed by the same
  `/api/v1/setup/*` endpoints.
- Curated profiles come first, with advanced provider/model controls available
  behind disclosure.
- Model downloads and package installs require an explicit `Provision now`
  action.
- TTS remains visible but secondary to transcription readiness.

The goal is not to create another setup system. The goal is to organize the
existing setup, config, audio bundle, install-plan, provider, and readiness
pieces into a first-run path that gets users to a known useful state.

## User-Approved Direction

During brainstorming, the approved product decisions were:

- Backend `/setup` is authoritative, but first-time WebUI setup must expose the
  same flow natively when setup is required.
- The WebUI implementation should call the same `/api/v1/setup/*` endpoints and
  provide a fallback link to backend `/setup`.
- V1 should use curated "ready for chat/RAG/transcription" profiles first, with
  advanced model/provider pickers available.
- Chat setup should support hosted providers, local OpenAI-compatible endpoints,
  and explicit skip.
- Provisioning must use a separate `Provision now` button after review.
- TTS should stay visible but secondary.
- Local first-run may be unauthenticated only while `/setup` is still required,
  matching the backend guard.
- After setup is complete, equivalent provisioning/setup controls become
  admin-only and use admin setup endpoints.
- Readiness should cover chat, embeddings/RAG, and speech readiness
  (transcription primary, TTS secondary), plus restart/config/permission
  overlays.

## Goals

1. Let a new self-hosted user choose useful defaults for chat, embeddings, and
   speech during first-time setup.
2. Let users pre-download or provision local model assets before the first
   ingest/search/transcription workflow.
3. Keep backend setup and WebUI setup backed by one shared API contract.
4. Preserve explicit user consent before downloads, package installs, config
   writes, and expensive verification.
5. Make setup completion depend on clear readiness, warnings, or explicit skips
   rather than hidden assumptions.
6. Support both hosted APIs and local OpenAI-compatible endpoints for chat
   defaults.
7. Preserve admin/operator control after first setup is complete.

## Non-Goals

- Do not replace the backend `/setup` page in V1.
- Do not create a second WebUI-only configuration system.
- Do not silently download model assets during page load or profile selection.
- Do not require every lane to be ready before setup can be completed; explicit
  skips are valid.
- Do not make regular users able to provision server-wide models after setup.
- Do not redesign every settings page or model-management surface.
- Do not guarantee GPU-accelerated stacks work in stock Docker without explicit
  Docker/runtime support.

## Current Repo Foundation

The repo already has several pieces that should be reused.

### Backend Setup

Relevant files:

- `tldw_Server_API/app/api/v1/endpoints/setup.py`
- `tldw_Server_API/app/api/v1/schemas/setup_schemas.py`
- `tldw_Server_API/app/core/Setup/setup_manager.py`
- `tldw_Server_API/app/core/Setup/install_schema.py`
- `tldw_Server_API/app/core/Setup/install_manager.py`
- `tldw_Server_API/app/core/Setup/audio_bundle_catalog.py`
- `tldw_Server_API/app/core/Setup/audio_readiness_store.py`
- `tldw_Server_API/app/api/v1/API_Deps/setup_deps.py`
- `tldw_Server_API/app/Setup_UI/setup.html`
- `tldw_Server_API/app/static/setup/js/setup.js`

Existing behavior:

- `/setup` is served by the API server when `enable_first_time_setup=true` and
  `setup_completed=false`.
- `/api/v1/setup/status` reports setup state and placeholder fields.
- `/api/v1/setup/config` reads and writes known `config.txt` keys.
- `/api/v1/setup/complete` can submit an install plan through a background
  task.
- `/api/v1/setup/audio/provision` runs existing audio bundle provisioning, and
  admin audio provisioning endpoints already exist after setup completion.
- Setup access is local-first and guarded by setup-specific dependencies.
- Audio recommendations, provisioning, verification, readiness, and pack import
  already exist.
- `InstallPlan` already supports STT, TTS, and embeddings.
- `install_manager` already supports explicit dependency installs and model
  downloads with skip flags such as `TLDW_SETUP_SKIP_DOWNLOADS` and
  `TLDW_SETUP_SKIP_PIP`.

### WebUI First-Run Setup

Relevant files:

- `apps/tldw-frontend/pages/setup.tsx`
- `apps/packages/ui/src/routes/option-setup.tsx`
- `apps/packages/ui/src/components/Option/Onboarding/OnboardingWizard.tsx`
- `apps/packages/ui/src/components/Option/Onboarding/OnboardingConnectForm.tsx`
- `apps/packages/ui/src/components/Option/Setup/AudioInstallerPanel.tsx`
- `apps/packages/ui/src/components/Option/Setup/hooks/useAudioInstaller.ts`
- `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- `apps/packages/ui/src/services/tldw/openapi-guard.ts`

Existing behavior:

- The WebUI `/setup` route currently renders connection onboarding, not the
  backend first-time config wizard.
- The WebUI onboarding form can store selected chat provider/model preferences,
  but it does not own server-side model provisioning.
- The shared UI already has an admin audio installer panel backed by setup admin
  endpoints.

### Config Surfaces

Relevant sections in `tldw_Server_API/Config_Files/config.txt`:

- `[Setup]`: first-time setup flags.
- `[API]`: hosted provider models, custom OpenAI-compatible provider settings,
  `default_api`, and `default_api_for_tasks`.
- `[Local-API]`: local provider endpoint/model settings such as Ollama and
  vLLM.
- `[Embeddings]`: `embedding_provider`, `embedding_model`,
  `embedding_api_url`, local model directories, and trusted HF models.
- `[STT-Settings]`: default transcriber, batch/streaming models, NeMo/Whisper
  settings, and Qwen ASR settings.
- `[TTS-Settings]`: default TTS provider, voice/model defaults, and local voice
  asset paths.

## Proposed Architecture

First-time setup should be one backend-owned readiness system with two UI
shells.

### Backend `/setup`

Backend `/setup` remains the authoritative recovery path. It continues to be
available directly from the API server when first-time setup is required. It can
also be used when the WebUI is unavailable or a user needs a lower-level config
view.

### Native WebUI Setup

The WebUI gets a first-run setup screen that consumes the same setup APIs. It
should not duplicate business rules. Its responsibilities are presentation,
routing, progress display, and user confirmation.

### Setup Readiness Profiles

Introduce a backend concept for setup readiness profiles. A profile is a
curated bundle of choices for the lanes a new user cares about:

- Chat defaults: hosted provider or local OpenAI-compatible endpoint, default
  model, optional test request.
- Embeddings/RAG: embedding provider/model, local download plan or remote
  config, smoke embedding test.
- Speech: STT bundle/profile, provisioning plan, verification target, and
  secondary TTS defaults/readiness metadata.

Profiles should be machine-aware but transparent. The backend returns the
recommendation, reasons, estimates, and warnings. The UI shows those details
before provisioning.

## Readiness Lanes

Canonical lane IDs should be:

- `chat`
- `embeddings_rag`
- `speech`

The `speech` lane owns transcription as the primary readiness target and TTS as
secondary readiness metadata. TTS only blocks completion when the selected
profile explicitly requires it.

All lanes should use a shared status vocabulary. Status values describe the
lane outcome only:

- `not_configured`: no useful selection has been made yet.
- `previewed`: a selection exists and config/provisioning changes have been
  previewed, but the user has not clicked `Provision now`.
- `provisioning`: approved provisioning is running.
- `ready`: verification passed.
- `ready_with_warnings`: the lane is usable, but follow-up work is recommended.
- `failed`: verification or provisioning failed and the lane is not usable.
- `blocked`: setup cannot proceed for this lane until an external prerequisite
  changes, such as missing permission, disabled downloads, disabled package
  installs, missing OS dependency, or insufficient disk.
- `skipped`: the user explicitly chose not to configure this lane now.

The frontend may group or label these statuses differently for presentation, but
the backend should make the underlying state explicit enough that completion
gating is not inferred from display text.

Cross-lane overlays should be reported separately from lane status:

- `restart_required`: config was changed and the server must restart before the
  new configuration can be trusted.
- `requires_admin`: the current user/session can view status but cannot perform
  server-wide provisioning.
- `remote_setup_blocked`: the WebUI cannot use first-run setup APIs because the
  backend local/proxy setup guard rejected the request.
- `network_unavailable`, `downloads_disabled`, and `package_installs_disabled`:
  provisioning cannot continue until the operator changes environment or setup
  flags.

A lane must not be displayed as fully `ready` solely because config was written
when `restart_required` is true. Until the restarted server is verified, show
the lane as `previewed`, `ready_with_warnings`, or another explicit state plus
the restart overlay.

### Chat Defaults

Purpose:
Make chat/model calls usable when the user wants hosted or local chat on day
one.

Choices:

- Hosted provider: provider, API key, default model.
- Local endpoint: base URL and model name.
- Skip for now: chat remains limited until configured later.

Readiness:

- `ready`: a tiny test request succeeds, or the local endpoint/model validates.
- `ready_with_warnings`: provider is configured but optional model discovery or
  fallback validation is incomplete.
- `skipped`: user explicitly skipped chat setup.
- `failed`: provider credentials, endpoint reachability, or model validation
  failed.

Hosted-provider API keys are write-only setup inputs. Preview responses may
report `present`, `absent`, or `placeholder`, but must never echo the submitted
secret. If implementation planning cannot reuse a safe existing secret write
path, the WebUI should show manual `.env` or provider-secret instructions and
mark the chat lane `blocked` or `skipped` instead of writing raw keys to an
unsafe config surface.

### Embeddings And RAG

Purpose:
Make media ingestion and RAG search useful without requiring the user to later
discover missing embeddings.

Choices:

- Hosted embedding provider where configured.
- Local Hugging Face model download.
- Local/custom trusted HF model.
- ONNX/local path where supported.
- Skip for now.

Readiness:

- `ready`: selected provider/model can produce a tiny embedding or load
  successfully.
- `ready_with_warnings`: config is present but a non-critical capability is
  missing, for example optional cache or model-discovery metadata.
- `skipped`: user explicitly skips embeddings setup.
- `failed`: selected model cannot load/respond, dependencies are missing, or
  download failed.

Mutation rule:
Profile selection should preview config changes. Config changes and downloads
must not happen until the user confirms and clicks `Provision now`.

Custom or trusted Hugging Face model entries belong in advanced mode only. The
curated default profiles should use allowlisted known-safe choices and must not
enable trust-remote-code-style behavior without a separate explicit
acknowledgement.

### Speech: Transcription Primary

Purpose:
Make media/audio ingestion workflows usable when users start transcribing files
or generating searchable transcripts.

Choices:

- Reuse existing audio bundle recommendations.
- Curated profiles: light, balanced, performance.
- Advanced STT engine/model override where needed.

Readiness:

- `ready`: existing audio verification reports primary STT usable.
- `ready_with_warnings`: STT is usable but secondary audio pieces need attention.
- `skipped`: user explicitly skips speech setup.
- `failed`: primary STT path is unusable.

### Speech: Secondary TTS

Purpose:
Keep voice output readiness visible without making it the headline for
ingestion/RAG.

Behavior:

- Show the selected TTS provider/voice inside the speech lane.
- Include TTS provisioning in curated speech profiles where existing bundles do
  so.
- Keep TTS readiness secondary in copy and visual hierarchy.
- Let TTS failures create warnings when transcription is otherwise ready, unless
  the selected profile explicitly requires TTS.

## API Design

The exact endpoint names can be finalized during implementation, but V1 should
extend the existing setup API rather than creating a parallel namespace.

Recommended additions:

- `GET /api/v1/setup/readiness/profiles`
  - Returns machine profile, curated readiness profiles, recommendation order,
    existing readiness, and setup/admin availability.
- `POST /api/v1/setup/readiness/preview`
  - Accepts selected profile and advanced overrides.
  - Returns config changes, install plan, disk estimates, warning list, and
    verification targets.
- `POST /api/v1/setup/readiness/provision`
  - Requires explicit confirmation.
  - Applies approved config changes and starts approved provisioning work.
  - Returns quickly with an operation ID, status snapshot, or polling URL; the
    WebUI must not depend on a long-held HTTP request for model downloads or
    package installation.
- `GET /api/v1/setup/readiness/status`
  - Returns lane-level readiness, install progress, last verification, warnings,
    and restart-required state.
- `POST /api/v1/setup/readiness/verify`
  - Runs lane-specific verification for chat, embeddings, speech, or all lanes.

Admin equivalents may either be separate admin paths or resolved through the
existing setup guard layer. The important contract is behavioral:

- setup-required local first-run can use local unauthenticated setup access
- post-setup use is admin-only
- regular users never get server-wide provisioning controls

V1 may reuse the existing setup install status machinery for first-run
provisioning. If post-setup admin provisioning grows pause/resume/drain,
multi-operation history, quotas, or admin operations beyond the current shared
installer behavior, implementation planning should use the Jobs system rather
than inventing a second durable operation runner.

## WebUI Design

### Entry

When the WebUI detects a setup-required server, it should show a native setup
screen instead of only the current connection onboarding. The screen should
include a fallback link to the backend `/setup` page.

The native WebUI path is available only when the configured API base URL can
satisfy the backend setup guard: local first-run access while setup is required,
or authenticated admin access when remote setup has explicitly been enabled.
If the guard rejects WebUI requests because of host, proxy, origin, or remote
access constraints, the WebUI should keep the fallback `/setup` link visible and
show a concise operator-facing explanation rather than asking the user to weaken
remote setup protections from the browser.

When setup is complete:

- admins can open the model readiness screen from settings/admin
- non-admin users see a clear admin-required state

### Layout

Use a compact readiness dashboard rather than a long generic config form.

Primary regions:

1. Profile picker
2. Readiness lanes
3. Preview and `Provision now`
4. Verification progress
5. Completion and next actions

Lanes:

- Chat
- Embeddings/RAG
- Speech, with transcription primary and TTS visually secondary

### Profile Picker

Default profiles should be curated and explain the tradeoff:

- Local Light: low disk/memory, useful for constrained machines.
- Local Balanced: recommended default for most local installs.
- Local Performance: larger local footprint, better quality/throughput.
- Hosted Plus Local Speech: hosted chat/embedding defaults with local speech.
- Advanced Custom: no single recommendation, user chooses each lane.

The backend may alter labels and recommendations based on machine profile,
available disk, platform, GPU/Apple Silicon signals, and existing config.

### Advanced Mode

Advanced mode should expose exact provider/model settings without changing the
default experience:

- Chat provider, write-only API key field, model, local endpoint URL, local
  model name.
- Embedding provider, model ID, local path/ONNX option, trusted custom HF repo
  behind explicit acknowledgement.
- STT engine/model/profile.
- TTS provider/voice/model.

Advanced mode must preserve the same preview and explicit provisioning gate.

## Data Flow

1. WebUI checks `/api/v1/setup/status`.
2. If setup is required and the backend setup guard allows the request, WebUI
   loads native setup using setup-first-run access.
3. If the setup guard blocks the WebUI path, WebUI shows the fallback backend
   `/setup` link and explains the local/proxy/admin requirement.
4. WebUI fetches readiness profiles and current readiness state.
5. User selects a profile or advanced overrides.
6. WebUI asks backend for a preview.
7. Backend returns:
   - proposed config updates
   - install plan
   - download/package estimates
   - verification targets
   - warnings and blockers
   - restart-required prediction
8. User clicks `Provision now`.
9. Backend persists approved config changes and starts approved installs or
   downloads as a pollable operation.
10. WebUI polls readiness/install status.
11. User runs verification, or verification runs after provisioning only when
   the profile preview already disclosed that check and the check is cheap,
   local, and non-mutating. Hosted model calls, expensive local model loads, and
   long audio checks require an explicit verification action.
12. If config changes require restart, WebUI keeps the restart overlay visible
    and does not present affected lanes as fully verified until the restarted
    server reports readiness.
13. User marks setup complete through the existing `/api/v1/setup/complete`
    flow, or a revised equivalent if implementation planning determines the
    existing endpoint must be extended. Completion is allowed only when all
    lanes are ready, ready with warnings, or explicitly skipped.

## Completion Rules

Setup completion should be allowed when:

- every lane is `ready`, `ready_with_warnings`, or `skipped`
- config writes are complete
- the UI has clearly called out any required restart
- skipped critical lanes include consequences and next actions, for example
  "RAG search will be limited until embeddings are configured"

A `failed` lane is never treated as complete by itself. The user must either
remediate it until it reaches `ready` or `ready_with_warnings`, or explicitly
change that lane to `skipped`. A `blocked` lane follows the same rule: the
external blocker must be resolved, or the user must explicitly skip the lane.

Setup completion should not be blocked by:

- skipped hosted chat
- skipped embeddings
- secondary TTS warnings when transcription is ready
- missing optional local acceleration

Setup completion should be blocked by:

- pending unconfirmed config changes
- active provisioning without a final status
- unknown lane state after a failed status fetch
- permission mismatch

## Error Handling

The flow should make failures recoverable.

Expected failures:

- Network or downloads disabled.
- Package installs disabled.
- Missing FFmpeg, eSpeak, CUDA, MLX, or other OS/runtime prerequisite.
- Hosted provider key invalid.
- Local endpoint unreachable.
- Local endpoint reachable but model missing.
- Embedding model download succeeds but smoke test fails.
- STT model installed but verification fails.
- Config write requires restart.
- User lacks admin permission after setup completion.

Rules:

- Fail the smallest lane possible.
- Preserve user choices after failure.
- Avoid raw stack traces or endpoint strings in primary UI.
- Show technical details behind disclosure.
- Provide a next action: retry, safe rerun, verify again, edit selection, skip,
  open backend `/setup`, or ask admin.
- Never start another download/install automatically after failure without a new
  user action.

## Security And Permissions

Security rules:

- While setup is required, local first-run access can remain unauthenticated
  under the existing setup guard.
- Remote setup remains controlled by the existing remote setup flags and
  allowlist/denylist behavior.
- After setup is complete, server-wide setup/provisioning actions require admin
  permissions.
- API keys and secrets must never be returned to the WebUI after save.
- Preview payloads may say a secret is present, absent, or placeholder, but
  must not reveal values.
- Provider tests must sanitize provider errors before display.
- Hosted provider secrets should prefer existing `.env` or secret-handling
  mechanisms. Raw secrets should not be written to `config.txt` unless the
  current setup writer already treats that key as a write-only masked secret.
- Regular users may see personal model preferences elsewhere, but this design is
  for server readiness and provisioning, not per-user preference editing.

## Testing Plan

### Backend Tests

- Profile recommendation unit tests for CPU, Apple Silicon, CUDA, no network,
  low disk, hosted preference, and local preference.
- Preview tests proving config changes are explicit and limited to known keys.
- Provision tests proving no download/install starts before `Provision now`.
- Provision tests proving the endpoint returns a pollable status/operation and
  does not require a long-held HTTP request.
- Install-plan tests for embeddings, STT, and secondary TTS profile expansion.
- Readiness tests for chat success/failure/skip.
- Readiness tests for embeddings success/failure/skip.
- Audio readiness tests using existing audio bundle verification paths.
- Permission tests for local setup-required access and post-setup admin-only
  access.
- Permission tests for WebUI fallback behavior when the setup guard rejects a
  host/proxy/remote request.
- Sanitization tests for provider/model verification errors.
- Secret tests proving preview/status payloads never echo provider API keys.
- Advanced trusted-model tests proving acknowledgement is required before a
  custom trusted HF model can be provisioned.

### Frontend Tests

- WebUI setup-required state renders the native readiness screen.
- Fallback link to backend `/setup` is always present.
- Curated profile selection updates lane previews.
- Advanced overrides update preview without mutating config.
- `Provision now` is required before provisioning starts.
- Lane status rendering covers ready, warning, failed, blocked, and skipped.
- Overlay rendering covers restart-required, admin-required, remote setup
  blocked, downloads disabled, and package installs disabled.
- Non-admin post-setup users see admin-required state.
- Admin post-setup users can load the admin readiness/provisioning path.

### End-To-End Tests

- First-run local profile with hosted chat skipped, local embeddings selected,
  and speech profile provisioned or safely skipped.
- Hosted provider profile with mocked valid/invalid key behavior.
- Local endpoint profile with reachable/unreachable endpoint behavior.
- Interrupted provisioning resumes from persisted status.
- Setup completion accepts ready-with-warnings and explicit skips.
- Restart-required flow keeps affected lanes from appearing fully verified until
  the restarted backend is checked.

## Implementation Staging Guidance

This design should be implemented in staged slices:

1. Backend readiness/profile API contract and tests.
2. Backend preview/provision wrappers using existing install-plan and config
   writers.
3. Backend chat and embeddings verification helpers.
4. Shared WebUI setup client.
5. Native WebUI first-run readiness screen.
6. Admin/post-setup entry point and permission states.
7. Backend `/setup` compatibility improvements if needed.
8. End-to-end verification and docs.

Each slice should have its own implementation plan and Backlog.md task if it is
large enough to review independently.

## Open Questions For Implementation Planning

These questions should be decided at the start of implementation planning so
endpoint naming, persistence, and verification boundaries do not diverge across
independent slices.

- Whether readiness status should reuse `setup_install_status.json`, extend
  `audio_readiness_store`, or introduce a small general readiness store.
- Whether admin readiness endpoints should mirror `/setup/admin/audio/*` naming
  or use a shared dependency that switches behavior based on setup state.
- Whether chat test requests should use existing chat completion APIs directly
  or a lightweight provider-validation helper.
- Whether embedding smoke tests should use `/api/v1/embeddings` or lower-level
  provider/load helpers.
- Whether restart-required overlays should be calculated per config key or
  returned as a conservative profile-level flag in V1.
- Whether first-run provisioning can stay on the existing setup installer status
  path, and exactly when post-setup admin provisioning should graduate to Jobs.

## Definition Of Done For The Spec

- Design records the approved backend-authoritative plus native-WebUI direction.
- Design covers chat, embeddings/RAG, transcription, and secondary TTS.
- Design includes explicit `Provision now` behavior.
- Design covers permissions before and after setup completion.
- Design covers readiness, error handling, and testing.
- Design is linked from `TASK-426`.
