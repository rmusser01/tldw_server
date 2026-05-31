# First-Time Solo User Onboarding PRD Design

Date: 2026-05-31
Status: Ready for user review
Owner: Codex brainstorming session
Backlog: TASK-487

## Summary

Create one cohesive first-time onboarding experience for solo users, from a fresh checkout or download to a successful first chat in the WebUI.

The product should no longer feel like a set of separate setup systems: Getting Started docs, `make quickstart`, `tldw-setup`, backend `/setup`, WebUI connection onboarding, provider configuration, model readiness, and first-value guidance should all describe and enforce the same lifecycle.

The primary experience is a WebUI-led progressive setup wizard. The pre-WebUI path gets the user to a running app through clear Docker single-user or local single-user setup choices. Once the WebUI is reachable, the focused setup shell guides the user through security, provider/model configuration, ingest defaults, audio/STT/TTS defaults, optional advanced settings, and a real first chat. Normal navigation stays hidden until first chat succeeds or the user explicitly skips.

## Product Decisions

The approved direction is a unified WebUI-led funnel:

- Present Docker single-user and local single-user as peer first-time solo setup choices.
- Keep multi-user as an intentional exit ramp to the multi-user guide and operator checklist.
- Preserve `make quickstart` as the current public entrypoint, while defining the product requirement as one obvious start command or command sequence per setup path.
- Make the WebUI the primary guided solo-user onboarding surface after the app is reachable.
- Keep backend `/setup` as the operator and recovery surface, backed by the same setup and readiness APIs as the WebUI.
- Use backend-authoritative setup, readiness, skip, and completion state.
- Hide global app navigation during first-run setup until first chat succeeds or setup is explicitly skipped.
- Make hosted API-key setup the recommended fastest route to first chat.
- Support hosted providers, local Ollama, local llama.cpp, and generic OpenAI-compatible endpoints as first-class provider options.
- For V1, the provider screen should be generated from the backend's existing supported chat-provider catalog rather than hardcoding one provider. At minimum, it must cover the already-supported commercial chat providers and local/OpenAI-compatible providers listed in the project guide: OpenAI, Anthropic, Cohere, DeepSeek, Google, Groq, HuggingFace, Mistral, OpenRouter, Qwen, Moonshot, Z.AI, Ollama, llama.cpp, Kobold.cpp, Oobabooga, TabbyAPI, vLLM, Aphrodite, and Custom OpenAI-compatible.
- Guide and verify local model runtimes, but do not install Ollama or llama.cpp in this PRD slice.
- Collect provider keys and local endpoint settings in the UI so normal first-time users do not edit `.env` or `config.txt`.
- Require a real successful model response before setup can be marked complete.
- Treat browser extension setup as a post-onboarding add-on.
- Make adding a first source the first guided milestone after onboarding completes.

## Problem

First-time setup is currently fragmented across several entrypoints and mental models:

- Documentation has canonical profile work, but users still have to choose between many setup references and concepts.
- `/setup` exists as a backend setup surface, while the WebUI has separate connection onboarding.
- The WebUI first-run flow mixes connection, demo mode, auth mode, model choice, quick ingest, and broader app chrome before the user has one successful loop.
- Provider setup often implies manual config or env editing, even though the WebUI should be able to collect common settings.
- Single-user API-key auth is a visible hurdle even when the bundled WebUI and API are launched together.
- Existing setup flows can verify installation or search readiness without proving the user's desired first value: a real chat response.
- Optional feature setup, especially audio and model readiness, competes with the shortest path to first value.

The result is high friction: a solo user must understand implementation details before they can prove the product works for them.

## Goals

1. Let a first-time solo user choose between Docker single-user and local single-user setup as peer paths.
2. Get a solo user from checkout/download to WebUI onboarding through one obvious command sequence per path.
3. Let the user configure common first-run settings through the WebUI instead of manually editing `.env` or `config.txt`.
4. Make hosted provider setup the fastest happy path to first chat while preserving local model alternatives.
5. Hide single-user API-key auth in the bundled WebUI path when it can be handled automatically.
6. Make first-run progress backend-authoritative and consistent across WebUI, `/setup`, CLI verification, and restart.
7. Require a real successful chat response before marking onboarding complete.
8. Keep optional advanced configuration visible but deferrable.
9. Make the next guided milestone after first chat be adding a first source.
10. Clean up or demote conflicting onboarding entrypoints so there is one user-facing story.

## Non-Goals

- Do not make multi-user setup part of the solo-user completion gate.
- Do not replace backend `/setup`; keep it as an operator and recovery surface.
- Do not require browser extension setup before first chat.
- Do not require RAG, embeddings, storage path tuning, audio, STT, or TTS readiness before first chat.
- Do not silently install Ollama, llama.cpp, or other local model runtimes.
- Do not guarantee GPU-accelerated media or speech stacks work in the default first-run path.
- Do not redesign the full app shell or every settings page in this slice.
- Do not remove manual config editing for advanced operators; make it unnecessary for the normal first-time path.

## Target Users

### Primary: Solo Self-Hoster

One person wants to run tldw for themselves on a local machine or homelab. They may be comfortable with a terminal, but they should not need to understand internal config structure, API-key auth mechanics, provider routing, or setup flags before getting a first chat response.

### Secondary: Local Power User

One person prefers local Python install, host-visible files, local model runtimes, and direct control. Local single-user setup must feel like a peer path, not an afterthought.

### Out Of Critical Path: Multi-User Operator

A user choosing multi-user/shared-server setup should be routed to the multi-user setup guide and operator checklist. Multi-user setup has admin bootstrap, JWT/session secrets, database, and production security decisions that should not be collapsed into the solo-user wizard.

## End-To-End Journey

### Phase 1: Start

The user starts from docs, README, or an app landing entrypoint and sees three setup intents:

1. Solo, Docker
2. Solo, local install
3. Multi-user or shared server

Docker single-user and local single-user are peer options. Docker can be labeled as the easiest containerized path, but local install must remain first-class.

For both solo paths:

- show prerequisites in plain language;
- offer one obvious command sequence;
- prepare, start, and verify with matching semantics;
- print or reveal the WebUI URL;
- classify failures with recovery guidance;
- hand off to the same WebUI onboarding flow.

If the user chooses multi-user:

- explain that the solo wizard is not the right path;
- route to the multi-user setup guide and operator checklist;
- explain the extra decisions: admin bootstrap, JWT/session secrets, database choice, production access, and shared-user security.

### Phase 2: WebUI Progressive Setup

Before onboarding completion, the WebUI uses a focused setup shell. It should expose only:

- setup progress;
- the current step and next action;
- diagnostics and logs links;
- docs/recovery links;
- theme and accessibility controls;
- explicit skip/defer actions where safe.

It should not show the normal app header, route rail, chat shortcuts, global search, or broad feature navigation.

The wizard guides the user through sequential screens:

1. Setup path and server verification
2. Privacy and security
3. Chat and providers
4. Ingest defaults
5. Audio, STT, and TTS
6. Optional advanced setup
7. First chat

Each step has:

- one primary action;
- safe defaults;
- clear validation status;
- a skip or defer path where allowed;
- plain-language errors;
- diagnostics behind disclosure rather than exposed raw failures.

### Phase 3: First Chat Completion

The user chooses a default provider/model and sends a real chat request. Setup is marked complete only after the backend records a successful response from the selected model.

Provider validation alone is not completion. If the user chooses to skip first chat, the backend records `skipped`, not `completed`.

### Phase 4: Next Milestone

After first chat completion, the normal app shell is revealed. The next guided milestone is adding a first source. Once a source is added, the product should guide the user toward asking a grounded question over that source.

Browser extension setup, audio deep setup, advanced RAG tuning, storage path tuning, admin configuration, and multi-user operations appear as follow-up milestones or settings surfaces, not first-run blockers.

## Wizard Requirements

### 1. Setup Path And Server Verification

Purpose: confirm how the user is running tldw and whether the WebUI can reach the backend.

Requirements:

- Present Docker single-user and local single-user as peer solo setup paths.
- Present multi-user/shared server as an exit path to the multi-user guide.
- Detect whether the WebUI is connected to the expected backend.
- Distinguish frontend origin from backend/API origin in user-facing language.
- Detect bundled single-user WebUI/API mode where possible.
- Use generated single-user API-key auth automatically where the bundled path supports it.
- Expose manual API-key entry only as diagnostics, advanced setup, or non-bundled connection recovery.

Recovery categories:

- backend unreachable;
- wrong URL or frontend URL entered as backend URL;
- auth failed;
- setup not enabled or already completed;
- CORS/proxy mismatch;
- port conflict or service unhealthy.

### 2. Privacy And Security

Purpose: set safe solo defaults before the user adds secrets or exposes services.

Requirements:

- Confirm solo single-user mode.
- Explain local-only versus LAN/reverse-proxy access expectations.
- Keep remote setup access disabled by default.
- Warn before enabling remote setup access.
- Show whether the current browser is local, LAN, or remote from the backend's perspective when available.
- Confirm how provider secrets are stored and masked.
- Explain that manual `.env` and `config.txt` edits are advanced recovery paths.
- Surface production/security warnings if the user is on a non-local or advanced deployment path.

### 3. Chat And Providers

Purpose: configure one or more providers and select the default provider/model for first chat.

Requirements:

- Let users select all providers they have access to, not only one provider.
- Support the existing chat-provider catalog in the setup UI. V1 minimum coverage is OpenAI, Anthropic, Cohere, DeepSeek, Google, Groq, HuggingFace, Mistral, OpenRouter, Qwen, Moonshot, Z.AI, Ollama, llama.cpp, Kobold.cpp, Oobabooga, TabbyAPI, vLLM, Aphrodite, and Custom OpenAI-compatible.
- Support hosted providers with API-key entry in the UI.
- Support local Ollama endpoint configuration.
- Support local llama.cpp endpoint configuration.
- Support generic OpenAI-compatible endpoint configuration.
- Collect local endpoint values such as host/IP, port, base URL, model name, and optional auth token.
- Save and mask keys for all supported hosted providers.
- Validate hosted provider credentials where practical. Providers without stable preflight validation can rely on syntax/presence checks plus the first-chat verification when selected as the default.
- Verify local endpoint reachability and API shape.
- Mask saved secrets after entry.
- Persist provider configuration through backend setup APIs.
- Let users choose a default provider/model for first chat.
- Show validation states: saved, reachable, auth failed, invalid key, unsupported API shape, model unavailable, restart required, skipped.
- Make hosted API-key setup the recommended fastest path, without burying local alternatives.

Non-requirements:

- Do not install local model runtimes.
- Do not download local model weights unless a later implementation plan explicitly adds user-approved model provisioning.

### 4. Ingest Defaults

Purpose: set practical ingestion defaults before the user adds sources after first chat.

Requirements:

- Configure common upload and source behavior in plain language.
- Let users review local path access rules if local file ingestion is enabled.
- Offer safe file handling defaults.
- Offer friendly chunking defaults without exposing every advanced parameter.
- Let users select metadata defaults or leave automatic defaults.
- Do not require a first source before setup completion.
- Defer advanced ingestion, connector, and bulk processing settings to post-onboarding settings.

### 5. Audio, STT, And TTS

Purpose: make speech features visible and configurable without blocking first chat.

Requirements:

- Offer a simple audio setup decision: configure now, use defaults, or skip.
- Present CPU/default, accelerated/local, and provider-backed choices in plain language.
- Let users select default STT/transcription behavior where available.
- Let users select TTS provider/voice defaults where available.
- Show when a choice requires external prerequisites, model files, package installs, container rebuild, or restart.
- Do not make audio/STT/TTS readiness mandatory for first chat.
- Route deeper audio setup to the existing audio setup docs and backend setup/recovery surfaces.

### 6. Optional Advanced Setup

Purpose: let power users configure important advanced settings without forcing every first-time user through them.

Requirements:

- Include optional RAG/embeddings setup.
- Include optional storage path configuration.
- Explain why each advanced area matters.
- Allow configure now, skip, or defer.
- Do not block first chat on these settings.
- Make deferred items easy to find later in settings/admin surfaces.

### 7. First Chat

Purpose: prove the setup works with the selected default model.

Requirements:

- Send a real chat request using the selected default provider/model.
- Show the actual model response.
- Record first chat success in backend setup state.
- Mark setup complete only after successful response and required screen acknowledgements.
- If the request fails, show recoverable failure categories and keep the user in the wizard.
- If the user explicitly skips, record `skipped` and reveal the normal app shell with a clear incomplete/setup-skipped status.

## State Model

Setup state should be backend-authoritative.

Recommended top-level states:

- `not_started`: setup has not begun.
- `in_progress`: user is moving through the wizard.
- `blocked`: setup cannot continue until an external prerequisite changes.
- `skipped`: user explicitly skipped first-run setup or first chat.
- `first_chat_complete`: a real chat response succeeded.
- `completed`: first chat succeeded and required setup screens were acknowledged.

Frontend state is allowed for:

- current unsaved form values;
- temporary resume affordances;
- dismissed tips;
- local UI expansions/collapses;
- non-authoritative cached display.

Frontend state must not be the source of truth for whether first setup is complete.

## Backend And API Requirements

The PRD should be implemented through shared backend setup and readiness APIs consumed by both the WebUI and backend `/setup`.

Required capabilities:

- Return setup status, completion state, skip state, and current step.
- Return auth mode, bundled single-user auth availability, and whether manual auth is required.
- Return setup path metadata and multi-user exit guidance.
- Read and write supported first-run configuration settings.
- Persist provider API keys and local endpoint settings securely.
- Mask secrets in responses.
- Validate hosted provider keys where practical.
- Validate local provider endpoints.
- Return provider/model candidates for first chat.
- Persist selected default provider/model.
- Persist ingest defaults.
- Persist audio/STT/TTS defaults where supported.
- Persist optional RAG/storage choices when configured.
- Execute or record first chat verification.
- Mark setup complete only through valid transitions.
- Return diagnostics in safe, user-facing categories with optional raw detail for operator surfaces.

`/setup` can expose lower-level operator controls, but it should use the same state names, validation categories, and readiness model.

### Setup API Access Boundary

Setup write APIs are allowed only inside the setup trust boundary.

Requirements:

- Setup write APIs are enabled only while setup is required or in progress.
- Before setup completion, unauthenticated setup writes are allowed only through the setup access guard: localhost and bundled same-origin WebUI by default.
- Remote setup access is disabled by default and requires explicit operator opt-in before any remote browser can write provider secrets or config.
- If remote setup is enabled, allowlist/denylist rules and setup diagnostics must be visible before secrets are collected.
- After setup completion, equivalent config, provider, and provisioning writes require authenticated admin/system-configure permissions.
- Secret-bearing setup responses must always return masked values, regardless of auth phase.
- Failed setup access checks must explain whether the issue is non-local access, setup already completed, setup disabled, missing admin permission, or allowlist/denylist rejection.

## CLI And Documentation Requirements

The docs and CLI should describe the same lifecycle as the WebUI.

Requirements:

- Getting Started should route solo users into a peer choice between Docker single-user and local single-user.
- Multi-user should be an explicit third choice that routes to the multi-user guide.
- Public setup docs should use the same stages: prepare, start, verify, open WebUI, complete first chat.
- `make quickstart` should remain valid as the current default entrypoint.
- Each solo path should have an obvious command sequence and matching verification behavior.
- CLI verification should not claim setup success if the WebUI first-chat requirement is incomplete; it can report installation readiness separately.
- CLI output should print the WebUI URL and next action.
- Failure output should classify common setup failures in plain language.
- Manual `.env` and `config.txt` editing should be documented as advanced or recovery behavior, not normal first-run behavior.

## Cleanup Requirements

The implementation should remove or demote conflicting onboarding surfaces.

Requirements:

- `/` resolves users based on backend setup state:
  - setup required or in progress -> focused setup shell;
  - setup skipped -> normal app with setup-skipped recovery status;
  - setup complete -> normal app;
  - degraded/unreachable -> recovery state.
- `/setup` identifies itself as backend/operator setup and recovery, not the primary solo-user wizard.
- Existing WebUI connection onboarding is folded into the unified first-run wizard.
- Existing setup test harnesses are removed from user-facing navigation or clearly marked internal.
- Docs that repeat setup commands outside canonical onboarding areas are redirected, trimmed, or linked into the unified lifecycle.
- Provider setup docs and settings pages should link back to the unified setup concepts.
- First-run copy should stop presenting multiple competing definitions of success.

## Error Handling And Recovery

Every wizard step should show one primary next action and clear recovery guidance.

Required failure categories:

- missing dependency;
- Docker unavailable;
- Python/venv unavailable;
- backend unreachable;
- WebUI unreachable;
- auth failed;
- bundled auth unavailable;
- provider key invalid;
- provider quota/rate limit;
- local provider endpoint unreachable;
- local provider unsupported API shape;
- selected model unavailable;
- config write failed;
- permission denied;
- restart required;
- setup access blocked;
- storage path invalid;
- external prerequisite required.

Raw exception text, stack traces, filesystem internals, and secrets must not be shown in primary user-facing errors. Operator diagnostics may expose more detail behind guarded disclosure.

## Success Metrics

Target metrics:

- Clean checkout/download to first successful chat in under 10 minutes for Docker single-user when prerequisites are present.
- Clean checkout/download to first successful chat in under 10 minutes for local single-user when prerequisites are present.
- WebUI loaded to first successful chat in under 3 minutes when the user has a hosted provider key ready.
- Zero normal first-run steps require direct `.env` or `config.txt` editing.
- Bundled single-user WebUI path does not ask the user to manually paste the generated API key.
- Users can add multiple providers during setup and choose one default for first chat.
- Setup completion cannot be recorded without a real chat response unless the user explicitly skips.
- First-run global navigation remains hidden until first chat succeeds or setup is skipped.
- After completion, the first guided milestone is adding a first source.

Quality indicators:

- Lower support burden from API-key auth confusion.
- Fewer failed first chats caused by unconfigured providers.
- Fewer docs loops between setup guides, `/setup`, and WebUI onboarding.
- Fewer users editing config files for basic first-run setup.

## Testing Strategy

### Docs And CLI Contract

- Docker solo and local solo docs present peer setup paths.
- Multi-user selection routes to the multi-user guide and checklist.
- Public setup docs share prepare/start/verify/WebUI/first-chat lifecycle language.
- `make quickstart` or its successor prints a WebUI URL and clear next step.
- CLI verification and WebUI readiness use matching success definitions.

### Backend Setup APIs

- Setup state persists across restart.
- State transitions reject invalid completion without first chat success or explicit skip.
- Provider secret writes are masked in responses.
- Hosted provider validation distinguishes invalid key, unavailable provider, and rate/quota failure when possible.
- Local endpoint validation distinguishes unreachable, auth failure, unsupported API shape, and model unavailable.
- `/setup` and WebUI consume the same state/readiness model.

### WebUI Onboarding

- First-run shell hides global navigation until completion or skip.
- Sequential wizard includes required screens: setup path, privacy/security, providers, ingest defaults, audio/STT/TTS, first chat.
- Optional RAG/storage screens are visible but deferrable.
- Selecting multi-user exits to the guide/checklist.
- Hosted provider key entry and local endpoint entry can both be completed without editing files.
- Multiple providers can be saved.
- One default provider/model can be selected for first chat.
- First chat success marks setup complete.
- Skip flow records skipped state and reveals normal navigation with recovery affordance.

### End-To-End

- Fresh Docker single-user path reaches WebUI onboarding.
- Fresh local single-user path reaches WebUI onboarding.
- Hosted-key path reaches first real chat and marks setup complete.
- Invalid hosted key recovers in place.
- Invalid local endpoint recovers in place.
- Restart during `in_progress` setup resumes from backend state.
- Post-completion next milestone prompts adding a first source.

## Acceptance Criteria

- A first-time solo user can choose Docker single-user or local single-user and reach the same WebUI onboarding flow.
- A user choosing multi-user is routed out of the solo wizard to the multi-user guide/checklist.
- The WebUI first-run flow is a progressive sequential wizard with required and optional steps.
- The normal first-time path does not require editing `.env` or `config.txt`.
- Hosted provider keys and local provider endpoints can be configured through the WebUI.
- Multiple providers can be configured during setup.
- A default provider/model is selected for first chat.
- Onboarding completion requires a real successful model response unless the user explicitly skips.
- Setup state is consistent across WebUI, backend `/setup`, CLI verification, and restart.
- Conflicting legacy onboarding surfaces are redirected, demoted, or clearly marked internal.

## Risks And Mitigations

### Risk: The wizard becomes too long

Mitigation: keep the wizard progressive. Make chat/provider setup required for first chat, but allow ingest defaults and audio setup to use recommended defaults quickly. Keep RAG/storage optional.

### Risk: Provider setup becomes a secret-management hazard

Mitigation: backend owns writes, responses mask secrets, logs never include secrets, and validation errors avoid echoing values.

### Risk: Local model setup blocks first-run success

Mitigation: local runtimes are guide-and-verify alternatives. Hosted-key setup remains the fastest recommended path.

### Risk: Backend `/setup` and WebUI drift again

Mitigation: shared setup/readiness APIs, shared state vocabulary, and tests that assert both surfaces use the same model.

### Risk: Local and Docker paths diverge in quality

Mitigation: define both as peer solo paths with equivalent lifecycle stages and success definitions.

## Open Questions For Implementation Planning

- Should first chat verification use a fixed prompt, a user-written prompt, or both?
- Which config fields are safe and stable enough for V1 WebUI writes?
- Which local endpoint checks should be supported first for Ollama, llama.cpp, and generic OpenAI-compatible APIs?
- How should CLI verification report "install ready but first chat not complete" without making automation brittle?
