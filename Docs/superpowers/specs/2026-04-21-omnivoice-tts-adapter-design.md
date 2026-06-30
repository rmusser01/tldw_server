# OmniVoice TTS Sidecar Adapter Design

- Date: 2026-04-21
- Project: tldw_server
- Topic: Add OmniVoice as a first-class TTS provider via a managed local sidecar
- Mode: Design for implementation

## 1. Objective

Add support for [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) as a first-class provider in the `tldw_server` TTS registry.

The integration must fit the current TTS architecture, support the existing `/api/v1/audio/speech` and voice-management flows, and remain operable in a self-hosted environment without pushing OmniVoice-specific runtime logic into generic API endpoints.

The approved direction is:

- first-class `omnivoice` provider in the TTS registry
- managed local sidecar, not direct in-process imports
- `tldw_server` owns sidecar startup, warmup, health, and shutdown
- local HTTP on loopback only
- broad OmniVoice feature exposure in v1
- main app keeps `custom:` voice resolution, request normalization, and final format conversion
- sidecar returns OmniVoice-native WAV/PCM output only

## 2. Scope

### In Scope

- add `omnivoice` as a registry-backed TTS provider
- add an OmniVoice adapter in the main app that talks to a managed local sidecar
- support OmniVoice auto voice, voice design, and voice cloning
- integrate OmniVoice with stored `custom:<voice_id>` voices through the existing voice manager
- support OmniVoice-specific advanced parameters such as `instruct`, `num_step`, `guidance_scale`, `duration`, `speed`, `denoise`, `t_shift`, and related generation controls
- provision an isolated OmniVoice runtime and managed sidecar entrypoint
- add setup, readiness, health, and verification coverage for the managed runtime
- add tests for registry behavior, supervisor lifecycle, protocol mapping, stored voice flows, and health/readiness

### Out Of Scope

- direct in-process Python embedding of OmniVoice into the main server runtime
- true incremental audio streaming from OmniVoice in v1
- a generic multi-provider model-runtime sidecar framework
- automatic speech-to-text derivation of `reference_text` for cloning in v1
- automatic promotion of OmniVoice into every curated setup bundle

## 3. Approved Constraints

### User-Approved Decisions

1. OmniVoice must be a first-class provider, not a one-off helper.
2. The integration should use a sidecar design rather than direct runtime coupling.
3. The sidecar should be a `tldw_server`-managed local service that the main app starts, checks, and stops.
4. Main app and sidecar communication should use local HTTP on `127.0.0.1` with a private port and health endpoint.
5. The sidecar boundary should stay narrow:
   - sidecar owns OmniVoice runtime concerns
   - main app owns voice-manager behavior, request normalization, `custom:` resolution, and public API semantics
6. Startup policy should be hybrid:
   - lazy start by default
   - optional eager warmup mode
7. Only one configured OmniVoice model should be loaded at a time.
8. Model changes should require a managed reload or restart.
9. Idle shutdown should be the default lifecycle policy, with resident mode optional.
10. The sidecar synth contract should return OmniVoice-native WAV/PCM output, while the main app performs final format conversion and public streaming behavior.
11. Cloning requires `reference_text` for both stored and direct flows in v1.

### Design Corrections From Review

1. Loopback sidecar traffic cannot silently reuse the normal shared outbound HTTP path because current egress policy blocks private IPs by default outside tests.
2. Sidecar ownership cannot live only on `app.state`, because the TTS stack is currently exposed through process-global singleton services.
3. Startup, retry, and crash-loop handling should reuse existing provider-registry failure backoff and local-runtime readiness patterns rather than inventing a second retry system.
4. Lazy sidecar startup must respect application draining so shutdown cannot trigger a fresh sidecar spawn.
5. Loopback binding alone is not enough as an internal trust boundary; the sidecar should require an ephemeral per-process shared secret.

## 4. Approaches Considered

### Recommended: Thin Managed Local HTTP Sidecar

Add `omnivoice` as a first-class provider in the main TTS registry, but run the OmniVoice runtime behind a narrowly scoped local HTTP sidecar that is supervised by `tldw_server`.

Pros:

- fits the existing provider architecture
- isolates heavy OmniVoice runtime dependencies from the main process
- keeps `custom:` voice logic and public API semantics in the current TTS stack
- provides a clean path to warm runtime behavior without over-generalizing

Cons:

- adds process supervision and internal protocol work
- requires explicit loopback-security and egress-policy handling

### Alternative: Rich OmniVoice Sidecar

Move more logic into the sidecar, including parts of voice preparation, request normalization, and response formatting.

Pros:

- thinner main-app adapter
- more OmniVoice behavior centralized in one process

Cons:

- cuts across current `TTSServiceV2` and voice-manager responsibilities
- makes public behavior harder to reason about
- increases drift from existing provider patterns

### Rejected: Direct CLI-Per-Request Adapter

Launch OmniVoice CLI processes directly from the main app for each request without a resident sidecar.

Reason rejected:

- colder startup path for every request
- weaker path toward future warm runtime behavior
- less aligned with the approved managed-sidecar direction

### Rejected: Generic Multi-Provider Runtime Sidecar First

Build a shared sidecar framework for several TTS providers before integrating OmniVoice.

Reason rejected:

- broader than the current need
- introduces abstraction before the concrete OmniVoice path is proven

## 5. Approved Design

Implement OmniVoice as a normal provider in the main TTS registry, but make the provider adapter a client to a managed local sidecar instead of a direct model runner.

At a high level:

1. register `omnivoice` in the TTS registry and request-resolution surfaces
2. add an `OmniVoiceAdapter` in the main app that translates normalized TTS requests into internal sidecar HTTP requests
3. add an OmniVoice supervisor service that owns sidecar startup, readiness, idle shutdown, and reload behavior
4. provision OmniVoice into an isolated managed runtime with explicit setup and verification support
5. keep stored `custom:` voice resolution, reference-text injection, output conversion, and public API semantics in the main app

## 6. Architecture

### Main App Responsibilities

The main `tldw_server` process continues to own:

- provider selection and registry behavior
- `custom:` voice lookup and stored-reference retrieval
- request normalization and OmniVoice-specific validation
- `reference_text` injection for stored voices
- provider fallback policy
- public streaming and non-streaming API behavior
- final response-format conversion
- public error mapping, observability, and history logging

This matches the current TTS service design, where `TTSServiceV2` already handles stored `custom:` voice resolution and provider-specific artifact injection before calling an adapter.

### Sidecar Responsibilities

The OmniVoice sidecar owns only OmniVoice runtime concerns:

- isolated runtime bootstrapping entrypoint
- model load/unload behavior for one configured model
- OmniVoice synthesis execution
- sidecar-local health and readiness status
- minimal warmup, shutdown, and reload control operations

The sidecar should not own:

- public OpenAI-compatible request semantics
- general voice-manager behavior
- cross-provider fallback
- final audio conversion to all public response formats

### Supervisor Ownership

The sidecar supervisor must be a TTS-owned singleton/service that is initialized and closed with the TTS service lifecycle.

It must not be designed as an `app.state`-only dependency, because the current TTS stack is primarily accessed through `get_tts_service_v2()` and other global TTS service paths rather than request-scoped app references.

## 7. Internal Transport And Security

### Loopback Transport Rule

The OmniVoice sidecar should listen on loopback only:

- `127.0.0.1` by default
- optional IPv6 loopback support if needed later

The internal HTTP client path must explicitly handle loopback sidecar traffic without weakening the general outbound egress policy used for arbitrary external requests.

The design requirement is:

- do not globally relax private-IP egress checks
- do not route OmniVoice sidecar requests through the same unrestricted path used for arbitrary external HTTP
- do introduce an explicit internal-loopback transport allowance or sidecar-specific client path

### Sidecar Authentication

Loopback binding is not sufficient on its own because any local process could otherwise hit synth or control endpoints.

The sidecar should therefore require an ephemeral shared secret generated by the supervisor at startup and attached to every internal request, for example in a dedicated header such as `X-TLDW-Sidecar-Token`.

This token should:

- be generated per sidecar process
- never be exposed through public APIs
- never be logged in plaintext
- be rotated on restart

### Port Selection

Port selection should follow the existing local-runtime pattern:

- probe for an available local port in a bounded range
- spawn the sidecar
- verify actual readiness over HTTP
- retry on collision or failed bind within a bounded startup path

The supervisor should treat port probing as advisory rather than authoritative and rely on post-spawn readiness verification before declaring the sidecar usable.

## 8. Request Flow

The request path should be:

1. public request enters existing `/api/v1/audio/speech` flow
2. `TTSServiceV2` normalizes provider selection and request semantics
3. stored `custom:` voices are resolved in-process through the current voice-manager path
4. OmniVoice-specific request extras are normalized in the main app
5. the OmniVoice adapter sends a narrow internal request to the sidecar
6. the sidecar returns OmniVoice-native WAV/PCM output plus minimal execution metadata
7. the main app converts the audio to the public response format and applies current streaming/non-streaming behavior

### Default Semantics

- provider key: `omnivoice`
- default model: provider-configured OmniVoice model
- default voice: `auto`

Because the public OpenAI-compatible schema still has a legacy default `voice`, OmniVoice request normalization must explicitly normalize the effective voice to `auto` when the caller did not intentionally choose a voice.

### Stored Voice Handling

When a request uses `voice="custom:<voice_id>"`, the existing `TTSServiceV2` flow should:

- load the stored reference audio
- load stored `reference_text`
- inject those values into the normalized OmniVoice request
- pass the request to the OmniVoice adapter exactly like a direct cloning request

This is intentional. The sidecar should not be responsible for voice-manager lookups.

### Stored Voice Upload And Encode Semantics

Stored OmniVoice voices should continue to use the existing `/api/v1/audio/voices/*` system.

For `/audio/voices/upload` with `provider="omnivoice"`:

- validate file format, size, and duration against OmniVoice requirements
- require `reference_text`
- normalize or convert the stored sample as needed through current voice-manager flows
- persist OmniVoice-compatible reference metadata

For `/audio/voices/encode` with `provider="omnivoice"`:

- treat the operation as staging and validation, not latent-token generation
- confirm the stored voice exists
- confirm `reference_text` exists
- mark the voice as OmniVoice-ready in provider metadata
- do not invent NeuTTS-style `ref_codes` artifacts that OmniVoice does not expose

## 9. Sidecar API Contract

### Endpoints

The sidecar should expose a minimal internal API:

- `GET /health`
- `GET /status`
- `POST /v1/synthesize`
- `POST /control/warmup`
- `POST /control/shutdown`
- optional `POST /control/reload`

### Synthesize Request

The synth request should include only OmniVoice-relevant fields, such as:

- text
- configured model identifier or path
- normalized language
- synthesis mode implied by presence of `instruct` and/or reference audio
- `instruct`
- `reference_text`
- generation controls such as `num_step`, `guidance_scale`, `duration`, `speed`, `denoise`, `t_shift`, `layer_penalty_factor`, `position_temperature`, and `class_temperature`
- managed reference-audio handle when cloning

### Reference Audio Transport

For cloning requests, the recommended v1 transport is:

- main app materializes reference audio into a managed temporary file under a supervisor-owned scratch area
- main app passes a safe managed handle or bounded path reference to the sidecar
- sidecar accepts only references under that managed area

This avoids repeated large base64 JSON payloads and keeps trust boundaries narrow.

The design should also define:

- bounded file size limits
- ownership of the scratch directory
- per-request cleanup or lease cleanup
- behavior when temp-file preparation fails before synthesis

### Synthesize Response

The recommended v1 synth response is:

- binary OmniVoice-native WAV or PCM payload
- sidecar metadata returned as headers or simple metadata fields, including:
  - sample rate
  - configured model id
  - cold-start vs warm-run marker
  - generation duration

The sidecar should not perform broad public-format packaging for `mp3`, `aac`, `opus`, or similar output types. That remains the main app's responsibility.

## 10. Lifecycle And Readiness

### Startup Policy

The sidecar lifecycle should be hybrid:

- lazy start by default
- optional eager warmup at app startup

The first-use startup path must be idempotent and concurrent requests must coalesce behind one startup/warmup path.

### Model Loading

Only one configured OmniVoice model should be loaded in memory at a time.

Changing the configured model should require:

- a managed reload if supported
- otherwise a managed stop and restart

The design should not assume per-request model switching.

### Idle Shutdown

Idle shutdown should be the default lifecycle mode.

The supervisor should:

- keep last-used timestamps
- stop the sidecar after a configurable inactivity timeout
- support a resident mode that disables idle shutdown

### Concurrency Policy

OmniVoice should use a conservative default concurrency policy in v1.

The default should be:

- one loaded model per sidecar
- one in-flight synthesis per sidecar by default

If the main app already exposes provider-specific concurrency control, the initial default should effectively behave like:

- `providers.omnivoice.max_concurrent_generations = 1`

This keeps the first version predictable while avoiding bursty cold-start or memory contention behavior.

### Drain Gate

No lazy startup, warmup, or auto-restart should begin once the app enters draining mode.

This should align with the existing lifecycle gate so shutdown cannot race with a new OmniVoice process launch.

### Readiness States

Health and status should distinguish at least:

- `disabled`
- `idle_stopped`
- `starting`
- `live_model_cold`
- `warming`
- `ready`
- `degraded`
- `shutting_down`

### Verification

Verification should be explicit and cheap:

- ensure the runtime exists
- ensure the sidecar entrypoint is launchable
- ensure the configured model is loadable
- run a tiny synth smoke test on demand
- confirm the output is non-empty and parseable

Normal health endpoints should report cached readiness and last verification results rather than running a synthesis probe on every call.

## 11. Setup And Runtime Provisioning

OmniVoice must be provisioned into a dedicated managed runtime rather than the main server interpreter.

Provisioning should create:

- isolated Python environment or equivalent runtime
- sidecar launch entrypoint inside that runtime
- recorded runtime metadata
- model asset metadata and provisioning provenance

### Source Preference

The local checkout at `../OmniVoice` should be supported as an explicit local-development source mode.

It should not become the silent default for general deployments.

The default managed path should remain a normal reproducible install/provision flow, with the sibling checkout used only when the operator explicitly chooses local-source provisioning.

## 12. Validation, Fallback, And Error Semantics

### Validation Rules

OmniVoice-specific validation should enforce:

- supported response formats
- supported text length ceilings
- supported speed range
- cloning requires `reference_text`
- allowed advanced generation parameters and ranges
- provider-specific voice-reference format, size, and duration requirements

### Fallback Rules

OmniVoice-specific requests should not silently cross-fallback to another provider when they depend on OmniVoice semantics, including:

- cloning
- stored `custom:` voices
- voice design via `extra_params.instruct`
- OmniVoice-only advanced parameters

Retry is allowed inside the OmniVoice provider boundary when the failure mode supports it.

### Error Classes

The design should distinguish at least:

- sidecar launch failure
- sidecar connectivity failure
- sidecar startup timeout
- model load failure
- synth execution failure
- invalid managed reference handle
- sidecar auth mismatch
- idle-stop race during request dispatch

Sidecar responses should use structured error codes so the main app can decide whether to:

- retry
- restart the sidecar
- mark the provider failed with backoff
- fail fast to the caller

## 13. Observability And Failure Backoff

The design should reuse current runtime patterns where possible.

### Startup And Health Pattern

Sidecar readiness should reuse the existing local-runtime approach:

- spawn subprocess
- poll internal HTTP readiness
- classify startup failure cleanly if the process exits or never becomes ready

### Registry Failure Backoff

Repeated OmniVoice startup failures should feed into the existing provider-registry failure/backoff behavior rather than introducing an unrelated retry system.

This gives OmniVoice the same coarse provider-availability semantics already used elsewhere in the TTS stack.

### Logging And Metrics

The implementation should log or track:

- request id and correlation id when available
- sidecar PID
- host and port
- configured model id
- startup duration
- cold vs warm synth count
- idle shutdown count
- restart count
- sidecar-specific failure codes

Logs must avoid leaking the sidecar shared secret or raw sensitive request payloads.

## 14. Expected Touchpoints

The implementation is expected to touch at least:

- TTS provider enum and registry entries
- OmniVoice provider config schema
- request-resolution and provider-default handling
- OmniVoice adapter implementation in the main app
- OmniVoice supervisor/service lifecycle wiring
- setup/install/runtime provisioning path
- health and readiness reporting
- voice-manager provider requirements
- TTS validation and fallback rules
- tests for adapter, supervisor, setup, and voice flows

## 15. Testing Requirements

### Unit Tests

- provider alias and default resolution
- OmniVoice config parsing
- sidecar request mapping
- shared-secret header injection
- managed reference-file preparation
- supervisor state transitions
- idle-timeout policy
- drain-gate behavior

### Protocol Tests

- `GET /health`
- `GET /status`
- `POST /v1/synthesize`
- `POST /control/warmup`
- `POST /control/shutdown`
- reload behavior if implemented

### Integration Tests

- lazy startup on first request
- optional eager warmup path
- idle shutdown and restart after idle stop
- startup coalescing under concurrent first-use requests
- sidecar unavailable failure path
- sidecar auth failure path
- model reload or restart behavior after config change
- `custom:<voice_id>` resolution with stored `reference_text`
- OmniVoice-specific no-cross-provider-fallback behavior

### Setup And Health Tests

- isolated runtime provisioning metadata
- explicit local-source install mode using `../OmniVoice`
- readiness state transitions
- cached health vs explicit verify behavior
- launch metadata recorded for diagnostics

## 16. Rollout Notes

The rollout should remain reversible:

- OmniVoice stays disabled until explicitly configured or provisioned
- no existing default TTS provider behavior changes unless the operator opts in
- the internal loopback allowance must stay scoped to the OmniVoice sidecar path only
- the sidecar remains host-local and non-public

## 17. Approved Outcome

The approved design is to add OmniVoice as a managed, first-class TTS provider with:

- a thin main-app adapter
- a `tldw_server`-supervised local sidecar
- isolated runtime provisioning
- stored custom voice support through the existing voice-manager path
- explicit loopback-security and egress-policy handling
- explicit readiness, verification, and idle-shutdown semantics
- buffered public streaming behavior in v1
- one-model-at-a-time runtime behavior

This gives `tldw_server` a usable OmniVoice integration that matches current TTS boundaries while avoiding direct runtime coupling and leaving a clean future path for optimization.
