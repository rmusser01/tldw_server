# OmniVoice TTS Adapter Design

- Date: 2026-04-21
- Project: tldw_server
- Topic: Add OmniVoice as a first-class TTS adapter in the registry with managed local runtime support
- Mode: Design for implementation

## 1. Objective

Add support for [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) as a first-class provider in the `tldw_server` TTS registry.

The integration must fit the existing adapter architecture, support the current `/audio/speech` and `/audio/voices/*` flows, and remain operable in a self-hosted environment without leaking OmniVoice-specific logic into generic endpoint code.

The approved direction is:

- first-class `omnivoice` provider in the TTS registry
- CLI/subprocess integration, not direct in-process Python imports
- fully managed by `tldw_server`
- broad OmniVoice feature exposure in v1
- buffered streaming in v1, with a clean future path to true streaming
- required `reference_text` for stored and direct cloning flows in v1

## 2. Scope

### In Scope

- add `omnivoice` as a registry-backed TTS provider
- create a dedicated OmniVoice adapter that shells out to a managed CLI runtime
- support OmniVoice auto voice, voice design, and voice cloning
- integrate OmniVoice with stored custom voices via `custom:<voice_id>`
- define OmniVoice-specific request validation, defaults, and provider limits
- add managed setup/readiness/health support for OmniVoice
- add tests for registry resolution, adapter behavior, stored voice flows, and readiness

### Out Of Scope

- true incremental audio streaming from OmniVoice in v1
- a long-lived OmniVoice sidecar service or warm worker process
- automatic Whisper-style reference transcription for cloning in v1
- a generic multi-provider subprocess abstraction beyond what OmniVoice directly needs
- adding OmniVoice to every curated setup bundle by default

## 3. Constraints And Decisions

### User-Approved Constraints

1. OmniVoice must be treated as a first-class provider, not a thin one-off helper.
2. Integration must use CLI/subprocess execution.
3. `tldw_server` should manage the runtime rather than expecting a manually prepared environment.
4. The feature surface should include:
   - auto voice
   - voice cloning
   - voice design via `extra_params.instruct`
   - advanced generation controls such as `num_step`, `guidance_scale`, `duration`, `speed`, `denoise`, and language selection
5. Stored voices must require `reference_text`.

### Design Corrections Identified During Review

1. OmniVoice must not be installed into the server's main Python environment because the current setup installer installs packages into the active interpreter.
2. `/audio/voices/encode` cannot mean "generate latent artifacts" for OmniVoice because the upstream CLI does not expose a compact pre-encode path analogous to NeuTTS `ref_codes`.
3. "Fully managed" requires setup, readiness, and health support in addition to adapter code.
4. A CLI-per-request provider must use a conservative concurrency default.

## 4. Approaches Considered

### Recommended: Dedicated OmniVoice CLI Adapter With Managed Runtime

Add a specific `omnivoice` provider with its own adapter, config, setup, health, and voice-manager integration.

Pros:

- matches the current registry-based TTS architecture
- keeps OmniVoice-specific behavior isolated
- minimizes endpoint churn
- leaves room for a future sidecar refactor if needed

Cons:

- subprocess startup cost remains visible
- setup/runtime management is broader than an adapter-only feature

### Alternative: Generic External-CLI TTS Framework First

Create a reusable subprocess-provider layer and implement OmniVoice on top of it.

Pros:

- more reusable if several CLI-only engines are expected soon

Cons:

- adds abstraction before concrete need is proven
- expands scope immediately

### Rejected: Sidecar Service First

Run OmniVoice as its own managed service and proxy requests to it.

Reason rejected:

- too large for the first integration
- operationally heavier than the current requirement
- unnecessary before proving the provider's usage and performance profile

## 5. Approved Design

Implement the recommended approach with one important adjustment:

- OmniVoice uses a dedicated, setup-managed isolated runtime rather than the server's main interpreter.

At a high level:

1. register `omnivoice` in the TTS registry and request-resolution surfaces
2. add an `OmniVoiceAdapter` that shells out to the isolated OmniVoice CLI
3. extend voice-manager and `TTSServiceV2` flows so stored `custom:` voices work for OmniVoice
4. add setup, verification, and health/readiness support for the managed runtime
5. keep v1 streaming buffered rather than truly incremental

## 6. Architecture

OmniVoice should be integrated as a normal TTS provider behind the existing `TTSRequest` and `TTSResponse` abstractions.

The adapter owns only provider-specific concerns:

- CLI path and runtime resolution
- temp-file lifecycle
- argument construction
- OmniVoice-specific validation
- stderr/exit-code translation
- output WAV collection and metadata

The generic TTS service continues to own:

- provider selection
- fallback and circuit-breaker behavior
- request normalization
- stored-voice lookup
- response-format conversion
- cross-provider observability

This keeps the OmniVoice boundary narrow and prevents provider-specific behavior from leaking into endpoint code.

## 7. Runtime Strategy

### Isolated Runtime

`tldw_server` should provision OmniVoice into a dedicated runtime directory managed by setup logic, for example under a provider-specific managed path inside the repository/runtime area.

That runtime must contain:

- a Python interpreter or virtualenv entrypoint
- the installed OmniVoice package
- the resolved `omnivoice-infer` command path
- recorded provenance about where OmniVoice was installed from

### Source Preference

The managed install flow should prefer the sibling checkout at `../OmniVoice` when it exists.

If the sibling checkout does not exist, the installer may fall back to a standard package source.

This preference is intentional because the current development context already includes a local OmniVoice clone one directory above the repo.

### Model Ownership

The OmniVoice model path or model identifier should be explicit in provider config.

The runtime may rely on a managed local model path or a provider-configured model identifier, but it should not guess implicitly at request time.

## 8. Request And Feature Mapping

OmniVoice v1 supports three request modes:

1. auto voice
2. voice design
3. voice cloning

The adapter behavior is determined by the normalized unified request:

- auto voice when there is no `voice_reference` and no `extra_params.instruct`
- voice design when `extra_params.instruct` is present
- voice cloning when `voice_reference` is present

### Default Semantics

- provider key: `omnivoice`
- default model: `omnivoice`
- default voice: `auto`

The `voice` field is not used as a named built-in voice inventory for OmniVoice. In OmniVoice requests, it serves only these roles:

- `auto` for default automatic voice selection
- `custom:<voice_id>` for stored custom voice cloning

### CLI Argument Mapping

The adapter should map the normalized request to CLI flags as follows:

- request text -> `--text`
- configured model id/path -> `--model`
- normalized language -> `--language`
- `speed` -> `--speed`
- `extra_params.duration` -> `--duration`
- `extra_params.instruct` -> `--instruct`
- `extra_params.num_step` -> `--num_step`
- `extra_params.guidance_scale` -> `--guidance_scale`
- `extra_params.t_shift` -> `--t_shift`
- `extra_params.denoise` -> `--denoise`
- `extra_params.layer_penalty_factor` -> `--layer_penalty_factor`
- `extra_params.position_temperature` -> `--position_temperature`
- `extra_params.class_temperature` -> `--class_temperature`

For cloning:

- write `voice_reference` bytes to a temp audio file
- pass that path to `--ref_audio`
- require `reference_text`
- accept direct-request reference text through `extra_params.reference_text` (or equivalent normalized aliases already used by the TTS service)
- pass the resolved reference text to `--ref_text`

## 9. Stored Voice Lifecycle

Stored OmniVoice voices should use the existing `/audio/voices` system instead of a provider-specific side channel.

### Upload

`/audio/voices/upload` with `provider="omnivoice"` should:

- validate file type, size, and duration against OmniVoice-specific requirements
- require `reference_text`
- normalize/store the processed reference sample through the existing voice-manager flow
- persist OmniVoice-compatible metadata

### Encode

`/audio/voices/encode` for OmniVoice is a staging/validation operation in v1, not a latent-token generation step.

It should:

- confirm the stored voice exists
- require `reference_text` to exist
- validate the stored sample against OmniVoice provider requirements
- persist provider metadata showing the voice is OmniVoice-ready
- return success without inventing `ref_codes`-style artifacts, leaving `ref_codes_len` unset/null in the response

This keeps API behavior consistent while acknowledging that OmniVoice does not expose a NeuTTS-like pre-encode artifact through the CLI path.

### Request-Time Resolution

When a request uses `voice="custom:<voice_id>"`, the existing `TTSServiceV2` custom-voice resolution path should:

- load the stored reference audio bytes
- load stored `reference_text`
- inject both into the normalized OmniVoice request
- let the adapter handle the request exactly like a direct cloning request

Direct one-off cloning requests and stored custom voices should share the same internal validation and temp-file preparation logic.

## 10. Streaming Behavior

True incremental streaming is out of scope for v1.

For `stream=true`, OmniVoice v1 should:

- generate the full WAV output first
- convert or package the result as needed
- stream the completed payload in chunks through the existing response path
- surface metadata that the transport was buffered rather than live

This preserves API compatibility while keeping the implementation honest about what the provider can actually do today.

Future work may replace this with a warm worker or sidecar process if true streaming becomes important.

## 11. Validation Rules

OmniVoice-specific validation should be added in the same places where other providers are validated today.

### Request Validation

- supported response formats
- text length ceiling
- supported speed range
- supported language semantics
- cloning requires `reference_text`
- voice design parameters allowed only through approved extra params
- advanced param ranges and types

### Voice Reference Validation

OmniVoice voice references should have provider requirements defined in the voice manager, including:

- allowed file formats
- max size
- min and max duration
- target sample rate / conversion target

The initial limits should be conservative and reflect upstream guidance rather than theoretical maximum tolerance.

## 12. Setup, Readiness, And Health

Because this provider is fully managed, readiness must be explicit.

OmniVoice should be considered healthy only when:

- the isolated runtime exists
- the OmniVoice CLI entrypoint is executable
- the configured source/model path resolves
- a lightweight verification flow succeeds

### Verification

Verification should be cheap and deterministic, for example:

- synthesize a very short text sample to a temp WAV
- confirm the output file exists
- confirm the output is non-empty and parseable

### Health Surface

Health/readiness should expose OmniVoice-specific failure reasons such as:

- `disabled`
- `runtime_missing`
- `cli_missing`
- `model_unavailable`
- `verification_failed`

This prevents "enabled in config" from being confused with real runtime availability.

## 13. Concurrency And Performance

OmniVoice is a heavy subprocess-based provider.

The design should therefore set a conservative default concurrency limit:

- `providers.omnivoice.max_concurrent_generations = 1` by default

This uses the provider-specific semaphore path already present in `TTSServiceV2` and avoids spawning multiple heavyweight model loads under bursty traffic.

Future optimization options:

- persistent warm worker
- process pooling
- sidecar service promotion

None of those are required for v1.

## 14. Error Handling

The adapter must translate subprocess failures into ordinary TTS exceptions.

Error classes should distinguish at least:

- validation errors
- runtime/readiness errors
- timeout errors
- CLI non-zero exit failures
- output file missing/invalid failures

User-facing APIs should receive sanitized structured errors. Internal logs may include richer subprocess stderr details for debugging.

Whenever possible, failures should happen before response streaming begins.

## 15. Setup Catalog Integration

OmniVoice should become installable through the setup system, but it should not automatically be added to every curated bundle choice.

The safer initial design is:

1. make OmniVoice a supported setup/install target
2. add readiness and health coverage
3. add it selectively to curated setup bundles where the machine profile and resource tier make sense

This prevents a heavyweight runtime from becoming an accidental default in lightweight CPU-oriented setup flows.

## 16. Expected Touchpoints

The implementation is expected to touch at least these areas:

- TTS registry and provider enums
- TTS request resolution defaults and provider inference
- OmniVoice adapter implementation
- OmniVoice config schema and provider settings
- TTS validation and provider limits
- voice-manager provider requirements and encode semantics
- setup installer schema and engine list
- setup install manager and readiness verification
- audio health/readiness surfaces
- public audio request schema/docs where provider lists are described
- unit and integration tests across TTS, setup, and voice flows

## 17. Testing Requirements

### Unit Tests

- provider alias/model inference
- OmniVoice config parsing
- CLI command construction
- request-to-flag mapping
- required `reference_text` behavior
- readiness classification

### Adapter Integration Tests

- successful auto-voice generation
- successful voice-design generation
- successful cloning generation
- timeout path
- non-zero CLI exit path
- missing runtime / missing CLI path
- buffered `stream=true` behavior

### Service And Voice Tests

- `custom:<voice_id>` resolution for OmniVoice
- stored `reference_text` injection
- OmniVoice upload validation
- OmniVoice encode-as-staging semantics

### Setup And Health Tests

- install metadata and source provenance
- readiness state transitions
- verification success/failure envelopes
- setup-catalog integration where applicable

## 18. Rollout Notes

The implementation should preserve reversibility:

- OmniVoice remains disabled until explicitly configured or provisioned
- no existing default provider behavior changes unless the operator opts in
- the isolated runtime keeps OmniVoice dependency churn separate from the main server environment

## 19. Approved Outcome

The approved design is to add OmniVoice as a managed, first-class, CLI-backed TTS provider with:

- isolated runtime provisioning
- broad feature exposure
- stored custom voice support through existing voice-manager flows
- explicit readiness/health semantics
- buffered v1 streaming
- conservative concurrency defaults

This gives `tldw_server` a clear, operable OmniVoice integration without overcommitting to a sidecar or true streaming architecture before it is needed.
