# OmniVoice Real Sidecar Synthesis Design

- Date: 2026-05-22
- Project: tldw_server
- Backlog: TASK-453
- Topic: Finish the managed OmniVoice TTS sidecar so it produces real audio

## Objective

Finish the existing `omnivoice` TTS provider integration by replacing the current sidecar contract stub with a real OmniVoice model runner.

The repository already has most of the first-class provider surface:

- `TTSProvider.OMNIVOICE`
- default registry adapter mapping
- `OmniVoiceAdapter`
- sidecar supervisor
- sidecar protocol/server skeleton
- provider config block
- validation rules
- stored voice metadata support
- installer helper
- unit tests and an earlier sidecar design spec

The missing runtime behavior is that the sidecar currently returns a generated silent WAV from `/v1/synthesize`. The next implementation slice should make that endpoint import OmniVoice in the sidecar runtime, load the configured model, call the real Python API, and return synthesized WAV/PCM audio to the main app.

## Scope

### In Scope

- Replace the silent-WAV sidecar stub with real OmniVoice synthesis.
- Keep the existing managed local sidecar architecture.
- Support upstream OmniVoice auto voice, voice design, and voice cloning modes.
- Require explicit install/verify before runtime synthesis; no model download should happen during an API request.
- Pass an optional language id through to OmniVoice without maintaining a large local language allowlist.
- Load one configured model/runtime per sidecar process.
- Allow request-level generation controls through a validated allowlist.
- Keep `reference_text` required for cloning in v1.
- Keep public output conversion in the main app.
- Add unit and opt-in integration test coverage for the real-runtime boundary.

### Out Of Scope

- Replacing the sidecar with direct in-process OmniVoice imports.
- Using the OmniVoice CLI as the normal per-request execution path.
- Runtime model downloads on first request.
- Per-request model or device switching.
- Incremental audio streaming from OmniVoice.
- Whisper auto-transcription for missing clone `reference_text`.
- Making OmniVoice part of default TTS bundles or changing default TTS provider behavior.

## Context From Upstream

OmniVoice exposes a Python API through `omnivoice.OmniVoice.from_pretrained(...)` and `model.generate(...)`.

The upstream README documents three generation modes:

- voice cloning with `ref_audio` and `ref_text`
- voice design with `instruct`
- auto voice with only `text`

It also documents optional generation controls such as `num_step`, `guidance_scale`, `denoise`, `t_shift`, `position_temperature`, `class_temperature`, `layer_penalty_factor`, `duration`, `speed`, `postprocess_output`, `audio_chunk_duration`, and `audio_chunk_threshold`.

Upstream says generated audio is returned as arrays at 24 kHz. The sidecar should serialize that output to WAV or PCM for the existing adapter boundary.

## Approach Considered

### Recommended: Python API Inside The Managed Sidecar

Keep the existing sidecar and make it a real model runner:

1. sidecar process imports heavy OmniVoice dependencies
2. sidecar loads one configured `OmniVoice` model
3. sidecar maps internal synth requests into `model.generate(...)`
4. sidecar serializes output to WAV or PCM
5. main app converts the result to public response formats

This matches the prior approved integration boundary and keeps heavy ML dependencies out of the FastAPI server process.

### Alternative: CLI Wrapper Inside The Sidecar

The sidecar could invoke `omnivoice-infer` per request and read the output WAV.

This would be easy to validate against upstream behavior, but it wastes the sidecar's resident-process benefits, makes startup/caching harder to control, and weakens structured error handling.

### Alternative: Direct Adapter Import

`OmniVoiceAdapter` could import OmniVoice directly and synthesize in-process.

This is smaller mechanically but conflicts with the existing isolation design and would pull heavy ML dependencies into the main API process.

## Approved Design

Use the Python API inside the existing managed sidecar.

The main app remains responsible for:

- provider selection
- public request normalization
- stored `custom:<voice_id>` lookup
- voice-reference retrieval
- provider validation
- fallback policy
- final output format conversion
- response history and observability
- public error mapping

The sidecar owns:

- importing OmniVoice and its ML dependencies
- loading one configured model
- running real synthesis
- serializing WAV/PCM audio
- runtime-local readiness and structured internal errors

The sidecar must not query the voice database or implement public OpenAI-compatible semantics.

## Runtime Configuration

The sidecar should load one configured runtime per process.

Provider config should support at least:

- `providers.omnivoice.model`: local model path or pre-cached model id
- `providers.omnivoice.sample_rate`: default `24000`
- `providers.omnivoice.max_concurrent_generations`: default `1`
- `providers.omnivoice.extra_params.python_path`
- `providers.omnivoice.extra_params.runtime_path`
- `providers.omnivoice.extra_params.device_map`
- `providers.omnivoice.extra_params.dtype`
- optional `providers.omnivoice.extra_params.attn_implementation` if upstream exposes it cleanly

The sidecar should not support per-request model, device, dtype, or attention backend overrides in v1.

## No-Download Runtime Policy

Runtime synthesis must not silently download model assets.

The installer/setup path should provision and verify before `providers.omnivoice.enabled` is useful. At runtime, missing assets should produce a structured readiness or synthesis failure such as `MODEL_NOT_AVAILABLE`.

The implementation should support reproducible provisioning through PyPI or a pinned git source, while preserving explicit local-source development through paths such as `../OmniVoice`.

## Sidecar API Contract

Keep the narrow internal HTTP contract and extend it only as needed.

Required endpoints:

- `GET /health`
- `GET /status`
- `POST /control/warmup`
- `POST /control/reload`
- `POST /control/shutdown`
- `POST /v1/synthesize`

All endpoints must continue to require the ephemeral sidecar token header.

### Synthesize Request

The request schema should include:

- `text`
- `mode`: `auto`, `design`, or `clone`
- optional `voice`
- optional `instruct`
- optional `language_id`
- optional `reference_audio_path`
- optional `reference_text`
- optional `requested_sample_rate` for diagnostics only
- `generation`: a nested object containing only allowlisted generation parameters

`mode` can be explicit or derived:

- `clone` when `reference_audio_path` is present
- `design` when `instruct` is present
- `auto` otherwise

Clone mode requires both `reference_audio_path` and `reference_text`.

The canonical sidecar protocol must use the nested `generation` object. The adapter may accept legacy or public top-level `extra_params` fields, but it must normalize them into `generation` before calling the sidecar. The sidecar should reject unknown top-level request keys and unknown `generation` keys.

Explicit `mode` conflicts should fail validation rather than being silently corrected:

- `mode="auto"` cannot include `instruct` or `reference_audio_path`.
- `mode="design"` requires `instruct` and cannot include `reference_audio_path`.
- `mode="clone"` requires `reference_audio_path` and `reference_text`.
- If both `instruct` and `reference_audio_path` are provided, the request is invalid unless a future explicit mixed mode is designed.

### Sample Rate Ownership

OmniVoice synthesis is treated as native 24 kHz output in v1. The sidecar should serialize the real generated audio at the native model sample rate and report that value in `X-OmniVoice-Sample-Rate`.

`requested_sample_rate` in the internal request is optional diagnostic metadata only. It must not cause the sidecar to resample or rewrite headers to a non-native rate. Public target sample-rate conversion remains the main app adapter's responsibility alongside public format conversion.

### Synthesize Response

The response should be binary audio plus metadata headers:

- `X-OmniVoice-Audio-Format`: `wav` or `pcm`
- `X-OmniVoice-Sample-Rate`
- `X-OmniVoice-Channels`
- `X-OmniVoice-Provider`
- `X-OmniVoice-Mode`
- optional `X-OmniVoice-Model`
- optional `X-OmniVoice-Cold-Start`

The sidecar should return only WAV or PCM. The main app keeps converting to `mp3`, `opus`, `aac`, `flac`, `wav`, or `pcm`.

## Request Mapping

### Auto Voice

For auto voice:

```python
audio = model.generate(text=text, language_id=language_id, **generation_kwargs)
```

If `language_id` is omitted, do not synthesize a default language id inside tldw. Let OmniVoice use its own behavior.

### Voice Design

For voice design:

```python
audio = model.generate(
    text=text,
    instruct=instruct,
    language_id=language_id,
    **generation_kwargs,
)
```

`instruct` should come from `extra_params.instruct` or an OmniVoice-specific alias documented for the public API.

Accepted public aliases for voice design are:

- `extra_params.instruct`
- `extra_params.voice_design`
- `extra_params.voice_description`

The adapter should normalize these to sidecar `instruct`. If more than one alias is present with different values, fail validation.

### Voice Cloning

For voice cloning:

```python
audio = model.generate(
    text=text,
    ref_audio=reference_audio_path,
    ref_text=reference_text,
    language_id=language_id,
    **generation_kwargs,
)
```

The main app should continue materializing direct or stored voice references into a managed temporary file before calling the sidecar. The sidecar should only accept reference paths under its configured scratch/runtime area or another explicitly allowed managed directory.

## Language Handling

V1 should use optional language passthrough rather than an explicit 600-language allowlist.

Resolution order:

1. `extra_params.language_id`
2. `extra_params.language`
3. normalized public request language
4. omitted

The current hard-coded English-only validation for `omnivoice` should be relaxed so supported languages are not artificially blocked by tldw.

The sidecar payload should use only `language_id`. The adapter owns alias resolution and should fail validation if `extra_params.language_id`, `extra_params.language`, and the normalized public request language conflict.

## Generation Parameter Validation

The main app should validate public request extras and pass the canonical sidecar `generation` object through a conservative allowlist:

- `num_step`: positive int, recommended bounded range such as `1..128`
- `guidance_scale`: non-negative float, bounded to a practical range
- `denoise`: bool
- `t_shift`: float
- `position_temperature`: non-negative float
- `class_temperature`: non-negative float
- `layer_penalty_factor`: non-negative float
- `duration`: positive float
- `speed`: positive float
- `postprocess_output`: bool
- `preprocess_prompt`: bool
- `audio_chunk_duration`: positive float
- `audio_chunk_threshold`: positive float

`duration` takes priority over `speed`, matching upstream behavior.

Unknown OmniVoice-specific generation parameters should be rejected with `TTSValidationError` rather than silently ignored.

## Model Runner Design

Add a sidecar-local runner object with a small interface:

```python
class OmniVoiceRuntime:
    async def load(self) -> None: ...
    async def synthesize(self, request: OmniVoiceSynthesizeRequest) -> OmniVoiceSynthesizeResult: ...
    async def status(self) -> OmniVoiceRuntimeStatus: ...
    async def unload(self) -> None: ...
```

The runner should:

- lazy-load on first synth or warmup
- enforce one model load at a time
- serialize generation with an async lock by default
- run blocking model work off the event loop
- convert `np.ndarray` output into WAV bytes with `soundfile` or `wave`
- reject empty model output
- avoid logging raw text or reference transcripts

Tests should be able to inject a fake runner into the FastAPI sidecar app.

## Readiness And Health

Health should be cheap and should not synthesize audio.

Readiness states should include:

- `disabled`
- `runtime_missing`
- `model_unavailable`
- `idle_stopped`
- `starting`
- `loading`
- `ready`
- `degraded`
- `shutting_down`

An explicit verify path may run a tiny real synth smoke test, but normal health polling should rely on cached runtime state.

## Concurrency

Default behavior should remain conservative:

- one sidecar process
- one configured model
- one in-flight synthesis
- startup coalesced through the existing supervisor lock

If there is a queue policy, v1 should queue behind the generation lock by default. A future config flag can add fail-fast behavior for overloaded providers.

## Error Handling

Sidecar errors should be structured and sanitized.

Recommended internal error codes:

- `RUNTIME_IMPORT_FAILED`
- `MODEL_NOT_AVAILABLE`
- `MODEL_LOAD_FAILED`
- `SYNTHESIS_FAILED`
- `INVALID_REFERENCE_AUDIO`
- `INVALID_GENERATION_PARAMETER`
- `REFERENCE_PATH_NOT_ALLOWED`
- `EMPTY_AUDIO_OUTPUT`
- `SIDECAR_SHUTTING_DOWN`

The main adapter maps them into existing TTS exceptions:

- setup/runtime missing -> `TTSProviderNotConfiguredError`
- invalid inputs -> `TTSValidationError`
- synthesis/runtime failure -> `TTSGenerationError`

The adapter should not expose sidecar token values, full local paths, raw request text, raw reference text, or stack traces in public responses.

## Fallback Semantics

Explicit OmniVoice semantics should not silently fall back to another provider.

No fallback when:

- provider is explicitly `omnivoice`
- model is explicitly `omnivoice` or an OmniVoice alias
- request uses `custom:` voice
- request includes direct `voice_reference`
- request includes `instruct`
- request includes OmniVoice-only generation parameters

Generic fallback remains acceptable when OmniVoice is merely an implicit priority candidate and the request has no OmniVoice-specific semantics.

## Setup And Verification

The installer should:

- create the dedicated sidecar venv
- install PyTorch according to operator-selected backend instructions or document the required pre-step
- install OmniVoice from PyPI, pinned git, or explicit local checkout
- install sidecar dependencies
- patch only `providers.omnivoice`
- record `python_path`, `runtime_path`, and optional logs path
- optionally run a load-only verification
- optionally run a tiny smoke synth when the model is available

The setup path should not enable surprise runtime downloads by default.

## Testing Requirements

### Unit Tests

- provider aliases and default resolution still map to `omnivoice`
- validation accepts `auto`, `design`, and `clone`
- validation requires `reference_text` for clone/custom voices
- language id passthrough does not require an English-only allowlist
- generation parameter allowlist accepts valid values and rejects invalid values
- sidecar auth still protects all endpoints
- sidecar synth rejects invalid reference paths
- fake runner receives the expected `model.generate(...)` arguments
- fake runner output is serialized to parseable WAV
- fake runner errors map to structured sidecar responses
- adapter maps structured sidecar errors to existing TTS exceptions
- no public error includes sidecar token, raw text, raw reference text, or full sensitive paths

### Integration Tests

Real OmniVoice integration tests should be opt-in, for example:

- `TLDW_TEST_OMNIVOICE_REAL=1`
- configured local model/cache required
- skip cleanly when runtime or model is unavailable

Opt-in real tests should cover:

- tiny auto voice synth
- tiny voice design synth
- cloning synth with short reference audio and `reference_text`
- output is non-empty and parseable as WAV
- runtime does not download during request

## Rollout

OmniVoice remains disabled by default.

This slice should not alter:

- default provider priority behavior
- curated setup bundles
- non-OmniVoice TTS behavior
- public response format support

Docs should describe OmniVoice as a heavier optional provider that supports buffered synthesis through `/api/v1/audio/speech`, not incremental streaming in v1.

## Acceptance Criteria

- `/v1/synthesize` no longer returns hard-coded silent audio when a real runner is configured.
- Sidecar can synthesize auto voice, voice design, and cloning through the OmniVoice Python API.
- Runtime synthesis does not download model assets implicitly.
- Missing runtime/model state is reported as a clear provider-unavailable/readiness failure.
- Public API continues using the existing `omnivoice` registry path and output conversion behavior.
- Tests cover request mapping, validation, structured errors, sidecar auth, and fake-runner audio serialization.
- Real OmniVoice smoke tests are available behind an explicit opt-in environment flag.
