# audio.cpp TTS Provider And Setup Integration Design

## Status

Accepted for specification after brainstorming on 2026-07-03.

## Related Task

- `TASK-12124`

## Purpose

Integrate [`0xShug0/audio.cpp`](https://github.com/0xShug0/audio.cpp) as an optional tldw_server audio backend. The first implementation should follow Approach A: add a production-shaped `audio_cpp` TTS provider and setup path first, then move toward the broader Audio Studio runtime platform after the TTS path proves reliable.

The first slice must not bypass the existing audio API surface. `POST /api/v1/audio/speech` should continue to route through `TTSServiceV2`, the TTS adapter registry, existing auth, quotas, fallback behavior, history, generated-file storage, metrics, and endpoint error mapping.

## Source Facts And Constraints

The upstream `audiocpp_server` exposes:

- `GET /health`
- `GET /v1/models`
- `POST /v1/audio/speech`
- `POST /v1/audio/transcriptions`
- `POST /v1/tasks/run`

The upstream server speech route accepts OpenAI-style JSON and returns `audio/wav` by default. It accepts `response_format: "json"` for base64 WAV in JSON. Its examples use `voice_ref` as a server-local path and request options such as `max_tokens` and `seed`.

The server README describes `audiocpp_server` as CUDA-only for the current HTTP adapter. The broader project README documents CPU, CUDA, Vulkan, and Metal build paths, but the first managed server target should treat CUDA as the verified server backend and make other server backends explicit future verification work.

The server registers configured model ids on startup when `lazy_load` is true, but loaded models and task sessions remain resident after first use until the server exits. tldw must surface that memory-residency behavior in setup docs and health/status messaging.

`audio.cpp` is Apache-2.0 licensed. tldw_server is GPLv2 per project notes. The first design treats `audio.cpp` as an optional external executable/service installed by explicit user action. Vendoring source, statically linking, or bundling prebuilt binaries requires separate legal/package review.

## Architecture

Add `audio_cpp` as a first-class TTS provider, not endpoint-specific glue.

Request path:

```text
POST /api/v1/audio/speech
  -> TTSServiceV2
  -> TTSAdapterFactory / TTSAdapterRegistry
  -> AudioCppTTSAdapter
  -> AudioCppClient
  -> audiocpp_server /v1/audio/speech
```

The provider has two runtime modes behind the same adapter:

- External server mode: `providers.audio_cpp.base_url` points at an already running `audiocpp_server`.
- Managed sidecar mode: `providers.audio_cpp.managed: true` and `server_binary_path` let tldw start `audiocpp_server --config <server_config_path>`, wait on `/health`, reuse the same HTTP client path, and stop the process on shutdown.

Core units:

- `AudioCppTTSAdapter`: implements the tldw TTS adapter contract, capabilities, request validation, response conversion, metadata, and provider error mapping.
- `AudioCppClient`: small HTTP client for `/health`, `/v1/models`, and `/v1/audio/speech`, using existing retry, timeout, and HTTP client patterns where practical.
- `AudioCppSidecarSupervisor`: optional process owner for managed mode, modeled after existing sidecar patterns but scoped to `audio_cpp`.
- `AudioCppServerConfig`: renderer/validator for the upstream server JSON file.
- `install_tts_audio_cpp.py`: setup/helper script with pure functions for layout and config patching plus explicit opt-in clone/build/model-manager actions.

Do not add generic Audio Studio `/v1/tasks/run` abstractions in the first implementation. The TTS provider should leave room for those by reusing `AudioCppClient` and the sidecar supervisor.

## Provider Registration And Routing

The implementation should register `audio_cpp` explicitly in the existing TTS provider registry:

- Add `TTSProvider.AUDIO_CPP = "audio_cpp"`.
- Add default adapter mapping for `AudioCppTTSAdapter`.
- Add provider aliases for `audio_cpp`, `audio-cpp`, and `audiocpp`.
- Add `audio_cpp` to provider priority only as disabled-by-default documentation/config; it should not affect default selection unless enabled.
- Add `audio_cpp` to format preferences after the adapter's conversion behavior is verified.

Model routing should avoid stealing existing aliases. Today, generic `pocket-tts` routes to the existing PocketTTS provider. The first `audio_cpp` implementation should prefer explicit provider selection or namespaced model aliases such as:

- `audio_cpp:pocket-tts`
- `audio-cpp/pocket-tts`
- `audiocpp/pocket-tts`

Do not remap bare `pocket-tts` to `audio_cpp` unless a later migration intentionally changes the existing PocketTTS behavior and covers that compatibility break in tests and release notes.

## Configuration

Add `audio_cpp` to `tldw_Server_API/Config_Files/tts_providers_config.yaml`, disabled by default.

```yaml
providers:
  audio_cpp:
    enabled: false
    backend: "cuda"  # setup/build hint; not necessarily emitted into server.json
    base_url: "http://127.0.0.1:8080"
    model: "audio-cpp/pocket-tts"
    model_path: "models/audio_cpp/pocket-tts"
    binary_path: null
    device: "cuda"
    timeout: 300
    sample_rate: 24000
    max_concurrent_generations: 1
    auto_download: false
    extra_params:
      managed: false
      allow_remote_base_url: false
      server:
        host: "127.0.0.1"
        port: 8080
        autoselect_port: true
        port_probe_max: 10
        startup_timeout_seconds: 30
        healthcheck_interval_seconds: 0.25
        startup_backoff_seconds: 5
        idle_shutdown_seconds: 900
        terminate_timeout_seconds: 10
        server_config_path: "models/audio_cpp/server.json"
        models_root: "models/audio_cpp"
        shared_scratch_dir: "models/audio_cpp/runtime/scratch"
        lazy_load: true
        device: 0
        threads: 1
        model:
          id: "pocket-tts"
          family: "pocket_tts"
          path: "models/audio_cpp/pocket-tts"
          task: "tts"
          mode: "offline"
          load_options:
            language: "english"
          session_options:
            language: "english"
      retain_request_artifacts: false
      external_voice_reference_mode: "disabled"  # disabled | shared_path
      request_option_allowlist:
        - max_tokens
        - seed
      voices:
        alba:
          upstream_value: "alba"
          request_field: null  # set only after upstream server support is verified
```

The current `ProviderConfig` schema preserves known provider fields plus `extra_params`. The first implementation should either extend that schema deliberately or keep audio.cpp-specific runtime fields under `extra_params` as shown above. Do not place new top-level provider keys in YAML unless the config schema is updated in the same change, because otherwise those fields may be dropped during Pydantic parsing or config serialization.

For managed mode, tldw renders an upstream-compatible server config with explicit model entries. Provider settings such as `backend` are setup/build hints unless upstream server config documents a matching field. The first implementation should support one configured TTS model entry and leave multi-model config expansion for a later pass unless it is trivial inside the config renderer.

For external server mode, `shared_scratch_dir` is required only for requests that need server-local files, such as reference audio. If the external server cannot read the configured scratch directory, `voice_reference` requests must fail with a clear provider validation error. Basic text-to-speech can work with only `base_url`.

External reference-audio support must be opt-in. Because upstream currently documents request-time audio paths as server-local and does not document a cheap file-read probe endpoint, tldw cannot fully verify external server readability during normal initialization. The default `external_voice_reference_mode` should be `disabled`; `shared_path` should be treated as an admin assertion that the external server can read the configured path.

## Setup And Installer Scope

The setup story should support three paths:

1. Existing server: user provides `base_url`; tldw verifies `/health` and `/v1/models`.
2. Managed sidecar: user provides or installer creates `server_binary_path`; tldw starts the server, waits for `/health`, and shuts it down with the app.
3. Full setup/admin flow: helper clones/builds `audio.cpp`, patches `tts_providers_config.yaml`, and guides model installation.

Model installation must be explicit. Upstream ships `tools/model_manager.py` for package listing, package info, and model installation. Some packages are gated, some need Hugging Face tokens, and some require source files or conversion. tldw should wrap or document that tool instead of reimplementing model download logic in the first pass.

No model download should happen silently from normal server startup or a speech request.

Installer behavior:

- Use pure helper functions for repo-root resolution, runtime layout, config rendering, and YAML patching.
- Offer CLI flags for clone/build/config update/model package installation.
- Keep network operations explicit and user/admin initiated.
- Do not put secrets or tokens in generated config files.
- Keep platform support truthful: CUDA server build is the first managed target; CPU/Vulkan/Metal server support is future verification unless upstream server docs change.

Managed sidecar behavior:

- Bind managed sidecars to loopback hosts only.
- Autoselect a free port by default and derive `base_url` from the selected host and port.
- Wait for `/health` with configurable timeout and polling interval.
- Back off after startup failure to avoid tight restart loops.
- Support idle shutdown because upstream keeps loaded models and sessions resident until process exit.
- Keep stdout/stderr handling sanitized and avoid surfacing arbitrary process output in user-facing errors.

## TTS Request Behavior

The adapter translates `TTSRequest` into `audiocpp_server` speech JSON.

Supported first-pass fields:

- `model`: configured audio.cpp model id by default, or request model when it resolves to `audio_cpp`.
- `input`: request text.
- `voice_reference`: staged WAV file path passed as `voice_ref` when the runtime can read the staged file.
- `voice`: mapped to upstream voice ids only through configured `providers.audio_cpp.voices`; generic tldw voices are not passed blindly.
- `response_format`: request `wav` or upstream JSON/base64 WAV, then use tldw's existing conversion path for `mp3`, `opus`, `flac`, `aac`, and `pcm`.
- `speed`: pass only if the configured model advertises or the config explicitly maps it to an allowed request option; otherwise ignore and record metadata.
- `extra_params`: pass only allowlisted scalar request options, initially `max_tokens` and `seed` unless configuration expands the allowlist.

Reference-audio handling:

- Managed sidecar mode can materialize reference bytes under `shared_scratch_dir` because tldw controls both the path and the server.
- External server mode must verify or require a shared scratch directory readable by the external server before accepting `voice_reference`.
- Reference audio should be validated and converted to a server-readable WAV shape before `voice_ref` is passed.
- Temp names must be generated by tldw, not derived from user filenames.
- Files are deleted after the request unless `retain_request_artifacts` is enabled for diagnostics.

Configured voice mappings need a verified upstream request field before they are sent. The current server README documents `voice_ref` for `/v1/audio/speech`, but it does not document a built-in voice field for that route. Until implementation verifies the server source or a real runtime for fields such as `voice`, `voice_id`, or another request key, configured voices should be exposed as catalog metadata only or fail with a clear validation error when requested without a reference audio path.

Streaming semantics:

- Upstream warns that framework-wide streaming inference is not generally supported and models should be treated as offline-only.
- tldw should advertise `supports_streaming: true` only in the compatibility sense required by `/audio/speech` streaming responses, while metadata should state `incremental_streaming: false`.
- For `stream=true`, the adapter generates full audio and returns it as a single chunk through the existing streaming response path.
- Do not claim token/audio incremental streaming until a specific upstream server route and model family are verified.

## Response Handling

Native response:

- Prefer `audio/wav` bytes from `/v1/audio/speech` when the requested tldw output can be served as WAV or converted locally.
- Use `response_format: "json"` only when it materially simplifies metadata or base64 decoding.

Output conversion:

- If tldw request format differs from upstream WAV, convert using existing `AudioConverter` behavior in `TTSServiceV2` or adapter-local conversion consistent with existing providers.
- The adapter response should include provider metadata: `provider=audio_cpp`, upstream `model`, `base_url` redacted to origin only when useful, `managed`, `incremental_streaming=false`, `voice_reference_mode`, and ignored/unsupported request options.
- Advertise only formats the adapter can actually return or convert. The first pass should not advertise `ogg`, `webm`, or `ulaw` just because the public request schema accepts them, unless conversion for those formats is verified and tested.

Voice catalog:

- Expose configured voices from `providers.audio_cpp.voices`.
- Do not try to infer model-owned voice ids from `/v1/models` unless upstream adds a stable field for it.

## Error Handling

Map failures into existing TTS exception types:

- Unreachable server: `TTSNetworkError` or `TTSProviderUnavailableError`.
- `/health` failure or managed startup timeout: `TTSProviderInitializationError`.
- Missing model in `/v1/models`: `TTSModelNotFoundError` or provider configuration error depending on whether the model was request-selected or configured.
- Unsupported format or unsupported reference-audio mode: `TTSValidationError`.
- Upstream 4xx/5xx: `TTSProviderError` with sanitized details.
- Conversion failure: `TTSGenerationError`.

User-facing errors must not include raw request text, secrets, full local model paths, arbitrary stderr, or generated scratch paths. Logs may include sanitized diagnostics and structured error categories.

## Security And Privacy

Primary risks:

- Server-local file path exposure through `voice_ref` and future task routes.
- Arbitrary option passthrough that could reach model/session/server internals.
- Installer clone/build/download operations with network and toolchain side effects.
- Resident model memory pressure after lazy first use.
- License and distribution boundary confusion.

Controls:

- Keep managed config, scratch, and logs under tldw-controlled roots.
- Validate paths with resolved absolute path checks before writing or passing them to the sidecar.
- In external mode, require explicit shared-scratch configuration for file-backed requests.
- Restrict `base_url` to loopback origins by default. Allow non-loopback origins only through an admin-controlled `allow_remote_base_url` setting.
- Use an allowlist for request options.
- Do not accept arbitrary server command args, environment variables, config JSON, or file paths from normal speech requests.
- Do not auto-download models during inference.
- Treat `audio.cpp` as an optional external component; no vendoring or bundled binaries in the first implementation.

## Testing

Default CI should not require a real `audio.cpp` binary or model.

Required tests:

- Unit tests for `AudioCppClient` request construction, health/model parsing, response decoding, and error mapping with mocked HTTP.
- Adapter tests for request translation, configured voice mapping, unsupported generic voice behavior, reference-audio staging/cleanup, stream-compatible single-chunk response, response metadata, and conversion handoff.
- Sidecar supervisor tests with fake process/startup probes.
- Server config renderer tests for model entries, lazy load, CUDA backend fields, and path validation.
- Installer helper tests for runtime layout and `tts_providers_config.yaml` patching.
- Endpoint-level test that `audio_cpp` can be selected through model/provider routing without bypassing TTSServiceV2, using mocks.
- Registry tests for `TTSProvider.AUDIO_CPP`, provider aliases, default adapter mapping, and namespaced model aliases that do not change existing `pocket-tts` routing.
- Config tests proving audio.cpp-specific fields survive load/serialize either through explicit `ProviderConfig` fields or through `extra_params`.
- Security tests for loopback-only default `base_url`, remote opt-in, path containment, and disabled external reference-audio mode.
- Format tests proving unsupported public schema formats are rejected unless conversion is implemented.

Optional real-runtime checks:

- Smoke helper for an installed `audiocpp_server` and configured model.
- Manual verification for managed CUDA server startup, `/health`, `/v1/models`, basic TTS, reference-audio TTS, and shutdown cleanup.

Bandit should run on touched backend/helper paths before implementation completion. For this design-only task, record a non-code skip.

## Documentation

Update or add docs for:

- TTS provider setup guide entry for `audio_cpp`.
- First-time CPU/GPU audio setup pages, with CUDA server caveat and external-server option.
- Admin/setup installer card or equivalent setup UI guidance, if implementation reaches UI.
- Audio API provider catalog docs showing `incremental_streaming=false`.
- Troubleshooting for server unreachable, model missing, shared scratch unreadable, model memory residency, gated model packages, and build prerequisites.

## Approach C Follow-up Path

After Approach A works, move toward Approach C in staged follow-ups:

1. Reuse `AudioCppClient` and `AudioCppSidecarSupervisor` for `/v1/tasks/run`.
2. Add Audio Studio provider adapters for VAD, diarization, source separation, voice conversion, music generation, and pipelines.
3. Add STT provider support around `/v1/audio/transcriptions`.
4. Promote upstream model package discovery and model-manager integration into setup/admin UI.
5. Add broader capability envelopes so WebUI routes can show which `audio.cpp` tasks are installed, configured, and ready.

The key boundary is that TTS must prove the runtime management, path safety, error mapping, and setup story before the generic task platform is introduced.

## Open Questions For Implementation Planning

- Which initial model package should the installer present first: `pocket_tts`, `qwen3_tts_0_6b_base`, or a small non-gated model if upstream supports one well enough?
- Should managed mode support multiple configured TTS models in the first implementation, or one model entry plus later expansion?
- Should the health/status endpoint expose resident-model memory warnings for `audio_cpp` specifically, or rely on provider metadata first?
- Which upstream server request field, if any, should configured built-in voices use for `/v1/audio/speech`?
