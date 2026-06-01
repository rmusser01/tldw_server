# TTS (Text-to-Speech) Module

## 1. Descriptive of Current Feature Set

- Purpose: Unified, production-grade TTS across local and cloud engines with OpenAI-compatible APIs, streaming, and voice management.
- Capabilities:
  - Multi-provider adapters: OpenAI, ElevenLabs, Kokoro (local ONNX), PocketTTS (local ONNX), Higgs, Dia, Chatterbox, VibeVoice, IndexTTS2, NeuTTS; mock provider for tests.
  - Streaming-first synthesis with graceful fallback and configurable error streaming-as-audio.
  - Voice cloning support and user voice management (upload, list, delete, preview).
  - Unified config with provider priority and per-provider settings; env/config.txt/YAML layering.
  - Metrics, rate limiting, and scope-aware auth on endpoints.
- Inputs/Outputs:
  - Input: Text and optional voice reference metadata; OpenAI-compatible JSON (see schema).
  - Output: Streaming or buffered audio bytes in mp3, opus, aac, flac, wav, or raw pcm.
- Related Endpoints:
  - `tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py` - POST `/api/v1/audio/speech`, GET `/api/v1/audio/voices/catalog`
  - `tldw_Server_API/app/api/v1/endpoints/audio/audio_voices.py` - POST `/api/v1/audio/voices/upload`, GET/DELETE `/api/v1/audio/voices/{voice_id}`, and POST `/api/v1/audio/voices/{voice_id}/preview`
- Related Schemas:
  - tldw_Server_API/app/api/v1/schemas/audio_schemas.py:44 — OpenAISpeechRequest
  - tldw_Server_API/app/core/TTS/voice_manager.py:74 — VoiceUploadRequest
  - tldw_Server_API/app/core/TTS/voice_manager.py:89 — VoiceInfo
  - tldw_Server_API/app/core/TTS/voice_manager.py:104 — VoiceUploadResponse

Provider support snapshot (indicative): OpenAI (cloud), ElevenLabs (cloud, cloning), Kokoro (local), PocketTTS (local, cloning), Higgs/Dia/Chatterbox/VibeVoice/NeuTTS (local, cloning), IndexTTS2 (cloud/local). See adapters/ for exact capabilities.

## 2. Technical Details of Features

- Architecture & Data Flow:
  - Entrypoint `TTSServiceV2` orchestrates provider selection, fallback, and streaming; adapters implement provider specifics.
  - `TTSAdapterRegistry` lazily resolves adapters from dotted paths, honors enablement, and manages retry windows for failed initializations.
  - `tts_validation` sanitizes text and validates voice references before generation; `voice_manager` enforces provider-specific sample constraints and quotas.
  - `tts_resource_manager` tracks shared resources (HTTP clients, temp files, memory/GPU) and performs cleanup; circuit breakers guard providers.
- Key Classes/Functions:
  - tldw_Server_API/app/core/TTS/tts_service_v2.py:1 — TTSServiceV2, get_tts_service_v2()
  - tldw_Server_API/app/core/TTS/adapter_registry.py:1 — TTSAdapterRegistry, TTSProvider
  - tldw_Server_API/app/core/TTS/adapters/base.py:1 — TTSAdapter interface and request/response models
  - tldw_Server_API/app/core/TTS/tts_validation.py:1 — Input validation utilities
  - tldw_Server_API/app/core/TTS/voice_manager.py:1 — Upload, registry, quotas for voices
- Configuration:
  - Canonical YAML: `tldw_Server_API/Config_Files/tts_providers_config.yaml` (providers, priority, performance, fallback, logging)
  - Config file: `Config_Files/config.txt` -> `[TTS-Settings]` (canonical defaults: `default_provider`, `default_voice`, `default_speed`, `local_device`)
  - Loader precedence: environment variables > `config.txt` > YAML > defaults
  - Deprecated aliases in `[TTS-Settings]`:
    - `default_tts_provider` -> `default_provider`
    - `default_tts_voice` -> `default_voice`
    - `default_tts_speed` -> `default_speed`
    - `local_tts_device`/`tts_device` -> `local_device`
    - Removal target for aliases: after 2026-06-30
  - Environment:
    - `TTS_STREAM_ERRORS_AS_AUDIO` — override streaming error behavior
    - `TTS_VOICE_REGISTRY_ENABLED` — toggle persistent voice registry backing store (`true` default). Set `false` for compatibility-mode runtime/filesystem registry only; compatibility mode is deprecated and targeted for removal after 2026-12-31.
    - Secrets via `${ENV_VAR}` in YAML
- Concurrency & Performance:
  - Global semaphore `performance.max_concurrent_generations` (default 4); provider-specific limits may apply inside adapters.
  - Streaming chunk size and connection pools are configurable; backoff on retryable failures; optional caching hooks.
- Error Handling:
  - Rich exception taxonomy in `tts_exceptions.py` with retry classification; circuit breaker integration; optional streaming of errors as audio.
- Security:
  - Endpoints use `TokenScopeGuard` and per-route rate limits; inputs validated and sanitized; uploads validated for type/size/path.
- Metrics:
  - Registered counters/histograms (e.g., `tts_requests_total`, `tts_request_duration_seconds`, `tts_fallback_attempts`, text length/audio size metrics) via Metrics registry.

## 2.1 Egress and Error Behavior (Ops Overview)

- Outbound HTTP for TTS providers (OpenAI, ElevenLabs, Index-style engines) is centralized in `http_client.py`:
  - All adapter POSTs either use the `apost`/`afetch` helpers or the shared `AsyncClient` from `tts_resource_manager`, which calls `_validate_egress_or_raise(url)` before sending.
  - Egress policy is configured via `EGRESS_ALLOWLIST` / `EGRESS_DENYLIST` / `WORKFLOWS_EGRESS_*` and enforces:
    - Allowed schemes (`http/https`) and ports.
    - Host allow/deny lists.
    - Private/reserved IP blocking (SSRF protection), except for selected test-only relaxations.
  - Denials raise `EgressPolicyError` and increment `http_client_egress_denials_total` with a reason (e.g., `Host could not be resolved`, `URL resolves to a private or reserved address`).

- Provider-specific HTTP behavior:
  - **OpenAI**
    - Uses `apost` for `/v1/audio/speech` POSTs (non-stream and streaming), so:
      - Egress is always enforced, but the underlying `httpx.HTTPStatusError` is preserved for mapping.
      - `401` → `TTSAuthenticationError`, `429` → `TTSRateLimitError` (with `retry_after`), other 4xx/5xx → `TTSProviderError` with details.
      - Transport failures (`ConnectError`, DNS/TLS issues, retry exhaustion) surface as `TTSNetworkError` or `TTSTimeoutError` and are treated as retryable at the service layer.
  - **ElevenLabs**
    - Non-streaming generation uses `afetch` for `POST /text-to-speech/{voice_id}`.
    - Streaming uses `AsyncClient.stream("POST", ...)` with `_validate_egress_or_raise(url)` preflight.
    - Error mapping in `_raise_mapped_http_error`:
      - `401/403` → `TTSAuthenticationError`.
      - `429` with `status=rate_limit_exceeded` → `TTSRateLimitError` (with `retry_after` if present).
      - `429` with `status=quota_exceeded` → `TTSQuotaExceededError`.
      - `400` with `status=invalid_voice_id` → `TTSValidationError` (“Invalid voice id” / provider message).
  - **Kokoro**
    - Purely local; no HTTP egress.
    - Errors are dominated by model/file issues (`TTSModelNotFoundError`, `TTSModelLoadError`) and resource problems (`TTSResourceError`, `TTSInsufficientMemoryError`).
    - Optional phoneme overrides via YAML/JSON config or per-request parameters:
      - Load from `Config_Files/tts_phonemes.yaml` (configurable via env `TTS_PHONEME_OVERRIDES_PATH`).
      - Entry format: `{term, phonemes, lang?, boundary?, provider?}`.
      - Request-level overrides (via `extra_params.phoneme_overrides`) take precedence over config-file overrides.
      - Sample configuration: `Config_Files/tts_phonemes.sample.yaml`.

- Service-level behavior (`TTSServiceV2.generate_speech`):
  - Wraps adapter errors into the unified TTS exception taxonomy and records metrics:
    - `tts_requests_total{provider,model,voice,format,status}` for success/failure.
    - `tts_request_duration_seconds`, `tts_text_length_characters`, and `tts_audio_size_bytes` (on success).
    - `tts_ttfb_seconds{provider,voice,format}` and `voice_to_voice_seconds{provider,route}` for latency slicing.
  - Fallback:
    - For retryable errors (`TTSNetworkError`, `TTSTimeoutError`, `TTSRateLimitError`, selected `TTSProviderError`), the service can attempt a fallback provider and increments `tts_fallback_attempts{from_provider,to_provider,success}`.
    - Categorized fallback outcomes are emitted via `tts_fallback_outcomes_total{from_provider,to_provider,outcome,category}`.
  - For non-retryable errors (validation, auth, configuration), no fallback is attempted by default.
  - Streaming vs. HTTP errors:
    - Default: `_stream_errors_as_audio == False` → errors propagate as structured HTTP responses / raised exceptions; streaming generators raise on failure.
    - Legacy/compat mode (`TTS_STREAM_ERRORS_AS_AUDIO=1` or `performance.stream_errors_as_audio=true`): streaming paths emit `"ERROR: ..."` chunks instead of raising, which is useful when mirroring OpenAI’s error-as-audio behavior.
    - `/api/v1/audio/speech` enforces that at least one non-empty audio chunk is produced:
      - If streaming yields no data, the endpoint logs and returns HTTP 500 “Audio generation failed to produce data.”
      - Non-streaming mode accumulates all chunks and similarly errors on empty output.

**Operational tips**

- When debugging TTS failures:
  - Check logs for `TTSServiceV2` messages (“Error generating speech with …”) and provider adapter logs; note the provider, model, and endpoint.
  - Inspect `tts_requests_total` / `tts_fallback_attempts` and HTTP client metrics to distinguish provider errors from egress/transport issues.
  - For OpenAI/ElevenLabs:
    - `TTSAuthenticationError` → misconfigured API key.
    - `TTSRateLimitError` (429) with non-zero `retry_after` → back off or adjust quotas.
    - `TTSQuotaExceededError` → account-level quota; requires provider-side changes.
    - `TTSNetworkError` / `TTSTimeoutError` → network connectivity, DNS, or upstream instability.
  - For Kokoro/local engines:
    - `TTSModelNotFoundError` / `TTSModelLoadError` → verify local model paths and dependencies.
    - `TTSInsufficientMemoryError` / `TTSResourceError` → adjust concurrency, model size, or host resources.

## 3. Developer-Related/Relevant Information for Contributors

- Folder Structure:
  - adapters/ — concrete providers (cloud + local) using `adapters/base.py` contract
  - tts_service_v2.py — orchestration (selection, fallback, streaming, metrics)
  - adapter_registry.py — registry/factory and provider enum
  - tts_config.py — unified configuration manager (env/config.txt/YAML)
  - voice_manager.py — user voice storage/validation/quotas + preview
  - app/core/DB_Management/Voice_Registry_DB.py — persistent per-user voice registry (`voice_registry.db`)
  - audio_converter.py, streaming_audio_writer.py — format conversion and stream shaping
- Extension Points:
  - Add a provider by creating `adapters/<name>_adapter.py` implementing `TTSAdapter` methods, register in `adapter_registry.py` (enum + DEFAULT_ADAPTERS), and add defaults to YAML.
  - Expose capabilities and supported formats via `get_capabilities()` so voice catalog and routing behave correctly.
- Tests:
  - Suites: `tldw_Server_API/tests/TTS/`, `tldw_Server_API/tests/TTS_NEW/`
  - Run: `python -m pytest tldw_Server_API/tests/TTS -v` and `python -m pytest tldw_Server_API/tests/TTS_NEW -v`
  - Adapters: see `tldw_Server_API/tests/TTS/adapters/` for mocks and integration stubs.
- Local Dev Tips:
  - Verify ffmpeg and soundfile availability; configure provider keys in `.env` or YAML; set `TTS_STREAM_ERRORS_AS_AUDIO=0` to use HTTP error codes during development.
  - Quick synth: POST `/api/v1/audio/speech` with `OpenAISpeechRequest` JSON; for streaming, set `stream=true`.
- Pitfalls & Gotchas:
  - Some providers require specific sample rates or short reference durations (e.g., Higgs 3–10s); voice uploads enforce provider constraints.
  - Missing or misconfigured adapters are skipped after failure; optional retry window controlled by `adapter_failure_retry_seconds`.
  - Quotas/rate limits may short-circuit requests; check Usage/Audio quota logs when debugging.
- Follow-up candidates:
  - AllTalk adapter (enum/config placeholder only; requests currently return "provider not configured").
  - Adaptive chunk shaping; provider health probes and proactive warmups; richer voice metadata unification.

Example: programmatic usage

```python
from tldw_Server_API.app.core.TTS.tts_service_v2 import get_tts_service_v2
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest

async def synthesize():
    service = await get_tts_service_v2()
    req = OpenAISpeechRequest(input="Hello from TLDW", model="tts-1", voice="alloy", response_format="mp3", stream=True)
    async for chunk in service.generate_speech(req):
        handle(chunk)
```

Additional References

- tldw_Server_API/app/core/TTS/TTS-DEPLOYMENT.md:1
- tldw_Server_API/app/core/TTS/TTS-VOICE-CLONING.md:1
- Docs/STT-TTS/TTS-SETUP-GUIDE.md:1
- Docs/Product/TTS_Module_PRD.md:1
