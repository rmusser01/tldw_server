# Audio: STT and TTS (server)

This document describes the audio surface: the API-facing normalization layer (model aliases, input sanitization, quotas, streaming plumbing), the TTS v2 provider stack with its adapter registry and gateway fallback, the long-form TTS jobs worker, OpenAI-compatible transcriptions, and realtime streaming.

## Authoritative files

All paths under `tldw_Server_API/` unless noted.

| Concern | File | Key symbols |
| --- | --- | --- |
| Normalization layer | `app/core/Audio/{transcription_service,tts_service,streaming_service}.py` | whisper model-alias mapping, `_sanitize_speech_request`, provider inference, exception→HTTP mapping; the WS streaming runtime (JWT validation, IP policy, close codes) |
| Taxonomies | `app/core/Audio/{dictation_error_taxonomy,error_payloads,quota_helpers,tokenizer_service}.py` | canonical STT error classes, quota fail-open helpers, Qwen3-TTS tokenizer |
| Realtime | `app/core/Audio/Realtime/` | realtime transcription session machinery |
| TTS core | `app/core/TTS/tts_service_v2.py`, `adapter_registry.py`, `adapters/` | `TTSServiceV2` (provider selection, fallback, streaming, metrics); lazy dotted-path adapter resolution with retry windows; adapters: OpenAI, ElevenLabs, Kokoro, PocketTTS, Higgs, Dia, Chatterbox, VibeVoice, IndexTTS2, NeuTTS, mock |
| Gateway | `app/core/TTS/{gateway_config,gateway_execution,gateway_catalog,gateway_preflight}.py` | provider/model/voice format specs, the speech executor, fallback policy, preflight |
| Support | `app/core/TTS/{tts_validation,voice_manager,tts_resource_manager,circuit_breaker,tts_exceptions}.py` | input validation, voice registry/quotas, shared resources, breakers, exception taxonomy |
| Long-form | `app/core/TTS/tts_jobs_worker.py` | the `tts_longform` jobs worker (max 3 retries) |
| Routes | `app/api/v1/endpoints/audio/*.py` | `/api/v1/audio`: transcriptions/translations, speech + speech/jobs, voices, presets, jobs, history, health, studio, voice conversion, tokenizer, audiobooks, realtime + streaming WS routes (dynamically imported) |
| Adjacent cores | `app/core/Audio_Studio/`, `app/core/Audiobooks/` | studio render/export jobs; audiobook alignment/subtitles/tag parsing |

## TTS request (dataflow)

1. `POST /api/v1/audio/speech` — auth + token-scope guard + rate limit.
2. Sanitization: Unicode normalization, HTML stripping; strict mode (default) rejects dangerous patterns with 400, relaxed strips (configurable).
3. Provider resolution: explicit backend → gateway preflight validates model/voice/format against specs and resolves fallback; else provider inferred from the model id with BYOK credential resolution.
4. `TTSServiceV2.generate_speech`: adapter from the registry (breaker-guarded), a global concurrency semaphore (default 4), streaming or buffered output, fallback chain per policy.
5. All provider egress goes through the shared HTTP client's egress validation (SSRF allow/deny lists).
6. Errors map through the TTS exception taxonomy (auth, rate-limit, quota, model-not-found/load, …) to HTTP.
7. Response: audio bytes with optional sample-rate and alignment headers; metrics record totals, durations, TTFB.
8. Long-form: `POST /speech/jobs` enqueues an audio-domain `tts_longform` job forcing non-streaming; artifacts are listed per job. The job goes on the `tts` queue. `TTS_JOBS_QUEUE` overrides it: set the same value for the API server and the TTS worker, and register a custom name in `JOBS_ALLOWED_QUEUES_AUDIO`. `default` and `transcribe` are refused, because the audio pipeline workers read them without a job-type filter. The TTS jobs worker reads that queue and takes only `tts_longform` jobs. It runs in-process wherever the audio routes do (`TTS_JOBS_WORKER_ENABLED`, default on by route policy); in sidecar mode it runs as the `tts` sidecar worker (`Dockerfiles/docker-compose.workers.yml`), and it can also run standalone as `python -m tldw_Server_API.app.core.TTS.tts_jobs_worker`.

## STT (condensed)

`POST /api/v1/audio/transcriptions` (OpenAI-compatible multipart): quota pre-checks (a pessimistic monthly-minute check plus a daily call check) with usage trued up post-transcription; whisper model aliases map (`whisper-1` → `large-v3`; unknown models also fall back to `large-v3`); backends include faster_whisper, NeMo (Parakeet/Canary), Qwen2Audio, and VibeVoice-ASR; `task=transcribe|translate`; optional TreeSeg segmentation; realtime via the WS routes.

## Boundaries

- `app/core/Audio/` is API-facing normalization; the heavy providers live in `app/core/TTS/` and the ingestion audio processor.
- TTS job requests reject embedded credential fields outright.
- Voice uploads are managed by the voice manager (type/size/path validation, quotas).

## Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| Dangerous input, strict validation | 400 |
| Missing provider credentials | Typed auth error |
| Provider 401/429/quota | Mapped auth/rate-limit/quota exceptions |
| Circuit open / adapter init failure | Breaker error; registry retry window; fallback chain |
| Mid-stream synthesis error | Optionally streamed as audio (flag-gated) |
| STT quota exhausted | Pre-check rejects before processing |
| WS auth failure | Close code 4401 (policy close) |

## Config keys (selection)

Canonical provider config: `Config_Files/tts_providers_config.yaml`; `[TTS-Settings]` keys `default_provider/default_voice/default_speed/local_device` (legacy aliases slated for removal); secrets via `${ENV_VAR}` in YAML with precedence env > config.txt > YAML > defaults. Env: `TTS_STREAM_ERRORS_AS_AUDIO`, `TTS_VOICE_REGISTRY_ENABLED`, `TTS_PHONEME_OVERRIDES_PATH`; egress `EGRESS_ALLOWLIST/_DENYLIST`; quotas via the billing limit categories (daily API calls, monthly transcription minutes).

## Verified gotchas

1. `whisper-1` silently maps to `large-v3`, and so does **every** unknown model name — a typo'd model doesn't error, it transcribes with large-v3.
2. STT quota is pre-checked pessimistically at one minute and trued up after — a request can pass pre-check and still exceed the month quota.
3. `app/core/TTS/TTS-README.md` and `app/core/TTS/README.md` duplicate content; the audio streaming module is ~200 KB and dynamically imported (tests patch the shims).
4. `Docs/STT-TTS/` is the setup-guide family (NeMo, Qwen3, VibeVoice, Chatterbox…); ADR-011 (audio API semantics) and ADR-047 (per-user preset ownership) govern.

## Related docs

- [ingestion.md](./ingestion.md) — ingestion-time audio processing
- [jobs-background.md](./jobs-background.md) — the long-form worker substrate
