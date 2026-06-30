# Audio

Audio provides API-facing support services for transcription, translation, text-to-speech, streaming speech chat, tokenizer utilities, quota handling, and audio-specific error mapping. The heavy provider implementations live in adjacent modules such as `TTS` and `Ingestion_Media_Processing`; this module normalizes request handling, credentials, model aliases, quotas, streaming queues, and endpoint error responses.

## Start Here

- `transcription_service.py` maps OpenAI-compatible transcription models to local Whisper/faster-whisper model identifiers.
- `tts_service.py` sanitizes speech requests, resolves provider credentials, infers TTS providers from model ids, and maps TTS exceptions to HTTP errors.
- `streaming_service.py` contains shared WebSocket streaming helpers, queue/backpressure behavior, and metrics hooks.
- `quota_helpers.py` centralizes audio quota and fail-open policy helpers.
- `tokenizer_service.py` handles Qwen3-TTS tokenizer encode/decode helpers.
- Related API surface: `tldw_Server_API/app/api/v1/endpoints/audio/`.
- Related schemas: `audio_schemas.py`, `audio_health.py`, and `audio_presets.py`.
- Related tests: `tldw_Server_API/tests/Audio/` and `tldw_Server_API/tests/AudioJobs/`.

## Responsibilities

- Normalize OpenAI-compatible transcription and translation requests.
- Map model aliases such as `whisper-1` to configured local transcription backends.
- Validate and sanitize TTS input before provider execution.
- Resolve BYOK or configured provider credentials for TTS providers.
- Convert TTS and streaming exceptions into consistent HTTP response payloads.
- Maintain streaming queue, status, limit, and fail-open behavior for audio chat flows.
- Apply quota helpers and bounded fail-open policy for audio usage.
- Encode and decode Qwen3-TTS tokenizer payloads with configured limits.

## Module Map

- `transcription_service.py`: transcription model alias handling and transcription service helpers.
- `tts_service.py`: TTS request sanitization, provider inference, credential resolution, and exception mapping.
- `streaming_service.py`: WebSocket streaming runtime helpers, queue behavior, and metrics registration.
- `quota_helpers.py`: quota lookup, quota exception handling, and fail-open minutes.
- `tokenizer_service.py`: tokenizer request limits and audio token serialization helpers.
- `dictation_error_taxonomy.py`: canonical dictation and STT error classes plus fallback policy.
- `streaming_exceptions.py`, `error_payloads.py`: audio exception and HTTP detail helpers.

## How It Connects

- `endpoints/audio/audio.py` aggregates TTS, history, presets, tokenizer, transcriptions, health, and voices routers and dynamically loads streaming routes.
- `audio_tts.py` calls `tts_service.py` and the `TTS` core providers for `/speech` and related speech endpoints.
- `audio_transcriptions.py` connects this module to `Ingestion_Media_Processing.Audio` transcription adapters and policy helpers.
- `audio_streaming.py` uses `streaming_service.py` for speech chat, stream status, limits, and WebSocket behavior.
- Audio endpoints use AuthNZ dependencies, BYOK runtime helpers, usage quota modules, metrics, Jobs for audio job queues, and ChaChaNotes DB for speech chat history.
- Provider-specific work is delegated to adjacent TTS, STT, LLM, and ingestion modules rather than implemented directly here.

## Extension Points

- Add a transcription model alias in `transcription_service.py` and cover it with audio transcription tests.
- Add TTS provider inference in `_infer_tts_provider_from_model` inside `tts_service.py`.
- Add or adjust TTS HTTP error behavior in `_raise_for_tts_error`.
- Change quota behavior in `quota_helpers.py` and check audio quota tests.
- Extend streaming behavior in `streaming_service.py` and endpoint tests that patch streaming imports.
- Extend tokenizer support in `tokenizer_service.py` and tokenizer endpoint tests.

## Testing

- Direct audio coverage lives under `tldw_Server_API/tests/Audio/`.
- Audio job API coverage lives under `tldw_Server_API/tests/AudioJobs/`.
- Relevant tests include transcription endpoint tests, TTS error mapping tests, WebSocket and stream status tests, audio quota tests, preset endpoint tests, tokenizer tests, and audio router import-resilience tests.
- Provider-heavy behavior is commonly mocked because several audio backends are optional local or external dependencies.

## Gotchas

- Minimal test and deployment profiles may not install heavy audio dependencies; keep imports lazy or guarded where existing code does so.
- Streaming routes are dynamically loaded by the aggregate audio router, so import-time side effects can break minimal startup.
- Fail-open quota behavior is bounded by configuration and should not be treated as unlimited access.
- Missing cloud provider credentials are intentionally surfaced as provider credential errors rather than silently falling back to another provider.
