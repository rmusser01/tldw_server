# Fish S2 Commercial API Design

## Context

The existing `codex/fish-s2-pro-tts` worktree adds a `fish_s2` TTS provider that targets a self-hosted Fish Speech HTTP server. That branch is internally consistent and its Fish-focused tests pass, but the hosted Fish Audio commercial API has a different contract:

- `POST https://api.fish.audio/v1/tts`
- `Authorization: Bearer <token>`
- required `model` request header, such as `s2-pro`
- JSON/MessagePack TTS body fields including `reference_id`, `references`, `prosody`, `sample_rate`, `mp3_bitrate`, `opus_bitrate`, `latency`, `max_new_tokens`, `min_chunk_length`, `condition_on_previous_chunks`, and `early_stop_threshold`
- reusable voice creation through `POST /model`, not `/v1/references/add`

## Goal

Add hosted Fish Audio commercial S2 TTS support without removing or weakening the existing self-hosted `native_http` backend.

## Recommended Approach

Add a second Fish backend named `commercial_api` behind the existing `FishS2Adapter`. The provider key remains `fish_s2`; configuration selects the backend:

```yaml
providers:
  fish_s2:
    enabled: true
    backend: "commercial_api"
    base_url: "https://api.fish.audio"
    api_key: ${FISH_AUDIO_API_KEY}
    model: "s2-pro"
```

This keeps existing API clients on one provider name and lets operators choose self-hosted or hosted Fish by config. It also keeps the adapter-level model aliases (`fish_s2`, `fish-s2-pro`, `s2-pro`, `fishaudio/s2-pro`) intact.

## Components

- `fish_s2_base.py`: extend the backend protocol so reference creation can accept voice metadata and return a hosted model ID.
- `fish_s2_commercial_api.py`: new HTTP backend for Fish Audio hosted API.
- `fish_s2_native_http.py`: retain existing behavior for self-hosted Fish Speech.
- `fish_s2_adapter.py`: instantiate `commercial_api` when configured and pass commercial generation fields through.
- `tts_service_v2.py`: keep local `voice_id` to remote Fish model ID mapping, but allow the backend to choose the remote ID returned by `/model`.
- `tts_config.py`: add Fish API key environment override support.
- `tts_validation.py`: allow Fish-supported hosted output formats such as `opus`; backend-specific constraints stay in the backend layer.
- `audio_voices.py`: keep the existing user-scoped Fish reference endpoints, but document that commercial mode creates Fish hosted models.

## Data Flow

1. A normal `/api/v1/audio/speech` request resolves to provider `fish_s2`.
2. The adapter builds a provider request and delegates to the configured backend.
3. In `commercial_api` mode, the backend sends JSON to `/v1/tts`, sets `Authorization` and `model` headers, and returns audio bytes or an async stream.
4. For `voice=custom:<voice_id>` or local `extra_params.reference_id`, `TTSServiceV2` resolves local voice metadata first.
5. If metadata has `provider_artifacts.fish_s2.remote_reference_id`, that hosted model ID is sent as Fish `reference_id`.
6. If no hosted ID exists and a user explicitly syncs the voice, the service calls backend reference creation, stores the returned Fish model `_id`, and reuses it on later TTS requests.

## Error Handling

Commercial API status mapping follows existing TTS exceptions:

- `401` and `403`: `TTSAuthenticationError`
- `402`: `TTSProviderError` with payment/quota context
- `429`: `TTSRateLimitError`
- `400`, `404`, and `422`: `TTSValidationError`
- `408` and `504`: `TTSTimeoutError`
- `5xx`: `TTSProviderError`
- transport failures: `TTSNetworkError`

The implementation must not log API keys or uploaded audio.

## Testing

Use TDD:

- backend tests for `/v1/tts` commercial payload/header construction
- backend tests for streaming helper use
- backend tests for `/model` multipart creation and returned `_id`
- backend tests for auth/payment/rate-limit/validation error mapping
- adapter tests for selecting `commercial_api`
- service tests for persisting a remote Fish `_id` returned by the commercial backend
- endpoint tests to keep user-scoped Fish reference routes stable

## Non-Goals

- Do not implement Fish WebSocket realtime streaming in this slice.
- Do not remove the self-hosted Fish Speech backend.
- Do not add a frontend provider management UI in this slice.
- Do not perform live Fish API calls in automated tests.
