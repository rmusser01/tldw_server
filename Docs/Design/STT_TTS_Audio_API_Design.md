# STT/TTS Audio API Design

Status: Implemented
Covered by: `Docs/ADR/011-audio-api-semantics.md`
Last verified against code: 2026-02-22 (`915632a97ad41dc7712101612113928dbea6b358`)

## Purpose and scope

This design doc is the architecture reference for:

- `POST /api/v1/audio/speech`
- `POST /api/v1/audio/transcriptions`
- `WS /api/v1/audio/stream/transcribe`

It exists so the user guide can point to one implementation-aligned source for:

- provider selection order and adapter retry behavior
- auth mode behavior (single-user and multi-user)
- storage download-link headers for TTS
- streaming protocol and error semantics

## Related documents

- [STT_Parakeet_MLX_Parity.md](./STT_Parakeet_MLX_Parity.md) - parity requirements for STT outputs between Parakeet MLX and existing response formats.
- [2026-02-25-parakeet-onnx-default-transcription-design.md](./2026-02-25-parakeet-onnx-default-transcription-design.md) - default runtime/provider decisions for Parakeet ONNX transcription.
- [Workspace_Persistence_Architecture.md](./Workspace_Persistence_Architecture.md) - persistence and recovery model used by clients that invoke audio APIs across sessions.
- [Meeting_Intelligence_API.md](./Meeting_Intelligence_API.md) - downstream consumers of transcript and artifact generation patterns from audio pipelines.

## Decision summary

1. Auth is centralized: HTTP uses `get_request_user`, WebSocket uses `_audio_ws_authenticate`.
2. TTS provider routing is model-first, then fallback/priority-based.
3. Failed adapter initialization can be retried using a cooldown window.
4. Streaming TTS errors default to structured failures, not embedded audio, unless explicitly enabled.
5. `return_download_link` is non-streaming-only and adds storage headers while still returning audio bytes.

## Auth modes and credential semantics

### HTTP endpoints (`/audio/speech`, `/audio/transcriptions`)

Auth dependency: `get_request_user` (`tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`).

- Accepted credentials:
  - `X-API-KEY: <key>`
  - `Authorization: Bearer <token>`
- In `single_user` mode, Bearer values are treated as API keys.
- In `multi_user` mode, Bearer JWT is preferred; non-JWT Bearer can be treated as API key compatibility path.
- Missing credentials return `401` with detail `"Not authenticated (provide Bearer token or X-API-KEY)"`.

### WebSocket endpoint (`/audio/stream/transcribe`)

Auth helper: `_audio_ws_authenticate` (`tldw_Server_API/app/core/Audio/streaming_service.py`).

- Multi-user supports:
  - `X-API-KEY` header
  - `Authorization: Bearer <JWT>`
  - `?token=` query parameter (API key or JWT path)
  - first-frame auth fallback (`{"type":"auth","token":"..."}`) for JWT
- Single-user supports:
  - `X-API-KEY` header
  - Bearer token matching single-user API key
  - `?token=` query parameter
  - first-frame auth message (`{"type":"auth","token":"<single_user_key>"}`)

## Provider selection and adapter retry behavior

### Provider selection flow

Primary model-to-provider routing is in `TTSAdapterFactory.MODEL_PROVIDER_MAP` (`tldw_Server_API/app/core/TTS/adapter_registry.py`).

- Explicit model examples:
  - `tts-1` -> `openai`
  - `kokoro` -> `kokoro`
- If model mapping is unavailable, provider aliases are resolved.
- Fallback adapter search uses capability requirements and registry ordering.

Priority order key:

- `provider_priority` in `tldw_Server_API/Config_Files/tts_providers_config.yaml`
- `TTSConfigManager.get_provider_priority()` filters that list to enabled providers only.

### Adapter init retry cooldown

Key:

- `performance.adapter_failure_retry_seconds` (same YAML file)

Behavior:

- Failed provider initialization is marked failed in shared provider registry.
- If retry seconds is configured and positive, provider is skipped until cooldown expires, then retried.
- If unset or `<= 0`, failure is treated as effectively permanent for process lifetime (until restart/reset).

### Streaming error mode decision

Keys:

- `performance.stream_errors_as_audio` (YAML)
- `TTS_STREAM_ERRORS_AS_AUDIO` (env override)

Behavior in `TTSServiceV2` (`tldw_Server_API/app/core/TTS/tts_service_v2.py`):

- Default: `False` (structured failures / raised errors).
- If `True`: generator can emit chunks like `ERROR: ...` as audio bytes for compatibility mode.

## Endpoint design

### `POST /api/v1/audio/speech`

Implementation: `tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py`.

Core behavior:

- OpenAI-compatible request body (`OpenAISpeechRequest`).
- `stream` defaults to `true`.
- `return_download_link` requires `stream=false`; otherwise `400`.
- Streaming mode returns `StreamingResponse` chunks.
- Non-streaming mode buffers all bytes and returns one audio response.

Example request (as used in getting-started guide):

```json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": "Hello from tldw_server",
  "response_format": "mp3",
  "stream": false,
  "return_download_link": true
}
```

### Storage header semantics (`return_download_link`)

When `stream=false` and `return_download_link=true`:

- server persists generated audio via storage registration path
- response still contains audio bytes in body
- response headers include:
  - `X-Download-Path: /api/v1/storage/files/{id}/download`
  - `X-Generated-File-Id: {id}`

When `stream=true` and `return_download_link=true`:

- request is rejected with `400` (`"return_download_link requires stream=false"`).

These semantics are tested in:

- `tldw_Server_API/tests/Storage/test_tts_storage_integration.py`

### `POST /api/v1/audio/transcriptions`

Implementation: `tldw_Server_API/app/api/v1/endpoints/audio/audio_transcriptions.py`.

Core behavior:

- OpenAI-compatible multipart upload endpoint.
- Uses STT registry/provider resolution based on model.
- Supports `response_format` values: `json`, `text`, `srt`, `verbose_json`, `vtt`.
- Applies per-user file-size/concurrency/daily-minute checks.
- Returns typed HTTP errors for provider/model availability and transient failures.

Example request (guide-aligned):

```bash
curl -X POST http://127.0.0.1:8000/api/v1/audio/transcriptions \
  -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -F "file=@sample.wav" \
  -F "model=whisper-large-v3" \
  -F "language=en"
```

### `WS /api/v1/audio/stream/transcribe`

Implementation: `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`.

Core behavior:

- Real-time transcription over WebSocket with per-user stream + minute quota enforcement.
- Uses default streaming config if client does not send config first.
- Optional transcript persistence to media DB when enabled by query/config hints.

Client message types:

- `auth` (fallback token frame)
- `config`
- `audio` (base64 audio chunk)
- `commit`

Server frame types:

- `partial`
- `transcription`
- `full_transcript`
- `warning`
- `error`

Quota/error behavior:

- quota breach sends error payload (`code: "quota_exceeded"`, quota metadata)
- connection closes with `4003` by default
- if `AUDIO_WS_QUOTA_CLOSE_1008=1`, close code is `1008`
- compatibility alias `error_type` can be included when `AUDIO_WS_COMPAT_ERROR_TYPE=1`

Example connect:

```bash
wscat -c ws://127.0.0.1:8000/api/v1/audio/stream/transcribe \
  -H "X-API-KEY: $SINGLE_USER_API_KEY"
```

## Guide-snippet config key reference

All keys below are defined in:

- `tldw_Server_API/Config_Files/tts_providers_config.yaml`

Keys:

- `provider_priority`
- `performance.adapter_failure_retry_seconds`
- `performance.stream_errors_as_audio`

## Implementation map (for code/design cross-reference)

- `tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py`
- `tldw_Server_API/app/api/v1/endpoints/audio/audio_transcriptions.py`
- `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`
- `tldw_Server_API/app/core/Audio/streaming_service.py`
- `tldw_Server_API/app/core/TTS/adapter_registry.py`
- `tldw_Server_API/app/core/TTS/tts_service_v2.py`
- `tldw_Server_API/app/core/TTS/tts_config.py`
- `tldw_Server_API/app/core/Infrastructure/provider_registry.py`
- `tldw_Server_API/tests/Storage/test_tts_storage_integration.py`
- `tldw_Server_API/tests/TTS_NEW/unit/service/test_tts_error_streaming_policy.py`
