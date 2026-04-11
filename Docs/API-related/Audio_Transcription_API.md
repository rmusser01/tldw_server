# Audio Transcription API Documentation

## Overview

The tldw_server provides a comprehensive audio transcription API that is fully compatible with OpenAI's Audio API while offering additional transcription engines including NVIDIA Nemo models (Canary and Parakeet) for improved performance and flexibility.

## User Guide Map

- [Getting Started — STT and TTS](../User_Guides/WebUI_Extension/Getting-Started-STT_and_TTS.md) — quickstart for first successful speech requests.
- [TTS Providers Getting Started](../User_Guides/WebUI_Extension/TTS_Getting_Started.md) — provider selection and first successful synthesis.
- [TTS Provider Setup Guide](../User_Guides/WebUI_Extension/TTS-SETUP-GUIDE.md) — runbook index for deep provider setup/tuning.
- [Qwen3-ASR Setup Guide](../STT-TTS/QWEN3_ASR_SETUP.md) — Qwen3-ASR model setup details.

## Auth + Rate Limits
- Single-user: `X-API-KEY: <key>`
- Multi-user: `Authorization: Bearer <JWT>`
- Transcriptions/Translations: 20 requests/minute, keyed per user when authenticated (falls back to IP).
- Real-time WebSocket transcription: per-user concurrent stream limits and daily minutes quotas enforced.

## Table of Contents
- [Features](#features)
- [Supported Models](#supported-models)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Live Transcription](#live-transcription)
- [Usage Examples](#usage-examples)
- [Performance Comparison](#performance-comparison)
- [Notes & Limitations](#notes--limitations)

## Features

### Core Capabilities
- **OpenAI API Compatible**: Drop-in replacement for OpenAI's audio transcription endpoints
- **Multiple Transcription Engines**: Choose from faster-whisper, NVIDIA Nemo models, Qwen2Audio, Qwen3-ASR, or VibeVoice
- **Live Transcription**: Real-time audio streaming with VAD and silence detection
- **Model Optimization**: Support for ONNX and MLX variants for better performance
- **Multi-format Support**: Handle various audio formats (WAV, MP3, M4A, etc.)
- **Response Formats**: JSON, text, SRT, VTT, verbose JSON

### Advanced Features
- **Voice Activity Detection (VAD)**: Intelligent speech segmentation
- **Streaming Support**: Process long audio files efficiently
- **Language Detection**: Automatic language identification (Whisper). When no `language` is provided, the API returns the detected language in JSON.
- **Partial Transcriptions**: Get interim results during live transcription
- **Model Caching**: Efficient model management for repeated use

## Supported Models

### 1. Whisper (faster-whisper)
- **Model**: `whisper-1` (OpenAI compatible name)
- **Variants**: tiny, base, small, medium, large-v3
- **Languages**: 99+ languages
- **Best For**: General-purpose transcription, multi-language support

### 2. NVIDIA Canary-1b
- **Model**: `canary`
- **Size**: 1 billion parameters
- **Languages**: English, Spanish, German, French
- **Best For**: Multi-lingual transcription with high accuracy
- **Special Features**: Built-in punctuation and capitalization

### 3. NVIDIA Parakeet TDT
- **Model**: `parakeet`
- **Size**: 0.6 billion parameters
- **Variants**:
  - Standard (PyTorch)
  - ONNX (optimized for CPU/GPU)
  - MLX (optimized for Apple Silicon)
- **Languages**: English (primarily)
- **Best For**: Fast, efficient transcription with good accuracy

### 4. Qwen2Audio
- **Model**: `qwen2audio`
- **Size**: 7 billion parameters
- **Languages**: Multiple languages
- **Best For**: Complex audio understanding tasks

### 5. Qwen3-ASR
- **Model**: `qwen3-asr-1.7b`, `qwen3-asr-0.6b`, `qwen3-asr`
- **Variants**:
  - **1.7B** (default): Production quality, ~8-16GB VRAM
  - **0.6B**: Resource-constrained / high-throughput, ~2-4GB VRAM
- **Languages**: 30 languages + 22 Chinese dialects (auto-detected)
- **Best For**: Chinese transcription, high-accuracy multilingual content
- **Special Features**: Optional word-level timestamps via Forced Aligner
- **Note**: Requires manual model download. See [Qwen3-ASR Setup Guide](../STT-TTS/QWEN3_ASR_SETUP.md); for end-to-end first run, start with [Getting Started — STT and TTS](../User_Guides/WebUI_Extension/Getting-Started-STT_and_TTS.md).

### 6. VibeVoice-ASR
- **Model**: `vibevoice-asr`, `vibevoice`
- **Size**: 7 billion parameters
- **Languages**: ~50 languages
- **Best For**: Long-form audio, speaker-aware transcripts, domain-specific vocabularies
- **Special Features**: Built-in diarization metadata, hotwords support

#### Model ID patterns (HTTP + ingestion)

The `model` string for `/api/v1/audio/transcriptions` is parsed via the same logic as the ingestion pipeline (`parse_transcription_model` in `Audio_Transcription_Lib.py`), so the following patterns are accepted:

- **Whisper / faster-whisper**
  - `whisper-1`, `whisper` (aliases for the default faster-whisper Whisper model)
  - Raw faster-whisper ids such as `large-v3`, `distil-whisper-large-v3`, or full HF ids (e.g. `openai/whisper-large-v3`).
- **NVIDIA NeMo Parakeet**
  - `parakeet`, `parakeet-standard`, `parakeet-onnx`, `parakeet-mlx`
  - Any string that `parse_transcription_model` resolves to provider `"parakeet"` (e.g., some `nemo-parakeet-*` ids).
- **NVIDIA NeMo Canary**
  - `canary` (and related aliases whose provider resolves to `"canary"`).
- **Qwen2Audio**
  - `qwen2audio`, `qwen2audio-*` (all map to provider `"qwen2audio"`)
  - Convenience alias `qwen` also maps to `qwen2audio` in the HTTP API.
- **Qwen3-ASR**
  - `qwen3-asr-1.7b`, `qwen3-asr-0.6b`, `qwen3-asr` (all map to provider `"qwen3-asr"`)
  - Bare `qwen3-asr` defaults to the configured model path (typically 1.7B)
  - Underscore variants also accepted: `qwen3_asr_1.7b`, `qwen3_asr_0.6b`
- **VibeVoice-ASR**
  - `vibevoice-asr`, `vibevoice`, `vibevoice_asr` (all map to provider `"vibevoice"`)

## API Endpoints

Authentication
- Single-user mode: send `X-API-KEY: <your_key>`
- Multi-user mode (JWT): send `Authorization: Bearer <JWT>`

Base path
- All endpoints in this document are served under `/api/v1`.

### POST /api/v1/audio/transcriptions

Transcribe audio into text.

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | file | Yes | The audio file to transcribe (default max 25MB; actual limit may vary by quota tier) |
| model | string | No | Model to use. Supported examples: `whisper-1` (`whisper` alias), raw faster-whisper ids like `large-v3` or `distil-whisper-large-v3`; NVIDIA variants such as `parakeet`, `parakeet-onnx`, `parakeet-mlx`; Canary via `canary`; Qwen via `qwen2audio` or `qwen2audio-*`; Qwen3-ASR via `qwen3-asr-1.7b`, `qwen3-asr-0.6b`, or `qwen3-asr`; VibeVoice via `vibevoice-asr` (default when omitted: `[STT-Settings].default_batch_transcription_model`, shipping default `parakeet-onnx`). |
| language | string | No | Language hint. ISO-639-1 codes are always accepted (for example `en`, `es`). BCP-47 locale hints (for example `en-US`, `pt-BR`) are accepted and normalized per provider: providers that require ISO-style hints receive base codes, providers with locale-capable routing keep locale hints. When omitted, Whisper models auto-detect the language and the detected code is included in the JSON response. |
| prompt | string | No | Optional text to guide the model's style |
| response_format | string | No | Output format: `json`, `text`, `srt`, `vtt`, `verbose_json` (default: `json`) |
| temperature | float | No | Sampling temperature 0-1 (default: 0) |
| task | string | No | For Whisper-based models, decoding task: `transcribe` (default) or `translate`. For non-Whisper providers this hint is ignored and a plain transcription is performed. |
| timestamp_granularities | string | No | Comma-separated values or JSON array. Supported tokens: `segment`, `word` |
| segment | boolean | No | If true and JSON response, also run transcript segmentation (TreeSeg) and include `segmentation` in the JSON |
| seg_K | integer | No | Max segments for TreeSeg (default 6) |
| seg_min_segment_size | integer | No | Min items per segment (default 5) |
| seg_lambda_balance | number | No | Balance penalty (default 0.01) |
| seg_utterance_expansion_width | integer | No | Context width per block (default 2) |
| seg_embeddings_provider | string | No | Embeddings provider override (optional) |
| seg_embeddings_model | string | No | Embeddings model override (optional) |

When `timestamp_granularities` includes `word` (Whisper only), each segment includes a `words` array with `{start, end, word}` entries.

Accepted Content-Types:
- `audio/wav`, `audio/x-wav`, `audio/mpeg`, `audio/mp3`, `audio/mp4`, `audio/m4a`, `audio/x-m4a`, `audio/flac`, `audio/ogg`, `audio/opus`, `audio/webm`.
Unsupported types return 415.

**Response (JSON format):**
```json
{
  "text": "Transcribed text here",
  "language": "en",
  "duration": 10.5,
  "segmentation": {
    "transitions": [0,0,1,0],
    "transition_indices": [2],
    "segments": [
      {"indices":[0,1],"start_index":0,"end_index":1,"speakers":[],"text":"..."}
    ]
  },
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 10.5,
      "text": "Transcribed text here"
    }
  ]
}
```

Notes:
- For `response_format: text|srt|vtt` responses, outputs are simple best-effort formats; precise per-segment timings require JSON.
- For `response_format: verbose_json`, the response includes `task` and `duration` fields.
- For Whisper-based models, the underlying `speech_to_text(...)` helper prepends a metadata header (model + detected language) to the first segment. The HTTP API always calls `strip_whisper_metadata_header(...)` before returning JSON/text so clients see only user content. If you use `speech_to_text` directly (e.g., in workflows or custom tools), call `strip_whisper_metadata_header` on segment lists, or `_strip_whisper_metadata_header_from_text` (speech chat) before presenting text to end users.

### Retention and Redaction Policy

- REST transcription resolves an effective STT policy before persistence and response emission.
- In multi-user mode, effective policy is `org override -> global STT defaults`.
- In single-user mode, only the global STT defaults apply.
- Request-level overrides may only be stricter than the effective policy:
  - shorter retention TTL is allowed
  - enabling delete-after-success is allowed
  - enabling redaction or adding redact categories is allowed
  - weakening a tenant-required retention/redaction rule is rejected
- When effective policy requires redaction, the persisted transcript and HTTP response are redacted before serialization.
- Retained raw-audio artifacts are indexed through `generated_files`; when retention is not enabled, delete-after-success remains the default behavior.

### Dictation Error Taxonomy

Structured error payloads include:
- `dictation_error_class`: canonical failure class.
- `dictation_fallback_allowed`: whether automatic fallback (`auto` strategy) is allowed for that class.

Classes:
- `permission_denied`
- `unsupported_api`
- `auth_error`
- `quota_error`
- `provider_unavailable`
- `model_unavailable` (includes `status: model_downloading`)
- `transient_failure`
- `empty_transcript`
- `unknown_error`

Fallback policy:
- Auto-fallback allowed: `unsupported_api`, `provider_unavailable`, `model_unavailable`, `transient_failure`.
- Auto-fallback disallowed: `permission_denied`, `auth_error`, `quota_error`, `empty_transcript`, `unknown_error`.

### Client Dictation Diagnostics (WebUI + Extension)

WebUI `/chat` and extension sidepanel emit a sanitized diagnostics event for dictation strategy transitions:
- Event name: `tldw:dictation:diagnostics`
- Purpose: explain mode resolution and fallback behavior without logging sensitive content.

Payload schema:

| Field | Type | Description |
|---|---|---|
| `version` | number | Schema version (`1`) |
| `at` | string | ISO-8601 timestamp |
| `surface` | string | `playground` or `sidepanel` |
| `kind` | string | `toggle`, `server_error`, or `server_success` |
| `requested_mode` | string | `auto`, `server`, `browser`, or `unknown` |
| `resolved_mode` | string | `server`, `browser`, `unavailable`, or `unknown` |
| `speech_available` | boolean | Whether dictation is available on this surface |
| `speech_uses_server` | boolean | Whether current resolved mode routes through server STT |
| `toggle_intent` | string/null | `start_*`/`stop_*` intent for toggle events |
| `error_class` | string/null | Dictation taxonomy class for terminal server errors |
| `fallback_applied` | boolean | Whether auto-fallback was applied after server error |
| `fallback_reason` | string/null | Error class that triggered fallback, if any |

Privacy contract:
- Diagnostics payloads never include transcript text, prompt text, raw audio, or binary payloads.
- Only strategy state and taxonomy metadata are serialized.

Internal STT helpers:
- `speech_to_text(...)` (file or NumPy input) is the canonical segment-based helper used by media ingestion and offline workers; it returns a list of segments (or `(segments, language)` when requested).
- `transcribe_audio(...)` (NumPy waveform input) is the canonical plain-text helper used by this HTTP endpoint, speech-chat, and streaming sinks; it routes to the configured provider and returns a single transcript string. Provider failures are surfaced as error sentinel strings (for example, `"[Transcription error] Qwen2Audio ..."`), which HTTP handlers detect via `is_transcription_error_message(...)` and map to appropriate HTTP error responses rather than returning the sentinel text as user content.

### Word-level Timestamps Example

When `timestamp_granularities` includes `word`, each segment contains `words` with start/end per tokenized word.

**Supported providers:**
- **Whisper**: Built-in word timestamp support
- **Qwen3-ASR**: Via Forced Aligner (requires `qwen3_asr_aligner_enabled=true` in config)

```json
{
  "text": "hello world",
  "language": "en",
  "duration": 2.1,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.1,
      "text": "hello world",
      "words": [
        { "start": 0.12, "end": 0.42, "word": "hello" },
        { "start": 0.55, "end": 0.92, "word": "world" }
      ]
    }
  ]
}
```

### POST /api/v1/audio/translations

Translate audio into English.

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | file | Yes | The audio file to translate |
| model | string | No | Model to use (default: `whisper-1`) |
| prompt | string | No | Optional text to guide the model's style |
| response_format | string | No | Output format (default: `json`) |
| temperature | float | No | Sampling temperature 0-1 |

For Whisper models, this endpoint internally calls the transcription endpoint
with `task=translate` and no explicit `language`, allowing the backend to
auto-detect the source language and return English output. Non-Whisper
providers treat `task` as a no-op and perform a regular transcription.

## Configuration

### config.txt Settings

Add the following section to your `config.txt`:

```ini
[STT-Settings]
# Explicit defaults when the client omits `model`
default_batch_transcription_model = parakeet-onnx
default_streaming_transcription_model = parakeet-onnx

# Nemo model variant (for Parakeet)
nemo_model_variant = standard
# Options: standard, onnx, mlx

# Parakeet ONNX model source
parakeet_onnx_model_id = istupakov/parakeet-tdt-0.6b-v3-onnx
# Optional: pin exact commit/tag for deterministic downloads
parakeet_onnx_revision =

# Streaming fallback policy (default fail-fast)
streaming_fallback_to_whisper = false

# Device for Nemo models
nemo_device = cuda
# Options: cpu, cuda

# Cache directory for downloaded models
nemo_cache_dir = ./models/nemo

# VibeVoice-ASR (local inference)
vibevoice_enabled = false
vibevoice_model_id = microsoft/VibeVoice-ASR
vibevoice_device = cuda
vibevoice_dtype = bfloat16
vibevoice_cache_dir = ./models/vibevoice

# Optional: route VibeVoice-ASR to a vLLM HTTP server
vibevoice_vllm_enabled = false
vibevoice_vllm_base_url = http://127.0.0.1:8001
vibevoice_vllm_model_id = microsoft/VibeVoice-ASR
vibevoice_vllm_timeout_seconds = 600
```

Hotwords: VibeVoice-ASR supports the `hotwords` form field on `/api/v1/audio/transcriptions`
and the `hotwords` option on media ingestion endpoints. You can pass CSV (e.g., `alpha,beta`)
or a JSON list (e.g., `["alpha","beta"]`).

### Environment Variables

Note: STT configuration is read from `Config_Files/config.txt` (`[STT-Settings]`). Environment overrides are limited; use `config.txt` to change batch/streaming defaults, Nemo device/variant, fallback policy, and cache directories.

Additional streaming quota/env controls:
- `AUDIO_TIER_LIMITS_JSON`: JSON mapping to override per-tier limits, e.g. `{ "free": { "daily_minutes": 60, "concurrent_streams": 2 } }`
- `AUDIO_STREAM_TTL_SECONDS`: TTL for Redis stream counters (default 120) to mitigate counter leaks on abrupt disconnects
- `AUDIO_FAILOPEN_CAP_MINUTES`: Bounded fail-open allowance (minutes) per WebSocket connection when the quota backing store (DB/Redis) is unavailable. Defaults to `5.0`. Set to a positive float to change.

STT vNext controls exposed through `get_stt_config()`:
- `STT_WS_CONTROL_V2_ENABLED`: enable explicit WebSocket control v2 negotiation (`protocol_version=2`)
- `STT_PAUSED_AUDIO_QUEUE_CAP_SECONDS`: paused-audio queue cap for v2 sessions (default `2.0`)
- `STT_OVERFLOW_WARNING_INTERVAL_SECONDS`: rate limit for paused-queue overflow warnings (default `5.0`)
- `STT_TRANSCRIPT_DIAGNOSTICS_ENABLED`: include deterministic final/full transcript diagnostics
- `STT_DELETE_AUDIO_AFTER_SUCCESS` / `STT_DELETE_AUDIO_AFTER`: default raw-audio delete-after-success policy
- `STT_AUDIO_RETENTION_HOURS`: default retained-audio TTL when retention is enabled
- `STT_REDACT_PII`: default transcript redaction toggle
- `STT_ALLOW_UNREDACTED_PARTIALS`: allow unredacted partial frames when policy permits it
- `STT_REDACT_CATEGORIES`: comma-separated or JSON list of category names to redact

Multi-user deployments can override the effective STT policy per org through:
- `GET /api/v1/admin/orgs/{org_id}/stt/settings`
- `PATCH /api/v1/admin/orgs/{org_id}/stt/settings`

Single-user mode does not use org policy rows; global STT config defaults are authoritative.

Config file overrides (Config_Files/config.txt):
```ini
[Audio-Quota]
free_daily_minutes = 60
free_concurrent_streams = 2
free_concurrent_jobs = 1
free_max_file_size_mb = 25
standard_daily_minutes = 480
premium_daily_minutes = unlimited  # or 'none'
# Optional bounded fail-open allowance (minutes) per connection when quota store is unavailable
failopen_cap_minutes = 5.0

[Audio]
# You can also specify the fail-open cap here if [Audio-Quota] is not present
failopen_cap_minutes = 5.0
```

## Live Transcription

### WebSocket API (Real-time)

- Endpoint: `ws://localhost:8000/api/v1/audio/stream/transcribe`
- Authentication:
  - Single-user: `?token=<SINGLE_USER_API_KEY>` in the query OR first message `{ "type": "auth", "token": "<SINGLE_USER_API_KEY>" }`
  - Multi-user JWT: `Authorization: Bearer <JWT>` on the upgrade request, or first message `{ "type": "auth", "token": "<JWT>" }`.
  - Multi-user API Keys: `X-API-KEY` header supported; keys can be scoped to endpoints (must include `audio.stream.transcribe`) and optionally path-prefixed allowlists. Quotas may be enforced per key.
- Protocol:
  - Client may send config after auth: `{ "type": "config", "sample_rate": 16000, "language": "en", "model_variant": "standard|onnx|mlx", "protocol_version": 2 }`
  - Send audio chunks: `{ "type": "audio", "data": "<base64 float32 little-endian mono>" }`
  - Legacy finalize/reset/stop remain valid: `{ "type": "commit" }`, `{ "type": "reset" }`, `{ "type": "stop" }`
  - WebSocket control v2 is opt-in. When the initial config includes `protocol_version: 2` and `STT_WS_CONTROL_V2_ENABLED=true`, clients may also send `{ "type": "control", "action": "pause|resume|commit|stop" }`.
- If no client `model` is provided, the server uses `[STT-Settings].default_streaming_transcription_model` (default: `parakeet-onnx`).
- Streaming model-init fallback to Whisper is opt-in via `[STT-Settings].streaming_fallback_to_whisper=true`; default is fail-fast.
- Server messages include:
    - `{ "type": "status", "message": "Authenticated" }` or `"Authenticated (JWT)"`
    - v2 lifecycle acknowledgements: `{ "type": "status", "state": "configured|paused|resumed|closing", "protocol_version": 2 }`
    - legacy reset acknowledgement: `{ "type": "status", "state": "reset" }`
    - `{ "type": "partial", "text": "...", "timestamp": ..., "is_final": false, "segment_id": 3, "segment_start": 12.5, "segment_end": 15.0 }`
    - `{ "type": "final", "text": "...", "timestamp": ..., "is_final": true, "segment_id": 3, "segment_start": 12.5, "segment_end": 14.0, "overlap": 0.5, "speaker_id": 1, "speaker_label": "SPEAKER_1" }` (speaker fields appear when diarization is enabled)
    - `{ "type": "full_transcript", "text": "...", "auto_commit": false, "vad_status": "enabled|disabled|fail_open", "diarization_status": "enabled|disabled|unavailable", "diarization_details": { "code": "...", "summary": "..." }? }`
    - `{ "type": "insight", "stage": "live|final", "summary": [...], "action_items": [...], ... }` when live meeting notes are enabled
    - `{ "type": "diarization_summary", "speaker_map": [...], "audio_path": "...", "speakers": [...] }` after `commit` when diarization is enabled
    - `{ "type": "error", "message": "..." }`
    - v2 control errors: `{ "type": "error", "error_type": "invalid_control", "message": "..." }`
    - v2 paused-queue overflow warning: `{ "type": "warning", "warning_type": "audio_dropped_during_pause", "message": "..." }`
    - Quota exceeded (structured): `{ "type": "error", "error_type": "quota_exceeded", "quota": "daily_minutes" }` followed by close with code `4003`.

#### Observability: Fail-open metrics

When the quota backing store is unavailable, the server allows a bounded amount of streaming time per connection (fail-open). The following metrics are emitted:

- `audio_failopen_minutes_total{reason=db_check|db_record}`: Minutes allowed during fail-open when quota checks or recording fail.
- `audio_failopen_events_total{reason=db_check|db_record}`: Count of fail-open allowance events.
- `audio_failopen_cap_exhausted_total{reason=db_check|db_record}`: Count of connections that hit the fail-open cap and were closed with `quota_exceeded`.

Use these to build dashboards/alerts on fail-open frequency and potential quota-store outages.

  - Metadata fields (`segment_id`, `segment_start`, `segment_end`, `chunk_start`, `chunk_end`, `overlap`) allow clients to align transcripts on a timeline or build diarization overlays.
  - WS final/full transcript frames follow the same effective redaction policy as REST responses. Partial frames are only allowed to bypass redaction when the effective policy explicitly permits unredacted partials.

#### WS Protocol Versions

- `v1` is the default when `protocol_version` is omitted.
- `v2` requires explicit `protocol_version: 2` in the initial config frame.
- Control frames are rejected with `invalid_control` unless the session negotiated `v2`.
- `pause` buffers inbound audio up to the configured cap; overflow uses `drop_oldest` semantics and emits the rate-limited `audio_dropped_during_pause` warning.
- `resume` drains buffered audio in FIFO order.
- `stop` drops any still-paused queued audio, emits `closing`, and closes the socket after already-processed audio is finalized.

Helper endpoints
- `GET /api/v1/audio/stream/status` → returns availability and supported models/variants and features
- `GET /api/v1/audio/stream/limits` → per-user limits, minutes remaining, active streams
- `POST /api/v1/audio/stream/test` → runs a built-in quick test of streaming setup

Examples (wscat)
```bash
wscat -c "ws://localhost:8000/api/v1/audio/stream/transcribe?token=$API_KEY"
wscat -H "Authorization: Bearer $JWT" -c "ws://localhost:8000/api/v1/audio/stream/transcribe"
```

For multilingual Nemo streaming with Canary:

- Use `model: "canary"` in the initial config message.
- Set `"task": "transcribe"` for same-language ASR, or `"task": "translate"` to request English translations (mirrors the `/audio/translations` HTTP endpoint semantics).

For low-latency English-only streaming with NVIDIA Parakeet-Realtime-EOU:

- Keep `model: "parakeet"` and enable the RNNT backend with `"parakeet_use_rnnt_streamer": true`.
- Set `"parakeet_rnnt_model_name": "nvidia/parakeet_realtime_eou_120m-v1"` in the config message to use the new realtime EOU model.
- The server strips the literal `<EOU>` token from transcripts while still using it internally as an utterance boundary hint.

#### Live Insights Configuration (Granola-style Notes)

Send an `insights` object inside the initial `{ "type": "config" }` message to enable live meeting summaries, action items, and decision tracking:

```json
{
  "type": "config",
  "model": "parakeet-onnx",
  "sample_rate": 16000,
  "insights": {
    "enabled": true,
    "provider": "openai",
    "model": "gpt-4o",
    "summary_interval_seconds": 90,
    "context_window_segments": 6,
    "live_updates": true,
    "final_summary": true,
    "generate_action_items": true,
    "generate_decisions": true
  }
}
```

- `summary_interval_seconds`: cadence for live summaries (set to `0` for “every segment”).
- `context_window_segments`: how many recent finalized segments are considered in each update.
- `live_updates`: toggle real-time `{"type":"insight","stage":"live"}` messages.
- `final_summary`: emit a final `{"type":"insight","stage":"final"}` after commit.
- Provider/model values fall back to the server’s default chat provider when omitted.

The insight payload mirrors granola-style UX:

```json
{
  "type": "insight",
  "stage": "live",
  "summary": ["Key bullet point", "..."],
  "action_items": [{"description": "Follow up with Alex", "owner": "Alex"}],
  "decisions": ["Ship v1 this week"],
  "topics": ["Roadmap"],
  "source": {"segment_range": [3,4], "start": 45.0, "end": 62.0}
}
```

### Auth & Close Codes

- Auth modes
  - Single-user: pass `?token=<API_KEY>` query, or `X-API-KEY` header, or `Authorization: Bearer <API_KEY>`, or first message `{ "type":"auth", "token":"..." }`.
  - Multi-user: prefer `Authorization: Bearer <JWT>`; first-message JWT also accepted. Virtual API keys via `X-API-KEY` are supported with endpoint/path allowlists and DB-backed quotas.
- Quotas
  - Concurrent streams and daily minutes enforced per user; Redis is used when available for cross-process counters; otherwise in-process.
  - On quota violations, the server emits `{ "type":"error", "error_type":"quota_exceeded", "quota":"daily_minutes|concurrent_streams" }` and closes with code `4003`.
- Common close codes
  - `4401` Unauthorized (auth missing/invalid)
  - `4403` Forbidden (endpoint/path not allowed or key/JWT quota exceeded)
  - `4003` Application quota violation (daily minutes / concurrent streams)
  - `1008` Policy violation (e.g., IP not on allowlist)
  - `1011` Internal error (e.g., no models available, or fallback failed when explicitly enabled)
  - `4400` Unsupported protocol version on WS surfaces that do not accept the requested version


#### Speaker Diarization & Audio Persistence

Add a `diarization` object inside the config message to enable per-segment speaker tagging:

```json
{
  "type": "config",
  "model": "parakeet",
  "sample_rate": 16000,
  "diarization": {
    "enabled": true,
    "num_speakers": 3,
    "store_audio": true,
    "storage_dir": "/tmp/meeting-audio"
  }
}
```

- When enabled, every finalized segment includes `speaker_id`/`speaker_label`.
- On `commit`, the server emits a `diarization_summary` frame containing `speaker_map`, aggregate speaker stats, and (optionally) the path to the persisted WAV file for replay or offline reprocessing.
- `store_audio` writes the full session audio to the provided directory (defaults to the system temp directory).

##### VAD Fallback Behavior

- The diarization pipeline uses Silero VAD to detect speech regions. Loading Silero via `torch.hub` can be network-bound and may fail in locked-down environments.
- When VAD is unavailable or fails at runtime, the server can optionally fall back to a single full-span speech region so diarization and transcript alignment can still proceed.
- This behavior is controlled by a configuration flag: `diarization.allow_vad_fallback` (default: `true`).
  - `true`: On VAD failures, use one region from 0.0s to full duration.
  - `false`: Treat VAD failure as fatal for diarization and return an error.
- Torch Hub cache directory is configured via `TORCH_HOME` (preferred) or `TORCH_HUB`, and the server sets `torch.hub.set_dir(...)` to ensure the directory is respected.
- To run in a locked-down/no-network environment, set `diarization.enable_torch_hub_fetch=false` to disable hub fetching entirely. With `diarization.allow_vad_fallback=true` (default), the server will fall back to a single full-span speech region when VAD is not available.
- Audio persistence prefers `soundfile`. If not available, the server falls back to `scipy.io.wavfile` or the standard `wave` module (16-bit PCM). A warning is logged when falling back.

##### Embedding Model Local-Only Mode

- The diarization pipeline uses a speaker embedding model (default: `speechbrain/spkrec-ecapa-voxceleb`). By default, the server may download this model when missing.
- To run fully offline, set `diarization.embedding_local_only=true`. In this mode, the server will only load models from local paths and will never attempt a network fetch.
- Resolution order when `embedding_local_only=true`:
  1) If `diarization.embedding_model` is a local filesystem path that exists, load from that directory.
  2) Else, look under the pre-seeded cache directory: `pretrained_models/<sanitized_name>`.
  3) If neither exists, diarization raises a structured error indicating local files are required.

Example config snippet (config.txt or env-equivalent):

```
[diarization]
embedding_model = /opt/models/speechbrain/spkrec-ecapa-voxceleb
embedding_local_only = true
```

Expected directory layout for a SpeechBrain model (simplified):

```
/opt/models/speechbrain/spkrec-ecapa-voxceleb/
├── hyperparams.yaml
├── model.ckpt          # or equivalent checkpoint
├── README.md           # optional
└── additional files…
```

Notes:
- `embedding_model` also accepts repo identifiers (e.g., `speechbrain/spkrec-ecapa-voxceleb`) when `embedding_local_only=false` (default). In that case the server caches into `pretrained_models/<sanitized_name>/`.
- Combine with `diarization.enable_torch_hub_fetch=false` and `diarization.allow_vad_fallback=true` to operate in fully offline/locked-down environments.

Example error payloads when files are missing and `embedding_local_only=true`:

- WebSocket (unified streaming) warning frame on initialization/finalize:

```
{
  "type": "warning",
  "state": "diarization_unavailable",
  "message": "Diarization disabled: initialization failed",
  "details": "Embedding model files not found locally. Set embedding_local_only=false to allow download or provide a local path in embedding_model."
}
```

- Generic structured error shape for non-WS callers (illustrative):

```
{
  "error": true,
  "error_type": "diarization_model_unavailable",
  "message": "Embedding model files not found locally",
  "details": {
    "embedding_model": "/opt/models/speechbrain/spkrec-ecapa-voxceleb",
    "embedding_local_only": true
  }
}
```

### Basic Live Transcription (Local Python)

```python
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Live_Transcription_Nemo import (
    create_live_transcriber
)

# Create transcriber with callbacks
def on_transcription(text):
    print(f"Final: {text}")

def on_partial(text):
    print(f"Partial: {text}")

transcriber = create_live_transcriber(
    model='parakeet',
    mode='silence_based',
    on_transcription=on_transcription,
    on_partial=on_partial
)

# Start transcription
transcriber.start()
# ... speak into microphone ...
transcriber.stop()
```

### Streaming File Transcription (Local Python)

```python
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Live_Transcription_Nemo import (
    NemoStreamingTranscriber
)

# Create streaming transcriber
transcriber = NemoStreamingTranscriber(
    model='parakeet',
    variant='onnx',
    chunk_duration=5.0
)

# Initialize with sample rate
transcriber.initialize(sample_rate=16000)

# Process audio chunks
for chunk in audio_chunks:
    text = transcriber.process_chunk(chunk)
    if text:
        print(f"Transcribed: {text}")

# Get complete transcription
full_text = transcriber.get_full_transcription()
```

### Transcription Modes

1. **Continuous Mode**: Process audio continuously without pause detection
2. **VAD-Based Mode**: Use Voice Activity Detection for intelligent segmentation
3. **Silence-Based Mode**: Simple amplitude-based silence detection (default)

## Usage Examples

### Using curl

```bash
# Basic transcription with Whisper
curl -X POST "http://localhost:8000/api/v1/audio/transcriptions" \
  -H "X-API-KEY: YOUR_SINGLE_USER_API_KEY" \
  -F "file=@audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=json"

# Fast transcription with Parakeet
curl -X POST "http://localhost:8000/api/v1/audio/transcriptions" \
  -H "X-API-KEY: YOUR_SINGLE_USER_API_KEY" \
  -F "file=@audio.wav" \
  -F "model=parakeet" \
  -F "response_format=json"

# Multi-lingual with Canary (Spanish)
curl -X POST "http://localhost:8000/api/v1/audio/transcriptions" \
  -H "X-API-KEY: YOUR_SINGLE_USER_API_KEY" \
  -F "file=@spanish_audio.wav" \
  -F "model=canary" \
  -F "language=es"

# Get SRT subtitles
curl -X POST "http://localhost:8000/api/v1/audio/transcriptions" \
  -H "X-API-KEY: YOUR_SINGLE_USER_API_KEY" \
  -F "file=@video_audio.wav" \
  -F "model=whisper-1" \
  -F "response_format=srt"
```

### Using Python (OpenAI Client)

```python
from openai import OpenAI

# Configure client to use tldw_server
client = OpenAI(
    base_url="http://localhost:8000/api/v1",
    # In single-user mode, the OpenAI client sends Bearer by default.
    # Provide your API key via X-API-KEY header instead:
    api_key="not-used",
    default_headers={"X-API-KEY": "YOUR_SINGLE_USER_API_KEY"}
)

# Basic transcription
with open("audio.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="json"
    )
    print(transcript.text)

# Using Parakeet for faster transcription
with open("audio.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="parakeet",
        file=audio_file,
        response_format="json"
    )
    print(transcript.text)

# Multi-lingual transcription with Canary
with open("spanish_audio.wav", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="canary",
        file=audio_file,
        language="es",
        response_format="verbose_json"
    )
    print(f"Language: {transcript.language}")
    print(f"Text: {transcript.text}")
    print(f"Duration: {transcript.duration}")

# Translation to English
with open("foreign_audio.wav", "rb") as audio_file:
    translation = client.audio.translations.create(
        model="whisper-1",
        file=audio_file
    )
    print(translation.text)
```

### Using Python (Direct API)

Note: This manual multipart example is minimal and no-deps; for production clients, prefer a well-tested multipart library.

```python
import json
import uuid
from urllib.request import Request, urlopen

# Transcribe with Parakeet
url = "http://localhost:8000/api/v1/audio/transcriptions"
headers = {"X-API-KEY": "YOUR_SINGLE_USER_API_KEY"}

def encode_multipart(fields, files):
    boundary = uuid.uuid4().hex
    body = bytearray()

    def add_line(line):
        body.extend(line.encode("utf-8"))
        body.extend(b"\r\n")

    for name, value in fields.items():
        add_line(f"--{boundary}")
        add_line(f'Content-Disposition: form-data; name="{name}"')
        add_line("")
        add_line(str(value))

    for name, filename, content, content_type in files:
        add_line(f"--{boundary}")
        add_line(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"')
        add_line(f"Content-Type: {content_type or 'application/octet-stream'}")
        add_line("")
        body.extend(content)
        body.extend(b"\r\n")

    add_line(f"--{boundary}--")
    return boundary, bytes(body)

with open("audio.wav", "rb") as f:
    data = {
        "model": "parakeet",
        "response_format": "json"
    }

    boundary, body = encode_multipart(
        data,
        [("file", "audio.wav", f.read(), "audio/wav")],
    )
    upload_headers = {**headers, "Content-Type": f"multipart/form-data; boundary={boundary}"}
    req = Request(url, data=body, headers=upload_headers, method="POST")
    with urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        print(result["text"])
```

### Live Transcription Example

```python
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.ARCHIVE.Desktop_Live_Audio_Samples import (
    LiveAudioStreamer,
)

# Configure for Parakeet with ONNX (desktop sample)
streamer = LiveAudioStreamer(
    transcription_provider='parakeet',
    nemo_variant='onnx',
    silence_threshold=0.01,
    silence_duration=1.5
)

# Custom handler for transcribed text
def handle_text(text):
    print(f"Transcribed: {text}")
    # Process text (save, send to chat, etc.)

streamer.handle_transcribed_text = handle_text

# Start live transcription (desktop-only sample)
streamer.start()
print("Listening... Press Ctrl+C to stop")

try:
    import time
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    streamer.stop()
    print("Stopped")
```

## Performance Comparison

### Speed Comparison (Relative to Real-Time)

| Model | Speed | Accuracy | Memory Usage | Best Use Case |
|-------|-------|----------|--------------|---------------|
| Whisper (tiny) | 10-15x | Good | 1GB | Quick drafts |
| Whisper (base) | 8-12x | Better | 1.5GB | General use |
| Whisper (large-v3) | 2-4x | Best | 10GB | High accuracy |
| Parakeet (standard) | 15-20x | Very Good | 2GB | Fast transcription |
| Parakeet (ONNX) | 20-30x | Very Good | 1.5GB | CPU optimization |
| Parakeet (MLX) | 25-35x | Very Good | 1.5GB | Apple Silicon |
| Canary-1b | 8-12x | Excellent | 4GB | Multi-lingual |
| Qwen2Audio | 1-2x | Excellent | 14GB | Complex audio |

### Recommendations

1. **For Speed**: Use Parakeet with ONNX or MLX variant
2. **For Accuracy**: Use Whisper large-v3 or Canary
3. **For Multi-lingual**: Use Canary (4 languages) or Whisper (99+ languages)
4. **For Live Transcription**: Use Parakeet with VAD mode
5. **For Resource-Constrained**: Use Parakeet ONNX or Whisper tiny

## Notes & Limitations

- Endpoint paths include `/api/v1` (examples reflect this; headings updated accordingly).
- `timestamp_granularities` supports `segment` and `word`; send as CSV or JSON array. Word-level timestamps are available for Whisper only.
- Language detection: When `language` is omitted and Whisper is used, the API returns the detected language in the JSON response.
- Authentication: Single-user mode uses `X-API-KEY`. The OpenAI Python client defaults to Bearer; pass `default_headers={"X-API-KEY": "..."}`.
- SRT/VTT outputs are basic placeholders without precise per-segment timings.
- File size limit is quota-aware; defaults to 25MB but can be increased/decreased per user tier. Requests over the limit return 413.
- Daily minutes are enforced for both batch and streaming transcription. When exceeded:
  - Batch/file transcription returns 402 (Payment Required) with `"Transcription quota exceeded (daily minutes)"`.
  - WebSocket streaming emits a structured error and closes with code 4003.

## Troubleshooting

### Common Issues

1. **Model Download Fails**
   - Check internet connection
   - Ensure sufficient disk space in cache directory
   - Try manual download from Hugging Face

2. **CUDA Out of Memory**
   - Use smaller model variant
   - Set `nemo_device = cpu` in config
   - Use ONNX variant for better memory efficiency

3. **Slow Transcription**
   - Use Parakeet instead of Whisper
   - Enable GPU acceleration (`nemo_device = cuda`)
   - Use ONNX or MLX variants

4. **Poor Accuracy**
   - Use larger model (Whisper large-v3 or Canary)
   - Specify correct language parameter
   - Provide prompt for context

### Debug Logging

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## API Rate Limits

- Transcription endpoint: 20 requests/minute (per user when authenticated; falls back to IP)
- Translation endpoint: 20 requests/minute (per user when authenticated; falls back to IP)
- File size limit: 25MB per request (tier-adjusted)

WebSocket limits
- Per-user concurrent streams and daily minutes enforced (exact values depend on server quotas). Structured errors emitted when quotas are exceeded.

TTS
- `POST /api/v1/audio/speech`: 10 requests/minute; OpenAI-compatible request with `model`, `input`, `voice`, `response_format` (mp3, opus, aac, flac, wav, pcm).
- Non-streaming responses may include `X-TTS-Alignment` (base64url JSON) when alignment metadata is available.
- Streaming alignment support: `POST /api/v1/audio/speech/metadata` with the same payload to return alignment JSON (200) or no-content (204).
- `GET /api/v1/audio/voices/catalog`: Lists available TTS voices across providers; optional `provider` filter.

## Security Considerations

1. **Authentication**: Always use Bearer token authentication in production
2. **File Validation**: The API validates file types and sizes
3. **Rate Limiting**: Built-in protection against abuse
4. **Input Sanitization**: All inputs are validated and sanitized

## Future Enhancements

- [ ] Batch transcription API (Jobs-backed, multi-stage fan-out)
- [ ] WebSocket JWT auth + per-user quotas/limits
- [ ] Speaker diarization with Nemo models
- [ ] Custom vocabulary support
- [ ] Fine-tuning support for domain-specific transcription
- [ ] Multi-GPU support for parallel processing

## Related Documentation

- [API Overview](./API_README.md)
- [Configuration Guide](../User_Guides/Configuration.md)
- [Live Transcription Guide](../User_Guides/Live_Transcription.md)
- [Model Selection Guide](../User_Guides/Model_Selection.md)
- For non-JSON responses (`text`, `srt`, `vtt`), `segment=true` is ignored and no `segmentation` is returned.
- TreeSeg embeddings use the configured embedding service unless `seg_embeddings_provider`/`seg_embeddings_model` overrides are supplied.
- If you have per-utterance segments from your STT provider, you can call the dedicated segmentation endpoint with those entries for better alignment.
- Errors:
  - 400: No file, invalid params, or bad `timestamp_granularities`
  - 402: Daily minutes quota exceeded
  - 413: File too large
  - 415: Unsupported media type
  - 429: Rate limit exceeded
  - 500: Transcription failed
