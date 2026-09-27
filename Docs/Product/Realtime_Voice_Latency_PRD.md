# Realtime Voice Latency PRD

Owner: Core Voice & API Team
Status: In Progress (Partially Implemented; updated 2026-07-01)

## Overview

Elevate the realtime voice experience (STT → LLM → TTS) to deliver natural, interruption‑friendly conversations with sub‑second voice‑to‑voice latency. Build on existing unified streaming STT, Kokoro streaming TTS, and OpenAI‑compatible APIs. Introduce precise turn detection, structured LLM streaming, low‑overhead audio transport options, and actionable end‑to‑end latency metrics.

## Goals

- Voice‑to‑voice latency (user stops speaking → first audible TTS): p50 ≤ 1.0s, p90 ≤ 1.8s.
- STT final transcript latency (end‑of‑speech → final text): p50 ≤ 600ms.
- TTS time‑to‑first‑byte (TTFB): p50 ≤ 250ms.
- Structured LLM streaming: speakable text to TTS immediately; code blocks and links to UI in parallel.
- Add reliable, lightweight metrics and a measurement harness.

## Non‑Goals

- Replacing existing RAG or LLM provider systems.
- Forcing WebRTC in all deployments (optional Phase 3 only).
- Vendor‑specific autoscaling mechanics (remain self‑host first).

## Personas & Use Cases

- Developers embedding voice agents in web apps who need:
  - Fast and reliable end‑of‑utterance detection with interruption handling.
  - Low TTS TTFB and smooth, continuous playback.
  - Structured results where “speakable” text is voiced and code/links render in UI.

## Success Metrics

- p50/p90 voice‑to‑voice latency meets targets above.
- <1% stream errors; 0 underruns in happy path.
- Backwards compatible APIs (no breaking changes to current REST).

## Scope & Phasing

### Phase 1: Core Latency + Metrics (Required)
- VAD/turn detection in streaming STT to trigger fast finalization.
- TTS TTFB + STT finalization latency metrics; compute voice‑to‑voice.
- PCM streaming option (lowest overhead) documented end‑to‑end.
- Phoneme/lexicon overrides for consistent pronunciation of brand/technical terms.

### Phase 2: Structured Streaming + WS TTS (Optional but Recommended)
- Streaming JSON parser: stream “spoke_response” to TTS; route code blocks/links to UI channel.
- WebSocket TTS endpoint for ultra‑low‑overhead PCM16 streaming.

### Phase 3: WebRTC Egress (Optional)
- Add a minimal WebRTC transport for browser playback where ultra‑low latency is required.

## Implementation Status (2026-07-01)

- [x] Phase 1 core implementation is present in code (VAD auto-commit, latency metrics, PCM support, phoneme mapping hooks).
- [ ] Phase 1 performance targets are not yet validated on the reference benchmark setup.
- [x] Phase 2 WS TTS endpoint is implemented (`/api/v1/audio/stream/tts` and realtime variant).
- [x] OpenAI-compatible Stage 1 realtime speech routes are implemented (`/api/v1/audio/realtime` and `/v1/realtime`),
  carrying `session_id`, `turn_id`, `response_id`, and `generation_id` for future cancellation and interruption work.
- [ ] Phase 2 structured streaming (`spoke_response`/`code_blocks`/`links`) is deferred.
- [ ] Realtime speech latency and interruption benchmarks remain Stage 2 validation work.
- [ ] Phase 3 WebRTC egress is deferred.

## Reference Setup

- Hardware/OS: 8‑core CPU, optional NVIDIA GPU (if Parakeet GPU path used); macOS 14 or Ubuntu 22.04
- Runtime: Python 3.11, ffmpeg ≥ 6.0, `av` ≥ 11.0.0, optional `espeak-ng` (phonemizer backend), optional `pyannote`
- Network: Localhost loopback during measurement; avoid WAN variability
- Test audio: 10 s of 16 kHz float32 speech, single speaker, 250 ms trailing silence
- Browser client (when applicable): latest Chrome/Edge/Firefox on same machine

## Functional Requirements

### STT Turn Detection
- Add Silero VAD‑based turn detection to the unified streaming STT path.
- Emit “commit” when end‑of‑speech is detected to finalize transcripts promptly.
- Expose safe server defaults and client‑configurable tunables (threshold, min silence, stop secs).

### TTS PCM Streaming
- Support `response_format=pcm` through `/api/v1/audio/speech` and document as recommended for ultra‑low latency clients.
- Keep MP3/Opus/AAC/FLAC for compatibility.

REST PCM details:
- Response header `Content-Type: audio/L16; rate=<sr>; channels=<n>`; default `rate=24000`, `channels=1`.
- Include `X-Audio-Sample-Rate: <sr>` header.
- Negotiation: default to provider/sample pipeline rate; optional `target_sample_rate` honored when supported.
- Example curl: `-d '{"model":"tts-1","input":"Hello","voice":"alloy","response_format":"pcm","stream":true}'`
- Example client: Web Audio API or Python playback snippet will be included in docs.

### Phoneme/Lexicon Overrides
- Optional phoneme mapping (config‑driven) to stabilize pronunciation of product names and domain terms.
- Provider‑aware behavior (e.g., Kokoro ONNX/PyTorch; espeak/IPA support where applicable).

### Structured LLM Streaming
- Add a streaming JSON parser to split:
  - `spoke_response` → stream chars immediately to TTS.
  - `code_blocks` and `links` → deliver to UI channel as soon as arrays complete (with optional async link validation).
- Make structured mode opt‑in (per request or model) to maintain backwards compatibility.

Schema and examples (opt‑in mode):
- Request flag: `structured_streaming: true` (per API call) or model‑level default
- Server stream examples:
  - `{ "type": "spoke_response", "text": "Great question..." }`
  - `{ "type": "code_block", "lang": "python", "text": "print('hello')" }`
  - `{ "type": "links", "items": [{"title": "Docs", "url": "https://..."}] }`
Interaction with OpenAI compatible `/chat/completions`:
- When enabled, server emits structured JSON chunks on the stream; speakable text is forwarded to TTS immediately; non‑speakable metadata is routed to the UI channel.

### WebSocket TTS Endpoint (Optional)
- New WS endpoint `/api/v1/audio/stream/tts` that accepts prompt frames and streams PCM16 bytes continuously with backpressure handling.

WebSocket TTS API details:
- Auth/Quotas: mirror STT WS. Support API key/JWT, endpoint allowlist checks, standardized close codes on quota.
- Client → Server frames: `{type:"prompt", text, voice?, speed?, format?:"pcm"}`; optional `request_id`.
- Server → Client frames: binary PCM16 audio frames (20–40 ms) with bounded queue; error frames as `{type:"error", "message": "..."}`.
- Backpressure: drop or throttle when the queue exceeds limit; increment `audio_stream_underruns_total` and emit warning status.

## Non‑Functional Requirements

- Low overhead: avoid heavy per‑chunk work; keep encoders warmed.
- Robustness: consistent behavior with disconnects, slow readers, and quotas.
- Observability: gated logs; metrics first for timing paths.

## Architecture & Components

Key touchpoints:
- STT WS handler: `tldw_Server_API/app/api/v1/endpoints/audio.py:1209` (stream/transcribe)
- Unified streaming STT: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Streaming_Unified.py`
- TTS REST endpoint: `tldw_Server_API/app/api/v1/endpoints/audio.py:268` (/audio/speech)
- Kokoro adapter streaming path: `tldw_Server_API/app/core/TTS/adapters/kokoro_adapter.py`
- Streaming encoder: `tldw_Server_API/app/core/TTS/streaming_audio_writer.py`
- TTS orchestrator: `tldw_Server_API/app/core/TTS/tts_service_v2.py`

Design changes:
- Introduce VAD in Unified STT pipeline; on VAD end → finalize chunk with Parakeet.
- Track event timestamps for end‑of‑speech, final transcript emission, TTS start, and first audio chunk write.
- Add PCM passthrough branch in TTS streaming for minimal overhead; preserve encoded formats via `StreamingAudioWriter`.
- Add phoneme pre‑processing hook in Kokoro adapter with config‑based mapping.
- Add optional WS TTS service that streams PCM16 frames directly.

## API Changes

REST (existing): `/api/v1/audio/speech`
- Support `response_format=pcm` (documented default for low‑latency clients).

WebSocket (existing): `/api/v1/audio/stream/transcribe`
- Accept optional client config to tune VAD/turn parameters (server defaults remain authoritative).
- Emit final transcripts promptly at turn end.

WebSocket (new, optional): `/api/v1/audio/stream/tts`
- Client → Server (text frames): `{type:"prompt", text, voice?, speed?, format?:"pcm"}`
- Server → Client (binary): PCM16 frames. Error frames as `{type:"error", message}`.

Structured LLM streaming (optional flag)
- When enabled, server parses JSON streams and routes fields: speech vs. UI metadata.

## Configuration

STT‑Settings:
- `vad_enabled` (bool, default true)
- `vad_threshold` (float, default 0.5)
- `turn_stop_secs` (float, default 0.2)
- `min_silence_ms` (int, default 250)

TTS‑Settings:
- `tts_pcm_enabled` (bool, default true)
- `phoneme_map_path` (str, optional JSON/YAML)
- `target_sample_rate` (int, default 24000)

Metrics:
- `enable_voice_latency_metrics` (bool, default true)

Feature Flags:
- `tts_pcm_enabled` (bool, default true)
- `enable_ws_tts` (bool, default false)
Dependencies:
- Required: `ffmpeg`, `av`
- Optional: `espeak-ng` (phonemizer), `pyannote`

 Security & Privacy:
- Do not log raw audio payloads; scrub PII from logs/metrics
- Configurable retention for any persisted audio (opt‑in diarization workflows)
- Avoid secrets in metric labels; bound label cardinality

## Measurement Model

Timestamps (server‑side):
- `EOS_detected_at`: VAD detects end‑of‑speech in WS STT loop
- `STT_final_emitted_at`: final transcript frame emitted on WS
- `TTS_request_started_at`: TTS handler receives request (REST) or prompt (WS‑TTS)
- `TTS_first_chunk_sent_at`: first audio bytes written to socket/response

Derived metrics:
- `stt_final_latency_seconds = STT_final_emitted_at - EOS_detected_at`
- `tts_ttfb_seconds = TTS_first_chunk_sent_at - TTS_request_started_at`
- `voice_to_voice_seconds = TTS_first_chunk_sent_at - EOS_detected_at`

Correlation:
- Propagate `X-Request-Id` (or generate UUIDv4) across WS/REST; include in logs/spans.

## Telemetry & Metrics

Histograms
- `voice_to_voice_seconds{provider,route}`: end‑of‑speech → first audio byte sent.
- `stt_final_latency_seconds{model,variant}`: end‑of‑speech → final transcript emit.
- `tts_ttfb_seconds{provider,voice,format}`: TTS request → first audio chunk emitted.
  - Buckets for all histograms: `[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]`

Counters
- `audio_stream_underruns_total{provider}`
- `audio_stream_errors_total{component,provider}`

Correlation
- Include `request_id` and `conversation_id` on event timelines where available.

Gauges
- Reuse `tts_active_requests{provider}` from TTS service v2

Endpoints
- Prometheus: `/metrics`; JSON: `/api/v1/metrics` (when `metrics` feature enabled)

## Testing Strategy

Unit
- JSON streaming parser: chunked inputs, escapes, array completion.
- Phoneme mapper: word‑boundary correctness, idempotence.

Integration
- STT WS: VAD commit timing; latency assertions (mock clocks).
- TTS streaming: PCM first‑chunk timing and multi‑format correctness.

Performance
- Synthetic end‑to‑end voice‑to‑voice harness; compute p50/p90, store summaries.
- Optional diarization on recorded sessions (pyannote) for verification (local opt‑in).
 - Negative‑path: slow reader/underrun, disconnects mid‑stream, silent/high‑noise input, malformed WS frames.

## Rollout Plan

Phase 1 (default on via flags)
- Shipped in code: VAD turn detection, latency metrics, PCM format, phoneme map hooks.
- Remaining: benchmark/validation pass against PRD latency targets.

Phase 2 (opt‑in)
- WS TTS shipped in code.
- Structured JSON streaming remains deferred.

Phase 3 (optional)
- WebRTC egress (aiortc) behind feature flag and environment readiness guide.

Documentation
- Update API docs, WebUI help, and latency tuning guidelines.

## Risks & Mitigations

- VAD misfires cause premature finals → conservative defaults; tunables; quick rollback.
- PCM clients mishandle raw streams → clear examples; fall back to MP3/Opus.
- Over‑instrumentation overhead → light timers; sampling; config‑gated metrics.

## Open Questions

- Default to structured JSON streaming for voice chat, or keep opt‑in per request/model?
- Preferred UI channel for code/links (reuse existing WS vs. SSE)?
- Region/affinity hints for distributed/self‑host deployments?

## Out of Scope

- New LLM providers and unrelated RAG changes.
- Browser TURN/STUN provisioning; full WebRTC infra (unless Phase 3 explicitly enabled).

## Acceptance Criteria

- [ ] p50 voice‑to‑voice ≤ 1.0s on a local reference setup; p90 ≤ 1.8s. (Pending benchmark validation)
- [ ] p50 STT final latency ≤ 600ms; p50 TTS TTFB ≤ 250ms (reference setup). (Pending benchmark validation)
- [ ] PCM streaming option documented and validated with example clients. (Partially implemented in code; docs/client validation pending)
- [x] Optional phoneme map configurable and applied in Kokoro path. (Implemented)
- [ ] Structured streaming mode available and tested end‑to‑end. (Deferred)
- [x] Metrics exported and visible in existing registry with labels. (Implemented)
- [ ] No regressions in quotas/auth for audio endpoints; REST streaming remains backwards‑compatible. (Pending regression sweep)
