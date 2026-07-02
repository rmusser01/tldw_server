Audio Streaming Protocol (Core Parakeet)
=======================================

Overview
- WebSocket-based real-time transcription using the Parakeet Core Streaming module.
- Supports model variants, partial/final frames, live insights (optional), and diarization (optional).

WebSocket Endpoint
- Unified endpoint: `/api/v1/audio/stream/transcribe` (primary; includes auth/quotas/fallback)
- Core demo endpoint: `/core/parakeet/stream` (portable router; no auth/quotas)

Server-side handler (observability-enabled)
```python
from tldw_Server_API.app.core.Streaming.streams import WebSocketStream

async def handle_audio_ws(websocket):
    # Use labels to tag metrics with low-cardinality identifiers
    stream = WebSocketStream(
        websocket,
        heartbeat_interval_s=10,
        idle_timeout_s=120,
        compat_error_type=True,  # transitional alias; gate with AUDIO_WS_COMPAT_ERROR_TYPE during migration
        close_on_done=True,
        labels={"component": "audio", "endpoint": "audio_ws"},
    )
    await stream.start()
    try:
        # domain payloads are sent as-is (no event frames)
        await stream.send_json({"type": "status", "state": "ready"})
        # ... process messages, emit partial/final results ...
    except Exception as e:
        await stream.error("internal_error", str(e))
    finally:
        await stream.stop()
```

Config Frame
- Send this JSON as the first message to configure the session. All fields are optional unless noted.

{
  "type": "config",                       // required
  "model": "parakeet",                    // default: parakeet
  "model_variant": "standard|onnx|mlx",   // default: standard
  "sample_rate": 16000,                    // default: 16000
  "chunk_duration": 2.0,                   // seconds per final segment
  "overlap_duration": 0.5,                 // seconds kept as context between segments
  "language": "en",                       // optional language hint
  "enable_partial": true,                  // emit partial results on a cadence
  "insights": { ... },                     // optional live insights configuration
  "diarization": true                      // or "diarize": true; enable speaker diarization
}

Audio Frame
- Base64-encoded float32 mono PCM audio samples.

{
  "type": "audio",
  "data": "<base64 float32 mono>"
}

Partial Frame
- Emitted periodically when `enable_partial` is true and buffer has enough audio.

{
  "type": "partial",
  "text": "...",
  "is_final": false,
  // Segment metadata (example keys)
  "segment_id": 1,
  "segment_start": 0.0,
  "segment_end": 0.8,
  "buffer_duration": 0.8,
  "cumulative_audio": 0.0
}

Final Frame
- Emitted when `chunk_duration` is reached; includes detailed segment metadata.

{
  "type": "final",
  "text": "...",
  "is_final": true,
  // Segment metadata
  "segment_id": 1,
  "segment_start": 0.0,
  "segment_end": 1.0,
  "chunk_duration": 1.0,
  "overlap": 0.0,
  "chunk_start": 0.0,
  "chunk_end": 1.0,
  "new_audio_duration": 1.0,
  "cumulative_audio": 1.0
}

Commit & Full Transcript
- After sending `{ "type": "commit" }`, the server flushes remaining audio and returns the full transcript.
- When diarization is enabled and available, a `diarization_summary` frame is also emitted.

// Flush response (if any pending text)
{ "type": "final", "text": "...", "is_final": true, ...metadata }

// Full transcript
{ "type": "full_transcript", "text": "..." }

// Optional diarization summary
{
  "type": "diarization_summary",
  "speaker_map": [
    { "segment_id": 1, "speaker_id": 0, "speaker_label": "SPEAKER_00" }
  ],
  "audio_path": null,
  "speakers": [ {"speaker_id": 0, "label": "SPEAKER_00"} ]
}

Other Control Frames
- Reset: `{ "type": "reset" }` → `{ "type": "status", "state": "reset" }`
- Stop: `{ "type": "stop" }` → closes session
- Ping/Pong: `{ "type": "ping" }` → `{ "type": "pong" }`

Notes
- Custom vocabulary post-processing applies to text results when enabled (see `Audio_Custom_Vocabulary`).
- Unified endpoint handles auth, quotas, Whisper fallback, and integrates the same core transcriber via an adapter.
- If Nemo capability probing cannot be imported/resolved at runtime, the unified endpoint fail-safes to Whisper defaults instead of terminating the WS session.

Client Examples
---------------

Python (websockets) - base64 JSON frames

import asyncio, json, base64, numpy as np
import websockets

async def main():
    url = "ws://127.0.0.1:8000/api/v1/audio/stream/transcribe?token=YOUR_API_KEY"
    async with websockets.connect(url, max_size=2**23) as ws:
        # 1) Send config
        await ws.send(json.dumps({
            "type": "config",
            "model": "parakeet",
            "model_variant": "onnx",  # or standard|mlx
            "sample_rate": 16000,
            "chunk_duration": 2.0,
            "overlap_duration": 0.5,
            "enable_partial": True,
            "diarization": True,
            "insights": {"enabled": True}
        }))

        # 2) Send audio frames as base64 float32 mono
        sr = 16000
        samples = (np.zeros(sr//2, dtype=np.float32)).tobytes()  # 0.5s silence
        payload = base64.b64encode(samples).decode("ascii")
        await ws.send(json.dumps({"type": "audio", "data": payload}))

        # 3) Commit
        await ws.send(json.dumps({"type": "commit"}))
        while True:
            msg = await ws.recv()
            print(json.loads(msg))

asyncio.run(main())

Node.js (ws) - base64 JSON frames

const WebSocket = require('ws');
const sr = 16000;
const zeros = Buffer.alloc((sr/2)*4); // 0.5s float32 zeros
const ws = new WebSocket('ws://127.0.0.1:8000/api/v1/audio/stream/transcribe?token=YOUR_API_KEY');

ws.on('open', () => {
  ws.send(JSON.stringify({
    type: 'config', model: 'parakeet', model_variant: 'standard', sample_rate: 16000,
    chunk_duration: 2.0, overlap_duration: 0.5, enable_partial: true
  }));
  ws.send(JSON.stringify({ type: 'audio', data: zeros.toString('base64') }));
  ws.send(JSON.stringify({ type: 'commit' }));
});
ws.on('message', (data) => console.log(JSON.parse(data)));

Python - raw float32 usage (library-level)
- If you embed the core transcriber in your own service, pass `numpy.float32` arrays directly. The Parakeet core transcriber accepts raw float32 and handles chunking, overlap, and metadata.

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Parakeet_Core_Streaming.transcriber import ParakeetCoreTranscriber
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Parakeet_Core_Streaming.config import StreamingConfig
import numpy as np, asyncio

async def run_local():
    cfg = StreamingConfig(sample_rate=16000, chunk_duration=2.0, overlap_duration=0.5,
                          model='parakeet', model_variant='standard', enable_partial=True)
    def decode_fn(audio_np, sr):
        # Your Parakeet decode here
        return "hello"
    tx = ParakeetCoreTranscriber(cfg, decode_fn=decode_fn)
    audio = np.zeros(24000, dtype=np.float32)  # 1.5s
    frame = await tx.process_audio_chunk(audio)
    print(frame)
    frame2 = await tx.flush()
    print(frame2)

asyncio.run(run_local())

---

WebSocket TTS (PCM)
-------------------

Endpoint: `/api/v1/audio/stream/tts`

- Auth/quotas: mirrors streaming STT (API key/JWT/single-user key) with per-user concurrent stream guard. Credentials
  may be supplied via `X-API-KEY`, `Authorization: Bearer <JWT>`, a `token` query parameter (API key or JWT), or
  an initial auth message; single-user setups additionally accept the fixed API key via these sources.
- Client → server frames: one JSON prompt/config frame

```json
{
  "type": "prompt",
  "text": "Hello world",
  "voice": "af_heart",
  "format": "pcm",
  "model": "kokoro",
  "provider": "kokoro",
  "speed": 1.0,
  "lang": "en",
  "extra_params": {}
}
```

- Server → client frames:
  - Binary PCM16 audio frames streamed as they are generated.
  - Error frames (normalized): `{ "type": "error", "code": "...", "message": "...", "request_id": "...", "data": {...} }`.
    - Compatibility alias: when `AUDIO_WS_COMPAT_ERROR_TYPE=1` (default), payloads also include `error_type`.
  - Finalizer: `{ "type": "done" }` then the socket closes (policy close code on quota errors when configured).

Notes:
- Backpressure: bounded producer/consumer queue; when full, the oldest chunk is dropped and the new chunk is enqueued. Each overflow increments `audio_stream_underruns_total{provider}`.
- Queue depth control:
  - `AUDIO_TTS_WS_QUEUE_MAXSIZE` (preferred) or `AUDIO_WS_TTS_QUEUE_MAXSIZE` (fallback)
  - Default `8`, clamped to `2..256`
- Close-code mapping:
  - `4400`: malformed/missing prompt frame or unsupported request payload
  - `4401`: authentication required/invalid credentials
  - `4403`: token/key is valid but not authorized for endpoint/path
  - `4003` (or `1008` when `AUDIO_WS_QUOTA_CLOSE_1008=1`): quota/concurrency denial
  - `1011`: transport failure while writing binary audio to the socket
- Metrics: streaming TTS emits `tts_ttfb_seconds`/`voice_to_voice_seconds{route="audio.stream.tts"}` and error counters on transport failures.
- Default format is `pcm`; `mp3|opus|aac|flac|wav` are accepted but PCM is preferred for low latency.

WebSocket TTS Realtime
----------------------

Endpoint: `/api/v1/audio/stream/tts/realtime`

- Auth/quotas: same as `/stream/tts` (API key/JWT/single-user key, per-user concurrent stream guard).
- Transport: JSON control frames from client, binary audio frames from server.

Client -> server frames:
- First frame: `type=config` (preferred) or `type=text`/`prompt`
  - `config` fields (all optional): `provider`, `model`, `voice`, `format`, `speed`, `lang`, `extra_params`,
    `auto_flush_ms`, `auto_flush_tokens`
- `type=text`/`input`: `{ "type": "text", "delta": "..." }` (accepted keys: `delta|text|input`)
- `type=commit`: flush buffered text to synthesis
- `type=interrupt`: cancel the current synthesis window and reopen a fresh realtime session on the same socket
- `type=final`: flush + close session
- `type=ping`: server replies `pong`

Server -> client frames:
- `ready`: `{ "type": "ready", "provider": "...", "format": "pcm", "sample_rate": 24000, "request_id": "..." }`
- `warning`: `{ "type": "warning", "message": "...", "request_id": "..." }` (e.g., fallback to buffered TTS)
- `error`: `{ "type": "error", "code": "...", "message": "...", "request_id": "..." }`
- `interrupted`: `{ "type": "interrupted", "phase": "tts", "reason": "...", "request_id": "..." }`
- `done`: `{ "type": "done" }`
- Binary audio frames (format from config/ready; for `pcm`, raw PCM16 LE)

Behavior notes:
- Auto-flush: when `auto_flush_ms` elapses after the last input or `auto_flush_tokens` is exceeded, the server issues
  an internal commit. Set either to `0` to disable.
- `interrupt` does not close the WebSocket; it rotates to a new realtime session so clients can continue sending text.
- Config updates after session start are ignored.
- Defaults: provider `vibevoice_realtime`, model `vibevoice-realtime-0.5b`, format `pcm`.
- Example client: `Helper_Scripts/voice_latency_harness/examples/ws_tts_realtime_client.py`

Example:
```bash
python Helper_Scripts/voice_latency_harness/examples/ws_tts_realtime_client.py --text "Hello realtime"
ffplay -f s16le -ar 24000 -ac 1 out_ws_tts_realtime.pcm
```

OpenAI-Compatible Realtime Speech
---------------------------------

The realtime speech endpoint exposes a Stage 1 OpenAI-compatible JSON event
protocol over WebSocket. It bridges committed PCM16 input audio through the
existing STT -> chat -> TTS pipeline and emits OpenAI-style realtime events.

### Routes

- `WS /api/v1/audio/realtime`: native tldw route using the OpenAI-compatible event shape.
- `WS /v1/realtime`: OpenAI-compatible route for clients that expect the upstream path.
- `GET /api/v1/audio/realtime/capabilities`: runtime metadata for supported routes, events, audio formats, limits,
  close codes, and unsupported features.

Both WebSocket routes are guarded by the `audio-realtime` route toggle and use the `audio.realtime` AuthNZ endpoint id.

### Authentication

Stage 1 realtime speech authenticates during the WebSocket handshake and does not consume an initial protocol frame
for auth.

- Single-user mode accepts `Authorization: Bearer <SINGLE_USER_API_KEY>` or `X-API-KEY: <SINGLE_USER_API_KEY>`.
- Multi-user mode accepts `Authorization: Bearer <JWT>` or `X-API-KEY: <virtual API key>`.
- Legacy `?token=` query-string auth remains disabled by default and is only accepted when
  `AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH=1`.
- `/v1/realtime` does not accept first-message auth such as `{"type":"auth","token":"..."}`. Unauthenticated
  connections close with `4401`.

### Audio Contract

- Input audio: mono PCM16, 16 kHz, little-endian, base64 encoded in `input_audio_buffer.append.audio`.
- Output audio: mono PCM16, 24 kHz, base64 encoded in the `delta` field of `response.output_audio.delta` events.
- Maximum JSON frame size: 262144 bytes.
- Maximum buffered input audio: 30 seconds, 960000 bytes.
- Maximum output audio delta chunk: 65536 bytes.

### Supported Client Events

- `session.update`
- `input_audio_buffer.append`
- `input_audio_buffer.commit`
- `input_audio_buffer.clear`
- `response.create`
- `response.cancel`

Manual turn flow:

1. Connect with auth headers.
2. Receive `session.created` and `rate_limits.updated`.
3. Optionally send `session.update`.
4. Send one or more `input_audio_buffer.append` frames with base64 PCM16 audio.
5. Send `input_audio_buffer.commit`.
6. Send `response.create` to run STT, chat generation, and TTS for the committed turn.
7. Read streamed text, transcript, audio deltas, and `response.done`.

`response.create` accepts the default/empty response object in Stage 1. Response-scoped overrides such as
`modalities`, `model`, `voice`, `instructions`, and `audio` are rejected; apply supported session configuration with
`session.update` before creating a response.

### Supported Server Events

- `session.created`
- `session.updated`
- `input_audio_buffer.speech_started`
- `input_audio_buffer.speech_stopped`
- `input_audio_buffer.committed`
- `conversation.item.created`
- `conversation.item.done`
- `response.created`
- `response.output_item.added`
- `response.content_part.added`
- `response.output_text.delta`
- `response.output_text.done`
- `response.output_audio.delta`
- `response.output_audio.done`
- `response.output_audio_transcript.delta`
- `response.output_audio_transcript.done`
- `response.content_part.done`
- `response.output_item.done`
- `response.done`
- `rate_limits.updated`
- `error`

`rate_limits.updated` is emitted for tldw quota compatibility. It is not an OpenAI quota-parity guarantee.

### Unsupported In Stage 1

- `conversation.item.create`
- Tool calls
- Server-side VAD turn detection
- Client-selected `input_audio_format`
- Client-selected `output_audio_format`
- Response-scoped overrides (`modalities`, `model`, `voice`, `instructions`, `audio`)
- Binary WebSocket audio frames

Unsupported client events return an `error` event while keeping the WebSocket open when the frame is otherwise valid.

### Persistence And Capabilities

Sessions are ephemeral by default. Optional turn persistence is enabled through session metadata:
`metadata.tldw.persist=true` plus `metadata.tldw.conversation_id=<integer>`. Stage 1 does not persist raw audio.

Use `GET /api/v1/audio/realtime/capabilities` to discover persistence metadata, optional/deferred events, audio
limits, close codes, and route support.

### Provider Hints

The default realtime pipeline uses the existing configured STT, chat, and TTS provider stacks. Optional env overrides:

- `REALTIME_CHAT_MODEL`: default chat model.
- `REALTIME_TTS_MODEL`: default TTS model.
- `REALTIME_TTS_VOICE`: default TTS voice.
- `REALTIME_CHAT_PROVIDER_HINT`: provider hint for chat generation.
- `REALTIME_TTS_PROVIDER_HINT`: provider hint for realtime/buffered TTS.
- `REALTIME_PROVIDER_HINT`: compatibility fallback used when a chat- or TTS-specific hint is not set.

WebSocket Voice Chat v2
-----------------------

Endpoint: `/api/v1/audio/chat/stream`

- Auth/quotas: same as `/stream/transcribe` (JWT/X-API-KEY/single-user key) with support for a `token` query parameter
  (API key or JWT), `can_start_stream` concurrency guard, per-chunk minute accounting with bounded fail-open; closes
  with code 4003 (or 1008 when `AUDIO_WS_QUOTA_CLOSE_1008=1`) on quota failures.
- Client → server frames:
  - `config` (required first): STT knobs (`model|variant|sample_rate|enable_vad|min_silence_ms|turn_stop_secs`), `llm` (`provider|model|temperature|max_tokens|system|extra_params`), `tts` (`voice|model|provider|format|speed|extra_params`), optional `session_id|metadata`.
  - `audio`: base64 float32/PCM chunks (same shape as `/stream/transcribe`).
  - `commit`: finalize the current turn (also auto-triggered by VAD when enabled).
  - `interrupt`: cancel in-flight generation/synthesis for the active turn without closing the socket.
  - `reset` / `stop`: reset buffers or close the stream.
- Server → client frames:
  - STT partials/finals: mirrors `/stream/transcribe` plus `full_transcript` with `voice_to_voice_start` timestamp and `auto_commit` hint.
  - LLM streaming: `{"type":"llm_delta","delta": "<text>"}` per SSE chunk, then `llm_message` + `assistant_summary` (finish_reason/usage).
  - TTS streaming: binary audio frames (PCM default; `mp3|opus|aac|flac|wav` accepted) preceded by `tts_start` and terminated by `tts_done`; underruns surfaced via `audio_stream_underruns_total`.
  - Interrupt ack: `{"type":"interrupted","turn_id":"turn-N|null","phase":"both","reason":"..."}`.
  - Overlapped flow: `tts_start` and first audio bytes may be emitted before final `llm_message`.
  - Errors use `{type:"error", code:"...", message, data?}`; with compatibility on (`AUDIO_WS_COMPAT_ERROR_TYPE=1`), `error_type` is also present. Quota/rate errors include `data.quota` (and legacy top-level `quota` while compatibility mode is enabled).
- VAD: Silero-based auto-commit when enabled (`enable_vad=true`, `vad_threshold`, `min_silence_ms`, `turn_stop_secs`, `min_utterance_secs`).
- Metrics: `stt_final_latency_seconds{endpoint="audio.chat.stream"}`, `voice_to_voice_seconds{route="audio.chat.stream"}`, `audio_stream_underruns_total`, `audio_stream_errors_total`, plus provider metrics from LLM/TTS.

Deployment Notes
----------------

Dependencies
- Parakeet variants
  - standard (NeMo): `pip install nemo_toolkit[asr]`
  - onnx: `pip install onnxruntime` (+ model/tokenizer loader in the codebase)
  - mlx (Apple Silicon): `pip install mlx parakeet-mlx`
- Whisper fallback (unified handler): `pip install faster-whisper` and `ffmpeg`
- Diarization (optional): depends on `Diarization_Lib` backends; if unavailable, diarization is disabled gracefully.

Quotas and Redis (optional)
- Per-user quotas for concurrent streams and daily minutes are tracked in-process or via Redis.
- Redis enables TTL-based leak safety for abrupt disconnects; configuration precedence:
  - Env `AUDIO_STREAM_TTL_SECONDS`
  - Config `[Audio-Quota].stream_ttl_seconds`
  - Default 120s (clamped to 30-3600)
- Without Redis, concurrency counters are in-process only.

Health
- Check variant availability: `GET /api/v1/audio/stream/status`
- Lists available models (feature probe) and the streaming WS endpoint.
