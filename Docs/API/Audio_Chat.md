# /api/v1/audio/chat (Non-Streaming Speech Chat)

Purpose: speech in → transcript → LLM → speech out (optional action execution) in a single REST call. This is the v1 “voice command” path.

## Request (JSON)
- `input_audio` (base64, required): raw audio bytes (no data: URI prefix).
- `input_audio_format` (str, required): `wav|mp3|ogg|opus|aac|flac|webm|m4a`.
- `session_id` (str, optional): reuse an existing chat session; new session is created when omitted.
- `stt_config` (optional): provider/language hints (defaults from server).
- `llm_config` (required): `model` + `api_provider` at minimum; extra params passed to chat orchestrator. Optional `extra_params.action` can hint an action/tool to run.
- `tts_config` (optional): provider/voice/format/speed; defaults to `mp3` if omitted.
- `metadata` (optional): arbitrary client metadata; `metadata.action` can also hint an action/tool.
- `store_audio` (optional): hint to retain raw audio when server-side retention is enabled (default is not to store audio).

## Response
- `session_id`: resolved chat session.
- `user_transcript`: text transcript of user audio (STT).
- `assistant_text`: LLM reply text.
- `output_audio` / `output_audio_mime_type`: base64 audio of TTS reply.
- `timing`: ms timings for STT/LLM/TTS.
- `token_usage`: optional LLM token counts.
- `metadata`: echo + server annotations.
- `action_result`: optional structured result when actions are enabled and executed.

## Limits & Validation
- Size: `AUDIO_CHAT_MAX_BYTES` (default 20MB); oversize → HTTP 413 before STT.
- Duration: `AUDIO_CHAT_MAX_DURATION_SEC` (default 120s); over-duration → HTTP 400 before STT.
- Formats: rejects unsupported `input_audio_format` with HTTP 400.
- Auth: API key or JWT, same as other v1 audio/chat endpoints.
- Quotas: inherits global rate limiting; per-user concurrency for `/audio/chat` TBD (align with STT quotas if tightened).

## Actions / Tools
- Disabled by default. Enable with `AUDIO_CHAT_ENABLE_ACTIONS=1`.
- Action hint: `metadata.action` or `llm_config.extra_params.action`.
- Execution: routed to MCP modules via `execute_tool(action_name, {"input": transcript}, context)`.
- Persistence: when present, `action_result` is also saved as a `tool` message in conversation history.
- If no module provides the action, `action_result.status` is `not_found`; errors are returned as `error` with message.

## Metrics
- `audio_chat_latency_seconds{stt_provider,llm_provider,tts_provider}`: end-to-end REST turn latency.
- Reuses existing voice metrics: `stt_final_latency_seconds`, `tts_ttfb_seconds`, `voice_to_voice_seconds`.
- Correlate with `X-Request-Id` for logs/spans.

## Errors (common)
- 400: invalid base64, unsupported format, duration exceeded, missing model in `llm_config`.
- 413: size exceeded.
- 429: rate limit/quota.
- 5xx: STT/LLM/TTS/action errors (structured detail; no raw audio logged).

## Testing / Harness
- Unit: validation, action execution, error mapping.
- Integration: `tldw_Server_API/tests/Audio/test_audio_chat_endpoint.py`, `test_speech_chat_service.py`.
- Harness: `python Helper_Scripts/voice_latency_harness/run.py --out out.json --short` (metrics scrape) or run a real turn (omit `--short`; pass `--api-key`/`--base-url` if needed). Output JSON includes `run_id`, `fixture`, `runs`, and `metrics` with p50/p90 values for STT/TTS/voice-to-voice (and audio_chat when available).

# /api/v1/audio/chat/stream (Streaming Voice Chat v1)

Purpose: low-latency voice commands with partial STT, streaming LLM deltas, and streaming TTS audio over a single WebSocket.

## Protocol
- First post-auth frame must be `{"type":"config", ...}` with strict v1 audio fields:
  - `protocol_version`: `1`.
  - `mode`: `"voice_chat"` or `"push_to_talk"`.
  - `audio_format`: `"pcm16"`.
  - `sample_rate`: `16000`.
  - `channels`: `1`.
  - `stt`: `model|variant|sample_rate|enable_vad|vad_threshold|min_silence_ms|turn_stop_secs|min_utterance_secs`.
  - `llm`: `provider|model|temperature|max_tokens|system|extra_params`.
  - `tts`: `voice|model|provider|format|speed|extra_params` (`pcm` default; `mp3|opus|aac|flac|wav` supported).
  - Optional: `session_id`, `metadata`.
- Stream audio as `{"type":"audio","data":"<base64 PCM16 little-endian mono>"}`. In `voice_chat` mode, server VAD can auto-commit; clients may also send `{"type":"commit"}`. In `push_to_talk` mode, send `{"type":"push_to_talk_release"}` to commit without depending on VAD. `reset` clears buffers; `stop` closes the socket.
- Optional interruption: send `{"type":"interrupt","reason":"barge_in|user_stop|..."}` to cancel the in-flight turn synthesis/generation window without closing the socket.

## Server Frames
- STT partial/final frames mirror `/audio/stream/transcribe` plus `full_transcript` with `voice_to_voice_start` + `auto_commit`.
- LLM streaming: `llm_delta` for each text chunk; `llm_message` + `assistant_summary` at end.
- TTS streaming: binary audio frames; bracketed by `tts_start` / `tts_done`.
- Interrupt ack: `{"type":"interrupted","turn_id":"turn-N|null","phase":"both","reason":"..."}`.
- Overlap behavior: in overlapped mode, `tts_start` and early audio bytes can arrive before final `llm_message`.
- Errors use canonical `{type:"error", code:"...", message, data?}`; compatibility alias `error_type` is included when `AUDIO_WS_COMPAT_ERROR_TYPE=1`.

## Limits & Metrics
- Auth/quotas: API key/JWT/single-user key; per-user concurrency guard; minute accounting per chunk with bounded fail-open; quota/rate errors close with 4003 (1008 when `AUDIO_WS_QUOTA_CLOSE_1008=1`).
- Metrics: `stt_final_latency_seconds{endpoint="audio.chat.stream"}`, `voice_to_voice_seconds{route="audio.chat.stream"}`, `audio_stream_underruns_total`, `audio_stream_errors_total`, plus LLM/TTS provider metrics. Underruns logged when backpressure drops audio.

## Testing
- Happy path/unit: `tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py` (LLM deltas + TTS frames).
- Run alongside `Helper_Scripts/voice_latency_harness/run.py` for latency snapshots (`voice_to_voice_seconds`).
