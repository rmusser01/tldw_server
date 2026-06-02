# Streaming

Streaming contains shared runtime primitives for server-sent events, WebSocket wrappers, phrase chunking, and a speech-chat turn pipeline. The package is used by chat, media ingest jobs, MCP adapters, audio tests, and voice-style flows that need queue-backed streaming or incremental text-to-speech chunking.

## Start Here

- `streams.py` defines `SSEStream` and WebSocket stream helpers with heartbeats, idle limits, metrics, and termination handling.
- `phrase_chunker.py` incrementally groups text fragments into speakable phrases for streaming TTS consumers.
- `speech_chat_service.py` coordinates a non-streaming speech-to-speech chat turn from audio input through STT, LLM response, optional MCP actions, and TTS output.
- Related tests: `tests/Streaming/` and `tests/Audio/test_phrase_chunker.py`.

## Responsibilities

- Provide queue-backed SSE events with JSON/raw payload helpers, heartbeat support, provider-control passthrough, error events, and done events.
- Wrap WebSocket send/receive behavior with idle and ping handling for stream consumers.
- Chunk partial text into phrase-sized units suitable for speech synthesis without waiting for the full response.
- Run speech-chat turns by decoding user audio, transcribing it, calling the LLM provider, saving chat history, and synthesizing response audio.
- Emit metrics and labels that streaming tests and monitoring code can assert against.

## Module Map

- `streams.py` - SSE and WebSocket stream primitives.
- `phrase_chunker.py` - incremental phrase segmentation for TTS.
- `speech_chat_service.py` - speech-chat orchestration service.

## How It Connects

- `app/core/Chat/chat_service.py` uses `SSEStream` for unified chat streaming.
- `app/api/v1/endpoints/media/ingest_jobs.py` streams ingest job events through `SSEStream`.
- `app/core/MCP_unified/adapters/tldw_runtime.py` imports WebSocket stream helpers.
- Audio and voice-adjacent flows use `phrase_chunker.py` and `speech_chat_service.py` for speech output timing and turn orchestration.

## Architecture Notes

### Core Flow

- `SSEStream` owns queue-backed event emission, heartbeats, provider-control filtering, idle/max-duration checks, error events, and a single done event.
- `WebSocketStream` wraps accept/send/error/done/close behavior with ping and idle tracking for endpoint-specific WebSocket consumers.
- `PhraseChunker` buffers streaming text until sentence, clause, whitespace, or max-character boundaries make a useful TTS chunk.
- `speech_chat_service.py` decodes audio, enforces audio constraints, transcribes, optionally executes a speech action, calls the LLM, records chat history, and maps TTS errors into API-compatible failures.

### State And Data

- Stream queues and heartbeat timers are per request; this package does not store auth or tenant state.
- Metrics labels include endpoint/component details and must stay stable for monitoring assertions.
- Speech-chat state crosses STT, LLM, optional MCP action execution, chat history, and TTS services; keep those boundaries explicit when changing turn behavior.

### Security And Operations

- Authorization, moderation, and business rules belong to the caller before the stream helper is created.
- SSE and WebSocket streams should terminate once through done or error handling so clients do not hang.
- Backpressure and heartbeat behavior are covered by tests; preserve queue limits and idle/max-duration safeguards.
- Logs for provider, metric, and TTS failures should not include raw secrets, private paths, or full user audio payloads.

### Extension Checklist

- New SSE event behavior: update `streams.py` and `tests/Streaming/test_streams.py`.
- New WebSocket timeout or ping behavior: update `streams.py` and WebSocket label tests.
- New speech-chat provider boundary: update `speech_chat_service.py`, audio constraint tests, and TTS/STT error mapping tests.

## Extension Points

- For new SSE event types, update `streams.py` and add focused assertions in `tests/Streaming/test_streams.py`.
- For WebSocket ping/idle behavior, inspect `streams.py` and the `tests/Streaming/test_ws_pings_labels_multi.py` coverage.
- For phrase boundaries, update `phrase_chunker.py` and add deterministic examples in `tests/Audio/test_phrase_chunker.py`.
- For speech-chat provider changes, start in `speech_chat_service.py` and keep STT, LLM, chat history, and TTS boundaries explicit.

## Testing

- `tests/Streaming/test_streams.py`
- `tests/Streaming/test_chat_completions_sse_unified_flag.py`
- `tests/Streaming/test_chat_doc_stream_unified_flag.py`
- `tests/Streaming/test_character_chat_sse_unified_flag.py`
- `tests/Streaming/test_provider_streaming_smoke.py`
- `tests/Streaming/test_ws_pings_labels_multi.py`
- `tests/Audio/test_phrase_chunker.py`

## Gotchas

- Stream helpers are transport/runtime primitives; endpoint-specific authorization and business rules live in the caller.
- SSE streams should terminate with a single completion path so consumers do not hang or receive duplicate done events.
- Speech-chat orchestration crosses STT, LLM, TTS, MCP, and chat-history storage, so provider failures should stay isolated and logged with sanitized context.
