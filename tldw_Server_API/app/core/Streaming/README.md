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
