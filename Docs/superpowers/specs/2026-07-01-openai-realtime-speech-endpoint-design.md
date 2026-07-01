# OpenAI-Compatible Realtime Speech Endpoint Design

Date: 2026-07-01
Backlog: TASK-12088
Status: Accepted for implementation

## Summary

Add an adapter-first OpenAI GA Realtime-compatible speech-to-speech WebSocket layer over the existing tldw audio pipeline. The first stage provides strict OpenAI-shaped behavior for the supported core speech lifecycle while keeping protocol details at the edge of the system.

The roadmap order is:

1. Add OpenAI-compatible realtime speech endpoint support.
2. Harden latency, interruption, cancellation, and stale-output suppression.
3. Add structured speech output so spoken text, UI text, tool data, links, and code can be separated cleanly.

Stage 1 must not replace the existing custom audio endpoints. It adds a focused compatibility facade and internal realtime session model that can support both the native tldw route and the OpenAI-compatible alias.

## Goals

- Expose a native realtime speech endpoint at `/api/v1/audio/realtime`.
- Expose an OpenAI-compatible alias at `/v1/realtime`.
- Target OpenAI GA Realtime voice-agent WebSocket event shapes for the supported lifecycle.
- Keep OpenAI protocol names out of the STT, LLM, TTS, persistence, quota, and metrics internals.
- Support ephemeral sessions by default, with opt-in persistence when tldw-specific session metadata is provided.
- Carry `session_id`, `turn_id`, `response_id`, and `generation_id` from Stage 1 so Stage 2 cancellation and interruption work does not require a structural rewrite.
- Add a capability endpoint that documents what Stage 1 supports and what is deferred.

## Non-Goals

- Full OpenAI Realtime parity in Stage 1.
- WebRTC, SIP, or translation-session support.
- Replacing existing `/api/v1/audio/chat/stream`, `/api/v1/audio/stream/transcribe`, or `/api/v1/audio/stream/tts/realtime`.
- Tool calls, MCP tools, full conversation item CRUD, server-side controls, or full conversation truncation support.
- Live-provider benchmark validation in the first compatibility slice.

## Source Context

Relevant existing tldw surfaces:

- Existing speech chat WebSocket: `/api/v1/audio/chat/stream`.
- Existing realtime TTS WebSocket: `/api/v1/audio/stream/tts/realtime`.
- Existing streaming transcription WebSocket: `/api/v1/audio/stream/transcribe`.
- Existing aggregate audio router: `tldw_Server_API/app/api/v1/endpoints/audio/audio.py`.
- Existing audio WebSocket helpers and auth behavior: `tldw_Server_API/app/core/Audio/streaming_service.py`.
- Existing route grouping and registration: `tldw_Server_API/app/api/v1/router_groups/content.py`, `tldw_Server_API/app/api/v1/router_groups/minimal.py`, and `tldw_Server_API/app/api/v1/router_registry.py`.
- Existing latency PRD notes: `Docs/Product/Realtime_Voice_Latency_PRD.md`.

External compatibility references:

- Hugging Face `speech-to-speech` uses a cascaded VAD -> STT -> LLM -> TTS architecture and exposes an OpenAI Realtime-compatible `/v1/realtime` style protocol.
- OpenAI Realtime GA docs identify `/v1/realtime` as the voice-agent conversation session endpoint and use event names such as `session.update`, `response.output_audio.delta`, `response.output_text.delta`, and `response.done`.
- OpenAI Realtime WebSocket docs describe the lifecycle around session initialization, input audio buffering, response creation, output deltas, done events, and rate limit updates.

## Recommended Approach

Use an adapter-first compatibility facade.

The core shape is:

```text
Routes:
  /api/v1/audio/realtime
  /v1/realtime
        |
WebSocket transport and auth adapter
        |
Protocol adapter
  OpenAI GA Realtime JSON <-> internal commands/events
        |
Realtime session orchestrator
  session config, audio buffer, active response, generation IDs,
  cancellation, quotas, optional persistence metadata
        |
Pipeline adapters
  existing STT -> LLM -> TTS services
        |
Internal events -> protocol adapter -> JSON WebSocket server events
```

The OpenAI mapper is an edge concern. Inbound OpenAI client events map into internal commands, and internal events map back to OpenAI server events. Lower-level STT, LLM, TTS, persistence, quota, and metrics code should never need to know OpenAI event names.

This avoids the two main failure modes:

- A thin wrapper around existing WebSocket handlers would be quick but would entangle the OpenAI protocol with the custom tldw protocol.
- A fresh speech runtime would be cleaner in isolation but too large for the first improvement slice.

## Route Strategy

Register two routes backed by the same handler and session orchestration:

- `/api/v1/audio/realtime`: canonical tldw endpoint.
- `/v1/realtime`: OpenAI-compatible alias.

The native endpoint can be mounted through the existing audio route group. The compatibility alias cannot be a child of the audio router because that router is mounted under `/api/v1/audio`. It needs a dedicated top-level router spec with prefix `/v1`.

Use an explicit route key/config flag:

- `audio-realtime`

The first implementation should treat this route as experimental. It should document how `audio-realtime` relates to existing `audio` and `audio-websocket` route flags.

## Components

### Realtime Routers

Thin endpoint/router layer only. Responsibilities:

- Accept WebSocket connections.
- Resolve route identity and path for auth/quota checks.
- Delegate to the shared realtime handler.
- Provide `/api/v1/audio/realtime/capabilities` as an HTTP capability endpoint.

The routers should be import-light. They should not import heavy STT or TTS runtime dependencies at module import time.

### Auth Adapter

The compatibility route must feel OpenAI-compatible without bypassing tldw AuthNZ.

Requirements:

- `/v1/realtime` accepts `Authorization: Bearer ...` where the bearer value resolves through normal tldw auth rules.
- `/v1/realtime` accepts `X-API-KEY` where supported by tldw auth mode.
- `/v1/realtime` may accept supported `Sec-WebSocket-Protocol` auth forms, but must echo the selected subprotocol correctly when accepting the socket.
- `/v1/realtime` must not use first-message auth fallback, because an OpenAI client may send `session.update` as its first event.
- `/api/v1/audio/realtime` may allow native tldw auth patterns, but should still prefer header or subprotocol auth.
- Auth should attach the normal internal principal/context so quota, billing, permissions, org resolution, and per-user persistence work consistently.

Endpoint identifiers for allowed endpoint/path checks should include:

- `audio.realtime`
- `/api/v1/audio/realtime`
- `/v1/realtime`

### Protocol Adapter

Validates OpenAI GA Realtime client events and converts them into internal commands. It also converts internal session events back into OpenAI-shaped server events.

Responsibilities:

- Parse known event types.
- Preserve event IDs where useful.
- Validate required fields and supported options.
- Convert base64 audio payloads into internal audio buffers.
- Convert internal audio chunks into JSON `response.output_audio.delta` events with base64 payloads.
- Emit OpenAI-shaped `error` events for invalid or unsupported input.
- Keep unsupported but recoverable events from closing the socket unnecessarily.

All audio over `/v1/realtime` and `/api/v1/audio/realtime` should be sent as JSON event frames. Do not stream raw binary frames on the compatibility surface.

### GA Session Shape

Stage 1 targets OpenAI Realtime GA shapes, not beta-era shapes.

Requirements:

- `session.update` handling must expect and validate `session.type` where the client provides it.
- Output audio configuration must live under `session.audio.output` when clients provide GA audio options.
- Beta-only event names and beta-only session fields are unsupported unless explicitly mapped and tested.
- The `OpenAI-Beta: realtime=v1` header must not enable beta behavior on the compatibility route.
- Docs and fixtures should use GA event names such as `response.output_text.delta`, `response.output_audio.delta`, and `response.output_audio_transcript.delta`.

### Session Orchestrator

Owns realtime session state and coordinates pipeline adapters.

Core state:

- `session_id`
- `turn_id`
- `response_id`
- `generation_id`
- session config
- input audio buffer state
- active response state
- cancellation state
- quota/accounting state
- optional persistence metadata

Responsibilities:

- Apply `session.update`.
- Append, commit, and clear input audio buffers.
- Start response generation.
- Cancel active response generation.
- Suppress stale output whose `generation_id` no longer matches the active response.
- Emit internal session, transcript, text, audio, done, error, and rate-limit events.
- Coordinate optional persistence.

Stage 1 should include the identifiers and stale-output checks even if Stage 2 later adds the full latency/interruption benchmark work.

### Pipeline Adapters

Call lower-level STT, LLM, and TTS services directly. They should not call existing WebSocket endpoint handlers as subroutines because those handlers own their own receive/send loops and protocol assumptions.

Pipeline adapters should:

- Convert committed input audio into STT requests.
- Convert transcripts and session history into LLM calls through existing chat provider abstractions.
- Stream generated text into TTS where supported.
- Stream internal text/audio/transcript events back to the orchestrator.
- Be swappable for deterministic fake adapters in tests.

### Persistence Adapter

Default behavior is ephemeral. No chat/session history is written unless the client provides explicit tldw persistence metadata.

When persistence is enabled:

- Persist user transcript turns.
- Persist assistant text turns.
- Link records to the resolved tldw conversation/session ID.
- Do not persist raw audio by default.
- Do not persist unsupported event payloads.

### Audio Format Contract

The first implementation must define exact Stage 1 input and output audio formats before exposing the route. Those formats should be chosen from what the existing STT/TTS services can support reliably through the realtime adapters.

Requirements:

- Supported input formats and sample rates must be listed in the capability response.
- Supported output formats and sample rates must be listed in the capability response.
- Unsupported declared formats must be rejected during `session.update` with an OpenAI-shaped `unsupported_session_option` error.
- Invalid or undecodable audio payloads must be rejected with an OpenAI-shaped `invalid_audio` error.
- The implementation plan must select concrete defaults rather than leaving format negotiation implicit.

## Supported Stage 1 Events

Supported client events:

- `session.update`
- `input_audio_buffer.append`
- `input_audio_buffer.commit`
- `input_audio_buffer.clear`
- `response.create`
- `response.cancel`

Optional client event for the first implementation slice:

- Basic `conversation.item.create` for text or complete audio input. The implementation plan must either include this explicitly with tests or defer it with an `unsupported_event` response. It is not required for the Stage 1 acceptance criteria.

Supported server events:

- `session.created`
- `session.updated`
- `input_audio_buffer.speech_started`
- `input_audio_buffer.speech_stopped`
- `input_audio_buffer.committed`
- `conversation.item.added`
- `conversation.item.done`
- `response.created`
- `response.output_item.created`
- `response.content_part.added`
- `response.output_audio.delta`
- `response.output_audio.done`
- `response.output_audio_transcript.delta`
- `response.output_audio_transcript.done`
- `response.output_text.delta`
- `response.output_text.done`
- `response.done`
- `rate_limits.updated`
- `error`

## Compatibility Boundaries

Stage 1 compatibility is strict for the supported core lifecycle. Unsupported features must be explicit.

Required behavior:

- Unsupported client event types produce OpenAI-shaped `error` events with stable codes.
- Unsupported session options produce OpenAI-shaped `error` events and should not silently fall back to different behavior.
- Recoverable protocol errors keep the socket open where possible.
- Auth failure, quota rejection, oversized frames, and unrecoverable internal failures may close with documented close codes.
- Manual `input_audio_buffer.commit` is supported.
- Server VAD is supported only where it maps cleanly to existing turn detection. Unsupported VAD modes or options must be rejected explicitly.

Stable error codes should include:

- `invalid_event`
- `unsupported_event`
- `unsupported_session_option`
- `invalid_audio`
- `payload_too_large`
- `quota_exceeded`
- `authentication_failed`
- `internal_error`

Deferred features:

- WebRTC
- SIP
- Translation sessions
- Realtime transcription-only sessions
- `/v1/realtime/client_secrets` ephemeral credential issuance
- Tool calls and MCP integration
- Full conversation item CRUD
- Server-side control APIs
- Full truncation and context-management parity

## Rate Limit Events

Stage 1 should emit `rate_limits.updated` as a tldw quota compatibility event, not as a claim of exact OpenAI quota parity.

The event should:

- Use the OpenAI server-event `type` value: `rate_limits.updated`.
- Report tldw-enforced realtime/audio quotas and remaining allowance where available.
- Avoid inventing OpenAI organization-level quota fields that tldw cannot compute.
- Be documented in the capability endpoint as tldw quota compatibility semantics.
- Be covered by golden fixture tests so downstream clients get stable shapes.

## Capability Endpoint

Add:

- `GET /api/v1/audio/realtime/capabilities`

The response should report:

- Supported routes.
- Supported modalities.
- Supported input audio formats.
- Supported output audio formats.
- Supported VAD/turn detection modes.
- Max frame size.
- Max buffered audio seconds/bytes.
- Supported client events.
- Optional client events available in the running build, including whether `conversation.item.create` is supported.
- Supported server events.
- Deferred features.
- Rate limit event semantics.
- Whether persistence is supported and how it is enabled.
- Whether the route is experimental.

This keeps Stage 1 honest and helps clients make feature decisions without trial-and-error.

## Testing Strategy

Default tests should use fake STT, LLM, and TTS adapters. Live provider tests should be marked and skipped by default.

### Protocol Adapter Tests

Use golden fixtures for:

- `session.update`
- `input_audio_buffer.append`
- `input_audio_buffer.commit`
- `input_audio_buffer.clear`
- `response.create`
- `response.cancel`
- output text deltas
- output audio deltas
- output transcript deltas
- `rate_limits.updated`
- done events
- error events

Adapter tests should compare exact JSON shapes for supported OpenAI GA events.

### Session Orchestrator Tests

Cover:

- Session creation and update.
- Audio append, commit, and clear.
- Response creation.
- Response cancellation.
- `generation_id` changes and stale output suppression.
- Optional persistence metadata.
- Recoverable and fatal error flows.

### Handshake And Auth Tests

Cover:

- `Authorization: Bearer ...`.
- `X-API-KEY`.
- Supported `Sec-WebSocket-Protocol` auth forms.
- Correct accepted subprotocol echo.
- `/v1/realtime` does not consume `session.update` as an auth message.
- Endpoint/path identifiers work with allowed endpoint/path auth restrictions.

### Payload And Backpressure Tests

Cover:

- Malformed base64 audio.
- Oversized JSON frames.
- Excess buffered audio seconds/bytes.
- Queue pressure.
- Graceful error events before close when possible.
- Documented close behavior for fatal cases.

### WebSocket Integration Tests

Use fake pipeline adapters and assert:

- Event order for a basic speech turn.
- Event order for manual commit.
- Event order for response cancel.
- JSON-only outbound frames.
- Base64 audio deltas.
- No binary frames on the compatibility surface.

### Persistence Tests

Cover:

- Ephemeral sessions write no conversation history.
- Opt-in sessions persist user and assistant turns.
- Persistence failures do not corrupt active realtime state.

### Capability Tests

Cover:

- Capability endpoint includes supported routes, formats, VAD modes, limits, events, and deferred features.
- Capability endpoint reports optional `conversation.item.create` support accurately for the running build.
- Capability endpoint documents `rate_limits.updated` as tldw quota compatibility semantics.
- Capability output changes only intentionally when Stage 2 or Stage 3 expands support.

### Live Provider Smoke Tests

Live-provider tests should be:

- Marked separately.
- Skipped by default.
- Documented with required environment variables.
- Limited to provider-specific smoke validation, not normal CI.

## Rollout Plan

1. Add route/config key `audio-realtime`.
2. Add import-light native router for `/api/v1/audio/realtime`.
3. Add separate top-level router spec for `/v1/realtime`.
4. Add `/api/v1/audio/realtime/capabilities`.
5. Add internal command/event models and OpenAI GA protocol adapter.
6. Add session orchestrator with `session_id`, `turn_id`, `response_id`, `generation_id`, cancellation state, and stale-output suppression.
7. Add pipeline adapters over lower-level STT, LLM, and TTS services.
8. Add optional persistence adapter.
9. Add golden fixture, unit, integration, auth, payload, and capability tests.
10. Update audio streaming docs with supported Stage 1 coverage and deferred features.

## Risks And Mitigations

- Risk: The compatibility alias accidentally inherits custom tldw protocol behavior.
  - Mitigation: Keep the protocol adapter at the edge and use exact OpenAI-shaped golden fixtures.

- Risk: WebSocket auth consumes the first client event.
  - Mitigation: Do not use first-message auth fallback on `/v1/realtime`; require header or subprotocol auth.

- Risk: Large base64 audio frames create memory pressure.
  - Mitigation: Enforce max frame size, max buffered audio, and queue limits before decoding or processing too much data.

- Risk: `audio_streaming.py` grows further.
  - Mitigation: Keep endpoint code thin and put models, adapters, and orchestration in focused realtime modules.

- Risk: Existing route registration cannot expose `/v1/realtime`.
  - Mitigation: Add a separate top-level router spec instead of mounting the alias under the audio router.

- Risk: Stage 1 claims more compatibility than it provides.
  - Mitigation: Add capability endpoint and docs that list supported events and deferred features.

- Risk: Cancellation is bolted on later.
  - Mitigation: Include response and generation identifiers from Stage 1.

## Open Questions For Implementation Planning

- Exact module path: likely `tldw_Server_API/app/core/Audio/Realtime/`, but the implementation plan should verify package naming conventions before creating files.
- Exact fake adapter injection mechanism for tests.
- Exact close code policy for auth failure, quota rejection, oversized frames, and fatal internal errors.
- Whether basic `conversation.item.create` should land in the first implementation slice or be deferred behind explicit `unsupported_event` behavior.

## Acceptance Criteria For Stage 1

- Both `/api/v1/audio/realtime` and `/v1/realtime` mount when `audio-realtime` is enabled.
- `/v1/realtime` speaks JSON OpenAI GA Realtime-shaped events for the supported lifecycle.
- `/v1/realtime` does not require or consume an initial auth message.
- Unsupported events/options return OpenAI-shaped errors.
- Audio deltas are base64 JSON events, not binary frames.
- GA session shapes are accepted and beta-only Realtime shapes are not silently enabled.
- Exact supported input/output audio formats are exposed through capabilities and unsupported formats are rejected explicitly.
- Internal STT/LLM/TTS services do not depend on OpenAI event names.
- Ephemeral sessions do not persist by default.
- Opt-in persistence writes expected user and assistant turns.
- `generation_id` stale-output suppression is present.
- Capability endpoint documents supported and deferred features.
- Capability endpoint documents optional `conversation.item.create` support and tldw `rate_limits.updated` semantics.
- Default tests do not require live providers.
