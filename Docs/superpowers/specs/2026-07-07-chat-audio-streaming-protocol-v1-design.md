# Chat Audio Streaming Protocol V1 Design

## Status

Implemented in the protocol-v1 rollout branch on 2026-07-08 using
`Docs/superpowers/plans/2026-07-08-chat-audio-streaming-protocol-v1.md`.

Backlog tasks: TASK-12912 (design), TASK-12913 (implementation plan),
TASK-12914 (implementation).

## Problem

The current WebUI and browser-extension chat audio paths do not share one explicit
streaming contract. That makes dictation, turn detection, and VAD fragile because
the client and server can disagree about audio format, initialization order, and
the meaning of control events.

The immediate user-facing failure was the voice chat path sending microphone
chunks in a format that the chat websocket path did not interpret correctly. VAD
and turn detection then saw unusable audio and could not reliably identify speech
or end a turn. A short-term compatibility fix can unblock current usage, but the
long-term fix is an explicit protocol that all chat audio clients must follow.

Pre-implementation problems this design addressed:

- The browser extension live STT path still forwards raw binary chunks to the
  transcription websocket with no explicit config.
- The transcription websocket and chat websocket have different assumptions about
  whether config is optional and how audio frames are shaped.
- Dictation uses a separate MediaRecorder upload path, so it does not share
  streaming behavior, partial transcript handling, or audio ownership rules with
  voice chat.
- Push-to-talk, continuous voice chat, dictation, and captions are different
  product modes but are not represented as first-class protocol modes.

## Goals

- Define one strict websocket protocol for chat-adjacent audio streaming.
- Keep the existing websocket endpoint URLs to avoid a large route migration.
- Use one shared server parser behind both endpoints.
- Make wire audio PCM16 mono at 16 kHz in v1.
- Normalize server-side audio to Float32 mono before STT, VAD, or turn logic sees
  it.
- Make server VAD authoritative for continuous voice chat.
- Make push-to-talk release an explicit end-of-turn hint that does not depend on
  VAD.
- Make streaming dictation insert transcript text into the composer without
  invoking LLM or TTS.
- Migrate the browser extension STT path in the same cutover so it no longer
  sends raw binary audio.
- Fail fast on invalid protocol usage instead of silently guessing.

## Non-Goals

- Migrating persona live voice to this protocol in the first implementation.
  Persona can remain a reference pattern because it already normalizes explicit
  audio formats.
- Removing the file-upload transcription endpoint. It remains available for
  non-streaming transcription.
- Keeping browser SpeechRecognition as part of the v1 streaming contract. It can
  remain a legacy fallback outside this protocol where currently supported.
- Supporting multi-channel audio, arbitrary sample rates, or Float32 on the wire
  in v1.
- Redesigning the chat toolbar visuals beyond the minimum controls and states
  needed for the modes below.

## Architecture

Keep these endpoint URLs:

- `/api/v1/audio/chat/stream`
- `/api/v1/audio/stream/transcribe`

Both endpoints use a shared `AudioStreamProtocol` layer before endpoint-specific
business logic runs.

The parser responsibilities are:

- Validate the first post-auth frame is a config frame.
- Validate the endpoint allows the requested mode.
- Validate the v1 audio contract.
- Decode JSON audio frames.
- Base64-decode PCM16 payloads.
- Convert PCM16 mono samples to Float32 mono bytes.
- Emit typed protocol events to endpoint handlers.

Endpoint handlers stay responsible for product behavior:

- `/api/v1/audio/chat/stream` accepts `voice_chat` and `push_to_talk`.
- `/api/v1/audio/stream/transcribe` accepts `dictate` and `captions`.
- Chat stream handlers perform STT, turn handling, LLM calls, and TTS when the
  mode allows it.
- Transcription stream handlers only produce transcript events.

Persona live voice is out of scope for the first implementation, but its existing
explicit audio-format normalization is the model for this parser.

## Protocol Contract

Authentication may be the first frame when required by the endpoint. The first
post-auth frame must be config. No audio or control frame is valid before config.

### Config Frame

```json
{
  "type": "config",
  "protocol_version": 1,
  "mode": "voice_chat",
  "audio_format": "pcm16",
  "sample_rate": 16000,
  "channels": 1
}
```

Strict v1 fields:

- `protocol_version`: must be `1`.
- `mode`: must be allowed by the endpoint.
- `audio_format`: must be `"pcm16"`.
- `sample_rate`: must be `16000`.
- `channels`: must be `1`.

The server may accept additional existing endpoint-specific config fields after
the strict fields validate, but the v1 audio contract is not inferred from those
fields.

### Audio Frame

```json
{
  "type": "audio",
  "data": "<base64 pcm16 mono 16khz>"
}
```

Raw binary websocket frames are not valid in v1. The browser extension STT path
must send the same JSON audio frames as the WebUI.

### Control Frames

Common controls:

```json
{ "type": "commit" }
{ "type": "reset" }
{ "type": "stop" }
```

Push-to-talk release:

```json
{ "type": "push_to_talk_release" }
```

The parser normalizes `push_to_talk_release` to the same internal commit intent
used by existing chat audio processing, with source metadata set to
`push_to_talk_release`.

## Modes

### `voice_chat`

Endpoint: `/api/v1/audio/chat/stream`

Continuous listening mode. The server performs STT, server-authoritative VAD,
turn finalization, LLM response generation, and TTS response streaming.

When VAD is unavailable, the server sends a warning and continues without
auto-commit. The UI must show that state instead of implying normal turn
detection is active.

### `push_to_talk`

Endpoint: `/api/v1/audio/chat/stream`

Records while the user holds the control. Release sends
`push_to_talk_release`, which finalizes and sends the chat turn. This mode does
not require VAD.

After release, the client returns to idle unless push-to-talk was launched inside
an already active voice chat session. In that case, the existing voice chat
auto-resume setting controls whether listening resumes.

### `dictate`

Endpoint: `/api/v1/audio/stream/transcribe`

Streaming dictation mode. Partial transcript events update a preview. Final
transcript events insert text into the active dictation span in the composer.

Dictation never calls the LLM and never starts TTS. In v1, server streaming is
required for the primary dictation control. Existing browser SpeechRecognition
behavior is legacy fallback behavior outside this protocol.

### `captions`

Endpoint: `/api/v1/audio/stream/transcribe`

Compatibility mode for existing extension/live-caption style use cases. Captions
use the same strict config and JSON audio frames as dictation.

Captions are protocol-supported but should not become a primary toolbar control
unless already exposed by an existing surface.

## Validation And Errors

Protocol violations should send a structured error frame when possible, then
close the websocket with `4400`.

Examples of contract violations:

- Audio before config.
- Control before config.
- Missing `protocol_version`.
- Unsupported `protocol_version`.
- Unsupported endpoint/mode combination.
- Non-PCM16 `audio_format`.
- Sample rate other than 16 kHz.
- Channel count other than one.
- Raw binary audio frame.
- Malformed JSON frame.
- Malformed base64 audio payload.

Oversized frames, quota failures, and rate-limit failures should continue to use
the existing quota and rate-limit behavior for the endpoint.

VAD unavailable is not a protocol violation. It is a warning event. For
`voice_chat`, the server continues without auto-commit. For `push_to_talk`, VAD
availability is irrelevant.

## Frontend Behavior

Primary voice controls expose:

- Dictate
- Voice Chat
- Push-to-talk

Only one audio owner may hold the microphone at a time. The v1 owners are:

- `dictation`
- `voice_chat`
- `push_to_talk`
- `captions`

Starting one owner must stop the previous owner before opening a new stream.
Dictation and voice chat must stop each other explicitly.

Streaming voice modes are unavailable when the backend streaming path is
unavailable. The UI should surface that directly instead of silently falling back
to a different behavior.

The UI must surface server warnings and errors that matter for usage:

- Authentication failure.
- Unsupported mode.
- Invalid protocol contract.
- VAD unavailable.
- Quota or rate-limit failure.

Dictation transcript handling:

- Partial transcript text is preview-only.
- Final transcript text inserts into the active dictation span.
- Partial updates must not overwrite user edits made after dictation started.
- Final insertion must not auto-submit the composer.

## Testing

Backend unit tests:

- Config validation accepts the strict v1 contract.
- Config validation rejects missing or unsupported fields.
- Endpoint/mode allowlists reject cross-endpoint modes.
- PCM16 payloads normalize to Float32 mono bytes.
- Malformed base64 is rejected.
- `push_to_talk_release` maps to internal commit intent with source metadata.

Backend websocket tests:

- Auth followed by config succeeds.
- Audio before config is rejected.
- Wrong endpoint/mode combinations are rejected.
- Dictation and captions never call LLM or TTS paths.
- Push-to-talk commits on release without VAD.
- Voice chat sends a warning and continues when VAD is unavailable.

Frontend hook tests:

- Each streaming mode sends strict config before audio.
- Audio frames are JSON base64 PCM16 frames, not raw binary frames.
- Dictation partial preview does not overwrite user edits.
- Dictation final transcript inserts into the composer without submitting.
- Starting one audio owner stops the previous owner.
- Browser extension STT no longer sends raw binary audio.

## Rollout

This is a strict v1 cutover implemented as one coordinated change:

- Update the shared WebUI audio capture client.
- Update chat voice streaming clients.
- Update streaming dictation clients.
- Update browser-extension background STT forwarding.
- Update both backend websocket endpoints to use the shared parser.
- Update tests and docs that currently imply config can be omitted or inferred.

The short-term compatibility fix that let voice chat use Float32 was treated as
a bridge only. The implemented stable contract is PCM16 on the wire with
server-side Float32 normalization.

## Risks And Open Questions

- Strict cutover requires frontend, extension, and backend changes to land
  together. Splitting the cutover risks breaking one surface.
- Existing docs or comments may still describe default config behavior. They need
  to be updated in the implementation pass.
- The first implementation intentionally supports only PCM16 mono 16 kHz. Future
  versions can negotiate additional formats through a new `protocol_version`.
- Browser SpeechRecognition fallback behavior should be audited separately after
  v1 streaming dictation is stable.
