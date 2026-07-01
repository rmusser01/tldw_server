# OpenAI Realtime Speech Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an adapter-first OpenAI GA Realtime-compatible speech-to-speech WebSocket surface at `/api/v1/audio/realtime` and `/v1/realtime`, backed by the existing tldw STT, chat, TTS, auth, route gating, quota, and optional persistence systems.

**Architecture:** Keep OpenAI Realtime event names at the protocol edge. Route handlers authenticate and pass JSON frames to a realtime handler. The handler delegates to a protocol adapter and internal session orchestrator. The orchestrator owns session state, audio buffers, response IDs, generation IDs, cancellation, stale-output suppression, and optional persistence. Pipeline adapters call lower-level STT, LLM, and TTS services directly.

**Tech Stack:** FastAPI WebSockets, Pydantic and dataclasses, existing AuthNZ helpers, existing route group registry, existing resource governor policy mapping, existing `perform_chat_api_call_async`, existing `OpenAISpeechRequest` and `TTSServiceV2.open_realtime_session`, pytest, TestClient WebSocket helpers, Bandit.

Backlog: `TASK-12088`
Spec: `Docs/superpowers/specs/2026-07-01-openai-realtime-speech-endpoint-design.md`

---

## Stage 1: Protocol, Capabilities, And Limits

**Goal:** Define the exact supported OpenAI GA event subset, internal event model, audio format contract, and capability response before touching transport code.

**Success Criteria:** The adapter accepts supported JSON client events, rejects unsupported/beta shapes with stable OpenAI-shaped errors, emits exact server event dictionaries, and exposes concrete capability data for Stage 1.

**Tests:** Protocol golden tests and capability unit tests run without importing heavy audio providers.

**Status:** Complete

### Files

Create:

- `tldw_Server_API/app/core/Audio/Realtime/__init__.py`
- `tldw_Server_API/app/core/Audio/Realtime/constants.py`
- `tldw_Server_API/app/core/Audio/Realtime/models.py`
- `tldw_Server_API/app/core/Audio/Realtime/protocol.py`
- `tldw_Server_API/app/core/Audio/Realtime/capabilities.py`
- `tldw_Server_API/tests/Audio/test_realtime_protocol_adapter.py`
- `tldw_Server_API/tests/Audio/test_realtime_capabilities.py`

### Exact Contract

Supported Stage 1 client events:

- `session.update`
- `input_audio_buffer.append`
- `input_audio_buffer.commit`
- `input_audio_buffer.clear`
- `response.create`
- `response.cancel`

Explicitly unsupported in Stage 1:

- `conversation.item.create`
- tool calls
- server VAD configuration beyond manual turns
- beta-era top-level audio fields such as `output_audio_format`
- binary WebSocket frames

Supported input audio:

- format: `pcm16`
- channels: `1`
- sample rate: `16000`
- wire encoding: base64 inside JSON `audio`

Supported output audio:

- OpenAI wire format: `pcm16`
- tldw TTS request format: `pcm`
- channels: `1`
- sample rate: `24000`
- wire encoding: base64 inside JSON `response.output_audio.delta`

Limits:

- `REALTIME_MAX_JSON_FRAME_BYTES = 262144`
- `REALTIME_MAX_BUFFERED_AUDIO_SECONDS = 30`
- `REALTIME_INPUT_SAMPLE_RATE_HZ = 16000`
- `REALTIME_INPUT_SAMPLE_WIDTH_BYTES = 2`
- `REALTIME_MAX_BUFFERED_AUDIO_BYTES = 960000`
- `REALTIME_MAX_OUTPUT_CHUNK_BYTES = 65536`

Close code policy:

- auth failure: `4401`
- path or endpoint denied: `4403`
- quota denied: `4003`, or `1008` when `AUDIO_WS_QUOTA_CLOSE_1008` is true
- oversized frame: emit `payload_too_large` when possible, then close `1009`
- fatal internal error: emit `internal_error` when possible, then close `1011`
- normal completion: `1000`

### Task 1.1: Write Protocol Adapter Tests First

- [ ] Add `test_realtime_protocol_adapter.py`.
- [ ] Test that a GA `session.update` with `session.type == "realtime"` parses into `UpdateSessionCommand`.
- [ ] Test that `session.audio.output.format == "pcm16"` and `sample_rate_hz == 24000` are accepted.
- [ ] Test that missing `session.type` is accepted for clients that omit it, but `session.type == "transcription"` is rejected with `unsupported_session_option`.
- [ ] Test that top-level `output_audio_format` is rejected as beta-only with `unsupported_session_option`.
- [ ] Test that `input_audio_buffer.append` decodes valid base64 PCM bytes.
- [ ] Test that malformed base64 returns `invalid_audio`.
- [ ] Test that a decoded append larger than `REALTIME_MAX_BUFFERED_AUDIO_BYTES` returns `payload_too_large`.
- [ ] Test exact OpenAI-shaped server dictionaries for:
  - `session.created`
  - `session.updated`
  - `response.created`
  - `response.output_text.delta`
  - `response.output_audio.delta`
  - `response.output_audio_transcript.delta`
  - `response.done`
  - `rate_limits.updated`
  - `error`
- [ ] Run the red test:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio/test_realtime_protocol_adapter.py -v
```

Expected red result: pytest imports the new test module and fails because `tldw_Server_API.app.core.Audio.Realtime.protocol` does not exist.

### Task 1.2: Implement Protocol Models And Adapter

- [ ] Add constants in `constants.py` for event names, formats, sample rates, and limits.
- [ ] Add internal command dataclasses in `models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ClientEventType = Literal[
    "session.update",
    "input_audio_buffer.append",
    "input_audio_buffer.commit",
    "input_audio_buffer.clear",
    "response.create",
    "response.cancel",
]


@dataclass(frozen=True)
class RealtimeSessionConfig:
    model: str | None = None
    voice: str | None = None
    instructions: str | None = None
    input_format: str = "pcm16"
    input_sample_rate_hz: int = 16000
    output_format: str = "pcm16"
    output_sample_rate_hz: int = 24000
    turn_detection: str = "manual"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UpdateSessionCommand:
    event_id: str | None
    config: RealtimeSessionConfig
```

- [ ] Add remaining command dataclasses: `AppendAudioCommand`, `CommitAudioCommand`, `ClearAudioCommand`, `CreateResponseCommand`, `CancelResponseCommand`, `UnsupportedCommand`.
- [ ] Add server event dataclasses: `SessionCreatedEvent`, `SessionUpdatedEvent`, `InputAudioCommittedEvent`, `ConversationItemAddedEvent`, `ResponseCreatedEvent`, `ResponseTextDeltaEvent`, `ResponseAudioDeltaEvent`, `ResponseTranscriptDeltaEvent`, `ResponseDoneEvent`, `RateLimitsUpdatedEvent`, `RealtimeErrorEvent`.
- [ ] Implement `parse_client_event(payload: dict[str, Any], limits: RealtimeLimits) -> ClientCommand | RealtimeErrorEvent`.
- [ ] Implement `to_openai_server_event(event: RealtimeServerEvent) -> dict[str, Any]`.
- [ ] Keep all base64 encoding and decoding in `protocol.py`.
- [ ] Keep OpenAI field names out of session and pipeline modules except where converting through the adapter.
- [ ] Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio/test_realtime_protocol_adapter.py -v
```

Expected result: all protocol adapter tests pass.

### Task 1.3: Implement Capabilities

- [ ] Add `RealtimeCapabilities` and `build_realtime_capabilities()` in `capabilities.py`.
- [ ] Report supported routes:
  - `/api/v1/audio/realtime`
  - `/v1/realtime`
- [ ] Report modalities:
  - input: `audio`
  - output: `audio`, `text`
- [ ] Report `conversation.item.create` as unsupported.
- [ ] Report turn detection as manual only.
- [ ] Report `rate_limits.updated` semantics as tldw quota compatibility, not OpenAI quota parity.
- [ ] Add `test_realtime_capabilities.py`.
- [ ] Assert capability output includes every event listed in the accepted spec.
- [ ] Assert capability output includes exact format and limit constants.
- [ ] Assert capability output marks the route experimental.
- [ ] Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio/test_realtime_capabilities.py -v
```

Expected result: all capability tests pass.

---

## Stage 2: Session Orchestrator And Pipeline Boundary

**Goal:** Build the internal realtime session state machine and a swappable pipeline interface that can be driven by fake adapters in tests and real STT/LLM/TTS adapters later.

**Success Criteria:** The session can apply updates, append and commit audio, create and cancel responses, suppress stale output, and stream internal events without knowing OpenAI event names.

**Tests:** Orchestrator tests use fake pipeline and fake persistence only.

**Status:** Complete

### Files

Create:

- `tldw_Server_API/app/core/Audio/Realtime/pipeline.py`
- `tldw_Server_API/app/core/Audio/Realtime/session.py`
- `tldw_Server_API/app/core/Audio/Realtime/persistence.py`
- `tldw_Server_API/tests/Audio/test_realtime_session.py`
- `tldw_Server_API/tests/Audio/test_realtime_persistence.py`

### Task 2.1: Write Session Tests First

- [ ] Add a deterministic `FakeRealtimePipeline` inside `test_realtime_session.py`.
- [ ] Add a deterministic `FakeRealtimePersistenceAdapter` inside `test_realtime_persistence.py`.
- [ ] Test session construction emits a `SessionCreatedEvent` with a stable `session_id` prefix of `sess_`.
- [ ] Test `UpdateSessionCommand` changes voice, model, instructions, output format, and metadata.
- [ ] Test first audio append emits `InputAudioSpeechStartedEvent`.
- [ ] Test manual commit emits `InputAudioSpeechStoppedEvent`, `InputAudioCommittedEvent`, and starts a turn.
- [ ] Test `CreateResponseCommand` creates a `response_id` prefix of `resp_` and a monotonically increasing `generation_id`.
- [ ] Test fake pipeline text chunks become internal `ResponseTextDeltaEvent` instances.
- [ ] Test fake pipeline transcript chunks become internal `ResponseTranscriptDeltaEvent` instances.
- [ ] Test fake pipeline audio chunks become internal `ResponseAudioDeltaEvent` instances.
- [ ] Test `CancelResponseCommand` increments `generation_id` and suppresses late chunks from the canceled generation.
- [ ] Test ephemeral sessions call no persistence writes.
- [ ] Test metadata `{"tldw": {"persist": True, "conversation_id": "abc"}}` writes user transcript and assistant text through the persistence adapter.
- [ ] Run the red test:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Audio/test_realtime_session.py \
  tldw_Server_API/tests/Audio/test_realtime_persistence.py \
  -v
```

Expected red result: pytest imports the new tests and fails because `session.py`, `pipeline.py`, and `persistence.py` do not exist.

### Task 2.2: Implement Pipeline Protocol

- [ ] Add `RealtimePipelineEvent` dataclasses in `pipeline.py` for text delta, audio chunk, transcript delta, and done markers.
- [ ] Add `RealtimePipeline` protocol:

```python
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from tldw_Server_API.app.core.Audio.Realtime.models import RealtimeSessionConfig


class RealtimePipeline(Protocol):
    async def transcribe_pcm16(self, audio: bytes, *, sample_rate_hz: int, language: str | None) -> str:
        raise NotImplementedError

    def stream_turn(self, transcript: str, *, config: RealtimeSessionConfig) -> AsyncIterator[RealtimePipelineEvent]:
        raise NotImplementedError
```

- [ ] Keep provider-specific imports out of `pipeline.py`.
- [ ] Make `stream_turn` yield text, assistant-audio-transcript, and audio events so the session does not have to tee a one-shot async text iterator.
- [ ] Make fake pipelines in tests implement this protocol.

### Task 2.3: Implement Session State Machine

- [ ] Add `RealtimeSession` in `session.py`.
- [ ] Store:
  - `session_id`
  - `turn_index`
  - `active_response_id`
  - `generation_id`
  - `config`
  - `input_audio_buffer`
  - `buffer_started`
  - `closed`
  - `active_task`
- [ ] Use `secrets.token_urlsafe(12)` or `uuid.uuid4().hex` for IDs; do not use predictable counters for externally visible IDs.
- [ ] Use a local monotonic integer for `generation_id`.
- [ ] Bound input buffers using `REALTIME_MAX_BUFFERED_AUDIO_BYTES`.
- [ ] On commit, snapshot and clear the buffer before calling the pipeline.
- [ ] On cancel, increment `generation_id`, cancel the active task, and emit a done event with status `cancelled`.
- [ ] Suppress every pipeline output whose captured generation ID does not equal the current active generation ID.
- [ ] Convert pipeline exceptions into `RealtimeErrorEvent(code="internal_error")` and a response done status of `failed`.
- [ ] Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Audio/test_realtime_session.py \
  tldw_Server_API/tests/Audio/test_realtime_persistence.py \
  -v
```

Expected result: all Stage 2 session and persistence tests pass.

### Task 2.4: Implement Persistence Adapter Interface

- [ ] Add `RealtimePersistenceAdapter` protocol in `persistence.py`.
- [ ] Add `NoopRealtimePersistenceAdapter`.
- [ ] Add `RealtimePersistenceConfig` with:
  - `enabled: bool`
  - `conversation_id: str | None`
  - `store_raw_audio: bool = False`
- [ ] Add `persistence_config_from_metadata(metadata: dict[str, Any]) -> RealtimePersistenceConfig`.
- [ ] Enable persistence only when metadata contains `tldw.persist == True`.
- [ ] Reject raw audio persistence in Stage 1 even if requested; keep the config value false.
- [ ] Record user transcript and assistant text only after the turn reaches done status.
- [ ] Add tests that persistence adapter failures yield `RealtimeErrorEvent(code="internal_error")` only after the response completes and do not discard already streamed output.

---

## Stage 3: Auth, WebSocket Handler, And Route Registration

**Goal:** Expose native and OpenAI-compatible WebSocket routes with correct auth behavior, route gating, capability HTTP response, and policy mappings.

**Success Criteria:** `/api/v1/audio/realtime` and `/v1/realtime` mount only when `audio-realtime` is enabled, `/v1/realtime` never consumes `session.update` as an auth message, and both routes produce JSON-only OpenAI-shaped frames.

**Tests:** WebSocket auth and integration tests run against fake pipeline without live providers.

**Status:** Complete

### Files

Create:

- `tldw_Server_API/app/core/Audio/Realtime/auth.py`
- `tldw_Server_API/app/core/Audio/Realtime/handler.py`
- `tldw_Server_API/app/api/v1/endpoints/audio/audio_realtime.py`
- `tldw_Server_API/app/api/v1/endpoints/realtime_compat.py`
- `tldw_Server_API/tests/Audio/test_realtime_auth.py`
- `tldw_Server_API/tests/Audio/test_realtime_websocket.py`
- `tldw_Server_API/tests/Resource_Governance/test_realtime_route_policy.py`

Modify:

- `tldw_Server_API/app/core/Audio/streaming_service.py`
- `tldw_Server_API/app/api/v1/router_groups/content.py`
- `tldw_Server_API/app/api/v1/router_groups/minimal.py`
- `tldw_Server_API/Config_Files/README.md`
- `tldw_Server_API/Config_Files/privilege_catalog.yaml`
- `tldw_Server_API/Config_Files/resource_governor_policies.yaml`

### Task 3.1: Refactor Existing WebSocket Auth Safely

- [ ] Modify `_audio_ws_authenticate` to accept a keyword-only argument:

```python
allow_initial_auth_message: bool = True
```

- [ ] In multi-user mode, skip the first-message auth fallback when `allow_initial_auth_message` is false.
- [ ] In single-user mode, skip the first-message auth fallback when `allow_initial_auth_message` is false.
- [ ] When no header/subprotocol credential is present and first-message auth is disabled, emit the existing auth error payload and close with `4401`.
- [ ] Preserve existing behavior for current WebSocket routes by leaving the default true.
- [ ] Add `test_realtime_auth.py` coverage that `session.update` is not read by auth on `/v1/realtime`.
- [ ] Add regression coverage that existing `/api/v1/audio/stream/transcribe` behavior is unchanged when first-message auth is still enabled.

### Task 3.2: Add Realtime Auth Adapter

- [ ] Implement `authenticate_realtime_websocket(websocket, route_kind)` in `auth.py`.
- [ ] For route kind `openai_compat`, call `_audio_ws_authenticate` with:
  - `endpoint_id="audio.realtime"`
  - `ws_path="/v1/realtime"`
  - `allow_initial_auth_message=False`
- [ ] For route kind `native`, call `_audio_ws_authenticate` with:
  - `endpoint_id="audio.realtime"`
  - `ws_path="/api/v1/audio/realtime"`
  - `allow_initial_auth_message=False`
- [ ] Accept `Authorization: Bearer <single-user-api-key>` in single-user mode through the refactored helper.
- [ ] Accept `X-API-KEY` through the refactored helper.
- [ ] Add subprotocol auth support only if it can echo the selected subprotocol through `websocket.accept(subprotocol=selected)`. If this cannot be implemented in the first route slice, reject subprotocol-only auth with `authentication_failed` and document header auth as the Stage 1 path.

### Task 3.3: Implement WebSocket Handler

- [ ] Implement `handle_realtime_websocket(websocket, route_kind, pipeline_factory, persistence_factory)` in `handler.py`.
- [ ] Accept the WebSocket only after authentication succeeds.
- [ ] Enforce `REALTIME_MAX_JSON_FRAME_BYTES` before JSON parsing.
- [ ] Reject binary frames with `invalid_event`.
- [ ] Parse text frames as JSON dictionaries only.
- [ ] Feed parsed dictionaries into `parse_client_event`.
- [ ] Send every internal event through `to_openai_server_event`.
- [ ] Send `session.created` immediately after accept.
- [ ] Emit `rate_limits.updated` after session creation with available tldw quota context, or an empty `rate_limits` list and documented semantics when no quota detail is available.
- [ ] Keep recoverable protocol errors on the socket.
- [ ] Close with documented codes for auth, oversized payloads, fatal internal errors, and client disconnect.

### Task 3.4: Add Routers With Feature Flag Isolation

- [ ] Add `audio_realtime.py` with:
  - `router = APIRouter(tags=["Audio Realtime"])`
  - `ws_router = APIRouter(tags=["Audio Realtime"])`
  - `GET /realtime/capabilities`
  - `WS /realtime`
- [ ] Add `realtime_compat.py` with:
  - `router = APIRouter(tags=["OpenAI Realtime Compatibility"])`
  - `WS /realtime`
- [ ] Do not include realtime routers inside aggregate `audio.py`; register them as separate imported router specs so `audio-realtime` gates them independently.
- [ ] In `content.py`, append imported specs when audio imports are enabled:

```python
ImportedRouterSpec(
    import_path="tldw_Server_API.app.api.v1.endpoints.audio.audio_realtime",
    log_name="audio_realtime",
    prefix=f"{API_V1_PREFIX}/audio",
    tags=("audio-realtime",),
    route_key="audio-realtime",
)
ImportedRouterSpec(
    import_path="tldw_Server_API.app.api.v1.endpoints.audio.audio_realtime",
    log_name="audio_realtime_ws",
    prefix=f"{API_V1_PREFIX}/audio",
    tags=("audio-realtime",),
    route_key="audio-realtime",
    attr_name="ws_router",
)
ImportedRouterSpec(
    import_path="tldw_Server_API.app.api.v1.endpoints.realtime_compat",
    log_name="openai_realtime_compat",
    prefix="/v1",
    tags=("audio-realtime",),
    route_key="audio-realtime",
)
```

- [ ] Mirror the same specs in `minimal.py` under the audio optional router block.
- [ ] Add route toggle documentation to `Config_Files/README.md`, including `audio-realtime` in the route key list.
- [ ] Add privilege `audio.realtime` to `privilege_catalog.yaml`.
- [ ] Add `audio-realtime: audio.default` to `resource_governor_policies.yaml` `by_route`.
- [ ] Add `"/v1/realtime": audio.default` to `resource_governor_policies.yaml` `by_path`.
- [ ] Keep `"/api/v1/audio*": audio.default` as the native path fallback.

### Task 3.5: Add WebSocket And Policy Tests

- [ ] Add `test_realtime_websocket.py`.
- [ ] Use `ws_client_without_lifespan` from `tldw_Server_API/tests/Audio/ws_test_helpers.py`.
- [ ] Set `MINIMAL_TEST_INCLUDE_AUDIO=1` for tests that need audio router imports.
- [ ] Inject fake pipeline and persistence factories through app state or monkeypatchable module-level factories in `audio_realtime.py` and `realtime_compat.py`.
- [ ] Test `/v1/realtime` basic manual turn event order:
  - `session.created`
  - `rate_limits.updated`
  - `session.updated`
  - `input_audio_buffer.speech_started`
  - `input_audio_buffer.speech_stopped`
  - `input_audio_buffer.committed`
  - `conversation.item.added`
  - `conversation.item.done`
  - `response.created`
  - `response.output_item.created`
  - `response.content_part.added`
  - `response.output_text.delta`
  - `response.output_audio_transcript.delta`
  - `response.output_audio.delta`
  - `response.output_text.done`
  - `response.output_audio_transcript.done`
  - `response.output_audio.done`
  - `response.done`
- [ ] Test `/api/v1/audio/realtime` uses the same event shape.
- [ ] Test unsupported `conversation.item.create` returns `error.code == "unsupported_event"` and leaves the socket open.
- [ ] Test oversized JSON frame closes with `1009`.
- [ ] Test binary frame receives `invalid_event` or closes with documented code based on Starlette behavior.
- [ ] Test route disabled by `ROUTES_DISABLE=audio-realtime` removes both routes.
- [ ] Test resource governor policy lookup maps `audio-realtime` and `/v1/realtime`.
- [ ] Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Audio/test_realtime_auth.py \
  tldw_Server_API/tests/Audio/test_realtime_websocket.py \
  tldw_Server_API/tests/Resource_Governance/test_realtime_route_policy.py \
  -v
```

Expected result: all Stage 3 tests pass.

---

## Stage 4: Real Pipeline Adapter

**Goal:** Wire the realtime orchestrator to existing STT, LLM, and TTS services without routing through older WebSocket handlers.

**Success Criteria:** The default adapter can transcribe committed PCM16 audio, stream chat text, and stream PCM audio using existing lower-level services, while tests remain deterministic through dependency injection.

**Tests:** Unit tests monkeypatch service callables and do not call external providers by default.

**Status:** Complete

### Files

Create:

- `tldw_Server_API/app/core/Audio/Realtime/default_pipeline.py`
- `tldw_Server_API/tests/Audio/test_realtime_default_pipeline.py`

Modify if needed after inspecting signatures during execution:

- `tldw_Server_API/app/api/v1/endpoints/audio/audio_transcriptions.py`
- `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`
- `tldw_Server_API/app/core/Audio/transcription_service.py`

### Task 4.1: Write Default Pipeline Tests First

- [x] Add `test_realtime_default_pipeline.py`.
- [x] Monkeypatch the STT callable to return `"hello world"`.
- [x] Monkeypatch `perform_chat_api_call_async` to return an async text iterator yielding `"hello "` and `"there"`.
- [x] Monkeypatch TTS service `open_realtime_session` to return a fake realtime TTS session yielding two PCM chunks.
- [x] Test the adapter sends `OpenAISpeechRequest(response_format="pcm", stream=True, target_sample_rate=24000)`.
- [x] Test text is passed into the TTS realtime session incrementally and committed at turn end.
- [x] Test `stream_turn` yields text deltas before done and yields audio chunks from the fake TTS session.
- [x] Test assistant transcript deltas mirror the spoken text emitted by the adapter.
- [x] Test non-streaming chat return values are normalized into one text delta when the provider does not stream.
- [x] Test STT exceptions become `RealtimePipelineError(stage="stt")`.
- [x] Test LLM exceptions become `RealtimePipelineError(stage="llm")`.
- [x] Test TTS exceptions become `RealtimePipelineError(stage="tts")`.
- [x] Run the red test:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio/test_realtime_default_pipeline.py -v
```

Expected red result: pytest imports the test module and fails because `default_pipeline.py` does not exist.

### Task 4.2: Implement Default Pipeline Adapter

- [x] Add `DefaultRealtimePipeline` in `default_pipeline.py`.
- [x] Constructor arguments:
  - `stt_transcribe_pcm16: Callable[..., Awaitable[str]]`
  - `chat_call: Callable[..., Awaitable[Any]]`
  - `tts_service_factory: Callable[[], Any]`
  - `default_model: str`
  - `default_voice: str`
  - `provider_hint: str | None`
  - `user_id: int | None`
- [x] Provide a module-level factory `build_default_realtime_pipeline(principal) -> DefaultRealtimePipeline`.
- [x] Reuse existing STT code by extracting or wrapping a lower-level batch transcription helper. The helper must accept raw PCM16 bytes plus sample rate and must not require an HTTP `UploadFile`.
- [x] Use `perform_chat_api_call_async` for LLM calls.
- [x] Normalize streaming chat chunks and non-streaming chat responses into an async iterator of text deltas.
- [x] Implement `stream_turn` as the only public streaming method on the default pipeline.
- [x] Within `stream_turn`, push each LLM text delta to the realtime TTS session and yield a typed text event for the same delta.
- [x] Within `stream_turn`, yield transcript events for the assistant text that will be spoken.
- [x] Within `stream_turn`, concurrently drain TTS audio chunks and yield typed audio events until the TTS session finishes.
- [x] Use `TTSServiceV2.open_realtime_session` when available.
- [x] Fall back to `generate_speech` through `BufferedRealtimeSession` when a provider lacks native realtime TTS.
- [x] Use `OpenAISpeechRequest` with:
  - `response_format="pcm"`
  - `stream=True`
  - `target_sample_rate=24000`
  - `voice=<resolved voice>`
  - `model=<resolved TTS model>`
- [x] Do not import heavy STT or TTS model modules at import time; resolve them inside the factory or call path.
- [x] Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio/test_realtime_default_pipeline.py -v
```

Expected result: all default pipeline tests pass.

### Task 4.3: Wire Default Factories Into Routers

- [x] In `audio_realtime.py`, set the production pipeline factory to `build_default_realtime_pipeline`.
- [x] In `realtime_compat.py`, set the production pipeline factory to the same factory.
- [x] Keep tests able to monkeypatch factories without importing heavy providers.
- [x] Run the Stage 3 WebSocket tests again with fake factories:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio/test_realtime_websocket.py -v
```

Expected result: all WebSocket tests still pass.

---

## Stage 5: Documentation, Live Smoke Markers, And Final Verification

**Goal:** Document the supported Stage 1 surface, add opt-in smoke tests for configured providers, and run focused verification plus Bandit on touched code.

**Success Criteria:** Docs describe exact supported behavior and deferred features. Default CI tests stay provider-free. Security scan on touched implementation paths reports no new findings.

**Tests:** Focused unit/integration tests and Bandit pass.

**Status:** Not Started

### Files

Create:

- `tldw_Server_API/tests/Audio/test_realtime_live_smoke.py`

Modify:

- `Docs/Audio_Streaming_Protocol.md`
- `Docs/Product/Realtime_Voice_Latency_PRD.md`
- `Docs/superpowers/specs/2026-07-01-openai-realtime-speech-endpoint-design.md`
- `backlog/tasks/task-12088 - Design-OpenAI-compatible-realtime-speech-endpoint.md`

### Task 5.1: Update Docs

- [ ] Add a `OpenAI-Compatible Realtime Speech` section to `Docs/Audio_Streaming_Protocol.md`.
- [ ] Document both routes:
  - `WS /api/v1/audio/realtime`
  - `WS /v1/realtime`
  - `GET /api/v1/audio/realtime/capabilities`
- [ ] Document required auth headers for Stage 1.
- [ ] Document that `/v1/realtime` does not accept first-message auth.
- [ ] Document the input/output audio formats, sample rates, and limits.
- [ ] Document supported client events and server events.
- [ ] Document `conversation.item.create` as explicitly unsupported in Stage 1.
- [ ] Document `rate_limits.updated` as tldw quota compatibility semantics.
- [ ] Update `Docs/Product/Realtime_Voice_Latency_PRD.md` with a note that the new endpoint carries `generation_id` from Stage 1 and that latency/interruption benchmarks remain Stage 2 work.
- [ ] Update the design spec status from `Draft for user review` to `Accepted for implementation` after implementation begins.

### Task 5.2: Add Live Smoke Test Marker

- [ ] Add `test_realtime_live_smoke.py`.
- [ ] Mark the module with `pytestmark = [pytest.mark.external_api, pytest.mark.local_llm_service]`.
- [ ] Skip unless `TLDW_REALTIME_LIVE_SMOKE=1`.
- [ ] Require explicit provider environment variables for STT, LLM, and TTS.
- [ ] Send a short generated PCM16 silence-plus-tone fixture through `/v1/realtime`.
- [ ] Assert a `response.done` event arrives.
- [ ] Keep this test out of default verification commands.

### Task 5.3: Run Focused Verification

- [ ] Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Audio/test_realtime_protocol_adapter.py \
  tldw_Server_API/tests/Audio/test_realtime_capabilities.py \
  tldw_Server_API/tests/Audio/test_realtime_session.py \
  tldw_Server_API/tests/Audio/test_realtime_persistence.py \
  tldw_Server_API/tests/Audio/test_realtime_auth.py \
  tldw_Server_API/tests/Audio/test_realtime_websocket.py \
  tldw_Server_API/tests/Audio/test_realtime_default_pipeline.py \
  tldw_Server_API/tests/Resource_Governance/test_realtime_route_policy.py \
  -v
```

Expected result: all focused tests pass.

- [ ] Run route/config regression tests:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Config/test_route_toggle_evaluations_regressions.py \
  tldw_Server_API/tests/Resource_Governance/test_auth_route_map_coverage.py \
  tldw_Server_API/tests/Resource_Governance/test_policy_loader_route_map_db_store.py \
  -v
```

Expected result: all selected route and policy tests pass.

- [ ] Run Bandit on touched implementation paths:

```bash
source .venv/bin/activate
python -m bandit \
  -r tldw_Server_API/app/core/Audio/Realtime \
  tldw_Server_API/app/api/v1/endpoints/audio/audio_realtime.py \
  tldw_Server_API/app/api/v1/endpoints/realtime_compat.py \
  tldw_Server_API/app/core/Audio/streaming_service.py \
  -f json \
  -o /tmp/bandit_audio_realtime.json
```

Expected result: Bandit exits successfully, and `/tmp/bandit_audio_realtime.json` contains no new high or medium findings in touched code.

### Task 5.4: Finalize Backlog And Commit

- [ ] Update Backlog task metadata with implementation docs, touched files, verification commands, Bandit result path, known live smoke skip conditions, and final summary.
- [ ] Run diff checks:

```bash
git diff --check
git status --short
```

Expected result: no whitespace errors. Git status shows only files intentionally changed for the realtime endpoint work plus unrelated pre-existing workspace changes.

- [ ] Stage only files touched for this implementation.
- [ ] Commit with a message that references the feature and includes the reason for the adapter-first design:

```bash
git commit -m "feat: add OpenAI-compatible realtime speech endpoint"
```

Expected result: commit succeeds without bypassing hooks.

---

## Implementation Notes

- Use a separate `audio_realtime` router spec instead of adding the realtime route to aggregate `audio.py`. This preserves independent `audio-realtime` route gating.
- Keep all OpenAI compatibility field names in `protocol.py`, `handler.py`, and endpoint docs. Session, pipeline, and persistence modules should use internal names.
- Keep default tests deterministic and provider-free. Live provider validation belongs only in the opt-in smoke module.
- Do not persist raw audio in Stage 1.
- Do not silently accept beta-only fields. Return `unsupported_session_option`.
- Do not call existing WebSocket handlers as subroutines. They own receive/send loops and custom protocol assumptions.

## Self-Review Checklist

- [x] Every acceptance criterion in the accepted spec maps to at least one task above.
- [x] Exact Stage 1 audio formats and limits are selected.
- [x] `/v1/realtime` auth cannot consume the first OpenAI client event.
- [x] `audio-realtime` route gating is independent from generic `audio` and `audio-websocket`.
- [x] Test commands use `source .venv/bin/activate`.
- [x] Bandit command covers touched implementation paths.
- [x] Default tests avoid live providers.
