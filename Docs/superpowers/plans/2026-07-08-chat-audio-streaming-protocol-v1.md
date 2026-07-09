# Chat Audio Streaming Protocol V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement strict v1 websocket audio streaming for WebUI and browser-extension chat audio paths so dictation, push-to-talk, voice chat, turn detection, and VAD all use one explicit contract.

**Architecture:** Add one small shared backend parser that validates config frames and converts PCM16 wire audio to Float32 bytes before existing STT/VAD handlers see it. Keep the existing websocket routes and frontend hooks; update them to send strict config first, PCM16 JSON audio frames, and explicit mode/control events.

**Tech Stack:** FastAPI websockets, Python stdlib `base64` plus existing `numpy`, existing `UnifiedStreamingTranscriber`, React hooks, Vitest, Bun, pytest.

## Global Constraints

- Source design: `Docs/superpowers/specs/2026-07-07-chat-audio-streaming-protocol-v1-design.md`.
- Backlog task for this plan: TASK-12913.
- No new websocket endpoint URLs.
- No new Python or frontend dependencies.
- V1 accepts only `protocol_version: 1`, `audio_format: "pcm16"`, `sample_rate: 16000`, `channels: 1`.
- `/api/v1/audio/chat/stream` accepts only `voice_chat` and `push_to_talk`.
- `/api/v1/audio/stream/transcribe` accepts only `dictate` and `captions`.
- Auth may be first. The first post-auth frame must be `type=config`.
- Raw binary websocket audio is invalid in v1.
- Server VAD is authoritative for `voice_chat`.
- `push_to_talk` does not depend on VAD.
- Dictation and captions never call LLM or TTS.
- Keep the existing file-upload transcription endpoint unchanged.
- Use TDD for each task. Run the smallest test that proves the task before moving on.
- Run Bandit on touched Python scopes before completion.

---

## Review Findings This Plan Must Close

- `WSControlSession.apply_config()` silently maps missing/unsupported protocol versions to `1`; do not reuse it as the v1 validator.
- `/stream/transcribe` currently proceeds with defaults on missing or invalid config; strict v1 must close `4400`.
- PCM16 must be normalized before quota, VAD, and paused-buffer duration accounting.
- `useMicStream()` hardcodes `live_voice` as the audio owner; v1 needs mode-specific owners.
- Streaming dictation needs a partial preview and final insertion path that does not overwrite user edits.
- Extension STT must send config before signaling open and must stop sending raw binary frames.

## File Structure

- Create `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/audio_stream_protocol.py`
  - Strict v1 config validation.
  - Endpoint/mode allowlist.
  - PCM16 JSON audio frame decoding.
  - PCM16-to-Float32 conversion.
  - Consistent error payload helper.

- Create `tldw_Server_API/tests/Audio/test_audio_stream_protocol_v1.py`
  - Unit coverage for strict config, endpoint mode allowlists, base64 validation, PCM16 normalization, and duration accounting.

- Modify `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Streaming_Unified.py`
  - Use the shared parser for `/audio/stream/transcribe`.
  - Remove config fail-open behavior for the websocket path.
  - Process Float32 bytes and parser-computed seconds.

- Modify `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`
  - Use the shared parser for `/audio/chat/stream`.
  - Decode and normalize audio before quota/VAD/transcriber.
  - Add `push_to_talk_release` handling.
  - Include `commit_source` in full transcript payloads.

- Modify existing backend tests:
  - `tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py`
  - `tldw_Server_API/tests/Audio/test_ws_transcribe_partial_persistence.py`
  - `tldw_Server_API/tests/Audio/test_ws_fallbacks.py`
  - Add focused tests only where current tests do not cover strict v1 behavior.

- Modify `apps/packages/ui/src/hooks/useMicStream.ts`
  - Add `owner` option while keeping PCM16 as default format.

- Modify `apps/packages/ui/src/hooks/useVoiceChatStream.tsx`
  - Send strict v1 voice-chat config.
  - Stop requesting Float32 capture.
  - Use audio owner `voice_chat`.
  - Add push-to-talk release sender when the UI calls it.

- Modify `apps/packages/ui/src/hooks/useServerDictation.tsx`
  - Replace MediaRecorder upload internals with streaming WS dictation.
  - Use `useMicStream(..., { owner: "dictation" })`.
  - Send strict v1 dictation config.
  - Surface partial preview separately from final transcript.

- Modify `apps/packages/ui/src/components/Chat/composer/hooks/useComposerVoiceChat.ts`
  - Preserve final `onTranscript(text)` behavior.
  - Add optional partial callback path for streaming dictation preview.
  - Stop the other dictation owner before starting a new one.

- Modify `apps/packages/ui/src/entries/background.ts`
  - Send strict `captions` config after auth and before posting `open`.
  - Convert extension STT chunks to JSON base64 audio frames.

- Modify or add frontend tests:
  - `apps/packages/ui/src/hooks/__tests__/useMicStream.test.tsx`
  - `apps/packages/ui/src/hooks/__tests__/useVoiceChatStream.defaults.test.tsx`
  - `apps/packages/ui/src/hooks/__tests__/useServerDictation.source.test.tsx`
  - `apps/packages/ui/src/components/Chat/composer/__tests__/useComposerVoiceChat.test.tsx`
  - `apps/packages/ui/src/entries/__tests__/background.stt-protocol.test.ts`

---

### Task 1: Backend Strict Protocol Parser

**Files:**
- Create: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/audio_stream_protocol.py`
- Create: `tldw_Server_API/tests/Audio/test_audio_stream_protocol_v1.py`

**Interfaces:**
- Produces:
  - `AudioProtocolError(code: str, message: str, close_code: int = 4400)`
  - `AudioProtocolConfig(endpoint_id: str, mode: str, sample_rate: int, channels: int, audio_format: str)`
  - `DecodedAudioFrame(float32_bytes: bytes, seconds: float, sample_rate: int)`
  - `validate_audio_stream_config(frame: dict[str, Any], endpoint_id: str) -> AudioProtocolConfig`
  - `decode_audio_frame(frame: dict[str, Any], config: AudioProtocolConfig) -> DecodedAudioFrame`
  - `audio_protocol_error_payload(exc: AudioProtocolError, request_id: str | None = None) -> dict[str, Any]`
- Consumes: no project-local helpers except existing `numpy`.

- [ ] **Step 1: Write failing parser tests**

Add this file:

```python
# tldw_Server_API/tests/Audio/test_audio_stream_protocol_v1.py
import base64
import binascii
import struct

import numpy as np
import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.audio_stream_protocol import (
    AudioProtocolError,
    decode_audio_frame,
    validate_audio_stream_config,
)


def _pcm16_frame(samples: list[int]) -> dict[str, str]:
    raw = struct.pack("<" + "h" * len(samples), *samples)
    return {"type": "audio", "data": base64.b64encode(raw).decode("ascii")}


def test_validate_chat_voice_config_accepts_strict_v1():
    cfg = validate_audio_stream_config(
        {
            "type": "config",
            "protocol_version": 1,
            "mode": "voice_chat",
            "audio_format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        },
        "audio.chat.stream",
    )

    assert cfg.mode == "voice_chat"
    assert cfg.sample_rate == 16000


@pytest.mark.parametrize(
    "frame",
    [
        {"type": "config", "mode": "voice_chat", "audio_format": "pcm16", "sample_rate": 16000, "channels": 1},
        {"type": "config", "protocol_version": 2, "mode": "voice_chat", "audio_format": "pcm16", "sample_rate": 16000, "channels": 1},
        {"type": "config", "protocol_version": 1, "mode": "voice_chat", "audio_format": "float32", "sample_rate": 16000, "channels": 1},
        {"type": "config", "protocol_version": 1, "mode": "voice_chat", "audio_format": "pcm16", "sample_rate": 48000, "channels": 1},
        {"type": "config", "protocol_version": 1, "mode": "voice_chat", "audio_format": "pcm16", "sample_rate": 16000, "channels": 2},
    ],
)
def test_validate_config_rejects_non_strict_v1(frame):
    with pytest.raises(AudioProtocolError):
        validate_audio_stream_config(frame, "audio.chat.stream")


def test_validate_config_rejects_wrong_endpoint_mode():
    with pytest.raises(AudioProtocolError):
        validate_audio_stream_config(
            {
                "type": "config",
                "protocol_version": 1,
                "mode": "voice_chat",
                "audio_format": "pcm16",
                "sample_rate": 16000,
                "channels": 1,
            },
            "audio.stream.transcribe",
        )


def test_decode_audio_frame_converts_pcm16_to_float32_and_seconds():
    cfg = validate_audio_stream_config(
        {
            "type": "config",
            "protocol_version": 1,
            "mode": "dictate",
            "audio_format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        },
        "audio.stream.transcribe",
    )

    decoded = decode_audio_frame(_pcm16_frame([0, 32767, -32768]), cfg)
    audio = np.frombuffer(decoded.float32_bytes, dtype=np.float32)

    assert decoded.sample_rate == 16000
    assert decoded.seconds == pytest.approx(3 / 16000)
    assert audio.tolist() == pytest.approx([0.0, 32767 / 32768, -1.0])


def test_decode_audio_frame_rejects_invalid_base64():
    cfg = validate_audio_stream_config(
        {
            "type": "config",
            "protocol_version": 1,
            "mode": "captions",
            "audio_format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        },
        "audio.stream.transcribe",
    )

    with pytest.raises(AudioProtocolError):
        decode_audio_frame({"type": "audio", "data": "not base64 ***"}, cfg)


def test_decode_audio_frame_rejects_odd_pcm16_byte_count():
    cfg = validate_audio_stream_config(
        {
            "type": "config",
            "protocol_version": 1,
            "mode": "captions",
            "audio_format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        },
        "audio.stream.transcribe",
    )
    frame = {"type": "audio", "data": base64.b64encode(b"\x00").decode("ascii")}

    with pytest.raises(AudioProtocolError):
        decode_audio_frame(frame, cfg)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio/test_audio_stream_protocol_v1.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `audio_stream_protocol`.

- [ ] **Step 3: Implement the minimal parser**

Create:

```python
# tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/audio_stream_protocol.py
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import numpy as np

AUDIO_CHAT_ENDPOINT = "audio.chat.stream"
AUDIO_TRANSCRIBE_ENDPOINT = "audio.stream.transcribe"

_ALLOWED_MODES = {
    AUDIO_CHAT_ENDPOINT: {"voice_chat", "push_to_talk"},
    AUDIO_TRANSCRIBE_ENDPOINT: {"dictate", "captions"},
}


class AudioProtocolError(ValueError):
    def __init__(self, code: str, message: str, close_code: int = 4400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.close_code = close_code


@dataclass(frozen=True, slots=True)
class AudioProtocolConfig:
    endpoint_id: str
    mode: str
    sample_rate: int
    channels: int
    audio_format: str


@dataclass(frozen=True, slots=True)
class DecodedAudioFrame:
    float32_bytes: bytes
    seconds: float
    sample_rate: int


def validate_audio_stream_config(frame: dict[str, Any], endpoint_id: str) -> AudioProtocolConfig:
    if not isinstance(frame, dict) or frame.get("type") != "config":
        raise AudioProtocolError("bad_request", "First post-auth frame must be type=config")
    if frame.get("protocol_version") != 1:
        raise AudioProtocolError("bad_request", "protocol_version must be 1")

    mode = str(frame.get("mode") or "").strip()
    allowed = _ALLOWED_MODES.get(endpoint_id)
    if allowed is None:
        raise AudioProtocolError("bad_request", f"Unsupported audio endpoint {endpoint_id}")
    if mode not in allowed:
        raise AudioProtocolError("bad_request", f"Mode {mode or 'missing'} is not allowed for {endpoint_id}")

    if frame.get("audio_format") != "pcm16":
        raise AudioProtocolError("bad_request", "audio_format must be pcm16")
    if frame.get("sample_rate") != 16000:
        raise AudioProtocolError("bad_request", "sample_rate must be 16000")
    if frame.get("channels") != 1:
        raise AudioProtocolError("bad_request", "channels must be 1")

    return AudioProtocolConfig(
        endpoint_id=endpoint_id,
        mode=mode,
        sample_rate=16000,
        channels=1,
        audio_format="pcm16",
    )


def decode_audio_frame(frame: dict[str, Any], config: AudioProtocolConfig) -> DecodedAudioFrame:
    if not isinstance(frame, dict) or frame.get("type") != "audio":
        raise AudioProtocolError("bad_request", "Audio frame must be type=audio")
    data = frame.get("data")
    if not isinstance(data, str) or not data:
        raise AudioProtocolError("bad_request", "Audio frame data must be base64 PCM16")
    try:
        pcm16_bytes = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise AudioProtocolError("bad_request", "Invalid base64 audio frame") from exc
    if len(pcm16_bytes) % 2:
        raise AudioProtocolError("bad_request", "PCM16 audio frame has an odd byte count")

    pcm16 = np.frombuffer(pcm16_bytes, dtype="<i2")
    float32 = (pcm16.astype(np.float32) / 32768.0).astype(np.float32, copy=False)
    return DecodedAudioFrame(
        float32_bytes=float32.tobytes(),
        seconds=float(pcm16.size) / float(config.sample_rate),
        sample_rate=config.sample_rate,
    )


def audio_protocol_error_payload(
    exc: AudioProtocolError,
    request_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": "error", "code": exc.code, "message": exc.message}
    if request_id:
        payload["request_id"] = request_id
    return payload
```

- [ ] **Step 4: Run parser tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio/test_audio_stream_protocol_v1.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/audio_stream_protocol.py tldw_Server_API/tests/Audio/test_audio_stream_protocol_v1.py
git commit -m "feat: add strict audio stream protocol parser"
```

---

### Task 2: Enforce Strict V1 On Transcription Websocket

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Streaming_Unified.py`
- Modify: `tldw_Server_API/tests/Audio/test_ws_fallbacks.py`
- Modify: `tldw_Server_API/tests/Audio/test_ws_transcribe_partial_persistence.py`

**Interfaces:**
- Consumes:
  - `validate_audio_stream_config(frame, "audio.stream.transcribe")`
  - `decode_audio_frame(frame, protocol_config)`
  - `audio_protocol_error_payload(error)`
- Produces:
  - `/api/v1/audio/stream/transcribe` rejects invalid first post-auth frames with `4400`.
  - Handler duration accounting uses parser-provided `seconds`.

- [ ] **Step 1: Write failing strict-config tests**

Add two tests to `tldw_Server_API/tests/Audio/test_ws_fallbacks.py` using the existing fake websocket style in that file:

```python
@pytest.mark.asyncio
async def test_transcribe_ws_rejects_audio_before_config(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Streaming_Unified as unified

    ws = _FakeWebSocket([
        json.dumps({"type": "audio", "data": base64.b64encode(b"\x00\x00").decode("ascii")}),
    ])

    await unified.handle_unified_websocket(ws, unified.UnifiedStreamingConfig())

    assert any(msg.get("type") == "error" and msg.get("code") == "bad_request" for msg in ws.sent_json)
    assert ws.close_code == 4400


@pytest.mark.asyncio
async def test_transcribe_ws_rejects_wrong_mode(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Streaming_Unified as unified

    ws = _FakeWebSocket([
        json.dumps({
            "type": "config",
            "protocol_version": 1,
            "mode": "voice_chat",
            "audio_format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        }),
    ])

    await unified.handle_unified_websocket(ws, unified.UnifiedStreamingConfig())

    assert any("not allowed" in msg.get("message", "") for msg in ws.sent_json)
    assert ws.close_code == 4400
```

If `_FakeWebSocket` in that file uses a different sent-message attribute, use its existing attribute name and keep the assertions identical in meaning.

- [ ] **Step 2: Run focused tests to verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio/test_ws_fallbacks.py::test_transcribe_ws_rejects_audio_before_config tldw_Server_API/tests/Audio/test_ws_fallbacks.py::test_transcribe_ws_rejects_wrong_mode -q
```

Expected: FAIL because the handler currently falls back to defaults.

- [ ] **Step 3: Add strict config handling**

In `Audio_Streaming_Unified.py`, import:

```python
from .audio_stream_protocol import (
    AUDIO_TRANSCRIBE_ENDPOINT,
    AudioProtocolError,
    audio_protocol_error_payload,
    decode_audio_frame,
    validate_audio_stream_config,
)
```

Replace the initial config receive block inside `handle_unified_websocket()` with this shape:

```python
        config_payload: dict[str, Any] = {}
        protocol_config = None
        try:
            logger.info("Waiting for required v1 configuration message from client...")
            config_message = await asyncio.wait_for(websocket.receive_text(), timeout=15.0)
            config_data = json.loads(config_message)
            if not isinstance(config_data, dict):
                raise AudioProtocolError("bad_request", "First post-auth frame must be a JSON object")
            protocol_config = validate_audio_stream_config(config_data, AUDIO_TRANSCRIBE_ENDPOINT)
            config_payload = config_data
            config_received = True
        except AudioProtocolError as exc:
            await stream.send_json(audio_protocol_error_payload(exc))
            with contextlib.suppress(_AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS):
                await websocket.close(code=exc.close_code)
            return
        except (asyncio.TimeoutError, json.JSONDecodeError) as exc:
            protocol_error = AudioProtocolError("bad_request", "config frame required")
            await stream.send_json(audio_protocol_error_payload(protocol_error))
            with contextlib.suppress(_AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS):
                await websocket.close(code=4400)
            return
```

Then keep the existing model/language/diarization config parsing under `if config_received:` and force the v1 audio fields after it:

```python
        if protocol_config is None:
            protocol_error = AudioProtocolError("bad_request", "config frame required")
            await stream.send_json(audio_protocol_error_payload(protocol_error))
            with contextlib.suppress(_AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS):
                await websocket.close(code=4400)
            return

        config.sample_rate = protocol_config.sample_rate
```

- [ ] **Step 4: Decode transcribe audio with the parser**

In the message loop, replace:

```python
                    audio_base64 = data.get("data", "")
                    audio_bytes = base64.b64decode(audio_base64)
```

with:

```python
                    try:
                        decoded = decode_audio_frame(data, protocol_config)
                    except AudioProtocolError as exc:
                        await stream.send_json(audio_protocol_error_payload(exc))
                        with contextlib.suppress(_AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS):
                            await websocket.close(code=exc.close_code)
                        return
                    audio_bytes = decoded.float32_bytes
```

Replace both calls to `_estimate_audio_seconds(audio_bytes, int(config.sample_rate or 16000))` in that audio branch with `decoded.seconds`.

- [ ] **Step 5: Update existing transcribe websocket tests that intentionally omitted config**

For any direct `handle_unified_websocket()` test that was not specifically testing fallback behavior, make the first fake incoming message this config:

```python
json.dumps({
    "type": "config",
    "protocol_version": 1,
    "mode": "dictate",
    "audio_format": "pcm16",
    "sample_rate": 16000,
    "channels": 1,
})
```

For any fake audio payload in those tests, encode PCM16 bytes:

```python
json.dumps({"type": "audio", "data": base64.b64encode(b"\x00\x00" * 1600).decode("ascii")})
```

- [ ] **Step 6: Run transcribe tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio/test_audio_stream_protocol_v1.py tldw_Server_API/tests/Audio/test_ws_fallbacks.py tldw_Server_API/tests/Audio/test_ws_transcribe_partial_persistence.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Streaming_Unified.py tldw_Server_API/tests/Audio/test_ws_fallbacks.py tldw_Server_API/tests/Audio/test_ws_transcribe_partial_persistence.py
git commit -m "feat: enforce audio stream protocol on transcription ws"
```

---

### Task 3: Enforce Strict V1 On Chat Audio Websocket

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py`
- Modify: `tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py`

**Interfaces:**
- Consumes:
  - `validate_audio_stream_config(frame, "audio.chat.stream")`
  - `decode_audio_frame(frame, protocol_config)`
  - `audio_protocol_error_payload(error, request_id)`
- Produces:
  - Chat stream rejects wrong endpoint modes.
  - Chat stream processes Float32 bytes after parser normalization.
  - `push_to_talk_release` commits with `commit_source: "push_to_talk_release"`.

- [ ] **Step 1: Write failing chat strict-mode tests**

Add tests to `tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py`:

```python
@pytest.mark.asyncio
async def test_audio_chat_rejects_dictate_mode(monkeypatch):
    ws = FakeWebSocket([
        json.dumps({
            "type": "config",
            "protocol_version": 1,
            "mode": "dictate",
            "audio_format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        }),
    ])

    await audio.websocket_audio_chat_stream(ws, token=None)

    assert any("not allowed" in msg.get("message", "") for msg in ws.sent_json)
    assert ws.close_code == 4400


@pytest.mark.asyncio
async def test_audio_chat_push_to_talk_release_commits_without_vad(monkeypatch):
    _patch_audio_chat_success_dependencies(monkeypatch)
    ws = FakeWebSocket([
        json.dumps({
            "type": "config",
            "protocol_version": 1,
            "mode": "push_to_talk",
            "audio_format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
            "stt": {"enable_vad": False},
        }),
        json.dumps({"type": "audio", "data": base64.b64encode(b"\x00\x00" * 1600).decode("ascii")}),
        json.dumps({"type": "push_to_talk_release"}),
        json.dumps({"type": "stop"}),
    ])

    await audio.websocket_audio_chat_stream(ws, token=None)

    transcript = next(msg for msg in ws.sent_json if msg.get("type") == "full_transcript")
    assert transcript["commit_source"] == "push_to_talk_release"
    assert transcript["auto_commit"] is False
```

Use the existing fake websocket and dependency patch names in the file. If the helper names differ, use the local helpers already used by the closest passing audio-chat tests.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py::test_audio_chat_rejects_dictate_mode tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py::test_audio_chat_push_to_talk_release_commits_without_vad -q
```

Expected: FAIL because chat currently lacks endpoint/mode validation and `push_to_talk_release`.

- [ ] **Step 3: Add parser imports**

In `audio_streaming.py`, add:

```python
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.audio_stream_protocol import (
    AUDIO_CHAT_ENDPOINT,
    AudioProtocolError,
    audio_protocol_error_payload,
    decode_audio_frame,
    validate_audio_stream_config,
)
```

- [ ] **Step 4: Validate initial chat config**

After `cfg_data` is parsed and before `stt_cfg = cfg_data.get("stt") or cfg_data`, add:

```python
        try:
            protocol_config = validate_audio_stream_config(cfg_data, AUDIO_CHAT_ENDPOINT)
        except AudioProtocolError as exc:
            stt_request_status = "bad_request"
            stt_session_close_reason = "error"
            emit_stt_error_total(
                endpoint="audio.chat.stream",
                provider=stt_metrics_provider,
                reason="validation_error",
            )
            if _outer_stream:
                await _outer_stream.send_json(audio_protocol_error_payload(exc, request_id=request_id))
            await websocket.close(code=exc.close_code)
            return
```

After `config = _new_unified_streaming_config()`, force the protocol sample rate:

```python
        config.sample_rate = protocol_config.sample_rate
```

When applying `stt_cfg`, do not let client `stt.sample_rate` override v1. Replace:

```python
            config.sample_rate = stt_cfg.get("sample_rate", config.sample_rate)
```

with:

```python
            config.sample_rate = protocol_config.sample_rate
```

- [ ] **Step 5: Add commit source to turn finalization**

Change `_finalize_turn` signature:

```python
        async def _finalize_turn(
            commit_at: Optional[float],
            *,
            auto: bool = False,
            turn_id: Optional[str] = None,
            commit_source: str = "manual_commit",
        ) -> None:
```

Use this exact payload addition after `payload.update(...)`:

```python
                payload["commit_source"] = commit_source
```

Change `_start_turn` signature:

```python
        async def _start_turn(
            commit_at: Optional[float],
            *,
            auto: bool,
            commit_source: str = "manual_commit",
        ) -> Optional[str]:
```

Pass it into `_finalize_turn`:

```python
                    await _finalize_turn(
                        commit_at,
                        auto=auto,
                        turn_id=turn_id,
                        commit_source=commit_source,
                    )
```

For VAD auto-commit, call:

```python
                    commit_source="vad",
```

For manual commit, call:

```python
                        await _start_turn(time.time(), auto=False, commit_source="manual_commit")
```

- [ ] **Step 6: Decode chat audio with parser before quota/VAD**

In the chat message loop, replace the `base64.b64decode` block with:

```python
                    try:
                        decoded = decode_audio_frame(data, protocol_config)
                    except AudioProtocolError as exc:
                        emit_stt_error_total(
                            endpoint="audio.chat.stream",
                            provider=stt_metrics_provider,
                            reason="validation_error",
                        )
                        if _outer_stream:
                            await _outer_stream.send_json(audio_protocol_error_payload(exc, request_id=request_id))
                        await websocket.close(code=exc.close_code)
                        break
                    audio_bytes = decoded.float32_bytes
                    seconds = decoded.seconds
```

Delete the old `_estimate_stream_audio_seconds(...)` fallback block in that branch. Keep `_on_audio_quota(seconds, decoded.sample_rate)`.

Call `_process_chat_audio` with mode-aware auto-commit:

```python
                    await _process_chat_audio(
                        audio_bytes,
                        allow_auto_commit=protocol_config.mode == "voice_chat",
                    )
```

- [ ] **Step 7: Handle push-to-talk release**

Add this branch before the existing `elif msg_type in {"control", "commit", "reset", "stop"}`:

```python
                elif msg_type == "push_to_talk_release":
                    if protocol_config.mode != "push_to_talk":
                        exc = AudioProtocolError(
                            "bad_request",
                            "push_to_talk_release is only valid in push_to_talk mode",
                        )
                        if _outer_stream:
                            await _outer_stream.send_json(audio_protocol_error_payload(exc, request_id=request_id))
                        await websocket.close(code=exc.close_code)
                        break
                    await _start_turn(
                        time.time(),
                        auto=False,
                        commit_source="push_to_talk_release",
                    )
```

- [ ] **Step 8: Run chat audio tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py
git commit -m "feat: enforce audio stream protocol on chat ws"
```

---

### Task 4: Send PCM16 Voice Chat Frames And Use Mode-Specific Mic Owners

**Files:**
- Modify: `apps/packages/ui/src/hooks/useMicStream.ts`
- Modify: `apps/packages/ui/src/hooks/useVoiceChatStream.tsx`
- Modify: `apps/packages/ui/src/hooks/__tests__/useMicStream.test.tsx`
- Modify: `apps/packages/ui/src/hooks/__tests__/useVoiceChatStream.defaults.test.tsx`
- Modify: `apps/packages/ui/src/hooks/__tests__/useVoiceChatStream.interrupt.test.tsx`

**Interfaces:**
- Consumes:
  - Existing `arrayBufferToBase64`.
  - Existing `useMicStream(onChunk, options)`.
- Produces:
  - `MicStreamOptions.owner?: "live_voice" | "voice_chat" | "push_to_talk" | "dictation" | "captions"`.
  - `useVoiceChatStream` sends top-level strict v1 config fields.
  - `useVoiceChatStream` captures PCM16 by default.

- [ ] **Step 1: Update failing hook expectations**

In `useVoiceChatStream.defaults.test.tsx`, change the mic options assertion to:

```ts
expect(lastCall?.[1]).toEqual({ owner: "voice_chat" })
```

Add an assertion to the config test:

```ts
expect(JSON.parse(MockWebSocket.instances[0].sentMessages.at(-1) as string)).toMatchObject({
  type: "config",
  protocol_version: 1,
  mode: "voice_chat",
  audio_format: "pcm16",
  sample_rate: 16000,
  channels: 1
})
```

- [ ] **Step 2: Run frontend tests to verify failure**

Run:

```bash
cd apps/packages/ui && bun run test src/hooks/__tests__/useVoiceChatStream.defaults.test.tsx src/hooks/__tests__/useMicStream.test.tsx
```

Expected: FAIL because `useVoiceChatStream` still requests Float32 and config lacks strict v1 fields.

- [ ] **Step 3: Add owner option to `useMicStream`**

In `useMicStream.ts`, replace the option type with:

```ts
export type MicStreamOwner =
  | "live_voice"
  | "voice_chat"
  | "push_to_talk"
  | "dictation"
  | "captions"

export type MicStreamOptions = {
  format?: MicStreamFormat
  owner?: MicStreamOwner
}
```

After `const streamFormat = options.format ?? "pcm16"`, add:

```ts
const captureOwner = options.owner ?? "live_voice"
```

Replace both hardcoded `"live_voice"` coordinator calls:

```ts
getAudioCaptureSessionCoordinator().release(captureOwner)
coordinator.claim(captureOwner)
```

Add `captureOwner` to both `useCallback` dependency arrays that use it.

- [ ] **Step 4: Update voice chat stream config and mic options**

In `useVoiceChatStream.tsx`, replace:

```ts
    { format: "float32" }
```

with:

```ts
    { owner: "voice_chat" }
```

In the websocket `onopen` config send, add the strict top-level fields:

```ts
              JSON.stringify({
                type: "config",
                protocol_version: 1,
                mode: "voice_chat",
                audio_format: "pcm16",
                sample_rate: 16000,
                channels: 1,
                stt: sttConfig,
                llm: preflight.llm,
                tts: preflight.tts
              })
```

- [ ] **Step 5: Run hook tests**

Run:

```bash
cd apps/packages/ui && bun run test src/hooks/__tests__/useMicStream.test.tsx src/hooks/__tests__/useVoiceChatStream.defaults.test.tsx src/hooks/__tests__/useVoiceChatStream.interrupt.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/hooks/useMicStream.ts apps/packages/ui/src/hooks/useVoiceChatStream.tsx apps/packages/ui/src/hooks/__tests__/useMicStream.test.tsx apps/packages/ui/src/hooks/__tests__/useVoiceChatStream.defaults.test.tsx apps/packages/ui/src/hooks/__tests__/useVoiceChatStream.interrupt.test.tsx
git commit -m "feat: send voice chat audio protocol v1 frames"
```

---

### Task 5: Convert Server Dictation To Streaming V1

**Files:**
- Modify: `apps/packages/ui/src/hooks/useServerDictation.tsx`
- Modify: `apps/packages/ui/src/components/Chat/composer/hooks/useComposerVoiceChat.ts`
- Modify: `apps/packages/ui/src/hooks/__tests__/useServerDictation.source.test.tsx`
- Modify: `apps/packages/ui/src/components/Chat/composer/__tests__/useComposerVoiceChat.test.tsx`

**Interfaces:**
- Consumes:
  - `useMicStream(onChunk, { owner: "dictation" })`.
  - Existing auth/config services used by `useVoiceChatStream`.
- Produces:
  - `UseServerDictationOptions.onPartialTranscript?: (text: string) => void`.
  - Streaming dictation config with `mode: "dictate"`.
  - Final transcript calls existing `onTranscript(text)`.

- [ ] **Step 1: Write dictation streaming tests**

In `useServerDictation.source.test.tsx`, mock `WebSocket` the same way `useVoiceChatStream.defaults.test.tsx` does and mock `useMicStream` so the test can emit chunks.

Add this assertion:

```ts
it("sends strict dictate config before audio frames", async () => {
  const { result } = renderHook(() => buildHook())

  await act(async () => {
    await result.current.startServerDictation()
  })
  MockWebSocket.instances[0].open()
  micState.callback?.(new ArrayBuffer(2))

  const sent = MockWebSocket.instances[0].sentMessages.map((msg) =>
    typeof msg === "string" ? JSON.parse(msg) : msg
  )

  expect(sent[0]).toMatchObject({
    type: "auth",
  })
  expect(sent[1]).toMatchObject({
    type: "config",
    protocol_version: 1,
    mode: "dictate",
    audio_format: "pcm16",
    sample_rate: 16000,
    channels: 1,
  })
  expect(sent[2]).toMatchObject({ type: "audio" })
})
```

Add this partial/final assertion:

```ts
it("emits partial preview separately and final transcript once", async () => {
  const onPartialTranscript = vi.fn()
  const onTranscript = vi.fn()
  const { result } = renderHook(() =>
    buildHook({ onPartialTranscript, onTranscript })
  )

  await act(async () => {
    await result.current.startServerDictation()
  })
  MockWebSocket.instances[0].open()
  MockWebSocket.instances[0].message({ type: "partial", text: "hel" })
  MockWebSocket.instances[0].message({ type: "full_transcript", text: "hello" })

  expect(onPartialTranscript).toHaveBeenCalledWith("hel")
  expect(onTranscript).toHaveBeenCalledWith("hello")
})
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd apps/packages/ui && bun run test src/hooks/__tests__/useServerDictation.source.test.tsx
```

Expected: FAIL because dictation still uses MediaRecorder upload.

- [ ] **Step 3: Replace MediaRecorder internals with websocket streaming**

In `useServerDictation.tsx`, add:

```ts
import { arrayBufferToBase64 } from "@/utils/compress"
import { useMicStream } from "@/hooks/useMicStream"
```

Add to options:

```ts
onPartialTranscript?: (text: string) => void
```

Replace `serverRecorderRef`, `serverStreamRef`, and `serverChunksRef` with:

```ts
const wsRef = React.useRef<WebSocket | null>(null)
```

Create the mic stream at hook top level:

```ts
const { start: micStart, stop: micStop, active: micActive } = useMicStream(
  (chunk) => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    ws.send(JSON.stringify({ type: "audio", data: arrayBufferToBase64(chunk) }))
  },
  { owner: "dictation" }
)
```

In `stopServerDictation`, send stop and close:

```ts
const ws = wsRef.current
try {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "stop" }))
  }
} catch {}
try {
  ws?.close()
} catch {}
wsRef.current = null
micStop()
setIsServerDictating(false)
```

In `startServerDictation`, after `canUseServerStt` checks, create a websocket to `/api/v1/audio/stream/transcribe`, send auth when token exists, send strict config, then start mic:

```ts
const config = await tldwClient.getConfig()
const serverUrl = String(config?.serverUrl || "").trim()
if (!serverUrl) {
  throw new Error("tldw server not configured")
}
const token =
  config?.authMode === "multi-user"
    ? String(config?.accessToken || "").trim()
    : String(config?.apiKey || "").trim()
const wsUrl = `${serverUrl.replace(/^http/, "ws").replace(/\/$/, "")}/api/v1/audio/stream/transcribe`
const ws = new WebSocket(wsUrl)
wsRef.current = ws
ws.onopen = async () => {
  if (token) {
    ws.send(JSON.stringify({ type: "auth", token }))
  }
  ws.send(JSON.stringify({
    type: "config",
    protocol_version: 1,
    mode: "dictate",
    audio_format: "pcm16",
    sample_rate: 16000,
    channels: 1,
    language: speechToTextLanguage,
    model: sttSettings.model?.trim() || undefined,
  }))
  await micStart({ deviceId: requestedDeviceId })
  setIsServerDictating(true)
}
ws.onmessage = (event) => {
  const payload = typeof event.data === "string" ? JSON.parse(event.data) : null
  if (!payload || typeof payload !== "object") return
  if (payload.type === "partial") {
    onPartialTranscript?.(String(payload.text || ""))
  }
  if (payload.type === "full_transcript" || payload.type === "transcription") {
    const text = String(payload.text || "").trim()
    if (text) onTranscript(text)
    onSuccess?.()
  }
  if (payload.type === "error") {
    reportError(payload)
  }
}
ws.onerror = () => reportError(new Error("Dictation websocket error"))
ws.onclose = () => {
  micStop()
  wsRef.current = null
  setIsServerDictating(false)
}
```

- [ ] **Step 4: Wire partial preview through composer hook**

In `UseComposerVoiceChatOptions`, add:

```ts
onPartialTranscript?: (text: string) => void
```

Pass it to `useServerDictation`:

```ts
onPartialTranscript: (text) => options.onPartialTranscript?.(text),
```

Keep the existing final path:

```ts
onTranscript: (text) => onTranscriptRef.current(text),
```

- [ ] **Step 5: Add composer regression test**

In `useComposerVoiceChat.test.tsx`, add:

```ts
it("keeps server dictation partials out of final transcript callback", () => {
  const onTranscript = vi.fn()
  const onPartialTranscript = vi.fn()
  renderHook(() =>
    useComposerVoiceChat({
      ...defaultOptions,
      onTranscript,
      onPartialTranscript,
      canUseServerStt: true,
      dictationModeOverride: "server",
    })
  )

  capturedServerDictationOptions.onPartialTranscript("draft")
  capturedServerDictationOptions.onTranscript("draft final")

  expect(onPartialTranscript).toHaveBeenCalledWith("draft")
  expect(onTranscript).toHaveBeenCalledWith("draft final")
})
```

Use the existing captured-options variable name from that test file.

- [ ] **Step 6: Run dictation tests**

Run:

```bash
cd apps/packages/ui && bun run test src/hooks/__tests__/useServerDictation.source.test.tsx src/components/Chat/composer/__tests__/useComposerVoiceChat.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/hooks/useServerDictation.tsx apps/packages/ui/src/components/Chat/composer/hooks/useComposerVoiceChat.ts apps/packages/ui/src/hooks/__tests__/useServerDictation.source.test.tsx apps/packages/ui/src/components/Chat/composer/__tests__/useComposerVoiceChat.test.tsx
git commit -m "feat: stream server dictation over audio protocol v1"
```

---

### Task 6: Migrate Browser Extension STT To V1 JSON Frames

**Files:**
- Modify: `apps/packages/ui/src/entries/background.ts`
- Create: `apps/packages/ui/src/entries/__tests__/background.stt-protocol.test.ts`

**Interfaces:**
- Consumes:
  - Existing extension storage config.
  - Existing `arrayBufferToBase64` from `@/utils/compress`.
- Produces:
  - Extension STT sends auth, config, then `open`.
  - Extension STT sends JSON base64 audio frames.

- [ ] **Step 1: Write failing background STT protocol test**

Create `apps/packages/ui/src/entries/__tests__/background.stt-protocol.test.ts` with the project’s existing background test mocks and this core assertion:

```ts
it("sends captions config before reporting STT open and wraps audio as JSON", async () => {
  const port = connectRuntimePort("tldw:stt")
  port.postMessage({ action: "connect" })
  const ws = MockWebSocket.instances[0]

  ws.open()

  expect(JSON.parse(ws.sent[0])).toMatchObject({ type: "auth" })
  expect(JSON.parse(ws.sent[1])).toMatchObject({
    type: "config",
    protocol_version: 1,
    mode: "captions",
    audio_format: "pcm16",
    sample_rate: 16000,
    channels: 1,
  })
  expect(port.messages.at(-1)).toEqual({ event: "open" })

  port.postMessage({ action: "audio", data: new Uint8Array([0, 0]).buffer })

  expect(JSON.parse(ws.sent[2])).toMatchObject({
    type: "audio",
    data: "AAA=",
  })
})
```

Use helper names from existing `apps/packages/ui/src/entries/__tests__/background-session-store.test.ts` and `background.web-clipper.test.ts`; keep this test’s behavior assertions unchanged.

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd apps/packages/ui && bun run test src/entries/__tests__/background.stt-protocol.test.ts
```

Expected: FAIL because the background currently posts `open` after auth only and sends raw binary.

- [ ] **Step 3: Send config before open**

In `background.ts`, import:

```ts
import { arrayBufferToBase64 } from "@/utils/compress";
```

In the `tldw:stt` `ws.onopen` handler, after auth send:

```ts
ws?.send(
  JSON.stringify({
    type: "config",
    protocol_version: 1,
    mode: "captions",
    audio_format: "pcm16",
    sample_rate: 16000,
    channels: 1,
  }),
);
```

Keep `safePost({ event: "open" })` after this config send.

- [ ] **Step 4: Wrap extension audio as JSON**

Replace the raw binary send block:

```ts
              if (msg.data instanceof ArrayBuffer) {
                ws.send(msg.data);
              } else if (msg.data?.buffer) {
                ws.send(msg.data.buffer);
              }
```

with:

```ts
              const data =
                msg.data instanceof ArrayBuffer
                  ? msg.data
                  : msg.data?.buffer instanceof ArrayBuffer
                    ? msg.data.buffer
                    : null;
              if (data) {
                ws.send(
                  JSON.stringify({
                    type: "audio",
                    data: arrayBufferToBase64(data),
                  }),
                );
              }
```

- [ ] **Step 5: Run background test**

Run:

```bash
cd apps/packages/ui && bun run test src/entries/__tests__/background.stt-protocol.test.ts
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/entries/background.ts apps/packages/ui/src/entries/__tests__/background.stt-protocol.test.ts
git commit -m "feat: migrate extension stt to audio protocol v1"
```

---

### Task 7: Docs, Full Focused Verification, And Cleanup

**Files:**
- Modify: `Docs/superpowers/specs/2026-07-07-chat-audio-streaming-protocol-v1-design.md`
- Modify: any audio docs or comments that still claim websocket config is optional.
- Modify: `backlog/tasks/task-12913 - Plan-chat-audio-streaming-protocol-v1-implementation.md`

**Interfaces:**
- Consumes: completed tasks 1-6.
- Produces: final verification evidence and updated task notes.

- [ ] **Step 1: Find stale optional-config docs/comments**

Run:

```bash
rg -n "default streaming configuration|does not provide|optional client configuration|Using default configuration|No valid config received|raw binary" tldw_Server_API Docs apps/packages/ui/src
```

Expected: Only intentional historical notes remain. Any route docstring for `/stream/transcribe` or `/chat/stream` must describe strict config.

- [ ] **Step 2: Update stale comments**

Replace stale `/stream/transcribe` docstring text in `audio_streaming.py`:

```python
Supported incoming message types: "auth" (for token-based auth), "config" (required first post-auth frame), "audio" (base64-encoded PCM16 JSON chunks), and "commit" (finalize current utterance).
```

Remove any comment that says server defaults are used when clients omit config on these chat audio websocket paths.

- [ ] **Step 3: Run backend focused tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Audio/test_audio_stream_protocol_v1.py tldw_Server_API/tests/Audio/test_ws_fallbacks.py tldw_Server_API/tests/Audio/test_ws_transcribe_partial_persistence.py tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py -q
```

Expected: PASS.

- [ ] **Step 4: Run frontend focused tests**

Run:

```bash
cd apps/packages/ui && bun run test src/hooks/__tests__/useMicStream.test.tsx src/hooks/__tests__/useVoiceChatStream.defaults.test.tsx src/hooks/__tests__/useVoiceChatStream.interrupt.test.tsx src/hooks/__tests__/useServerDictation.source.test.tsx src/components/Chat/composer/__tests__/useComposerVoiceChat.test.tsx src/entries/__tests__/background.stt-protocol.test.ts
```

Expected: PASS.

- [ ] **Step 5: Run scoped type checks**

Run:

```bash
cd apps/packages/ui && bun run typecheck
```

Expected: PASS or only pre-existing unrelated errors. If unrelated errors exist, record exact first five unrelated file paths in TASK-12913 notes.

- [ ] **Step 6: Run Bandit on touched backend scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/audio_stream_protocol.py tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Streaming_Unified.py tldw_Server_API/app/api/v1/endpoints/audio/audio_streaming.py -f json -o /tmp/bandit_chat_audio_protocol_v1.json
```

Expected: PASS with no new findings in touched code.

- [ ] **Step 7: Run diff checks**

Run:

```bash
git diff --check
git diff --cached --check
```

Expected: both exit 0.

- [ ] **Step 8: Update TASK-12913 final notes**

Run:

```bash
backlog task edit TASK-12913 --append-notes "Implemented chat audio streaming protocol v1 using one strict parser, existing websocket endpoints, PCM16 wire audio, server-side Float32 normalization, mode allowlists, push-to-talk release commit, streaming dictation, and extension STT JSON frames. Verification commands and any known skips are recorded in this task." --plain
```

- [ ] **Step 9: Commit final docs/task update**

```bash
git add Docs/superpowers/specs/2026-07-07-chat-audio-streaming-protocol-v1-design.md "backlog/tasks/task-12913 - Plan-chat-audio-streaming-protocol-v1-implementation.md"
git commit -m "docs: close audio protocol v1 rollout notes"
```

---

## Execution Notes

- The lazy implementation is one backend parser plus edits to existing handlers. Do not introduce a protocol registry, new websocket routes, or a second audio capture coordinator.
- `WSControlSession` remains the control-state helper. It must not be used for strict v1 config validation.
- Existing Persona live voice remains out of scope unless a test proves the shared `useMicStream` owner option breaks it. If that happens, fix the owner default only.
- Existing file-upload transcription remains available through its existing API client path; this plan changes chat-composer server dictation to streaming.

## Self-Review Checklist

- Spec coverage: tasks cover strict config, endpoint modes, PCM16 normalization, server VAD, push-to-talk release, dictation, captions/extension STT, tests, and rollout docs.
- Review findings coverage: every finding from the pre-plan review is mapped to Task 1, 2, 3, 4, 5, or 6.
- Placeholder scan target: this plan must not contain unresolved placeholder markers from the writing-plans skill.
- Type consistency target: parser names introduced in Task 1 are the same names used in Tasks 2 and 3.
