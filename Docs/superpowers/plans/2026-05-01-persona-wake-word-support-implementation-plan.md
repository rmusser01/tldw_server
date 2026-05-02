# Persona Wake Word Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add manually armed, client-side wake phrase support for Persona Live while keeping all persona turns on the existing persona websocket runtime.

**Architecture:** V1 wake detection runs in the active browser or extension surface through a frontend `WakeDetector` abstraction. The detector matches the selected persona profile's saved `voice_chat_trigger_phrases`, then sends a scoped `wake_activation` frame to the existing `/api/v1/persona/stream` websocket. The backend stores short-lived session-local wake state, validates activations against the saved persona profile and current `voice_runtime`, and only bypasses trigger phrase gating inside the active post-wake window.

**Tech Stack:** FastAPI, Pydantic, pytest, React, TypeScript, Ant Design, browser `SpeechRecognition` feature detection, Vitest, Testing Library.

---

## Source Spec

- `Docs/superpowers/specs/2026-04-30-persona-wake-word-support-design.md`

## Stages

### Stage 1: Persist Wake Defaults

**Goal:** Store and return the persona default wake behavior without changing existing profiles.

**Success Criteria:** Profile create, read, and update preserve `voice_defaults.wake_behavior`; invalid values fail validation; missing values remain accepted.

**Tests:** Backend profile API tests covering roundtrip, missing value, and invalid enum values.

**Status:** Not Started

### Stage 2: Backend Wake Activation Runtime

**Goal:** Add websocket `wake_activation` and `wake_deactivation` handling with session-local state and existing trigger gate preservation.

**Success Criteria:** A valid activation allows the next post-wake turn to omit the trigger phrase; invalid activations leave gating unchanged; deactivation clears the bypass.

**Tests:** Focused persona websocket tests for valid activation, invalid phrase, explicit deactivation, one-shot expiry, and no-command timeout.

**Status:** Not Started

### Stage 3: Frontend Types, Defaults, And Detector

**Goal:** Add wake behavior to frontend persona voice defaults and introduce the browser transcript detector behind a small interface.

**Success Criteria:** Defaults resolve to `one_shot`; profile settings can persist another behavior; detector normalizes and matches configured phrases only.

**Tests:** Hook tests for default resolution, panel tests for persistence, detector unit tests for matching and unavailable state.

**Status:** Not Started

### Stage 4: Persona Live Wake Controls

**Goal:** Wire manual arm/disarm, session-local behavior selection, detector lifecycle, and websocket activation into Persona Live.

**Success Criteria:** Users can arm wake listening only when the selected saved profile has trigger phrases; wake mode sends activation/deactivation frames and suspends during active mic capture.

**Tests:** Controller tests, voice card tests, and sidepanel route tests.

**Status:** Not Started

### Stage 5: Verification And Documentation

**Goal:** Run focused backend/frontend checks, Bandit on touched backend files, and document V1 limitations without overwriting unrelated work.

**Success Criteria:** Focused tests pass; `git diff --check` is clean; Bandit report has no new findings in touched backend code; docs explain that V1 is manually armed and browser-lifecycle bounded.

**Tests:** Focused pytest and Vitest commands plus static checks.

**Status:** Not Started

## File Map

### Backend

- Modify `tldw_Server_API/app/api/v1/schemas/persona.py`
  - Add `PersonaWakeBehavior` literal type.
  - Add nullable `wake_behavior` to `PersonaVoiceDefaults`.
  - Rely on Pydantic enum validation for invalid values.

- Modify `tldw_Server_API/app/api/v1/endpoints/persona.py`
  - Add wake behavior and deactivation reason constants.
  - Add a configurable no-command timeout helper.
  - Add helper functions for canonical wake phrase normalization/matching.
  - Add session-local wake activation state for `/api/v1/persona/stream`.
  - Add `wake_activation` and `wake_deactivation` websocket handlers.
  - Update `_commit_persona_live_turn` so active wake state can bypass trigger gating only inside the scoped window.
  - Clear wake state during websocket/session cleanup.

- Modify `tldw_Server_API/tests/Persona/test_persona_profiles_api.py`
  - Extend the existing voice defaults roundtrip test.
  - Add a validation test for invalid wake behavior.

- Modify `tldw_Server_API/tests/Persona/test_persona_ws.py`
  - Add or extend seed helpers for persona profile `voice_defaults`.
  - Add focused wake activation/deactivation websocket tests.

### Frontend

- Modify `apps/packages/ui/src/hooks/useResolvedPersonaVoiceDefaults.tsx`
  - Add `PersonaWakeBehavior`.
  - Add nullable saved `wake_behavior`.
  - Add resolved `wakeBehavior` with fixed fallback `one_shot`.

- Create `apps/packages/ui/src/hooks/personaWakeDetector.ts`
  - Export detector types and `BrowserTranscriptWakeDetector`.
  - Export pure phrase normalization and matching helpers.
  - Feature-detect `window.SpeechRecognition` and `window.webkitSpeechRecognition`.

- Modify `apps/packages/ui/src/hooks/usePersonaLiveVoiceController.tsx`
  - Accept raw saved `wakeTriggerPhrases`.
  - Accept or create a `WakeDetector`.
  - Own manual arming state, detector state, wake warnings, and session-local wake behavior.
  - Send `wake_activation` and `wake_deactivation` frames.
  - Stop or pause the detector during active Persona Live mic capture.

- Modify `apps/packages/ui/src/components/PersonaGarden/AssistantDefaultsPanel.tsx`
  - Add persistent default wake behavior control near voice defaults.
  - Include `wake_behavior` in form state, payload, and resolved preview.

- Modify `apps/packages/ui/src/components/PersonaGarden/AssistantVoiceCard.tsx`
  - Add explicit `Listen for wake phrase` arm/disarm control.
  - Add wake status, behavior selector, no-phrases blocked state, and unavailable/error copy.

- Modify `apps/packages/ui/src/routes/sidepanel-persona.tsx`
  - Derive `wakeTriggerPhrases` from the raw saved selected profile `voice_defaults.voice_chat_trigger_phrases`.
  - Do not use `resolvedDefaults.voiceChatTriggerPhrases` for wake arming because that value can include global fallback phrases.
  - Pass wake controller state and actions into `AssistantVoiceCard`.
  - Stop wake listening before persona/session switches.

- Modify `apps/packages/ui/src/routes/personaTypes.ts`
  - Re-export or consume updated `PersonaVoiceDefaults` shape.
  - Keep turn-detection helpers focused on VAD fields; do not mix wake behavior into VAD change detection.

### Frontend Tests

- Modify `apps/packages/ui/src/hooks/__tests__/useResolvedPersonaVoiceDefaults.test.tsx`
- Create `apps/packages/ui/src/hooks/__tests__/personaWakeDetector.test.ts`
- Modify `apps/packages/ui/src/hooks/__tests__/usePersonaLiveVoiceController.test.tsx`
- Modify `apps/packages/ui/src/components/PersonaGarden/__tests__/AssistantDefaultsPanel.test.tsx`
- Modify `apps/packages/ui/src/components/PersonaGarden/__tests__/LiveSessionPanel.test.tsx`
- Modify `apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx`

### Documentation

- Prefer updating `Docs/Design/Personas.md` only if it has no unrelated working-tree changes when implementation reaches the docs task.
- If `Docs/Design/Personas.md` is still dirty with unrelated edits, create `Docs/Design/Persona_Wake_Word_Support.md` instead and leave the dirty file untouched.

## Cross-Cutting Invariants

- V1 must never auto-arm wake listening.
- V1 must not send pre-wake audio to the tldw server.
- Wake arming must use raw saved selected-profile trigger phrases only.
- Existing ordinary Persona Live trigger phrase behavior may continue to use existing resolved fallback phrases.
- `wake_activation` and `wake_behavior` must not authorize tools, memory access, or policy bypass.
- Invalid wake activation must be recoverable and non-fatal.
- Websocket cleanup must clear session-local wake state even if the frontend fails to send `wake_deactivation`.
- Do not touch unrelated dirty changes. At plan creation time `Docs/Design/Personas.md` had an unrelated modification.

## Task 1: Backend Wake Behavior Schema

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/persona.py`
- Modify: `tldw_Server_API/tests/Persona/test_persona_profiles_api.py`

- [ ] **Step 1: Add failing profile roundtrip assertions**

In `tldw_Server_API/tests/Persona/test_persona_profiles_api.py`, extend `test_persona_profile_voice_defaults_roundtrip`:

```python
created_voice_defaults = payload["voice_defaults"]
assert created_voice_defaults["voice_chat_trigger_phrases"] == [
    "hey helper",
    "okay helper",
]
assert created_voice_defaults["wake_behavior"] == "continuous"
```

Assert the create, fetch, and update responses:

```python
assert payload["voice_defaults"]["wake_behavior"] == "continuous"
assert fetched_payload["voice_defaults"]["wake_behavior"] == "continuous"
assert updated_payload["voice_defaults"]["wake_behavior"] == "push_to_talk_after_wake"
```

Add `"wake_behavior": "continuous"` to the create payload and `"wake_behavior": "push_to_talk_after_wake"` to the patch payload.

- [ ] **Step 2: Add failing invalid-value test**

Add a new test in `tldw_Server_API/tests/Persona/test_persona_profiles_api.py`:

```python
def test_persona_profile_voice_defaults_rejects_invalid_wake_behavior(
    persona_db: CharactersRAGDB,
):
    with _client_for_user(1, persona_db) as client:
        created = client.post(
            "/api/v1/persona/profiles",
            json={
                "name": "Bad Wake Helper",
                "mode": "persistent_scoped",
                "voice_defaults": {"wake_behavior": "always_on_background"},
            },
        )
        assert created.status_code == 422, created.text

    fastapi_app.dependency_overrides.clear()
```

- [ ] **Step 3: Run the failing tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_profiles_api.py -q -k "voice_defaults"
```

Expected: FAIL because `wake_behavior` is not accepted or not returned yet.

- [ ] **Step 4: Implement schema field**

In `tldw_Server_API/app/api/v1/schemas/persona.py`, add the literal near the existing persona mode literals:

```python
PersonaWakeBehavior = Literal["one_shot", "continuous", "push_to_talk_after_wake"]
```

Add the nullable field to `PersonaVoiceDefaults`:

```python
wake_behavior: PersonaWakeBehavior | None = None
```

Do not add a separate `wake_phrases` field.

- [ ] **Step 5: Run focused tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_profiles_api.py -q -k "voice_defaults"
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add tldw_Server_API/app/api/v1/schemas/persona.py tldw_Server_API/tests/Persona/test_persona_profiles_api.py
git commit -m "feat(persona): persist wake behavior default"
```

## Task 2: Backend Wake Activation State

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/persona.py`
- Modify: `tldw_Server_API/tests/Persona/test_persona_ws.py`

- [ ] **Step 1: Extend the websocket seed helper**

In `tldw_Server_API/tests/Persona/test_persona_ws.py`, update `_seed_persona_session` with an optional `voice_defaults` parameter:

```python
def _seed_persona_session(
    tmp_path,
    monkeypatch,
    *,
    user_id: str,
    session_id: str,
    mode: str,
    use_persona_state_context_default: bool = True,
    scope_snapshot_json: dict | None = None,
    preferences_json: dict | None = None,
    voice_defaults: dict | None = None,
) -> None:
```

Include it in the persona profile row:

```python
"voice_defaults": dict(voice_defaults or {}),
```

`CharactersRAGDB.create_persona_profile` hydrates `voice_defaults_json` from a `voice_defaults` dict, so the seed helper should pass `voice_defaults`.

- [ ] **Step 2: Add failing valid activation test**

Add reusable fake voice helpers near the existing audio chunk tests:

```python
class _WakeFakePersonaTranscriber:
    def __init__(self, transcript: str):
        self.transcript = transcript
        self.initialize_called = False

    def initialize(self):
        self.initialize_called = True

    async def process_audio_chunk(self, audio_data: bytes):
        return {
            "type": "partial",
            "text": self.transcript,
            "is_final": False,
        }

    def get_full_transcript(self) -> str:
        return self.transcript

    def reset(self):
        return None

    def cleanup(self):
        return None


class _WakeFakeTurnDetector:
    def __init__(self):
        self.available = True
        self.unavailable_reason = None
        self.last_trigger_at = None
        self._triggered = False

    def observe(self, audio_data: bytes) -> bool:
        if self._triggered:
            return False
        self._triggered = True
        self.last_trigger_at = 123.456
        return True

    def reset(self):
        self._triggered = False


def _install_wake_voice_fakes(monkeypatch, transcript: str):
    fake_transcriber = _WakeFakePersonaTranscriber(transcript)
    fake_turn_detector = _WakeFakeTurnDetector()
    monkeypatch.setattr(
        persona_ep,
        "_create_persona_live_stt_transcriber",
        lambda *args, **kwargs: fake_transcriber,
        raising=False,
    )
    monkeypatch.setattr(
        persona_ep,
        "_create_persona_live_turn_detector",
        lambda *args, **kwargs: fake_turn_detector,
        raising=False,
    )
    return fake_transcriber, fake_turn_detector
```

Then add a test near existing audio chunk trigger phrase tests:

```python
def test_persona_wake_activation_allows_next_voice_turn_without_trigger(
    tmp_path,
    monkeypatch,
):
    _install_wake_voice_fakes(monkeypatch, "summarize the current note")
    _seed_persona_session(
        tmp_path,
        monkeypatch,
        user_id="1",
        session_id="sess_wake_valid",
        mode="session_scoped",
        voice_defaults={
            "voice_chat_trigger_phrases": ["hey helper"],
            "wake_behavior": "one_shot",
        },
    )

    with TestClient(fastapi_app) as c:
        with c.websocket_connect("/api/v1/persona/stream") as ws:
            _ = json.loads(ws.receive_text())
            ws.send_text(json.dumps({
                "type": "voice_config",
                "session_id": "sess_wake_valid",
                "voice": {
                    "trigger_phrases": ["hey helper"],
                    "wake_behavior": "one_shot",
                },
                "stt": {"enable_vad": True},
            }))
            _ = _recv_until(
                ws,
                lambda d: d.get("event") == "notice"
                and d.get("reason_code") == "VOICE_CONFIG_UPDATED",
            )
            ws.send_text(json.dumps({
                "type": "wake_activation",
                "session_id": "sess_wake_valid",
                "matched_phrase": "hey helper",
                "detector_kind": "browser_transcript",
                "detected_at_ms": 1714500000000,
            }))
            accepted = _recv_until(
                ws,
                lambda d: d.get("event") == "notice"
                and d.get("reason_code") == "WAKE_ACTIVATION_ACCEPTED",
            )
            assert accepted.get("session_id") == "sess_wake_valid"

            ws.send_text(json.dumps({
                "type": "audio_chunk",
                "session_id": "sess_wake_valid",
                "audio_format": "pcm16",
                "bytes_base64": base64.b64encode(b"\x00\x00\xff\x7f\x00\x80").decode("ascii"),
            }))
            plan = _recv_until(ws, lambda d: d.get("event") == "tool_plan")
            assert plan.get("session_id") == "sess_wake_valid"
```

Expected current failure: no `wake_activation` handler and the no-trigger transcript is ignored.

- [ ] **Step 3: Add failing invalid activation tests**

Add a helper that asserts rejection does not disable the existing trigger gate:

```python
def _assert_wake_activation_rejected_keeps_trigger_gate(
    *,
    tmp_path,
    monkeypatch,
    session_id: str,
    saved_phrases: list[str],
    runtime_phrases: list[str],
    matched_phrase: str,
    wake_behavior: str = "one_shot",
) -> None:
    _install_wake_voice_fakes(monkeypatch, "summarize the current note")
    _seed_persona_session(
        tmp_path,
        monkeypatch,
        user_id="1",
        session_id=session_id,
        mode="session_scoped",
        voice_defaults={
            "voice_chat_trigger_phrases": saved_phrases,
            "wake_behavior": "one_shot",
        },
    )

    with TestClient(fastapi_app) as c:
        with c.websocket_connect("/api/v1/persona/stream") as ws:
            _ = json.loads(ws.receive_text())
            audio_payload = base64.b64encode(b"\x00\x00\xff\x7f\x00\x80").decode("ascii")
            ws.send_text(json.dumps({
                "type": "voice_config",
                "session_id": session_id,
                "voice": {
                    "trigger_phrases": runtime_phrases,
                    "wake_behavior": wake_behavior,
                },
                "stt": {"model": "whisper-1", "language": "en-US"},
            }))
            _ = _recv_until(ws, lambda d: d.get("reason_code") == "VOICE_CONFIG_UPDATED")
            ws.send_text(json.dumps({
                "type": "wake_activation",
                "session_id": session_id,
                "matched_phrase": matched_phrase,
                "detector_kind": "browser_transcript",
                "detected_at_ms": 1714500000000,
            }))
            rejected = _recv_until(
                ws,
                lambda d: d.get("event") == "notice"
                and d.get("reason_code") == "WAKE_ACTIVATION_REJECTED",
            )
            assert rejected.get("session_id") == session_id
            ws.send_text(json.dumps({
                "type": "audio_chunk",
                "session_id": session_id,
                "audio_format": "pcm16",
                "bytes_base64": audio_payload,
            }))
            ignored = _recv_until(
                ws,
                lambda d: d.get("event") == "notice"
                and d.get("reason_code") == "VOICE_TRIGGER_NOT_HEARD",
            )
            assert ignored.get("session_id") == session_id
```

Add these rejection tests:

```python
def test_persona_wake_activation_rejects_phrase_not_saved_in_profile(tmp_path, monkeypatch):
    _assert_wake_activation_rejected_keeps_trigger_gate(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        session_id="sess_wake_reject_saved",
        saved_phrases=["hey helper"],
        runtime_phrases=["runtime only"],
        matched_phrase="runtime only",
    )


def test_persona_wake_activation_rejects_phrase_missing_from_runtime_config(tmp_path, monkeypatch):
    _assert_wake_activation_rejected_keeps_trigger_gate(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        session_id="sess_wake_reject_runtime",
        saved_phrases=["hey helper"],
        runtime_phrases=["okay helper"],
        matched_phrase="hey helper",
    )
```

Also add a positive test proving the server derives wake behavior from `voice_config.voice.wake_behavior` and ignores any client-supplied `wake_behavior` on `wake_activation`.

- [ ] **Step 4: Add failing deactivation and expiry tests**

Add `test_persona_wake_deactivation_restores_trigger_gating`:

```python
def test_persona_wake_deactivation_restores_trigger_gating(tmp_path, monkeypatch):
    _install_wake_voice_fakes(monkeypatch, "summarize the current note")
    _seed_persona_session(
        tmp_path,
        monkeypatch,
        user_id="1",
        session_id="sess_wake_deactivated",
        mode="session_scoped",
        voice_defaults={
            "voice_chat_trigger_phrases": ["hey helper"],
            "wake_behavior": "continuous",
        },
    )

    with TestClient(fastapi_app) as c:
        with c.websocket_connect("/api/v1/persona/stream") as ws:
            _ = json.loads(ws.receive_text())
            audio_payload = base64.b64encode(b"\x00\x00\xff\x7f\x00\x80").decode("ascii")
            ws.send_text(json.dumps({
                "type": "voice_config",
                "session_id": "sess_wake_deactivated",
                "voice": {
                    "trigger_phrases": ["hey helper"],
                    "wake_behavior": "continuous",
                },
                "stt": {"model": "whisper-1", "language": "en-US"},
            }))
            _ = _recv_until(ws, lambda d: d.get("reason_code") == "VOICE_CONFIG_UPDATED")
            ws.send_text(json.dumps({
                "type": "wake_activation",
                "session_id": "sess_wake_deactivated",
                "matched_phrase": "hey helper",
                "detector_kind": "browser_transcript",
                "detected_at_ms": 1714500000000,
            }))
            _ = _recv_until(ws, lambda d: d.get("reason_code") == "WAKE_ACTIVATION_ACCEPTED")
            ws.send_text(json.dumps({
                "type": "wake_deactivation",
                "session_id": "sess_wake_deactivated",
                "reason": "disarmed",
            }))
            _ = _recv_until(ws, lambda d: d.get("reason_code") == "WAKE_DEACTIVATED")
            ws.send_text(json.dumps({
                "type": "audio_chunk",
                "session_id": "sess_wake_deactivated",
                "audio_format": "pcm16",
                "bytes_base64": audio_payload,
            }))
            ignored = _recv_until(
                ws,
                lambda d: d.get("event") == "notice"
                and d.get("reason_code") == "VOICE_TRIGGER_NOT_HEARD",
            )
            assert ignored.get("session_id") == "sess_wake_deactivated"
```

Add `test_persona_wake_activation_one_shot_expires_after_commit`:

```python
def test_persona_wake_activation_one_shot_expires_after_commit(tmp_path, monkeypatch):
    _install_wake_voice_fakes(monkeypatch, "summarize the current note")
    _seed_persona_session(
        tmp_path,
        monkeypatch,
        user_id="1",
        session_id="sess_wake_one_shot",
        mode="session_scoped",
        voice_defaults={
            "voice_chat_trigger_phrases": ["hey helper"],
            "wake_behavior": "one_shot",
        },
    )

    with TestClient(fastapi_app) as c:
        with c.websocket_connect("/api/v1/persona/stream") as ws:
            _ = json.loads(ws.receive_text())
            audio_payload = base64.b64encode(b"\x00\x00\xff\x7f\x00\x80").decode("ascii")
            ws.send_text(json.dumps({
                "type": "voice_config",
                "session_id": "sess_wake_one_shot",
                "voice": {
                    "trigger_phrases": ["hey helper"],
                    "wake_behavior": "one_shot",
                },
                "stt": {"model": "whisper-1", "language": "en-US"},
            }))
            _ = _recv_until(ws, lambda d: d.get("reason_code") == "VOICE_CONFIG_UPDATED")
            ws.send_text(json.dumps({
                "type": "wake_activation",
                "session_id": "sess_wake_one_shot",
                "matched_phrase": "hey helper",
                "detector_kind": "browser_transcript",
                "detected_at_ms": 1714500000000,
            }))
            _ = _recv_until(ws, lambda d: d.get("reason_code") == "WAKE_ACTIVATION_ACCEPTED")
            ws.send_text(json.dumps({
                "type": "audio_chunk",
                "session_id": "sess_wake_one_shot",
                "audio_format": "pcm16",
                "bytes_base64": audio_payload,
            }))
            _ = _recv_until(ws, lambda d: d.get("reason_code") == "VOICE_TURN_COMMITTED")
            _ = _recv_until(ws, lambda d: d.get("event") == "tool_plan")
            ws.send_text(json.dumps({
                "type": "audio_chunk",
                "session_id": "sess_wake_one_shot",
                "audio_format": "pcm16",
                "bytes_base64": audio_payload,
            }))
            ignored = _recv_until(
                ws,
                lambda d: d.get("event") == "notice"
                and d.get("reason_code") == "VOICE_TRIGGER_NOT_HEARD",
            )
            assert ignored.get("session_id") == "sess_wake_one_shot"
```

Add `test_persona_wake_activation_expires_after_no_command_timeout`:

```python
def test_persona_wake_activation_expires_after_no_command_timeout(tmp_path, monkeypatch):
    import time

    _install_wake_voice_fakes(monkeypatch, "summarize the current note")
    monkeypatch.setattr(persona_ep, "_get_persona_wake_no_command_timeout_s", lambda: 0.01)
    _seed_persona_session(
        tmp_path,
        monkeypatch,
        user_id="1",
        session_id="sess_wake_timeout",
        mode="session_scoped",
        voice_defaults={
            "voice_chat_trigger_phrases": ["hey helper"],
            "wake_behavior": "one_shot",
        },
    )

    with TestClient(fastapi_app) as c:
        with c.websocket_connect("/api/v1/persona/stream") as ws:
            _ = json.loads(ws.receive_text())
            audio_payload = base64.b64encode(b"\x00\x00\xff\x7f\x00\x80").decode("ascii")
            ws.send_text(json.dumps({
                "type": "voice_config",
                "session_id": "sess_wake_timeout",
                "voice": {
                    "trigger_phrases": ["hey helper"],
                    "wake_behavior": "one_shot",
                },
                "stt": {"model": "whisper-1", "language": "en-US"},
            }))
            _ = _recv_until(ws, lambda d: d.get("reason_code") == "VOICE_CONFIG_UPDATED")
            ws.send_text(json.dumps({
                "type": "wake_activation",
                "session_id": "sess_wake_timeout",
                "matched_phrase": "hey helper",
                "detector_kind": "browser_transcript",
                "detected_at_ms": 1714500000000,
            }))
            _ = _recv_until(ws, lambda d: d.get("reason_code") == "WAKE_ACTIVATION_ACCEPTED")
            time.sleep(0.02)
            ws.send_text(json.dumps({
                "type": "audio_chunk",
                "session_id": "sess_wake_timeout",
                "audio_format": "pcm16",
                "bytes_base64": audio_payload,
            }))
            ignored = _recv_until(
                ws,
                lambda d: d.get("event") == "notice"
                and d.get("reason_code") == "VOICE_TRIGGER_NOT_HEARD",
            )
            assert ignored.get("session_id") == "sess_wake_timeout"
```

For timeout, monkeypatch the new helper planned below:

```python
monkeypatch.setattr(persona_ep, "_get_persona_wake_no_command_timeout_s", lambda: 0.01)
```

The timeout test should sleep briefly before committing:

```python
import time
time.sleep(0.02)
```

Expected: transcript without trigger is ignored after timeout.

- [ ] **Step 5: Run failing websocket tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_ws.py -q -k "wake_activation or wake_deactivation or persona_audio_chunk_vad_auto_commit_ignores_missing_trigger_phrase"
```

Expected: FAIL on new wake tests; existing missing-trigger test should still pass.

- [ ] **Step 6: Add constants and helpers**

In `tldw_Server_API/app/api/v1/endpoints/persona.py`, near other persona live constants/helpers, add:

```python
_PERSONA_WAKE_BEHAVIORS = {
    "one_shot",
    "continuous",
    "push_to_talk_after_wake",
}
_PERSONA_WAKE_DEACTIVATION_REASONS = {
    "disarmed",
    "stop_live_voice",
    "persona_switch",
    "route_leave",
    "session_close",
}


def _get_persona_wake_no_command_timeout_s() -> float:
    try:
        return max(
            1.0,
            min(300.0, float(os.getenv("PERSONA_WAKE_NO_COMMAND_TIMEOUT_S", "30"))),
        )
    except (TypeError, ValueError):
        return 30.0
```

If `os` is not already imported in the file, add the import.

Add pure normalization helpers close to `_apply_persona_live_trigger_phrases`:

```python
def _normalize_persona_wake_phrase(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return re.sub(r"(^[^\w]+|[^\w]+$)", "", text)


def _match_persona_wake_phrase(
    matched_phrase: object,
    configured_phrases: list[str] | None,
) -> str | None:
    normalized_match = _normalize_persona_wake_phrase(matched_phrase)
    if not normalized_match:
        return None
    for phrase in configured_phrases or []:
        if _normalize_persona_wake_phrase(phrase) == normalized_match:
            return str(phrase or "").strip()
    return None
```

- [ ] **Step 7: Add websocket-scoped wake state**

Inside `stream_persona` where `persona_live_stt_state_by_session` is used, add a sibling dict:

```python
persona_live_wake_state_by_session: dict[str, dict[str, Any]] = {}
```

Add nested helpers so they can reuse `session_manager`, `persona_scope_db`, and authenticated user values:

```python
def _get_active_wake_state(session_id: str) -> dict[str, Any] | None:
    state = persona_live_wake_state_by_session.get(session_id)
    if not isinstance(state, dict):
        return None
    expires_at = state.get("expires_at_monotonic")
    if isinstance(expires_at, (int, float)) and time.monotonic() > float(expires_at):
        persona_live_wake_state_by_session.pop(session_id, None)
        return None
    return state


def _clear_wake_state(session_id: str) -> None:
    persona_live_wake_state_by_session.pop(session_id, None)
```

If `time` is already imported, reuse it; otherwise add the import.

- [ ] **Step 8: Validate saved profile phrases**

Add a nested helper to resolve saved phrases for the authenticated session:

```python
def _get_saved_wake_phrases_for_session(session_id: str) -> list[str]:
    runtime_context = _load_persona_policy_rules_for_session(
        persona_scope_db,
        session_id=session_id,
        user_id=authenticated_user_id,
    )
    persona_id = str(runtime_context.get("persona_id") or "").strip()
    if not persona_id:
        return []
    profile = persona_scope_db.get_persona_profile(
        persona_id=persona_id,
        user_id=authenticated_user_id,
    )
    if not isinstance(profile, dict):
        return []
    voice_defaults = profile.get("voice_defaults")
    if not isinstance(voice_defaults, dict):
        return []
    return [
        str(phrase or "").strip()
        for phrase in voice_defaults.get("voice_chat_trigger_phrases") or []
        if str(phrase or "").strip()
    ]
```

Adjust the profile access if the DB helper returns JSON field names instead of hydrated `voice_defaults`; inspect `CharactersRAGDB.get_persona_profile` before implementing.

- [ ] **Step 9: Add `wake_activation` handler**

In the websocket message loop, add a branch before `voice_commit`:

```python
elif mtype == "wake_activation":
    original_session_id = msg.get("session_id")
    session_id = _normalize_ws_identifier(original_session_id, fallback="")
    if not session_id:
        await _emit_notice(
            session_id=default_session_id,
            level="error",
            message="session_id is required",
            reason_code="SESSION_ID_REQUIRED",
        )
        continue

    matched_phrase = str(msg.get("matched_phrase") or "").strip()
    detector_kind = _bounded_label(
        msg.get("detector_kind"),
        allowed={"browser_transcript", "native_companion", "test"},
        fallback="browser_transcript",
    )
    saved_phrases = _get_saved_wake_phrases_for_session(session_id)
    runtime_preferences = session_manager.get_preferences(
        session_id=session_id,
        user_id=connection_user_id,
    )
    voice_runtime = runtime_preferences.get("voice_runtime")
    runtime_phrases = (
        list(voice_runtime.get("trigger_phrases") or [])
        if isinstance(voice_runtime, dict)
        else []
    )
    wake_behavior = _bounded_label(
        voice_runtime.get("wake_behavior")
        if isinstance(voice_runtime, dict)
        else None,
        allowed=_PERSONA_WAKE_BEHAVIORS,
        fallback="one_shot",
    )
    saved_match = _match_persona_wake_phrase(matched_phrase, saved_phrases)
    runtime_match = _match_persona_wake_phrase(matched_phrase, runtime_phrases)
    if not saved_match or not runtime_match:
        await _emit_notice(
            session_id=session_id,
            level="warning",
            reason_code="WAKE_ACTIVATION_REJECTED",
            message="Wake activation was rejected for this persona session.",
        )
        continue

    expires_at = None
    if wake_behavior in {"one_shot", "push_to_talk_after_wake"}:
        expires_at = time.monotonic() + _get_persona_wake_no_command_timeout_s()
    persona_live_wake_state_by_session[session_id] = {
        "wake_behavior": wake_behavior,
        "matched_phrase": saved_match,
        "detector_kind": detector_kind,
        "expires_at_monotonic": expires_at,
    }
    await _emit_notice(
        session_id=session_id,
        level="info",
        reason_code="WAKE_ACTIVATION_ACCEPTED",
        message="Wake activation accepted for this live session.",
        wake_behavior=wake_behavior,
        detector_kind=detector_kind,
    )
```

- [ ] **Step 10: Add `wake_deactivation` handler**

Add another branch:

```python
elif mtype == "wake_deactivation":
    session_id = _normalize_ws_identifier(msg.get("session_id"), fallback="")
    if not session_id:
        await _emit_notice(
            session_id=default_session_id,
            level="error",
            message="session_id is required",
            reason_code="SESSION_ID_REQUIRED",
        )
        continue
    reason = _bounded_label(
        msg.get("reason"),
        allowed=_PERSONA_WAKE_DEACTIVATION_REASONS,
        fallback="disarmed",
    )
    _clear_wake_state(session_id)
    await _emit_notice(
        session_id=session_id,
        level="info",
        reason_code="WAKE_DEACTIVATED",
        message="Wake activation cleared for this live session.",
        reason=reason,
    )
```

- [ ] **Step 11: Update trigger gating in `_commit_persona_live_turn`**

Inside `_commit_persona_live_turn`, before `_apply_persona_live_trigger_phrases`, read active state:

```python
wake_state = _get_active_wake_state(session_id)
if wake_state:
    trigger_matched = True
    cleaned_transcript = str(transcript or "").strip()
else:
    trigger_matched, cleaned_transcript = _apply_persona_live_trigger_phrases(
        transcript,
        trigger_phrases=trigger_phrases,
    )
```

After a successful commit and before returning, expire one-shot style activations:

```python
if wake_state and wake_state.get("wake_behavior") in {
    "one_shot",
    "push_to_talk_after_wake",
}:
    _clear_wake_state(session_id)
```

Do not clear `continuous` here.

- [ ] **Step 12: Clear wake state on voice config and cleanup**

When `voice_config` successfully updates runtime preferences, clear any stale wake state for that session before emitting `VOICE_CONFIG_UPDATED`:

```python
_clear_wake_state(session_id)
```

In websocket cleanup or disconnect handling, clear wake state for known sessions. If the function already tracks connected session ids, iterate those. If not, clear `default_session_id` and any `persona_live_wake_state_by_session` keys before returning:

```python
persona_live_wake_state_by_session.clear()
```

This dict is scoped to the websocket connection, so clearing it on disconnect is correct.

- [ ] **Step 13: Run focused websocket tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_ws.py -q -k "wake_activation or wake_deactivation or persona_audio_chunk_vad_auto_commit_ignores_missing_trigger_phrase"
```

Expected: PASS.

- [ ] **Step 14: Commit**

Run:

```bash
git add tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/tests/Persona/test_persona_ws.py
git commit -m "feat(persona): add wake activation websocket state"
```

## Task 3: Frontend Wake Defaults

**Files:**
- Modify: `apps/packages/ui/src/hooks/useResolvedPersonaVoiceDefaults.tsx`
- Modify: `apps/packages/ui/src/routes/personaTypes.ts`
- Modify: `apps/packages/ui/src/components/PersonaGarden/AssistantDefaultsPanel.tsx`
- Modify: `apps/packages/ui/src/hooks/__tests__/useResolvedPersonaVoiceDefaults.test.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/__tests__/AssistantDefaultsPanel.test.tsx`

- [ ] **Step 1: Add failing resolver tests**

In `apps/packages/ui/src/hooks/__tests__/useResolvedPersonaVoiceDefaults.test.tsx`, add:

```ts
it("resolves explicit persona wake behavior", () => {
  const { result } = renderHook(() =>
    useResolvedPersonaVoiceDefaults({
      wake_behavior: "continuous"
    })
  )

  expect(result.current.wakeBehavior).toBe("continuous")
})

it("falls back to one_shot when wake behavior is unset", () => {
  const { result } = renderHook(() => useResolvedPersonaVoiceDefaults(null))

  expect(result.current.wakeBehavior).toBe("one_shot")
})
```

- [ ] **Step 2: Add failing defaults panel test**

In `apps/packages/ui/src/components/PersonaGarden/__tests__/AssistantDefaultsPanel.test.tsx`, add a test that:

1. Renders `AssistantDefaultsPanel` with `voice_defaults.wake_behavior = "continuous"`.
2. Changes the wake behavior select to `push_to_talk_after_wake`.
3. Saves.
4. Asserts the API payload includes:

```ts
expect(fetchMock).toHaveBeenCalledWith(
  expect.any(String),
  expect.objectContaining({
    body: expect.stringContaining('"wake_behavior":"push_to_talk_after_wake"')
  })
)
```

Follow the existing fetch mock and save assertions in this test file.

- [ ] **Step 3: Run failing frontend defaults tests**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/hooks/__tests__/useResolvedPersonaVoiceDefaults.test.tsx ../packages/ui/src/components/PersonaGarden/__tests__/AssistantDefaultsPanel.test.tsx
```

Expected: FAIL because `wakeBehavior` and the panel field do not exist yet.

- [ ] **Step 4: Add frontend wake behavior types**

In `apps/packages/ui/src/hooks/useResolvedPersonaVoiceDefaults.tsx`, add:

```ts
export type PersonaWakeBehavior =
  | "one_shot"
  | "continuous"
  | "push_to_talk_after_wake"
```

Extend `PersonaVoiceDefaults`:

```ts
wake_behavior?: PersonaWakeBehavior | null
```

Extend `ResolvedPersonaVoiceDefaults`:

```ts
wakeBehavior: PersonaWakeBehavior
```

Add:

```ts
const DEFAULT_WAKE_BEHAVIOR: PersonaWakeBehavior = "one_shot"
const normalizeWakeBehavior = (
  value: PersonaWakeBehavior | null | undefined
): PersonaWakeBehavior =>
  value === "continuous" || value === "push_to_talk_after_wake" || value === "one_shot"
    ? value
    : DEFAULT_WAKE_BEHAVIOR
```

Return:

```ts
wakeBehavior: normalizeWakeBehavior(personaVoiceDefaults?.wake_behavior)
```

- [ ] **Step 5: Wire defaults panel form state**

In `apps/packages/ui/src/components/PersonaGarden/AssistantDefaultsPanel.tsx`:

Add to `AssistantDefaultsFormState`:

```ts
wakeBehavior: PersonaWakeBehavior
```

Import the type:

```ts
type PersonaWakeBehavior
```

Add to `buildFormState`:

```ts
wakeBehavior: voiceDefaults?.wake_behavior || "one_shot",
```

Add to `buildPayload`:

```ts
wake_behavior: formState.wakeBehavior,
```

Add an Ant Design `Select` near the trigger phrases controls:

```tsx
<Select
  data-testid="assistant-wake-behavior"
  value={formState.wakeBehavior}
  onChange={(wakeBehavior: PersonaWakeBehavior) =>
    setFormState((current) => ({ ...current, wakeBehavior }))
  }
  options={[
    { value: "one_shot", label: "One turn after wake" },
    { value: "continuous", label: "Continuous after wake" },
    { value: "push_to_talk_after_wake", label: "Push to talk after wake" }
  ]}
/>
```

Add `Wake behavior` to the resolved preview.

- [ ] **Step 6: Keep route types aligned**

In `apps/packages/ui/src/routes/personaTypes.ts`, make no independent duplicate enum if it can import `PersonaWakeBehavior` from `useResolvedPersonaVoiceDefaults.tsx`. If any helper enumerates voice default keys, include `wake_behavior` there.

Do not include `wake_behavior` in `buildTurnDetectionValuesFromSavedDefaults`; that helper is for VAD change detection only.

- [ ] **Step 7: Run defaults tests**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/hooks/__tests__/useResolvedPersonaVoiceDefaults.test.tsx ../packages/ui/src/components/PersonaGarden/__tests__/AssistantDefaultsPanel.test.tsx
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
git add apps/packages/ui/src/hooks/useResolvedPersonaVoiceDefaults.tsx apps/packages/ui/src/routes/personaTypes.ts apps/packages/ui/src/components/PersonaGarden/AssistantDefaultsPanel.tsx apps/packages/ui/src/hooks/__tests__/useResolvedPersonaVoiceDefaults.test.tsx apps/packages/ui/src/components/PersonaGarden/__tests__/AssistantDefaultsPanel.test.tsx
git commit -m "feat(persona): expose wake behavior defaults"
```

## Task 4: Frontend WakeDetector Abstraction

**Files:**
- Create: `apps/packages/ui/src/hooks/personaWakeDetector.ts`
- Create: `apps/packages/ui/src/hooks/__tests__/personaWakeDetector.test.ts`

- [ ] **Step 1: Add failing detector tests**

Create `apps/packages/ui/src/hooks/__tests__/personaWakeDetector.test.ts`:

```ts
import {
  BrowserTranscriptWakeDetector,
  findCanonicalWakePhrase,
  normalizeWakePhraseText
} from "../personaWakeDetector"

describe("personaWakeDetector", () => {
  it("normalizes phrase text for matching", () => {
    expect(normalizeWakePhraseText("  Hey,   Helper! ")).toBe("hey helper")
  })

  it("matches whole normalized phrase sequences", () => {
    expect(
      findCanonicalWakePhrase("um hey helper please wake up", ["Hey Helper"])
    ).toBe("Hey Helper")
    expect(
      findCanonicalWakePhrase("the helper is nearby", ["hey helper"])
    ).toBeNull()
  })

  it("reports unavailable when SpeechRecognition is absent", async () => {
    ;(window as any).SpeechRecognition = undefined
    ;(window as any).webkitSpeechRecognition = undefined

    const detector = new BrowserTranscriptWakeDetector()
    await expect(detector.isAvailable()).resolves.toBe(false)
  })
})
```

- [ ] **Step 2: Run failing detector tests**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/hooks/__tests__/personaWakeDetector.test.ts
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement detector types and pure helpers**

Create `apps/packages/ui/src/hooks/personaWakeDetector.ts`:

```ts
export type WakeDetectorState =
  | "idle"
  | "starting"
  | "listening"
  | "detected"
  | "unavailable"
  | "error"

export type WakeDetectorKind = "browser_transcript"

export type WakeDetectedEvent = {
  canonicalPhrase: string
  transcript: string
  detectedAtMs: number
  detectorKind: WakeDetectorKind
}

export type WakeDetectorError = {
  message: string
  code?: string
}

export type WakeDetectorConfig = {
  phrases: string[]
  locale?: string
  onWake: (event: WakeDetectedEvent) => void
  onStateChange?: (state: WakeDetectorState) => void
  onError?: (error: WakeDetectorError) => void
}

export interface WakeDetector {
  isAvailable(): Promise<boolean>
  start(config: WakeDetectorConfig): Promise<void>
  stop(): Promise<void>
}
```

Implement helpers:

```ts
export const normalizeWakePhraseText = (value: unknown): string =>
  String(value || "")
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim()

export const findCanonicalWakePhrase = (
  transcript: unknown,
  phrases: string[]
): string | null => {
  const normalizedTranscript = ` ${normalizeWakePhraseText(transcript)} `
  if (!normalizedTranscript.trim()) return null
  for (const phrase of phrases || []) {
    const normalizedPhrase = normalizeWakePhraseText(phrase)
    if (!normalizedPhrase) continue
    if (normalizedTranscript.includes(` ${normalizedPhrase} `)) {
      return String(phrase || "").trim()
    }
  }
  return null
}
```

- [ ] **Step 4: Implement `BrowserTranscriptWakeDetector`**

Add a class in the same file:

```ts
type SpeechRecognitionCtor = new () => {
  continuous: boolean
  interimResults: boolean
  lang: string
  onresult: ((event: any) => void) | null
  onerror: ((event: any) => void) | null
  onend: (() => void) | null
  start: () => void
  stop: () => void
}

const getSpeechRecognitionCtor = (): SpeechRecognitionCtor | null => {
  if (typeof window === "undefined") return null
  return (
    (window as any).SpeechRecognition ||
    (window as any).webkitSpeechRecognition ||
    null
  )
}

export class BrowserTranscriptWakeDetector implements WakeDetector {
  private recognition: InstanceType<SpeechRecognitionCtor> | null = null
  private active = false

  async isAvailable(): Promise<boolean> {
    return Boolean(getSpeechRecognitionCtor())
  }

  async start(config: WakeDetectorConfig): Promise<void> {
    await this.stop()
    const Ctor = getSpeechRecognitionCtor()
    const phrases = (config.phrases || []).map((phrase) => String(phrase || "").trim()).filter(Boolean)
    if (!Ctor || phrases.length === 0) {
      config.onStateChange?.("unavailable")
      return
    }
    config.onStateChange?.("starting")
    const recognition = new Ctor()
    this.recognition = recognition
    this.active = true
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = config.locale || "en-US"
    recognition.onresult = (event: any) => {
      if (!this.active) return
      const transcript = Array.from(event?.results || [])
        .map((result: any) => result?.[0]?.transcript || "")
        .join(" ")
      const canonicalPhrase = findCanonicalWakePhrase(transcript, phrases)
      if (!canonicalPhrase) return
      config.onStateChange?.("detected")
      config.onWake({
        canonicalPhrase,
        transcript,
        detectedAtMs: Date.now(),
        detectorKind: "browser_transcript"
      })
    }
    recognition.onerror = (event: any) => {
      config.onStateChange?.("error")
      config.onError?.({
        code: String(event?.error || ""),
        message: String(event?.message || event?.error || "Wake detector error")
      })
    }
    recognition.onend = () => {
      if (this.active) {
        config.onStateChange?.("idle")
      }
    }
    recognition.start()
    config.onStateChange?.("listening")
  }

  async stop(): Promise<void> {
    this.active = false
    const recognition = this.recognition
    this.recognition = null
    try {
      recognition?.stop()
    } catch {
      // SpeechRecognition implementations throw when stop races with end.
    }
  }
}
```

If lint flags `any`, replace with the local structural types already used in `useSpeechRecognition.tsx`.

- [ ] **Step 5: Add event lifecycle tests**

Extend `personaWakeDetector.test.ts` with a mock recognition constructor and assert:

- `start` emits `starting` then `listening`.
- matching result calls `onWake` with canonical configured phrase.
- `stop` calls recognition `stop`.
- empty phrase lists produce `unavailable` and do not start recognition.

- [ ] **Step 6: Run detector tests**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/hooks/__tests__/personaWakeDetector.test.ts
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git add apps/packages/ui/src/hooks/personaWakeDetector.ts apps/packages/ui/src/hooks/__tests__/personaWakeDetector.test.ts
git commit -m "feat(persona): add wake detector abstraction"
```

## Task 5: Frontend Controller Wake Runtime

**Files:**
- Modify: `apps/packages/ui/src/hooks/usePersonaLiveVoiceController.tsx`
- Modify: `apps/packages/ui/src/hooks/__tests__/usePersonaLiveVoiceController.test.tsx`

- [ ] **Step 1: Add test detector helper**

In `apps/packages/ui/src/hooks/__tests__/usePersonaLiveVoiceController.test.tsx`, add a fake detector:

```ts
const createWakeDetectorHarness = () => {
  let config: WakeDetectorConfig | null = null
  const detector: WakeDetector = {
    isAvailable: vi.fn(async () => true),
    start: vi.fn(async (nextConfig) => {
      config = nextConfig
      nextConfig.onStateChange?.("listening")
    }),
    stop: vi.fn(async () => undefined)
  }
  return {
    detector,
    fireWake: (canonicalPhrase = "hey helper") => {
      config?.onWake({
        canonicalPhrase,
        transcript: `${canonicalPhrase} status`,
        detectedAtMs: 1714500000000,
        detectorKind: "browser_transcript"
      })
    }
  }
}
```

Import `WakeDetector` and `WakeDetectorConfig` from `../personaWakeDetector`.

- [ ] **Step 2: Add failing arming tests**

Add tests that assert:

```ts
it("blocks wake arming when the selected profile has no saved trigger phrases", async () => {
  const { result } = renderHook(() =>
    usePersonaLiveVoiceController({
      ...baseArgs,
      wakeTriggerPhrases: [],
      wakeDetectorFactory: () => detector
    })
  )
  await act(async () => result.current.toggleWakeArmed())
  expect(detector.start).not.toHaveBeenCalled()
  expect(result.current.wakeWarning).toMatch(/trigger phrase/i)
})
```

And:

```ts
it("starts the wake detector with raw saved phrases", async () => {
  const { result } = renderHook(() =>
    usePersonaLiveVoiceController({
      ...baseArgs,
      wakeTriggerPhrases: ["hey helper"],
      wakeDetectorFactory: () => detector
    })
  )
  await act(async () => result.current.toggleWakeArmed())
  expect(detector.start).toHaveBeenCalledWith(expect.objectContaining({
    phrases: ["hey helper"]
  }))
})
```

- [ ] **Step 3: Add failing activation/deactivation tests**

Add tests that assert:

- `fireWake()` sends activation metadata only; wake behavior is already carried by `voice_config.voice.wake_behavior`:

```ts
{
  type: "wake_activation",
  session_id: "sess-1",
  matched_phrase: "hey helper",
  detector_kind: "browser_transcript",
  detected_at_ms: 1714500000000
}
```

- `toggleWakeArmed()` from armed to disarmed sends:

```ts
{
  type: "wake_deactivation",
  session_id: "sess-1",
  reason: "disarmed"
}
```

- Unmount sends `wake_deactivation` with `reason: "route_leave"` when armed and websocket is open.

- Starting live mic capture stops the detector to avoid microphone contention.

- [ ] **Step 4: Run failing controller tests**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/hooks/__tests__/usePersonaLiveVoiceController.test.tsx
```

Expected: FAIL on new wake controller tests.

- [ ] **Step 5: Extend controller args and state**

In `apps/packages/ui/src/hooks/usePersonaLiveVoiceController.tsx`, import detector pieces:

```ts
import {
  BrowserTranscriptWakeDetector,
  type WakeDetector,
  type WakeDetectorState,
  type WakeDetectedEvent
} from "@/hooks/personaWakeDetector"
import type { PersonaWakeBehavior } from "@/hooks/useResolvedPersonaVoiceDefaults"
```

Extend args:

```ts
wakeTriggerPhrases?: string[]
wakeDetectorFactory?: () => WakeDetector
```

Add state:

```ts
const [wakeArmed, setWakeArmed] = React.useState(false)
const [wakeDetectorState, setWakeDetectorState] =
  React.useState<WakeDetectorState>("idle")
const [wakeWarning, setWakeWarning] = React.useState<string | null>(null)
const [sessionWakeBehavior, setSessionWakeBehavior] =
  React.useState<PersonaWakeBehavior>(resolvedDefaults.wakeBehavior)
```

Add refs:

```ts
const wakeDetectorRef = React.useRef<WakeDetector | null>(null)
const wakeActiveRef = React.useRef(false)
```

- [ ] **Step 6: Add send helpers**

Add callback helpers:

```ts
const sendWakeActivation = React.useCallback((event: WakeDetectedEvent) => {
  if (!connected || !sessionId || !ws || ws.readyState !== WebSocket.OPEN) {
    setWakeWarning("Wake phrase heard, but Persona Live is not connected.")
    return
  }
  ws.send(JSON.stringify({
    type: "wake_activation",
    session_id: sessionId,
    matched_phrase: event.canonicalPhrase,
    detector_kind: event.detectorKind,
    detected_at_ms: event.detectedAtMs
  }))
}, [connected, sessionId, ws])

const sendWakeDeactivation = React.useCallback((reason: string) => {
  if (!sessionId || !ws || ws.readyState !== WebSocket.OPEN) return
  ws.send(JSON.stringify({
    type: "wake_deactivation",
    session_id: sessionId,
    reason
  }))
}, [sessionId, ws])
```

Wrap sends in `try/catch` consistent with the existing `voice_config` send effect.

- [ ] **Step 7: Add detector start/stop callbacks**

Add:

```ts
const stopWakeListening = React.useCallback(async (reason: "disarmed" | "route_leave" | "stop_live_voice" | "persona_switch" = "disarmed") => {
  wakeActiveRef.current = false
  await wakeDetectorRef.current?.stop()
  wakeDetectorRef.current = null
  setWakeDetectorState("idle")
  setWakeArmed(false)
  sendWakeDeactivation(reason)
}, [sendWakeDeactivation])

const startWakeListening = React.useCallback(async () => {
  const phrases = (wakeTriggerPhrases || []).map((phrase) => String(phrase || "").trim()).filter(Boolean)
  if (phrases.length === 0) {
    setWakeWarning("Add a persona trigger phrase before arming wake listening.")
    return
  }
  const detector = wakeDetectorFactory?.() || new BrowserTranscriptWakeDetector()
  wakeDetectorRef.current = detector
  const available = await detector.isAvailable()
  if (!available) {
    setWakeDetectorState("unavailable")
    setWakeWarning("Wake listening is unavailable in this browser context.")
    return
  }
  setWakeArmed(true)
  setWakeWarning(null)
  await detector.start({
    phrases,
    locale: resolvedDefaults.sttLanguage,
    onStateChange: setWakeDetectorState,
    onError: (error) => setWakeWarning(error.message),
    onWake: (event) => {
      if (wakeActiveRef.current) return
      wakeActiveRef.current = true
      void detector.stop()
      sendWakeActivation(event)
      if (sessionWakeBehavior !== "push_to_talk_after_wake") {
        void startListening()
      }
    }
  })
}, [resolvedDefaults.sttLanguage, sendWakeActivation, sessionWakeBehavior, startListening, wakeDetectorFactory, wakeTriggerPhrases])
```

If dependency order creates a circular callback issue because `startListening` is declared later, move wake callbacks below the listening callbacks.

Add `toggleWakeArmed`:

```ts
const toggleWakeArmed = React.useCallback(() => {
  if (wakeArmed) {
    void stopWakeListening("disarmed")
    return
  }
  void startWakeListening()
}, [startWakeListening, stopWakeListening, wakeArmed])
```

- [ ] **Step 8: Suspend detector during live mic capture**

At the start of `startListening`, before opening the mic stream, stop the detector without disarming the UI:

```ts
if (wakeArmed) {
  await wakeDetectorRef.current?.stop()
  wakeDetectorRef.current = null
  setWakeDetectorState("idle")
}
```

At `stopListening` or `finishVoiceTurn`, decide behavior:

- For `one_shot`: if still `wakeArmed`, set `wakeActiveRef.current = false` and call `startWakeListening()` after the assistant turn fully completes.
- For `continuous`: leave Persona Live in the normal live voice loop until the user stops/disarms.
- For `push_to_talk_after_wake`: do not auto-start live mic capture; keep manual controls available.

Use existing `finishVoiceTurn`, `pendingResumeRef`, `audioFinish`, and TTS/text-only completion points so wake resumes after the turn is actually done, not immediately after commit.

- [ ] **Step 9: Handle server notices**

In `handlePayload`, add notice handling:

```ts
if (reasonCode === "WAKE_ACTIVATION_ACCEPTED") {
  setWakeWarning(null)
  return
}
if (reasonCode === "WAKE_ACTIVATION_REJECTED") {
  wakeActiveRef.current = false
  setWakeWarning(String(payload?.message || "Wake activation was rejected."))
  if (wakeArmed) void startWakeListening()
  return
}
if (reasonCode === "WAKE_DEACTIVATED") {
  wakeActiveRef.current = false
  return
}
```

When `VOICE_TRIGGER_NOT_HEARD` or `VOICE_EMPTY_COMMAND_AFTER_TRIGGER` fires and `wakeArmed` is true, clear `wakeActiveRef` and restart wake listening if behavior is `one_shot`.

- [ ] **Step 10: Cleanup on unmount and session/persona changes**

Extend the existing cleanup effect:

```ts
void stopWakeListening("route_leave")
```

Add an effect watching `personaId` and `sessionId`:

```ts
React.useEffect(() => {
  return () => {
    if (wakeArmed) {
      void stopWakeListening("persona_switch")
    }
  }
}, [personaId, sessionId])
```

Be careful not to send duplicate deactivation on ordinary rerenders; tests should assert at most one deactivation for unmount.

- [ ] **Step 11: Return wake state and actions**

Add to the hook return object:

```ts
wakeArmed,
wakeDetectorState,
wakeWarning,
sessionWakeBehavior,
wakeTriggerPhrases,
toggleWakeArmed,
stopWakeListening,
setSessionWakeBehavior,
```

- [ ] **Step 12: Run controller tests**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/hooks/__tests__/usePersonaLiveVoiceController.test.tsx
```

Expected: PASS.

- [ ] **Step 13: Commit**

Run:

```bash
git add apps/packages/ui/src/hooks/usePersonaLiveVoiceController.tsx apps/packages/ui/src/hooks/__tests__/usePersonaLiveVoiceController.test.tsx
git commit -m "feat(persona): wire wake activation controller"
```

## Task 6: Persona Live UI And Route Wiring

**Files:**
- Modify: `apps/packages/ui/src/components/PersonaGarden/AssistantVoiceCard.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/__tests__/LiveSessionPanel.test.tsx`
- Modify: `apps/packages/ui/src/routes/sidepanel-persona.tsx`
- Modify: `apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx`

- [ ] **Step 1: Add failing voice card tests**

In `apps/packages/ui/src/components/PersonaGarden/__tests__/LiveSessionPanel.test.tsx`, update the default props with wake fields:

```ts
wakeArmed: false,
wakeDetectorState: "idle",
wakeWarning: null,
wakeTriggerPhrases: ["hey helper"],
sessionWakeBehavior: "one_shot",
onToggleWakeArmed: vi.fn(),
onSessionWakeBehaviorChange: vi.fn()
```

Add tests:

```ts
it("renders wake phrase controls and current saved wake phrases", () => {
  render(<AssistantVoiceCard {...defaultVoiceCardProps()} />)
  expect(screen.getByTestId("live-wake-toggle")).toHaveTextContent(/listen for wake phrase/i)
  expect(screen.getByTestId("live-wake-phrases")).toHaveTextContent("hey helper")
})

it("blocks wake toggle display when no saved wake phrases are configured", () => {
  render(<AssistantVoiceCard {...defaultVoiceCardProps()} wakeTriggerPhrases={[]} />)
  expect(screen.getByTestId("live-wake-toggle")).toBeDisabled()
  expect(screen.getByText(/add a trigger phrase/i)).toBeInTheDocument()
})
```

- [ ] **Step 2: Add failing route test**

In `apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx`, add or extend a Persona Live test so the mocked profile response has:

```ts
voice_defaults: {
  voice_chat_trigger_phrases: ["saved wake phrase"],
  wake_behavior: "continuous"
}
```

Set global voice chat trigger phrases in the test to a different fallback if the harness exposes it, then assert the controller/card receives or renders `saved wake phrase`, not the fallback phrase.

- [ ] **Step 3: Run failing UI route tests**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/PersonaGarden/__tests__/LiveSessionPanel.test.tsx ../packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx
```

Expected: FAIL because wake props and route wiring do not exist yet.

- [ ] **Step 4: Add wake props and UI**

In `apps/packages/ui/src/components/PersonaGarden/AssistantVoiceCard.tsx`, import:

```ts
import type { PersonaWakeBehavior } from "@/hooks/useResolvedPersonaVoiceDefaults"
import type { WakeDetectorState } from "@/hooks/personaWakeDetector"
```

Extend props:

```ts
wakeArmed: boolean
wakeDetectorState: WakeDetectorState
wakeWarning: string | null
wakeTriggerPhrases: string[]
sessionWakeBehavior: PersonaWakeBehavior
onToggleWakeArmed: () => void
onSessionWakeBehaviorChange: (next: PersonaWakeBehavior) => void
```

Add a compact wake control section near the top of the card:

```tsx
<div className="mt-3 rounded-md border border-border bg-surface2 p-3 text-xs text-text">
  <div className="flex flex-wrap items-center justify-between gap-2">
    <div>
      <Typography.Text strong>Wake phrase</Typography.Text>
      <Typography.Text type="secondary" className="mt-1 block text-xs">
        Active only while this Persona Live surface is open.
      </Typography.Text>
    </div>
    <Button
      data-testid="live-wake-toggle"
      size="small"
      type={wakeArmed ? "default" : "primary"}
      disabled={sessionControlsDisabled || wakeTriggerPhrases.length === 0}
      onClick={onToggleWakeArmed}
    >
      {wakeArmed ? "Stop wake listening" : "Listen for wake phrase"}
    </Button>
  </div>
  <div className="mt-2 grid gap-2 sm:grid-cols-3">
    <div>
      <div className="text-text-muted">Saved wake phrases</div>
      <div data-testid="live-wake-phrases">
        {wakeTriggerPhrases.length > 0 ? wakeTriggerPhrases.join(", ") : "Add a trigger phrase in voice defaults."}
      </div>
    </div>
    <div>
      <div className="text-text-muted">Wake state</div>
      <div data-testid="live-wake-state">{wakeArmed ? wakeDetectorState : "idle"}</div>
    </div>
    <div>
      <div className="text-text-muted">After wake</div>
      <Select
        data-testid="live-wake-behavior"
        size="small"
        value={sessionWakeBehavior}
        disabled={sessionControlsDisabled}
        onChange={onSessionWakeBehaviorChange}
        options={[
          { value: "one_shot", label: "One turn" },
          { value: "continuous", label: "Continuous" },
          { value: "push_to_talk_after_wake", label: "Push to talk" }
        ]}
      />
    </div>
  </div>
  {wakeWarning ? <Alert className="mt-2" type="warning" showIcon message={wakeWarning} /> : null}
</div>
```

Add `Alert` and `Select` to the existing Ant Design import.

- [ ] **Step 5: Wire raw saved phrases in the route**

In `apps/packages/ui/src/routes/sidepanel-persona.tsx`, add a local normalizer near other small route helpers:

```ts
const normalizeWakeTriggerPhrases = (phrases?: string[] | null): string[] => {
  const seen = new Set<string>()
  const next: string[] = []
  for (const phrase of phrases || []) {
    const trimmed = String(phrase || "").trim()
    if (!trimmed || seen.has(trimmed)) continue
    seen.add(trimmed)
    next.push(trimmed)
  }
  return next
}
```

Add memoized raw saved phrases:

```ts
const wakeTriggerPhrases = React.useMemo(
  () =>
    normalizeWakeTriggerPhrases(
      savedPersonaVoiceDefaults?.voice_chat_trigger_phrases
    ),
  [savedPersonaVoiceDefaults]
)
```

Do not derive wake phrases from `liveSessionVoiceDefaultsBaseline`; that value is a session snapshot and can become stale after profile saves. Wake arming should reflect the selected profile's current saved trigger phrases.

Pass them into the controller:

```ts
const liveVoiceController = usePersonaLiveVoiceController({
  ws,
  connected,
  sessionId,
  personaId,
  resolvedDefaults: resolvedLivePersonaVoiceDefaults,
  canUseServerStt,
  wakeTriggerPhrases
})
```

Pass returned wake values into `AssistantVoiceCard`.

- [ ] **Step 6: Stop wake mode on route/session actions**

Before actions that intentionally change the active live persona session, call:

```ts
void liveVoiceController.stopWakeListening("persona_switch")
```

Use `route_leave` only in hook unmount cleanup. Use `stop_live_voice` when a user action stops live voice but does not leave the route.

- [ ] **Step 7: Run UI route tests**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/components/PersonaGarden/__tests__/LiveSessionPanel.test.tsx ../packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
git add apps/packages/ui/src/components/PersonaGarden/AssistantVoiceCard.tsx apps/packages/ui/src/components/PersonaGarden/__tests__/LiveSessionPanel.test.tsx apps/packages/ui/src/routes/sidepanel-persona.tsx apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx
git commit -m "feat(persona): add wake controls to live voice"
```

## Task 7: Documentation And Final Verification

**Files:**
- Modify if clean: `Docs/Design/Personas.md`
- Or create: `Docs/Design/Persona_Wake_Word_Support.md`
- Verify touched files from Tasks 1 through 6.

- [ ] **Step 1: Check documentation working tree state**

Run:

```bash
git status --short Docs/Design/Personas.md
```

Expected: If this shows unrelated user changes, do not edit `Docs/Design/Personas.md`.

- [ ] **Step 2: Add V1 limitations documentation**

If `Docs/Design/Personas.md` is clean, add a short `Wake Phrase Support` section there.

If it is dirty with unrelated changes, create `Docs/Design/Persona_Wake_Word_Support.md` containing:

```md
# Persona Wake Phrase Support

Persona Live wake phrase support is manually armed per session. V1 uses the selected persona profile's saved `voice_chat_trigger_phrases` as wake phrases and detects them in the active browser or extension surface before sending a scoped `wake_activation` frame to `/api/v1/persona/stream`.

V1 is not true background always-on listening. Wake listening is visible, manually armed, and only active while the Persona Live surface is open and the browser speech recognition API is available. Pre-wake audio is not sent to the tldw server.
```

- [ ] **Step 3: Run backend focused tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_profiles_api.py tldw_Server_API/tests/Persona/test_persona_ws.py -q -k "voice_defaults or wake_activation or wake_deactivation or persona_audio_chunk_vad_auto_commit_ignores_missing_trigger_phrase"
```

Expected: PASS.

- [ ] **Step 4: Run frontend focused tests**

Run:

```bash
cd apps/tldw-frontend && bunx vitest run ../packages/ui/src/hooks/__tests__/useResolvedPersonaVoiceDefaults.test.tsx ../packages/ui/src/hooks/__tests__/personaWakeDetector.test.ts ../packages/ui/src/hooks/__tests__/usePersonaLiveVoiceController.test.tsx ../packages/ui/src/components/PersonaGarden/__tests__/AssistantDefaultsPanel.test.tsx ../packages/ui/src/components/PersonaGarden/__tests__/LiveSessionPanel.test.tsx ../packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Compile touched backend files**

Run:

```bash
source .venv/bin/activate && python -m py_compile tldw_Server_API/app/api/v1/schemas/persona.py tldw_Server_API/app/api/v1/endpoints/persona.py
```

Expected: no output and exit code 0.

- [ ] **Step 6: Run Bandit on touched backend files**

Run:

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/schemas/persona.py tldw_Server_API/app/api/v1/endpoints/persona.py -f json -o /tmp/bandit_persona_wake_word.json
```

Expected: exit code 0 or only pre-existing findings outside changed lines. Inspect `/tmp/bandit_persona_wake_word.json` before proceeding.

- [ ] **Step 7: Run whitespace diff check**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 8: Commit docs and final verification fixes**

Run:

```bash
git add Docs/Design/Persona_Wake_Word_Support.md Docs/Design/Personas.md
git commit -m "docs(persona): document wake phrase limitations"
```

If only one docs file exists or changed, stage only that file. Do not stage unrelated pre-existing edits.

## Final Review Checklist

- [ ] `wake_behavior` is saved as nullable profile data and resolves to `one_shot` in the frontend.
- [ ] Wake arming uses raw saved selected-profile trigger phrases, not global fallback phrases.
- [ ] Browser detector is unavailable-safe and can be replaced by a future native companion.
- [ ] `wake_activation` validates both saved profile phrases and current session runtime phrases.
- [ ] Trigger phrase gating remains unchanged outside active wake windows.
- [ ] One-shot and push-to-talk wake activations expire after a turn or no-command timeout.
- [ ] Continuous wake activation clears on deactivation, session close, persona switch, or route leave.
- [ ] Frontend stops detector before Persona Live mic capture.
- [ ] UI states are visible for idle, listening, detected, unavailable, and error.
- [ ] No pre-wake transcript is persisted or sent as a persona turn.
- [ ] Focused tests, backend compile, Bandit, and `git diff --check` have been run.

## Execution Notes

- Use `superpowers:test-driven-development` for each task.
- Use `superpowers:systematic-debugging` if a test fails unexpectedly.
- Use `superpowers:verification-before-completion` before claiming implementation complete.
- Use `superpowers:requesting-code-review` before merging or opening a PR.
- Keep commits small and in the order listed above.
- Do not use `--no-verify`.
- Do not stage unrelated changes in `Docs/Design/Personas.md`.
