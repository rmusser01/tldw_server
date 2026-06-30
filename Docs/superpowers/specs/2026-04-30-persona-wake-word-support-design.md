# Persona Wake Word Support Design

Date: 2026-04-30
Status: Approved for planning
Owner: Codex brainstorming pass

## Summary

Add wake-word support to Persona Live without creating a parallel persona runtime.

V1 wake mode is manually armed from an active Persona Live surface. A client-side `WakeDetector` listens for the selected persona's existing `voice_chat_trigger_phrases`, then activates the existing `/api/v1/persona/stream` live voice flow. The first detector implementation uses browser-side transcript matching where the browser/device supports it. A later native/local companion can implement the same wake activation contract without changing persona profile semantics or the persona websocket turn executor.

## Goals

- Let users arm wake listening for the current Persona Live session.
- Reuse existing persona trigger phrases as wake phrases in V1.
- Let each persona define the default post-wake behavior.
- Keep actual persona turns on the existing persona websocket path.
- Keep V1 honest about browser and extension lifecycle limits.
- Preserve a clean extension point for a future native/local companion.
- Avoid automatic background listening in V1.

## Non-Goals

- Do not add server-side always-on audio streaming for wake detection.
- Do not add a separate wake phrase model in V1.
- Do not create a second persona or assistant runtime.
- Do not make wake listening start automatically when the app or extension starts.
- Do not solve native packaging, platform startup, or OS-level microphone permissions in V1.
- Do not make generic `/api/v1/audio/*` voice chat a persona wake runtime.

## Existing Context

The current backend already has a persona live voice path:

- `/api/v1/persona/stream` accepts persona websocket traffic.
- `voice_config` stores session-scoped live voice runtime preferences.
- `audio_chunk` feeds persona live STT and VAD.
- `voice_commit` and VAD auto-commit route spoken turns through the persona turn flow.
- Server-side trigger phrase gating strips configured trigger phrases before creating a persona turn.
- Persona turns use existing policy, scope, memory, tool planning, assistant deltas, TTS, and analytics.

The current persona profile schema already includes `voice_chat_trigger_phrases` under persona voice defaults. Those phrases should be reused as wake phrases in V1. Wake arming should use the selected persona profile's saved trigger phrases only; it should not invent global fallback wake phrases.

## Key Design Decisions

### 1. Wake Detection Is Client/Device Side

V1 wake detection happens in the active frontend or extension surface, not on the server.

This means the backend does not receive continuous pre-wake microphone audio. The frontend only opens or activates the normal persona voice flow after local wake detection.

### 2. Manual Session Arming Only

Wake listening starts only after the user explicitly arms it in Persona Live.

Disarming, leaving the route, closing the session, switching personas, losing microphone permission, or auth failure must stop wake listening. V1 must never auto-arm based only on saved persona defaults.

### 3. Reuse `voice_chat_trigger_phrases`

V1 treats existing persona trigger phrases as wake phrases.

This keeps the mental model simple:

- The same phrase wakes the persona locally.
- The same phrase remains the server-side guard for non-wake voice commits.

Separate wake phrases can be added later if the native companion needs richer routing, but V1 should avoid that extra model.

### 4. Add A WakeDetector Interface

The frontend should define a detector abstraction instead of binding Persona Live directly to one browser API.

The first implementation is `BrowserTranscriptWakeDetector`, which listens in the active browser context and matches normalized transcript text against the configured trigger phrases.

Future implementations can include:

- a native/local companion bridge
- a dedicated wake-word model
- an extension-specific offscreen-document detector where supported

### 5. Persona Profiles Own Default Wake Behavior

Each persona should store a default post-wake behavior under voice defaults:

- `one_shot`
- `continuous`
- `push_to_talk_after_wake`

The current live session can override the behavior temporarily, but that override should not persist unless the user explicitly saves it in profile settings.

## Data Model

Extend `PersonaVoiceDefaults` with:

```python
wake_behavior: Literal["one_shot", "continuous", "push_to_talk_after_wake"] | None
```

Default resolution should be:

1. explicit persona `voice_defaults.wake_behavior`
2. fixed fallback of `one_shot`

The field is nullable so existing profiles remain valid and so unsaved profiles can inherit the product default.

Do not add separate `wake_phrases` in V1. Continue to use `voice_chat_trigger_phrases`.

Wake arming must use the raw saved persona profile phrases, not the broader resolved Persona Live defaults. Existing non-wake Persona Live behavior may continue to use fallback trigger phrases if that is already supported, but wake mode must not arm from global voice-chat fallback phrases.

## Frontend Components

### WakeDetector Interface

The interface should be small and implementation-neutral:

```ts
type WakeDetectorState =
  | "idle"
  | "starting"
  | "listening"
  | "detected"
  | "unavailable"
  | "error";

interface WakeDetector {
  isAvailable(): Promise<boolean>;
  start(config: WakeDetectorConfig): Promise<void>;
  stop(): Promise<void>;
}

interface WakeDetectorConfig {
  phrases: string[];
  locale?: string;
  onWake: (event: WakeDetectedEvent) => void;
  onStateChange?: (state: WakeDetectorState) => void;
  onError?: (error: WakeDetectorError) => void;
}
```

`WakeDetectedEvent` should include the canonical configured phrase that matched, the heard transcript if available, a timestamp, and the detector kind. The heard transcript is diagnostic in V1; it should not directly become a persona turn.

### BrowserTranscriptWakeDetector

The V1 detector should:

- normalize phrase and transcript text before matching
- ignore empty phrase lists
- debounce repeated wake events while a turn is active
- expose unavailable/error states without breaking normal Persona Live
- stop cleanly on disarm or route/session teardown

The implementation can use browser speech recognition where available. The UI must describe it as browser/device-side wake listening, not guaranteed offline recognition, because browser speech APIs vary by platform and may use browser-managed services.

### Persona Live Controller

`usePersonaLiveVoiceController` should own:

- `armedForWake`
- detector availability
- detector state
- wake behavior for the current session
- wake activation handoff into the existing live voice flow

The controller should not create a second persona session manager. It should continue to use the existing persona websocket session for actual turns.

Wake mode should pass the controller a separate `wakeTriggerPhrases` value derived from `profile.voice_defaults.voice_chat_trigger_phrases`. Do not derive wake arming from `resolvedDefaults.voiceChatTriggerPhrases`, because that value may include global fallback phrases used by ordinary live voice.

### Persona Live UI

The Live UI should add:

- an explicit `Listen for wake phrase` arm/disarm control
- visible armed/listening/detected/error state
- current wake phrases
- a blocked state when no trigger phrases are configured
- a session-local wake behavior selector
- a clear indication that V1 wake listening works only while the Persona Live surface is active

Profile settings should add the persistent default wake behavior near existing voice defaults.

## Runtime Flow

### Arming

1. User opens Persona Live and selects a persona.
2. UI resolves live voice defaults and separately reads saved persona wake trigger phrases.
3. User toggles `Listen for wake phrase`.
4. If there are no trigger phrases, arming is blocked.
5. If the detector is unavailable, UI shows a recoverable unavailable state.
6. If available, the detector starts and the UI shows an armed/listening state.

### Wake Detection

1. Detector hears a transcript locally.
2. Detector matches one configured trigger phrase.
3. Controller debounces duplicate detections.
4. Controller ensures the persona websocket session is connected.
5. Controller sends or refreshes `voice_config` with the saved persona wake trigger phrases for this wake-armed session.
6. Controller sends a `wake_activation` frame with the canonical matched phrase, detector kind, and timestamp.
7. Server derives the effective wake behavior from the current session `voice_runtime.wake_behavior`, which was sent through `voice_config`.
8. Controller applies the current session-local wake behavior for UI flow.

### Wake Activation Frame

Add a small persona websocket frame for V1:

```json
{
  "type": "wake_activation",
  "session_id": "<persona-session-id>",
  "matched_phrase": "hey helper",
  "detector_kind": "browser_transcript",
  "detected_at_ms": 1714500000000
}
```

The client must not be treated as authoritative for wake behavior in this frame. The selected behavior is sent through `voice_config.voice.wake_behavior`; the server normalizes that runtime value and stores it with the accepted wake activation as session-local runtime state. It is not a persisted memory, not a command, and not an authorization grant.

The server should accept `wake_activation` only when all of these are true:

- `session_id` normalizes to an existing session for the authenticated user.
- The session has current `voice_runtime.trigger_phrases` from a prior `voice_config`.
- The server can derive `voice_runtime.wake_behavior` as one of `one_shot`, `continuous`, or `push_to_talk_after_wake`, falling back to `one_shot` when unset.
- The server can resolve the session's persona profile for the authenticated user.
- `matched_phrase` is non-empty.
- `matched_phrase` is the canonical configured phrase after detector normalization, not the raw recognized text.
- `matched_phrase` matches one of the saved persona profile `voice_defaults.voice_chat_trigger_phrases`.
- `matched_phrase` is also present in the current session `voice_runtime.trigger_phrases`, so the active runtime state and saved persona wake phrases agree.

If validation fails, the server should emit a non-fatal `WAKE_ACTIVATION_REJECTED` notice and leave trigger phrase gating unchanged.

While a valid wake activation is active, the next committed voice turn does not need to contain the trigger phrase. This is required because pre-wake audio stays local and the post-wake server transcript normally starts after the wake phrase. Outside an active wake activation, existing server-side trigger phrase gating still applies.

For `one_shot` and `push_to_talk_after_wake`, the wake activation expires after the next successful committed turn, after a short no-command timeout, or when the user disarms/stops the session. The no-command timeout should default to 30 seconds and be configurable. For `continuous`, the wake activation remains active until the user stops live voice, disarms wake mode, switches personas, or the session closes.

The local detector transcript remains diagnostic. V1 captures the user command after wake activation through the existing persona live STT/VAD or manual send flow, so users may need a short post-wake pause or cue before speaking the command.

### Wake Deactivation Frame

Add a matching explicit deactivation frame:

```json
{
  "type": "wake_deactivation",
  "session_id": "<persona-session-id>",
  "reason": "disarmed"
}
```

Allowed reasons should include `disarmed`, `stop_live_voice`, `persona_switch`, `route_leave`, and `session_close`.

The frontend should send `wake_deactivation` before disarming, stopping live voice, switching personas, or leaving the route when the websocket is still open. The server should also clear wake activation state during websocket cleanup, even if no deactivation frame arrives.

### Post-Wake Behaviors

`one_shot`:

- Capture the post-wake utterance.
- Send one turn through the existing persona voice path.
- Return to wake listening after the assistant turn completes if the session remains armed.

`continuous`:

- Enter the normal Persona Live voice loop after wake.
- Keep listening and auto-committing until the user stops live voice, disarms wake mode, or the session ends.
- Send `wake_deactivation` before returning to armed-idle or closed state while the websocket remains open.

`push_to_talk_after_wake`:

- Wake opens or focuses the session.
- UI exposes manual listen/send controls.
- Continuous microphone capture does not start automatically after wake.

## Backend Responsibilities

The backend should remain additive:

- Validate and persist `voice_defaults.wake_behavior`.
- Return the saved nullable `voice_defaults.wake_behavior` in persona profile responses; frontend resolution applies the `one_shot` fallback.
- Accept `wake_activation` websocket frames and store active wake state in session-local runtime preferences using server-derived `voice_runtime.wake_behavior`.
- Accept `wake_deactivation` websocket frames and clear active wake state for the session.
- Validate wake activation phrases against the saved persona profile, not only client-supplied `voice_config`.
- Keep existing server-side trigger phrase gating on committed transcripts when no wake activation is active.
- Let active wake state bypass trigger phrase gating only for the scoped post-wake behavior window.
- Keep persona policy, scope, tool planning, memory, and TTS behavior unchanged.

The server should not authorize actions based on `wake_behavior` or `wake_activation`. They are UI/runtime behavior settings, not permission grants.

## Native Companion Compatibility

The future companion should call the same conceptual activation contract:

- persona id
- session id or request to create/focus a session
- matched phrase
- detector kind
- timestamp
- desired wake behavior

The frontend can initially handle this through an internal controller method that sends the persona websocket `wake_activation` frame. If a companion is added later, that method can be exposed through a local bridge without changing the persona profile data model.

## Error Handling

- No trigger phrases: block arming and provide a setup path to voice defaults.
- Detector unavailable: show wake listening unavailable; normal Persona Live remains usable.
- Microphone permission denied: stop detector and show a permission error.
- Wake detected while a turn is active: ignore or debounce the duplicate wake.
- Websocket connect failure after wake: show a recoverable connection error and return to armed-idle or disarmed state based on controller policy.
- Server rejects the committed transcript because no trigger phrase remains and no wake activation is active: show the existing ignored/no-command notice and return to wake listening if still armed.
- Server rejects `wake_activation`: keep normal trigger phrase gating, show a recoverable wake activation error, and return to armed wake listening if still armed.
- Wake activation expires before the post-wake utterance is committed: return to armed wake listening and require a fresh wake phrase.
- Persona switch: stop detector before changing active persona/session state.
- Route leave or session close: stop detector and release microphone resources.

## Privacy And Security

V1 must be explicit that wake listening is manually armed and active only while the Persona Live surface is active.

Privacy requirements:

- No automatic arming.
- No pre-wake audio sent to the tldw server.
- Visible armed/listening state.
- Stop detector on teardown and permission loss.
- Do not persist raw wake transcripts unless they become normal persona turns after the persona voice turn path accepts them.

Security requirements:

- Existing persona websocket auth remains authoritative.
- Server-side trigger phrase gating remains a guard outside active wake windows.
- `wake_behavior` and `wake_activation` must not bypass persona policy or scope rules.
- The native companion design must not embed long-lived secrets in detector events.

## Testing Strategy

Backend:

- `PersonaVoiceDefaults` accepts valid `wake_behavior` values.
- invalid `wake_behavior` values are rejected.
- profile create/update/read preserves `wake_behavior`.
- existing profiles without `wake_behavior` still validate.
- websocket `wake_activation` allows the next post-wake voice turn to omit the trigger phrase.
- invalid `wake_activation` frames are rejected without disabling trigger phrase gating.
- `wake_activation` is rejected when `matched_phrase` exists only in client-supplied runtime config and not in saved persona profile trigger phrases.
- client-supplied `wake_behavior` in `wake_activation`, if present, is ignored in favor of the server-side runtime value.
- websocket `wake_deactivation` clears continuous wake activation while the socket stays open.
- websocket trigger phrase gating still rejects transcripts without trigger phrases when no wake activation is active.
- `one_shot` wake activation expires after one committed voice turn.
- `continuous` wake activation expires on explicit stop/disarm/session close.
- persona websocket behavior remains unchanged for typed and spoken turns.

Frontend:

- `BrowserTranscriptWakeDetector` normalizes and matches configured phrases.
- detector reports unavailable when the browser API is absent.
- arming is blocked when no trigger phrases are configured.
- disarm stops the detector.
- persona switch stops the detector and reloads phrases.
- duplicate wake events are debounced while a turn is active.
- wake activation sends a `wake_activation` frame before post-wake capture starts.
- wake listening suspends during post-wake Persona Live microphone capture to avoid microphone contention and duplicate wake events.
- wake deactivation sends a `wake_deactivation` frame before continuous active mode is stopped while the socket remains open.
- `one_shot` returns to armed wake listening after completion.
- `continuous` enters the live voice loop after wake.
- `push_to_talk_after_wake` focuses the session without starting continuous capture.
- profile settings persist the default wake behavior.

E2E or smoke:

- active Persona Live session can be armed.
- wake phrase detection activates a persona turn.
- browser unsupported state degrades cleanly.
- route leave stops armed listening.

## Rollout Plan

1. Add backend schema and persistence for `wake_behavior`.
2. Add the frontend `WakeDetector` interface and detector unit tests.
3. Implement `BrowserTranscriptWakeDetector`.
4. Add Persona Live manual arm/disarm UI and blocked/unavailable states.
5. Wire wake activation into the existing persona live voice controller.
6. Add profile editor support for default wake behavior.
7. Add focused backend, frontend, and smoke coverage.
8. Document V1 limitations and the future native companion integration point.

## Open Implementation Notes

- The implementation plan should inspect current frontend file names before assigning exact paths, because the repository has both current app routes and historical design docs for Persona Garden.
- The first shipping detector should be feature-detected at runtime rather than assumed available.
- The UI should avoid calling V1 "true background always-on" because browser and extension lifecycles cannot guarantee that behavior.
