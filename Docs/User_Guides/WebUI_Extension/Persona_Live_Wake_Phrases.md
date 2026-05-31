# Persona Live Wake Phrase Guide

Persona Live can listen for a saved trigger phrase before accepting a spoken turn. This is useful when you want a visible, session-scoped voice workflow without keeping the live mic capture open for every utterance.

Wake phrase support is not background always-on listening. It works only while the Persona Garden Live Session surface is open, connected, and manually armed.

## Requirements

- A persona profile with at least one saved trigger phrase.
- A connected Persona Garden Live Session.
- A browser or extension surface that exposes `SpeechRecognition` or `webkitSpeechRecognition`.
- Microphone permission for the current browser/site context.
- Working Persona Live voice configuration for the command after wake detection.

Saved trigger phrases are stored on the persona profile as `voice_defaults.voice_chat_trigger_phrases`. Wake behavior is stored as `voice_defaults.wake_behavior`; if no behavior is saved, Persona Live uses one turn after wake.

## Setup

1. Open `Persona Garden`.
2. Select the persona you want to use.
3. In the persona assistant voice defaults, add one trigger phrase per line under `Trigger phrases`.
4. Choose `Wake behavior`:
   - `One turn after wake`: the next spoken turn is accepted, then wake gating resumes.
   - `Continuous after wake`: spoken turns continue after the first accepted wake phrase until wake mode is stopped or the session changes.
   - `Push to talk after wake`: wake activation allows push-to-talk style live turns after detection.
5. Save the persona defaults.
6. Open `Live Session` for that persona and connect Persona Live.
7. In the `Wake phrase` panel, click `Listen for wake phrase`.
8. Say one of the saved phrases, then speak the command you want the persona to handle.

The Live Session panel shows saved wake phrases, detector state, and the active after-wake behavior. Session behavior can be changed for the live session without changing the saved persona default.

## Privacy and Safety

- Pre-wake audio is not streamed to the tldw server by this feature.
- The browser-side detector listens locally for transcript matches before sending a scoped `wake_activation` frame to `/api/v1/persona/stream`.
- Browser speech recognition support varies. Some browsers may use browser-managed speech recognition services and may not run fully offline.
- Wake activation does not bypass persona policy, MCP/tool permissions, or approval settings. Existing confirmation and access controls still apply.
- Wake listening stops when the live session disconnects, the persona changes, the route changes, or the user stops wake listening.

## Troubleshooting

### The wake button is disabled

Add at least one saved trigger phrase in the selected persona's assistant voice defaults. Persona Live does not arm wake listening from global fallback trigger phrases.

### The detector is unavailable

Use a browser context that supports `SpeechRecognition` or `webkitSpeechRecognition`. Also check that the page is running in a secure or otherwise browser-approved context and that the extension/page has microphone access.

### Permission was denied

Grant microphone permission for the site or extension, then arm wake listening again. Some browsers require a direct user action before speech recognition can start.

### The phrase is heard, but no command runs

Confirm that Persona Live is connected and still on the same persona. Use the exact saved trigger phrase, try a shorter phrase, and leave a small pause before the command. If the phrase exists only in runtime or fallback settings but not in the selected persona profile, the server rejects the wake activation.

### Wake listening stops after switching tabs or personas

That is expected. Wake listening is scoped to the active Persona Live session and is intentionally stopped on tab, route, persona, and disconnect changes.

### The wrong after-wake behavior seems active

Check both places:
- The saved `Wake behavior` in assistant voice defaults controls the default for future sessions.
- The `After wake` selector in the Live Session panel controls the current session.

Reconnect the live session if you want to reload saved defaults into a fresh session.
