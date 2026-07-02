---
id: TASK-12109
title: Harden real-time voice (barge-in, backpressure, handshake timeout, WS unmount leaks)
status: To Do
labels:
- bug
- medium
- audio
- voice
- packages-ui
documentation:
- apps/FRONTEND_AUDIT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Severity: Medium (user-visible + leaks).** Round-2 audit finding R8. Paths under apps/packages/ui/src/.

- **Barge-in doesn't stop TTS** — `hooks/useVoiceChatStream.tsx:382-404` handles server `interrupted` but never calls `audioStop()` (only referenced in `cleanupSession`), so already-buffered assistant audio keeps playing while the user speaks. Also floods `{type:"interrupt"}` on every audio chunk (~4/sec) with no once-guard (`:221-223`).
- **No backpressure** — `:216-229` `ws.send()`s each mic PCM frame without checking `ws.bufferedAmount` (checked nowhere in the codebase); a slow uplink grows the buffer unbounded until the browser force-closes the socket.
- **No handshake timeout** — `:476-604` if `onopen` never fires, `connectingRef` stays true and the restart guard blocks recovery → UI wedged in "connecting". (Contrast `background.ts:3335`, which has a 10s connect timer.)
- **WS leak on unmount-mid-connect** — `usePersonaLiveSession.tsx:412-591` and `usePersonaLiveControl.tsx:264-269` create the WebSocket *after* a multi-await handshake with no attempt-id/mounted guard, so an unmount during the awaits runs cleanup (closing the then-null socket) and the resumed `connect()` creates a fresh socket nothing ever closes.
- Low: error state clobbered by `onclose`→idle (`useVoiceChatStream.tsx:491-510`); onerror without close leaves a half-broken "connected" socket; unmount close without detaching listeners.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Barge-in / server `interrupted` calls `audioStop()` so buffered TTS stops immediately; the `interrupt` message is sent once per barge-in, not per chunk.
- [ ] #2 Mic→WS send checks `ws.bufferedAmount` and drops/coalesces frames under backpressure.
- [ ] #3 `useVoiceChatStream` and `usePersonaLiveControl.ensureStreamSocket` add a connection/handshake timeout that clears the connecting state and allows restart.
- [ ] #4 `usePersonaLiveSession.connect` and `ensureStreamSocket` add an attempt-id/mounted guard after their awaits so an unmount can't leak an open WebSocket.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
