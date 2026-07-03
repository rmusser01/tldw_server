---
id: TASK-12112
title: Fix microphone left live after recorder error or double-start
status: Done
labels:
- bug
- high
- privacy
- audio
- packages-ui
documentation:
- apps/FRONTEND_AUDIT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Severity: High (privacy — mic indicator stuck on).** Round-2 audit finding R2.

`hooks/useAudioRecorder.ts:110-130` sets `ondataavailable` and `onstop` but **no `onerror`**. `stopMediaTracks()` runs only in `onstop`, so a `MediaRecorder` error that doesn't also fire `stop` (device unplugged, hardware glitch) leaves the `getUserMedia` track live — the browser mic/recording indicator stays on indefinitely — and `captureOwnerRef` is never released, blocking all future capture. Both sibling hooks set `onerror`; this one is the outlier. Verified by direct read.

Related mic-leak paths (same class):
- `useAudioRecorder.ts:100-108` — `startRecording` has no synchronous re-entry guard, so a double-clicked record button runs a second `getUserMedia` and orphans the first stream.
- `hooks/useServerDictation.tsx:302-331` — the acquired `stream` is block-scoped inside `try`, so a `MediaRecorder` ctor/`start()` throw can't stop it; plus the same double-click race (`:114-120`).
- `components/Option/Speech/SpeechPlaygroundPage.tsx:592-594` — `recordingStreamRef` is assigned *after* `new MediaRecorder(stream)`, so a ctor throw leaks the mic; same double-start race.

`hooks/useMicStream.ts` is the correct template: synchronous `startingRef` re-entry guard, stream held in a ref reachable from `catch`, tracks stopped on every path (normal/error/unmount/race), AudioContext closed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `useAudioRecorder` sets a `MediaRecorder` `onerror` handler that stops all tracks, releases the capture owner, and resets status.
- [ ] #2 All four mic-capture sites (`useAudioRecorder`, `useServerDictation`, `SpeechPlaygroundPage`, and any other `getUserMedia`) hold the acquired `MediaStream` in a ref reachable from a synchronous `catch` and stop its tracks on the error path.
- [ ] #3 Re-entry is guarded synchronously (a ref, not lagging React state) so a rapid double-start cannot orphan a stream; the record/dictation buttons are disabled while a start is in flight.
- [ ] #4 Tests: simulate a `MediaRecorder` error → assert tracks stopped + owner released; simulate a double-start → assert only one active stream and the first is stopped.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
