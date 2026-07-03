---
id: TASK-12107
title: Fix TTS playback resource leaks and overlapping playback
status: Done
labels:
- bug
- medium
- audio
- tts
- packages-ui
documentation:
- apps/FRONTEND_AUDIT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Severity: Medium (steady memory leaks + audible double-playback).** Round-2 audit findings R6 + R7. Paths under apps/packages/ui/src/.

1. **MediaSource blob URL leak on stream→buffered fallback** — `hooks/useStreamingAudioPlayer.tsx:273` sets `objectUrlRef.current = url` with a fresh blob, overwriting the MediaSource blob URL stored at `:204` without revoking it, so `cleanup()`/`revokeObjectUrl()` can never free it. Every failed stream (onerror/appendBuffer throw/play reject) leaks one blob URL + MediaSource.
2. **Blob URL leak + orphaned generator on cancel-during-playback** — `hooks/useTTS.tsx:283-330`: `cancel()` calls `audioElement.pause()`, which fires neither `onended` nor `onerror`, so `playAudio()`'s promise never settles, `await Promise.all` never resolves, and the `finally { URL.revokeObjectURL(url) }` never runs. Stop/unmount mid-playback leaks the segment URL and silently stops remaining sentences.
3. **Overlapping playback** — `components/Sidepanel/Chat/TtsClipsDrawer.tsx:128-160`: `handleTogglePlay` only stops when `playingClipId === clip.id`; for a *different* clip it creates a new AbortController and overwrites `audioRef` without aborting/stopping the previous one → two clips play at once. Verified by read.
4. **Audiobook cancel doesn't abort synthesis** — `hooks/useAudiobookGeneration.tsx:90` calls `context.synthesize(...)` with no `{ signal }`; the abort is only polled between chapters, so Cancel doesn't stop the in-flight chapter.

Lower: play/pause `AbortError` races and undetached audio listeners in `hooks/document-workspace/useDocumentTTS.ts`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `useStreamingAudioPlayer` revokes the previous object URL before overwriting `objectUrlRef` (no leak on the streaming→buffered fallback path).
- [ ] #2 `useTTS.cancel()` settles the in-flight `playAudio` promise (or revokes the URL directly) so the object URL is always freed and the generator isn't orphaned.
- [ ] #3 `TtsClipsDrawer.handleTogglePlay` stops/aborts any currently-playing clip before starting a different one (no overlapping audio).
- [ ] #4 Audiobook `cancelGeneration` aborts the in-flight chapter synthesis (thread the abort signal into `synthesize`).
- [ ] #5 Tests/assertions where practical: cancel during playback frees the URL; starting a second clip stops the first.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
