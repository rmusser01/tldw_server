---
id: TASK-12101
title: Fix default request timeouts aborting normal LLM generations and TTS
status: Done
labels:
- bug
- high
- chat
- frontend
- packages-ui
documentation:
- apps/FRONTEND_AUDIT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Severity: High (normal generations fail as "Network error").** From the 2026-07-02 frontend audit (finding H12). Paths relative to `apps/packages/ui/src/`.

- `services/tldw/request-core.ts:95-100` — `deriveRequestTimeout` defaults `/api/v1/chat/completions` to a **10s total** timeout unless the user set `chatRequestTimeoutMs`. LLM generations routinely exceed 10s, so unconfigured non-stream chat aborts mid-generation and surfaces as `status: 0` "Network error" (inconsistent with the 45s stream-idle default).
- `services/background-proxy.ts:31-32,271-281,696-708` — `resolveRuntimeMessageTimeoutMs` returns 10s when `timeoutMs` is unset (`Number(undefined) → NaN`). The MV3 worker only replies after the whole fetch completes, so any write > 10s (non-stream chat, media-processing kickoff, character export) throws "Extension messaging timeout" while the server keeps running and the result is lost. Same in `bgUpload` (`:1406-1420`).
- `TldwApiClient.ts:6478` `synthesizeSpeech` passes no `timeoutMs` → TTS inherits the 10s default; synthesizing more than a short paragraph on local Kokoro aborts while the server renders.
- Related (fold in): request timeout only covers headers (`request-core.ts:443-468`); the subsequent body read is unbounded, so a server that sends headers then stalls yields an infinite spinner despite the "timeout".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Generation endpoints (non-stream chat completions, TTS, media-processing kickoff, exports) use a timeout appropriate for long-running work, not the 10s default.
- [ ] #2 The extension messaging-ack timeout is decoupled from the request completion time (an in-flight long request is not killed at 10s by `Number(undefined)`).
- [ ] #3 `synthesizeSpeech` (and other long POST wrappers) pass an appropriate `timeoutMs`.
- [ ] #4 The request timeout covers the body read (not just headers), so a stalled body doesn't hang forever.
- [ ] #5 A test asserts a >10s non-stream chat/TTS completes instead of aborting.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
