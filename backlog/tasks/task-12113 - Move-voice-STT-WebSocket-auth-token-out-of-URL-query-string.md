---
id: TASK-12113
title: Move voice/STT WebSocket auth token out of the URL query string
status: In Progress
updated_date: 2026-07-05 00:26
labels:
- bug
- high
- security
- audio
- packages-ui
- needs-server-testing
documentation:
- apps/FRONTEND_AUDIT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Severity: High (credential log exposure).** Round-2 audit finding R3.

The webui voice/STT WebSocket clients put the auth credential directly in the connection URL query string, so it lands in server access logs, reverse-proxy logs, and any WS-URL telemetry (replayable by anyone with log access):
- `services/persona-stream.ts:20,26` — `?token=<jwt>` (multi-user) / `?api_key=<key>` (single-user, long-lived key). Verified by read.
- `services/tldw/voice-conversation.ts:361` — `/api/v1/audio/chat/stream?token=...`.
- `entries/background.ts:3332` — `/api/v1/audio/stream/transcribe?token=...` (extension worker).

**The backend already supports better methods** (so this is genuinely fixable client-side):
- `persona.py:3688-3712` `_extract_auth_credentials` reads the token from the `Authorization` header and from the WebSocket **subprotocol**; the query `token`/`api_key` are only a fallback.
- `audio_streaming.py:866,874-881` — query-token auth is **legacy and disabled by default** (`AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH`); the endpoint supports an `{type:"auth"}` **first message** after connect.

The client currently authenticates **only** via the URL token (no subprotocol 2nd arg to `new WebSocket`, no `{type:"auth"}` send — verified). So the fix is a client auth-method switch.

**Why this is ticketed, not auto-fixed:** switching the client to subprotocol/first-message auth and removing the URL token will break ALL voice/STT if the exact format or timing is wrong (browser subprotocol character constraints; auth-before-config ordering), and it must be validated against a running server — which wasn't possible in the audit-remediation pass. Treat like a change that needs a live smoke test before merge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Persona WS auth uses the `Sec-WebSocket-Protocol` subprotocol (or a first-message auth) matching `persona.py:3688-3712`; the token is no longer in the URL.
- [x] #2 Audio STT/voice WS auth uses the `{type:"auth"}` first message matching `audio_streaming.py`; the token is no longer in the URL. Extension STT (`background.ts:3332`) updated the same way.
- [x] #3 Auth is sent before any config/audio message so the server authenticates the connection first.
- [ ] #4 Validated against a running backend: persona live, voice chat, and streaming transcription all connect and authenticate with no token in the connection URL (checked in browser devtools / server logs).
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-04 current repo-audit slice started on branch codex/audit-audio-ws-auth-contract-2026-07-04 from origin/dev fd5c152b065c408e4e8ee5f08da41589f21cb7f5. Scope for this branch: AUDIT-2026-06-27-WEBUI-002 and AUDIT-2026-06-27-APIWEB-001 audio TTS/STT/voice-chat WebSocket query-token drift. Implementation plan: Docs/superpowers/plans/2026-07-04-audio-websocket-auth-contract-plan.md. Existing TASK-12113 also mentions persona WebSocket auth and live backend smoke; those remain tracked residuals unless this branch explicitly validates them.

Audio WebSocket remediation validation on branch codex/audit-audio-ws-auth-contract-2026-07-04: focused frontend tests passed (5 files, 63 tests) covering shared audio URL/auth helper, voice preflight, voice stream auth frame ordering, and Speech page compile; backend audio auth contract test passed (7 selected, 5 deselected) covering query-token rejection and first-frame auth acceptance across TTS, STT, and voice chat routes; git diff --check passed; production source token scan found no audio WebSocket query-token strings outside negative tests. Bandit on the touched Python test file reported only LOW B101 pytest assert findings. Persona WS auth and live backend browser smoke remain open in TASK-12113.

Draft PR opened for the audio portion of this task: https://github.com/rmusser01/tldw_server/pull/2630. The PR is intentionally draft pending the repository-required human-written Change summary. TASK-12113 remains In Progress because persona WebSocket auth and live backend/browser smoke acceptance criteria are still open.
2026-07-04 PR #2630 review follow-up before the later rebase from origin/dev fd5c152b065c408e4e8ee5f08da41589f21cb7f5: addressed review comments on the Speech Playground TTS WebSocket auth frame path. Added a source-level contract update using flexible regex matching instead of strict indexOf string matching, plus a regression requiring the auth/initialization failure handler to close the WebSocket before any prompt payload is sent. Red verification: source contract failed before production change with "Missing auth failure handler". Production change: wrapped the WebSocket onopen initialization flow from auth frame through prompt send in try/catch; failures now set stream error state, close the socket, and avoid unhandled async rejection. Green verification: npm test -- src/services/__tests__/audio-websocket-auth.test.ts passed (6 tests); git diff --check passed. Typecheck limitation: npm exec tsc was blocked by root-owned npm cache; direct tsc via main workspace dependencies reached Node default heap OOM; a larger-heap retry was blocked by the environment approval/usage guard, so no full typecheck result was available in this run.
Post-rebase validation on current origin/dev 4c1ca5d8358bff2a5a7fb5c75d60d1bd6728e702: rebased codex/audit-audio-ws-auth-contract-2026-07-04 so merge-base equals current origin/dev. Fresh verification after rebase: npm test -- src/services/__tests__/audio-websocket-auth.test.ts passed (6 tests); git diff --check HEAD~1..HEAD passed. Larger-heap UI package typecheck completed but failed on pre-existing/unrelated diagnostics outside the touched files (ChatGreetingPicker test checksum property, missing OpenUI @openuidev modules/types, background session fixture missing quickIngestBatches, missing typescript module resolution in route-registry helper, TldwChat abort spread tuple typing, and character-export SSRF tuple cast). No typecheck diagnostic referenced the touched WebSocket component or audio-websocket-auth contract test.
2026-07-04 current-dev refresh: rebased `codex/audit-audio-ws-auth-contract-2026-07-04` onto `origin/dev` `09d9ec901e1d4548f7924f1c6bcefa963fadd9bd`; merge-base matches `origin/dev`. Validation: `npm test -- src/services/__tests__/audio-websocket-auth.test.ts` passed with 6 tests; `git diff --check HEAD~1..HEAD` passed. Typecheck was rerun with `NODE_OPTIONS=--max-old-space-size=8192 ../voice-assistant-sdk/node_modules/.bin/tsc --noEmit --pretty false` because this worktree's UI install lacks a local `node_modules/.bin/tsc`; it failed only on existing diagnostics outside the touched audio WebSocket files: ChatGreetingPicker checksum property, missing OpenUI modules/types, background session fixture missing `quickIngestBatches`, route registry helper missing `typescript` module resolution, `TldwChat.abort` spread tuple typing, and `character-export.ssrf` tuple cast.
2026-07-04 latest-dev refresh: rebased and validated PR #2630 on origin/dev 6b727b221e55646eba663a03571e38302f7fafc2. Tested head c096bb70c2f2. Verification: npm test -- src/services/__tests__/audio-websocket-auth.test.ts => 6 passed; git diff --check HEAD~1..HEAD => clean. TypeScript package check with NODE_OPTIONS=--max-old-space-size=8192 ../voice-assistant-sdk/node_modules/.bin/tsc --noEmit --pretty false exited 2 with 12 known unrelated diagnostics and no audio-websocket/audio-service file matches.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the frontend audio websocket auth contract. Final refresh validated against origin/dev 6b727b221e55646eba663a03571e38302f7fafc2 with focused Vitest coverage passing and whitespace check clean; package-level TypeScript remains blocked by unrelated diagnostics outside the touched audio websocket files.
<!-- SECTION:FINAL_SUMMARY:END -->
