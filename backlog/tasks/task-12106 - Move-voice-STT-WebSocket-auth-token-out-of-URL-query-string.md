---
id: TASK-12106
title: Move voice/STT WebSocket auth token out of the URL query string
status: To Do
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
- [ ] #2 Audio STT/voice WS auth uses the `{type:"auth"}` first message matching `audio_streaming.py`; the token is no longer in the URL. Extension STT (`background.ts:3332`) updated the same way.
- [ ] #3 Auth is sent before any config/audio message so the server authenticates the connection first.
- [ ] #4 Validated against a running backend: persona live, voice chat, and streaming transcription all connect and authenticate with no token in the connection URL (checked in browser devtools / server logs).
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
