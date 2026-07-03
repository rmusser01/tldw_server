---
id: TASK-12098
title: Stop replaying non-idempotent chat POST on stream connection timeout
status: Done
labels:
- bug
- high
- chat
- extension
- packages-ui
documentation:
- apps/FRONTEND_AUDIT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Severity: High (duplicate generation + duplicate saved messages).** From the 2026-07-02 frontend audit (finding H6). Independently flagged by **three** reviewers.

`apps/packages/ui/src/services/background-proxy.ts:1236-1243,1320-1324`: the port-stream path has a hard-coded 5s `CONNECTION_TIMEOUT_MS`. If no stream byte arrives within 5s, `bgStream` disconnects the port and **re-sends the entire request** via `bgStreamDirect`. `/api/v1/chat/completions` (with `save_to_db`/`conversation_id`) and `/api/v1/chats/{id}/complete-v2` are not idempotent, and time-to-first-byte > 5s is normal for large prompts, RAG, or a cold local model. Result: the server processes the request twice → duplicate generation and duplicate persisted messages. Same replay on the early transport-error path (`:1325-1332`).

Related dead-code hazard (fold in): the `stream_transport_interrupted` sentinel that `bgStream` synthesizes (`:1334-1349`) can never match in normal/RAG/tab/document modes, because the token extractor drops non-string chunks (`services/tldw/TldwChat.ts:574`, `models/ChatTldw.ts:217`) before `chatModePipeline.ts:582-599` can see it. So an extension port loss mid-answer silently truncates and saves as complete (character chat handles it correctly — divergence).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The stream connection timeout is derived from config (e.g. `streamIdleTimeoutMs`/a generation-appropriate value), not a hard-coded 5s.
- [ ] #2 Non-idempotent POSTs (chat completions, complete-v2) are not silently re-sent after the connection timeout or an early transport error.
- [ ] #3 A slow first token (>5s) on a non-idempotent endpoint results in exactly one server-side generation and one persisted message.
- [ ] #4 The `stream_transport_interrupted` event is surfaced through the token pipeline so normal/RAG modes mark a truncated stream as interrupted (parity with character chat).
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
