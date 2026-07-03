---
id: TASK-12097
title: Fix chat abort and stream lifecycle
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
**Severity: High (core chat feature unreliable).** From the 2026-07-02 frontend audit (finding H5). Paths relative to `apps/packages/ui/src/`.

Several related defects in the shared chat send/stream pipeline:

1. **Stop doesn't abort the transport** (normal/RAG modes). `models/ChatTldw.ts:181-223` receives the UI `signal` but calls `tldwChat.streamMessage` without it; the signal is only polled at loop top, so the fetch/port stays open (server keeps generating + persisting) until the next token or the 30s idle timeout. Character chat threads the real signal (`domains/chat-rag.ts:1168`), so behavior diverges by mode.
2. **Singleton controller collisions.** `tldwChat` is a module singleton with one `currentController`, and every `streamMessage()` starts with `this.cancelStream()` (`services/tldw/TldwChat.ts:443-446`). Any two concurrent streams cancel each other — Compare mode (N parallel models) has N-1 die with "Request cancelled"; also breaks double-send and regenerate-while-streaming.
3. **Shared-controller clobber.** A finishing turn's `finally` unconditionally resets the shared streaming flag + abort controller (`chat-modes/chatModePipeline.ts:808-812`; also `useChatActions.ts:2082`), so an old turn re-enables the send button and nulls a newer in-flight turn's controller (which then can't be stopped).
4. **Stuck-streaming on early throw.** `onSubmit` sets `setStreaming(true)` then `await`s `buildChatModeParams` OUTSIDE the try (`useChatActions.ts:2335-2394`, try at `:2441`); a throw leaves the spinner + disabled send button stuck until reload.
5. **Abort takes the success path.** A user abort mid-stream `break`s the loop and then finalizes/saves the partial as a complete answer with no interrupted marker (`chatModePipeline.ts:579-652`); aborting before the first token can persist an empty/error assistant bubble.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The UI `AbortSignal` is threaded through `ChatTldw.stream` → `streamMessage` → `bgStream`, so Stop aborts the underlying fetch/port in all modes.
- [ ] #2 `tldwChat` uses per-call abort controllers (not a module singleton), so concurrent streams (Compare mode, double-send) don't cancel each other.
- [ ] #3 Each turn's `finally` resets the shared streaming flag/controller only if it still owns the current controller.
- [ ] #4 `buildChatModeParams` (and other pre-stream awaits) run inside `onSubmit`'s try, so a failure resets streaming state and drains the queue.
- [ ] #5 A user-aborted turn is finalized/saved as interrupted (or discarded), never as a complete answer; aborting before the first token does not persist an empty/error bubble.
- [ ] #6 Tests cover: Stop aborts the network stream; Compare mode runs N models without mutual cancellation; a pre-stream throw does not strand the UI.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
