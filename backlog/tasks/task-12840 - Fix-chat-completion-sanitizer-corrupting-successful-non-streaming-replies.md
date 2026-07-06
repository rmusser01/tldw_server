---
id: TASK-12840
title: Fix chat-completion sanitizer corrupting successful non-streaming replies
status: Done
labels:
- bug
- critical
- frontend
- chat
- packages-ui
documentation:
- apps/FRONTEND_AUDIT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Severity: Critical (silent data loss on normal use).** From the 2026-07-02 frontend audit (finding C1).

`createChatCompletion` in `apps/packages/ui/src/services/tldw/TldwApiClient.ts:2649-2661` passes the entire **successful** non-streaming response through `normalizeChatCompletionResponseBody` → `sanitizeChatCompletionPayload` (`:156-217`), which recurses into every value and replaces any string matching `/traceback|stack|exception|error|\/Users\/|[A-Za-z]:\\|\.py:\d+/i` with the literal `"Chat completion failed."`, then wraps the result in a fake `status: 200` Response.

Because `choices[0].message.content` is a string at arbitrary depth, an ordinary assistant reply that contains the word "error"/"exception", a stack-trace snippet, or a file path is silently replaced. Object keys named `error`/`errors` are overwritten and `details`/`stack`/`traceback` keys are dropped. The **streaming** path (`streamChatCompletion:2664`) does NOT apply the sanitizer, so the same prompt succeeds when streamed and is corrupted when not — making this look like a model problem rather than a client bug.

Verified by direct read of `TldwApiClient.ts:156-217,2637-2679`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Successful (2xx) non-streaming chat completions are returned to consumers unmodified — assistant content containing "error"/"exception"/stack-trace text/file paths is preserved verbatim.
- [ ] #2 If defensive scrubbing of genuine server error payloads is still desired, it is scoped to actual error responses (non-2xx / error-shaped bodies), not applied to successful `choices[].message.content`.
- [ ] #3 Non-streaming and streaming chat return equivalent content for the same successful response.
- [ ] #4 A regression test covers a successful completion whose content matches the old regex (e.g. "To handle this exception, wrap it in try/catch") and asserts it is preserved.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
