---
id: TASK-13373
title: Move provider-adapter SSE loops onto streaming.iter_sse_lines_requests
status: Done
assignee: []
created_date: '2026-09-24 02:13'
updated_date: '2026-09-28 00:28'
labels:
  - tech-debt
  - streaming
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Nine provider adapters keep their own SSE read loops instead of streaming.iter_sse_lines_requests / aiter helpers. Migrating changes decode (errors='replace') and error-frame behaviour, so it needs per-adapter tests pinning the current frames first. Split from TASK-13370.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each migrated adapter has a test pinning its error frame and decode behaviour before the switch
- [x] #2 No adapter keeps a hand-rolled SSE loop, or the exceptions are documented
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-09-27 (32945b26fb, 43dc53fc5a, 04d16a16d4). Found: iter_sse_lines_requests called response.iter_lines(decode_unicode=...), but the legacy session facade returns an httpx response whose iter_lines() takes no args; the TypeError was swallowed into the error frame, so Moonshot (its only caller) and Z.AI (same call in its own loop) streamed only an error frame in production. Helper now falls back to iter_lines() on TypeError; Z.AI moved onto the helper (same provider_unavailable frame + DONE). The eight OpenAI-shaped adapters (OpenAI, Groq, OpenRouter, DeepSeek, HuggingFace, Qwen, Mistral, Bedrock) cannot use iter_sse_lines_requests without a client-visible change: it swallows in-band errors and transport failures into a generic frame, while they raise the status-aware sanitized ChatAPIError (chat layer maps it; OpenAI credential-refresh retry depends on it). Added streaming.iter_sse_lines_raising with their exact contract and deleted the eight copies. Kept own loops, documented in code: Anthropic, Google, Cohere (event translation), custom OpenAI and local adapters (stop at DONE/error, defer error frame/DONE until the response context exits). Behaviour differences: Z.AI now canonicalizes provider DONE to 'data: [DONE]' and honours STREAM_PROVIDER_CONTROL_PASSTHRU; Z.AI stream failures log via the helper's bounded debug log instead of log_provider_failure; Mistral's opt-in per-chunk debug log (LLM_ADAPTERS_STREAM_DEBUG) is gone; dead bytes-decode fallbacks removed (httpx yields str, so decode is httpx's errors=replace before and after). Tests: tests/LLM_Calls/test_adapter_sse_loop_pinning.py (42) drives each adapter through httpx MockTransport: deltas + one DONE, invalid UTF-8 -> U+FFFD, in-band error (ChatProviderError 502 for raising adapters; bounded frame for zai/moonshot), mid-stream ReadError. 36 green before any source change; the 6 zai/moonshot httpx cases were red (the bug) and are green after. LLM_Calls+Streaming: 698 passed, 12 failed both before and after (same pre-existing local-adapter strict-filter/ollama and character_chat_sse_unified_flag failures, verified on clean source). LLM_Adapters + chat streaming normalization + chat fallback: 1071 passed. ruff F clean on touched adapters. Bandit (uvx bandit -q) on touched source files: no findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Eight identical OpenAI-shaped adapter SSE loops replaced by streaming.iter_sse_lines_raising, Z.AI moved onto iter_sse_lines_requests, and the helper fixed for httpx responses, which un-breaks Moonshot and Z.AI streaming. Anthropic, Google, Cohere, custom OpenAI and local adapters keep documented own loops.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
