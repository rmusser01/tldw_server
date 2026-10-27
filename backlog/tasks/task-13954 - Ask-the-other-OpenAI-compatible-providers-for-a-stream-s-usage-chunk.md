---
id: TASK-13954
title: Ask the other OpenAI-compatible providers for a stream's usage chunk
status: To Do
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-13951, which asks only api.openai.com for a stream's usage chunk (stream_options.include_usage in OpenAIAdapter.stream). Streams from DeepSeek, Groq, Mistral, Qwen, Moonshot, Z.AI, HuggingFace and Bedrock reach both recorders with usage only if the provider sends a usage chunk unasked or the caller sets stream_options in extra_body; otherwise they are costed from an estimate that misses cache rates and reasoning tokens (for example deepseek-reasoner). Cohere's adapter stream turns the stream-end event, which carries Cohere's usage, into DONE (cohere_adapter.py). For each provider, read its API docs: if it documents a usage flag, ask for it in its adapter, only on the provider's own host as the OpenAI adapter does; if it sends usage unasked, pin that with a test; otherwise list it as estimated in Usage_Quotas.md. Local and custom OpenAI-compatible servers are left alone, since they may reject the field. Note: the /chat/completions forwarding guard drops only an empty-choices usage chunk for a client that did not ask; a provider whose usage chunk has no choices key at all is forwarded, so check each adapter's chunk shape.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed provider's stream is asked for usage (with a test), or is shown to send it unasked (with a test), or is listed as estimated in Usage_Quotas.md
- [ ] #2 Cohere's stream passes its usage on, with a test, or Usage_Quotas.md says it is estimated
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Final-review note (TASK-13951, Minor 5): a top-level stream_options from an OpenAI-SDK client never reaches the provider. ChatCompletionRequest keeps it in model_extra, and /chat/completions uses it only to decide whether to forward the usage chunk, but _build_openai_payload drops it and only extra_body is merged into the provider payload. So such a client gets the usage chunk from OpenAI (the adapter asks anyway) and from OpenRouter (always), but from no other OpenAI-compatible provider, even one that accepts the field. Consider forwarding the client's top-level stream_options into extra_body (or into the provider payload) alongside asking each provider.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
