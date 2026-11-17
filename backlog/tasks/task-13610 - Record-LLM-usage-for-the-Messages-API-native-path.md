---
id: TASK-13610
title: Record LLM usage for the Messages API native path
status: Done
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Anthropic-compatible Messages API (api/v1/endpoints/messages.py) sends requests for the providers in MESSAGES_NATIVE_PROVIDERS (anthropic, llama.cpp) straight to the provider over HTTP (_native_post_json, _prepare_native_stream_iterator). Those calls never reach a ChatProvider adapter, so the provider base class (ADR-067) records nothing for them and a check at the call (LLM usage accounting PR 2b) cannot see them. PR 2a gates the endpoint at the request boundary. Parent: TASK-13437.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A native Messages API call writes one llm_usage_log row for the user, streaming and not
- [x] #2 The row's tokens come from the provider's usage block when it sends one
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
RED: tests/Chat/unit/test_messages_native_usage_recording.py 9 failed, 2 passed on dev, each ValueError: not enough values to unpack (no row written); the two record-nothing tests are guards. GREEN: 11 passed. (The plan's prototype showed its first-draft reader failing 4 of them: a null message_delta field gave (0, 9), and the cached cases (3, 100) for (62003, 100) and (60003, 100).) Messages, LLM_Calls and Usage suites: 610 passed, 1 skipped. Other tests naming the recorder or the cache fields: 233 passed, 1 skipped (the Postgres file, no Postgres).
Owner decisions (2026-10-18): D1 (count cache_creation_input_tokens and cache_read_input_tokens as prompt tokens for an Anthropic-shaped block) accepted; D3 (the new ADR) accepted, so ADR-075 is Accepted. llama.cpp's /v1/messages reports input_tokens = n_prompt_tokens - n_prompt_tokens_cache with cache_read_input_tokens (llama.cpp tools/server/server-task.cpp), so the sum applies to it too. Cache rates and the Anthropic adapter's cache fields: follow-up TASK-13858.
ADR check: ADR required: yes. ADR path: Docs/ADR/075-native-messages-and-mcp-tools-use-the-provider-recorder.md (Accepted), extending ADR-067 (governing). Reason: two public recorder entry points outside ChatProvider extend ADR-067's one recording point, a durable provider-integration rule. Number: the plan proposed 076; 075 was free on dev and in every open PR on 2026-10-18 (PR #3226 holds 072-074), so 075 was taken.
No check at the call on the native path: both routes into _handle_messages carry Depends(require_llm_quota), which marks the request prechecked, so llm_quota_gate.user_to_check() returns None for any call-site check there. test_messages_refuses_an_over_quota_user_before_any_provider_call already pins the 402 for the native path, streaming and not. count_tokens generates nothing and stays unrecorded.
Bandit: uvx bandit -ll on the touched source, 0 findings (also 0 at all severities; bandit is not in .venv). Postgres-only tests skip locally; this change touches no counter or SQL. Not tested, by design: estimate rows on the native path count no Anthropic content and no system prompt (ADR-075).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Messages API calls to Anthropic and llama.cpp now write one llm_usage_log row for the user, streaming and not, through the provider base class's recorder (record_direct_response, accounted_direct_stream in LLM_Calls/usage_accounting.py; ADR-075). Tokens come from the reply's usage block; for a stream, input tokens from message_start and the final output count from message_delta, and a null field never erases a count. Cache reads and writes count as prompt tokens, because Anthropic's input_tokens is only the uncached remainder. An invalid 200 writes nothing. The boundary check already admits every native call, so no check at the call was added. Cache pricing is follow-up TASK-13858. ADR-075, ADR-067, the spec and Usage_Quotas.md updated.
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
