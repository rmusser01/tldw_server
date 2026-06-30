---
id: TASK-384
title: Implement paid-provider prompt cache intents
status: Done
assignee: []
created_date: '2026-05-15 15:54'
updated_date: '2026-05-15 16:14'
labels:
  - llm-cache
  - llm-adapters
  - cost-control
  - implementation
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-15-chat-worldbook-cache-cost-control-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-chat-worldbook-cache-cost-control-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add opt-in provider-neutral billing prompt-cache intent handling and adapter-level translation for paid providers with documented cache semantics. Cache behavior must remain unchanged unless explicitly requested, and diagnostics must distinguish requested cache intent from provider-proven cache usage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Provider cache intent is ignored by default and only affects requests when explicitly enabled.
- [x] #2 Provider-specific payload changes are isolated behind adapter-level helpers with exact outbound payload tests.
- [x] #3 OpenAI, Anthropic, Gemini, and OpenRouter handling follows current official documentation checked during implementation.
- [x] #4 Diagnostics distinguish cache intent requested from authoritative provider usage cache hits or writes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Provider docs checked on 2026-05-15:
- OpenAI prompt caching guide: https://platform.openai.com/docs/guides/prompt-caching (automatic prompt caching; prompt_cache_key and prompt_cache_retention request controls; cached_tokens usage proof).
- Anthropic prompt caching docs: https://platform.claude.com/docs/en/build-with-claude/prompt-caching (cache_control blocks/top-level controls; default 5m TTL, optional 1h TTL; cache_creation/cache_read usage fields).
- Google Gemini context caching docs: https://ai.google.dev/gemini-api/docs/caching (implicit caching on newer Gemini models; explicit caching uses cached content resources referenced as cached_content/cachedContent).
- OpenRouter prompt caching docs: https://openrouter.ai/docs/guides/best-practices/prompt-caching (automatic provider sticky routing; prompt_tokens_details cached/cache_write fields; Anthropic cache_control and provider routing constraints).

Implementation completed:
- Added provider-neutral billing prompt cache intent parsing/diagnostics in cache_intents.py.
- Wired OpenAI prompt_cache_key/prompt_cache_retention, Anthropic cache_control blocks, Gemini cachedContent references, and OpenRouter whitelisted provider/cache_control metadata.
- Added schema/API propagation for /chat/completions and character completion v2.
- Documented that extra_body remains an escape hatch but is not proof of cache activation.

Verification:
- RED: python -m pytest tldw_Server_API/tests/LLM_Calls/test_cache_intents.py tldw_Server_API/tests/LLM_Adapters/unit/test_provider_prompt_cache_intents.py -q --tb=short failed with missing cache_intents module and unsupported billing_prompt_cache_intent fields.
- GREEN: same focused new-test command passed: 11 passed.
- Focused changed-provider pack passed: python -m pytest tldw_Server_API/tests/LLM_Calls/test_cache_intents.py tldw_Server_API/tests/LLM_Calls/test_capability_registry.py tldw_Server_API/tests/LLM_Adapters/unit/test_provider_prompt_cache_intents.py tldw_Server_API/tests/LLM_Adapters/unit/test_openai_native_http.py tldw_Server_API/tests/LLM_Adapters/unit/test_anthropic_native_http.py tldw_Server_API/tests/LLM_Adapters/unit/test_google_gemini_tools_and_images.py tldw_Server_API/tests/LLM_Adapters/unit/test_openrouter_native_http.py -q --tb=short --disable-warnings => 45 passed.
- Adjacent schema/character pack passed: python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_request_schemas.py tldw_Server_API/tests/LLM_Calls/test_llamacpp_request_extensions.py tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_session_error_mapping.py -q --tb=short --disable-warnings => 40 passed.
- py_compile touched Python files passed.
- git diff --check passed.
- Bandit touched Python scope: /tmp/bandit_cache_intents_stage6.json, exit 0, results empty.

Known verification note:
- Full tests/LLM_Adapters/unit was attempted but failed on unrelated CustomOpenAIAdapter2 explicit-base-URL baseline and then hit the repo-level 300s app-fixture timeout.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented opt-in paid-provider prompt cache intents with provider-specific translations isolated to adapter payload boundaries. Defaults remain no-op; requested intent metadata is reported separately from authoritative provider usage cache hits/writes, and extra_body is not treated as confirmed cache activation.
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
