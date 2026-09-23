---
id: TASK-13316
title: >-
  create_session_with_retries returns a different class under pytest than in
  production
status: Done
assignee: []
created_date: '2026-09-22 04:54'
updated_date: '2026-09-23 23:13'
labels:
  - bug
  - llm
  - tests
dependencies: []
references:
  - 'tldw_Server_API/app/core/LLM_Calls/chat_calls.py:108'
  - 'tldw_Server_API/app/core/LLM_Calls/http_helpers.py:55'
  - 'tldw_Server_API/app/core/LLM_Calls/chat_calls.py:62'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The factory branches on os.getenv("PYTEST_CURRENT_TEST") and returns the legacy _RetrySession under pytest but _SessionShim in production. The two have different POST paths: _RetrySession uses a dedicated client with its own pool and TLS context; _SessionShim calls fetch() with no client argument, taking the default transport-adapter path.

PYTEST_CURRENT_TEST is set for every test, so NO TEST EVER EXERCISES THE PRODUCTION OBJECT. Seven call sites are affected: cohere_adapter.py:259, moonshot_adapter.py:205/232, zai_adapter.py:129/192, chat_calls.py:227/320. A regression in _SessionShim.post (timeout default, header, pooling) is invisible to the whole suite.

test_provider_unsafe_post_no_retry.py:97-98 has to monkeypatch the name back to http_helpers.create_session_with_retries to test anything, which is the suite conceding the point.

Source: synthesis F17
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Factory returns the same object in tests and production
- [x] #2 Tests that relied on the legacy iter_lines shape are migrated rather than deleted
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in 61ae349512. create_session_with_retries no longer branches on PYTEST_CURRENT_TEST; it always returns _SessionShim. Streaming still delegates to the legacy facade inside the shim, so iter_lines callers are unchanged. AC2: test_provider_unsafe_post_no_retry no longer monkeypatches the factory back to http_helpers; it drives the real shim, injecting the mock transport into both paths (http_helpers._hc_create_client for streaming, http_client._get_httpx_client for non-streaming) and now asserts the shim does NOT close the shared cached client on the non-streaming path. Added test_session_factory_returns_the_production_shim_under_pytest. Red on ea1cbc6941: 7 failed (6 non-streaming no-retry cases + the new test); green after: 13 passed. tests/LLM_Calls + tests/LLM_Adapters + the other files that touch the factory: 9 failed/1260 passed before and after, identical FAILED lists (all 9 are pre-existing 'local model provider is currently unavailable' in local-LLM strict-filter tests, unrelated). Bandit clean. Docs: the factory docstring and header comment updated; no external docs reference the pytest branch.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Tests now get the same _SessionShim production uses; the one test that had to swap the factory back now exercises the real shim on both paths.
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
