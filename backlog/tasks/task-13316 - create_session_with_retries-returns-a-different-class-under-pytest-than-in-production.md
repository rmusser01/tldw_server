---
id: TASK-13316
title: >-
  create_session_with_retries returns a different class under pytest than in
  production
status: To Do
assignee: []
created_date: '2026-09-22 04:54'
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
- [ ] #1 Factory returns the same object in tests and production
- [ ] #2 Tests that relied on the legacy iter_lines shape are migrated rather than deleted
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
