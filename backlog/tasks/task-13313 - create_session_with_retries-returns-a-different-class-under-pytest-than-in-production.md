---
id: TASK-13313
title: >-
  create_session_with_retries returns a different class under pytest than in
  production
status: Done
assignee: []
created_date: '2026-09-22 04:54'
updated_date: '2026-09-23 19:34'
labels:
  - bug
  - llm
  - testing
dependencies: []
references:
  - 'tldw_Server_API/app/core/LLM_Calls/chat_calls.py:108'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`core/LLM_Calls/chat_calls.py:create_session_with_retries` branches on the test environment and returns a **different class** in each case — and its own docstring says so:

```python
"""
- Under pytest, return the legacy session facade so tests can patch ...
- In production, return a shim that routes non-streaming POSTs through ...
"""
if _os.getenv("PYTEST_CURRENT_TEST"):
    return _legacy_create_session_with_retries(...)
return _SessionShim(...)
```

Since `PYTEST_CURRENT_TEST` is set for **every** test, `_SessionShim` — the production object — is never exercised by the suite. The two have different POST paths (dedicated client vs default transport), so any defect introduced into the production path is undetectable by CI.

Affected providers: Cohere, Moonshot, Zai, and the legacy embeddings path.

The test seam already exists without this branch: 20 test files monkeypatch `chat_calls.create_session_with_retries` directly, and `test_provider_unsafe_post_no_retry.py:97-98` explicitly re-points it at the `http_helpers` version to get deterministic behaviour — i.e. the suite is already working around this gate.

Related, same class, same module: 14 provider call sites short-circuit on `PYTEST_CURRENT_TEST` before consulting their own `_use_native_http()` kill switch, so `LLM_ADAPTERS_NATIVE_HTTP_<P>=0` is unreachable under test and no test sets it to a false value. An operator setting it in production hits an unhandled `RuntimeError` on every chat call while CI stays green.

Found by the comprehensive core-module review; independently verified by the orchestrator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Tests inject the session factory rather than relying on a PYTEST_CURRENT_TEST branch in production code
- [ ] #2 The PYTEST_CURRENT_TEST branch is removed from create_session_with_retries
- [ ] #3 A test exercises the production _SessionShim POST path for at least one affected provider
- [ ] #4 A test sets LLM_ADAPTERS_NATIVE_HTTP_<P> to a false value and asserts the documented behaviour rather than an unhandled RuntimeError
- [ ] #5 The 14 provider call sites no longer short-circuit their own kill switch under test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Duplicate of TASK-13316 (filed twice during the 2026-09-22 review). Work and status are tracked there.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed as duplicate of TASK-13316.
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
