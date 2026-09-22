---
id: TASK-13313
title: >-
  create_session_with_retries returns a different class under pytest than in
  production
status: To Do
assignee: []
created_date: '2026-09-22 04:54'
updated_date: '2026-09-22 20:51'
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
- [x] #1 Tests inject the session factory rather than relying on a PYTEST_CURRENT_TEST branch in production code
- [x] #2 The PYTEST_CURRENT_TEST branch is removed from create_session_with_retries
- [x] #3 A test exercises the production _SessionShim POST path for at least one affected provider
- [x] #4 A test sets LLM_ADAPTERS_NATIVE_HTTP_<P> to a false value and asserts the documented behaviour rather than an unhandled RuntimeError
- [x] #5 The 14 provider call sites no longer short-circuit their own kill switch under test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed on branch fix/llm-test-env-gating (off dev).

Two test files, both red first.

tests/LLM_Adapters/unit/test_native_http_kill_switch_reachable.py -- 16 failed / 19 passed before, 35 passed after. The red pattern was self-diagnosing: openai, parametrised as the control because it already uses the bare 'if self._use_native_http():' form, passed all seven cases; groq and openrouter failed only the chat/stream reachability cases; anthropic and custom_openai additionally failed the spelling cases, because their short-circuit sits inside _use_native_http() itself and bypasses the 0/false/no/off parsing entirely.

tests/LLM_Adapters/unit/test_session_shim_is_the_tested_object.py -- 3 failed / 2 passed before, 5 passed after. The failure message named the object it actually got: _RetrySession.

AC #5 fix, with a count correction: 11 sites across 4 adapters, not 14. Six were 'if _prefer_httpx_in_tests() or os.getenv("PYTEST_CURRENT_TEST") or self._use_native_http():' (groq, openrouter, anthropic, two each), two were 'if os.getenv("PYTEST_CURRENT_TEST"): return True' opening _use_native_http (anthropic, custom_openai), and three were the _prefer_httpx_in_tests helper definitions. All now read 'if self._use_native_http():', which is what openai, bedrock and the other adapters already did -- this is conformance to the existing majority pattern, not a new one.

Worth recording: the first two terms were the same check written twice. _prefer_httpx_in_tests() is defined as bool(os.getenv("PYTEST_CURRENT_TEST")), so 'A or A or B' was the literal shape.

The change is behaviour-neutral for every existing test, because each _use_native_http() defaults to True when its env var is unset -- which is why the regression run came back exactly at baseline.

AC #2 fix: the branch in create_session_with_retries is gone; it always returns _SessionShim now. log_runtime_deprecation is still called on the shim's streaming path, so the 'llm_chat_legacy_session' key stays live and tests/lint/test_no_new_runtime_compat_markers.py still passes.

TWO CLAIMS IN THE DESCRIPTION NEED QUALIFYING.

1. 'The two have different POST paths (dedicated client vs default transport), so any defect introduced into the production path is undetectable by CI.' The paths converge: _RetrySession.post and _SessionShim.post both end at http_client.fetch(method="POST", ...) and from there at the same _get_transport_adapter("httpx").request(). The only difference is that _RetrySession passes client=<create_client() instance> and the shim does not. For streaming they are identical, because the shim delegates to _RetrySession. So the risk was never divergent behaviour; it is that _SessionShim was never constructed under test, so a future edit to it is unobserved. That is still worth fixing, and is what makes removing the branch safe.

2. 'An operator ... hits an unhandled RuntimeError on every chat call.' The RuntimeError is deliberate and documented -- openai_adapter.py carries the comment 'If disabled explicitly, raise clear error rather than falling back'. The defect is that the switch could not be exercised by any test, not that the error is unhandled. The new test asserts that documented behaviour, which is what AC #4 asked for.

Verification: LLM_Adapters + LLM_Calls on this branch = 18 failed / 1230 passed. Baseline on origin/dev by detached checkout = 18 failed / 1195 passed, the identical 18 failures (all local adapters -- ollama, vllm, llamacpp, ooba, tabbyapi, aphrodite, local-llm -- none of which import any file changed here). Delta is +35 passing, exactly the new tests. Zero regressions. Adding tests/lint to the run gives 19 failed / 1242 passed; the extra is test_endpoint_auth_deps_import_boundary, also confirmed failing on a clean tree. ruff clean on all seven touched files.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
