---
id: TASK-13381
title: Chat cannot reach its own HTTP-status extraction for NetworkError
status: Done
assignee: []
created_date: '2026-09-23 17:54'
updated_date: '2026-09-28 19:39'
labels:
  - bug
  - chat
  - llm
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Chat/chat_orchestrator.py` calls `_get_http_status_from_exception` in three places (`:552`, `:1181`, `:1659`), each guarded by `except _CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS`. `NetworkError` is not in that tuple, so for a `NetworkError` none of those handlers run and the extraction is unreachable on the Chat path.

Verified on dev, 2026-09-23:

```
tuple members: AssertionError, AttributeError, ConnectionError, ImportError, KeyError,
               LookupError, OSError, RuntimeError, TimeoutError, TypeError, ValueError,
               UnicodeDecodeError, RequestException, RequestError, HTTPError
issubclass(NetworkError, tuple) -> False
NetworkError.__mro__ -> NetworkError, Exception, BaseException, object
```

`NetworkError` derives straight from `Exception` (`core/exceptions.py:545`), so nothing in the tuple covers it incidentally.

Consequences:
1. The status-extraction fix landed in #2981 and consolidated in #2994 is **inert on the Chat path** for a `NetworkError`. It works when called; Chat never calls it.
2. The downstream `ChatProviderError` 5xx branch in each handler is unreachable for `NetworkError`.
3. There is no FastAPI exception handler registered for `NetworkError` either, so one raised on the Chat path propagates uncaught and surfaces as a generic **500** -- losing the upstream status entirely, with no Retry-After and no rate-limit classification for a caller keyed on 429.

This matters because `http_client._AiohttpResponse.raise_for_status` raises `NetworkError(f"HTTP {status}")` with no `.status_code` attribute, and `Embeddings/connection_pool.py` does the same -- the exact shape whose message parsing #2981 repaired.

Already tripwired: `tests/LLM_Calls/test_http_status_extraction_parity.py::test_chat_orchestrator_cannot_yet_reach_this_path` asserts `not issubclass(NetworkError, _CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS)`. It is written to **fail** once this is fixed, so whoever fixes it is told to retire the note and this task.

Split out of TASK-13287 deliberately: widening what the Chat error handler catches changes behaviour at three call sites covering every Chat provider call, so it needs its own blast-radius assessment rather than riding along with a regex repair.

Open question for whoever takes it: adding `NetworkError` to the tuple is the small change, but it also makes the handler swallow transport failures it currently lets through, which may alter retry behaviour upstream. Check that before assuming the one-line version is correct.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A NetworkError carrying an HTTP status in its message reaches Chat's status extraction, or a recorded decision says it should not
- [x] #2 The blast radius of widening the tuple is assessed at all three call sites, including any retry behaviour that currently depends on NetworkError propagating
- [x] #3 test_chat_orchestrator_cannot_yet_reach_this_path is retired or inverted, and TASK-13287's note removed
- [x] #4 An upstream 429 on the Chat path surfaces as ChatRateLimitError (HTTP 429) rather than 500; Retry-After is forwarded only where the upstream exception carries headers
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Closed 2026-09-28, verified on dev after #3011 merged. AC1: NetworkError and RetryExhaustedError are in _CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS (chat_orchestrator.py:50,99); tests/Chat/unit/test_orchestrator_network_exception_routing.py pins that they reach the handler and that NetworkError('HTTP 429') classifies as 429, which the handler maps to ChatRateLimitError. AC3: nothing in core/Chat, core/Character_Chat or endpoints/chat.py catches a raw NetworkError (the only reference is the _is_network_exception classifier), so no retry behaviour depended on it escaping. AC4: the tripwire test is gone from dev. AC2 amended below: a message-only NetworkError('HTTP 429') has no response headers, so there is no upstream Retry-After to forward.

Evidence correction (Qodo on #3048): test_orchestrator_network_exception_routing.py's first two tests only pin classification and tuple membership. Added test_chat_api_call_maps_a_network_error_through_the_real_handler, which drives chat_api_call itself with the dispatcher raising NetworkError: 'HTTP 429' -> ChatRateLimitError(429), status-less -> ChatProviderError(504). Probed: both cases fail with NetworkError removed from the tuple.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
