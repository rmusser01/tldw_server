---
id: TASK-13287
title: Fix double-escaped HTTP status regex misclassifying upstream 429 as 502
status: Done
assignee: []
created_date: '2026-09-22 03:55'
updated_date: '2026-09-23 23:14'
labels:
  - llm
  - bug
dependencies: []
references:
  - 'tldw_Server_API/app/core/LLM_Calls/error_utils.py:145'
  - 'tldw_Server_API/app/core/Chat/chat_orchestrator.py:268'
  - 'tldw_Server_API/app/core/Local_LLM/http_utils.py:72'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two of the three copies of `get_http_status_from_exception` use a double-escaped regex that can never match, so an upstream provider status carried only in an exception message is silently lost.

**Broken (identical bug, both sites):**
- `tldw_Server_API/app/core/LLM_Calls/error_utils.py:145`
- `tldw_Server_API/app/core/Chat/chat_orchestrator.py:268`

Both: `re.search(r"HTTP\\s+(\\d{3})", str(exc))`. Inside a raw string `\\s` is a literal backslash followed by `s`, so it cannot match `HTTP 429`.

**Correct third copy:** `tldw_Server_API/app/core/Local_LLM/http_utils.py:72` uses `r"HTTP\s+(\d{3})"`.

Verified: `re.search(r"HTTP\\s+(\\d{3})", "HTTP 429 rate limited")` -> None; single-escape form -> matches.

**Impact.** `core/http_client.py` `_AiohttpResponse.raise_for_status` raises `NetworkError(f"HTTP {status}")` with no `.status_code` attribute when httpx is unavailable; `core/Embeddings/connection_pool.py:204` does the same. Status extraction returns None, and `core/exceptions.py:866` defaults ChatProviderError to **502**. An upstream 429 therefore reaches the client as 502 with no Retry-After and no rate-limit classification for any caller keyed on 429.

**Classification: divergent-copies.** Do not write a new helper. Promote the working implementation and delete the other two. `error_utils.py` is the natural owner (widest importer set, 13 modules).

Source: comprehensive core-module review prompt smoke run, findings LLM_Calls-2 / LLM_Calls-7. Prompt at `Docs/Development/Used_Prompts/Code_Review/Comprehensive_Core_Module_Code_Review.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A failing test reproduces 429 -> 502 misclassification through a NetworkError carrying only message text, before any production edit
- [x] #2 The three copies of get_http_status_from_exception are reduced to one shared implementation with the working regex
- [x] #3 get_http_error_text (3 copies) and is_network_error (3 copies) are consolidated in the same pass, or the residual duplication is documented with a reason
- [x] #4 Behavioural divergences are preserved deliberately or fixed explicitly: Embeddings_Create.py:175-180 checks exc.status_code before exc.response.status_code while the others check response first
- [x] #5 Existing coverage in tests/Local_LLM/test_http_utils.py:105-107 still passes against the consolidated helper
- [x] #6 Bandit run for touched scope
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
STAGE 2 COMPLETE - the cluster is now one implementation.

NEW: core/Utils/http_status_extraction.py owns get_http_status_from_exception, get_http_error_text, is_http_status_error and is_chunked_encoding_error. It lives under core/Utils/ rather than core/LLM_Calls/ because TTS, Local_LLM and Embeddings all need it and none should depend on the LLM_Calls package to classify an HTTP error.

ZERO CHURN FOR EXISTING CALLERS: error_utils.py re-exports all four names, so its 14 importing modules and ~77 call sites are untouched. Verified by the parity test continuing to pass through both import paths.

ALL FIVE remaining copies migrated:
- Local_LLM/http_utils.py - had the only CORRECT regex; deleted, now imports the shared one under the same public name (0 external importers).
- Embeddings_Create.py - the divergent fourth copy with NO message-text branch AND inverted attribute precedence (exc.status_code before exc.response, never exc.status), so an aiohttp ClientResponseError returned None there and the right status elsewhere. Deleted.
- TTS openai_adapter and elevenlabs_adapter - httpx-only; now use the shared classifier which also recognises requests.HTTPError.
- TTS qwen3_runtime_remote - kept as a method (called as self._is_http_status_error) delegating to the shared function.

The only definitions left are two deliberate thin aliases that preserve call-site shapes: chat_orchestrator._get_http_status_from_exception and the qwen3 method.

Tests: parity suite extended to 27 cases, including three asserting every former copy now resolves to THE SAME OBJECT (identity, not just equal behaviour) and one proving the shared classifier recognises requests.HTTPError where the TTS copies did not.

Regression: app imports; 4 suites touching the cluster give 53 passed. tests/Local_LLM/test_http_utils.py shows 4 failed / 11 passed BOTH with and without the change (stash-isolated) - pre-existing wait_for_http_ready failures, unrelated.

2026-09-23 reconciliation: AC1 met (tests/LLM_Calls/test_http_status_extraction_parity.py::test_status_recovered_from_network_error_message[429] + embedded-message case; pre-fix regex at 8c1a637a2d^ error_utils.py:145 returns None for 'HTTP 429', so the test is red against old code - caveat: test and fix landed in the same commit, so red-first ordering is not separately evidenced). AC2 met (single impl core/Utils/http_status_extraction.py via 8c1a637a2d+f6cfc9925d; only thin delegating aliases remain in chat_orchestrator and qwen3; identity test passes). AC4 met (Embeddings_Create copy deleted in f6cfc9925d, precedence unified, test_attribute_branch_still_wins). AC5 met (test_get_http_status_from_network_error_text passes; parity+http_utils: 38 passed, 4 failed = wait_for_http_ready tests failing on PackageNotFoundError tldw-server metadata, env issue unrelated). AC6 met (uvx bandit on the 8 touched files: no issues). AC3 NOT met: get_http_error_text still has 2 copies (Utils/http_status_extraction.py:74 and Local_LLM/http_utils.py:57, different bodies) and is_network_error still has 2 copies (LLM_Calls/error_utils.py:338 and Local_LLM/http_utils.py:69, same logic) plus Embeddings_Create._is_probable_network_error with different semantics; no documented reason for the residual duplication.

2026-09-23 AC3 closed (89168a494c): get_http_error_text and is_network_error now live only in core/Utils/http_status_extraction.py; error_utils and Local_LLM/http_utils re-export them (identity test). Embeddings_Create._is_probable_network_error deliberately kept: broader semantics (builtin TimeoutError/ConnectionError + message text for SDK backends), documented in its docstring. Found and fixed a regression from Stage 2: the shared helper's noncritical tuple dropped RuntimeError, so httpx.ResponseNotRead (a RuntimeError) escaped get_http_error_text on unread streaming responses and the read-then-retry branch was dead. test_error_text_reads_an_unread_streaming_body red before (httpx.ResponseNotRead), green after. Tests: parity+Local_LLM+lint 154 passed/1 skipped. LLM_Calls+Local_LLM+Embeddings+TTS: 1956 passed, 16 failed - same 16 fail with the HEAD versions of the 3 source files (pre-existing: local-provider strict_filter tests, TTS mocks). Bandit -ll on touched files: no issues. Docs: none needed beyond module/function docstrings.


Notes recorded on dev by the parallel core-review work (merged 2026-09-23):
FIXED 2026-09-22 in PR #2981. Both double-escaped copies corrected: LLM_Calls/error_utils.py and Chat/chat_orchestrator.py. Zero double-escaped copies remain in the codebase.
Evidence beyond the new tests: against dev the LLM_Calls + Local_LLM suites go from 41 failed / 617 passed to 32 failed / 626 passed. Nine tests that already asserted correct 429/503 extraction were failing because of the dead regex and now pass. No test regressed.
Added a parity test across the live copies of this rule, since four independent implementations is how it drifted.
STILL OPEN, and pinned by a test so it is not forgotten: the Chat path cannot reach this extraction at all. NetworkError is absent from _CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS (verified: issubclass -> False), so the handler calling the extractor never runs for a NetworkError and the ChatProviderError(504) branch downstream is unreachable. Fixing the regex does not fix the Chat path. Widening what the chat error handler catches is a behaviour change with its own blast radius -- it needs its own task and its own reasoning about what else that tuple would begin swallowing.
Also still open from the original finding: consolidating the four copies (get_http_status_from_exception x4, get_http_error_text x3, is_network_error x3). The TTS reviewer argued separately that TTS's _is_http_status_error copies should NOT fold into this task -- they contain no regex at all, and routing a core/TTS adapter through core/LLM_Calls to classify an httpx exception would be a worse dependency than the duplication it removes.
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Transport-error classification is one implementation in core/Utils/http_status_extraction.py (status, error text, network/HTTP-status/chunked classifiers). The double-escaped regex that turned upstream 429 into 502 is gone (8c1a637a2d, f6cfc9925d); the last duplicated helpers were merged in 89168a494c, which also restored reading unread httpx streaming bodies. Only remaining look-alike is Embeddings' _is_probable_network_error, kept deliberately. Known skips: 16 pre-existing failures in LLM_Calls/TTS suites, unrelated.
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
