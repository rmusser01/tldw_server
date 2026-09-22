---
id: TASK-13287
title: Fix double-escaped HTTP status regex misclassifying upstream 429 as 502
status: In Progress
assignee: []
created_date: '2026-09-22 03:55'
updated_date: '2026-09-22 20:51'
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
- [ ] #1 A failing test reproduces 429 -> 502 misclassification through a NetworkError carrying only message text, before any production edit
- [ ] #2 The three copies of get_http_status_from_exception are reduced to one shared implementation with the working regex
- [ ] #3 get_http_error_text (3 copies) and is_network_error (3 copies) are consolidated in the same pass, or the residual duplication is documented with a reason
- [ ] #4 Behavioural divergences are preserved deliberately or fixed explicitly: Embeddings_Create.py:175-180 checks exc.status_code before exc.response.status_code while the others check response first
- [ ] #5 Existing coverage in tests/Local_LLM/test_http_utils.py:105-107 still passes against the consolidated helper
- [ ] #6 Bandit run for touched scope
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
