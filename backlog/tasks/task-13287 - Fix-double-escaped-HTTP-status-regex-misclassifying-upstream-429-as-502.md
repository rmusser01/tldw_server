---
id: TASK-13287
title: Fix double-escaped HTTP status regex misclassifying upstream 429 as 502
status: To Do
assignee: []
created_date: '2026-09-22 03:55'
updated_date: '2026-09-22 18:56'
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
FIXED 2026-09-22 in PR #2981. Both double-escaped copies corrected: LLM_Calls/error_utils.py and Chat/chat_orchestrator.py. Zero double-escaped copies remain in the codebase.

Evidence beyond the new tests: against dev the LLM_Calls + Local_LLM suites go from 41 failed / 617 passed to 32 failed / 626 passed. Nine tests that already asserted correct 429/503 extraction were failing because of the dead regex and now pass. No test regressed.

Added a parity test across the live copies of this rule, since four independent implementations is how it drifted.

STILL OPEN, and pinned by a test so it is not forgotten: the Chat path cannot reach this extraction at all. NetworkError is absent from _CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS (verified: issubclass -> False), so the handler calling the extractor never runs for a NetworkError and the ChatProviderError(504) branch downstream is unreachable. Fixing the regex does not fix the Chat path. Widening what the chat error handler catches is a behaviour change with its own blast radius -- it needs its own task and its own reasoning about what else that tuple would begin swallowing.

Also still open from the original finding: consolidating the four copies (get_http_status_from_exception x4, get_http_error_text x3, is_network_error x3). The TTS reviewer argued separately that TTS's _is_http_status_error copies should NOT fold into this task -- they contain no regex at all, and routing a core/TTS adapter through core/LLM_Calls to classify an httpx exception would be a worse dependency than the duplication it removes.
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
