---
id: TASK-13287
title: Fix double-escaped HTTP status regex misclassifying upstream 429 as 502
status: In Progress
assignee: []
created_date: '2026-09-22 03:55'
updated_date: '2026-09-22 14:29'
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
STAGE 1 APPLIED. Three changes, all verified.

1) REGEX FIXED at core/LLM_Calls/error_utils.py:145 - r"HTTP\\s+(\\d{3})" -> r"HTTP\s+(\d{3})".

2) COPY DELETED, NOT JUST FIXED. core/Chat/chat_orchestrator.py had a third full copy of the same walk; its body is gone and it is now a thin alias delegating to the LLM_Calls canonical (the file already imported from LLM_Calls at :73, and the private copy had exactly one in-module caller). That is the promote-one-and-delete this task specifies. The 4th copy, Embeddings_Create.py:174, is untouched - different implementation, different precedence, belongs with the Stage 2 destination module.

3) THE COMPOUNDING DEFECT IS FIXED TOO, and this is the part the original ticket did not cover. NetworkError and RetryExhaustedError were absent from _CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS, verified at runtime: "NetworkError in tuple: False, caught by tuple: False". So NetworkError escaped chat_api_call UNMAPPED and the ChatProviderError(504) branch was unreachable - fixing the regex alone would NOT have fixed the Chat path. Conclusive evidence it was an omission rather than a design choice: _is_network_exception:282 explicitly names both types, and the handler has a branch for them. Both added to the tuple.

TESTS (both new, red before / green after):
- tests/LLM_Calls/test_http_status_extraction_parity.py - 24 tests asserting all three extractor copies agree across 400/401/429/500/503, embedded text, absent status, and attribute-branch precedence. Was 12 failed / 12 passed (the exact 2-of-3 split), now 24 passed.
- tests/Chat/unit/test_orchestrator_network_exception_routing.py - asserts the invariant that the tuple must catch everything _is_network_exception classifies. 3 passed.

REGRESSION: tests/Chat/unit + tests/LLM_Calls went from 33 failed / 2259 passed to 18 failed / 2274 passed. Net 15 fixed, 0 broken. 14 of the 15 are the new tests; one pre-existing test was also repaired. The remaining 18 are pre-existing and unrelated (tabbyapi/vllm strict filters, plus 2 hypothesis collection errors from a missing declared dep).

Still open: the Stage 2 consolidation into core/Utils/http_status_extraction.py covering all four copies plus the is_http_status_error cluster.
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
