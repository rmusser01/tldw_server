---
id: TASK-13331
title: >-
  Add a lenient tiktoken resolver instead of migrating to the strict-by-design
  canonical
status: Done
assignee: []
created_date: '2026-09-22 04:58'
updated_date: '2026-09-23 19:38'
labels:
  - duplication
  - llm
dependencies: []
references:
  - 'tldw_Server_API/app/core/LLM_Calls/tokenizer_resolver.py:938'
  - 'tldw_Server_API/app/core/Workflows/adapters/text/nlp.py:479'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The tiktoken encoding_for_model -> cl100k_base fallback is rewritten 11 times in five exception dialects. resolve_tiktoken_encoding (938-942) exists, is lru_cached and importable - and RAISES TokenizerUnavailable instead of falling back, DELIBERATELY, to serve strict token counting. That is WHY nobody adopted it.

So this is true-duplication needing a lenient twin, NOT an adoption campaign. Reporting it as an adoption gap would have recommended breaking the strict callers.

Three of the eleven (Workflows/adapters/text/nlp.py:479, RAG/rag_service/utils.py:26, Workflows/adapters/evaluation/eval.py:459) catch only KeyError, but encoding_for_model(None) raises AttributeError - so a workflow step with model: null CRASHES where its three bare-except siblings degrade to cl100k_base and return a count.

Add resolve_tiktoken_encoding_or_default(model) beside the strict twin. 3 of the 11 sites are owner-only (under app/api/v1/**).

Source: synthesis F31
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A lenient resolver exists beside the strict one, both memoized
- [x] #2 The three KeyError-only sites no longer crash on a non-string model
- [x] #3 Strict callers are unaffected
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
REGRESSION CONFIRMED CLEAN: 331 passed / 0 failed across the 7 test files that actually import tokenizer_resolver, TokenCounter, run_token_count_adapter or run_context_window_check_adapter. 1m55s.

The earlier unscoped attempt (whole tests/Workflows + tests/RAG) ran ~5 hours without producing output and was stopped; it was killed BEFORE its git stash fired, so no state was left behind (verified: stash list empty, all four changed files intact). Scoping the regression to importers is both faster and a better signal.

2026-09-23 reconciliation: AC1 met - resolve_tiktoken_encoding_or_default added beside the strict resolver in LLM_Calls/tokenizer_resolver.py:944-971, memoized via lru_cache on _resolve_tiktoken_encoding_or_default_cached (strict twin already lru_cached) (commit 8c1a637a2d). AC2 met - nlp.py:484, eval.py:462 and RAG/rag_service/utils.py:29 all call the lenient resolver; tests/LLM_Calls/test_tiktoken_lenient_resolver.py 9 passed (covers None/123/''/whitespace, TokenCounter(model=None), run_token_count_adapter model=None); eval.py run_context_window_check_adapter has no test, verified by direct probe: {'model': None, 'text': 'hello world'} -> token_count 2, no crash. AC3 met - 8c1a637a2d diff to tokenizer_resolver.py is purely additive; test_strict_twin_is_unchanged asserts TokenizerUnavailable still raised; strict callers (Sharing/shared_workspace_chat_service.py:985, tokenizer_resolver.py:976/1716/1730) untouched. Bandit (uvx bandit on the 4 touched files): 1 Low B110 try/except/pass at tokenizer_resolver.py:952 - the intended fallback-to-cl100k_base, annotated noqa BLE001. Known scope note: only the 3 KeyError-only sites were migrated; the other 8 hand-rolled fallbacks remain (not required by any AC).
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Lenient twin resolve_tiktoken_encoding_or_default(model) added next to the strict resolve_tiktoken_encoding in core/LLM_Calls/tokenizer_resolver.py, both lru_cached (commit 8c1a637a2d). The three KeyError-only sites (Workflows text/nlp.py, Workflows evaluation/eval.py, RAG rag_service/utils.py) now use it and no longer crash on model=None; the strict function is byte-unchanged and still raises TokenizerUnavailable. Evidence: tests/LLM_Calls/test_tiktoken_lenient_resolver.py 9 passed on 2026-09-23, plus a direct probe of run_context_window_check_adapter with model=None. Bandit: one Low B110 on the deliberate fallback. The remaining 8 hand-rolled fallback sites were not migrated; no AC required it.
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
