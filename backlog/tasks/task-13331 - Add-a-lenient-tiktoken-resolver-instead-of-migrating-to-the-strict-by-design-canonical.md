---
id: TASK-13331
title: >-
  Add a lenient tiktoken resolver instead of migrating to the strict-by-design
  canonical
status: In Progress
assignee: []
created_date: '2026-09-22 04:58'
updated_date: '2026-09-22 20:01'
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
- [ ] #1 A lenient resolver exists beside the strict one, both memoized
- [ ] #2 The three KeyError-only sites no longer crash on a non-string model
- [ ] #3 Strict callers are unaffected
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
REGRESSION CONFIRMED CLEAN: 331 passed / 0 failed across the 7 test files that actually import tokenizer_resolver, TokenCounter, run_token_count_adapter or run_context_window_check_adapter. 1m55s.

The earlier unscoped attempt (whole tests/Workflows + tests/RAG) ran ~5 hours without producing output and was stopped; it was killed BEFORE its git stash fired, so no state was left behind (verified: stash list empty, all four changed files intact). Scoping the regression to importers is both faster and a better signal.
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
