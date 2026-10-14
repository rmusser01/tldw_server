---
id: TASK-13635
title: RAG custom completeness metric never calls a provider
status: To Do
labels:
- rag
- evaluations
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
RAGCustomMetrics.evaluate_response_completeness (core/RAG/rag_custom_metrics.py, near line 233) awaits analyze(prompt, "..."), but analyze is a sync function that needs api_name, input_data and custom_prompt_arg. The call raises TypeError before any provider call, and the except clause falls back to the heuristic score, so the LLM path is dead code. Fixing it needs a choice of grading provider and model, and the call must run off the event loop (asyncio.to_thread) so it can be quota-checked. Found while planning LLM usage accounting PR 2b (TASK-13437).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The completeness metric calls a configured provider through analyze with the right arguments, off the event loop
- [ ] #2 With no provider configured it keeps the heuristic, and says so in the result
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
