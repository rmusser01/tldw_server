---
id: TASK-13635
title: RAG custom completeness metric never calls a provider
status: In Progress
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
- [ ] #1 evaluate_response_completeness and evaluate_all_metrics, which nothing in the app or the WebUI calls, are deleted, so rag_custom_metrics makes no LLM call
- [ ] #2 The docs that name the metric (the service prompt inventory, the RAG pipeline evaluation design and the LLM usage accounting design section 4b) say it was deleted
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: plan_quota_callers_rag_metric.md (PR B, tasks 10-11). Owner decision 4: delete the metric. The original ACs assumed a caller: git grep on origin/dev finds none in tldw_Server_API/app, apps/ or the tests other than test_rag_custom_metrics_sanitizers.py, and eval_runner uses only the coverage and diversity metrics. ACs amended to match.
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
