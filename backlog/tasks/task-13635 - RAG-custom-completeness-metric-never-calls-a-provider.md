---
id: TASK-13635
title: RAG custom completeness metric never calls a provider
status: Done
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
- [x] #1 evaluate_response_completeness and evaluate_all_metrics, which nothing in the app or the WebUI calls, are deleted, so rag_custom_metrics makes no LLM call
- [x] #2 The docs that name the metric (the service prompt inventory, the RAG pipeline evaluation design and the LLM usage accounting design section 4b) say it was deleted
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: plan_quota_callers_rag_metric.md (PR B, tasks 10-11). Owner decision 4: delete the metric. The original ACs assumed a caller: git grep on origin/dev finds none in tldw_Server_API/app, apps/ or the tests other than test_rag_custom_metrics_sanitizers.py, and eval_runner uses only the coverage and diversity metrics. ACs amended to match.
Before: git grep on origin/dev found 7 lines (2 definitions, the aggregate's call, 4 lines in test_rag_custom_metrics_sanitizers.py); after: none. Deleted 152 lines and the two-test sanitizer file, which covered only the deleted methods. tests/RAG (without test_restricted_postgres_media_retrieval.py): 684 passed; eval_runner tests (unit/test_eval_runner.py and test_evaluations_stage2_user_isolation_and_usage_accounting.py, which import get_custom_metrics): 35 passed; tests/Docs 212 passed; ruff on rag_custom_metrics.py: all checks passed; test-quality ratchet: triage --enforce exit 0, test_test_quality_ratchet.py 1 passed. refresh_docs_published.sh changes nothing: the three edited Design docs are not mirrored.
Bandit -ll on rag_custom_metrics.py: no issues identified.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
evaluate_response_completeness and its only caller, evaluate_all_metrics, are deleted: nothing in the app or the WebUI called them, and the LLM path raised TypeError before any provider call. rag_custom_metrics now makes no LLM call. The service prompt inventory, the RAG pipeline evaluation design and the LLM usage accounting design say so. Wiring a completeness metric into eval_runner starts from git history.
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
