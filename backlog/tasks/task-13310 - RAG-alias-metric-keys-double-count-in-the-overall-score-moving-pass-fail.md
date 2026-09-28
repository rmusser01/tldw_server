---
id: TASK-13310
title: RAG alias metric keys double-count in the overall score moving pass/fail
status: Done
assignee: []
created_date: '2026-09-22 04:53'
updated_date: '2026-09-22 14:31'
labels:
  - bug
  - rag
  - evaluations
dependencies: []
references:
  - 'tldw_Server_API/app/core/Evaluations/rag_evaluator.py:381'
  - 'tldw_Server_API/app/core/Evaluations/rag_evaluator.py:401'
  - 'tldw_Server_API/app/core/Evaluations/rag_evaluator.py:1042'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
evaluate() setdefaults answer_relevance / answer_faithfulness pointing at the SAME metric dicts as the canonical keys. The dedup that would remove the canonical key runs only "if not explicit_metrics", but eval_runner.py:1478 always passes a non-None metrics list (_normalize_metrics returns a list unconditionally), so on the API path the dedup NEVER runs.

Executed: metrics=["relevance","context_relevance"] with relevance=1.0 and context_relevance=0.0 gives overall 0.6666666666666666 with alias dupes vs 0.5 without. It propagates into per-sample avg_score, the 0.7 _evaluate_passed threshold and mean_score, so it moves results across pass/fail.

Source: synthesis F13
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Alias keys excluded from _calculate_overall_score, or emitted only in the response projection
- [ ] #2 Test pins the arithmetic for a metric set containing exactly one aliased metric plus another
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ALREADY FIXED - verified 2026-09-22, no action needed from this task.

Fixed in commit 9061081c0c "fix(rag,evals): restore PostgreSQL notes retrieval and stop alias keys inflating scores" (2026-09-21 22:07), which is an ancestor of HEAD. rag_evaluator._calculate_overall_score:1065-1073 now filters alias keys whose canonical twin is present, with a comment stating the exact defect.

Behaviour re-verified at runtime: metrics {relevance: 1.0, context_relevance: 0.0} now scores 0.5 both with and without the answer_relevance alias present. Previously 0.667 with the alias.

Closing as already-addressed rather than implementing. Synthesis F13 is stale as written.
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
