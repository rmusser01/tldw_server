---
id: TASK-13311
title: RAG alias metric keys inflate the overall score and move the pass threshold
status: Done
assignee: []
created_date: '2026-09-22 04:54'
updated_date: '2026-09-23 23:20'
labels:
  - bug
  - evaluations
  - rag
dependencies: []
references:
  - 'tldw_Server_API/app/core/Evaluations/rag_evaluator.py:393'
  - 'tldw_Server_API/app/core/Evaluations/rag_evaluator.py:422'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`core/Evaluations/rag_evaluator.py` records each metric under its canonical key and then adds an **alias pointing at the same result dict**:

```python
results["metrics"][metric_name] = metric_result          # canonical
...
results["metrics"].setdefault("answer_relevance", metric_result)     # :393  alias -> same object
results["metrics"].setdefault("answer_faithfulness", metric_result)  # :395  alias -> same object
```

`_calculate_overall_score` is then called over that dict **including the aliases** (`:422`), so each aliased metric is counted twice.

The dedup that would drop the canonical key is gated on `if not explicit_metrics` (`:405`) — but `eval_runner.py:1484` always passes `metrics=self._normalize_metrics(eval_spec.get("metrics"), ...)`, i.e. an explicit list. **So on the runner path the dedup never executes and the double-count always applies.**

Executed:
```
two metrics, scores 1.0 and 0.0
  without aliases : 0.5
  with one alias  : 0.6666666666666666
```

**Effect:** the overall score is inflated by the alias weighting, and it propagates into `avg_score`, `mean_score`, and the **0.7 pass threshold** — so evaluations pass that should fail. The magnitude depends on which metrics score high, so it is not a constant offset that could be calibrated away.

Found by the comprehensive core-module review; independently reproduced by the orchestrator.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A failing test asserts overall_score for two metrics scoring 1.0 and 0.0 is 0.5, not 0.667
- [x] #2 Alias keys are excluded from overall-score computation, or aliasing happens after scoring
- [x] #3 The fix holds on the eval_runner path where explicit_metrics is always truthy
- [x] #4 avg_score, mean_score and the 0.7 pass-threshold decision are verified against the corrected score
- [x] #5 Alias keys remain present in the response for OpenAI-style compatibility
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Premise partly already fixed: 9061081c0c (earlier on this branch) deduped aliases inside RAGEvaluator._calculate_overall_score and added tests/Evaluations/unit/test_rag_evaluator_alias_scoring.py (1.0/0.0 -> 0.5, 0.9/0.3 -> 0.6 < 0.7). That covers overall_score, which unified_evaluation_service and the eval_runner pipeline path (overall_score at eval_runner.py ~810) consume.

Remaining bug found while verifying AC3/AC4: EvaluationRunner._eval_rag never reads overall_score; it averages every key in result['metrics'] itself, and on the runner path (explicit metrics) the alias keys are always present, so avg_score, the aggregate mean_score built from it, and the pass decision still double-counted relevance/faithfulness. Fixed in 0123ee97d6: avg_score skips an alias whose canonical twin is present (reuses rag_evaluator._CANONICAL_METRIC_FOR_ALIAS); the per-sample scores dict keeps the alias keys.

Test test_runner_avg_score_and_pass_gate_ignore_alias_keys drives the real RAGEvaluator.evaluate (stubbed metric calls) through runner._eval_rag with explicit metrics relevance=1.0, faithfulness=0.0, context_relevance=0.0 and threshold 0.35. RED before 0123ee97d6: avg_score 0.4 != 1/3 (and would pass the 0.35 gate). GREEN: avg_score 1/3, passed False, aggregate mean_score 1/3, pass_rate 0.0, alias keys still in scores. (A symmetric 1.0/0.0 pair hides the bug because both aliases are added, so the test uses three metrics.) Evaluations/unit with RUN_EVALUATIONS=1: 268 passed after. Bandit -ll on eval_runner.py: no findings. Ruff: 2 pre-existing findings, same count on ea1cbc6941.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Alias keys (answer_relevance, answer_faithfulness) no longer inflate RAG scores: overall_score was fixed in 9061081c0c, and the eval_runner's own avg_score/mean_score/pass decision is fixed in 0123ee97d6. Alias keys remain in the response. No known skips.
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
