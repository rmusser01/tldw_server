---
id: TASK-13311
title: RAG alias metric keys inflate the overall score and move the pass threshold
status: To Do
assignee: []
created_date: '2026-09-22 04:54'
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
- [ ] #1 A failing test asserts overall_score for two metrics scoring 1.0 and 0.0 is 0.5, not 0.667
- [ ] #2 Alias keys are excluded from overall-score computation, or aliasing happens after scoring
- [ ] #3 The fix holds on the eval_runner path where explicit_metrics is always truthy
- [ ] #4 avg_score, mean_score and the 0.7 pass-threshold decision are verified against the corrected score
- [ ] #5 Alias keys remain present in the response for OpenAI-style compatibility
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
