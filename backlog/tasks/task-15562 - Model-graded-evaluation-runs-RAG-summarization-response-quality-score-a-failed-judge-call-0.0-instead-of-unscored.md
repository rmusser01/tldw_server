---
id: TASK-15562
title: Model-graded evaluation runs (RAG, summarization, response quality) score a
  failed judge call 0.0 instead of unscored
status: To Do
labels:
- evaluations
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the final review of PR #3499 (I3). The owner's rule: a failed judge or model call, or a reply with no readable score, is unscored (score None), a sample error, and out of every average; never 0.0. PR #3499 applies it to label-choice and NLI runs only. Model-graded runs still turn a failed judge into 0.0 before the aggregate sees it: eval_runner.py _eval_rag gives avg_score 0 when every metric failed (RAGEvaluator drops a metric whose judge raised); _eval_summarization does the same when G-Eval yields no metric; a None metric score becomes 0.0 in both; response_quality_evaluator.py scores a failed metric judge 0.0 (around lines 272, 340, 405, 473, 608). Measured example (evaluators stubbed): a RAG sample whose metrics all failed gives {'scores': {}, 'avg_score': 0, 'error': None}; aggregated with one passing 1.0 sample that gives mean_score 0.5, pass_rate 0.5, failed_samples 0.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A RAG or summarization sample whose judge calls all failed is a sample error (no avg_score), counted in failed_samples and in no mean_score or pass_rate; a test with a stubbed judge pins it
- [ ] #2 A metric whose judge failed or returned no readable score is left out of the sample's scores and average, not scored 0.0, in _eval_rag, _eval_summarization and response_quality_evaluator.py; tests pin each
- [ ] #3 Evaluations_API_Reference.md no longer carves out model-graded runs from the failed_samples note
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
