---
id: TASK-13328
title: >-
  Likert score normalization is re-derived 16 times with five answers and the
  correct one is dead
status: Done
assignee: []
created_date: '2026-09-22 04:57'
updated_date: '2026-09-23 23:59'
labels:
  - duplication
  - evaluations
dependencies: []
references:
  - 'tldw_Server_API/app/core/Evaluations/rag_evaluator.py:1027'
  - 'tldw_Server_API/app/core/Evaluations/response_quality_evaluator.py:226'
  - 'tldw_Server_API/app/core/Evaluations/eval_runner.py:1424'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
rag_evaluator._normalize_score (1027-1039) implements (score-1)/4, is tested, and has ZERO production callers. Meanwhile raw/5.0 is inlined at 11 sites, and three further incompatible schemes exist: eval_runner._eval_summarization uses raw/max conditionally (max=3 for fluency), recipes use a caller-supplied max, evaluation_manager uses raw/10.0.

Executed: _normalize_score(1) = 0.0 vs inline 1/5.0 = 0.2; _normalize_score(3) = 0.5 vs 0.6. A retrieval system whose judge rates every context "1 = completely irrelevant" reports context_relevance 0.2 - a 20% floor under every metric - which also shifts the observed-range clamp and every threshold comparison. Both formulas agree at 5, so it is invisible on happy-path fixtures and only distorts the bottom of the range.

Within _eval_summarization the two branches disagree with EACH OTHER: dict {"fluency": 0.8} scores 0.8 while the identical string "fluency: 0.8" scores 0.267.

The tests pin the DEAD function contract, which is precisely why the divergence survived.

Destination: core/Evaluations/scoring.py - parse a judge raw score and convert it on a declared scale to 0-1. Changing the formula changes published scores, so it needs a recorded decision.

Source: synthesis F27
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One normalizer with an explicit declared scale
- [x] #2 The chosen formula is recorded as a decision
- [x] #3 Tests cover the bottom of the range, not just the top
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
APPLIED - decision recorded, migration done, impact being measured.

DECISION (recorded in the module docstring at core/Evaluations/scoring.py): the affine mapping of the declared range wins. On a 1-5 Likert scale the minimum observable score is 1, not 0, so 1 must map to 0.0. raw/5.0 mapped 1 to 0.2, putting a 20% FLOOR under every metric - a judge rating every context "1 = completely irrelevant" reported 0.2, which also shifted the observed-range clamp in _calculate_overall_score and every threshold built on it. Both formulas agree at 5, which is why the divergence was invisible on happy-path fixtures.
That mapping is exactly what RAGEvaluator._normalize_score always implemented - correctly, and with tests - while having ZERO production callers.

NEW: core/Evaluations/scoring.py with normalize_likert(raw, *, scale_min=1, scale_max=5) and parse_judge_score(). Out-of-range input is clamped rather than rejected. Other scales pass their own bounds instead of growing another copy.

MIGRATED 11 inline raw/5.0 sites: rag_evaluator.py 6, response_quality_evaluator.py 5. Zero "/ 5.0" remains in either. _normalize_score now delegates to the shared function, so there is one source of truth rather than a correct-but-dead one.

NOT migrated (genuinely different scales, not the same knowledge): eval_runner._eval_summarization uses max=3 for fluency, recipes/summarization_quality takes a caller-supplied max, evaluation_manager uses /10.0. Each can pass its own bounds to normalize_likert; tracked here, not done.

Tests: tests/Evaluations/unit/test_scoring_normalization.py, 21 cases including one asserting the shared function matches the previously-dead canonical for 0..6.

IMPACT SO FAR: baseline for tests/Evaluations/unit + property was 259 passed / 0 failed; after the change 280 passed / 0 failed. Exactly ONE assertion changed - test_response_quality_provider_boundary asserted 0.8 for a judge score of 4; the correct value is (4-1)/4 = 0.75. Updated WITH the reason in a comment, not silently; the score is incidental to that test, which covers executor bypass.

The wider tests/Evaluations dir shows 9 failed vs a 6-failed baseline, so ~3 further tests appear affected. Identifying them precisely before claiming otherwise.

2026-09-23 reconciliation:
AC2 met - decision recorded as an explicit DECISION section in the core/Evaluations/scoring.py module docstring (affine (raw-min)/(max-min); commit 8c1a637a2d). No ADR exists; if an ADR is required, that is a follow-up, but the AC wording is satisfied.
AC3 met - tests/Evaluations/unit/test_scoring_normalization.py (21 passed, none skipped with or without RUN_EVALUATIONS=1) includes test_the_bottom_of_the_range_is_zero_not_a_floor, clamping below min, and parity with the old _normalize_score for 0..6.
AC1 NOT met - normalize_likert exists with declared scale and the 11 raw/5.0 sites in rag_evaluator/response_quality_evaluator use it, but independent schemes remain: eval_runner.py:1429-1439 (raw/max_score, max=3 for fluency, still the dict/string branch inconsistency), recipes/summarization_quality.py:289 _normalize_score(value, max_score), evaluation_manager.py:613,632 (/10.0), and a site the earlier notes missed: recipes/rag_answer_quality_execution.py:1112 _coerce_unit_score returns numeric/5.0 for values >1 (the same 20%-floor bug). Also unresolved from the previous note: ~3 extra failures in the wider tests/Evaluations run were never identified.

2026-09-23 completion (commit da3f397302):
AC1 met. Every remaining judge-score normalizer now routes through core/Evaluations/scoring.py: eval_runner._eval_summarization (dict + string branches), recipes/summarization_quality._coerce_metrics, recipes/rag_answer_quality_execution (_coerce_unit_score -> _coerce_geval_score), evaluation_manager.evaluate_custom_metric (JSON + regex branches, declared 1-10), and a site the earlier notes missed: api/v1/endpoints/evaluations/evaluations_unified.py _normalize_geval_metric (raw/5 + an unreachable fluency scale-guess, dropped). New helpers: normalize_judge_score (value in [0,1) on a 1-based scale passes through as already normalized; exactly 1 is on-scale -> 0.0) and normalize_geval_metric (fluency 1-3, others 1-5). No raw/max normalization of judge scores remains in core/Evaluations.
Bugs found while consolidating, each pinned by a test that FAILED on 60d5d30f05 and passes now (tests/Evaluations/unit/test_scoring_normalization.py, 37 passed): summarization_quality and rag_answer_quality scored G-Eval raw 1 (worst) as 1.0 (perfect); rag_answer_quality put fluency on 1-5 so a perfect 3 scored 0.6; eval_runner dict vs string disagreed (0.8 vs 0.267) and floored 1 at 0.2; evaluation_manager regex mapped 'Score: 1' to 1.0 and JSON 0.8 to 0.08. Red run: 15 failed (assertion failures on the four sites + ImportError for the new helper).
VISIBLE SCORE SHIFT: G-Eval 4/5 0.8 -> 0.75; custom metric 8.8 0.88 -> 0.867; worst rating 0.0 everywhere. Stored results are not rewritten; before/after runs are not directly comparable. Updated with inline reason: test_eval_runner (2 asserts coherence 0.8 -> 0.75), test_evaluation_manager TestCustomMetrics (3 tests, /10 -> (x-1)/9).
Suite: tests/Evaluations with RUN_EVALUATIONS=1 -n 8: before 19 failed/863 passed, after 18 failed/880 passed. FAILED-set diff: no new failures; the one that disappeared (property test_overall_score_bounds) was a Hypothesis FlakyFailure and passes 3/3 in isolation. The earlier '~3 unidentified extra failures' question is resolved: all 18 remaining failures are environmental (503 credential_store_unavailable in the local credential store, route-mount tests) and none asserts a score.
Endpoint-level test for evaluations_unified was not added: the existing test_geval_endpoint itself fails locally with credential_store_unavailable, so the endpoint is covered through the unit-tested shared helper instead.
Bandit -ll on touched files: one pre-existing B608 at evaluation_manager.py:853 (untouched code).
Out of scope follow-up: core/RAG/rag_service/analytics_system.py:959,968 divide user feedback stars by 5 (1 -> 0.2) into stored analytics history; different module, changing it would break trend continuity, so it needs its own decision.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
One Likert normalizer with a declared scale (scoring.normalize_likert, plus normalize_judge_score / normalize_geval_metric for mixed raw/normalized inputs) is now the only live formula; all 16+ sites use it. The affine decision is recorded in the scoring.py docstring. Consolidating exposed real bugs (worst G-Eval rating reported as perfect in two recipes, fluency on the wrong scale, 1-10 custom metric 'Score: 1' -> 1.0), each pinned with a failing-first test. Scores below the top of the scale shift down; noted in commit da3f397302.
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
