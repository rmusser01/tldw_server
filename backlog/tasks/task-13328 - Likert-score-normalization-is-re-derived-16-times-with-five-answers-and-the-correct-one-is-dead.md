---
id: TASK-13328
title: >-
  Likert score normalization is re-derived 16 times with five answers and the
  correct one is dead
status: In Progress
assignee: []
created_date: '2026-09-22 04:57'
updated_date: '2026-09-22 19:46'
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
- [ ] #1 One normalizer with an explicit declared scale
- [ ] #2 The chosen formula is recorded as a decision
- [ ] #3 Tests cover the bottom of the range, not just the top
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
