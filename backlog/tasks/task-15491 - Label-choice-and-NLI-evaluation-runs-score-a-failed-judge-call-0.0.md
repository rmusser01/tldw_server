---
id: TASK-15491
title: Label-choice and NLI evaluation runs score a failed judge call 0.0
status: To Do
labels:
- evaluations
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-15426 PR C's plan refresh and review (2026-11-12). Pre-existing: the same on the server's key. A label-choice or NLI fact-check evaluation run (POST /api/v1/evaluations/{eval_id}/runs) scores a sample whose judge call failed as a wrong answer. _eval_label_choice and _eval_nli_factcheck wrap their _call_adapter_text call in except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS and set the reply to an empty string (core/Evaluations/eval_runner.py:1929-1931 and :2123-2125; lines on 04905e02ea). That tuple (:74-84) holds ChatConfigurationError, ConnectionError and TimeoutError, as well as OSError, ValueError, TypeError, KeyError, RuntimeError, AttributeError and the httpx and aiohttp client errors. The empty reply parses to no label, so the sample scores accuracy 0.0 with passed False (:1982-1995, :2175-2187). A vendor 401, 403 or 429 and a quota refusal are not in the tuple: they already become a sample error in _process_batch (:1408-1420). With TASK-15426 PR C the same 0.0 also covers a handle refused for another provider (an evaluation edited between the run route's read and the task's) and a ConnectionError or TimeoutError on the user's own key. Intended, following the owner's 2026-11-10 ruling for benchmark items (TASK-15485: a failed judge call is not a score; neither a success nor a failure, left out of every average): such a sample becomes a sample error, not 0.0. Note: the run's aggregate scores every sample error 0.0 today, for every run type (_calculate_aggregate_results, :2319-2322, and pass_rate divides by every sample, :2348), while by_metric leaves it out (_calculate_metric_stats), so a sample error alone does not leave mean_score.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A label-choice run whose judge call raises an error _eval_label_choice catches today (ChatConfigurationError, ConnectionError, TimeoutError) records that sample as a sample error (its error set, counted in the run's failed_samples, no accuracy score), not accuracy 0.0, with a test
- [ ] #2 The same for an NLI fact-check run, with a test
- [ ] #3 The run's aggregate leaves such a sample out of mean_score and pass_rate, as by_metric already does, with a test. This changes the aggregate of every run type with a sample error; the owner confirmed it on 2026-11-12 ("yes")
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
