---
id: TASK-15489
title: The eval custom CLI command calls evaluate_custom_metric with arguments it
  does not take
status: To Do
labels:
- evaluations
- cli
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`tldw_Server_API/cli/commands/evaluation.py` `_run_custom_evaluation` is reached by the `eval custom` command and by a batch spec with `type: custom`. It calls `EvaluationManager.evaluate_custom_metric(text=, metric_name=, custom_prompt=, api_key=, api_name=, save_results=)`. The method takes `metric_name`, `description`, `evaluation_prompt`, `input_data`, `scoring_criteria`, `api_name` and `api_key`, so the call raises `TypeError` before any provider call. The command then reads `metrics` and `evaluation_id` from the result, which the method never returns (it returns `metric_name`, `score`, `explanation` and `raw_output`).

Found by code, not run, by the planner of TASK-15485's unscored fold (#3445).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `eval custom` and a batch spec with `type: custom` call `evaluate_custom_metric` with its real parameters (the text as input data, the prompt as the evaluation prompt) and print the score and explanation; a CLI test with a stubbed provider pins it.
- [ ] #2 An item with no score (`score: None` since TASK-15485) prints as unscored, not as 0.
- [ ] #3 `--save/--no-save` either stores the result or is removed: `evaluate_custom_metric` has no save parameter.
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
