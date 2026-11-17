---
id: TASK-15489
title: The eval custom CLI command calls evaluate_custom_metric with arguments it
  does not take
status: Done
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
- [x] #1 `eval custom` and a batch spec with `type: custom` call `evaluate_custom_metric` with its real parameters (the text as input data, the prompt as the evaluation prompt) and print the score and explanation; a CLI test with a stubbed provider pins it.
- [x] #2 An item with no score (`score: None` since TASK-15485) prints as unscored, not as 0.
- [x] #3 `--save/--no-save` either stores the result or is removed: `evaluate_custom_metric` has no save parameter.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Owner decisions 2026-11-16 (plan Docs/superpowers/plans/2026-11-17-evals-runner-cli.md): (1) a label-choice or NLI call that succeeds but returns an unparseable reply keeps accuracy 0.0, since that call is the system under test, not a judge; (2) the known-provider check is the built-in adapter table (ChatProviderRegistry.DEFAULT_ADAPTERS plus canonical_provider_name aliases); (3) --save/--no-save removed from eval custom; (4) an unscored eval custom exits 1; (5) an explicitly named provider stops the prefix:/prefix/ split; (6) eval rag and eval quality runners are filed as TASK-15561, not fixed here; (7) ADR-004 waiver covers this PR. PR #3499.
Evidence: AC1 tldw_Server_API/tests/Evaluations/unit/test_evals_cli_custom_command.py (6 failed RED, 6 passed GREEN); AC2 the unscored print and exit-1 test and the batch Unscored row; AC3 --save/--no-save removed (the command never stored anything). --model is accepted but not sent because evaluate_custom_metric has no model parameter (pre-existing). Sibling runners: TASK-15561.
Final review fixes (PR #3499): uvx bandit on eval_runner.py, cli/commands/evaluation.py and cli/utils/output.py found no issues.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
eval custom and a batch type: custom spec call evaluate_custom_metric with its real parameters; no score prints as unscored and exits 1; --save/--no-save removed. CLI tests: 6 failed on base, 6 passed; CLI group 19 passed; wider set 316 passed. PR #3499.
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
