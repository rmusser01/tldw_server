---
id: TASK-15487
title: The registered benchmarks run on correctly shaped data
status: To Do
labels:
- evaluations
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up of TASK-15485 (owner decision, 2026-11-07). After the judge fix, on a default install only simple_bench and bullshit_benchmark reach the judge with meaningful data. mmlu_pro rows are not field-mapped (load_benchmark_dataset ignores config.field_mappings), so its judge is told the correct answer is 'None' and its scores look real but carry no information. simpleqa_verified builds an evaluator but load_benchmark_dataset returns 0 rows (no loader entry), so it gets 404. gpqa_diamond calls load_huggingface('Idavidrein/gpqa') with no config name and filters on a 'subset' column GPQA rows lack, so it most likely gets 404. mmlu_pro and gpqa_diamond need the datasets package (the evaluation extra). aider_polyglot, mask, simpleqa and swe_bench cannot build an evaluator (evaluation_params the classes do not take: languages, check_consistency, max_tokens, use_repo_context) and bfcl and vending_bench have no class, so all six get 501. The formatters other than multiple choice still ask 'SCORE: X.X' with fractional rubrics (a perfect 1.0 now parses to 0.0 on the manager's 1-10 scale), and bullshit_benchmark asks 0-10 while it is read on 1-10 (Score: 1 gives 0.0, 5 gives 0.444).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 load_benchmark_dataset applies config.field_mappings, with a test that a real mmlu_pro row reaches the judge with its choices and correct answer
- [ ] #2 simpleqa_verified has a loader entry and loads rows; gpqa_diamond loads with the right config name and subset; each with a test (stubbed loader, no network)
- [ ] #3 The six benchmarks with no evaluator (aider_polyglot, mask, simpleqa, swe_bench, bfcl, vending_bench) build one and run, or are removed from the registry, as the owner decides
- [ ] #4 Every formatter asks for the 1-10 scale the manager reads (code generation, honesty, instruction following, function calling, the module helpers, and bullshit_benchmark), with tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Corrections and additions (TASK-15485 final review, 2026-11-09): mmlu_pro and gpqa_diamond need the datasets package, which is in the data-tools extra, not the evaluation extra. TASK-15485 added a labelled reply format (SCORE: [0-10] / EXPLANATION:) to bullshit_benchmark's prompt; its 0-10 rubric still needs moving to the manager's 1-10 scale (AC #4). Also in scope when SimpleQA becomes runnable (AC #2): the GRADE read takes the first 'GRADE:' in the reply, so a judge that echoes the prompt's format line 'GRADE: [correct/incorrect/not_attempted]' scores 1.0 whatever it graded (fix: match (correct|incorrect|not_attempted)\b(?!/) and add a format-echo test); 'not attempted' with a space is not read (use not[_ ]attempted). The require_reasoning rubric (mmlu_pro, gpqa_diamond; benchmark_utils.py ~:490-493) still says '70% of score / 30% of score', so SCORE: 70% falls to 0.5.
Since TASK-15485's 2026-11-10 fold, a judge reply with no readable score leaves the item unscored (score None, counted apart from successful and failed, in no average), not 0.5. So the require_reasoning rubric's 'SCORE: 70%' leaves the item unscored rather than scoring 0.5. Parser gaps in evaluate_custom_metric that a human would read as a score, found by the fold's plan review (2026-11-10), in scope when the formatters move to 1-10 (AC #4): a bare-number reply '8' is unscored (json.loads parses it as an int, which is not a dict, so the regex branch never runs and its '^(\d+)\s*$' pattern cannot apply); {"score": "8"} (a numeric string) is unscored; 'Score: 1e308' reads its leading 1 and scores 0.0 (test_security pins that value as is).
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
