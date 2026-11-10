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
