---
id: TASK-15487
title: The registered benchmarks run on correctly shaped data
status: Done
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
- [x] #1 load_benchmark_dataset applies config.field_mappings, with a test that a real mmlu_pro row reaches the judge with its choices and correct answer
- [x] #2 simpleqa_verified has a loader entry and loads rows; gpqa_diamond loads with the right config name and subset; each with a test (stubbed loader, no network)
- [x] #3 The six benchmarks with no evaluator (aider_polyglot, mask, simpleqa, swe_bench, bfcl, vending_bench) build one and run, or are removed from the registry, as the owner decides
- [x] #4 Every formatter asks for the 1-10 scale the manager reads (code generation, honesty, instruction following, function calling, the module helpers, and bullshit_benchmark), with tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Corrections and additions (TASK-15485 final review, 2026-11-09): mmlu_pro and gpqa_diamond need the datasets package, which is in the data-tools extra, not the evaluation extra. TASK-15485 added a labelled reply format (SCORE: [0-10] / EXPLANATION:) to bullshit_benchmark's prompt; its 0-10 rubric still needs moving to the manager's 1-10 scale (AC #4). Also in scope when SimpleQA becomes runnable (AC #2): the GRADE read takes the first 'GRADE:' in the reply, so a judge that echoes the prompt's format line 'GRADE: [correct/incorrect/not_attempted]' scores 1.0 whatever it graded (fix: match (correct|incorrect|not_attempted)\b(?!/) and add a format-echo test); 'not attempted' with a space is not read (use not[_ ]attempted). The require_reasoning rubric (mmlu_pro, gpqa_diamond; benchmark_utils.py ~:490-493) still says '70% of score / 30% of score', so SCORE: 70% falls to 0.5.
Since TASK-15485's 2026-11-10 fold, a judge reply with no readable score leaves the item unscored (score None, counted apart from successful and failed, in no average), not 0.5. So the require_reasoning rubric's 'SCORE: 70%' leaves the item unscored rather than scoring 0.5. Parser gaps in evaluate_custom_metric that a human would read as a score, found by the fold's plan review (2026-11-10), in scope when the formatters move to 1-10 (AC #4): a bare-number reply '8' is unscored (json.loads parses it as an int, which is not a dict, so the regex branch never runs and its '^(\d+)\s*$' pattern cannot apply); {"score": "8"} (a numeric string) is unscored; 'Score: 1e308' reads its leading 1 and scores 0.0 (test_security pins that value as is).
Owner decisions 2026-11-16: D1 build simpleqa (OpenAI's published CSV) and remove aider_polyglot, swe_bench, mask, bfcl, vending_bench from the registry; D2 GPQA choices get a seeded shuffle per item (seeded by Record ID); D3 delete the dead BenchmarkRegistry.load_dataset; D4 simpleqa topics stay 'general', no follow-up filed. ADR-004 waiver covers this PR.
Tests: RED with only the test hunks applied (new test_benchmark_datasets.py, test_evaluation_manager.py, test_security.py) over the 3 files: 28 failed, 57 passed (plan: 28 RED incl. the known ssrf artifact). GREEN over the plan's 13-file touched set: 1 failed, 164 passed (plan: 1 failed, 164 passed); the 1 failure is the known test_ssrf_protection_private_ip probe artifact. Probe log on that run: 2 lines, both from the known test_security webhook tests (test_ssrf_protection_private_ip, test_blocked_ports); none from this plan's tests.
Wide set tests/Docs + tests/lint + tests/CI (-n 2): 24 failed, 1588 passed, 6 skipped, 1 error, netlines 0. The 24 are the known dev failures (13 test_playwright_browser_jobs_use_supported_ubuntu_image + 9 test_ux_browser_jobs_select_a_host_supported_by_locked_playwright = 22 ubuntu-latest workflow contracts, plus 2 test_jobs_postgres_exports_recorded_coverage); the 1 error is the environment-only test_runtime_package_discovery (no setuptools). Function-size ratchet: 13 passed. test_quality_triage --enforce rc 0. check_shard_coverage new_uncovered=0. check_backlog_task_ids ok. git diff --check clean. Bandit (uvx): only pre-existing B105 benchmark_utils.py:859/:949 and B608 evaluation_manager.py:898.
Ruling: commits group Tasks 1+2 (loaders, registry, docs examples, guide mirror) and Task 3 (formatters, parser, all tests) instead of one commit per task, and the prototype was applied whole after the RED run, not hunk by hunk: the registry changes of Tasks 1 and 2 share a file. Not measured: real datasets.load_dataset calls and real downloads (MMLU-Pro, GPQA, OpenAI CSV); datasets is stubbed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Benchmark rows now reach the evaluator in the benchmark's own fields (load_benchmark_dataset applies the registry field_mappings; mmlu_pro maps the real answer column; gpqa_diamond loads config gpqa_diamond and shapes seeded-shuffled choices; simpleqa_verified loads the shipped zip; load_csv reads while the file is open). The registry lists only benchmarks that build an evaluator (simpleqa built; five removed; dead load_dataset deleted). Every judge prompt asks for 1-10 via a shared JUDGE_SCALE_LINE and the manager parser reads a bare number, a numeric string and the GRADE forms, leaving exponent forms unscored.
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
