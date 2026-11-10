---
id: TASK-15485
title: Benchmark judge calls never reach the provider, so every benchmark item scores
  0.5
status: In Progress
labels:
- evaluations
- backend
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-facing: every /benchmarks and /evaluations/benchmarks run on dev scores each item 0.5 whatever the model answered, so average_score, min_score, max_score and by_category carry no information. Found by TASK-15426 PR B's plan review (B-4); fixed before that PR on the owner's instruction (2026-11-07). Which benchmarks it reaches: on a default install only simple_bench and bullshit_benchmark (mmlu_pro and gpqa_diamond need the datasets package; mmlu_pro rows are not field-mapped, so its judge is told the correct answer is 'None'; simpleqa_verified loads 0 rows and gpqa_diamond most likely 404s; six benchmarks get 501). Root cause (7a020da14a): EvaluationManager.evaluate_custom_metric sends the judge json.dumps(runtime_input_data) through Summarization_General_Lib.analyze without input_is_literal_text (Evaluations/evaluation_manager.py:589-597), so extract_text_from_input (Summarization_General_Lib.py:846-887) parses it and keeps only title, description, transcription, segments, text or content. The benchmark formatters build question/choices/correct_answer/response (benchmark_utils.py:506-511; SimpleQA question/expected_answer/model_response/topic, simpleqa_eval.py:68-73), so analyze returns 'Error: Could not extract text content.' (:673) without a provider call, and the parser defaults to 0.5 (evaluation_manager.py:648-651). Behind it: the score regex is case-sensitive ([Ss]core, :624, searched at :630) while every formatter asks for 'SCORE: X.X'; the multiple-choice formatter asks for 1.0/0.0 (benchmark_utils.py:496) while the manager reads 1-10, where 1 is the worst rating (scoring.py, TASK-13328), so a correct answer would score 0.0; SimpleQA answers 'GRADE: ...' (simpleqa_eval.py:121), which the manager never reads. Measured: each registered benchmark that builds an evaluator made one adapter call per item (the model's answer) and scored 0.5 for every judge reply. Stored results (save_results, internal_evaluations) from before the fix are 0.5 placeholders; no route or WebUI view reads them back.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A benchmark item's judge call reaches the provider with the formatted prompt and the model's answer, with a test that runs a registered benchmark's own formatter through the real evaluate_custom_metric and analyze to a stubbed adapter and sees two adapter calls
- [ ] #2 The item's score is read from the judge's reply in the format its formatter asks for (multiple choice on 1-10: 10 gives 1.0, 1 gives 0.0, also when the judge restates the rubric or writes markdown labels; a SimpleQA grade: correct 1.0, otherwise 0.0), not the 0.5 default, with tests; a reply with neither still scores 0.5; a failed judge call (provider error, timeout, rate limit, truncated or empty reply) scores 0.0 with the error text, not 0.5
- [ ] #3 Benchmark results from before the fix are documented as 0.5 placeholders (the benchmark guide and its Docs/Published mirror), naming simple_bench and bullshit_benchmark as the benchmarks whose scores are now meaningful; no code rewrites stored results
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-11-07-benchmark-judge-reaches-the-provider.md. Fixed before TASK-15426 PR B, which touches evaluate_custom_metric too (owner, 2026-11-07).
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
