---
id: TASK-15561
title: eval rag and eval quality CLI runners call their evaluators with arguments
  they do not take
status: To Do
labels:
- evaluations
- cli
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the independent review of the TASK-15489 plan (I2), verified by reading the signatures. _run_rag_evaluation raises TypeError: __init__() got an unexpected keyword argument 'api_name' on RAGEvaluator(api_key=, api_name=), and its evaluate(save_results=) has no such parameter. _run_quality_evaluation raises TypeError: ResponseQualityEvaluator() takes no arguments; the real evaluate parameters are prompt, response, expected_format, custom_criteria, api_name, api_key. _run_geval_assessment is fine. Needs its own decisions on --save, on RAGEvaluator's embedding provider and on the result shape. Sibling of TASK-15489 (eval custom), PR #3499.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 eval rag, eval quality and a batch spec of that type call the evaluator with its real parameters and print the score; a CLI test with a stubbed provider pins it
- [ ] #2 An item with no score prints as unscored, not 0.0, and the command exits non-zero (owner rule: a failed judge or model call is unscored)
- [ ] #3 The --save flag on those commands either stores the result or is removed, as decided; a test pins it
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
