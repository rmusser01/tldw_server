---
id: TASK-13638
title: Callers that turn an LLM quota refusal into degraded output
status: To Do
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After LLM usage accounting PR 2b (TASK-13437) an LLM call for an over-quota user raises LLMQuotaExceededError at the call. These callers catch it broadly and go on with degraded output instead of reporting it: workflow adapters that return an error output (Workflows/adapters/content/generation.py, text/nlp.py, research/bibliography.py, content/image.py, content/summarize.py, llm/translate.py, which returns the input as a fallback translation; control/flow.py parallel drops it; control/state.py retry retries it); claims verification (Claims_Extraction/claims_service.py, whose _CLAIMS_NONCRITICAL_EXCEPTIONS holds HTTPException, falls back to heuristic verdicts); Prompt Studio test runs and evaluation (test_runner.py, evaluation_manager.py save error outputs and 0.0 scores) and benchmarks (evaluation_manager.py returns score 0.0); character memory extraction (character_memory_extraction.py returns an empty result). Decide for each whether to let the refusal through (except LLMQuotaExceededError: raise) or keep the degraded output and say so.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed caller either lets the refusal through or records that it was refused
- [ ] #2 The operator page lists what each one does
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
