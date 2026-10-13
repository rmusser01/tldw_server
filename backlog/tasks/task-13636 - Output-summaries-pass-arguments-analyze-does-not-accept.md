---
id: TASK-13636
title: Output summaries pass arguments analyze does not accept
status: To Do
labels:
- outputs
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
outputs_service._summarize_single_article (app/services/outputs_service.py, near line 666; the call near line 679) and _summarize_text_block (near line 869; the call near line 883) call analyze(input_data=..., custom_prompt=..., api_name=..., api_key=None, model=...). analyze takes custom_prompt_arg and model_override, so the call raises TypeError, the except clause logs a warning, and every summary falls back to the first 300 characters of the source text. Found while planning LLM usage accounting PR 2b (TASK-13437).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 _summarize_single_article and _summarize_text_block call analyze with custom_prompt_arg and model_override and return the model's summary
- [ ] #2 A test drives each through the real analyze signature
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
