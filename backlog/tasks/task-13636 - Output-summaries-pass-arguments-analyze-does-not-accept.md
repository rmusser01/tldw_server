---
id: TASK-13636
title: Output summaries pass arguments analyze does not accept
status: Done
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
- [x] #1 _summarize_single_article and _summarize_text_block call analyze with custom_prompt_arg and model_override and return the model's summary
- [x] #2 A test drives each through the real analyze signature
- [x] #3 Once the call works, an LLM quota refusal is let through or recorded in the output's status, not swallowed by the except Exception fallbacks (outputs_service.py near lines 688 and 892), so watchlist enrichment does not record a refused summary as completed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fixed: _summarize_single_article and _summarize_text_block call analyze(api_name, prompt, None, model_override=..., input_is_literal_text=True), so the text is sent once; _fill_prompt fills the template once (at {text}, or appended when absent), which also fixes custom prompts without {text} and braces in titles, since generate_group_summaries and classify_items_by_topic formatted twice. An Error: result falls back as before. LLMQuotaExceededError is re-raised by the two helpers, the per-group summary loop, classify_items_by_topic and POST /watchlists/outputs (whose broad tuple includes HTTPException), so the endpoint returns the 402 and enrich_output records enrichment_status failed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Watchlist output summaries never reached a provider: the helpers passed analyze keyword arguments it does not take, so every summary fell back to the first 300 characters. They now call analyze correctly, with the prompt filled once and a system prompt that follows it, an Error: result still falls back, and an LLM quota refusal is reported (402 on POST /watchlists/outputs; enrichment_status failed for scheduled enrichment) instead of being stored as a summary. Follow-ups: TASK-13642 (summary cache never persisted; fallback shown as an LLM summary).
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
