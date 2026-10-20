---
id: TASK-13879
title: Price OpenAI cached input tokens
status: To Do
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
OpenAI's prompt_tokens already include prompt_tokens_details.cached_tokens, and the usage normalizer already splits them out (cache_read_input_tokens, billable_input_tokens), but no OpenAI model in model_pricing.json has a cache_read rate, so compute_costs bills cached input at the full input rate. Found while planning TASK-13858. No adapter change is needed, only rates taken from OpenAI's published pricing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 model_pricing.json gives the OpenAI models that support prompt caching a cache_read rate from OpenAI's published pricing, with a test that a row with cached_tokens is costed with it
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
