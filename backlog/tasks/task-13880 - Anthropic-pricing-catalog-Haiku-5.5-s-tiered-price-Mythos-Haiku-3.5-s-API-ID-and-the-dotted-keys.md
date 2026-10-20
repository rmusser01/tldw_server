---
id: TASK-13880
title: 'Anthropic pricing catalog: Haiku 5.5''s tiered price, Mythos, Haiku 3.5''s
  API ID and the dotted keys'
status: To Do
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-13858 priced the flat-priced current Anthropic models. Left over, measured with PricingCatalog().get_rate_details: claude-haiku-5-5 costs the unknown-model rate (0.01 and 0.03 per 1K, about 100x its input price); Anthropic prices it by prompt length (above or below 100,000 tokens, cache tokens included), which a flat catalog entry cannot express. Mythos 5.1 and 5 (limited availability) have no entry. Haiku 3.5's API ID claude-3-5-haiku-20241022 costs the unknown rate, so the claude-haiku-3.5 entry never prices a real request. The dotted keys (claude-opus-4.5, claude-opus-4.1, claude-sonnet-4.5, claude-haiku-4.5) match no API ID and appear in the WebUI model list. Prices are on platform.claude.com/docs/en/about-claude/pricing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 claude-haiku-5-5 is costed at Anthropic's price for its prompt length, with a test on both sides of 100,000 tokens
- [ ] #2 Every Anthropic entry's key is an API ID or alias, and Mythos and Haiku 3.5's API ID are priced or deliberately left out
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From the #3286 clean review (M3): two more Anthropic keys are not API IDs. claude-opus-4-7-20260103 and claude-sonnet-4-6-20260105 are in the catalog (TASK-13858 gave them cache rates) and appear in list_provider_models("anthropic"), so the WebUI model picker offers two IDs Anthropic would reject; models from the 4.6 generation on have dateless IDs only. AC #2 covers them. Warning for the implementer: keep the claude-opus-4 and claude-sonnet-4 keys (or add the dated IDs they cover). They are not dotted, but they price the dated 4.x IDs by substring (claude-opus-4-1-*, claude-opus-4-20250514, claude-sonnet-4-5-*, claude-sonnet-4-20250514); deleting them as non-IDs sends those requests to the unknown-model rate. Extend test_anthropic_model_ids_get_their_list_prices_and_cache_rates with those dated IDs.
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
