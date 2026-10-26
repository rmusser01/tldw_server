---
id: TASK-13936
title: 'Price the provider tiers and caches the catalog ignores: Batch, OpenAI Flex
  and Fast, Gemini cached content, OpenRouter'
status: In Progress
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up 5 of Docs/superpowers/plans/2026-10-23-pricing.md, filed at the owner's request on 2026-10-25 so that LLM cost limits cover these. compute_costs (core/Usage) prices each call at the model's standard rate from model_pricing.json and pricing_catalog.py, so none of the following is costed correctly today. Anthropic fast mode is not in scope: TASK-13885 prices it. (1) Batch, 50% off at Anthropic and OpenAI: nothing in tldw_Server_API/app calls a Batch API today (no /v1/batches or messages/batches caller), but the Messages API endpoint accepts a usage block with service_tier 'batch' or 'priority' (endpoints/messages.py, the service_tier check) and the cost ignores it. (2) OpenAI's Flex and Fast (formerly Priority) service tiers: the OpenAI adapter never sets service_tier, but a caller can pass it through extra_body (merge_extra_body in openai_adapter.py), so a call can run on either tier. Cost it by the service_tier the response reports. (3) Gemini cached content: llm_usage_normalizer.py maps cachedContentTokenCount to cache_read_input_tokens and cache_intents.py can send cachedContent, but no google entry in model_pricing.json has a cache_read rate. (4) OpenRouter: pricing_catalog.py has an empty openrouter table ('sourced from model_pricing.json overrides'), so a call with no override costs the unknown-model rate; OpenRouter can report the call's own cost in its usage block. llm_usage_normalizer.py does not read service_tier at all today, so (1) and (2) start by carrying it into the usage record. Take the rates from each provider's pricing page on the day of the work, not from this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Gemini cached input tokens are costed at Gemini's cache-read rate for the catalogued Gemini models, with a test
- [ ] #2 An OpenRouter call is costed from OpenRouter's reported cost or a catalogued rate rather than the unknown-model rate, with a test, or the gap is documented in Usage_Quotas.md
- [ ] #3 Docs/Operations/Usage_Quotas.md says which tiers and caches are priced
- [ ] #4 An Anthropic call whose usage reports service_tier batch is costed at the Batch rate, with a test
- [ ] #5 An OpenAI call whose response reports service_tier flex, priority (Fast) or batch is costed at that tier's rate, with a test per tier
- [ ] #6 A service_tier value the catalog has no rate for is costed at the standard rate, with a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-26-pricing-tiers.md (owner-approved). ADR assessment: no new ADR; ADR-075 governs prompt_tokens including cache tokens, ADR-059 the backlog tool.
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
