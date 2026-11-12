---
id: TASK-13936
title: 'Price the provider tiers and caches the catalog ignores: Batch, OpenAI Flex
  and Fast, Gemini cached content, OpenRouter'
status: Done
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
- [x] #1 Gemini cached input tokens are costed at Gemini's cache-read rate for the catalogued Gemini models, with a test
- [x] #2 An OpenRouter call is costed from OpenRouter's reported cost or a catalogued rate rather than the unknown-model rate, with a test, or the gap is documented in Usage_Quotas.md
- [x] #3 Docs/Operations/Usage_Quotas.md says which tiers and caches are priced
- [x] #4 An Anthropic call whose usage reports service_tier batch is costed at the Batch rate, with a test
- [x] #5 An OpenAI call whose response reports service_tier flex, priority (Fast) or batch is costed at that tier's rate, with a test per tier
- [x] #6 A service_tier value the catalog has no rate for is costed at the standard rate, with a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-26-pricing-tiers.md (owner-approved). ADR assessment: no new ADR; ADR-075 governs prompt_tokens including cache tokens, ADR-059 the backlog tool.
RED/GREEN (real counts, 2026-10-26, branch fix/pricing-service-tiers-13936 from origin/dev 553b2da867): Task 1 84 failed -> 84 passed; Task 2 23 failed, 3 passed -> 26 passed; Task 3 13 failed -> 13 passed over the new file (the existing Gemini usage test in test_chat_service_content.py also passes, so the plan's four-file command gives 13 failed, 1 passed -> 14 passed); Task 4 4 failed, 6 passed -> 10 passed; new file 127 passed; focused set 1529 passed, 1 skipped on dev -> 1662 passed, 1 skipped; wider set 976 passed, 14 skipped on dev -> 976 passed, 14 skipped (13 need live provider keys, 1 standing TestClient skip); tests/Docs 212 passed; adapter suites 770 passed; model-list tests 593 passed, 1 skipped; shard coverage new_uncovered=0; test_quality_triage --enforce exit 0.
Sources, fetched 2026-10-26: platform.claude.com/docs/en/about-claude/pricing (Batch 50% on input and output, stacks with cache and data residency), platform.claude.com/docs/en/api/service-tiers (Priority Tier is a capacity commitment, no token price), developers.openai.com/api/docs/pricing (Batch, Flex, Fast tables), developers.openai.com/api/docs/guides/priority-processing (GPT-5.6 and earlier report Fast as service_tier priority; a downgraded Fast call reports default), developers.openai.com/api/docs/guides/batch (the output example has no service_tier), ai.google.dev/gemini-api/docs/pricing (context caching prices), ai.google.dev/api/generate-content (promptTokenCount includes the cached content), ai.google.dev/gemini-api/docs/thinking (response pricing is output plus thinking tokens), openrouter.ai/docs/use-cases/usage-accounting and /docs/faq (usage.cost always sent, credits are dollars, BYOK cost is the 5% fee and upstream_inference_cost the provider's charge).
Decisions (owner-approved plan): OpenRouter costed from usage.cost (+ BYOK upstream), catalog fallback; no service_tier DB column (raw_usage_metadata_json keeps it); per-model OpenAI service_tiers, Anthropic batch as a 0.5 code constant; Gemini explicit-cache storage not costed (documented); a long prompt on a tier with no long-context price costs the standard long-context rate.
AC #1 scope: every catalogued Gemini text model with a published cache rate carries cache_read: the six on Google's pricing page and the four 2.5 Flash and Flash-Lite -preview entries, which are priced exactly as their GA model. The three 2.0 Flash entries (gemini-2.0-flash, -001, -lite) are deliberately excluded: Google publishes no cache rate for them, so their cached tokens cost the input rate (an over-charge). The gemini-2.5-pro-high pseudo-entry was deleted; gemini-2.5-pro's long_prompt holds its price.
Not verified (pages do not say): gpt-5.5-pro Batch/Flex and gpt-5.5/gpt-5.4 Fast long-context prices (the pricing page shows '-', the gpt-5.5-pro model page gives none; costed at the standard long-context rate); the service_tier a model after GPT-5.6 reports for Fast (none is catalogued; TASK-13915 note). Known gap, out of scope: Gemini thinking tokens are not costed (the adapter drops thoughtsTokenCount; TASK-13952). The Anthropic adapter does not carry service_tier: no caller gets batch on that path and priority costs standard.
Docs updated: Docs/Operations/Usage_Quotas.md (four bullets: service tiers, Gemini cache reads, OpenRouter, streamed replies and router calls; the override sentence names service_tiers), Docs/Providers/Model_Pricing_Catalog.md (service_tiers format, links, OpenAI example; stale format example removed) and the model_pricing.json _catalog_note. Neither doc is mirrored under Docs/Published. Bandit (uvx bandit, 2026-10-26) on the six touched source files: 0 findings. Follow-ups: TASK-13951 (streamed /chat/completions costed from an estimate), TASK-13952 (Gemini pricing gaps, high). Each task got a spec + quality review; deferred minors are listed in the PR.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Calls are costed at the service tier the provider reports: OpenAI Batch, Flex and Fast from per-model catalog rates (service_tiers, written from OpenAI's tables), Anthropic Batch at half price on every token category; any other tier costs the standard rate. The tier is folded into the usage record by usage_with_service_tier and kept in raw_usage_metadata_json (no schema change). Gemini cached input costs the context-caching rate (the adapter keeps cachedContentTokenCount; ten Gemini entries carry cache_read, the two Pro models a 200K long_prompt; the 2.0 Flash entries have no published cache rate). OpenRouter calls are costed from the usage.cost OpenRouter reports, plus the upstream charge with BYOK, with the catalog as fallback. Usage_Quotas.md says which tiers and caches are priced and what is not (streamed /chat/completions, router calls, Gemini storage, audio and thinking tokens).
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
