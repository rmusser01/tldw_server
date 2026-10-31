---
id: TASK-13879
title: Price OpenAI cached input tokens
status: Done
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
- [x] #1 model_pricing.json gives the OpenAI models that support prompt caching a cache_read rate from OpenAI's published pricing, with a test that a row with cached_tokens is costed with it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-23-pricing.md (branch fix/llm-pricing-followups-fu2).
RED (Task 1): get_rate_details('openai', 'gpt-4.1') returned {'prompt': 0.002, 'completion': 0.008}, no cache_read. The 18 parametrized rate cases failed with KeyError 'cache_read'; the gpt-4.1 cost test billed 0.002 against 0.0008; the snapshot test found no snapshot with a cached rate (assert 0 >= 15); o3-pro substring-matched o3 ($2/$8). 21 failed, 5 passed (the five -pro 'no cached rate' cases). GREEN: 26 passed.
RED (Task 3, long context): the 8 two-sided cases (272,000 vs 272,001 prompt tokens) and the compute_costs test failed with TypeError (get_rate_details had no prompt_tokens keyword). GREEN after the long_prompt tier.
Sources: developers.openai.com/api/docs/pricing.md, re-fetched 2026-10-23; every figure matched the plan's 2026-10-22 table, and the long-context rows are quoted in the Task 3 commit. All 18 snapshots were confirmed on developers.openai.com/api/docs/models/<alias>.md (2026-10-23); none was skipped.
Left without a cached rate on purpose: gpt-5.5-pro, gpt-5.4-pro, gpt-5.2-pro, gpt-5-pro, o1-pro, o3-pro (the page shows '-'); gpt-4o-2024-05-13, o1-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo (no cached price on the page).
Beyond the AC: the o3-pro row ($20/$80 per MTok; it was priced as o3, 10x too low) and a long_prompt tier above 272,000 prompt tokens on gpt-5.5, gpt-5.4, gpt-5.5-pro, gpt-5.4-pro and their snapshots (2x input, 1.5x output, 2x cached input).
Sweep: focused set (tests/Usage, test_anthropic_native_http, test_messages_native_usage_recording, test_anthropic_dotted_model_names) 402 passed, 1 skipped (baseline on dev 298 passed, 1 skipped; the skip is pre-existing); the plan's wider regression set 1399 passed after Task 3 (baseline 1398; the final count is in the PR body); strict-selection endpoint tests 5 passed; other compute_costs/log_llm_usage callers 768 passed; tests/Docs 212 passed; test quality ratchet passed; OpenAPI drift check OK. No test skipped for Postgres or Docker in these sets.
Bandit: uvx bandit -ll on the 13 touched source files (pricing_catalog, llm_usage_normalizer, usage_tracker, anthropic_adapter, anthropic_model_ids, chat_service, messages, llm_providers, routing/metadata, llm_provider_overrides, llm_budget_middleware, tokenizer_resolver, standalone_html_provider): No issues identified.
ADR assessment: no new ADR (optional catalog keys and two billing multipliers inside the existing pipeline; no global default flips). Governing: ADR-075 (Anthropic prompt_tokens include cache tokens, so the Haiku 5.5 tier uses the whole prompt) and ADR-059 (backlog-py); no ADR addresses where multipliers live.
Docs updated: Usage_Quotas.md, Model_Pricing_Catalog.md and the model_pricing.json _catalog_note (plan Task 5).
Follow-ups: TASK-13915 (OpenAI models the catalog lacks, 30-minute and 12-hour cache writes, the regional-processing uplift). Not filed: other pricing tiers (Batch, Flex, Fast) and providers, which the plan files only if the owner wants cost limits to cover them.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
OpenAI cached input tokens are now costed at OpenAI's published cached-input rate: 18 models and the 18 snapshots their model pages name carry cache_read. Also added the o3-pro row and the 272,000-token long-context tier for gpt-5.5, gpt-5.4 and their -pro models.
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
