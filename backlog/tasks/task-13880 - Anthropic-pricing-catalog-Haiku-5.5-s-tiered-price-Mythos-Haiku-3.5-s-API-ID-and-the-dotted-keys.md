---
id: TASK-13880
title: 'Anthropic pricing catalog: Haiku 5.5''s tiered price, Mythos, Haiku 3.5''s
  API ID and the dotted keys'
status: Done
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
- [x] #1 claude-haiku-5-5 is costed at Anthropic's price for its prompt length, with a test on both sides of 100,000 tokens
- [x] #2 Every Anthropic entry's key is an API ID or alias, and Mythos and Haiku 3.5's API ID are priced or deliberately left out
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
From the #3286 clean review (M3): two more Anthropic keys are not API IDs. claude-opus-4-7-20260103 and claude-sonnet-4-6-20260105 are in the catalog (TASK-13858 gave them cache rates) and appear in list_provider_models("anthropic"), so the WebUI model picker offers two IDs Anthropic would reject; models from the 4.6 generation on have dateless IDs only. AC #2 covers them. Warning for the implementer: keep the claude-opus-4 and claude-sonnet-4 keys (or add the dated IDs they cover). They are not dotted, but they price the dated 4.x IDs by substring (claude-opus-4-1-*, claude-opus-4-20250514, claude-sonnet-4-5-*, claude-sonnet-4-20250514); deleting them as non-IDs sends those requests to the unknown-model rate. Extend test_anthropic_model_ids_get_their_list_prices_and_cache_rates with those dated IDs.
Plan: Docs/superpowers/plans/2026-10-23-pricing.md (branch fix/llm-pricing-followups-fu2).
RED: Haiku 5.5 tier tests TypeError (no prompt_tokens keyword); Mythos 5.1/5 and claude-3-5-haiku-20241022 returned the unknown rate; the listing test failed on claude-haiku-3.5 (dotted); hidden tests failed (Mythos unpriced, override flag dropped); helper import ModuleNotFoundError; adapter and Messages payload sent the dotted name; metadata test 3 failed (tool_support False after the key rename). With the dotted keys deleted and no mapping yet: is_model_known_for_provider False for all 5 dotted names, provider inference 'openai' instead of 'anthropic', the strict gate returned model_not_available, and the endpoint test got HTTP 400. GREEN: all pass.
Tier semantics: strictly above 100,000 prompt tokens, cache reads and writes counted (prompt_total, ADR-075). get_rates and compute_token_cost take no prompt length and use the base tier (estimators).
Added: claude-haiku-5-5 (long_prompt above 100,000), claude-mythos-5-1 and claude-mythos-5 (priced and hidden: the owner's decision, the plan's recommended option), the aliases claude-opus-4-1 and claude-sonnet-4-5. Renamed: claude-haiku-3.5 to claude-3-5-haiku-20241022, claude-sonnet-3.7 to claude-3-7-sonnet-20250219 (price kept: the current page no longer lists Sonnet 3.7). Deleted: claude-opus-4.5, claude-opus-4.1, claude-sonnet-4.5, claude-haiku-4.5 (also the three dotted DEFAULT_PRICING keys), claude-opus-4-7-20260103, claude-sonnet-4-6-20260105. claude-mythos-preview is deliberately unpriced (deprecated; the page gives no price).
Dotted names stay usable through canonical_anthropic_model_id (core/LLM_Calls/anthropic_model_ids.py): the price lookup, the no-provider estimator, is_model_known_for_provider, infer_provider_from_model_catalog, the adapter request builder, the Messages proxy and count_tokens payload, the tokenizer count and Slides requests, the provider-override and virtual-key allowlists, and the MODEL_METADATA and ROUTING_MODEL_RANKS lookups (keys renamed to dashed). A saved claude-sonnet-4.5 stays known, routes to Anthropic, keeps its tool/vision metadata and router rank, and is sent as claude-sonnet-4-5.
Rulings beyond the plan: provider inference tries the name as written before the dashed form, so a provider that lists the dotted name itself keeps it; the allowlists use a type-safe allowlist_has_model helper, because the virtual-key middleware skips enforcement on TypeError and re.sub raises one on a non-string model (one extra test).
Left for a follow-up: config.py:3861 default, Evaluations/cli/api_utils.py:158, Audio_Streaming_Insights.py:336, RAG quick_wins.py:309-311, claude-3-5-sonnet-latest, and docs/tests with dotted names.
AC #2 is met with stated exceptions: claude-opus-4 and claude-sonnet-4 stay as substring price keys for the dated 4.x IDs (as this task's note says), and claude-3-5-sonnet-latest (alias of a model retired 2025-10-28) is left to the dotted-names follow-up. The listing test checks the picker's shape (no dots, no invalid dated suffixes), not that every key is an API ID.
Unrelated flake seen once: tests/test_contextual_properties.py::test_expand_preserves_document_count_or_increases failed in a 4-file run (Hypothesis deadline=1000 ms, no pricing code on its path); it passed alone and in its file (10 passed). Recorded as load.
Sweep: focused set (tests/Usage, test_anthropic_native_http, test_messages_native_usage_recording, test_anthropic_dotted_model_names) 402 passed, 1 skipped (baseline on dev 298 passed, 1 skipped; the skip is pre-existing); the plan's wider regression set 1399 passed after Task 3 (baseline 1398; the final count is in the PR body); strict-selection endpoint tests 5 passed; other compute_costs/log_llm_usage callers 768 passed; tests/Docs 212 passed; test quality ratchet passed; OpenAPI drift check OK. No test skipped for Postgres or Docker in these sets.
Bandit: uvx bandit -ll on the 13 touched source files (pricing_catalog, llm_usage_normalizer, usage_tracker, anthropic_adapter, anthropic_model_ids, chat_service, messages, llm_providers, routing/metadata, llm_provider_overrides, llm_budget_middleware, tokenizer_resolver, standalone_html_provider): No issues identified.
ADR assessment: no new ADR (optional catalog keys and two billing multipliers inside the existing pipeline; no global default flips). Governing: ADR-075 (Anthropic prompt_tokens include cache tokens, so the Haiku 5.5 tier uses the whole prompt) and ADR-059 (backlog-py); no ADR addresses where multipliers live.
Docs updated: Usage_Quotas.md, Model_Pricing_Catalog.md and the model_pricing.json _catalog_note (plan Task 5).
Follow-ups: TASK-13913 (replace dotted Anthropic names with API IDs at the source, including claude-3-5-sonnet-latest and the config default).
Review fix round 1 (PR #3306, I1): apply_llm_provider_overrides_to_listing filtered the override allowlist by exact name, so an allowlist saved from the old picker (['claude-sonnet-4.5', 'claude-haiku-4.5']) emptied the Anthropic picker and the auto-router's Anthropic candidates. The models and models_info filters now use the same allowlist_has_model mapping as policy_error (Anthropic only). RED: the listing was []; GREEN: both dashed models listed. allowlist_has_model also returns False instead of raising on shapes it cannot compare (M4).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
claude-haiku-5-5 is priced by prompt length (long_prompt tier above 100,000 tokens); Mythos 5.1 and 5 are priced and hidden; Haiku 3.5 and Sonnet 3.7 use their API IDs; the dotted and invalid dated keys are gone, and dotted names saved in conversations and config stay usable as aliases of the dashed API IDs.
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
