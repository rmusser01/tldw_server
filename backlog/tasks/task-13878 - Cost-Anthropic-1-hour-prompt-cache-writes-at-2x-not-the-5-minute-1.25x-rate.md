---
id: TASK-13878
title: Cost Anthropic 1-hour prompt-cache writes at 2x, not the 5-minute 1.25x rate
status: Done
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-13858 gave the Anthropic models one cache_write rate, the 5-minute write price (1.25x input). Anthropic bills a 1-hour cache write at 2x input (platform.claude.com/docs/en/about-claude/pricing) and reports the split in usage.cache_creation.ephemeral_5m_input_tokens and ephemeral_1h_input_tokens. The server asks for 1-hour writes when a cache intent sets ttl 1h or ttl_seconds >= 3600 (LLM_Calls/cache_intents.py), and Messages API clients can ask directly, so those writes are costed 0.75x input too low and cost quotas undercount them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The usage normalizer reads the 1-hour share of cache writes, and compute_costs prices it at a cache_write_1h rate, with a test
- [x] #2 model_pricing.json gives the Anthropic models a cache_write_1h rate (2x input), and the Anthropic adapter keeps the cache_creation breakdown
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-23-pricing.md (branch fix/llm-pricing-followups-fu2).
RED: normalizer tests AttributeError (no cache_write_1h_input_tokens); compute_costs tests TypeError (unexpected keyword cache_write_1h_input_tokens); the 2x invariant test KeyError 'cache_write_1h'; the adapter test KeyError 'cache_creation'; the two Messages tests (native and streamed) costed 0.011259 against 0.015759 (every write at 1.25x). 9 failed. GREEN: all pass.
Source: the '1h cache writes' column of platform.claude.com/docs/en/about-claude/pricing (re-read 2026-10-23); every 1-hour price is exactly 2x the row's input price. 22 Anthropic entries got cache_write_1h.
The streamed Messages path keeps the split (message_start carries cache_creation; the merge keeps it), tested. A streamed reply through the AnthropicAdapter is still costed from an estimate (pre-existing gap, filed as a follow-up); Usage_Quotas.md says so.
Deliberate test change: the Anthropic rate constants in test_pricing_catalog.py gained cache_write_1h.
Sweep: focused set (tests/Usage, test_anthropic_native_http, test_messages_native_usage_recording, test_anthropic_dotted_model_names) 402 passed, 1 skipped (baseline on dev 298 passed, 1 skipped; the skip is pre-existing); the plan's wider regression set 1399 passed after Task 3 (baseline 1398; the final count is in the PR body); strict-selection endpoint tests 5 passed; other compute_costs/log_llm_usage callers 768 passed; tests/Docs 212 passed; test quality ratchet passed; OpenAPI drift check OK. No test skipped for Postgres or Docker in these sets.
Bandit: uvx bandit -ll on the 13 touched source files (pricing_catalog, llm_usage_normalizer, usage_tracker, anthropic_adapter, anthropic_model_ids, chat_service, messages, llm_providers, routing/metadata, llm_provider_overrides, llm_budget_middleware, tokenizer_resolver, standalone_html_provider): No issues identified.
ADR assessment: no new ADR (optional catalog keys and two billing multipliers inside the existing pipeline; no global default flips). Governing: ADR-075 (Anthropic prompt_tokens include cache tokens, so the Haiku 5.5 tier uses the whole prompt) and ADR-059 (backlog-py); no ADR addresses where multipliers live.
Docs updated: Usage_Quotas.md, Model_Pricing_Catalog.md and the model_pricing.json _catalog_note (plan Task 5).
Follow-ups: TASK-13916 (a streamed reply through the Anthropic adapter is costed from an estimate, so its 1-hour writes are missed); TASK-13914 (usage.iterations can bill other models).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Anthropic 1-hour cache writes are costed at 2x input: the normalizer reads usage.cache_creation.ephemeral_1h_input_tokens, the catalog carries cache_write_1h, compute_costs splits the writes, and the Anthropic adapter keeps the cache_creation breakdown.
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
