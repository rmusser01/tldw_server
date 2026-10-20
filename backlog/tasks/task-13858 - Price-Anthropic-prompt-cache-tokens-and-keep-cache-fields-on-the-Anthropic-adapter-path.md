---
id: TASK-13858
title: Price Anthropic prompt-cache tokens, and keep cache fields on the Anthropic
  adapter path
status: Done
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-13610 counts Anthropic cache reads and writes as prompt tokens on the Messages API native path (ADR-075). No provider in Config_Files/model_pricing.json has a cache_read or cache_write rate, so compute_costs (Usage/usage_tracker.py) bills cached tokens at the full prompt rate: about ten times Anthropic's cache-read price, and below its cache-write price. The Anthropic adapter (LLM_Calls/providers/anthropic_adapter.py, near lines 373-378) also builds prompt_tokens from input_tokens alone and drops cache_creation_input_tokens and cache_read_input_tokens, so a prompt-cached call through /chat/completions is under-counted the way the native path was. Found by the independent review of the TASK-13585/TASK-13610 plan.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 model_pricing.json gives the Anthropic models cache_read and cache_write rates, and a native /v1/messages row with cache fields is costed with them, with a test
- [x] #2 The Anthropic adapter keeps the cache fields in its usage block and counts them in prompt_tokens, with a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-20-anthropic-cache-token-pricing.md (branch fix/anthropic-cache-token-pricing).
Fixed:
- model_pricing.json gives the 14 Anthropic entries on Anthropic's price list cache_read (0.1x input) and cache_write (5-minute write, 1.25x), and adds the 8 flat-priced current models (claude-fable-5-1, claude-fable-5, claude-opus-5-5, claude-opus-5, claude-opus-4-8, claude-opus-4-5, claude-sonnet-5-5, claude-sonnet-5; 0.05x reads on Opus 5.5 and Sonnet 5.5, 0.025x on Fable 5.1), all from platform.claude.com/docs/en/about-claude/pricing (fetched 2026-10-20). The new entries are selectable through /api/v1/llm/providers. compute_costs, the catalog parser, the normalizer and the llm_usage_log cache columns already existed (SQLite migration 088, Postgres pg_migrations_extra), so no code or schema change was needed for pricing.
- AnthropicAdapter._normalize_to_openai_shape adds cache_creation_input_tokens and cache_read_input_tokens to prompt_tokens and keeps both in the usage block, so /chat/completions and provider-recorder rows count and cost cached input like the native path.
- A cached Sonnet 4.6 reply (3 uncached, 2000 written, 60000 read) now costs 0.025509 of prompt, not 0.186009.
Tests:
- T1 RED 15 failed, 16 passed; GREEN 31 passed.
- T2 RED 1 failed, 2 passed; GREEN 3 passed.
- Sweep (68 referencing files, 8 Anthropic-usage tests, plus Usage, LLM_Adapters and LLM_Calls): 3575 passed, 28 skipped. Docs 212 passed. Test-quality ratchet OK.
- Ruff: only the 13 findings dev already has (12 in test_anthropic_native_http.py, an unused import in test_pricing_catalog.py). Bandit clean.
- Token consumers now counting cached input in full: limits.llm_tokens_per_month, the org plan's llm_tokens_month, the RG token commit, the Prometheus token counters.
Follow-ups:
- TASK-13878: 1-hour cache writes at 2x (Decision 1).
- TASK-13879: OpenAI cached-input rates (Decision 2).
- TASK-13880: Haiku 5.5's tiered price, Mythos, Haiku 3.5's API ID, the dotted keys.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Anthropic prompt-cache reads and writes are costed at Anthropic's cache rates, and the flat-priced current Anthropic models (Fable 5.1 and 5, Opus 5.5, 5, 4.8 and 4.5, Sonnet 5.5 and 5) have their list prices. The Anthropic adapter counts the cache fields in prompt_tokens and keeps them in its usage block, so /chat/completions rows match the native Messages path. 1-hour writes at 2x, OpenAI cached-input rates, and Haiku 5.5, Mythos and the dotted keys are follow-ups.
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
