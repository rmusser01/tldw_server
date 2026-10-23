---
id: TASK-13885
title: Cost Anthropic fast mode (2x) and US-only inference (1.1x) price multipliers
status: Done
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-13858 prices Anthropic calls at list prices. Anthropic bills two request options above them (platform.claude.com/docs/en/about-claude/pricing, fetched 2026-10-20): fast mode (speed: fast) on Claude Opus 5.5 ($8/$40 per MTok, 2x), Opus 5 and Opus 4.8 ($10/$50, 2x), and US-only inference (inference_geo: us) on Claude 4.6 and later models, a 1.1x multiplier on input, output, cache writes and cache reads. They stack with each other and with the prompt-caching multipliers. The Messages API native path forwards client fields unchanged (AnthropicMessagesRequest is extra=allow) and Anthropic reports both in the usage block (usage.speed, usage.inference_geo; messages.py _MESSAGES_NATIVE_USAGE_FIELDS), but compute_costs uses list prices, so those calls are undercharged and cost limits undercount fast-mode traffic by half. Found by the #3286 clean review (M5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A recorded Anthropic call whose usage reports speed fast is costed at the fast-mode price of its model, with a test
- [x] #2 A recorded Anthropic call whose usage reports inference_geo us is costed at 1.1x on every token category, stacking with fast mode and the cache rates, with a test
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-23-pricing.md (branch fix/llm-pricing-followups-fu2).
RED: normalizer tests AttributeError (no speed); compute_costs tests TypeError (unexpected keyword speed / inference_geo); the Messages test costed the fast, us and combined cases at list price (0.042515 against 0.08503, 0.0467665, 0.093533). 9 failed, 2 passed (standard/global and the adapter guard). GREEN: all pass.
Design: ANTHROPIC_FAST_MODE_MULTIPLIER = 2.0 and ANTHROPIC_US_ONLY_MULTIPLIER = 1.1 are code constants in usage_tracker.py, applied once to the finished prompt and completion cost when the usage block reports speed 'fast' or inference_geo 'us'. No model gate: the API only reports the option on a call that used it, and the three published fast prices are exactly 2x list (Opus 5.5 $8/$40, Opus 5 and 4.8 $10/$50). Source: platform.claude.com/docs/en/about-claude/pricing, 'Fast mode pricing' and 'Data residency pricing', re-read 2026-10-23.
Unverified: what usage.inference_geo reports for a pre-4.6 model in a workspace with a default inference_geo (filed as a follow-up).
Sweep: focused set (tests/Usage, test_anthropic_native_http, test_messages_native_usage_recording, test_anthropic_dotted_model_names) 402 passed, 1 skipped (baseline on dev 298 passed, 1 skipped; the skip is pre-existing); the plan's wider regression set 1399 passed after Task 3 (baseline 1398; the final count is in the PR body); strict-selection endpoint tests 5 passed; other compute_costs/log_llm_usage callers 768 passed; tests/Docs 212 passed; test quality ratchet passed; OpenAPI drift check OK. No test skipped for Postgres or Docker in these sets.
Bandit: uvx bandit -ll on the 13 touched source files (pricing_catalog, llm_usage_normalizer, usage_tracker, anthropic_adapter, anthropic_model_ids, chat_service, messages, llm_providers, routing/metadata, llm_provider_overrides, llm_budget_middleware, tokenizer_resolver, standalone_html_provider): No issues identified.
ADR assessment: no new ADR (optional catalog keys and two billing multipliers inside the existing pipeline; no global default flips). Governing: ADR-075 (Anthropic prompt_tokens include cache tokens, so the Haiku 5.5 tier uses the whole prompt) and ADR-059 (backlog-py); no ADR addresses where multipliers live.
Docs updated: Usage_Quotas.md, Model_Pricing_Catalog.md and the model_pricing.json _catalog_note (plan Task 5).
Follow-ups: TASK-13917 (what usage.inference_geo reports for a pre-4.6 model under a workspace default); TASK-13916 (the streamed adapter path misses fast mode and US-only).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Anthropic fast mode (2x) and US-only inference (1.1x) are costed on every token category, stacking with each other, the cache rates and the Haiku 5.5 tier; the normalizer reads speed and inference_geo and the adapter keeps them.
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
