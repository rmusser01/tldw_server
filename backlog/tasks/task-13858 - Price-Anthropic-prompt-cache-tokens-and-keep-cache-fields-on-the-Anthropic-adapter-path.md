---
id: TASK-13858
title: Price Anthropic prompt-cache tokens, and keep cache fields on the Anthropic
  adapter path
status: To Do
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
- [ ] #1 model_pricing.json gives the Anthropic models cache_read and cache_write rates, and a native /v1/messages row with cache fields is costed with them, with a test
- [ ] #2 The Anthropic adapter keeps the cache fields in its usage block and counts them in prompt_tokens, with a test
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
